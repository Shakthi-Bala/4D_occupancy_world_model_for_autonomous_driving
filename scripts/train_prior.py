"""Stage 3: build the trajectory vocabulary and train the imitation prior.

    python scripts/train_prior.py --vae runs/vae/best.pt --out runs/prior
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import _path  # noqa: F401
from occwm.data import DEFAULT_ROOT, Occ3DIndex, OccFrames
from occwm.metrics import l2_stp3
from occwm.models.prior import TrajectoryPrior, soft_targets
from occwm.models.vae import OccVAE, latent_stats
from occwm.planning import build_anchors
from occwm.train_utils import Logger, cosine_lr, save_ckpt, setup_ddp

N_PAST, N_FUTURE = 5, 6


class CurrentFrame(Dataset):
    """Current occupancy + past/future ego poses for every planning window."""

    def __init__(self, index, split, flip=False):
        self.index, self.flip = index, flip
        self.rows = index.windows(split, N_PAST, N_FUTURE)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        t = self.rows[i]
        occ = np.array(self.index.semantics[t])
        poses = self.index.relative_poses(np.arange(t - N_PAST + 1, t + N_FUTURE + 1), t)
        if self.flip and np.random.rand() < 0.5:
            occ = occ[:, ::-1].copy()
            poses[:, 1:] *= -1
        return torch.from_numpy(occ), torch.from_numpy(poses)


def all_future_poses(index, split):
    rows = index.windows(split, N_PAST, N_FUTURE)
    return np.stack([index.relative_poses(np.arange(t + 1, t + N_FUTURE + 1), t) for t in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_ROOT)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", default="runs/prior")
    ap.add_argument("--anchors", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    _, _, device = setup_ddp()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    log = Logger(out)
    index = Occ3DIndex(args.data)

    fut = all_future_poses(index, "train")
    fut_mirror = fut * np.array([1, -1, -1], np.float32)  # vocabulary covers both mirror images
    anchors = build_anchors(np.concatenate([fut, fut_mirror]), args.anchors)
    np.save(out / "anchors.npy", anchors)
    gt_val = all_future_poses(index, "val")
    d = np.linalg.norm(anchors[None, :, :, :2] - gt_val[:, None, :, :2], axis=-1).mean(-1)
    oracle = l2_stp3(anchors[d.argmin(1)][..., :2], gt_val[..., :2])
    log.log(msg="anchors", n=args.anchors, oracle_l2_1s=float(oracle[1]), oracle_l2_2s=float(oracle[3]),
            oracle_l2_3s=float(oracle[5]))

    vck = torch.load(args.vae, map_location="cpu", weights_only=False)
    vae = OccVAE(latent_ch=vck["args"]["latent_ch"]).to(device)
    vae.load_state_dict(vck["model"])
    vae.eval()
    shift, scale = latent_stats(vae, OccFrames(index, "train"), device)
    shift, scale = shift.view(1, -1, 1, 1).to(device), scale.view(1, -1, 1, 1).to(device)

    def encode(occ):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            return (vae.encode(occ)[0].float() - shift) / scale

    model = TrajectoryPrior(args.anchors, vck["args"]["latent_ch"], N_PAST).to(device)
    A = torch.from_numpy(anchors).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_dl = DataLoader(CurrentFrame(index, "train", flip=True), args.bs, shuffle=True,
                          num_workers=args.workers, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(CurrentFrame(index, "val"), 64, num_workers=args.workers)
    total, step, best = args.epochs * len(train_dl), 0, 1e9
    for epoch in range(args.epochs):
        model.train()
        for occ, poses in train_dl:
            occ, poses = occ.to(device, non_blocking=True), poses.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, args.lr, warmup=500)
            logits = model(encode(occ), poses[:, :N_PAST])
            q = soft_targets(A, poses[:, N_PAST:])
            loss = -(q * logits.log_softmax(-1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 200 == 0:
                log.log(step=step, epoch=epoch, loss=loss)
            step += 1

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for occ, poses in val_dl:
                occ, poses = occ.to(device), poses.to(device)
                k = model(encode(occ), poses[:, :N_PAST]).argmax(-1)
                preds.append(A[k].cpu().numpy())
                gts.append(poses[:, N_PAST:].cpu().numpy())
        l2 = l2_stp3(np.concatenate(preds)[..., :2], np.concatenate(gts)[..., :2])
        avg = float(l2[[1, 3, 5]].mean())
        log.log(step=step, epoch=epoch, val_l2_1s=float(l2[1]), val_l2_2s=float(l2[3]),
                val_l2_3s=float(l2[5]), val_l2_avg=avg)
        extra = dict(args=vars(args), anchors=anchors, latent_shift=shift.flatten().cpu(),
                     latent_scale=scale.flatten().cpu(), epoch=epoch)
        save_ckpt(out / "last.pt", model, **extra)
        if avg < best:
            best = avg
            save_ckpt(out / "best.pt", model, val_l2_avg=avg, **extra)


if __name__ == "__main__":
    main()
