"""Stage 2: train the trajectory-conditioned world model on top of a trained VAE.

    torchrun --nproc_per_node 4 scripts/train_wm.py --vae runs/vae/best.pt --out runs/wm
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import _path  # noqa: F401
from occwm.data import DEFAULT_ROOT, Occ3DIndex, OccFrames, OccSequences
from occwm.losses import occupancy_loss
from occwm.metrics import IoUAccumulator
from occwm.models.vae import OccVAE, latent_stats
from occwm.models.world_model import OccWorldModel
from occwm.train_utils import (Logger, all_reduce_, cosine_lr, is_main, save_ckpt, setup_ddp,
                               unwrap)
from occwm.viz import bev_image, hstack, label, vstack

N_PAST, N_FUTURE = 5, 6


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    acc = IoUAccumulator(N_FUTURE, device)
    for batch in loader:
        occ = batch["occ"].to(device, non_blocking=True).long()
        poses = batch["poses"].to(device)
        with torch.autocast("cuda", torch.bfloat16):
            logits, _ = model(occ[:, :N_PAST], poses)
        acc.update(logits.argmax(2), occ[:, N_PAST:])
    all_reduce_(acc.tensors())
    model.train()
    return acc.miou(), acc.iou()


@torch.no_grad()
def snapshot(model, batch, device, path):
    model.eval()
    occ = batch["occ"][:1].to(device).long()
    poses = batch["poses"][:1].to(device)
    with torch.autocast("cuda", torch.bfloat16):
        pred = model(occ[:, :N_PAST], poses)[0].argmax(2)[0].cpu().numpy()
    gt = occ[0, N_PAST:].cpu().numpy()
    rows = [hstack([label(bev_image(gt[k], 1), f"GT +{(k + 1) / 2:.1f}s") for k in (1, 3, 5)]),
            hstack([label(bev_image(pred[k], 1), f"pred +{(k + 1) / 2:.1f}s") for k in (1, 3, 5)])]
    from PIL import Image
    Image.fromarray(vstack(rows)).save(path)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_ROOT)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", default="runs/wm")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=2, help="per GPU")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--decoder-lr", type=float, default=5e-5)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--latent-weight", type=float, default=1.0)
    ap.add_argument("--lovasz", type=float, default=1.0)
    ap.add_argument("--val-stride", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()

    rank, world, device = setup_ddp()
    out = Path(args.out)
    if is_main():
        out.mkdir(parents=True, exist_ok=True)
        json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    log = Logger(out)

    index = Occ3DIndex(args.data)
    vck = torch.load(args.vae, map_location="cpu", weights_only=False)
    vae = OccVAE(latent_ch=vck["args"]["latent_ch"])
    vae.load_state_dict(vck["model"])
    vae.to(device)
    shift, scale = latent_stats(vae, OccFrames(index, "train"), device)
    args.latent_ch = vck["args"]["latent_ch"]

    model = OccWorldModel(vae, shift, scale, N_PAST, N_FUTURE, args.dim, args.depth, args.heads).to(device)
    dec_params = [p for n, p in model.named_parameters() if n.startswith("vae.") and p.requires_grad]
    new_params = [p for n, p in model.named_parameters() if not n.startswith("vae.")]
    if is_main():
        print(f"world model: {sum(p.numel() for p in new_params) / 1e6:.1f} M new params, "
              f"{sum(p.numel() for p in dec_params) / 1e6:.1f} M fine-tuned decoder params", flush=True)
    if world > 1:
        model = DDP(model, device_ids=[device.index])
    opt = torch.optim.AdamW([{"params": new_params, "lr": args.lr, "base": args.lr},
                             {"params": dec_params, "lr": args.decoder_lr, "base": args.decoder_lr}],
                            weight_decay=0.01, betas=(0.9, 0.95))

    train_ds = OccSequences(index, "train", N_PAST, N_FUTURE, flip=True)
    val_ds = OccSequences(index, "val", N_PAST, N_FUTURE, stride=args.val_stride)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if world > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world > 1 else None
    train_dl = DataLoader(train_ds, args.bs, sampler=train_sampler, shuffle=train_sampler is None,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=True)
    val_dl = DataLoader(val_ds, args.bs, sampler=val_sampler, num_workers=args.workers)
    fixed = next(iter(DataLoader(val_ds, 1, shuffle=False)))

    total = args.epochs * len(train_dl)
    step, best = 0, -1.0
    raw = unwrap(model)
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_dl:
            occ = batch["occ"].to(device, non_blocking=True).long()
            poses = batch["poses"].to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, g["base"])
            with torch.autocast("cuda", torch.bfloat16):
                logits, z = model(occ[:, :N_PAST], poses)
                z_gt = raw.encode(occ[:, N_PAST:].flatten(0, 1)).reshape(z.shape)
            seg, parts = occupancy_loss(logits.flatten(0, 1), occ[:, N_PAST:].flatten(0, 1),
                                        lovasz_weight=args.lovasz)
            lat = F.mse_loss(z.float(), z_gt.float())
            loss = seg + args.latent_weight * lat
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                log.log(step=step, epoch=epoch, loss=loss, ce=parts["ce"], lovasz=parts["lovasz"],
                        latent=lat, grad_norm=gn, lr=opt.param_groups[0]["lr"])
            step += 1
            if args.max_steps and step >= args.max_steps:
                break

        miou, iou = evaluate(model, val_dl, device)
        avg_miou, avg_iou = float(miou[[1, 3, 5]].mean()), float(iou[[1, 3, 5]].mean())
        log.log(step=step, epoch=epoch, val_miou_avg=avg_miou, val_iou_avg=avg_iou,
                val_miou=[round(float(v), 2) for v in miou], val_iou=[round(float(v), 2) for v in iou])
        if is_main():
            snapshot(raw, fixed, device, out / "forecast.png")
            extra = dict(epoch=epoch, step=step, args=vars(args), latent_shift=shift, latent_scale=scale)
            save_ckpt(out / "last.pt", model, opt, **extra)
            if avg_miou > best:
                best = avg_miou
                save_ckpt(out / "best.pt", model, val_miou_avg=avg_miou, val_iou_avg=avg_iou, **extra)
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
