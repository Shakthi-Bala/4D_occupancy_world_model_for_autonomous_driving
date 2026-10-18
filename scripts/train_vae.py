"""Stage 1: train the occupancy VAE (tokenizer).

    torchrun --nproc_per_node 4 scripts/train_vae.py --out runs/vae
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import _path  # noqa: F401
from occwm.data import DEFAULT_ROOT, Occ3DIndex, OccFrames
from occwm.losses import occupancy_loss
from occwm.metrics import IoUAccumulator
from occwm.models.vae import OccVAE, kl_divergence
from occwm.train_utils import (Logger, all_reduce_, cosine_lr, is_main, save_ckpt, setup_ddp,
                               unwrap)
from occwm.viz import bev_image, hstack, label


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    acc = IoUAccumulator(1, device)
    for occ in loader:
        occ = occ.to(device, non_blocking=True)
        with torch.autocast("cuda", torch.bfloat16):
            logits, _, _ = model(occ, sample=False)
        acc.update(logits.argmax(1)[:, None], occ[:, None])
    all_reduce_(acc.tensors())
    model.train()
    return float(acc.miou()[0]), float(acc.iou()[0])


@torch.no_grad()
def snapshot(model, occ, path):
    model.eval()
    with torch.autocast("cuda", torch.bfloat16):
        pred = model(occ[:3], sample=False)[0].argmax(1).cpu().numpy()
    rows = [hstack([label(bev_image(g), "GT"), label(bev_image(p), "reconstruction")])
            for g, p in zip(occ[:3].cpu().numpy(), pred)]
    from PIL import Image
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="runs/vae")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8, help="per GPU")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--latent-ch", type=int, default=32)
    ap.add_argument("--kl", type=float, default=1e-3)
    ap.add_argument("--lovasz", type=float, default=1.0)
    ap.add_argument("--val-frames", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop early")
    args = ap.parse_args()

    rank, world, device = setup_ddp()
    out = Path(args.out)
    if is_main():
        out.mkdir(parents=True, exist_ok=True)
        json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    log = Logger(out)

    index = Occ3DIndex(args.data)
    train_ds = OccFrames(index, "train", flip=True)
    val_ds = OccFrames(index, "val")
    val_ds = Subset(val_ds, np.linspace(0, len(val_ds) - 1, args.val_frames).astype(int))
    train_sampler = DistributedSampler(train_ds, shuffle=True) if world > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world > 1 else None
    train_dl = DataLoader(train_ds, args.bs, sampler=train_sampler, shuffle=train_sampler is None,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=True)
    val_dl = DataLoader(val_ds, args.bs, sampler=val_sampler, num_workers=args.workers)

    model = OccVAE(latent_ch=args.latent_ch).to(device)
    if is_main():
        print(f"VAE params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
    if world > 1:
        model = DDP(model, device_ids=[device.index])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    total = args.epochs * len(train_dl)
    step, best = 0, -1.0
    fixed = next(iter(val_dl))[:3].to(device)
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for occ in train_dl:
            occ = occ.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, args.lr)
            with torch.autocast("cuda", torch.bfloat16):
                logits, mean, logvar = model(occ)
            rec, parts = occupancy_loss(logits, occ, lovasz_weight=args.lovasz)
            kl = kl_divergence(mean.float(), logvar.float())
            loss = rec + args.kl * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                log.log(step=step, epoch=epoch, loss=loss, ce=parts["ce"], lovasz=parts["lovasz"],
                        kl=kl, grad_norm=gn, lr=opt.param_groups[0]["lr"])
            step += 1
            if args.max_steps and step >= args.max_steps:
                break

        miou, iou = evaluate(model, val_dl, device)
        log.log(step=step, epoch=epoch, val_miou=miou, val_iou=iou)
        if is_main():
            snapshot(unwrap(model), fixed, out / "recon.png")
            save_ckpt(out / "last.pt", model, opt, epoch=epoch, step=step, args=vars(args))
            if miou > best:
                best = miou
                save_ckpt(out / "best.pt", model, epoch=epoch, step=step, args=vars(args),
                          val_miou=miou, val_iou=iou)
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
