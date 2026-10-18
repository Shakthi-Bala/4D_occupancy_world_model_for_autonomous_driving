"""Reconstruction mIoU / IoU of the occupancy VAE on the full val split.

    python scripts/eval_vae.py --ckpt runs/vae/best.pt
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _path  # noqa: F401
from occwm.data import DEFAULT_ROOT, Occ3DIndex, OccFrames
from occwm.metrics import IoUAccumulator
from occwm.models.vae import OccVAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    c = ck["args"]["latent_ch"]
    vae = OccVAE(latent_ch=c)
    vae.load_state_dict(ck["model"])
    vae.to(device).eval()
    ds = OccFrames(Occ3DIndex(args.data), "val")
    acc = IoUAccumulator(1, device)
    with torch.no_grad():
        for occ in DataLoader(ds, 16, num_workers=6, pin_memory=True):
            occ = occ.to(device, non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                logits, _, _ = vae(occ, sample=False)
            acc.update(logits.argmax(1)[:, None], occ[:, None])
    res = {"n_frames": len(ds), "miou": round(float(acc.miou()[0]), 2), "iou": round(float(acc.iou()[0]), 2),
           "latent": f"50x50x{c}", "compression": round(200 * 200 * 16 / (50 * 50 * c), 1)}
    Path(args.out).mkdir(exist_ok=True)
    json.dump(res, open(Path(args.out) / "vae_recon.json", "w"), indent=2)
    print(res)


if __name__ == "__main__":
    main()
