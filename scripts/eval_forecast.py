"""4D occupancy forecasting on the Occ3D-nuScenes val split (OccWorld protocol).

Inputs: 5 frames (2 s) of history. Targets: the next 6 frames (0.5 s ... 3 s), each in the ego
frame of its own timestamp. Reports mIoU / IoU per horizon and the 1/2/3 s average.

    python scripts/eval_forecast.py --method copy
    python scripts/eval_forecast.py --method warp
    python scripts/eval_forecast.py --method model --ckpt runs/wm/best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import _path  # noqa: F401
from occwm.constants import CLASS_NAMES, EVAL_CLASSES
from occwm.data import DEFAULT_ROOT, Occ3DIndex, OccSequences
from occwm.geometry import se2, warp_occupancy
from occwm.metrics import IoUAccumulator

N_PAST, N_FUTURE = 5, 6


def future_transforms(poses):
    """(B, T, 3) poses in the frame of t -> (B*F, 3, 3) transforms for the future frames."""
    fut = poses[:, N_PAST:]
    return se2(fut[..., 0], fut[..., 1], fut[..., 2]).reshape(-1, 3, 3)


def predict_copy(occ, poses):
    return occ[:, N_PAST - 1:N_PAST].expand(-1, N_FUTURE, -1, -1, -1)


def predict_warp(occ, poses):
    b = occ.shape[0]
    cur = occ[:, N_PAST - 1:N_PAST].expand(-1, N_FUTURE, -1, -1, -1).reshape(-1, *occ.shape[2:])
    return warp_occupancy(cur, future_transforms(poses)).reshape(b, N_FUTURE, *occ.shape[2:])


def build_model_predictor(ckpt, device):
    from occwm.models.world_model import load_world_model
    wm = load_world_model(ckpt, device)

    @torch.no_grad()
    def predict(occ, poses):
        with torch.autocast("cuda", torch.bfloat16):
            logits = wm.forecast(occ[:, :N_PAST], poses)
        return logits.argmax(2)
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["copy", "warp", "model"], required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--name", help="result name (default: method)")
    ap.add_argument("--data", default=DEFAULT_ROOT)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1, help="subsample val windows (debug)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = Occ3DIndex(args.data)
    ds = OccSequences(index, "val", N_PAST, N_FUTURE, stride=args.stride)
    dl = DataLoader(ds, args.bs, num_workers=6, pin_memory=True)
    predict = {"copy": predict_copy, "warp": predict_warp}.get(args.method)
    if args.method == "model":
        predict = build_model_predictor(args.ckpt, device)

    acc = IoUAccumulator(N_FUTURE, device)
    for i, batch in enumerate(dl):
        occ = batch["occ"].to(device, non_blocking=True).long()
        poses = batch["poses"].to(device)
        pred = predict(occ, poses)
        acc.update(pred, occ[:, N_PAST:])
        if i % 200 == 0:
            print(f"{i * args.bs}/{len(ds)}  mIoU {np.round(acc.miou(), 2)}", flush=True)

    miou, iou = acc.miou(), acc.iou()
    sec = [1, 3, 5]  # indices of the 1 s, 2 s and 3 s frames
    res = {
        "method": args.name or args.method, "ckpt": args.ckpt, "n_windows": len(ds),
        "miou": miou.round(2).tolist(), "iou": iou.round(2).tolist(),
        "miou_1s_2s_3s_avg": [*miou[sec].round(2).tolist(), round(float(miou[sec].mean()), 2)],
        "iou_1s_2s_3s_avg": [*iou[sec].round(2).tolist(), round(float(iou[sec].mean()), 2)],
        "per_class_iou_3s": dict(zip([CLASS_NAMES[c] for c in EVAL_CLASSES],
                                     (acc.per_class_iou()[5] * 100).cpu().numpy().round(2).tolist())),
    }
    Path(args.out).mkdir(exist_ok=True)
    path = Path(args.out) / f"forecast_{res['method']}.json"
    json.dump(res, open(path, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("method", "miou_1s_2s_3s_avg", "iou_1s_2s_3s_avg")}))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
