"""Pack Occ3D-nuScenes into memory-mappable arrays.

Reads the extracted `gts/<scene>/<token>/labels.npz` files plus Occ3D's `annotations.json`
(ego poses, timestamps, train/val split) and writes:

    <out>/semantics.npy     (N, 200, 200, 16) uint8 labels, one row per keyframe
    <out>/mask_camera.npy   (N, 200*200*16/8) bit-packed camera-visibility mask
    <out>/index.npz         per-frame scene id, token, timestamp, SE(2) ego pose;
                            per-scene name, split, start row, length

Frames are ordered by scene, then timestamp, so a scene is a contiguous slice.

    python scripts/prepare_data.py --root /scratch/$USER/data/occ3d_nuscenes
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import _path  # noqa: F401
from occwm.constants import GRID_SHAPE
from occwm.geometry import quat_to_yaw

VOX = int(np.prod(GRID_SHAPE))


def load(path):
    d = np.load(path)
    return d["semantics"], np.packbits(d["mask_camera"].astype(bool).ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding gts/ and annotations.json")
    ap.add_argument("--out", default=None, help="default: <root>/packed")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out or root / "packed")
    out.mkdir(parents=True, exist_ok=True)
    ann = json.load(open(root / "annotations.json"))
    split_of = {s: "train" for s in ann["train_split"]} | {s: "val" for s in ann["val_split"]}

    scene_names, scene_split, scene_start, scene_len = [], [], [], []
    files, tokens, scene_ids, stamps, poses = [], [], [], [], []
    for sid, name in enumerate(sorted(ann["scene_infos"])):
        frames = sorted(ann["scene_infos"][name].items(), key=lambda kv: kv[1]["timestamp"])
        scene_names.append(name)
        scene_split.append(split_of[name])
        scene_start.append(len(files))
        scene_len.append(len(frames))
        for token, info in frames:
            files.append(root / "gts" / name / token / "labels.npz")
            tokens.append(token)
            scene_ids.append(sid)
            stamps.append(info["timestamp"])
            t, q = info["ego_pose"]["translation"], info["ego_pose"]["rotation"]
            poses.append([t[0], t[1], quat_to_yaw(q)])

    n = len(files)
    missing = [f for f in files if not f.exists()]
    assert not missing, f"{len(missing)} label files missing, e.g. {missing[0]}"
    print(f"{len(scene_names)} scenes, {n} frames "
          f"({sum(l for l, s in zip(scene_len, scene_split) if s == 'train')} train)")

    sem = np.lib.format.open_memmap(out / "semantics.npy", "w+", np.uint8, (n, *GRID_SHAPE))
    cam = np.lib.format.open_memmap(out / "mask_camera.npy", "w+", np.uint8, (n, VOX // 8))
    with Pool(args.workers) as pool:
        for i, (s, m) in enumerate(pool.imap(load, files, chunksize=32)):
            sem[i], cam[i] = s, m
            if i % 5000 == 0:
                print(f"  {i}/{n}", flush=True)
    sem.flush()
    cam.flush()

    np.savez(out / "index.npz",
             token=np.array(tokens), scene_id=np.array(scene_ids, np.int32),
             timestamp=np.array(stamps, np.int64), pose=np.array(poses, np.float64),
             scene_name=np.array(scene_names), scene_split=np.array(scene_split),
             scene_start=np.array(scene_start, np.int64), scene_len=np.array(scene_len, np.int64))
    print(f"wrote {out} ({os.path.getsize(out / 'semantics.npy') / 1e9:.1f} GB labels)")


if __name__ == "__main__":
    main()
