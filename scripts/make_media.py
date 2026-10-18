"""3D GIFs and videos for the README.

    python scripts/make_media.py forecast --wm runs/wm/best.pt     # GT vs ego-warp vs model
    python scripts/make_media.py whatif   --wm runs/wm/best.pt     # one past, three plans

Scenes are picked automatically where the world model beats the ego-warp most on moving
agents (vehicles / pedestrians), unless --rows is given.
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import _path  # noqa: F401
from occwm.constants import AGENT_CLASSES
from occwm.data import DEFAULT_ROOT, Occ3DIndex
from occwm.geometry import se2, warp_occupancy
from occwm.models.world_model import load_world_model
from occwm.render import add_ego, render

N_PAST, N_FUTURE = 5, 6


def font(size=16):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def caption(img, text, sub=None):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 30 if sub is None else 50], fill=(0, 0, 0))
    d.text((10, 5), text, fill=(240, 240, 240), font=font(18))
    if sub:
        d.text((10, 28), sub, fill=(170, 170, 170), font=font(14))
    return np.asarray(im)


def grid(panels, gap=6):
    h = panels[0].shape[0]
    sep = np.full((h, gap, 3), 0, np.uint8)
    out = [panels[0]]
    for p in panels[1:]:
        out += [sep, p]
    return np.concatenate(out, 1)


def load_window(index, row, device):
    rows = np.arange(row - N_PAST + 1, row + N_FUTURE + 1)
    occ = torch.from_numpy(np.array(index.semantics[rows[0]:rows[-1] + 1])).long().to(device)
    poses = torch.from_numpy(index.relative_poses(rows, row)).to(device)
    return occ, poses


def warp_baseline(occ, poses):
    fut = poses[N_PAST:]
    T = se2(fut[:, 0], fut[:, 1], fut[:, 2])
    return warp_occupancy(occ[N_PAST - 1][None].expand(N_FUTURE, -1, -1, -1), T)


def agent_iou(pred, gt):
    p, g = torch.isin(pred, torch.tensor(AGENT_CLASSES, device=pred.device)), \
        torch.isin(gt, torch.tensor(AGENT_CLASSES, device=gt.device))
    return (p & g).sum().item() / max((p | g).sum().item(), 1)


@torch.no_grad()
def pick_rows(index, wm, device, n, stride=10):
    """Val windows where the model's gain over the warp on moving agents is largest."""
    scored = []
    for row in index.windows("val", N_PAST, N_FUTURE)[::stride]:
        occ, poses = load_window(index, row, device)
        gt = occ[N_PAST:]
        if torch.isin(gt, torch.tensor(AGENT_CLASSES, device=device)).sum() < 400:
            continue  # want scenes with traffic
        with torch.autocast("cuda", torch.bfloat16):
            pred = wm.forecast(occ[None, :N_PAST], poses[None])[0].argmax(1)
        scored.append((agent_iou(pred, gt) - agent_iou(warp_baseline(occ, poses), gt), int(row)))
    scored.sort(reverse=True)
    # one window per scene
    seen, rows = set(), []
    for gain, row in scored:
        s = index.scene_id[row]
        if s not in seen:
            seen.add(s)
            rows.append(row)
        if len(rows) == n:
            break
    return rows


RENDER = dict(width=560, height=360)


@torch.no_grad()
def forecast_gif(index, wm, row, out, device):
    occ, poses = load_window(index, row, device)
    with torch.autocast("cuda", torch.bfloat16):
        pred = wm.forecast(occ[None, :N_PAST], poses[None])[0].argmax(1)
    warp = warp_baseline(occ, poses)
    frames = []
    for k in range(N_PAST):  # history: the same observed frame in all three panels
        img = render(add_ego(occ[k]), **RENDER)
        t = (k - N_PAST + 1) * 0.5
        frames.append(grid([caption(img, "Ground truth", f"observed  t = {t:+.1f} s"),
                            caption(img, "Ego-warp baseline", f"observed  t = {t:+.1f} s"),
                            caption(img, "World model (ours)", f"observed  t = {t:+.1f} s")]))
    for k in range(N_FUTURE):
        t = (k + 1) * 0.5
        frames.append(grid([caption(render(add_ego(occ[N_PAST + k]), **RENDER), "Ground truth", f"future  t = +{t:.1f} s"),
                            caption(render(add_ego(warp[k]), **RENDER), "Ego-warp baseline", f"forecast  t = +{t:.1f} s"),
                            caption(render(add_ego(pred[k]), **RENDER), "World model (ours)", f"forecast  t = +{t:.1f} s")]))
    durations = [0.45] * N_PAST + [0.7] * (N_FUTURE - 1) + [2.0]
    imageio.mimsave(out, frames, duration=durations, loop=0)
    return frames


def bend(poses_fut, yaw_rate, accel):
    """Candidate built from the logged speed: constant yaw rate and acceleration."""
    v0 = float(np.linalg.norm(poses_fut[0, :2].cpu().numpy())) / 0.5
    x = y = yaw = 0.0
    out = []
    for k in range(N_FUTURE):
        v = max(v0 + accel * 0.5 * (k + 1), 0.0)
        yaw += yaw_rate * 0.5
        x += v * 0.5 * np.cos(yaw)
        y += v * 0.5 * np.sin(yaw)
        out.append([x, y, yaw])
    return torch.tensor(out, dtype=torch.float32, device=poses_fut.device)


@torch.no_grad()
def whatif_gif(index, wm, row, out, device):
    """Same past, three different plans: the world model's future follows the plan."""
    occ, poses = load_window(index, row, device)
    plans = {"Logged trajectory": poses[N_PAST:],
             "Hard brake": bend(poses[N_PAST:], 0.0, -4.0),
             "Steer left": bend(poses[N_PAST:], 0.35, 0.0)}
    preds = {}
    for name, fut in plans.items():
        p = torch.cat([poses[:N_PAST], fut])[None]
        with torch.autocast("cuda", torch.bfloat16):
            preds[name] = wm.forecast(occ[None, :N_PAST], p)[0].argmax(1)
    frames = []
    for k in range(N_FUTURE):
        t = (k + 1) * 0.5
        frames.append(grid([caption(render(add_ego(preds[n][k]), **RENDER), n, f"world model forecast  t = +{t:.1f} s")
                            for n in plans]))
    imageio.mimsave(out, frames, duration=[0.7] * (N_FUTURE - 1) + [2.0], loop=0)
    return frames


def to_mp4(frames_list, out, fps=2):
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    for frames in frames_list:
        for f in frames:
            w.append_data(f)
        for _ in range(fps):  # pause between clips
            w.append_data(frames[-1])
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["forecast", "whatif"])
    ap.add_argument("--wm", required=True)
    ap.add_argument("--data", default=DEFAULT_ROOT)
    ap.add_argument("--rows", type=int, nargs="*")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="assets")
    args = ap.parse_args()

    device = torch.device("cuda")
    index = Occ3DIndex(args.data)
    wm = load_world_model(args.wm, device)
    rows = args.rows or pick_rows(index, wm, device, args.n)
    print("rows", rows, [str(index.scene_name[index.scene_id[r]]) for r in rows])
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    clips = []
    for i, row in enumerate(rows):
        fn = forecast_gif if args.what == "forecast" else whatif_gif
        clips.append(fn(index, wm, row, out / f"{args.what}_{i}.gif", device))
    to_mp4(clips, out / f"{args.what}_all.mp4")


if __name__ == "__main__":
    main()
