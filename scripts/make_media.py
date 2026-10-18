"""3D GIFs and videos for the README.

    python scripts/make_media.py forecast --wm runs/wm/best.pt     # GT vs ego-warp vs model
    python scripts/make_media.py whatif   --wm runs/wm/best.pt     # one past, three plans
    python scripts/make_media.py planning --wm runs/wm/best.pt --prior runs/prior/best.pt --rows ...

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


# ---- planning: candidates scored by the world model -------------------------------------
def risk_color(c):
    """0 (safe) -> green, 1 (certain collision) -> red."""
    c = float(np.clip(c, 0, 1))
    return (int(60 + 195 * c), int(200 - 150 * c), 70)


def draw_box(img, pose, scale, color, width=2):
    from occwm.constants import EGO_LENGTH, EGO_REAR_OVERHANG, EGO_WIDTH
    from occwm.viz import metric_to_pixel
    x0, x1, hw = -EGO_REAR_OVERHANG, EGO_LENGTH - EGO_REAR_OVERHANG, EGO_WIDTH / 2
    c, s = np.cos(pose[2]), np.sin(pose[2])
    pts = [(pose[0] + c * x - s * y, pose[1] + s * x + c * y) for x, y in ((x0, -hw), (x1, -hw), (x1, hw), (x0, hw))]
    im = Image.fromarray(img)
    ImageDraw.Draw(im).polygon([metric_to_pixel(x, y, scale) for x, y in pts], outline=color, width=width)
    return np.asarray(im)


@torch.no_grad()
def planning_gif(index, wm, prior_ck, row, out, device, scale=3):
    from eval_planning import select, wm_costs
    from occwm.geometry import se2
    from occwm.models.prior import TrajectoryPrior
    from occwm.viz import bev_image, draw_traj
    occ, poses = load_window(index, row, device)
    prior = TrajectoryPrior(len(prior_ck["anchors"]), 32, N_PAST).to(device).eval()
    prior.load_state_dict(prior_ck["model"])
    A = torch.from_numpy(prior_ck["anchors"]).to(device)
    shift = prior_ck["latent_shift"].view(1, -1, 1, 1).to(device)
    scale_ = prior_ck["latent_scale"].view(1, -1, 1, 1).to(device)
    with torch.autocast("cuda", torch.bfloat16):
        z = (wm.vae.encode(occ[N_PAST - 1][None])[0].float() - shift) / scale_
        logp_all = prior(z, poses[None, :N_PAST]).float().log_softmax(-1)[0]
    top = logp_all.topk(32).indices
    cands, logp = A[top], logp_all[top]
    col, off = wm_costs(wm, occ, poses, cands, 16)
    best = select(logp, col, off, 5.0, 1.0)
    plan = cands[best]
    with torch.autocast("cuda", torch.bfloat16):
        pred = wm.forecast(occ[None, :N_PAST], torch.cat([poses[:N_PAST], plan])[None])[0].argmax(1)

    def to_frame_t(frames, fut):  # each frame k lives in pose fut[k]; resample into frame t
        T = torch.linalg.inv(se2(fut[:, 0], fut[:, 1], fut[:, 2]))
        return warp_occupancy(frames, T)

    pred_t = to_frame_t(pred, plan).cpu().numpy()
    gt_t = to_frame_t(occ[N_PAST:], poses[N_PAST:]).cpu().numpy()
    risk = col.amax(-1).cpu().numpy()
    gt_traj, plan_np = poses[N_PAST:].cpu().numpy(), plan.cpu().numpy()
    order = np.argsort(-risk)  # draw risky candidates first, safe ones on top
    frames = []
    for k in range(N_FUTURE):
        panels = []
        for bg, title in ((pred_t[k], "World-model forecast for the chosen plan"), (gt_t[k], "What actually happened")):
            img = bev_image(bg, scale)
            for i in order:
                img = draw_traj(img, cands[i].cpu().numpy(), risk_color(risk[i]), scale, width=1, dots=False)
            img = draw_traj(img, gt_traj, (240, 240, 240), scale, width=2)
            img = draw_traj(img, plan_np, (60, 140, 255), scale, width=4)
            img = draw_box(img, plan_np[k], scale, (60, 140, 255), 3)
            img = draw_box(img, np.zeros(3), scale, (200, 200, 200), 1)
            img = img[90:510, 120:480]  # 42 m x 36 m window around the ego
            panels.append(caption(img, title, f"t = +{(k + 1) * 0.5:.1f} s"))
        frames.append(grid(panels))
    legend = Image.new("RGB", (frames[0].shape[1], 34), (0, 0, 0))
    d = ImageDraw.Draw(legend)
    x = 10
    for text, colr in (("chosen plan", (60, 140, 255)), ("logged trajectory", (240, 240, 240)),
                       ("candidates: safe", (60, 200, 70)), ("risky", (255, 50, 70))):
        d.line([x, 17, x + 26, 17], fill=colr, width=4)
        d.text((x + 32, 8), text, fill=(220, 220, 220), font=font(14))
        x += 46 + int(d.textlength(text, font=font(14)))
    frames = [np.concatenate([f, np.asarray(legend)], 0) for f in frames]
    imageio.mimsave(out, frames, duration=[0.8] * (N_FUTURE - 1) + [2.0], loop=0)
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
    ap.add_argument("what", choices=["forecast", "whatif", "planning"])
    ap.add_argument("--wm", required=True)
    ap.add_argument("--prior", help="needed for `planning`")
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
        path = out / f"{args.what}_{i}.gif"
        if args.what == "planning":
            prior_ck = torch.load(args.prior, map_location="cpu", weights_only=False)
            clips.append(planning_gif(index, wm, prior_ck, row, path, device))
        else:
            fn = forecast_gif if args.what == "forecast" else whatif_gif
            clips.append(fn(index, wm, row, path, device))
    to_mp4(clips, out / f"{args.what}_all.mp4")


if __name__ == "__main__":
    main()
