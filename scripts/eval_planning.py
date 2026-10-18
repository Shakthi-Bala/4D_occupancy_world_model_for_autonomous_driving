"""Open-loop planning on Occ3D-nuScenes val: score candidate trajectories with the world model.

For every val window:
  1. The imitation prior scores all anchors; the top-M become candidates.
  2. The world model forecasts 3 s of occupancy *for each candidate*, in that candidate's own
     future ego frames, so the ego box always sits at the grid centre.
  3. cost = -log p_prior + w_col * P(obstacle in ego box) + w_off * P(not on road)

Planners compared (same candidates, only the safety cost changes):
  gt            the logged trajectory (floor for the collision metric)
  const_vel     constant velocity / yaw rate from the last two poses
  prior         argmax of the imitation prior, no safety term
  prior+static  safety cost from the *current* occupancy (static-world assumption, no model)
  prior+wm      safety cost from world-model forecasts
  prior+oracle  safety cost from the GT future occupancy (privileged upper bound)

Metrics (ST-P3 / OccWorld conventions, cumulative mean up to 1/2/3 s): L2 to the logged
trajectory, collision rate of the ego box against GT future occupancy (agents = vehicles +
pedestrians, and all obstacles), off-road rate.

    torchrun --nproc_per_node 4 scripts/eval_planning.py --wm runs/wm/best.pt --prior runs/prior/best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import _path  # noqa: F401
from occwm.constants import OBSTACLE_CLASSES
from occwm.data import DEFAULT_ROOT, Occ3DIndex, OccSequences
from occwm.metrics import l2_stp3, rate_stp3
from occwm.models.prior import TrajectoryPrior
from occwm.models.world_model import load_world_model
from occwm.planning import collisions, offroad, predicted_costs
from occwm.train_utils import is_main, setup_ddp

N_PAST, N_FUTURE = 5, 6
CROP = slice(68, 132)  # 25.6 m x 25.6 m window around the ego used for world-model costs


def const_velocity(past):
    """past: (P, 3) poses in frame t (last = identity). Roll the last step forward."""
    dx, dy, dyaw = past[-1] - past[-2]
    c, s = np.cos(-past[-2, 2]), np.sin(-past[-2, 2])
    step = np.array([c * dx - s * dy, s * dx + c * dy])  # last displacement in the frame of t-1
    out, pose = [], np.zeros(3)
    for _ in range(N_FUTURE):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        pose = pose + np.array([c * step[0] - s * step[1], s * step[0] + c * step[1], dyaw])
        out.append(pose.copy())
    return np.array(out, np.float32)


def maneuver(fut):
    """Coarse label of the logged future, used to slice the results."""
    dist_, yaw = np.linalg.norm(fut[-1, :2]), fut[-1, 2]
    if dist_ < 2.0:
        return "stopped"
    if yaw > np.deg2rad(20):
        return "left_turn"
    if yaw < -np.deg2rad(20):
        return "right_turn"
    return "straight"


@torch.no_grad()
def wm_costs(wm, occ, poses, cands, chunk):
    """World-model collision / off-road costs for (M, F, 3) candidates of one window."""
    ctx = wm.encode_context(occ[None, :N_PAST], poses[None, :N_PAST])
    cols, offs = [], []
    for i in range(0, len(cands), chunk):
        c = cands[i:i + chunk]
        cx = wm.expand_context(ctx, len(c))
        with torch.autocast("cuda", torch.bfloat16):
            z, _ = wm.forward_latent(cx, c)
            warped = wm.warp_now(cx, c)
            lat = slice(CROP.start // 4, CROP.stop // 4)
            logits = torch.stack([wm._decode_one(z[:, k, :, lat, lat], warped[:, k, CROP, CROP])
                                  for k in range(N_FUTURE)], 1)
        col, off = predicted_costs(logits, CROP.start, CROP.start)
        cols.append(col)
        offs.append(off)
    return torch.cat(cols), torch.cat(offs)


def select(logp, col, off, w_col, w_off):
    """Index of the best candidate. Safety terms use the worst (max) over the horizon."""
    cost = -logp + w_col * col.amax(-1) + w_off * off.mean(-1)
    return int(cost.argmin())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_ROOT)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--top-m", type=int, default=32)
    ap.add_argument("--w-col", type=float, default=5.0)
    ap.add_argument("--w-off", type=float, default=1.0)
    ap.add_argument("--sweep", type=float, nargs="*", default=[0, 0.5, 1, 2, 5, 10, 20, 50])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="planning")
    args = ap.parse_args()

    rank, world, device = setup_ddp()
    index = Occ3DIndex(args.data)
    ds = OccSequences(index, "val", N_PAST, N_FUTURE, stride=args.stride)
    sampler = DistributedSampler(ds, shuffle=False) if world > 1 else None
    dl = DataLoader(ds, 1, sampler=sampler, num_workers=4)

    wm = load_world_model(args.wm, device)
    pck = torch.load(args.prior, map_location="cpu", weights_only=False)
    prior = TrajectoryPrior(len(pck["anchors"]), pck["args"].get("latent_ch", 32), N_PAST)
    prior.load_state_dict(pck["model"])
    prior.to(device).eval()
    A = torch.from_numpy(pck["anchors"]).to(device)
    shift = pck["latent_shift"].view(1, -1, 1, 1).to(device)
    scale = pck["latent_scale"].view(1, -1, 1, 1).to(device)

    methods = ["gt", "const_vel", "prior", "prior+static", "prior+wm", "prior+oracle"]
    recs = []
    for it, batch in enumerate(dl):
        occ = batch["occ"][0].to(device).long()
        poses = batch["poses"][0].to(device)
        past, gt = poses[:N_PAST], poses[N_PAST:]
        fut_occ = occ[N_PAST:]

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            z_now = (wm.vae.encode(occ[N_PAST - 1][None])[0].float() - shift) / scale
            logp_all = prior(z_now, past[None]).float().log_softmax(-1)[0]
        top = logp_all.topk(args.top_m).indices
        cands, logp = A[top], logp_all[top]

        # Safety costs for every candidate under each world assumption.
        now = occ[N_PAST - 1][None].expand(N_FUTURE, -1, -1, -1)
        zero = torch.zeros(N_FUTURE, 3, device=device)
        costs = {
            "static": (collisions(cands, zero, now, OBSTACLE_CLASSES, margin=0.4).float(),
                       offroad(cands, zero, now)),
            "oracle": (collisions(cands, gt, fut_occ, OBSTACLE_CLASSES, margin=0.4).float(),
                       offroad(cands, gt, fut_occ)),
            "wm": wm_costs(wm, occ, poses, cands, args.chunk),
        }
        plans = {"gt": gt, "const_vel": torch.from_numpy(const_velocity(past.cpu().numpy())).to(device),
                 "prior": A[logp_all.argmax()]}
        for k, (col, off) in costs.items():
            plans[f"prior+{k}"] = cands[select(logp, col, off, args.w_col, args.w_off)]
        sweep = {k: [cands[select(logp, col, off, w, args.w_off)] for w in args.sweep]
                 for k, (col, off) in costs.items()}

        def evaluate(traj):
            traj = traj[None]
            return {"xy": traj[0, :, :2].cpu().numpy(),
                    "col_agent": collisions(traj, gt, fut_occ)[0].cpu().numpy(),
                    "col_obst": collisions(traj, gt, fut_occ, OBSTACLE_CLASSES)[0].cpu().numpy(),
                    "off": (offroad(traj, gt, fut_occ)[0] > 0.5).cpu().numpy()}

        rec = {"row": int(batch["row"][0]), "gt_xy": gt[:, :2].cpu().numpy(),
               "maneuver": maneuver(gt.cpu().numpy()),
               "plans": {m: evaluate(plans[m]) for m in methods},
               "sweep": {k: [evaluate(t) for t in v] for k, v in sweep.items()},
               "wm_col_of_choice": float(costs["wm"][0][select(logp, *costs["wm"], args.w_col, args.w_off)].amax())}
        recs.append(rec)
        if is_main() and it % 100 == 0:
            print(f"{it * world}/{len(ds)}", flush=True)

    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, recs)
        recs = [r for g in gathered for r in g]
    if not is_main():
        return
    recs = list({r["row"]: r for r in recs}.values())  # DistributedSampler pads with repeats

    def summarize(items, key):
        pred = np.stack([it[key]["xy"] if key else it["xy"] for it in items])
        gt_xy = np.stack([it["gt_xy"] for it in items])
        get = (lambda it, f: it[key][f]) if key else (lambda it, f: it[f])
        l2 = l2_stp3(pred, gt_xy)
        ca = rate_stp3(np.stack([get(it, "col_agent") for it in items]))
        co = rate_stp3(np.stack([get(it, "col_obst") for it in items]))
        off = rate_stp3(np.stack([get(it, "off") for it in items]))
        pick = [1, 3, 5]
        return {"l2": [*l2[pick].round(3).tolist(), round(float(l2[pick].mean()), 3)],
                "col_agent": [*ca[pick].round(3).tolist(), round(float(ca[pick].mean()), 3)],
                "col_obstacle": [*co[pick].round(3).tolist(), round(float(co[pick].mean()), 3)],
                "offroad": [*off[pick].round(3).tolist(), round(float(off[pick].mean()), 3)]}

    res = {"n_windows": len(recs), "top_m": args.top_m, "w_col": args.w_col, "w_off": args.w_off,
           "overall": {}, "by_maneuver": {}, "sweep": {}}
    for m in methods:
        res["overall"][m] = summarize([{**r, **{"m": r["plans"][m]}} for r in recs], "m")
    for man in sorted({r["maneuver"] for r in recs}):
        sub = [r for r in recs if r["maneuver"] == man]
        res["by_maneuver"][man] = {"n": len(sub), **{m: summarize([{**r, "m": r["plans"][m]} for r in sub], "m")
                                                       for m in methods}}
    for k in ("static", "wm", "oracle"):
        res["sweep"][k] = [{"w_col": w, **summarize([{**r, "m": r["sweep"][k][i]} for r in recs], "m")}
                           for i, w in enumerate(args.sweep)]
    Path(args.out).mkdir(exist_ok=True)
    json.dump(res, open(Path(args.out) / f"{args.name}.json", "w"), indent=2)
    np.save(Path(args.out) / f"{args.name}_records.npy", np.array(recs, dtype=object), allow_pickle=True)
    for m in methods:
        o = res["overall"][m]
        print(f"{m:14s} L2 {o['l2']}  col_agent% {o['col_agent']}  col_obst% {o['col_obstacle']}  off% {o['offroad']}")


if __name__ == "__main__":
    main()
