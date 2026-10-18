"""Trajectory vocabulary, collision / off-road checks and costs for occupancy-based planning.

All trajectories are (..., F, 3) arrays of (x, y, yaw) for the F future keyframes, expressed in
the ego frame of the current frame t.
"""

import numpy as np
import torch

from .constants import (AGENT_CLASSES, EGO_LENGTH, EGO_REAR_OVERHANG, EGO_WIDTH, EGO_Z_RANGE,
                        GRID_SHAPE, NON_DRIVABLE, NUM_CLASSES, OBSTACLE_CLASSES, PC_RANGE,
                        VOXEL_SIZE)
from .geometry import rel_pose, se2

Z_BODY = slice(int((EGO_Z_RANGE[0] - PC_RANGE[2]) / VOXEL_SIZE),
               int(np.ceil((EGO_Z_RANGE[1] - PC_RANGE[2]) / VOXEL_SIZE)))  # cells 4..6
Z_GROUND = slice(0, 4)


def build_anchors(future_poses, n=256, seed=0):
    """k-means over (x, y) waypoints of logged future trajectories -> (n, F, 3) anchors.
    Yaw of each anchor is the circular mean of its members' yaws."""
    from sklearn.cluster import KMeans
    f = future_poses.shape[1]
    km = KMeans(n_clusters=n, n_init=4, random_state=seed).fit(future_poses[:, :, :2].reshape(-1, 2 * f))
    anchors = np.zeros((n, f, 3), np.float32)
    for c in range(n):
        m = future_poses[km.labels_ == c]
        anchors[c, :, :2] = m[:, :, :2].mean(0)
        anchors[c, :, 2] = np.arctan2(np.sin(m[:, :, 2]).mean(0), np.cos(m[:, :, 2]).mean(0))
    return anchors


def footprint_points(margin=0.0, step=0.2, device="cpu"):
    """Points covering the ego footprint (rear-axle frame), (P, 2)."""
    xs = torch.arange(-EGO_REAR_OVERHANG - margin, EGO_LENGTH - EGO_REAR_OVERHANG + margin + 1e-6, step)
    ys = torch.arange(-EGO_WIDTH / 2 - margin, EGO_WIDTH / 2 + margin + 1e-6, step)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], -1).to(device)


def class_mask_2d(occ, classes, zs=Z_BODY):
    """(..., X, Y, Z) labels -> (..., X, Y) bool: any voxel of `classes` inside the z band."""
    lut = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=occ.device)
    lut[classes] = True
    return lut[occ[..., zs].long()].any(-1)


def lookup(mask, pts):
    """mask: (X, Y) bool; pts: (..., 2) metric -> (...,) bool, False outside the grid."""
    ix = torch.floor((pts[..., 0] - PC_RANGE[0]) / VOXEL_SIZE).long()
    iy = torch.floor((pts[..., 1] - PC_RANGE[1]) / VOXEL_SIZE).long()
    ok = (ix >= 0) & (ix < GRID_SHAPE[0]) & (iy >= 0) & (iy < GRID_SHAPE[1])
    return mask[ix.clamp(0, GRID_SHAPE[0] - 1), iy.clamp(0, GRID_SHAPE[1] - 1)] & ok


def place_footprint(traj, frame_poses, pts):
    """Footprint points of each trajectory waypoint, expressed in the given frames.

    traj: (N, F, 3) in frame t; frame_poses: (F, 3) pose of the frame to express each waypoint
    in (also in frame t); pts: (P, 2). Returns (N, F, P, 2).
    """
    T_t_c = se2(traj[..., 0], traj[..., 1], traj[..., 2])                       # N F 3 3
    T_t_f = se2(frame_poses[..., 0], frame_poses[..., 1], frame_poses[..., 2])  # F 3 3
    T_f_c = rel_pose(T_t_f[None], T_t_c)                                         # N F 3 3
    hom = torch.cat([pts, torch.ones_like(pts[:, :1])], -1)                      # P 3
    return torch.einsum("nfij,pj->nfpi", T_f_c, hom)[..., :2]


def collisions(traj, frame_poses, occ_frames, classes=AGENT_CLASSES, margin=0.0, ignore_ego=True):
    """Does the ego box at each waypoint overlap `classes` in the matching occupancy frame?

    traj (N, F, 3); frame_poses (F, 3); occ_frames (F, X, Y, Z). Returns (N, F) bool.
    Used both for evaluation (GT future frames + GT poses) and for privileged/static costs.

    ignore_ego: each occupancy frame was recorded with the ego at its centre, so anything
    labelled inside that box is label noise (~2% of Occ3D frames). Masking it makes the logged
    trajectory collision-free by construction without changing the rule for other plans.
    """
    pts = footprint_points(margin, device=traj.device)
    where = place_footprint(traj, frame_poses, pts)                    # N F P 2
    xs, ys = footprint_cells(0.0)
    out = torch.zeros(where.shape[:2], dtype=torch.bool, device=traj.device)
    for k in range(where.shape[1]):
        mask = class_mask_2d(occ_frames[k], classes)
        if ignore_ego:
            mask[xs, ys] = False
        out[:, k] = lookup(mask, where[:, k]).any(-1)
    return out


def offroad(traj, frame_poses, occ_frames):
    """Fraction of footprint ground columns that are a non-drivable surface, (N, F) float.
    Unobserved ground (free) counts as fine; see NON_DRIVABLE."""
    pts = footprint_points(0.0, step=0.4, device=traj.device)
    where = place_footprint(traj, frame_poses, pts)
    out = torch.zeros(where.shape[:2], device=traj.device)
    for k in range(where.shape[1]):
        bad = class_mask_2d(occ_frames[k], NON_DRIVABLE, Z_GROUND)
        out[:, k] = lookup(bad, where[:, k]).float().mean(-1)
    return out


def footprint_cells(margin=0.4):
    """Voxel index ranges of the (inflated) ego box when the ego sits at the grid centre."""
    def rng(lo, hi, axis):
        return slice(int(np.floor((lo - PC_RANGE[axis]) / VOXEL_SIZE)),
                     int(np.floor((hi - PC_RANGE[axis]) / VOXEL_SIZE)) + 1)
    xs = rng(-EGO_REAR_OVERHANG - margin, EGO_LENGTH - EGO_REAR_OVERHANG + margin, 0)
    ys = rng(-EGO_WIDTH / 2 - margin, EGO_WIDTH / 2 + margin, 1)
    return xs, ys


def predicted_costs(logits, x_off=0, y_off=0, margin=0.4):
    """Costs from world-model logits, where each predicted frame is centred on the candidate's
    own future pose, so the ego box sits at a fixed set of voxels.

    logits: (N, F, K, X', Y', Z), possibly a crop; (x_off, y_off) = crop origin in the full grid.
    Returns collision probability (N, F) = max obstacle probability inside the box, and
    off-road (N, F) = mean over box columns of the ground-level non-drivable probability.
    """
    xs, ys = footprint_cells(margin)
    xs = slice(xs.start - x_off, xs.stop - x_off)
    ys = slice(ys.start - y_off, ys.stop - y_off)
    box = logits[:, :, :, xs, ys].float().softmax(2)                    # N F K bx by Z
    p_obst = box[:, :, OBSTACLE_CLASSES][..., Z_BODY].sum(2)            # N F bx by zb
    col = p_obst.flatten(2).amax(-1)
    p_bad = box[:, :, NON_DRIVABLE][..., Z_GROUND].sum(2).amax(-1)     # N F bx by
    off = p_bad.flatten(2).mean(-1)
    return col, off
