"""SE(2) pose helpers and BEV warping.

Ego poses in nuScenes are full SE(3), but the ego always sits on the ground (z = 0 in the
global frame), so planning and warping are done in SE(2): (x, y, yaw).

A pose matrix T_a_b maps points from frame b into frame a. `rel_pose(T_w_t, T_w_k)` returns
T_t_k: the pose of frame k expressed in frame t, which is also the transform that maps a
point given in frame-k coordinates into frame-t coordinates.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .constants import FREE, PC_RANGE


def quat_to_yaw(q):
    """Yaw of a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def se2(x, y, yaw):
    """(..., ) arrays -> (..., 3, 3) homogeneous SE(2) matrices. Works on numpy or torch."""
    lib = torch if torch.is_tensor(x) else np
    c, s = lib.cos(yaw), lib.sin(yaw)
    one, zero = lib.ones_like(x), lib.zeros_like(x)
    rows = [lib.stack([c, -s, x], -1), lib.stack([s, c, y], -1), lib.stack([zero, zero, one], -1)]
    return lib.stack(rows, -2)


def se2_to_xyyaw(T):
    lib = torch if torch.is_tensor(T) else np
    return lib.stack([T[..., 0, 2], T[..., 1, 2], lib.arctan2(T[..., 1, 0], T[..., 0, 0])], -1)


def rel_pose(T_w_a, T_w_b):
    """Pose of frame b expressed in frame a (T_a_b)."""
    lib = torch if torch.is_tensor(T_w_a) else np
    return lib.linalg.inv(T_w_a) @ T_w_b


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _sampling_grid(T_src_tgt, height, width):
    """grid_sample grid that pulls each target BEV cell from the source frame.

    BEV tensors are laid out (..., X, Y): rows index x (forward), columns index y (left).
    With align_corners=False the normalised coordinate of a metric position is simply
    x / 40 (rows) and y / 40 (columns), because the grid spans [-40, 40) m.
    """
    half_x, half_y = PC_RANGE[3], PC_RANGE[4]
    dev, dt = T_src_tgt.device, T_src_tgt.dtype
    xs = (torch.arange(height, device=dev, dtype=dt) + 0.5) / height * 2 * half_x - half_x
    ys = (torch.arange(width, device=dev, dtype=dt) + 0.5) / width * 2 * half_y - half_y
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([gx, gy, torch.ones_like(gx)], -1).reshape(1, -1, 3)  # (1, H*W, 3)
    src = pts @ T_src_tgt.transpose(1, 2)  # (B, H*W, 3)
    grid = torch.stack([src[..., 1] / half_y, src[..., 0] / half_x], -1)
    return grid.reshape(-1, height, width, 2)


def warp_bev_features(feat, T_src_tgt, mode="bilinear"):
    """Resample a (B, C, X, Y) BEV feature map from a source ego frame into a target frame.

    T_src_tgt: (B, 3, 3) pose of the target frame expressed in the source frame.
    Cells that fall outside the source grid are zero.
    """
    grid = _sampling_grid(T_src_tgt.to(torch.float32), feat.shape[-2], feat.shape[-1])
    return F.grid_sample(feat, grid.to(feat.dtype), mode=mode, padding_mode="zeros",
                         align_corners=False)


def warp_occupancy(occ, T_src_tgt):
    """Resample a (B, X, Y, Z) semantic label grid into another ego frame (nearest neighbour).

    This is the static-world assumption: every voxel keeps its world position. Columns move
    as a whole (roll/pitch are ignored). Cells with no source data become `free`.
    """
    b, x, y, z = occ.shape
    feat = (occ.permute(0, 3, 1, 2).float() + 1.0)  # shift so zero padding marks "unknown"
    out = warp_bev_features(feat, T_src_tgt, mode="nearest").round().long() - 1
    out[out < 0] = FREE
    return out.permute(0, 2, 3, 1).to(occ.dtype)
