"""GPU voxel ray caster for 3D occupancy figures (no OpenGL needed on headless nodes).

Each pixel's ray walks the voxel grid with a vectorised 3D-DDA (Amanatides & Woo) until it
hits an occupied voxel. Faces get Lambert shading by their normal plus a thin dark edge, which
gives the familiar "voxel" look of occupancy papers.
"""

import math

import numpy as np
import torch

from .constants import (CLASS_COLORS, EGO_LENGTH, EGO_REAR_OVERHANG, EGO_WIDTH, FREE, GRID_SHAPE,
                        PC_RANGE, VOXEL_SIZE)

EGO_CLASS = 18
PALETTE = np.concatenate([CLASS_COLORS, [[245, 245, 245]]]).astype(np.float32) / 255.0


def add_ego(occ):
    """Paint the ego vehicle into a (X, Y, Z) label grid as EGO_CLASS (for display only)."""
    occ = occ.clone()
    x0 = int((-EGO_REAR_OVERHANG - PC_RANGE[0]) / VOXEL_SIZE)
    x1 = int((EGO_LENGTH - EGO_REAR_OVERHANG - PC_RANGE[0]) / VOXEL_SIZE)
    y0 = int((-EGO_WIDTH / 2 - PC_RANGE[1]) / VOXEL_SIZE)
    y1 = int((EGO_WIDTH / 2 - PC_RANGE[1]) / VOXEL_SIZE)
    occ[x0:x1 + 1, y0:y1 + 1, 3:6] = EGO_CLASS
    return occ


def camera_rays(eye, target, width, height, fov_deg, device):
    eye, target = torch.tensor(eye, device=device), torch.tensor(target, device=device)
    fwd = target - eye
    fwd = fwd / fwd.norm()
    right = torch.linalg.cross(fwd, torch.tensor([0.0, 0.0, 1.0], device=device))
    right = right / right.norm()
    up = torch.linalg.cross(right, fwd)
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    u = (torch.arange(width, device=device) + 0.5) / width * 2 - 1
    v = (torch.arange(height, device=device) + 0.5) / height * 2 - 1
    v, u = torch.meshgrid(v, u, indexing="ij")
    d = fwd * f + right * u[..., None] - up * (v[..., None] * height / width)
    return eye.expand_as(d).reshape(-1, 3), (d / d.norm(dim=-1, keepdim=True)).reshape(-1, 3)


@torch.no_grad()
def render(occ, eye=(-28.0, 0.0, 22.0), target=(10.0, 0.0, -2.0), width=640, height=400,
           fov_deg=55, hide=(FREE,), max_steps=700, light=(-0.4, 0.3, 1.0), bg=(0.07, 0.07, 0.09)):
    """occ: (X, Y, Z) int labels (torch, any device) -> (H, W, 3) uint8 image."""
    device = occ.device if torch.is_tensor(occ) else "cpu"
    occ = torch.as_tensor(occ, device=device).long()
    shape = torch.tensor(GRID_SHAPE, device=device)
    visible = torch.ones(EGO_CLASS + 1, dtype=torch.bool, device=device)
    visible[list(hide)] = False
    solid = visible[occ]

    o, d = camera_rays(eye, target, width, height, fov_deg, device)
    lo = torch.tensor(PC_RANGE[:3], device=device)
    o = (o - lo) / VOXEL_SIZE                                    # voxel units
    d = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)

    # enter the grid box (slab test)
    t0 = (0 - o) / d
    t1 = (shape - o) / d
    t_enter = torch.minimum(t0, t1).amax(-1).clamp(min=0)
    t_exit = torch.maximum(t0, t1).amin(-1)
    alive = t_enter < t_exit
    p = o + d * (t_enter[:, None] + 1e-4)
    idx = torch.floor(p).long().clamp(torch.zeros(3, dtype=torch.long, device=device), shape - 1)
    step = torch.sign(d).long()
    t_max = ((idx + (step > 0).long()).float() - o) / d
    t_delta = (1.0 / d).abs()
    normal_axis = torch.minimum(t0, t1).argmax(-1)               # face we entered through
    t_cur = t_enter.clone()

    n = o.shape[0]
    hit_cls = torch.full((n,), -1, dtype=torch.long, device=device)
    hit_axis = torch.zeros(n, dtype=torch.long, device=device)
    hit_t = torch.zeros(n, device=device)
    for _ in range(max_steps):
        inside = alive & (idx >= 0).all(-1) & (idx < shape).all(-1)
        if not inside.any():
            break
        ii = idx.clamp(torch.zeros(3, dtype=torch.long, device=device), shape - 1)
        is_hit = inside & solid[ii[:, 0], ii[:, 1], ii[:, 2]] & (hit_cls < 0)
        hit_cls[is_hit] = occ[ii[is_hit, 0], ii[is_hit, 1], ii[is_hit, 2]]
        hit_axis[is_hit] = normal_axis[is_hit]
        hit_t[is_hit] = t_cur[is_hit]
        alive = inside & (hit_cls < 0)
        axis = t_max.argmin(-1)
        ar = torch.arange(n, device=device)
        t_cur = t_max[ar, axis]
        idx[ar, axis] += step[ar, axis]
        t_max[ar, axis] += t_delta[ar, axis]
        normal_axis = axis

    img = torch.tensor(bg, device=device).expand(n, 3).clone()
    m = hit_cls >= 0
    if m.any():
        pal = torch.from_numpy(PALETTE).to(device)
        col = pal[hit_cls[m]]
        ax = hit_axis[m]
        normal = torch.zeros(int(m.sum()), 3, device=device)
        normal[torch.arange(len(ax), device=device), ax] = -torch.sign(d[m, :].gather(1, ax[:, None])[:, 0])
        lgt = torch.tensor(light, device=device)
        lgt = lgt / lgt.norm()
        shade = 0.45 + 0.55 * (normal @ lgt).clamp(min=0)
        # dark voxel edges: distance of the hit point to the face border
        ph = o[m] + d[m] * hit_t[m, None]
        frac = ph - torch.floor(ph + 1e-4)
        frac = torch.minimum(frac, 1 - frac)
        frac[torch.arange(len(ax), device=device), ax] = 1.0
        edge = frac.amin(-1) < 0.06
        shade = torch.where(edge, shade * 0.55, shade)
        fog = torch.exp(-hit_t[m] * VOXEL_SIZE / 160.0)
        img[m] = (col * shade[:, None]) * fog[:, None] + torch.tensor(bg, device=device) * (1 - fog[:, None])
    return (img.reshape(height, width, 3).clamp(0, 1) * 255).byte().cpu().numpy()
