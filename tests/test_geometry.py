import numpy as np
import torch

import _path  # noqa: F401
from occwm.constants import FREE, GRID_SHAPE
from occwm.geometry import rel_pose, se2, se2_to_xyyaw, warp_occupancy


def test_rel_pose_roundtrip():
    a, b = se2(3.0, -2.0, 0.7), se2(-1.0, 5.0, -2.1)
    T_ab = rel_pose(a, b)
    assert np.allclose(a @ T_ab, b)
    assert np.allclose(se2_to_xyyaw(se2(1.0, 2.0, 0.3)), [1.0, 2.0, 0.3])


def test_identity_warp_is_lossless():
    occ = torch.randint(0, 18, (1, *GRID_SHAPE), dtype=torch.uint8)
    out = warp_occupancy(occ, torch.eye(3)[None])
    assert torch.equal(out, occ)


def test_forward_motion_shifts_scene_backwards():
    """Ego drives 4 m forward: an obstacle 10 m ahead should appear 6 m ahead."""
    occ = torch.full((1, *GRID_SHAPE), FREE, dtype=torch.uint8)
    ix = int((10.0 + 40) / 0.4)          # voxel containing x = 10 m
    occ[0, ix, 100, 4] = 4
    out = warp_occupancy(occ, torch.tensor(se2(4.0, 0.0, 0.0), dtype=torch.float32)[None])
    xs = torch.nonzero(out[0] == 4)[:, 0]
    assert len(xs) == 1 and xs[0].item() == ix - 10


def test_left_turn_rotates_scene_clockwise():
    """Ego yaws +90 deg in place: an obstacle 10 m to its left is now 10 m straight ahead."""
    occ = torch.full((1, *GRID_SHAPE), FREE, dtype=torch.uint8)
    iy = int((10.0 + 40) / 0.4)
    occ[0, 100, iy, 4] = 4
    out = warp_occupancy(occ, torch.tensor(se2(0.0, 0.0, np.pi / 2), dtype=torch.float32)[None])
    x, y = torch.nonzero(out[0] == 4)[0, :2].tolist()
    assert abs(x - iy) <= 1 and abs(y - 100) <= 1
