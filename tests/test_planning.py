import torch

import _path  # noqa: F401
from occwm.constants import FREE, GRID_SHAPE, NUM_CLASSES
from occwm.planning import collisions, footprint_cells, predicted_costs

CAR = 4


def scene_with_car_at(x_m, y_m=0.0):
    occ = torch.full(GRID_SHAPE, FREE, dtype=torch.long)
    ix, iy = int((x_m + 40) / 0.4), int((y_m + 40) / 0.4)
    occ[ix - 2:ix + 3, iy - 2:iy + 3, 3:7] = CAR
    return occ


def straight(speed, n=6, dt=0.5):
    t = torch.arange(1, n + 1, dtype=torch.float32) * dt
    return torch.stack([speed * t, torch.zeros(n), torch.zeros(n)], -1)


def test_driving_into_a_parked_car_collides_and_stopping_does_not():
    occ = scene_with_car_at(12.0)[None].expand(6, -1, -1, -1)
    frames = torch.zeros(6, 3)  # static world: every waypoint checked against the same frame
    go, stop = straight(8.0), straight(0.0)
    hit = collisions(torch.stack([go, stop]), frames, occ)
    assert hit[0].any() and not hit[1].any()
    # waypoints at 4, 8, 12 m; the car spans 11.2-13.2 m and the front bumper is 3.08 m ahead
    # of the axle, so the first contact is the third waypoint
    assert hit[0].nonzero()[0].item() == 2


def test_collision_is_checked_in_the_matching_future_frame():
    """A car that is 12 m ahead in the *future* frame (ego already moved 8 m) is 20 m ahead of t."""
    occ = scene_with_car_at(12.0)[None].expand(6, -1, -1, -1)
    gt = straight(16.0)  # GT frames: ego moved 8 m per step
    plan = straight(16.0)
    assert not collisions(plan[None], gt, occ).any()  # same pose as GT: car always 12 m ahead


def test_predicted_cost_reads_the_box_at_the_grid_centre():
    xs, ys = footprint_cells(margin=0.0)
    logits = torch.full((1, 1, NUM_CLASSES, *GRID_SHAPE), -10.0)
    logits[:, :, FREE] = 10.0
    col, _ = predicted_costs(logits, margin=0.0)
    assert col.item() < 1e-3
    logits[0, 0, CAR, xs.start + 2, ys.start + 1, 5] = 20.0
    col, _ = predicted_costs(logits, margin=0.0)
    assert col.item() > 0.99


def test_footprint_cells_match_collision_checker():
    """The fixed-cell cost and the point-sampled checker must agree on what 'inside' means."""
    xs, ys = footprint_cells(margin=0.0)
    for ix, iy in [(xs.start, ys.start), (xs.stop - 1, ys.stop - 1)]:
        occ = torch.full(GRID_SHAPE, FREE, dtype=torch.long)
        occ[ix, iy, 5] = CAR
        assert collisions(torch.zeros(1, 1, 3), torch.zeros(1, 3), occ[None], ignore_ego=False).item()


def test_label_noise_inside_the_recorded_ego_box_is_ignored():
    xs, ys = footprint_cells(margin=0.0)
    occ = torch.full(GRID_SHAPE, FREE, dtype=torch.long)
    occ[xs.start + 3, ys.start + 2, 5] = CAR  # a stray voxel where the ego itself was
    assert not collisions(torch.zeros(1, 1, 3), torch.zeros(1, 3), occ[None]).item()
