import numpy as np
import torch

import _path  # noqa: F401
from occwm.constants import FREE
from occwm.metrics import IoUAccumulator, l2_stp3, rate_stp3


def random_occ(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    occ = torch.full(shape, FREE, dtype=torch.long)
    mask = torch.rand(shape, generator=g) < 0.2
    occ[mask] = torch.randint(0, 17, (int(mask.sum()),), generator=g)
    return occ


def test_perfect_prediction_scores_100():
    gt = random_occ((2, 3, 20, 20, 4))
    acc = IoUAccumulator(3)
    acc.update(gt, gt)
    assert np.allclose(acc.miou(), 100) and np.allclose(acc.iou(), 100)


def test_all_free_prediction_scores_zero_iou():
    gt = random_occ((1, 1, 20, 20, 4))
    acc = IoUAccumulator(1)
    acc.update(torch.full_like(gt, FREE), gt)
    assert acc.iou()[0] == 0


def test_miou_matches_bruteforce():
    gt, pred = random_occ((2, 1, 16, 16, 4), 1), random_occ((2, 1, 16, 16, 4), 2)
    acc = IoUAccumulator(1)
    acc.update(pred, gt)
    ious = []
    for c in range(17):
        i = ((pred == c) & (gt == c)).sum().item()
        u = ((pred == c) | (gt == c)).sum().item()
        ious.append(i / u if u else 1.0)
    assert np.isclose(acc.miou()[0], 100 * np.mean(ious))


def test_stp3_cumulative_means():
    pred = np.zeros((1, 3, 2))
    gt = np.array([[[1.0, 0], [2.0, 0], [3.0, 0]]])
    assert np.allclose(l2_stp3(pred, gt), [1.0, 1.5, 2.0])
    assert np.allclose(rate_stp3(np.array([[0, 1, 1]])), [0, 50, 200 / 3])
