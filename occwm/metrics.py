"""Occupancy and planning metrics.

Occupancy follows the OccWorld / DOME protocol: semantic mIoU over the 17 non-free classes and
binary IoU of occupied-vs-free, computed over all voxels (no camera mask), accumulated over the
whole split before dividing.
"""

import numpy as np
import torch

from .constants import EVAL_CLASSES, FREE


class IoUAccumulator:
    """Per-horizon confusion counts. Call `update(pred, gt)` with (B, T, X, Y, Z) labels."""

    def __init__(self, n_steps, device="cpu"):
        k = len(EVAL_CLASSES)
        self.inter = torch.zeros(n_steps, k, dtype=torch.float64, device=device)
        self.pred = torch.zeros_like(self.inter)
        self.gt = torch.zeros_like(self.inter)
        self.occ = torch.zeros(n_steps, 3, dtype=torch.float64, device=device)  # inter, pred, gt

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.long(), gt.long()
        for t in range(pred.shape[1]):
            p, g = pred[:, t].reshape(-1), gt[:, t].reshape(-1)
            # bincount over (gt, pred) pairs is one pass instead of a per-class loop
            cm = torch.bincount(g * 18 + p, minlength=18 * 18).reshape(18, 18).double()
            cls = torch.tensor(EVAL_CLASSES, device=cm.device)
            self.inter[t] += cm.diagonal()[cls]
            self.gt[t] += cm.sum(1)[cls]
            self.pred[t] += cm.sum(0)[cls]
            po, go = p != FREE, g != FREE
            self.occ[t] += torch.stack([(po & go).sum(), po.sum(), go.sum()]).double()

    def merge(self, other):
        for name in ("inter", "pred", "gt", "occ"):
            getattr(self, name).add_(getattr(other, name).to(getattr(self, name).device))

    def tensors(self):
        return [self.inter, self.pred, self.gt, self.occ]

    def per_class_iou(self):
        union = self.gt + self.pred - self.inter
        # classes never seen in GT nor predicted count as 1 (OccWorld convention)
        return torch.where(union > 0, self.inter / union.clamp(min=1), torch.ones_like(union))

    def miou(self):
        return (self.per_class_iou().mean(1) * 100).cpu().numpy()

    def iou(self):
        i, p, g = self.occ.unbind(1)
        return (i / (p + g - i).clamp(min=1) * 100).cpu().numpy()


def l2_stp3(pred_xy, gt_xy):
    """ST-P3 / OccWorld convention: L2 at horizon k = mean displacement over waypoints <= k.

    pred_xy, gt_xy: (N, T, 2). Returns (T,) array of cumulative-mean L2.
    """
    d = np.linalg.norm(pred_xy - gt_xy, axis=-1)                 # N T
    return (np.cumsum(d, 1) / np.arange(1, d.shape[1] + 1)).mean(0)


def rate_stp3(flags):
    """Same cumulative-mean convention for per-waypoint boolean events (N, T) -> (T,) in %."""
    f = flags.astype(np.float64)
    return (np.cumsum(f, 1) / np.arange(1, f.shape[1] + 1)).mean(0) * 100
