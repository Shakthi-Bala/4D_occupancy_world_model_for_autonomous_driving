"""Segmentation losses for voxel logits."""

import torch
import torch.nn.functional as F

from .constants import NUM_CLASSES


def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jac = 1.0 - inter / union
    jac[1:] = jac[1:] - jac[:-1]
    return jac


def lovasz_softmax(logits, labels, max_voxels=400_000):
    """Lovasz-softmax (Berman et al. 2018) over classes present in the batch.

    logits: (B, C, ...) labels: (B, ...). A random voxel subset keeps the sort affordable.
    """
    probs = logits.float().softmax(1).movedim(1, -1).reshape(-1, NUM_CLASSES)
    labels = labels.reshape(-1).long()
    if labels.numel() > max_voxels:
        keep = torch.randint(0, labels.numel(), (max_voxels,), device=labels.device)
        probs, labels = probs[keep], labels[keep]
    losses = []
    for c in labels.unique():
        fg = (labels == c).float()
        err = (fg - probs[:, c]).abs()
        err_sorted, perm = torch.sort(err, descending=True)
        losses.append(torch.dot(err_sorted, lovasz_grad(fg[perm])))
    return torch.stack(losses).mean()


def occupancy_loss(logits, labels, class_weight=None, lovasz_weight=1.0):
    ce = F.cross_entropy(logits.float(), labels.long(), weight=class_weight)
    lz = lovasz_softmax(logits, labels) if lovasz_weight > 0 else logits.new_zeros(())
    return ce + lovasz_weight * lz, {"ce": ce.detach(), "lovasz": lz.detach()}
