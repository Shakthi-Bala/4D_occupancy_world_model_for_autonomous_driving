"""Imitation prior over a fixed trajectory vocabulary.

Scores each anchor trajectory from the current scene (VAE latent of the present occupancy) and
the ego's recent motion. It plays the role of the "what would a human do here" term in the
planner; the world model then vetoes candidates that are predicted to be unsafe.
"""

import torch
import torch.nn as nn

from .vae import ResBlock
from .world_model import pose_features


class TrajectoryPrior(nn.Module):
    def __init__(self, n_anchors, latent_ch=32, n_past=5, width=128, hidden=512):
        super().__init__()
        self.bev = nn.Sequential(                                   # 50 -> 25 -> 13 -> 7
            nn.Conv2d(latent_ch, width, 3, padding=1), ResBlock(width),
            nn.Conv2d(width, width, 3, stride=2, padding=1), ResBlock(width),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), ResBlock(2 * width),
            nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1), ResBlock(2 * width),
            nn.GroupNorm(32, 2 * width), nn.SiLU(), nn.Flatten(),
            nn.Linear(2 * width * 7 * 7, hidden))
        self.ego = nn.Sequential(nn.Linear(4 * (n_past - 1) + 4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.SiLU(), nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, n_anchors))

    def forward(self, z_now, past_poses):
        """z_now: (B, C, 50, 50) normalised latent; past_poses: (B, P, 3) in the current frame."""
        past = past_poses[:, :-1]
        vel = past_poses[:, -1] - past_poses[:, -2]  # last displacement, a proxy for speed
        ego = torch.cat([pose_features(past).flatten(1), pose_features(vel)], -1)
        return self.head(torch.cat([self.bev(z_now), self.ego(ego)], -1))


def soft_targets(anchors, gt, tau=0.5):
    """anchors (K, F, 3), gt (B, F, 3) -> (B, K) soft labels from mean waypoint distance."""
    d = (anchors[None, :, :, :2] - gt[:, None, :, :2]).norm(dim=-1).mean(-1)
    return (-d / tau).softmax(-1)
