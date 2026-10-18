"""Occupancy VAE: compresses a 200x200x16 semantic grid into a 50x50xC BEV latent.

Height is folded into channels (each voxel gets a learned class embedding, and the 16 height
cells of a column are concatenated), so the network is a 2D conv net over the BEV plane. That
keeps the latent a plain image-like grid the world model can warp with ego motion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import GRID_SHAPE, NUM_CLASSES


def norm(c):
    return nn.GroupNorm(32 if c % 32 == 0 else 8, c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout=None):
        super().__init__()
        cout = cout or cin
        self.body = nn.Sequential(norm(cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1),
                                  norm(cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.skip(x) + self.body(x)


class OccVAE(nn.Module):
    def __init__(self, latent_ch=32, emb_dim=16, voxel_feat=16, widths=(128, 256, 256), blocks=2):
        super().__init__()
        self.z_cells = GRID_SHAPE[2]
        self.latent_ch, self.voxel_feat = latent_ch, voxel_feat
        self.class_emb = nn.Embedding(NUM_CLASSES, emb_dim)

        enc = [nn.Conv2d(emb_dim * self.z_cells, widths[0], 3, padding=1)]
        for i, w in enumerate(widths):
            cin = widths[max(i - 1, 0)]
            enc += [ResBlock(cin, w)] + [ResBlock(w) for _ in range(blocks - 1)]
            if i < len(widths) - 1:
                enc.append(nn.Conv2d(w, w, 3, stride=2, padding=1))  # 200 -> 100 -> 50
        enc += [norm(widths[-1]), nn.SiLU(), nn.Conv2d(widths[-1], 2 * latent_ch, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)

        rw = widths[::-1]
        dec = [nn.Conv2d(latent_ch, rw[0], 3, padding=1)]
        for i, w in enumerate(rw):
            cin = rw[max(i - 1, 0)]
            dec += [ResBlock(cin, w)] + [ResBlock(w) for _ in range(blocks - 1)]
            if i < len(rw) - 1:
                dec += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(w, w, 3, padding=1)]
        dec += [norm(rw[-1]), nn.SiLU(), nn.Conv2d(rw[-1], voxel_feat * self.z_cells, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)
        # Per-voxel head; the height embedding tells it which z-cell a feature belongs to.
        self.height_emb = nn.Parameter(torch.zeros(1, voxel_feat, 1, 1, self.z_cells))
        self.head = nn.Sequential(nn.Conv3d(voxel_feat, voxel_feat, 1), nn.SiLU(),
                                  nn.Conv3d(voxel_feat, NUM_CLASSES, 1))

    def encode(self, occ):
        """occ: (B, X, Y, Z) int labels -> (mean, logvar), each (B, C, X/4, Y/4)."""
        b, x, y, z = occ.shape
        e = self.class_emb(occ.long())                       # B X Y Z E
        e = e.permute(0, 3, 4, 1, 2).reshape(b, -1, x, y)    # B (Z E) X Y
        mean, logvar = self.encoder(e).chunk(2, dim=1)
        return mean, logvar.clamp(-30, 20)

    def decode_features(self, z):
        """z: (B, C, 50, 50) -> per-voxel features (B, voxel_feat, X, Y, Z)."""
        f = self.decoder(z)
        b, _, x, y = f.shape
        f = f.reshape(b, self.z_cells, self.voxel_feat, x, y).permute(0, 2, 3, 4, 1)
        return f + self.height_emb

    def decode(self, z):
        """z: (B, C, 50, 50) -> logits (B, NUM_CLASSES, X, Y, Z)."""
        return self.head(self.decode_features(z))

    def forward(self, occ, sample=True):
        mean, logvar = self.encode(occ)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
        return self.decode(z), mean, logvar


def kl_divergence(mean, logvar):
    return 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).mean()
