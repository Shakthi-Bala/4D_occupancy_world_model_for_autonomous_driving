"""Trajectory-conditioned occupancy world model: ego-warp prior + learned residual.

Given 2 s of past occupancy and a planned ego trajectory (6 future SE(2) poses), forecast the
semantic occupancy of each future frame in that frame's own ego coordinates.

Ego motion explains most of what changes in the ego frame, and a parameter-free warp of the
current frame along the trajectory is already a strong forecaster. So the model is built
around that warp instead of learning it:

  1. Past frames are encoded by the frozen VAE into 50x50 BEV latents.
  2. For every future pose, all past latents are warped into that future ego frame. Static
     structure now lines up across time, and moving agents show up as trails.
  3. A space-time transformer, modulated (adaLN) by the trajectory, predicts a residual on top
     of the warped current latent: other agents' motion, newly revealed areas.
  4. The VAE decoder turns latents into voxel features. A full-resolution refinement head
     fuses them with the warped current occupancy, so static detail is copied exactly instead
     of passing through the latent bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..constants import GRID_SHAPE, NUM_CLASSES
from ..geometry import rel_pose, se2, warp_bev_features, warp_occupancy
from .vae import OccVAE, ResBlock


def pose_features(p):
    """(..., 3) x, y, yaw -> (..., 4) scaled x, y, cos, sin."""
    return torch.cat([p[..., :2] / 10.0, torch.cos(p[..., 2:]), torch.sin(p[..., 2:])], -1)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(b, n, d))


class SpaceTimeBlock(nn.Module):
    """Spatial attention within each frame, temporal attention per BEV cell, then an MLP.
    Every sub-layer is adaLN-zero modulated by the per-frame trajectory embedding."""

    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in range(3)])
        self.space = Attention(dim, heads)
        self.time = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(approximate="tanh"),
                                 nn.Linear(mlp_ratio * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        """x: (B, F, N, D) tokens, c: (B, F, D) per-frame condition."""
        b, f, n, d = x.shape
        mods = self.ada(c)[:, :, None].chunk(9, dim=-1)  # each (B, F, 1, D)

        h = modulate(self.norms[0](x), mods[0], mods[1])
        x = x + mods[2] * self.space(h.reshape(b * f, n, d)).reshape(b, f, n, d)

        h = modulate(self.norms[1](x), mods[3], mods[4])
        h = h.permute(0, 2, 1, 3).reshape(b * n, f, d)
        x = x + mods[5] * self.time(h).reshape(b, n, f, d).permute(0, 2, 1, 3)

        h = modulate(self.norms[2](x), mods[6], mods[7])
        return x + mods[8] * self.mlp(h)


class RefineHead(nn.Module):
    """Full-resolution fusion of decoded voxel features with the warped current occupancy."""

    def __init__(self, voxel_feat, width=192):
        super().__init__()
        z = GRID_SHAPE[2]
        self.z, self.voxel_feat = z, voxel_feat
        self.net = nn.Sequential(nn.Conv2d((voxel_feat + NUM_CLASSES) * z, width, 3, padding=1),
                                 ResBlock(width), ResBlock(width),
                                 nn.GroupNorm(32, width), nn.SiLU(),
                                 nn.Conv2d(width, voxel_feat * z, 3, padding=1))
        self.head = nn.Sequential(nn.Conv3d(2 * voxel_feat, voxel_feat, 1), nn.SiLU(),
                                  nn.Conv3d(voxel_feat, NUM_CLASSES, 1))
        # Per-class gain on the warped labels: at init the model reproduces the warp baseline,
        # and training only has to learn where the static-world assumption is wrong.
        self.skip = nn.Parameter(torch.full((NUM_CLASSES,), 6.0))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, feat, warped_occ):
        """feat: (N, Cf, X, Y, Z) decoder features; warped_occ: (N, X, Y, Z) labels."""
        n, cf, x, y, z = feat.shape
        onehot = F.one_hot(warped_occ.long(), NUM_CLASSES).to(feat.dtype)       # N X Y Z K
        fold = torch.cat([feat.permute(0, 4, 1, 2, 3).reshape(n, z * cf, x, y),
                          onehot.permute(0, 3, 4, 1, 2).reshape(n, z * NUM_CLASSES, x, y)], 1)
        r = self.net(fold).reshape(n, z, cf, x, y).permute(0, 2, 3, 4, 1)
        skip = (onehot * self.skip).permute(0, 4, 1, 2, 3)
        return self.head(torch.cat([feat, r], 1)) + skip


class OccWorldModel(nn.Module):
    def __init__(self, vae: OccVAE, latent_shift, latent_scale, n_past=5, n_future=6, dim=512,
                 depth=8, heads=8, patch=2, finetune_decoder=True, checkpoint_decode=True, warp=True):
        super().__init__()
        # warp=False is the ablation: no geometric warping anywhere (identity transforms), so
        # the trajectory reaches the model only through adaLN conditioning, as in DOME.
        self.warp = warp
        self.vae = vae
        for p in vae.parameters():
            p.requires_grad = False
        if finetune_decoder:  # vae.head stays frozen: RefineHead replaces it
            for p in vae.decoder.parameters():
                p.requires_grad = True
            vae.height_emb.requires_grad = True
        self.register_buffer("latent_shift", torch.as_tensor(latent_shift).float().view(1, -1, 1, 1))
        self.register_buffer("latent_scale", torch.as_tensor(latent_scale).float().view(1, -1, 1, 1))
        self.n_past, self.n_future, self.patch = n_past, n_future, patch
        self.checkpoint_decode = checkpoint_decode
        c = vae.latent_ch
        self.lat_hw = GRID_SHAPE[0] // 4
        n_tok = (self.lat_hw // patch) ** 2

        in_ch = n_past * (c + 1)  # every warped past latent + its validity mask
        self.embed = nn.Linear(in_ch * patch * patch, dim)
        self.pos = nn.Parameter(torch.randn(1, 1, n_tok, dim) * 0.02)
        self.frame = nn.Parameter(torch.randn(1, n_future, 1, dim) * 0.02)
        cond_in = 4 + 4 + 4 * (n_past - 1)  # future pose, step motion, past ego poses
        self.cond = nn.Sequential(nn.Linear(cond_in, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.frame_cond = nn.Parameter(torch.zeros(1, n_future, dim))
        self.blocks = nn.ModuleList([SpaceTimeBlock(dim, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, c * patch * patch)
        nn.init.zeros_(self.out.weight)  # start as the pure warp prior
        nn.init.zeros_(self.out.bias)
        self.refine = RefineHead(vae.voxel_feat)

    # ---- latent helpers -------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, occ):
        """(N, X, Y, Z) labels -> normalised latent (N, C, 50, 50)."""
        mean, _ = self.vae.encode(occ)
        return (mean.float() - self.latent_shift) / self.latent_scale

    def decode_features(self, z):
        return self.vae.decode_features(z * self.latent_scale + self.latent_shift)

    def patchify(self, x):  # (N, C, H, W) -> (N, H/p * W/p, C p p)
        n, c, h, w = x.shape
        p = self.patch
        x = x.reshape(n, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(n, (h // p) * (w // p), c * p * p)

    def unpatchify(self, x, c):
        n, _, _ = x.shape
        p, g = self.patch, self.lat_hw // self.patch
        x = x.reshape(n, g, g, c, p, p).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(n, c, g * p, g * p)

    # ---- context (shared across candidate trajectories) -----------------------------------
    def encode_context(self, occ_past, past_poses):
        """occ_past: (B, P, X, Y, Z); past_poses: (B, P, 3) in the frame of the current (last)
        past frame. Everything the forecast needs that does not depend on the plan."""
        b, p = occ_past.shape[:2]
        z = self.encode(occ_past.flatten(0, 1)).reshape(b, p, -1, self.lat_hw, self.lat_hw)
        return {"z_past": z, "past_poses": past_poses, "occ_now": occ_past[:, -1]}

    def expand_context(self, ctx, k):
        """Repeat each context k times (for scoring k candidate trajectories per scene)."""
        return {key: v.repeat_interleave(k, 0) for key, v in ctx.items()}

    # ---- forecasting ----------------------------------------------------------------------
    def forward_latent(self, ctx, fut_poses):
        """fut_poses: (B, F, 3) in the current frame. Returns predicted normalised latents
        (B, F, C, 50, 50) and the warp prior latents."""
        z_past, past_poses = ctx["z_past"], ctx["past_poses"]
        b, p, c, h, w = z_past.shape
        f = fut_poses.shape[1]

        T_t_p = se2(past_poses[..., 0], past_poses[..., 1], past_poses[..., 2])   # B P 3 3
        T_t_k = se2(fut_poses[..., 0], fut_poses[..., 1], fut_poses[..., 2])      # B F 3 3
        T_p_k = rel_pose(T_t_p[:, None], T_t_k[:, :, None])                       # B F P 3 3
        if not self.warp:
            T_p_k = torch.eye(3, device=T_p_k.device).expand_as(T_p_k)

        src = torch.cat([z_past, torch.ones_like(z_past[:, :, :1])], 2)           # B P C+1 H W
        src = src[:, None].expand(b, f, p, c + 1, h, w).reshape(-1, c + 1, h, w)
        warped = warp_bev_features(src, T_p_k.reshape(-1, 3, 3))
        warped = warped.reshape(b, f, p, c + 1, h, w)
        prior = warped[:, :, -1, :c]                                              # warped z_t

        tokens = self.embed(self.patchify(warped.reshape(b * f, p * (c + 1), h, w)))
        x = tokens.reshape(b, f, -1, tokens.shape[-1]) + self.pos + self.frame

        prev = torch.cat([torch.zeros_like(fut_poses[:, :1]), fut_poses[:, :-1]], 1)
        step = fut_poses - prev
        past = pose_features(past_poses[:, :-1]).flatten(1)[:, None].expand(b, f, -1)
        cvec = self.cond(torch.cat([pose_features(fut_poses), pose_features(step), past], -1))
        cvec = cvec + self.frame_cond[:, :f]

        for blk in self.blocks:
            x = checkpoint(blk, x, cvec, use_reentrant=False) if self.training else blk(x, cvec)
        shift, scale = self.out_ada(cvec)[:, :, None].chunk(2, -1)
        delta = self.out(modulate(self.out_norm(x), shift, scale))
        delta = self.unpatchify(delta.reshape(b * f, -1, delta.shape[-1]), c).reshape(b, f, c, h, w)
        return prior + delta, prior

    def warp_now(self, ctx, fut_poses):
        """Current occupancy warped into every future frame: (B, F, X, Y, Z)."""
        b, f = fut_poses.shape[:2]
        T = se2(fut_poses[..., 0], fut_poses[..., 1], fut_poses[..., 2]).reshape(-1, 3, 3)
        if not self.warp:
            T = torch.eye(3, device=T.device).expand_as(T)
        occ = ctx["occ_now"][:, None].expand(b, f, *ctx["occ_now"].shape[1:]).flatten(0, 1)
        return warp_occupancy(occ, T).reshape(b, f, *occ.shape[1:])

    def _decode_one(self, z, warped):
        return self.refine(self.decode_features(z), warped)

    def decode(self, z, warped, chunk=6):
        """z: (B, F, C, 50, 50), warped: (B, F, X, Y, Z) -> logits (B, F, K, X, Y, Z)."""
        b, f = z.shape[:2]
        zf, wf = z.flatten(0, 1), warped.flatten(0, 1)
        outs = []
        for i in range(0, zf.shape[0], chunk):
            args = (zf[i:i + chunk], wf[i:i + chunk])
            if self.training and self.checkpoint_decode:
                outs.append(checkpoint(self._decode_one, *args, use_reentrant=False))
            else:
                outs.append(self._decode_one(*args))
        out = torch.cat(outs)
        return out.reshape(b, f, *out.shape[1:])

    def forward(self, occ_past, poses):
        """occ_past: (B, P, X, Y, Z); poses: (B, P+F, 3) all in the frame of the last past frame.
        Returns logits (B, F, K, X, Y, Z) and predicted latents."""
        p = self.n_past
        ctx = self.encode_context(occ_past, poses[:, :p])
        fut = poses[:, p:]
        z, _ = self.forward_latent(ctx, fut)
        return self.decode(z, self.warp_now(ctx, fut)), z

    @torch.no_grad()
    def forecast(self, occ_past, poses):
        return self.forward(occ_past, poses)[0]


def load_world_model(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    vae = OccVAE(latent_ch=a["latent_ch"])
    wm = OccWorldModel(vae, ck["latent_shift"], ck["latent_scale"], dim=a["dim"], depth=a["depth"],
                       heads=a["heads"], warp=not a.get("no_warp", False))
    wm.load_state_dict(ck["model"])
    return wm.to(device).eval()
