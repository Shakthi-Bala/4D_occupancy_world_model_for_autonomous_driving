"""Occ3D-nuScenes sequences backed by the memory-mapped arrays from scripts/prepare_data.py."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import rel_pose, se2, se2_to_xyyaw

DEFAULT_ROOT = "/scratch/sbalamurugan/data/occ3d_nuscenes/packed"


class Occ3DIndex:
    """Frame/scene index plus lazily opened label memmap. Cheap to share between datasets."""

    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root)
        idx = np.load(self.root / "index.npz")
        self.pose = idx["pose"]                  # (N, 3) global x, y, yaw
        self.token = idx["token"]
        self.scene_id = idx["scene_id"]
        self.scene_name = idx["scene_name"]
        self.scene_split = idx["scene_split"]
        self.scene_start = idx["scene_start"]
        self.scene_len = idx["scene_len"]
        self._sem = None

    @property
    def semantics(self):
        if self._sem is None:  # open per worker process, after fork
            self._sem = np.load(self.root / "semantics.npy", mmap_mode="r")
        return self._sem

    def scenes(self, split):
        return [i for i, s in enumerate(self.scene_split) if s == split]

    def windows(self, split, n_past, n_future):
        """Global row of the 'current' frame for every window that fits inside one scene."""
        rows = []
        for s in self.scenes(split):
            start, n = self.scene_start[s], self.scene_len[s]
            rows.extend(range(start + n_past - 1, start + n - n_future))
        return np.array(rows, np.int64)

    def relative_poses(self, rows, ref_row):
        """SE(2) (x, y, yaw) of each row's ego frame expressed in the frame of `ref_row`."""
        T_ref = se2(*self.pose[ref_row])
        T = se2(self.pose[rows, 0], self.pose[rows, 1], self.pose[rows, 2])
        return se2_to_xyyaw(rel_pose(T_ref[None], T)).astype(np.float32)


class OccFrames(Dataset):
    """Single frames, for training the occupancy VAE."""

    def __init__(self, index, split, flip=False):
        self.index, self.flip = index, flip
        self.rows = np.concatenate([np.arange(index.scene_start[s], index.scene_start[s] + index.scene_len[s])
                                    for s in index.scenes(split)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        occ = np.array(self.index.semantics[self.rows[i]])
        if self.flip and np.random.rand() < 0.5:
            occ = occ[:, ::-1].copy()  # mirror left/right (y -> -y)
        return torch.from_numpy(occ)


class OccSequences(Dataset):
    """Windows of `n_past` frames (ending at the current frame t) plus `n_future` frames.

    Returns
        occ:   (n_past + n_future, 200, 200, 16) uint8 labels, each frame in its own ego frame
        poses: (n_past + n_future, 3) float32 (x, y, yaw) of each frame in the frame of t
        row:   global row of t (lets evaluation code look up scene / token)
    """

    def __init__(self, index, split, n_past=5, n_future=6, flip=False, stride=1):
        self.index, self.n_past, self.n_future, self.flip = index, n_past, n_future, flip
        self.rows = index.windows(split, n_past, n_future)[::stride]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        t = self.rows[i]
        rows = np.arange(t - self.n_past + 1, t + self.n_future + 1)
        occ = np.array(self.index.semantics[rows[0]:rows[-1] + 1])
        poses = self.index.relative_poses(rows, t)
        if self.flip and np.random.rand() < 0.5:
            occ = occ[:, :, ::-1].copy()
            poses[:, 1] *= -1
            poses[:, 2] *= -1
        return {"occ": torch.from_numpy(occ), "poses": torch.from_numpy(poses), "row": int(t)}
