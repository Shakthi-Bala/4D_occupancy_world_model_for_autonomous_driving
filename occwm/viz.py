"""Fast top-down (BEV) rendering of semantic occupancy, used for training monitors and plots."""

import numpy as np
from PIL import Image, ImageDraw

from .constants import (CLASS_COLORS, EGO_LENGTH, EGO_REAR_OVERHANG, EGO_WIDTH, FREE, PC_RANGE,
                        VOXEL_SIZE)


def bev_image(occ, scale=2, bg=(18, 18, 22)):
    """(X, Y, Z) labels -> (H, W, 3) uint8: colour of the highest non-free voxel per column,
    shaded by height. Image up = +x (forward), image left = +y (left), ego at the centre."""
    occ = np.asarray(occ)
    z = occ.shape[2]
    occupied = occ != FREE
    top = np.where(occupied.any(2), z - 1 - np.argmax(occupied[:, :, ::-1], axis=2), -1)
    cls = np.take_along_axis(occ, np.clip(top, 0, None)[..., None], 2)[..., 0]
    rgb = CLASS_COLORS[cls].astype(np.float32)
    shade = 0.55 + 0.45 * np.clip(top, 0, None) / (z - 1)
    rgb = rgb * shade[..., None]
    rgb[top < 0] = bg
    img = rgb[::-1, ::-1].astype(np.uint8)  # rows: +x up; cols: +y left
    if scale != 1:
        img = np.kron(img, np.ones((scale, scale, 1), np.uint8))
    return img


def metric_to_pixel(x, y, scale=2):
    """Ego-frame metres -> pixel (col, row) in a bev_image of the given scale."""
    row = (PC_RANGE[3] - x) / VOXEL_SIZE * scale
    col = (PC_RANGE[4] - y) / VOXEL_SIZE * scale
    return col, row


def draw_ego(img, scale=2, color=(255, 255, 255)):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    x0, x1 = -EGO_REAR_OVERHANG, EGO_LENGTH - EGO_REAR_OVERHANG
    c0, r0 = metric_to_pixel(x1, EGO_WIDTH / 2, scale)
    c1, r1 = metric_to_pixel(x0, -EGO_WIDTH / 2, scale)
    d.rectangle([c0, r0, c1, r1], outline=color, width=max(1, scale // 2))
    return np.asarray(im)


def draw_traj(img, xy, color, scale=2, width=2, dots=True):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    pts = [metric_to_pixel(x, y, scale) for x, y in np.concatenate([[[0, 0]], np.asarray(xy)[:, :2]])]
    d.line(pts, fill=tuple(color), width=width)
    if dots:
        for c, r in pts[1:]:
            d.ellipse([c - width, r - width, c + width, r + width], fill=tuple(color))
    return np.asarray(im)


def label(img, text, pos=(6, 4), color=(235, 235, 235)):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text(pos, text, fill=color)
    return np.asarray(im)


def hstack(imgs, gap=4, bg=0):
    h = max(i.shape[0] for i in imgs)
    out = [np.pad(i, ((0, h - i.shape[0]), (0, gap), (0, 0)), constant_values=bg) for i in imgs]
    return np.concatenate(out, 1)[:, :-gap]


def vstack(imgs, gap=4, bg=0):
    w = max(i.shape[1] for i in imgs)
    out = [np.pad(i, ((0, gap), (0, w - i.shape[1]), (0, 0)), constant_values=bg) for i in imgs]
    return np.concatenate(out, 0)[:-gap]
