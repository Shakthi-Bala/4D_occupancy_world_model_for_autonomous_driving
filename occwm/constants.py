"""Occ3D-nuScenes grid layout and label set."""

import numpy as np

# Voxel grid: [X, Y, Z] in the ego frame (x forward, y left, z up), origin at the rear axle.
GRID_SHAPE = (200, 200, 16)
VOXEL_SIZE = 0.4
PC_RANGE = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)  # xmin, ymin, zmin, xmax, ymax, zmax

# Keyframes are annotated at 2 Hz.
DT = 0.5

CLASS_NAMES = [
    "others", "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
    "pedestrian", "traffic_cone", "trailer", "truck", "driveable_surface", "other_flat",
    "sidewalk", "terrain", "manmade", "vegetation", "free",
]
NUM_CLASSES = len(CLASS_NAMES)  # 18
FREE = 17
DRIVEABLE = 11

# Classes scored in mIoU. Follows OccWorld / DOME: every semantic class except `free`.
EVAL_CLASSES = list(range(17))

# Things the ego must not drive into. AGENTS follows the ST-P3 / OccWorld convention
# (vehicles + pedestrians); OBSTACLES adds static hazards.
AGENT_CLASSES = [2, 3, 4, 5, 6, 7, 9, 10]
OBSTACLE_CLASSES = AGENT_CLASSES + [0, 1, 8, 15, 16]

# nuScenes ego vehicle (Renault Zoe) footprint relative to the rear axle, metres.
EGO_LENGTH = 4.084
EGO_WIDTH = 1.85
EGO_REAR_OVERHANG = 1.0  # rear axle -> rear bumper
# Body height band used for collision checks (z in the ego frame).
EGO_Z_RANGE = (0.2, 1.8)

# Palette used by every figure in the repo (RGB 0-255), close to the Occ3D / OccWorld colours.
CLASS_COLORS = np.array([
    [0, 0, 0],        # others
    [255, 120, 50],   # barrier
    [255, 192, 203],  # bicycle
    [255, 255, 0],    # bus
    [0, 150, 245],    # car
    [0, 255, 255],    # construction_vehicle
    [200, 180, 0],    # motorcycle
    [255, 0, 0],      # pedestrian
    [255, 240, 150],  # traffic_cone
    [135, 60, 0],     # trailer
    [160, 32, 240],   # truck
    [255, 0, 255],    # driveable_surface
    [139, 137, 137],  # other_flat
    [75, 0, 75],      # sidewalk
    [150, 240, 80],   # terrain
    [230, 230, 250],  # manmade
    [0, 175, 0],      # vegetation
    [255, 255, 255],  # free
], dtype=np.uint8)


def voxel_centers_1d(axis):
    """Metric coordinate of every voxel centre along axis 0 (x), 1 (y) or 2 (z)."""
    lo, n = PC_RANGE[axis], GRID_SHAPE[axis]
    return lo + VOXEL_SIZE * (np.arange(n) + 0.5)
