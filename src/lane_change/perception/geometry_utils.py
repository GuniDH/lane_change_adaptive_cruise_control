"""
Geometric utility functions for perception module.

This module provides shared geometric calculations used across the perception pipeline,
including bounding box computations and distance calculations between objects.
"""

import numpy as np
from typing import Tuple


def compute_bbox_from_lidar_cluster(lidar_points: np.ndarray) -> Tuple[float, float]:
    """
    Compute 2D bounding box dimensions from LiDAR cluster points.

    Args:
        lidar_points: (N, 3) LiDAR points in vehicle frame

    Returns:
        (length, width) in meters
    """
    if len(lidar_points) == 0:
        return 4.5, 1.8  # Default car dimensions

    # Extract X, Y coordinates (ignore Z)
    x_coords = lidar_points[:, 0]
    y_coords = lidar_points[:, 1]

    # Compute bounding box dimensions
    length = float(np.max(x_coords) - np.min(x_coords))
    width = float(np.max(y_coords) - np.min(y_coords))

    # Clamp to reasonable minimums (LiDAR might underestimate)
    length = max(length, 3.5)  # Minimum car length
    width = max(width, 1.5)    # Minimum car width

    return length, width


def compute_min_distance_between_aabb(
    ego_center: np.ndarray,
    ego_half_length: float,
    ego_half_width: float,
    obj_center: np.ndarray,
    obj_half_length: float,
    obj_half_width: float
) -> float:
    """
    Compute minimum distance between two axis-aligned 2D bounding boxes.

    Both boxes are axis-aligned in vehicle frame (same yaw assumption).

    Args:
        ego_center: (2,) Ego bbox center [x, y] in vehicle frame
        ego_half_length: Ego half-length in meters (X direction)
        ego_half_width: Ego half-width in meters (Y direction)
        obj_center: (2,) Object bbox center [x, y] in vehicle frame
        obj_half_length: Object half-length in meters (X direction)
        obj_half_width: Object half-width in meters (Y direction)

    Returns:
        Minimum distance between boxes in meters
    """
    # Compute bounding box edges
    ego_x_min = ego_center[0] - ego_half_length
    ego_x_max = ego_center[0] + ego_half_length
    ego_y_min = ego_center[1] - ego_half_width
    ego_y_max = ego_center[1] + ego_half_width

    obj_x_min = obj_center[0] - obj_half_length
    obj_x_max = obj_center[0] + obj_half_length
    obj_y_min = obj_center[1] - obj_half_width
    obj_y_max = obj_center[1] + obj_half_width

    # Compute gap distance in X direction
    if obj_x_min > ego_x_max:
        dx = obj_x_min - ego_x_max  # Object to the right (ahead in vehicle frame)
    elif obj_x_max < ego_x_min:
        dx = ego_x_min - obj_x_max  # Object to the left (behind in vehicle frame)
    else:
        dx = 0  # Overlap in X

    # Compute gap distance in Y direction
    if obj_y_min > ego_y_max:
        dy = obj_y_min - ego_y_max  # Object above (right side in vehicle frame)
    elif obj_y_max < ego_y_min:
        dy = ego_y_min - obj_y_max  # Object below (left side in vehicle frame)
    else:
        dy = 0  # Overlap in Y

    # Euclidean distance (shortest distance between boxes)
    distance = np.sqrt(dx**2 + dy**2)

    return float(distance)
