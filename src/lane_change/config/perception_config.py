"""
Configuration dataclass for perception module.

This module centralizes all perception-related configuration parameters
to provide a single source of truth for tuning the perception pipeline.
"""

from dataclasses import dataclass


@dataclass
class PerceptionConfig:
    """Configuration for perception module."""

    # Hysteresis settings
    lane_hysteresis_frames: int = 7
    color_hysteresis_frames: int = 4

    # Lane detection optimization
    lane_detection_interval: int = 2

    # YOLO detection settings
    yolo_confidence_threshold: float = 0.40
    yolo_iou_threshold: float = 0.5

    # Lane detection settings (from lane_detector.py)
    lane_bbox_expand_x_ratio: float = 0.2
    lane_bbox_expand_y_ratio: float = 0.1
    lane_trim_top_px: int = 50
    lane_distance_threshold: int = 30
    lane_smooth_factor: float = 0.3

    # Velocity estimation settings (from carla_velocity_estimator.py)
    moving_threshold_mps: float = 0.3
    max_depth_meters: float = 50.0
    min_points_threshold: int = 5

    # Clustering settings
    cluster_eps: float = 0.5
    cluster_min_samples: int = 3

    # Parallel processing
    max_workers: int = 4
