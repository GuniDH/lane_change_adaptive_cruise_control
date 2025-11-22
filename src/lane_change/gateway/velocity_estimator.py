"""
Velocity Estimator - Interface for estimating vehicle velocities from sensors.

This module defines the abstract interface for velocity estimation using
multi-sensor fusion (camera, LiDAR, radar).
"""

from abc import ABC, abstractmethod
from typing import Optional
from .data_types import DetectedObject, CameraImage, LidarData, RadarData, VelocityEstimate
import numpy as np


class IVelocityEstimator(ABC):
    """Abstract interface for velocity estimation from sensor fusion."""

    @abstractmethod
    def estimate_velocity(
        self,
        detection: DetectedObject,
        camera_image: CameraImage,
        radar_data: RadarData,
        ego_velocity_world: np.ndarray,
        ego_rotation: float = 0.0,
        track_id: Optional[int] = None,
        radar_xyz_all: np.ndarray = None,
        radar_image_uv: np.ndarray = None
    ) -> Optional[VelocityEstimate]:
        """
        Estimate 3D velocity for a detected object using radar.

        Args:
            detection: YOLO detection with 2D bounding box
            camera_image: Camera image for calibration reference
            radar_data: Radar measurements
            ego_velocity_world: Ego vehicle velocity vector (3,) in m/s in world frame
            ego_rotation: Ego vehicle yaw rotation in radians
            track_id: Optional track ID for temporal smoothing

        Returns:
            VelocityEstimate or None if estimation fails
        """
        pass
