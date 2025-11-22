"""
Data structures for the Plant Gateway module.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class CameraImage:
    """Camera sensor data."""
    image_data: np.ndarray  # (H x W x 3) RGB image array
    width: int              # Image width in pixels
    height: int             # Image height in pixels


@dataclass
class LidarData:
    """LiDAR sensor point cloud data."""
    points: np.ndarray  # (N x 3) XYZ point cloud


@dataclass
class RadarData:
    """Radar sensor data with velocity information."""
    points: np.ndarray  # (N x 4) [velocity, azimuth, altitude, depth]


@dataclass
class BoundingBox:
    """2D bounding box in image coordinates."""
    x: float                # Top-left x coordinate
    y: float                # Top-left y coordinate
    width: float            # Box width in pixels
    height: float           # Box height in pixels
    class_label: str        # Class label from detection (includes track_id)


@dataclass
class Position3D:
    """3D position estimate from LiDAR clustering."""
    lidar_points: np.ndarray      # (N, 3) Clustered LiDAR points in vehicle frame
    center_3d: np.ndarray          # (3,) Position - mean of cluster points
    lidar_points_2d: Optional[np.ndarray] = None  # (N, 2) Projected 2D image coords [u, v]


@dataclass
class VelocityEstimate:
    """3D velocity estimate from radar fusion."""
    velocity_vector: np.ndarray  # (3,) XYZ velocity in m/s (world frame) - absolute velocity
    velocity_relative: np.ndarray  # (3,) XYZ velocity in m/s (world frame) - relative to ego
    speed_kmh: float             # Scalar speed in km/h (absolute)
    confidence: float            # Confidence score [0, 1]
    num_radar_points: int        # Number of radar points used
    radar_xyz: Optional[np.ndarray] = None  # (M, 3) Filtered radar points for visualization


@dataclass
class DetectedObject:
    """Raw detection from camera-based detector (YOLO)."""
    bounding_box: BoundingBox      # 2D bounding box in image
    confidence: float              # Detection confidence (0-1)
    class_id: int                  # YOLO class ID (2=car, 0=person, etc.)
    class_name: str                # Human-readable class name


@dataclass
class IEgoVehicle:
    """Abstract ego vehicle interface."""
    id: int                                       # Unique ID
    controller: 'IVehicleController'              # Vehicle controller interface
    sensor_manager: 'ISensorManager'             # Sensor manager interface

    def is_alive(self) -> bool:
        """Check if vehicle and all components are alive."""
        raise NotImplementedError

    def destroy(self) -> None:
        """Clean up all vehicle components."""
        raise NotImplementedError