"""
Gateway module for lane change autonomous system.

This module provides the abstraction layer between the perception/behavior
modules and the plant (simulator/real vehicle). Contains focused interfaces
following SOLID principles for swapping simulators and detection algorithms.
"""

from .vehicle_manager import IVehicleManager
from .sensor_manager import ISensorManager
from .vehicle_controller import IVehicleController
from .traffic_manager import ITrafficManager
from .detector import Detector
from .velocity_estimator import IVelocityEstimator
from .data_types import (
    CameraImage, LidarData, RadarData,
    DetectedObject, BoundingBox, VelocityEstimate
)

__all__ = [
    # SOLID-compliant interfaces
    'IVehicleManager', 'ISensorManager', 'IVehicleController', 'ITrafficManager',
    # Detection interface
    'Detector',
    # Velocity estimation interface
    'IVelocityEstimator',
    # Data types
    'CameraImage', 'LidarData', 'RadarData',
    'DetectedObject', 'BoundingBox', 'VelocityEstimate'
]
