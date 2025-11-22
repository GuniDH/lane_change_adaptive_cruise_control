"""
Sensor Manager - Interface for sensor management and data access.

This module defines the abstract interface for managing sensors and
accessing their data in a simulator-agnostic way.
"""

from abc import ABC, abstractmethod
from typing import Optional
from .data_types import CameraImage, LidarData, RadarData

class ISensorManager(ABC):
    """Abstract interface for sensor management and data access."""

    @abstractmethod
    def spawn_sensors(self) -> None:
        """Spawn sensors on the vehicle."""
        pass

    @abstractmethod
    def get_camera_image(self, sensor_id: str = 'front') -> Optional[CameraImage]:
        """
        Get the latest camera image from specified sensor.

        Args:
            sensor_id: Sensor identifier ('front', 'rear', etc.)

        Returns:
            CameraImage or None if not available
        """
        pass

    @abstractmethod
    def get_lidar_data(self, sensor_id: str = 'front') -> Optional[LidarData]:
        """
        Get the latest LiDAR point cloud from specified sensor.

        Args:
            sensor_id: Sensor identifier ('front', etc.)

        Returns:
            LidarData or None if not available
        """
        pass

    @abstractmethod
    def get_radar_data(self, sensor_id: str = 'front') -> Optional[RadarData]:
        """
        Get the latest radar data from specified sensor.

        Args:
            sensor_id: Sensor identifier ('front', etc.)

        Returns:
            RadarData or None if not available
        """
        pass

    @abstractmethod
    def destroy_sensors(self) -> None:
        """Clean up and destroy all sensors."""
        pass