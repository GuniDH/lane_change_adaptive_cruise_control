"""
Plant module for CARLA simulator integration.

This module provides CARLA-specific implementations following SOLID principles,
including focused controllers for world, vehicle, sensor, and traffic management.
"""

from .carla_vehicle_manager import CarlaVehicleManager
from .carla_vehicle_controller import CarlaVehicleController
from .carla_sensor_manager import CarlaSensorManager
from .carla_traffic_manager import CarlaTrafficManager

__all__ = [
    'CarlaVehicleManager',
    'CarlaVehicleController',
    'CarlaSensorManager',
    'CarlaTrafficManager'
]