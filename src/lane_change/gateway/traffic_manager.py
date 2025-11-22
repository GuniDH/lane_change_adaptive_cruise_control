"""
Traffic Manager - Interface for traffic vehicle management.

This module defines the abstract interface for managing traffic vehicles,
including spawning, speed control, autopilot, and cleanup.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict


class ITrafficManager(ABC):
    """Abstract interface for traffic vehicle management."""

    @abstractmethod
    def spawn_traffic(
        self,
        reference_location,
        num_vehicles: Optional[int] = None,
        distance_range: Tuple[float, float] = (15.0, 200.0),
        speed_range: Tuple[float, float] = (20.0, 60.0),
        min_spacing: float = 5.0,
        configs: Optional[List[Tuple[float, float, str]]] = None
    ) -> List[int]:
        """
        Spawn traffic vehicles.

        Args:
            reference_location: Reference location for spawning
            num_vehicles: Number of vehicles to spawn (random mode)
            distance_range: Distance range from reference (min, max) in meters
            speed_range: Speed range (min, max) in km/h
            min_spacing: Minimum spacing between vehicles in meters
            configs: Explicit configurations [(distance, speed, lane), ...] (explicit mode)

        Returns:
            List of spawned vehicle IDs
        """
        pass

    @abstractmethod
    def apply_speeds(self) -> None:
        """Apply stored speeds to all spawned traffic vehicles."""
        pass

    @abstractmethod
    def enable_autopilot(self) -> None:
        """Enable autopilot for all traffic vehicles (synchronized batch operation)."""
        pass

    @abstractmethod
    def destroy_all(self) -> None:
        """Destroy all spawned traffic vehicles."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources and destroy all traffic vehicles."""
        pass

    @abstractmethod
    def get_vehicle_count(self) -> int:
        """Get number of currently spawned traffic vehicles."""
        pass
