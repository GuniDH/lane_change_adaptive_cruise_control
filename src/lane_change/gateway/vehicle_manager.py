"""
Vehicle Manager - Interface for client-side vehicle management.

This module defines the abstract interface for managing vehicles from the client side,
including connection, ego vehicle spawning, and vehicle control.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class IVehicleManager(ABC):
    """Abstract interface for client-side vehicle management."""

    @abstractmethod
    def connect(self, map_name: Optional[str] = None) -> None:
        """
        Connect to the simulation world.

        Args:
            map_name: Ignored in multi-client mode (server loads map)
        """
        pass

    @abstractmethod
    def spawn_ego_vehicle(self) -> Any:
        """
        Spawn the ego vehicle in the simulation.

        Returns:
            Vehicle actor object (simulator-specific type)
        """
        pass

    @abstractmethod
    def get_world(self) -> Any:
        """
        Get the simulation world object.

        Returns:
            World object (simulator-specific type)
        """
        pass

    @abstractmethod
    def get_map(self) -> Any:
        """
        Get the simulation map object.

        Returns:
            Map object (simulator-specific type)
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up client's spawned actors and resources."""
        pass