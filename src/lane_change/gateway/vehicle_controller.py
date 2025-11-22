"""
Vehicle Controller - Interface for direct vehicle control.
"""

from abc import ABC, abstractmethod


class IVehicleController(ABC):
    """Interface for direct vehicle control with target speed and lane."""

    @abstractmethod
    def set_speed(self, speed: float) -> None:
        """Set target speed in km/h."""
        pass

    @abstractmethod
    def set_lane(self, lane: int) -> None:
        """Set target lane number (1-based from leftmost)."""
        pass

    @abstractmethod
    def emergency_stop(self) -> None:
        """Emergency stop - immediately stop vehicle."""
        pass