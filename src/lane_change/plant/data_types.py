"""
CARLA-specific data structures for the Plant module.
"""

from dataclasses import dataclass
from lane_change.gateway.data_types import IEgoVehicle


@dataclass
class CarlaEgoVehicle(IEgoVehicle):
    """CARLA-specific ego vehicle implementation."""
    carla_actor: 'carla.Actor'                    # CARLA vehicle actor (specific to CARLA)

    def is_alive(self) -> bool:
        """Check if vehicle and all components are alive."""
        return (self.carla_actor and self.carla_actor.is_alive and
                self.controller is not None and
                self.sensor_manager is not None)

    def destroy(self) -> None:
        """Clean up all vehicle components."""
        if self.sensor_manager:
            self.sensor_manager.destroy_sensors()
        if self.carla_actor and self.carla_actor.is_alive:
            self.carla_actor.destroy()