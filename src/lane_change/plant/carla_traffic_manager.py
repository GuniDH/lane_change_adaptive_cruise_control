"""
CARLA Traffic Manager - Manages traffic vehicles in CARLA simulation.

This module handles traffic spawning, speed control, autopilot activation,
and cleanup for both training and open-world modes.
"""

import carla
import random
import logging
from typing import List, Tuple, Optional, Dict
from lane_change.gateway.traffic_manager import ITrafficManager

logger = logging.getLogger(__name__)


class CarlaTrafficManager(ITrafficManager):
    """
    Manages traffic vehicles in CARLA simulation.

    Encapsulates all traffic-related operations including spawning,
    speed management, autopilot control, and cleanup.
    """

    def __init__(self, world: carla.World, client: carla.Client, traffic_manager: carla.TrafficManager):
        """
        Initialize traffic manager.

        Args:
            world: CARLA world object
            client: CARLA client object
            traffic_manager: CARLA Traffic Manager instance
        """
        self.world = world
        self.client = client
        self.traffic_manager = traffic_manager

        # Private state
        self._traffic_vehicles = []  # List of vehicle IDs (for batch operations)
        self._traffic_vehicle_speeds = {}  # Map: vehicle_id -> speed_kmh
        self._traffic_vehicle_actors = {}  # Map: vehicle_id -> actor (cached for performance)

    def spawn_traffic(
        self,
        reference_location: carla.Location,
        num_vehicles: Optional[int] = None,
        distance_range: Tuple[float, float] = (15.0, 200.0),
        speed_range: Tuple[float, float] = (20.0, 60.0),
        min_spacing: float = 5.0,
        configs: Optional[List[Tuple[float, float, str]]] = None
    ) -> List[int]:
        """
        Spawn traffic vehicles around reference location.

        Two modes:
        1. Random mode: Specify num_vehicles, distance_range, speed_range
        2. Explicit mode: Provide configs list with exact (distance, speed, lane) tuples

        Args:
            reference_location: Reference location for spawning
            num_vehicles: Number of vehicles to spawn (required if configs=None)
            distance_range: (min_distance, max_distance) in meters from reference
            speed_range: (min_speed, max_speed) in km/h
            min_spacing: Minimum spacing between vehicles in meters (random mode)
            configs: List of explicit configurations [(distance, speed, lane), ...]

        Returns:
            List of spawned vehicle IDs
        """
        # Mode 1: Explicit configurations
        if configs is not None:
            logger.info(f"Spawning {len(configs)} traffic vehicles (explicit mode)...")
            spawned_ids = []

            for distance, speed, lane in configs:
                vehicle_id = self._spawn_vehicle(reference_location, distance, speed, lane)
                if vehicle_id:
                    spawned_ids.append(vehicle_id)

            logger.info(f"Spawned {len(spawned_ids)}/{len(configs)} traffic vehicles (explicit mode)")
            return spawned_ids

        # Mode 2: Random generation
        if num_vehicles is None:
            raise ValueError("Either 'configs' or 'num_vehicles' must be provided")

        logger.info(f"Spawning {num_vehicles} traffic vehicles (random mode)...")
        logger.info(f"  Distance range: {distance_range[0]:.1f}m - {distance_range[1]:.1f}m")
        logger.info(f"  Speed range: {speed_range[0]:.1f} - {speed_range[1]:.1f} km/h")
        logger.info(f"  Min spacing: {min_spacing:.1f}m")

        # Generate random distances with smart spacing
        distances = self._generate_spaced_distances(num_vehicles, distance_range, min_spacing)
        logger.info(f"Generated {len(distances)} spawn distances (sorted with spacing ≥{min_spacing}m)")

        # Spawn vehicles
        spawned_ids = []
        lanes = ['ego', 'left', 'right']

        for distance in distances:
            speed = random.uniform(speed_range[0], speed_range[1])
            lane = random.choice(lanes)

            vehicle_id = self._spawn_vehicle(reference_location, distance, speed, lane)
            if vehicle_id:
                spawned_ids.append(vehicle_id)

        logger.info(f"Successfully spawned {len(spawned_ids)}/{len(distances)} traffic vehicles")
        return spawned_ids

    def _generate_spaced_distances(
        self,
        num_vehicles: int,
        distance_range: Tuple[float, float],
        min_spacing: float
    ) -> List[float]:
        """Generate distances with minimum spacing."""
        min_dist, max_dist = distance_range
        available_range = max_dist - min_dist
        required_spacing = (num_vehicles - 1) * min_spacing

        if required_spacing > available_range:
            logger.warning(
                f"Cannot fit {num_vehicles} vehicles with {min_spacing}m spacing in range "
                f"[{min_dist}, {max_dist}]. Reducing num_vehicles."
            )
            num_vehicles = int(available_range / min_spacing) + 1

        distances = []
        for i in range(num_vehicles):
            min_allowed = min_dist + i * min_spacing
            max_allowed = max_dist - (num_vehicles - i - 1) * min_spacing

            if min_allowed <= max_allowed:
                distance = random.uniform(min_allowed, max_allowed)
                distances.append(distance)
            else:
                logger.warning(f"Cannot place vehicle {i+1}, skipping")
                break

        distances.sort()
        return distances

    def _spawn_vehicle(
        self,
        reference_location: carla.Location,
        distance_meters: float,
        speed_kmh: float,
        lane: str = 'ego'
    ) -> Optional[int]:
        """Spawn a single vehicle at specified distance, speed, and lane."""
        carla_map = self.world.get_map()
        reference_waypoint = carla_map.get_waypoint(reference_location)

        # Navigate to target distance
        target_waypoint = reference_waypoint
        remaining_distance = abs(distance_meters)
        step_size = 2.0

        if distance_meters >= 0:
            # Ahead
            while remaining_distance > 0:
                next_waypoints = target_waypoint.next(step_size)
                if not next_waypoints:
                    logger.warning("Ran out of waypoints while searching ahead")
                    return None
                target_waypoint = next_waypoints[0]
                remaining_distance -= step_size
        else:
            # Behind
            while remaining_distance > 0:
                prev_waypoints = target_waypoint.previous(step_size)
                if not prev_waypoints:
                    logger.warning("Ran out of waypoints while searching behind")
                    return None
                target_waypoint = prev_waypoints[0]
                remaining_distance -= step_size

        # Change lane if requested
        if lane == 'left':
            left_waypoint = target_waypoint.get_left_lane()
            if left_waypoint and left_waypoint.lane_type == carla.LaneType.Driving:
                target_waypoint = left_waypoint
            else:
                logger.warning("No valid left lane available, using current lane")
        elif lane == 'right':
            right_waypoint = target_waypoint.get_right_lane()
            if right_waypoint and right_waypoint.lane_type == carla.LaneType.Driving:
                target_waypoint = right_waypoint
            else:
                logger.warning("No valid right lane available, using current lane")

        # Create spawn transform
        spawn_transform = carla.Transform(
            carla.Location(
                x=target_waypoint.transform.location.x,
                y=target_waypoint.transform.location.y,
                z=target_waypoint.transform.location.z+0.5 
            ),
            target_waypoint.transform.rotation
        )

        # Get vehicle blueprint
        blueprints = self.world.get_blueprint_library().filter('*vehicle.tesla.model3')
        blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']
        blueprint = random.choice(blueprints)

        # Set attributes
        blueprint.set_attribute('role_name', 'autopilot')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn vehicle
        vehicle = self.world.try_spawn_actor(blueprint, spawn_transform)

        if vehicle:
            self.traffic_manager.set_desired_speed(vehicle, 0.0)  # Set initial speed to 0
            self._traffic_vehicles.append(vehicle.id)
            self._traffic_vehicle_speeds[vehicle.id] = speed_kmh
            self._traffic_vehicle_actors[vehicle.id] = vehicle  # Cache actor reference

            logger.debug(
                f"Spawned vehicle {vehicle.id} at distance={distance_meters:.1f}m, "
                f"speed={speed_kmh:.1f}km/h, lane={lane}"
            )
            return vehicle.id

        logger.warning(f"Failed to spawn vehicle at distance={distance_meters:.1f}m, lane={lane}")
        return None

    def apply_speeds(self) -> None:
        """Apply stored speeds to all spawned traffic vehicles (optimized with cached actors)."""
        if not self._traffic_vehicle_speeds:
            logger.debug("No traffic vehicle speeds to apply")
            return

        applied_count = 0
        for vehicle_id, speed_kmh in self._traffic_vehicle_speeds.items():
            # Use cached actor reference (no get_actor() call needed)
            vehicle = self._traffic_vehicle_actors.get(vehicle_id)
            if vehicle:
                self.traffic_manager.set_desired_speed(vehicle, speed_kmh)
                applied_count += 1
            else:
                logger.warning(f"Traffic vehicle {vehicle_id} not found in cache when applying speed")

        logger.info(f"Applied speeds to {applied_count}/{len(self._traffic_vehicle_speeds)} traffic vehicles")

    def enable_autopilot(self) -> None:
        """Enable autopilot for all traffic vehicles using batch command (synchronized)."""
        if not self._traffic_vehicles:
            logger.debug("No traffic vehicles to enable autopilot")
            return

        # Use batch command for synchronized autopilot activation
        batch = [
            carla.command.SetAutopilot(vehicle_id, True, self.traffic_manager.get_port())
            for vehicle_id in self._traffic_vehicles
        ]
        self.client.apply_batch_sync(batch)

        logger.info(f"Enabled autopilot for {len(self._traffic_vehicles)} traffic vehicles (synchronized)")

    def destroy_all(self) -> None:
        """Destroy all spawned traffic vehicles."""
        if not self._traffic_vehicles:
            logger.debug("No traffic vehicles to destroy")
            return

        logger.info(f"Destroying {len(self._traffic_vehicles)} traffic vehicles")
        batch = [carla.command.DestroyActor(vehicle_id) for vehicle_id in self._traffic_vehicles]
        self.client.apply_batch_sync(batch)

        self._traffic_vehicles.clear()
        self._traffic_vehicle_speeds.clear()
        self._traffic_vehicle_actors.clear()
        logger.info("Traffic vehicles destroyed")

    def cleanup(self) -> None:
        """Cleanup resources and destroy all traffic vehicles."""
        self.destroy_all()

    def get_vehicle_count(self) -> int:
        """Get number of currently spawned traffic vehicles."""
        return len(self._traffic_vehicles)
