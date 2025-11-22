"""
Decision Core - Main decision layer for adaptive cruise control with lane change capability.

This module implements the decision-making algorithm that determines target speed
and lane based on perception data (TTC with vehicles in different lanes).
"""

import time
import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from lane_change.perception.perception_core import TrackedVehicle, TrackedPedestrian, TrackedMovingObject, TrackedTrafficLight
from lane_change.gateway.vehicle_controller import IVehicleController
from lane_change.decision.rl_agent import RLAgentBase, StateDict
from lane_change.config.decision_config import DecisionConfig


class DecisionLayer:
    """
    Decision layer implementing adaptive cruise control with lane change capability.

    Algorithm:
    1. Safety first: Maintain safe following distance based on TTC
    2. Try to achieve target speed when possible
    3. Change lanes when stuck behind slower vehicles
    """

    def __init__(
        self,
        rl_agent: RLAgentBase,
        ego_vehicle,
        perception_core,
        target_speed: float = 70.0,  # km/h
        vehicle_ttc_threshold: float = 2.0,  # seconds
        pedestrian_ttc_threshold: float = 5.0,  # seconds (more conservative)
        cooldown_period: float = 3.0,  # seconds
        config_path: Optional[str] = None,
        kpi_tracker: Optional['KPITracker'] = None,
        config: Optional[DecisionConfig] = None
    ):
        """
        Initialize decision layer.

        Args:
            rl_agent: RL agent for lane change decisions (TrainingAgent or InferenceAgent)
            ego_vehicle: Ego vehicle reference (not currently used but kept for future extensions)
            perception_core: Perception core reference for lane information (required)
            target_speed: Desired cruising speed in km/h (deprecated, use config)
            vehicle_ttc_threshold: Safety threshold for TTC with vehicles in seconds (deprecated, use config)
            pedestrian_ttc_threshold: Safety threshold for TTC with pedestrians in seconds (deprecated, use config)
            cooldown_period: Minimum time between lane changes in seconds (deprecated, use config)
            config_path: Path to carla_config.yaml (optional)
            kpi_tracker: Optional KPI tracker for performance metrics
            config: Optional decision configuration (uses defaults if not provided)
        """
        self.rl_agent = rl_agent
        self.config = config or DecisionConfig()

        # Use config values, but allow individual parameters to override for backward compatibility
        self.target_speed = target_speed if target_speed != 70.0 else self.config.target_speed_kmh
        self.vehicle_ttc_threshold = vehicle_ttc_threshold if vehicle_ttc_threshold != 2.0 else self.config.vehicle_ttc_threshold_sec
        self.pedestrian_ttc_threshold = pedestrian_ttc_threshold if pedestrian_ttc_threshold != 5.0 else self.config.pedestrian_ttc_threshold_sec
        self.cooldown_period = cooldown_period if cooldown_period != 3.0 else self.config.cooldown_period_sec

        self.ego_vehicle = ego_vehicle
        self.perception_core = perception_core
        self.kpi_tracker = kpi_tracker

        # State tracking
        self.last_lane_change_time = 0.0
        self.stopped_at_traffic_light = False

        # Logging
        self.logger = logging.getLogger(__name__)

    def process(
        self,
        tracked_vehicles: List['TrackedMovingObject'],
        vehicle_controller: IVehicleController,
        ego_info: Dict[str, Any],
        current_time: Optional[float] = None,
        closest_traffic_light: Optional['TrackedTrafficLight'] = None
    ) -> None:
        """
        Process perception data and make control decisions.

        Args:
            tracked_vehicles: List of tracked moving objects (vehicles and pedestrians) from perception
            vehicle_controller: Vehicle controller interface
            ego_info: Dictionary with ego vehicle information:
                - 'lane_idx': Current lane index (1-based from left)
                - 'speed_kmh': Current speed in km/h
            current_time: Current timestamp (uses time.time() if None)
            closest_traffic_light: Closest traffic light ahead (if detected)
        """
        if current_time is None:
            current_time = time.time()

        # Get ego state
        ego_lane_idx = ego_info['lane_idx']
        ego_speed = ego_info['speed_kmh']

        from lane_change.perception.perception_core import TrackedVehicle, TrackedPedestrian

        # 0. Traffic light check - highest priority
        if closest_traffic_light and closest_traffic_light.distance is not None:
            # Yellow light: stop if TTC < 3 seconds
            YELLOW_TTC_THRESHOLD = 3.0

            if closest_traffic_light.color == "red" or closest_traffic_light.color == "yellow" and closest_traffic_light.ttc is not None and closest_traffic_light.ttc < YELLOW_TTC_THRESHOLD and not self.stopped_at_traffic_light:
                self.logger.info(f"RED LIGHT: Stopping")
                self.stopped_at_traffic_light = True
                vehicle_controller.emergency_stop()
                return

            elif closest_traffic_light.color == "green":
                # Green light: resume if previously stopped
                if self.stopped_at_traffic_light:
                    self.logger.info("GREEN LIGHT: Resuming")
                    self.stopped_at_traffic_light = False

        # If we were stopped at a traffic light but it's no longer detected, stay stopped
        elif self.stopped_at_traffic_light:
            self.logger.debug("Traffic light not detected, but still stopped from previous light")
            vehicle_controller.emergency_stop()
            return

        # Find closest vehicle in current lane
        next_vehicle = self._find_next_object(tracked_vehicles, ego_lane_idx, TrackedVehicle)

        # Find closest pedestrian in current lane
        next_pedestrian = self._find_next_object(tracked_vehicles, ego_lane_idx, TrackedPedestrian)

        # 1. Pedestrian safety check - highest priority (emergency stop)
        if next_pedestrian and next_pedestrian.ttc and next_pedestrian.ttc < self.pedestrian_ttc_threshold:
            self.logger.info(f"PEDESTRIAN ALERT: TTC={next_pedestrian.ttc:.1f}s < {self.pedestrian_ttc_threshold}s - EMERGENCY STOP")
            vehicle_controller.emergency_stop()
            return

        # 2. Vehicle safety check - maintain safe following distance
        if next_vehicle and next_vehicle.ttc and next_vehicle.ttc < self.vehicle_ttc_threshold and next_vehicle.velocity_estimate.speed_kmh < ego_speed:
            # Too close to vehicle ahead - match their speed
            target = next_vehicle.velocity_estimate.speed_kmh
            self.logger.info(f"Safety: TTC={next_vehicle.ttc:.1f}s < threshold, matching speed {target:.1f} km/h")
            vehicle_controller.set_speed(target)
            return

        # 3. Check if we're already at target speed
        if ego_speed >= self.target_speed - 1:
            self.logger.debug(f"At target speed ({ego_speed:.1f} >= {self.target_speed} km/h)")
            vehicle_controller.set_speed(self.target_speed)

        # 4. We're below target speed, try to reach it
        else:
            # Check if road ahead is clear or fast vehicle ahead
            if not next_vehicle and not vehicle_controller.lane_changing:
                self.logger.debug(f"No vehicle ahead, accelerating to target speed (current: {ego_speed:.1f} km/h → target: {self.target_speed:.1f} km/h)")
                vehicle_controller.set_speed(self.target_speed)

            elif next_vehicle and next_vehicle.velocity_estimate and next_vehicle.velocity_estimate.speed_kmh >= self.target_speed:
                self.logger.debug(f"Fast vehicle ahead ({next_vehicle.velocity_estimate.speed_kmh:.1f} km/h), accelerating to target")
                vehicle_controller.set_speed(self.target_speed)


            elif next_vehicle and next_vehicle.velocity_estimate and next_vehicle.velocity_estimate.speed_kmh < self.target_speed: # Both ego vehicle and next vehicle are slower than target speed. If next vehicle is slower, safety ttc check will handle it.

                next_speed = next_vehicle.velocity_estimate.speed_kmh if next_vehicle.velocity_estimate else ego_speed

                # Try to change lanes (with cooldown check)
                time_since_last_change = current_time - self.last_lane_change_time
                if time_since_last_change >= self.cooldown_period:
                    # Get number of lanes from perception
                    num_lanes = len(self.perception_core.get_detected_lanes())

                    # Query RL agent for action
                    action = self._get_agent_action(tracked_vehicles, ego_lane_idx, ego_speed, num_lanes)

                    # Check if we should override to other lane (prefer more space)
                    if self._should_override_to_other_lane(action, tracked_vehicles, ego_lane_idx):
                        action = 'right' if action == 'left' else 'left'

                    # Safety validator: Check if target lane is occupied
                    is_occupied = self._is_lane_occupied(
                        action,
                        tracked_vehicles,
                        ego_lane_idx
                    )

                    if is_occupied:
                        self.logger.info(f"Lane change to {action} rejected: Target lane occupied (safety validator)")
                        # Try alternative lane
                        alternative_action = 'right' if action == 'left' else 'left'

                        is_alternative_occupied = self._is_lane_occupied(
                            alternative_action,
                            tracked_vehicles,
                            ego_lane_idx
                        )

                        if not is_alternative_occupied:
                            self.logger.info(f"Attempting alternative lane change: {alternative_action}")
                            alternative_accepted = vehicle_controller.set_lane(alternative_action)
                            if self.kpi_tracker:
                                self.kpi_tracker.record_lane_change_attempt(
                                    accepted=alternative_accepted,
                                    next_vehicle_ttc=next_vehicle.ttc if next_vehicle else None
                                )
                            if alternative_accepted:
                                self.last_lane_change_time = current_time
                        else:
                            self.logger.info(f"Both lanes occupied, matching speed of vehicle ahead: {next_speed:.1f} km/h")
                            vehicle_controller.set_speed(next_speed)
                    else:
                        # Target lane is clear - proceed with lane change
                        self.logger.info(f"Attempting lane change: {action} (safety validator: clear)")
                        lane_change_accepted = vehicle_controller.set_lane(action)
                        if self.kpi_tracker:
                            self.kpi_tracker.record_lane_change_attempt(
                                accepted=lane_change_accepted,
                                next_vehicle_ttc=next_vehicle.ttc if next_vehicle else None
                            )
                        if lane_change_accepted:
                            self.last_lane_change_time = current_time
                        else:
                            # Primary lane rejected (possibly doesn't exist) - check alternative lane safety
                            alternative_action = 'left' if action == 'right' else 'right'
                            is_alternative_occupied = self._is_lane_occupied(
                                alternative_action,
                                tracked_vehicles,
                                ego_lane_idx
                            )

                            if not is_alternative_occupied:
                                self.logger.info(f"Primary lane rejected, attempting alternative: {alternative_action}")
                                alternative_accepted = vehicle_controller.set_lane(alternative_action)
                                if self.kpi_tracker:
                                    self.kpi_tracker.record_lane_change_attempt(
                                        accepted=alternative_accepted,
                                        next_vehicle_ttc=next_vehicle.ttc if next_vehicle else None
                                    )
                                if alternative_accepted:
                                    self.last_lane_change_time = current_time
                            else:
                                self.logger.info(f"Alternative lane {alternative_action} is occupied, staying in current lane")
                if ego_speed < next_speed: # If next is faster, we adapt and get closer to target speed.
                    self.logger.info(f"Matching speed of vehicle ahead: {next_speed:.1f} km/h")
                    vehicle_controller.set_speed(next_speed)

    def _get_closest_vehicle_in_lane(
        self,
        lane_position: str,
        tracked_vehicles: List['TrackedVehicle'],
        ego_lane_idx: int
    ) -> Optional['TrackedVehicle']:
        """
        Get the closest vehicle in a specific lane (left/right relative to ego).

        Args:
            lane_position: Lane position ("left" or "right")
            tracked_vehicles: List of all tracked vehicles
            ego_lane_idx: Ego vehicle's current lane index

        Returns:
            Closest vehicle in the lane, or None if lane is empty
        """
        # Determine target lane index
        if lane_position == "left":
            target_lane_idx = ego_lane_idx - 1
        elif lane_position == "right":
            target_lane_idx = ego_lane_idx + 1
        else:
            return None

        # Filter vehicles in target lane with valid distance
        lane_vehicles = [
            v for v in tracked_vehicles
            if v.absolute_lane_idx == target_lane_idx and v.distance is not None
        ]

        if not lane_vehicles:
            return None

        # Return closest vehicle (minimum distance)
        return min(lane_vehicles, key=lambda v: v.distance)

    def _should_override_to_other_lane(
        self,
        agent_action: str,
        tracked_vehicles: List['TrackedVehicle'],
        ego_lane_idx: int
    ) -> bool:
        """
        Check if we should override agent's action to the other lane.

        Override if:
        1. Agent's chosen lane has vehicles AND other lane is empty
        2. Agent's chosen lane has closer vehicle than other lane (prefer more space)

        Args:
            agent_action: Agent's chosen action ("left" or "right")
            tracked_vehicles: List of all tracked vehicles
            ego_lane_idx: Ego vehicle's current lane index

        Returns:
            True if we should override to the other lane, False otherwise
        """
        other_action = 'right' if agent_action == 'left' else 'left'

        # Get closest vehicles in each lane
        agent_lane_vehicle = self._get_closest_vehicle_in_lane(agent_action, tracked_vehicles, ego_lane_idx)
        other_lane_vehicle = self._get_closest_vehicle_in_lane(other_action, tracked_vehicles, ego_lane_idx)

        # Case 1: Agent's lane has vehicles AND other lane is empty
        if agent_lane_vehicle is not None and other_lane_vehicle is None:
            self.logger.info(f"Overriding {agent_action} to {other_action}: agent's lane has vehicle, other lane empty")
            return True

        # Case 2: Both lanes have vehicles, but other lane's vehicle is farther (more space)
        if agent_lane_vehicle is not None and other_lane_vehicle is not None:
            if other_lane_vehicle.distance > agent_lane_vehicle.distance:
                self.logger.info(
                    f"Overriding {agent_action} to {other_action}: "
                    f"other lane vehicle farther ({other_lane_vehicle.distance:.1f}m vs {agent_lane_vehicle.distance:.1f}m)"
                )
                return True

        return False

    def _is_lane_occupied(
        self,
        lane_position: str,
        tracked_vehicles: List['TrackedVehicle'],
        ego_lane_idx: int,
        safety_gap_threshold: float = 0.3  # meters
    ) -> bool:
        """
        Check if target lane is occupied (unsafe for lane change).

        Returns True if ANY vehicle in target lane has gap < threshold OR ttc < threshold.

        Args:
            lane_position: Target lane position ("left" or "right")
            tracked_vehicles: List of all tracked vehicles
            ego_lane_idx: Ego vehicle's current lane index
            safety_gap_threshold: Minimum safe gap in meters

        Returns:
            True if lane is occupied (unsafe), False if clear (safe)
        """
        # Determine target lane index (adjacent lane only)
        if lane_position == "left":
            target_lane_idx = ego_lane_idx - 1
        elif lane_position == "right":
            target_lane_idx = ego_lane_idx + 1
        else:
            return False  # Invalid lane position

        # Filter vehicles in the specific adjacent target lane
        target_lane_vehicles = [
            v for v in tracked_vehicles
            if v.absolute_lane_idx == target_lane_idx
        ]

        if not target_lane_vehicles:
            return False  # Lane is clear

        # Check ALL vehicles in target lane (gap OR ttc)
        for vehicle in target_lane_vehicles:
            if vehicle.position_3d is None:
                self.logger.warning(
                    f"Lane {lane_position} occupied: vehicle without 3D position (tracking lost) - unsafe"
                )
                return True

            # Use pre-computed gap from perception (None if no LiDAR points)
            if vehicle.gap is None:
                self.logger.warning(
                    f"Lane {lane_position} occupied: vehicle without gap measurement - unsafe"
                )
                return True

            if vehicle.gap < safety_gap_threshold:
                self.logger.info(
                    f"Lane {lane_position} occupied: vehicle at gap={vehicle.gap:.2f}m < threshold={safety_gap_threshold}m"
                )
                return True  # Occupied - unsafe

            # Check TTC (only if vehicle is approaching)
            if vehicle.ttc is not None and vehicle.ttc < self.vehicle_ttc_threshold:
                self.logger.info(
                    f"Lane {lane_position} occupied: vehicle TTC={vehicle.ttc:.2f}s < threshold={self.vehicle_ttc_threshold}s"
                )
                return True  # Occupied - unsafe

        return False  # All vehicles have sufficient gap and TTC - safe

    def _find_next_object(
        self,
        tracked_objects: List['TrackedMovingObject'],
        ego_lane_idx: int,
        object_type: Optional[type] = None
    ) -> Optional['TrackedMovingObject']:
        """
        Find the closest object ahead in the same lane as ego, optionally filtered by type.

        Args:
            tracked_objects: List of all tracked moving objects
            ego_lane_idx: Ego vehicle's current lane index (1-based)
            object_type: Optional class type to filter by (e.g., TrackedVehicle, TrackedPedestrian)

        Returns:
            Closest object ahead in same lane, or None if lane is clear
        """
        # Filter objects in same lane that are ahead
        objects_in_lane = []
        for obj in tracked_objects:
            # Type filter if specified
            if object_type and not isinstance(obj, object_type):
                continue

            # Check if in same lane
            if obj.absolute_lane_idx == ego_lane_idx:
                # Check if ahead (positive distance means ahead)
                if obj.distance and obj.distance > 0:
                    objects_in_lane.append(obj)

        # Find closest one
        if objects_in_lane:
            closest = min(objects_in_lane, key=lambda o: o.distance)
            ttc_str = f"{closest.ttc:.1f}s" if closest.ttc else "N/A"
            type_name = object_type.__name__ if object_type else "object"
            self.logger.debug(
                f"Next {type_name}: distance={closest.distance:.1f}m, TTC={ttc_str}"
            )
            return closest

        return None

    def _get_agent_action(self, tracked_vehicles: List['TrackedMovingObject'],
                          ego_lane_idx: int, ego_speed: float, num_lanes: int) -> str:
        """
        Get lane change action from RL agent.

        Args:
            tracked_vehicles: List of tracked moving objects
            ego_lane_idx: Current ego lane index (1-based)
            ego_speed: Current ego speed in km/h
            num_lanes: Total number of detected lanes

        Returns:
            Action string: "left" or "right"
        """
        # Build state dict for RL agent
        state_dict = StateDict(
            tracked_vehicles=tracked_vehicles,
            ego_lane_idx=ego_lane_idx,
            ego_speed=ego_speed,
            target_speed=self.target_speed,
            num_lanes=num_lanes
        )

        # Query RL agent
        action_idx = self.rl_agent.get_action(state_dict)

        # Convert to string
        return ['left', 'right'][action_idx]

    def set_target_speed(self, speed_kmh: float) -> None:
        """Update target speed."""
        self.target_speed = speed_kmh
        self.logger.info(f"Target speed updated to {speed_kmh} km/h")