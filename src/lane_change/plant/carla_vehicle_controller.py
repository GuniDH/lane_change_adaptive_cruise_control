"""
CARLA Vehicle Controller - Direct control implementation.
"""

import agents.navigation.basic_agent as bs
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import get_speed as _get_speed
from lane_change.gateway.vehicle_controller import IVehicleController


class CarlaVehicleController(bs.BasicAgent, IVehicleController):
    """CARLA vehicle controller with direct speed and lane control."""

    def __init__(self, vehicle, perception_core=None):
        bs.BasicAgent.__init__(self, vehicle)
        self.perception_core = perception_core
        self.lane_changing = False
        self.lane_change_start_time = None

    def set_speed(self, speed: float) -> None:
        """Set target speed in km/h."""
        bs.BasicAgent.set_target_speed(self, speed)

    def set_lane(self, direction: str) -> bool:
        """Set target lane number (1-based from leftmost)."""
        if self.lane_changing:
            return False

        import time
        self.lane_changing = True
        self.lane_change_start_time = time.time()

        lane_change_accepted = self.lane_change(
            direction=direction,
            same_lane_time=0.5,
            other_lane_time=3,
            lane_change_time=2
        )

        if not lane_change_accepted:
            self.lane_changing = False
            self.lane_change_start_time = None
            return False

        return True

    def emergency_stop(self) -> None:
        """Emergency stop - immediately stop vehicle."""
        self.set_speed(0.0)

    def get_speed(self) -> float:
        """Get current speed in km/h."""
        return _get_speed(self._vehicle)
    
    def _straight_path(self):
        # Build a straight-ahead path
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        if not current_waypoint:
            return

        plan = []
        wp = current_waypoint

        # Build path with at least current waypoint
        plan.append((current_waypoint, RoadOption.LANEFOLLOW))

        # Add future waypoints
        for _ in range(20):
            next_wps = wp.next(5.0)
            if next_wps:
                wp = next_wps[0]
                plan.append((wp, RoadOption.LANEFOLLOW))
            else:
                break

        # Always set plan (at minimum has current waypoint)
        self._local_planner.set_global_plan(plan, stop_waypoint_creation=False, clean_queue=True)

    def lane_change(self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=2):
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        """
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution
        )
        if path:
            self.set_global_plan(path)
            return True
        return False
        '''
        else:
            self._straight_path()
            return False'''

    def run_step(self):
        """Check completion and start next maneuver (Overriding BasicAgent)."""
        # Check if we've completed the lane change or timeout (failsafe)
        if self.lane_changing:
            import time
            # Timeout after 7 seconds (failsafe)
            if time.time() - self.lane_change_start_time > 7.0:
                self.lane_changing = False
                self.lane_change_start_time = None
                self._straight_path()
            # Normal completion when done with path
            elif self.done():
                self.lane_changing = False
                self.lane_change_start_time = None
                self._straight_path()
        # Ensure path exists (failsafe for empty queue)
        elif self.done():
            self._straight_path()

        hazard_detected = False
        # Retrieve all relevant actors
        vehicle_speed = self.get_speed() / 3.6
        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._lights_list, max_tlight_distance)
        if False and affected_by_tlight:
            hazard_detected = True

        try:
            control = self._local_planner.run_step()
        except IndexError:
            # Waypoint queue empty - regenerate path and return no control
            self._straight_path()
            control = self._vehicle.get_control()

        if hazard_detected:
            control = self.add_emergency_stop(control)
        return control

    
    def _generate_lane_change_path(self, waypoint, direction='left', distance_same_lane=10,
                                   distance_other_lane=25, lane_change_distance=25,
                                   check=True, lane_changes=1, step_distance=2):
        """
        Override to prevent autonomous lane changes when disabled.
        """
        if not self.lane_changing:
            return []  # Return empty path to prevent autonomous lane changes
        return super()._generate_lane_change_path(
            waypoint, direction, distance_same_lane, distance_other_lane,
            lane_change_distance, check, lane_changes, step_distance
        )