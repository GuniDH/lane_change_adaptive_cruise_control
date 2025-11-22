"""
Collision Monitor - Safety validator for detecting collision violations.

This module provides a stateful monitor that tracks when vehicles violate
minimum safe distance thresholds for sustained periods.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class CollisionMonitor:
    """
    Monitors for collision violations using perception data.

    A collision is defined as maintaining a distance below the threshold
    for a sustained duration (default: distance < 0.3m for 3+ seconds).
    """

    def __init__(
        self,
        distance_threshold: float = 0.3,
        duration_threshold: float = 3.0
    ):
        """
        Initialize collision monitor.

        Args:
            distance_threshold: Minimum safe distance in meters (default: 0.3m)
            duration_threshold: Minimum violation duration in seconds (default: 3.0s)
        """
        self.distance_threshold = distance_threshold
        self.duration_threshold = duration_threshold
        self.violation_start: Optional[float] = None

    def check(
        self,
        tracked_vehicles: List,
        ego_lane_idx: int,
        current_time: float
    ) -> bool:
        """
        Check if collision violation occurred.

        Args:
            tracked_vehicles: List of tracked vehicles from perception
            ego_lane_idx: Current ego vehicle lane index
            current_time: Current timestamp in seconds

        Returns:
            True if minimum distance violated for duration threshold, False otherwise
        """
        from lane_change.perception.perception_core import TrackedVehicle

        if not tracked_vehicles:
            self.violation_start = None
            return False

        # Filter vehicles in ego lane only
        ego_lane_vehicles = [
            v for v in tracked_vehicles
            if isinstance(v, TrackedVehicle) and
               v.absolute_lane_idx == ego_lane_idx and
               v.distance is not None
        ]

        if not ego_lane_vehicles:
            self.violation_start = None
            return False

        min_distance = min([v.distance for v in ego_lane_vehicles])

        # Check if violating minimum distance threshold
        if min_distance <= self.distance_threshold:
            # Violation detected
            if self.violation_start is None:
                # Start tracking violation
                self.violation_start = current_time
                logger.debug(f"Collision violation started: min_distance={min_distance:.2f}m (threshold={self.distance_threshold}m)")
            else:
                # Check if violation lasted long enough
                violation_duration = current_time - self.violation_start
                if violation_duration >= self.duration_threshold:
                    logger.warning(f"Collision detected: min_distance={min_distance:.2f}m for {violation_duration:.1f}s")
                    return True
        else:
            # No violation - reset tracking
            if self.violation_start is not None:
                logger.debug(f"Collision violation cleared: min_distance={min_distance:.2f}m")
            self.violation_start = None

        return False

    def reset(self):
        """Reset violation tracking state."""
        self.violation_start = None
