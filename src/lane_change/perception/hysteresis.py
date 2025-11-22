"""
Generic hysteresis filter for temporal smoothing of tracked object attributes.

This module provides a reusable hysteresis filter that prevents flickering when
object attributes (lane, color, etc.) temporarily change for brief periods.
"""

import logging
from typing import Dict, Any, Set

logger = logging.getLogger(__name__)


class HysteresisFilter:
    """
    Generic hysteresis filter for temporal smoothing.

    Applies hysteresis to any field on any tracked object to prevent rapid fluctuations.
    Only changes the stable value after N consecutive frames show the new value.
    """

    def __init__(self, threshold_frames: int):
        """
        Initialize hysteresis filter.

        Args:
            threshold_frames: Number of consecutive frames required to confirm a change
        """
        self.threshold_frames = threshold_frames
        self.state: Dict[int, Dict[str, Any]] = {}

    def apply(self, track_id: int, detected_value: Any, field_name: str) -> Any:
        """
        Apply hysteresis to any field on any tracked object.

        Args:
            track_id: Object track ID (must be >= 0)
            detected_value: Value detected in current frame
            field_name: Name of the field (for logging)

        Returns:
            Stable value (with hysteresis applied)
        """
        if track_id < 0:
            return detected_value

        # Initialize state
        if track_id not in self.state:
            self.state[track_id] = {
                'stable': detected_value,
                'candidate': detected_value,
                'count': 0
            }
            return detected_value

        state = self.state[track_id]

        # Value matches stable - reset
        if detected_value == state['stable']:
            state['candidate'] = detected_value
            state['count'] = 0
            return state['stable']

        # Value matches candidate - increment
        if detected_value == state['candidate']:
            state['count'] += 1
            if state['count'] >= self.threshold_frames:
                state['stable'] = detected_value
                state['count'] = 0
                logger.debug(f"{field_name} change confirmed for track {track_id}: {state['stable']} -> {detected_value}")
                return detected_value
            return state['stable']

        # New candidate
        state['candidate'] = detected_value
        state['count'] = 1
        return state['stable']

    def cleanup(self, active_track_ids: Set[int]):
        """
        Remove stale entries for tracks that are no longer active.

        Args:
            active_track_ids: Set of currently active track IDs
        """
        stale = set(self.state.keys()) - active_track_ids
        for track_id in stale:
            del self.state[track_id]
