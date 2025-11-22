"""
Configuration dataclass for decision module.

This module centralizes all decision-related configuration parameters
to provide a single source of truth for tuning the decision pipeline.
"""

from dataclasses import dataclass


@dataclass
class DecisionConfig:
    """Configuration for decision module."""

    # Target speeds
    target_speed_kmh: float = 70.0
    speed_tolerance_kmh: float = 5.0

    # TTC thresholds
    vehicle_ttc_threshold_sec: float = 2.0
    pedestrian_ttc_threshold_sec: float = 5.0

    # Lane change settings
    cooldown_period_sec: float = 3.0
