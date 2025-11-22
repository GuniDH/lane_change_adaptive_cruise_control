"""
RL Agent - Interface for lane change decision making.

Provides unified interface for both training and inference modes.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class StateDict:
    """Container for raw state information from perception."""
    tracked_vehicles: List  # List of TrackedVehicle objects
    ego_lane_idx: int       # Current lane index (1-based)
    ego_speed: float        # km/h
    target_speed: float     # km/h
    num_lanes: int          # Total number of detected lanes


class RLAgentBase(ABC):
    """Abstract base class for RL agents."""

    @abstractmethod
    def get_action(self, state_dict: StateDict) -> int:
        """
        Return action index given state.

        Args:
            state_dict: Raw state information from perception

        Returns:
            Action index: 0=left, 1=right
        """
        pass


class TrainingAgent(RLAgentBase):
    """RL agent for training - uses epsilon-greedy exploration with callback to trainer."""

    def __init__(self, trainer=None):
        """
        Initialize training agent.

        Args:
            trainer: Reference to LaneChangeTrainer for callbacks (optional)
        """
        self.trainer = trainer

    def get_action(self, state_dict: StateDict) -> int:
        """
        Select action using trainer's epsilon-greedy policy.

        Args:
            state_dict: State information from decision layer

        Returns:
            Action index (0=left, 1=right)
        """
        # Notify trainer that agent is being queried
        if self.trainer:
            action = self.trainer.on_agent_queried(state_dict)
            return action
        else:
            # Fallback: random action if no trainer
            import random
            return random.randint(0, 1)


class InferenceAgent(RLAgentBase):
    """RL agent for inference - uses trained DQN model."""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize inference agent with trained model.

        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)

        # Import here to avoid circular dependency
        from .dqn_network import DuelingDQN

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        state_size = checkpoint['state_size']

        self.model = DuelingDQN(state_size=state_size, action_size=2)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def get_action(self, state_dict: StateDict) -> int:
        """
        Predict action using trained model.

        Args:
            state_dict: State information from perception

        Returns:
            Action index: 0=left, 1=right
        """
        # Build observation
        obs = self._build_observation(state_dict)

        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Get Q-values (deterministic)
        with torch.no_grad():
            q_values = self.model(obs_tensor)

        # Return action with highest Q-value
        action = q_values.argmax(dim=1).item()
        return action

    def _build_observation(self, state_dict: StateDict) -> np.ndarray:
        """
        Build observation vector from state dictionary.

        Args:
            state_dict: Raw state information

        Returns:
            Flattened observation vector
        """
        return build_observation(state_dict)


def build_observation(state_dict: StateDict) -> np.ndarray:
    """
    Build observation vector from state dictionary.

    State representation:
    - Grid: (5 lanes, 10 cells, 3 features) = 150 values
    - Ego speed deficit: 1 value
    - Total: 151 values

    Grid structure:
    - 5 lanes (max highway lanes, middle is ego, padded if fewer)
    - 10 cells per lane (5 ahead, 5 behind)
    - 3 features per cell: [ttc_danger, distance_norm, speed_norm]

    Cell distance ranges:
    - Ahead: [0-10m, 10-20m, 20-40m, 40-60m, 60-100m]
    - Behind: [-10-0m, -20--10m, -40--20m, -60--40m, -100--60m]

    Args:
        state_dict: Raw state information from perception

    Returns:
        Observation vector of shape (151,)
    """
    # Initialize grid (5 lanes, 10 cells, 3 features)
    grid = np.zeros((5, 10, 3), dtype=np.float32)

    # Cell boundaries (meters relative to ego)
    cell_boundaries = [
        # Ahead cells
        (0, 10), (10, 20), (20, 40), (40, 60), (60, 100),
        # Behind cells
        (-10, 0), (-20, -10), (-40, -20), (-60, -40), (-100, -60)
    ]

    # Compute lane offset (ego lane should be at index 2)
    ego_lane_idx = state_dict.ego_lane_idx  # 1-based
    num_lanes = state_dict.num_lanes

    # Map absolute lane indices to grid indices (centered on ego at index 2)
    lane_offset = 2 - (ego_lane_idx - 1)  # Convert to 0-based, then offset

    # Process each tracked vehicle
    for vehicle in state_dict.tracked_vehicles:
        # Get vehicle properties
        distance = vehicle.distance  # Signed distance (+ ahead, - behind)
        ttc = vehicle.ttc
        speed = vehicle.velocity_estimate.speed_kmh if vehicle.velocity_estimate else 0.0
        absolute_lane = vehicle.absolute_lane_idx  # 1-based

        # Map to grid lane index
        grid_lane_idx = absolute_lane - 1 + lane_offset  # Convert to 0-based, apply offset

        # Skip if outside grid bounds
        if grid_lane_idx < 0 or grid_lane_idx >= 5:
            continue

        # Find cell index
        cell_idx = None
        for idx, (min_dist, max_dist) in enumerate(cell_boundaries):
            if min_dist <= distance < max_dist:
                cell_idx = idx
                break

        # Skip if outside cell range
        if cell_idx is None:
            continue

        # Normalize features
        ttc_danger = 1.0 - min(ttc / 10.0, 1.0) if ttc is not None else 0.0
        distance_norm = 1.0 - min(abs(distance) / 100.0, 1.0)
        speed_norm = speed / 120.0

        # Update cell (take maximum if multiple vehicles in same cell)
        grid[grid_lane_idx, cell_idx, 0] = max(grid[grid_lane_idx, cell_idx, 0], ttc_danger)
        grid[grid_lane_idx, cell_idx, 1] = max(grid[grid_lane_idx, cell_idx, 1], distance_norm)
        grid[grid_lane_idx, cell_idx, 2] = max(grid[grid_lane_idx, cell_idx, 2], speed_norm)

    # Flatten grid
    grid_flat = grid.flatten()

    # Compute ego speed deficit
    speed_deficit = (state_dict.target_speed - state_dict.ego_speed) / state_dict.target_speed
    speed_deficit = np.clip(speed_deficit, 0.0, 1.0)  # Clip to [0, 1]

    # Concatenate
    observation = np.concatenate([grid_flat, [speed_deficit]])

    return observation
