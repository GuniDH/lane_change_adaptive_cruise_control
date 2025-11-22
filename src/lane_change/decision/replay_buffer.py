"""
Replay Buffer - Experience replay for DQN training.

Stores transitions and samples random batches for training.
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        """
        Add transition to buffer.

        Args:
            state: Current state observation
            action: Action taken (0 or 1)
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray]:
        """
        Sample random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each as numpy array with shape (batch_size, ...)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has {len(self.buffer)} transitions, "
                           f"cannot sample batch of {batch_size}")

        # Sample random batch
        batch = random.sample(self.buffer, batch_size)

        # Unpack into separate arrays
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()
