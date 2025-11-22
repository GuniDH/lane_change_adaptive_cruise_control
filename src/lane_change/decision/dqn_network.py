"""
Dueling Double DQN Network - Neural network for Q-value approximation.

Implements Dueling architecture that separates state value and action advantage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.

    Separates Q(s,a) into:
    - V(s): State value (scalar)
    - A(s,a): Action advantage (per action)

    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize Dueling DQN network.

        Args:
            state_size: Dimension of state observation
            action_size: Number of actions (2 for left/right)
            hidden_size: Size of hidden layers
        """
        super(DuelingDQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # Shared feature layers
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            state: State observation tensor (batch_size, state_size)

        Returns:
            Q-values for each action (batch_size, action_size)
        """
        # Shared features
        features = self.feature(state)

        # Value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine using dueling architecture
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def save(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size
        }, path)

    @staticmethod
    def load(path: str, device: str = 'cpu'):
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model on

        Returns:
            Loaded DuelingDQN model
        """
        checkpoint = torch.load(path, map_location=device)
        model = DuelingDQN(
            state_size=checkpoint['state_size'],
            action_size=checkpoint['action_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model
