"""
Lane Change Trainer - Custom training loop for lane change RL agent.

Implements Dueling Double DQN training with experience replay.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import logging
import random
import carla
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from torch.utils.tensorboard import SummaryWriter

from lane_change.decision.rl_agent import TrainingAgent, build_observation, StateDict
from lane_change.decision.dqn_network import DuelingDQN
from lane_change.decision.replay_buffer import ReplayBuffer
from lane_change.decision.decision_core import DecisionLayer
from lane_change.plant.carla_traffic_manager import CarlaTrafficManager
from lane_change.gateway.collision_monitor import CollisionMonitor

# Import pygame at module level for visualization (not inside loops)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

logger = logging.getLogger(__name__)


class LaneChangeTrainer:
    """
    Custom training loop for lane change RL agent.

    Integrates with existing architecture (async ticker + DecisionLayer).
    """

    def __init__(
        self,
        vehicle_manager,
        output_dir: str = './runs',
        gamma: float = 0.99,
        learning_rate: float = 0.0001,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_episodes: int = 50,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        visualize_callback = None,
        display_surface = None
    ):
        """
        Initialize trainer.

        Args:
            vehicle_manager: CARLA vehicle manager instance
            output_dir: Directory for checkpoints and tensorboard logs
            gamma: Discount factor
            learning_rate: Learning rate for optimizer
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            epsilon_start: Starting epsilon for exploration
            epsilon_end: Final epsilon
            epsilon_decay_episodes: Episodes to decay epsilon over
            device: Device to train on ('cpu' or 'cuda')
        """
        self.vehicle_manager = vehicle_manager
        self.ego_vehicle = vehicle_manager.ego_vehicle
        self.controller = self.ego_vehicle.controller
        self.perception = self.controller.perception_core
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.device = torch.device(device)

        # Networks
        state_size = 151  # 5*10*3 + 1
        action_size = 2  # left, right

        self.q_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training agent with callback to this trainer
        self.training_agent = TrainingAgent(trainer=self)

        # Decision layer with training agent
        self.decision_layer = DecisionLayer(
            rl_agent=self.training_agent,
            ego_vehicle=self.ego_vehicle,
            perception_core=self.perception,
            target_speed=39.15,  # Will be overridden per episode
            vehicle_ttc_threshold=2.0,
            pedestrian_ttc_threshold=5.0,
            cooldown_period=3.0
        )

        # Traffic manager for spawning and managing traffic
        self.traffic_manager = CarlaTrafficManager(
            world=vehicle_manager.world,
            client=vehicle_manager.client,
            traffic_manager=vehicle_manager.carla_tm
        )

        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.epsilon = epsilon_start

        # Pending transition tracking (for continuous mode)
        self.pending_transition = None
        self.was_lane_changing = False

        # Collision monitoring
        self.collision_monitor = CollisionMonitor(
            distance_threshold=0.3,  # meters
            duration_threshold=3.0   # seconds
        )

        # Visualization
        self.visualize_callback = visualize_callback
        self.display_surface = display_surface

        # Tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Q-network: {sum(p.numel() for p in self.q_network.parameters())} parameters")

    def train(self, num_episodes: int = 100, target_speed: float = 39.15):
        """
        Main training loop.

        Args:
            num_episodes: Number of episodes to train
            target_speed: Target speed for ego vehicle (km/h)
        """
        self.num_episodes = num_episodes  # Store for visualization

        logger.info("="*80)
        logger.info(f"Starting training: {num_episodes} episodes")
        logger.info(f"Target speed: {target_speed} km/h")
        logger.info("="*80)

        for episode in range(num_episodes):
            episode_start = time.time()

            # Soft reset
            self._reset_episode()

            # Update target speed
            self.decision_layer.target_speed = target_speed

            # Run episode
            episode_reward, episode_steps, episode_result = self._run_episode()

            # Update epsilon
            self._update_epsilon(episode)

            # Log episode
            episode_duration = time.time() - episode_start
            self._log_episode(episode, episode_reward, episode_steps, episode_result, episode_duration)

            self.episode_count += 1

            # Save checkpoint periodically
            if (episode + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_ep{episode+1}.pth')

        logger.info("="*80)
        logger.info("Training complete!")
        logger.info("="*80)

        # Save final model
        self._save_checkpoint('final_model.pth')
        self.writer.close()

    def _reset_episode(self):
        """Soft reset for new episode - faster than full reset."""
        # Destroy traffic
        self.traffic_manager.destroy_all()

        # Reset perception state (clear tracked vehicles and reset tracker)
        self.perception.reset()

        # Reset controller state
        self.controller.lane_changing = False
        self.controller.lane_change_start_time = None

        # Reset decision layer state
        self.decision_layer.last_lane_change_time = 0.0
        self.decision_layer.stopped_at_traffic_light = False

        # Teleport ego to configured spawn point (consistent training environment)
        spawn_point = self.vehicle_manager.get_configured_spawn_point()
        self.ego_vehicle.carla_actor.set_transform(spawn_point)

        # Reset velocity and controller speed
        self.ego_vehicle.carla_actor.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.controller.set_speed(self.decision_layer.target_speed)

        # Reset controller path to straight ahead (clears stale lane change paths)
        self.controller._straight_path()

        # Respawn traffic in new configuration
        num_vehicles = random.randint(4, 6)
        ego_location = self.vehicle_manager.ego_vehicle.carla_actor.get_location()

        self.traffic_manager.spawn_traffic(
            reference_location=ego_location,
            num_vehicles=num_vehicles,
            distance_range=(-20.0, 20.0),
            speed_range=(25.0, 45.0),
            min_spacing=4.0
        )

        logger.debug(f"Spawned {self.traffic_manager.get_vehicle_count()} traffic vehicles")

        # Initialize speeds (ego + traffic)
        self.controller.set_speed(self.decision_layer.target_speed)
        self.traffic_manager.apply_speeds()
        self.traffic_manager.enable_autopilot()

        # Wait for stabilization
        time.sleep(0.5)

        logger.debug("Episode reset complete")

    def on_agent_queried(self, state_dict: StateDict) -> int:
        """
        Callback when decision layer queries the agent.

        Args:
            state_dict: State information from decision layer

        Returns:
            Action index selected by epsilon-greedy policy
        """
        # Build observation from state
        state = build_observation(state_dict)

        # Select action using epsilon-greedy
        if random.random() < self.epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()

        # Store pending transition (will complete when lane change finishes)
        self.pending_transition = {
            'state': state,
            'action': action,
            'query_time': time.time()
        }

        logger.debug(f"Agent queried: action={action} ({'left' if action==0 else 'right'}), epsilon={self.epsilon:.3f}")

        return action

    def _run_episode(self) -> Tuple[float, int, str]:
        """
        Run single episode in continuous mode (like demo.py).

        Continuously calls decision layer, which internally queries agent when needed.
        Stores transitions when lane changes complete.

        Returns:
            Tuple of (total_reward, steps, result)
            result: 'collision', 'success', or 'timeout'
        """
        episode_reward = 0.0
        episode_steps = 0
        episode_start_time = time.time()
        episode_start_location = self.ego_vehicle.carla_actor.get_location()
        last_viz_time = time.time()

        done = False
        result = 'timeout'

        # Reset pending transition tracking
        self.pending_transition = None
        self.was_lane_changing = False

        # Reset collision monitoring
        self.collision_monitor.reset()

        while not done:
            # Handle pygame events to prevent freezing
            if self.visualize_callback:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        logger.info("Window closed - stopping training")
                        return episode_reward, episode_steps, 'interrupted'

            # Get current perception (continuous, like demo.py)
            tracked_vehicles, traffic_light = self.perception.process_frame()
            ego_info = {
                'lane_idx': self.perception.get_ego_lane(),
                'speed_kmh': self.controller.get_speed()
            }

            # Run decision layer continuously (like demo.py)
            # It will query agent via callback when conditions are met
            current_time = time.time()
            self.decision_layer.process(
                tracked_vehicles=tracked_vehicles,
                vehicle_controller=self.controller,
                ego_info=ego_info,
                current_time=current_time,
                closest_traffic_light=traffic_light
            )

            # Check for collision continuously (3+ seconds of violation)
            if self.collision_monitor.check(tracked_vehicles, ego_info['lane_idx'], current_time):
                # Collision detected - terminate episode
                reward = -100.0

                # Store transition if there's a pending one
                if self.pending_transition:
                    next_state = self._build_state(tracked_vehicles, ego_info)
                    self.replay_buffer.add(
                        self.pending_transition['state'],
                        self.pending_transition['action'],
                        reward,
                        next_state,
                        True
                    )
                    self._train_step()
                    self.pending_transition = None

                logger.info(f"Collision detected! Episode terminated.")
                return episode_reward + reward, episode_steps + 1, 'collision'

            # Check if lane change completed
            is_lane_changing = self.controller.lane_changing
            if self.was_lane_changing and not is_lane_changing:
                # Lane change just completed!
                if self.pending_transition:
                    # Store transition for completed lane change
                    next_state = self._build_state(tracked_vehicles, ego_info)
                    reward = self._compute_reward(tracked_vehicles, ego_info, self.pending_transition['action'])
                    self.replay_buffer.add(
                        self.pending_transition['state'],
                        self.pending_transition['action'],
                        reward,
                        next_state,
                        False
                    )
                    self._train_step()

                    episode_reward += reward
                    episode_steps += 1
                    self.total_steps += 1

                    logger.debug(f"Lane change completed: reward={reward:.2f}, total_steps={self.total_steps}")

                    # Update target network periodically
                    if self.total_steps % self.target_update_freq == 0:
                        self.target_network.load_state_dict(self.q_network.state_dict())
                        logger.debug(f"Target network updated at step {self.total_steps}")

                    self.pending_transition = None

            self.was_lane_changing = is_lane_changing

            # Visualize (always, like demo.py) - but limit to ~20 Hz to reduce overhead
            if self.visualize_callback and (time.time() - last_viz_time) > 0.05:
                camera_image = self.ego_vehicle.sensor_manager.get_camera_image('front')
                if camera_image:
                    self.display_surface.fill((0, 0, 0))
                    self.visualize_callback(
                        self.display_surface,
                        camera_image,
                        tracked_vehicles,
                        self.perception,
                        self.ego_vehicle,
                        self.episode_count + 1,
                        self.num_episodes,
                        episode_reward,
                        self.epsilon,
                        traffic_light
                    )
                    pygame.display.flip()
                    last_viz_time = time.time()

            # Check termination conditions
            elapsed = time.time() - episode_start_time

            # Success: Drove well for 15+ seconds
            if elapsed > 20.0:
                current_location = self.ego_vehicle.carla_actor.get_location()
                distance_traveled = episode_start_location.distance(current_location)
                avg_speed = (distance_traveled / elapsed) * 3.6

                if avg_speed > 0.90 * self.decision_layer.target_speed:
                    result = 'success'
                    done = True

                    # Add success bonus
                    success_reward = 50.0
                    episode_reward += success_reward

                    # Store terminal transition if there's a pending one
                    if self.pending_transition:
                        next_state = self._build_state(tracked_vehicles, ego_info)
                        self.replay_buffer.add(
                            self.pending_transition['state'],
                            self.pending_transition['action'],
                            success_reward,
                            next_state,
                            True  # Terminal
                        )
                        self._train_step()
                        self.pending_transition = None

                    logger.info(f"Episode success! Avg speed: {avg_speed:.1f} km/h, bonus: +{success_reward}")

            # Timeout: Max 60 seconds
            if elapsed > 60.0:
                result = 'timeout'
                done = True

                # Add timeout penalty
                timeout_penalty = -50.0
                episode_reward += timeout_penalty

                # Store terminal transition if there's a pending one
                if self.pending_transition:
                    next_state = self._build_state(tracked_vehicles, ego_info)
                    self.replay_buffer.add(
                        self.pending_transition['state'],
                        self.pending_transition['action'],
                        timeout_penalty,
                        next_state,
                        True  # Terminal
                    )
                    self._train_step()
                    self.pending_transition = None

                logger.info(f"Episode timeout after {elapsed:.1f}s, penalty: {timeout_penalty}")

        return episode_reward, episode_steps, result

    def _build_state(self, tracked_vehicles: List, ego_info: Dict) -> np.ndarray:
        """Build state observation."""
        num_lanes = len(self.perception.get_detected_lanes())
        state_dict = StateDict(
            tracked_vehicles=tracked_vehicles,
            ego_lane_idx=ego_info['lane_idx'],
            ego_speed=ego_info['speed_kmh'],
            target_speed=self.decision_layer.target_speed,
            num_lanes=num_lanes
        )
        return build_observation(state_dict)

    def _compute_reward(self, tracked_vehicles: List, ego_info: Dict, action: int) -> float:
        """
        Compute reward for completed action.

        Prioritizes safety first, then efficiency.
        """
        reward = 0.0

        # Speed efficiency (main reward)
        speed_ratio = ego_info['speed_kmh'] / self.decision_layer.target_speed
        reward += speed_ratio * 1.0

        # Front vehicle TTC danger penalty
        front_vehicle = self._find_front_vehicle(tracked_vehicles, ego_info['lane_idx'])
        if front_vehicle and front_vehicle.ttc is not None:
            if front_vehicle.ttc < 2.0:
                reward -= (2.0 - front_vehicle.ttc) * 3.0

        return reward

    def _find_front_vehicle(self, tracked_vehicles: List, ego_lane_idx: int):
        """Find closest vehicle ahead in ego lane."""
        from lane_change.perception.perception_core import TrackedVehicle

        vehicles_in_lane = [
            v for v in tracked_vehicles
            if isinstance(v, TrackedVehicle) and
               v.absolute_lane_idx == ego_lane_idx and
               v.distance is not None and v.distance > 0
        ]

        if vehicles_in_lane:
            return min(vehicles_in_lane, key=lambda v: v.distance)
        return None

    def _train_step(self):
        """Train Q-network on batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q(s, a)
        q_values = self.q_network(states).gather(1, actions)

        # Compute Double DQN target: r + gamma * Q_target(s', argmax_a Q(s', a))
        with torch.no_grad():
            # Use online network to select best action
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            # Use target network to evaluate action
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(q_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Log to tensorboard
        self.writer.add_scalar('train/loss', loss.item(), self.total_steps)
        self.writer.add_scalar('train/q_value_mean', q_values.mean().item(), self.total_steps)

    def _update_epsilon(self, episode: int):
        """Update epsilon for exploration."""
        if episode < self.epsilon_decay_episodes:
            self.epsilon = self.epsilon_start - \
                          (self.epsilon_start - self.epsilon_end) * (episode / self.epsilon_decay_episodes)
        else:
            self.epsilon = self.epsilon_end

    def _log_episode(self, episode: int, reward: float, steps: int, result: str, duration: float):
        """Log episode statistics."""
        logger.info(f"Episode {episode+1}: reward={reward:.2f}, steps={steps}, "
                   f"result={result}, duration={duration:.1f}s, epsilon={self.epsilon:.3f}")

        # Tensorboard
        self.writer.add_scalar('episode/reward', reward, episode)
        self.writer.add_scalar('episode/steps', steps, episode)
        self.writer.add_scalar('episode/duration', duration, episode)
        self.writer.add_scalar('episode/epsilon', self.epsilon, episode)
        self.writer.add_scalar('episode/success', 1.0 if result == 'success' else 0.0, episode)
        self.writer.add_scalar('episode/collision', 1.0 if result == 'collision' else 0.0, episode)

    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / filename
        torch.save({
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'model_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': 151,
            'action_size': 2
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
