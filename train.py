"""
Training script for lane change RL agent.

Trains a Dueling Double DQN agent to make lane change decisions.

Usage:
    # Start CARLA server first
    ./CarlaUE4.sh

    # Start training server (ticks world, no coordination)
    python training_server.py --map Town12

    # Start training
    python train.py --episodes 100 --device cuda
"""

import logging
import argparse
import yaml
import pygame
import numpy as np
import time
import threading
from pathlib import Path

from lane_change.plant.carla_vehicle_manager import CarlaVehicleManager
from lane_change.decision.rl_trainer import LaneChangeTrainer


def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Suppress verbose libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def visualize_training(surface, camera_image, tracked_vehicles, perception_core, ego_vehicle,
                       episode, total_episodes, episode_reward, epsilon, closest_traffic_light=None):
    """
    Visualize training progress with perception data.

    Args:
        surface: Pygame surface to draw on
        camera_image: CameraImage data from sensor manager
        tracked_vehicles: List of TrackedVehicle objects
        perception_core: Perception core for lane information
        ego_vehicle: Ego vehicle
        episode: Current episode number
        total_episodes: Total number of episodes
        episode_reward: Cumulative reward for current episode
        epsilon: Current exploration rate
        closest_traffic_light: Closest traffic light ahead
    """
    if camera_image is None:
        return

    # Draw camera image
    frame_surface = pygame.surfarray.make_surface(np.transpose(camera_image.image_data, (1, 0, 2)))
    surface.blit(frame_surface, (0, 0))

    # Setup font
    font = pygame.font.SysFont("Arial", 18)

    # Draw detected lane boundaries
    if perception_core:
        lanes = perception_core.get_detected_lanes()
        for i, lane in enumerate(lanes):
            if len(lane) > 1:
                points = [(int(x), int(y)) for x, y in lane]
                if len(points) > 1:
                    pygame.draw.lines(surface, (255, 255, 255), False, points, 2)
                if points:
                    bottom_point = points[-1]
                    lane_label = str(i + 1)
                    text_surface = font.render(lane_label, True, (255, 255, 0), (0, 0, 0))
                    surface.blit(text_surface, (bottom_point[0] - 10, bottom_point[1] + 10))

    # Lane position colors
    lane_position_colors = {
        'left': (255, 100, 100),
        'ego': (100, 255, 100),
        'right': (100, 100, 255)
    }

    # Get ego lane info
    ego_lane_idx = perception_core.get_ego_lane()
    num_lanes = len(perception_core.get_detected_lanes())

    # Draw tracked vehicles
    for tracked in tracked_vehicles:
        bbox = tracked.detection.bounding_box
        lane_position = tracked.get_lane_position(ego_lane_idx)
        lane_color = lane_position_colors.get(lane_position, (255, 255, 255))

        pygame.draw.rect(surface, lane_color,
                        pygame.Rect(bbox.x, bbox.y, bbox.width, bbox.height), 3)

        # Lane label
        if tracked.absolute_lane_idx == 0:
            lane_abs_display = "L-"
        elif tracked.absolute_lane_idx == num_lanes:
            lane_abs_display = "R+"
        else:
            lane_abs_display = str(tracked.absolute_lane_idx)

        if tracked.track_id >= 0:
            label = f"ID:{tracked.track_id} | Lane:{lane_abs_display} | {lane_position.upper()}"
        else:
            label = f"NEW | Lane:{lane_abs_display} | {lane_position.upper()}"

        velocity_label = f"{tracked.velocity_estimate.speed_kmh:.1f} km/h" if tracked.velocity_estimate else ''
        distance_label = f"Dist: {tracked.distance:.1f}m" if tracked.distance is not None else ''
        ttc_label = f"TTC: {tracked.ttc:.1f}s" if tracked.ttc is not None else "TTC: ---" if tracked.distance is not None else ''

        text_surface1 = font.render(label, True, (255, 255, 255), lane_color)
        text_surface2 = font.render(velocity_label, True, (255, 255, 255), (0, 0, 0))
        text_surface3 = font.render(distance_label, True, (255, 255, 255), (0, 0, 0))
        text_surface4 = font.render(ttc_label, True, (255, 255, 255), (0, 0, 0))

        surface.blit(text_surface1, (bbox.x, max(0, bbox.y - 80)))
        surface.blit(text_surface2, (bbox.x, max(0, bbox.y - 60)))
        surface.blit(text_surface3, (bbox.x, max(0, bbox.y - 40)))
        surface.blit(text_surface4, (bbox.x, max(0, bbox.y - 20)))

    # Draw traffic light if detected
    if closest_traffic_light:
        bbox = closest_traffic_light.detection.bounding_box
        color_map = {
            "red": (255, 0, 0),
            "yellow": (255, 165, 0),
            "green": (0, 255, 0),
            "irrelevant": (128, 128, 128)
        }
        tl_color = color_map.get(closest_traffic_light.color, (255, 255, 255))
        pygame.draw.rect(surface, tl_color,
                        pygame.Rect(bbox.x, bbox.y, bbox.width, bbox.height), 4)
        label = f"TRAFFIC LIGHT | {closest_traffic_light.color.upper()}"
        distance_label = f"Dist: {closest_traffic_light.distance:.1f}m" if closest_traffic_light.distance else "Dist: ---"
        text_surface1 = font.render(label, True, (255, 255, 255), tl_color)
        text_surface2 = font.render(distance_label, True, (255, 255, 255), (0, 0, 0))
        surface.blit(text_surface1, (bbox.x, max(0, bbox.y - 40)))
        surface.blit(text_surface2, (bbox.x, max(0, bbox.y - 20)))

    # Draw training stats overlay
    stats = [
        f"TRAINING MODE",
        f"Episode: {episode}/{total_episodes}",
        f"Reward: {episode_reward:.2f}",
        f"Epsilon: {epsilon:.3f}",
        f"Ego Speed: {ego_vehicle.controller.get_speed():.1f} km/h",
    ]

    y_offset = 10
    for stat in stats:
        text_surface = font.render(stat, True, (255, 255, 255), (0, 0, 0))
        surface.blit(text_surface, (10, y_offset))
        y_offset += 30


def main():
    parser = argparse.ArgumentParser(description='Train lane change RL agent')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--target-speed', type=float, default=39.15,
                       help='Target speed in km/h (default: 39.15)')
    parser.add_argument('--output-dir', type=str, default='./runs',
                       help='Output directory for checkpoints and logs (default: ./runs)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port (default: 2000)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on (default: cuda)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable pygame visualization (faster training)')

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Lane Change RL Training")
    logger.info("=" * 80)
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Target speed: {args.target_speed} km/h")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"CARLA server: {args.host}:{args.port}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Visualization: {'Disabled' if args.no_display else 'Enabled'}")
    logger.info("=" * 80)

    # Initialize pygame display if enabled
    display = None
    if not args.no_display:
        pygame.init()
        project_root = Path(__file__).parent
        config_path = project_root / "config" / "carla_config.yaml"
        with open(config_path, 'r') as f:
            carla_config = yaml.safe_load(f)
        width = carla_config['sensors']['camera']['image_size_x']
        height = carla_config['sensors']['camera']['image_size_y']
        display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Lane Change RL Training")
        logger.info(f"Pygame display initialized: {width}x{height}")

    # Initialize CARLA
    logger.info("Connecting to CARLA...")
    with CarlaVehicleManager(host=args.host, port=args.port) as vehicle_manager:
        vehicle_manager.connect()
        logger.info("Connected to CARLA")

        # Spawn ego vehicle
        logger.info("Spawning ego vehicle...")
        ego_vehicle = vehicle_manager.spawn_ego_vehicle()
        logger.info(f"Ego vehicle spawned: {ego_vehicle.id}")

        # Initialize ego vehicle speed (traffic spawned/managed by trainer)
        ego_vehicle.controller.set_speed(args.target_speed)
        logger.info(f"Ego vehicle speed set to {args.target_speed} km/h")

        # Start control thread (needed to actually drive the vehicle)
        control_thread_running = True

        def control_loop():
            """Background control thread that applies controls at 50 Hz."""
            nonlocal control_thread_running
            tick_count = 0
            target_period = 1.0 / 50.0  # 50 Hz
            next_tick_time = time.perf_counter()

            while control_thread_running:
                try:
                    # Generate and apply control
                    control = ego_vehicle.controller.run_step()
                    ego_vehicle.carla_actor.apply_control(control)

                    tick_count += 1

                    # Maintain constant 50 Hz timing
                    next_tick_time += target_period
                    sleep_time = next_tick_time - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_tick_time = time.perf_counter()

                except Exception as e:
                    logger.error(f"[Control Thread] Error at tick {tick_count}: {e}", exc_info=True)
                    tick_count += 1
                    next_tick_time = time.perf_counter()

        control_thread = threading.Thread(target=control_loop, daemon=True)
        control_thread.start()
        logger.info("Control thread started")

        # Create trainer
        logger.info("Initializing trainer...")
        trainer = LaneChangeTrainer(
            vehicle_manager=vehicle_manager,
            output_dir=args.output_dir,
            device=args.device,
            visualize_callback=visualize_training if display else None,
            display_surface=display
        )
        logger.info("Trainer initialized")

        # Train
        logger.info("Starting training...")
        try:
            trainer.train(
                num_episodes=args.episodes,
                target_speed=args.target_speed
            )
        finally:
            control_thread_running = False
            logger.info("Stopping control thread...")
            if display:
                pygame.quit()

        logger.info("Training complete!")


if __name__ == '__main__':
    main()
