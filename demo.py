"""
Open-World Demo Client - Multi-client autonomous driving demo.

This client connects to carla_server.py and joins the open-world simulation.
Traffic is managed by the server with hybrid physics.
Client IDs are auto-assigned by the server based on connection order.

Usage:
    python demo.py
    python demo.py --server-host 192.168.1.100
"""

import pygame
import numpy as np
import logging
import time
import argparse
import os
import threading
import yaml
import carla
import ctypes
from pathlib import Path
from lane_change.plant.carla_vehicle_manager import CarlaVehicleManager
from lane_change.decision import DecisionLayer, InferenceAgent
from lane_change.gateway.server_client import ServerClient
from lane_change.gateway.collision_monitor import CollisionMonitor
from typing import Optional

import psutil
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)

logger = logging.getLogger(__name__)


class KPITracker:
    """Tracks and reports KPIs for autonomous lane change performance."""

    def __init__(self, target_speed: float):
        """
        Initialize KPI tracker.

        Args:
            target_speed: Target cruising speed in km/h
        """
        self.target_speed = target_speed

        # KPI 1: Collisions
        self.collision_monitor = CollisionMonitor(
            distance_threshold=0.3,
            duration_threshold=3.0
        )
        self.collisions = 0

        # KPI 2: Target speed achievement
        self.total_frames = 0
        self.frames_at_target_speed = 0  # Within ±5 km/h

        # KPI 3: TTC violations during lane changes
        self.lane_changes_attempted = 0
        self.lane_changes_with_ttc_violation = 0  # TTC < 1.0 at decision

        # KPI 4: Average gap
        self.gap_measurements = []

        # KPI 5: Lane change success rate
        self.lane_changes_completed = 0

    def update_frame(
        self,
        tracked_vehicles: list,
        ego_lane_idx: int,
        ego_speed: float,
        next_vehicle_gap: Optional[float],
        current_time: float
    ):
        """
        Update KPIs for current frame.

        Args:
            tracked_vehicles: List of tracked vehicles from perception
            ego_lane_idx: Current ego lane index
            ego_speed: Current ego speed in km/h
            next_vehicle_gap: Gap to front vehicle in meters (None if no vehicle ahead)
            current_time: Current timestamp
        """
        self.total_frames += 1

        # KPI 1: Check for collision
        if self.collision_monitor.check(tracked_vehicles, ego_lane_idx, current_time):
            self.collisions += 1
            self.collision_monitor.reset()  # Reset after counting

        # KPI 2: Speed metric
        if abs(ego_speed - self.target_speed) <= 5.0:
            self.frames_at_target_speed += 1

        # KPI 4: Gap metric
        if next_vehicle_gap is not None:
            self.gap_measurements.append(next_vehicle_gap)

    def record_lane_change_attempt(
        self,
        accepted: bool,
        next_vehicle_ttc: Optional[float]
    ):
        """
        Record lane change attempt.

        Args:
            accepted: Whether the lane change was accepted
            next_vehicle_ttc: TTC to vehicle ahead at decision time (None if no vehicle)
        """
        self.lane_changes_attempted += 1

        # KPI 5: Success rate
        if accepted:
            self.lane_changes_completed += 1

        # KPI 3: TTC violation
        if next_vehicle_ttc is not None and next_vehicle_ttc < 1.0:
            self.lane_changes_with_ttc_violation += 1

    def print_summary(self):
        """Print KPI summary at end of run."""
        print("\n" + "=" * 60)
        print("AUTONOMOUS LANE CHANGE KPIs")
        print("=" * 60)

        # KPI 1
        print(f"Collisions: {self.collisions}")

        # KPI 2
        if self.total_frames > 0:
            speed_achievement = (self.frames_at_target_speed / self.total_frames) * 100
            print(f"Target Speed Achievement: {speed_achievement:.1f}% (within ±5 km/h of {self.target_speed} km/h)")
        else:
            print(f"Target Speed Achievement: N/A (no frames)")

        # KPI 3 & 5
        if self.lane_changes_attempted > 0:
            ttc_violation_pct = (self.lane_changes_with_ttc_violation / self.lane_changes_attempted) * 100
            success_rate = (self.lane_changes_completed / self.lane_changes_attempted) * 100
            print(f"Lane Changes: {self.lane_changes_completed}/{self.lane_changes_attempted} (success rate: {success_rate:.1f}%)")
            print(f"TTC Violations: {self.lane_changes_with_ttc_violation}/{self.lane_changes_attempted} ({ttc_violation_pct:.1f}% of lane changes)")
        else:
            print("Lane Changes: 0 (no attempts)")

        # KPI 4
        if self.gap_measurements:
            avg_gap = sum(self.gap_measurements) / len(self.gap_measurements)
            print(f"Average Gap with Front Vehicle: {avg_gap:.2f}m")
        else:
            print("Average Gap: N/A (no front vehicles)")

        print("=" * 60 + "\n")

# Windows-specific: Set high-resolution timer (system-wide)
# This prevents Windows from throttling background pygame windows
if hasattr(ctypes, 'windll'):
    ctypes.windll.winmm.timeBeginPeriod(1)

# Load CARLA configuration
project_root = Path(__file__).parent
config_path = project_root / "config" / "carla_config.yaml"
with open(config_path, 'r') as f:
    CARLA_CONFIG = yaml.safe_load(f)

# Extract display resolution from camera sensor config
DISPLAY_WIDTH = CARLA_CONFIG['sensors']['camera']['image_size_x']
DISPLAY_HEIGHT = CARLA_CONFIG['sensors']['camera']['image_size_y']


def setup_logging(verbose=False):
    """
    Configure logging for the entire application.

    Args:
        verbose: If True, set DEBUG level. Otherwise INFO level.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Suppress verbose third-party libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def visualize_yolo_tracking(surface, camera_image, tracked_vehicles, perception_core, ego_vehicle, closest_traffic_light=None):
    """
    Visualize YOLO detection results with tracking IDs, lane detection, and lane assignments.

    Args:
        surface: Pygame surface to draw on
        camera_image: CameraImage data from sensor manager
        tracked_vehicles: List of TrackedVehicle objects from perception
        perception_core: Perception core for lane information
        ego_vehicle: Ego vehicle for accessing sensor data
        closest_traffic_light: Closest traffic light ahead (if detected)
    """
    if camera_image is None:
        return

    # Draw camera image (already RGB)
    frame_surface = pygame.surfarray.make_surface(np.transpose(camera_image.image_data, (1, 0, 2)))
    surface.blit(frame_surface, (0, 0))

    # Setup font
    font = pygame.font.SysFont("Arial", 18)

    # Get velocity estimator for projection
    velocity_estimator = perception_core.velocity_estimator if hasattr(perception_core, 'velocity_estimator') else None

    # Draw detected lane boundaries
    if perception_core:
        lanes = perception_core.get_detected_lanes()

        # Draw as connected lines with labels
        for i, lane in enumerate(lanes):
            if len(lane) > 1:
                # Draw lane as connected lines
                points = [(int(x), int(y)) for x, y in lane]
                if len(points) > 1:
                    pygame.draw.lines(surface, (255, 255, 255), False, points, 2)

                # Label lanes at the bottom
                if points:
                    bottom_point = points[-1]
                    lane_label = str(i + 1)
                    text_surface = font.render(lane_label, True, (255, 255, 0), (0, 0, 0))
                    surface.blit(text_surface, (bottom_point[0] - 10, bottom_point[1] + 10))

    # Define colors for lane positions
    lane_position_colors = {
        'left': (255, 100, 100),
        'ego': (100, 255, 100),
        'right': (100, 100, 255)
    }

    # Get ego lane index for computing relative positions
    ego_lane_idx = perception_core.get_ego_lane()
    num_lanes = len(perception_core.get_detected_lanes())

    for tracked in tracked_vehicles:
        bbox = tracked.detection.bounding_box

        # Compute relative lane position from absolute lane index
        lane_position = tracked.get_lane_position(ego_lane_idx)

        # Select color based on lane position
        lane_color = lane_position_colors.get(lane_position, (255, 255, 255))

        # Draw 2D bounding box with lane position color
        pygame.draw.rect(surface, lane_color,
                        pygame.Rect(bbox.x, bbox.y, bbox.width, bbox.height), 3)

        # Prepare label with tracking, lane, and velocity info
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

        if tracked.velocity_estimate is not None:
            vel_est = tracked.velocity_estimate
            velocity_label = f"{vel_est.speed_kmh:.1f} km/h"
        else:
            velocity_label = ''

        # Prepare distance and TTC labels
        distance_label = ''
        ttc_label = ''

        if tracked.distance is not None:
            distance_label = f"Dist: {tracked.distance:.1f}m"

        if tracked.ttc is not None:
            ttc_label = f"TTC: {tracked.ttc:.1f}s"
        elif tracked.distance is not None:
            ttc_label = "TTC: ---"

        # Draw labels above box
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

        # Color based on traffic light color
        color_map = {
            "red": (255, 0, 0),
            "yellow": (255, 165, 0),
            "green": (0, 255, 0),
            "irrelevant": (128, 128, 128)
        }
        tl_color = color_map.get(closest_traffic_light.color, (255, 255, 255))

        # Draw bounding box with thick border
        pygame.draw.rect(surface, tl_color,
                        pygame.Rect(bbox.x, bbox.y, bbox.width, bbox.height), 4)

        # Prepare labels
        label = f"TRAFFIC LIGHT | {closest_traffic_light.color.upper()}"

        # Draw labels above box
        text_surface1 = font.render(label, True, (255, 255, 255), tl_color)

        surface.blit(text_surface1, (bbox.x, max(0, bbox.y - 40)))


def main():
    """Open-world demo client."""
    pygame.init()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Open-World Lane Change Demo')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging (DEBUG level)')
    parser.add_argument('--server-host', type=str, default='localhost',
                       help='Server host address (default: localhost)')
    parser.add_argument('--target-speed', type=float, default=39.15,
                       help='Target speed in km/h (default: 39.15)')
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    # Log high-resolution timer status
    if hasattr(ctypes, 'windll'):
        logger.info("High-resolution timer enabled (1ms) - all clients get equal performance")

    # Setup display
    width, height = DISPLAY_WIDTH, DISPLAY_HEIGHT
    display = pygame.display.set_mode(
        (width, height),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("Lane Change - Open World Demo")
    font = pygame.font.SysFont("Arial", 24)

    logger.info("=" * 80)
    logger.info("OPEN-WORLD DEMO CLIENT")
    logger.info(f"Server Host: {args.server_host}")
    logger.info("=" * 80)

    # Connect to server via ZeroMQ
    logger.info("Connecting to server...")
    with ServerClient(host=args.server_host) as server_client:
        # Register with server and get assigned client_id
        registration = server_client.register()
        assigned_client_id = registration.get('assigned_client_id', 0)
        phase = registration.get('phase', 'waiting')

        # Determine role name based on assigned client_id
        if assigned_client_id == 0:
            role_name = 'hero'
        else:
            role_name = f'hero_{assigned_client_id}'

        logger.info(f"Server assigned client_id: {assigned_client_id} (role: {role_name})")
        logger.info(f"Server phase: {phase}")

        # Connect to CARLA and spawn ego (pass server_client for coordinated cleanup)
        with CarlaVehicleManager(host='localhost', port=2000, server_client=server_client) as vehicle_manager:
            vehicle_manager.connect()

            # Spawn ego vehicle with correct role_name
            ego_vehicle = vehicle_manager.spawn_ego_vehicle(role_name=role_name)

            # If waiting phase, set speed to 0
            if phase == 'waiting':
                logger.info("Waiting phase - setting speed to 0")
                ego_vehicle.carla_actor.set_target_velocity(carla.Vector3D(0, 0, 0))

            # Use the vehicle's perception core
            perception = ego_vehicle.controller.perception_core

            # Load trained RL agent
            model_path = Path(__file__).parent / 'runs' / 'final_model.pth'
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Trained model not found at {model_path}. "
                    f"Please train the model first using: python train.py --episodes 100"
                )

            rl_agent = InferenceAgent(
                model_path=str(model_path),
                device='cuda' if __import__('torch').cuda.is_available() else 'cpu'
            )
            logger.info(f"Loaded trained model from {model_path}")

            # Initialize KPI tracker
            kpi_tracker = KPITracker(target_speed=args.target_speed)

            # Initialize decision layer
            decision_layer = DecisionLayer(
                rl_agent=rl_agent,
                ego_vehicle=ego_vehicle,
                perception_core=perception,
                target_speed=args.target_speed,
                vehicle_ttc_threshold=2.0,
                pedestrian_ttc_threshold=5.0,
                cooldown_period=3.0,
                kpi_tracker=kpi_tracker
            )

            logger.info("Starting open-world demo...")
            logger.info("Controls:")
            logger.info("  ESC: Quit | SPACE: Pause/Resume | R: Reset tracker")
            logger.info("  LEFT/RIGHT: Manual lane change override")
            logger.info("  UP/DOWN: Increase/Decrease target speed (+/- 10 km/h)")
            logger.info("  T: Toggle decision layer ON/OFF")
            logger.info(f"Decision layer: target_speed={decision_layer.target_speed} km/h, "
                       f"vehicle_TTC={decision_layer.vehicle_ttc_threshold}s")

            # State variables (defined early for shutdown callback)
            running = True

            # Register shutdown callback to gracefully exit
            def on_server_shutdown():
                nonlocal running
                logger.warning("Server shutting down - exiting demo gracefully...")
                running = False

            server_client.on_shutdown_callback = on_server_shutdown

            # Signal ready to server
            server_client.signal_ready(client_id=assigned_client_id)

            # Wait for GO if in waiting phase
            if phase == 'waiting':
                logger.info("Waiting for GO signal from server...")
                server_client.wait_for_go(timeout=120)  # 2 minutes timeout (seconds, not ms)
                logger.info("GO received - starting!")

                # Set initial speed to target speed
                ego_vehicle.controller.set_speed(args.target_speed)
                logger.info(f"Set initial speed to target: {args.target_speed} km/h")

            paused = False
            frame_count = 0
            fps_counter = 0
            fps_timer = time.time()
            decision_layer_enabled = True
            control_thread_running = True
            last_frame_time = time.time()
            STALE_FRAME_TIMEOUT = 5.0

            # Thread synchronization lock for controller access
            controller_lock = threading.Lock()
            server_alive = threading.Event()
            server_alive.set()

            def control_loop():
                """Background control thread that applies controls at 50 Hz."""
                nonlocal control_thread_running
                tick_count = 0
                target_period = 1.0 / 50.0
                next_tick_time = time.perf_counter()
                consecutive_control_failures = 0

                while control_thread_running and server_alive.is_set():
                    try:
                        loop_start = time.perf_counter()

                        # Log waypoint queue status every 50 ticks (1 second)
                        if tick_count % 50 == 0:
                            queue = ego_vehicle.controller._local_planner._waypoints_queue
                            queue_len = len(queue)
                            if queue_len > 0:
                                first_wp = queue[0][0]
                                vehicle_loc = ego_vehicle.carla_actor.get_location()
                                logger.debug(f"[Control Thread] Tick {tick_count}: Queue={queue_len}, "
                                           f"Vehicle=({vehicle_loc.x:.2f},{vehicle_loc.y:.2f}), "
                                           f"FirstWP=({first_wp.transform.location.x:.2f},{first_wp.transform.location.y:.2f})")
                            else:
                                logger.debug(f"[Control Thread] Tick {tick_count}: Queue EMPTY")

                        # Generate and apply control (thread-safe with timeout protection)
                        with controller_lock:
                            control = ego_vehicle.controller.run_step()
                            ego_vehicle.carla_actor.apply_control(control)

                        tick_count += 1
                        consecutive_control_failures = 0

                        # Maintain constant 50 Hz timing
                        next_tick_time += target_period
                        sleep_time = next_tick_time - time.perf_counter()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        else:
                            next_tick_time = time.perf_counter()

                    except Exception as e:
                        consecutive_control_failures += 1
                        logger.error(f"[Control Thread] Error at tick {tick_count} (failure {consecutive_control_failures}/5): {e}", exc_info=True)

                        if consecutive_control_failures >= 5:
                            logger.error("Too many consecutive control failures - signaling shutdown")
                            server_alive.clear()
                            break

                        tick_count += 1
                        next_tick_time = time.perf_counter()

                if not server_alive.is_set():
                    logger.error("Control thread exiting due to server failure")

            # Start control thread
            control_thread = threading.Thread(target=control_loop, daemon=True)
            control_thread.start()
            logger.info("Control thread started")

            try:
                while running and server_alive.is_set():
                    # Handle events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            logger.info("Window closed - ending demo")
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                logger.info("ESC pressed - ending demo")
                                running = False
                            elif event.key == pygame.K_LEFT:
                                with controller_lock:
                                    vehicle_manager.ego_vehicle.controller.set_lane('left')
                                logger.info("Lane change LEFT commanded")
                            elif event.key == pygame.K_RIGHT:
                                with controller_lock:
                                    vehicle_manager.ego_vehicle.controller.set_lane('right')
                                logger.info("Lane change RIGHT commanded")
                            elif event.key == pygame.K_r:
                                perception.reset()
                                logger.info("Tracker reset")
                            elif event.key == pygame.K_SPACE:
                                paused = not paused
                                logger.info(f"{'Paused' if paused else 'Resumed'}")
                            elif event.key == pygame.K_UP:
                                new_speed = decision_layer.target_speed + 10
                                decision_layer.set_target_speed(new_speed)
                                logger.info(f"Target speed increased to {new_speed} km/h")
                                if not decision_layer_enabled:
                                    with controller_lock:
                                        ego_vehicle.controller.set_speed(new_speed)
                            elif event.key == pygame.K_DOWN:
                                new_speed = max(0, decision_layer.target_speed - 10)
                                decision_layer.set_target_speed(new_speed)
                                logger.info(f"Target speed decreased to {new_speed} km/h")
                                if not decision_layer_enabled:
                                    with controller_lock:
                                        ego_vehicle.controller.set_speed(new_speed)
                            elif event.key == pygame.K_t:
                                decision_layer_enabled = not decision_layer_enabled
                                if not decision_layer_enabled:
                                    with controller_lock:
                                        current_speed = ego_vehicle.controller.get_speed()
                                        ego_vehicle.controller.set_speed(current_speed)
                                    logger.info(f"Decision layer DISABLED - Manual control mode")
                                else:
                                    logger.info("Decision layer ENABLED - Autonomous mode")

                    # Get latest sensor data
                    camera_image = ego_vehicle.sensor_manager.get_camera_image('front')

                    if camera_image:
                        last_frame_time = time.time()
                    elif not paused and (time.time() - last_frame_time) > STALE_FRAME_TIMEOUT:
                        logger.error(f"No camera frames for {STALE_FRAME_TIMEOUT}s - CARLA may be frozen")
                        server_alive.clear()
                        running = False

                    if not paused and camera_image:
                        # Process perception frame
                        tracked_vehicles, closest_traffic_light = perception.process_frame()

                        # Get ego info for decision and KPI tracking
                        ego_lane_idx = perception.get_ego_lane()
                        ego_speed = ego_vehicle.controller.get_speed()

                        # Run decision layer if enabled (thread-safe)
                        if decision_layer_enabled:
                            with controller_lock:
                                ego_info = {
                                    'lane_idx': ego_lane_idx,
                                    'speed_kmh': ego_speed
                                }
                                decision_layer.process(
                                    tracked_vehicles=tracked_vehicles,
                                    vehicle_controller=ego_vehicle.controller,
                                    ego_info=ego_info,
                                    current_time=time.time(),
                                    closest_traffic_light=closest_traffic_light
                                )

                        # Update KPIs
                        from lane_change.perception.tracked_objects import TrackedVehicle
                        next_vehicle_gap = None
                        for obj in tracked_vehicles:
                            if (isinstance(obj, TrackedVehicle) and
                                obj.absolute_lane_idx == ego_lane_idx and
                                obj.distance and obj.distance > 0):
                                if next_vehicle_gap is None or obj.gap < next_vehicle_gap:
                                    next_vehicle_gap = obj.gap

                        kpi_tracker.update_frame(
                            tracked_vehicles=tracked_vehicles,
                            ego_lane_idx=ego_lane_idx,
                            ego_speed=ego_speed,
                            next_vehicle_gap=next_vehicle_gap,
                            current_time=time.time()
                        )

                        # Clear display
                        display.fill((0, 0, 0))

                        # Visualize detections
                        visualize_yolo_tracking(display, camera_image, tracked_vehicles, perception,
                                              ego_vehicle, closest_traffic_light)

                        # Calculate FPS
                        fps_counter += 1
                        if time.time() - fps_timer > 1.0:
                            current_fps = fps_counter
                            logger.debug(f"FPS: {current_fps}")
                            fps_counter = 0
                            fps_timer = time.time()

                        # Draw stats overlay
                        stats = [
                            f"Ego Speed: {ego_vehicle.controller.get_speed():.1f} km/h",
                            f"Target Speed: {decision_layer.target_speed:.1f} km/h",
                        ]

                        y_offset = 10
                        for stat in stats:
                            text_surface = font.render(stat, True, (255, 255, 255), (0, 0, 0))
                            display.blit(text_surface, (10, y_offset))
                            y_offset += 30

                        frame_count += 1

                    else:
                        # Show paused message
                        pause_text = font.render("PAUSED - Press SPACE to resume", True,
                                                (255, 255, 0), (0, 0, 0))
                        text_rect = pause_text.get_rect(center=(width // 2, height // 2))
                        display.blit(pause_text, text_rect)

                    pygame.display.flip()

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                running = False

            finally:
                # Print KPI summary
                kpi_tracker.print_summary()

                # Check if we exited due to server failure
                if not server_alive.is_set():
                    logger.error("Exiting due to server failure")

                # Stop control thread
                control_thread_running = False
                logger.info("Stopping control thread...")

                # Wait briefly for control thread to finish
                if control_thread.is_alive():
                    control_thread.join(timeout=0.5)

    logger.info("Demo completed")
    pygame.quit()

    # Clean up high-resolution timer
    if hasattr(ctypes, 'windll'):
        ctypes.windll.winmm.timeEndPeriod(1)


if __name__ == '__main__':
    main()
