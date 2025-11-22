"""
CARLA Vehicle Manager - Manages vehicles for a single client.

This module handles vehicle spawning and control for a single client in both
open-world multi-client mode and training mode.
"""

import carla
import random
import logging
import yaml
from pathlib import Path
from lane_change.gateway.vehicle_manager import IVehicleManager

logger = logging.getLogger(__name__)


class CarlaVehicleManager(IVehicleManager):
    """
    Manages vehicles for a single client.

    Responsibilities:
    - Connect to CARLA (world already setup by server)
    - Spawn and manage ego vehicle
    - Spawn training scenario vehicles (for training mode)
    - Cleanup own actors on exit

    Does NOT:
    - Load maps (server handles this)
    - Configure world settings (server handles this)
    - Tick the world (server handles this)
    - Spawn global traffic (server handles this in open-world mode)
    """

    def __init__(self, host='localhost', port=2000, config=None, server_client=None):
        """
        Initialize vehicle manager.

        Args:
            host: CARLA server host address
            port: CARLA server port
            config: Configuration dictionary
            server_client: Optional ServerClient for coordinated cleanup
        """
        self.host = host
        self.port = port
        self.server_client = server_client

        # Load configuration
        if config is None:
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "carla_config.yaml"
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Loaded config from {config_path}")
            except FileNotFoundError:
                logger.warning(f"Config not found at {config_path}, using defaults")
                self.config = {}
        else:
            self.config = config

        # CARLA objects
        self.client = None
        self.world = None
        self.carla_tm = None

        # Spawned actors
        self.ego_vehicle = None
        self.my_ego_actor_id = None

    def connect(self, map_name=None):
        """
        Connect to CARLA server (world already setup by server).

        Args:
            map_name: Ignored (server loads map)
        """
        try:
            # Connect to CARLA
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(
                self.config.get('carla', {}).get('timeout', 120.0)
            )

            # Get world (already configured by server)
            self.world = self.client.get_world()
            logger.info(f"Connected to CARLA: {self.world.get_map().name}")

            # Verify synchronous mode enabled
            settings = self.world.get_settings()
            if not settings.synchronous_mode:
                raise RuntimeError(
                    "\n" + "="*60 + "\n"
                    "ERROR: No server detected!\n"
                    "\n"
                    "Synchronous mode not enabled.\n"
                    "\n"
                    "Steps:\n"
                    "1. Start CARLA server (./CarlaUE4.sh)\n"
                    "2. Start server: python carla_server.py --num-clients N  (open-world)\n"
                    "   OR: python training_server.py  (training)\n"
                    "3. Start client: python demo.py / python train.py\n"
                    "\n"
                    "The server must be running BEFORE starting clients.\n"
                    + "="*60
                )

            # Get Traffic Manager
            tm_port = self.config.get('traffic', {}).get('traffic_manager_port', 8000)
            self.carla_tm = self.client.get_trafficmanager(tm_port)

            logger.info("Client connected successfully")

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    def find_spawn_point_near(self, x, y, z=None):
        """
        Find the closest spawn point to given coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Optional Z coordinate (ignored for distance calculation)

        Returns:
            carla.Transform: Closest spawn point
        """
        all_spawn_points = self.world.get_map().get_spawn_points()

        min_distance = float('inf')
        closest_spawn = None

        for spawn in all_spawn_points:
            dx = spawn.location.x - x
            dy = spawn.location.y - y
            distance = (dx**2 + dy**2)**0.5

            if distance < min_distance:
                min_distance = distance
                closest_spawn = spawn

        logger.info(f"Found spawn point at ({closest_spawn.location.x:.2f}, {closest_spawn.location.y:.2f}, {closest_spawn.location.z:.2f}), distance: {min_distance:.2f}m from target ({x:.2f}, {y:.2f})")
        return closest_spawn

    def spawn_ego_vehicle(self, spawn_point=None, role_name=None):
        """
        Spawn ego vehicle with controller and sensors.

        Args:
            spawn_point: Optional specific spawn point
            role_name: Optional role name (e.g., 'hero', 'hero_1', 'hero_2').
                      If None, defaults to 'hero' for single-client mode.

        Returns:
            IEgoVehicle: Complete ego vehicle object
        """
        # Get vehicle blueprint
        ego_config = self.config.get('ego_vehicle', {})
        blueprint_filter = ego_config.get('blueprint', 'vehicle.tesla.model3')
        vehicle_bp = self.world.get_blueprint_library().filter(blueprint_filter)[0]

        # Set role name (default to 'hero' if not provided)
        if role_name is None:
            role_name = 'hero'
        vehicle_bp.set_attribute('role_name', role_name)
        logger.info(f"Spawning ego vehicle with role_name: {role_name}")

        # Get spawn point and spawn vehicle
        if spawn_point is None:
            spawn_location = ego_config.get('spawn_location')

            if spawn_location:
                # Multi-client support: Try configured location first, then closest alternatives
                x = spawn_location.get('x')
                y = spawn_location.get('y')

                # Get all spawn points sorted by distance from configured location
                all_spawn_points = self.world.get_map().get_spawn_points()
                target_location = carla.Location(x=x, y=y, z=0)

                # Sort by distance to configured location
                sorted_spawns = sorted(
                    all_spawn_points,
                    key=lambda sp: sp.location.distance(target_location)
                )

                # Try spawn points in order (closest first)
                carla_actor = None
                for i, sp in enumerate(sorted_spawns):
                    carla_actor = self.world.try_spawn_actor(vehicle_bp, sp)
                    if carla_actor:
                        spawn_point = sp
                        distance = sp.location.distance(target_location)
                        if i == 0:
                            logger.info(f"Spawned at configured location ({x:.1f}, {y:.1f})")
                        else:
                            logger.info(f"Configured location occupied, spawned at nearest available point ({distance:.1f}m away)")
                        break
            else:
                # No configured location - use random spawn points
                all_spawn_points = self.world.get_map().get_spawn_points()
                random.shuffle(all_spawn_points)

                carla_actor = None
                for sp in all_spawn_points:
                    carla_actor = self.world.try_spawn_actor(vehicle_bp, sp)
                    if carla_actor:
                        spawn_point = sp
                        logger.info(f"Spawned at random spawn point")
                        break
        else:
            # Use provided spawn point
            carla_actor = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Check if spawn succeeded
        if carla_actor is None:
            raise RuntimeError(
                "Failed to spawn ego vehicle at any spawn point. "
                "This usually means CARLA has orphaned actors from a previous crashed run. "
                "Try restarting the CARLA server, or run the cleanup again."
            )

        logger.info(f"Spawned ego vehicle: {carla_actor.type_id} at {spawn_point.location}")

        # CRITICAL: Wait for at least one world tick to ensure vehicle position is updated in CARLA
        # Without this, carla_actor.get_location() returns uninitialized position (near origin)
        # when LocalPlanner._init_controller() captures the first waypoint
        logger.info("Waiting for world tick to initialize vehicle position...")
        self.world.wait_for_tick(2.0)
        actual_location = carla_actor.get_location()
        logger.info(f"Vehicle position after tick: ({actual_location.x:.2f}, {actual_location.y:.2f}, {actual_location.z:.2f})")

        # Create sensor manager
        from lane_change.plant.carla_sensor_manager import CarlaSensorManager
        sensor_manager = CarlaSensorManager(carla_actor, self.world)
        sensor_manager.spawn_sensors()

        # Create velocity estimator with actual sensor parameters
        from lane_change.perception.carla_velocity_estimator import CarlaVelocityEstimator
        velocity_estimator = CarlaVelocityEstimator(
            camera_fov=sensor_manager.camera_config['fov'],
            camera_location=sensor_manager.camera_config['location'],
            camera_pitch=sensor_manager.camera_config['pitch'],
            lidar_location=sensor_manager.lidar_config['location'],
            radar_location=sensor_manager.radar_config['location']
        )

        # Create detector instances
        from lane_change.perception import YoloDetector, UFLDLaneDetector, TrafficLightYoloClassifier
        from pathlib import Path
        import numpy as np

        # Ego vehicle mask in normalized coordinates (for YOLO detector)
        ego_vehicle_mask_normalized = np.array([
            (0.840625, 0.997222), (0.625000, 0.658333), (0.596875, 0.655556),
            (0.584375, 0.633333), (0.426563, 0.633333), (0.403125, 0.655556),
            (0.365625, 0.661111), (0.153125, 0.997222)
        ], dtype=np.float32)

        detector = YoloDetector(ego_vehicle_mask_normalized=ego_vehicle_mask_normalized)
        lane_detector = UFLDLaneDetector(use_kalman=True)

        # Get project root and find latest traffic light classifier model
        project_root = Path(__file__).parent.parent.parent.parent
        model_dir = project_root / "models" / "traffic_light_classifier"
        model_folders = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("traffic_light_cls_")])
        if not model_folders:
            raise FileNotFoundError(f"No traffic light models found in {model_dir}")
        latest_model = model_folders[-1] / "weights" / "best.pt"
        traffic_light_classifier = TrafficLightYoloClassifier(model_path=str(latest_model))

        # Create perception configuration
        from lane_change.config.perception_config import PerceptionConfig
        perception_config = PerceptionConfig()

        # Create perception pipeline with dependency injection (without ego_vehicle reference yet)
        from lane_change.perception.perception_core import PerceptionPipeline
        perception_pipeline = PerceptionPipeline(
            sensor_manager=sensor_manager,
            detector=detector,
            lane_detector=lane_detector,
            velocity_estimator=velocity_estimator,
            traffic_light_classifier=traffic_light_classifier,
            ego_vehicle=None,  # Will be set after ego_vehicle is created
            config=perception_config
        )

        # Auto-create vehicle controller with perception pipeline
        from lane_change.plant.carla_vehicle_controller import CarlaVehicleController
        controller = CarlaVehicleController(carla_actor, perception_core=perception_pipeline)
        

        # Create CarlaEgoVehicle object
        from lane_change.plant.data_types import CarlaEgoVehicle
        ego_vehicle = CarlaEgoVehicle(
            id=1,
            carla_actor=carla_actor,
            controller=controller,
            sensor_manager=sensor_manager
        )

        # Set ego_vehicle reference now that circular dependency is resolved
        perception_pipeline.ego_vehicle = ego_vehicle

        # Store the ego vehicle and its actor ID
        self.ego_vehicle = ego_vehicle
        self.my_ego_actor_id = carla_actor.id

        ego_vehicle.controller.set_speed(0.0)

        logger.info(f"Created ego vehicle (actor_id={self.my_ego_actor_id})")

        return ego_vehicle

    def get_world(self):
        """Get the CARLA world object."""
        return self.world

    def get_map(self):
        """Get the CARLA map object."""
        return self.world.get_map() if self.world else None


    def get_random_spawn_point(self):
        """Get random spawn point from map."""
        import random
        all_spawn_points = self.world.get_map().get_spawn_points()
        return random.choice(all_spawn_points)

    def get_configured_spawn_point(self):
        """
        Get spawn point closest to configured location from carla_config.yaml.

        Returns:
            carla.Transform: Spawn point closest to configured location
        """
        ego_config = self.config.get('ego_vehicle', {})
        spawn_location = ego_config.get('spawn_location')

        if not spawn_location:
            logger.warning("No spawn_location configured, using random spawn point")
            return self.get_random_spawn_point()

        x = spawn_location.get('x')
        y = spawn_location.get('y')

        # Get all spawn points sorted by distance from configured location
        all_spawn_points = self.world.get_map().get_spawn_points()
        target_location = carla.Location(x=x, y=y, z=0)

        # Sort by distance and return closest
        sorted_spawns = sorted(
            all_spawn_points,
            key=lambda sp: sp.location.distance(target_location)
        )

        closest_spawn = sorted_spawns[0]
        distance = closest_spawn.location.distance(target_location)
        logger.debug(f"Using spawn point {distance:.1f}m from configured location ({x:.1f}, {y:.1f})")

        return closest_spawn

    def cleanup(self):
        """
        Clean up client-side resources.

        Server destroys actors - client just clears references.
        """
        try:
            if self.ego_vehicle:
                # Clear sensor manager (keeps sensor references alive to prevent __del__)
                if self.ego_vehicle.sensor_manager:
                    self.ego_vehicle.sensor_manager.destroy_sensors()

                # Clear ego vehicle reference
                self.ego_vehicle = None
                self.my_ego_actor_id = None

            logger.info("Client-side cleanup complete")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup on exit."""
        logger.info("Client exiting...")

        try:
            # Request server to destroy actors (if server_client provided)
            if self.server_client and self.my_ego_actor_id:
                logger.info("Requesting server cleanup...")
                self.server_client.request_cleanup(self.my_ego_actor_id)

            # Run local cleanup to clear references
            self.cleanup()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}", exc_info=True)

        logger.info("Client shutdown complete")
        return False
