"""
CARLA Base Server - Base class for all CARLA server implementations.

This module provides shared functionality for CARLA servers:
- Connection to CARLA
- Synchronous mode configuration
- Traffic Manager setup
- World ticking
- Cleanup

Subclasses:
- CarlaServer (open-world multi-client)
- TrainingServer (single-client training)
"""

import carla
import logging
import yaml
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CarlaBaseServer(ABC):
    """
    Base class for CARLA servers with shared functionality.

    Provides common CARLA setup and management logic.
    Subclasses implement specific server behavior.
    """

    def __init__(self, host='localhost', port=2000, config=None):
        """
        Initialize base server.

        Args:
            host: CARLA server host address
            port: CARLA server port
            config: Configuration dictionary (optional)
        """
        self.host = host
        self.port = port

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
        self.original_settings = None

        # Simulation state
        self.running = False

    def connect_to_carla(self, map_name=None):
        """
        Connect to CARLA server and load map with retry logic.

        Args:
            map_name: Map to load (e.g., 'Town12'). If None, uses config or current map.
        """
        import time

        max_retries = 5
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"Retry {attempt}/{max_retries}: Connecting to CARLA server at {self.host}:{self.port}")
                else:
                    logger.info(f"Connecting to CARLA server at {self.host}:{self.port}")

                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(
                    self.config.get('carla', {}).get('timeout', 10.0)
                )

                self.world = self.client.get_world()
                logger.info(f"Connected to CARLA world: {self.world.get_map().name}")

                # Store original settings for cleanup
                self.original_settings = self.world.get_settings()

                # Load map if specified
                if map_name is None:
                    map_name = self.config.get('carla', {}).get('map_name')

                if map_name:
                    current_map = self.world.get_map().name
                    if map_name not in current_map:
                        logger.info(f"Loading map: {map_name}")
                        self.world = self.client.load_world(map_name)
                        logger.info(f"Map loaded: {map_name}")
                    else:
                        logger.info(f"Map already loaded: {map_name}")

                return

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to connect to CARLA after {max_retries} attempts: {e}")
                    raise

    def setup_traffic_manager(self, synchronous_mode=True):
        """
        Setup Traffic Manager.

        Args:
            synchronous_mode: Whether to enable synchronous mode for TM
        """
        tm_port = self.config.get('traffic', {}).get('traffic_manager_port', 8000)
        self.carla_tm = self.client.get_trafficmanager(tm_port)
        self.carla_tm.set_synchronous_mode(synchronous_mode)

        self.carla_tm.set_global_distance_to_leading_vehicle(
            self.config.get('traffic', {}).get('global_distance_to_leading_vehicle', 2.5)
        )

        logger.info(f"Traffic Manager configured (sync={synchronous_mode})")

    def enable_synchronous_mode(self, delta_seconds=0.05):
        """
        Enable synchronous mode for deterministic simulation.

        Args:
            delta_seconds: Fixed time step in seconds (default: 0.05 = 20 Hz)
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta_seconds

        # Performance optimizations
        settings.tile_stream_distance = 500
        settings.actor_active_distance = 500

        self.world.apply_settings(settings)

        # Unload map layers for performance
        self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.Ground)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)
        self.world.unload_map_layer(carla.MapLayer.Walls)

        logger.info(f"Synchronous mode enabled: {delta_seconds}s per tick ({1.0/delta_seconds:.1f} Hz)")

    def tick_world(self):
        """Tick the world once."""
        self.world.tick()

    def cleanup(self):
        """Restore original world settings."""
        logger.info("Starting server cleanup...")

        if self.world and self.original_settings:
            try:
                logger.info("Restoring original world settings...")
                self.world.apply_settings(self.original_settings)
                settings = self.world.get_settings()
                settings.synchronous_mode = True # Always have sync mode to ensure clients dont tick themselves when server is down

                logger.info("World settings restored")
            except Exception as e:
                logger.error(f"Failed to restore settings: {e}")

        logger.info("Server cleanup complete")

    @abstractmethod
    def run(self):
        """
        Main server loop.

        Subclasses must implement this method with their specific logic.
        """
        pass
