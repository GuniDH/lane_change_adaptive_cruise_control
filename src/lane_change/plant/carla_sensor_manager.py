"""
CARLA Sensor Manager - Sensor abstraction for CARLA simulator.

This module handles sensor management and data conversion from
raw CARLA formats to abstract data types. Implements ISensorManager interface.
"""

import carla
import numpy as np
import logging
import yaml
import threading
import time
from pathlib import Path
from typing import Optional
from queue import Queue, Empty
from dataclasses import dataclass
from lane_change.gateway.sensor_manager import ISensorManager
from lane_change.gateway.data_types import CameraImage, LidarData, RadarData

logger = logging.getLogger(__name__)


@dataclass
class SensorHealth:
    """Tracks health status of a sensor for failover detection."""
    state: str = 'HEALTHY'
    last_frame: int = 0
    last_update_time: float = 0.0
    consecutive_failures: int = 0


class CarlaSensorManager(ISensorManager):
    """Manages sensors for the vehicle - CARLA sensor abstraction."""

    def __init__(self, vehicle: 'carla.Actor', world: 'carla.World', config_path: Optional[str] = None):
        """
        Initialize sensor manager for a vehicle.

        Args:
            vehicle: CARLA vehicle actor to attach sensors to
            world: CARLA world object
            config_path: Path to carla_config.yaml (optional, defaults to config/carla_config.yaml)
        """
        self.vehicle = vehicle
        self.world = world
        self.sensors = {}  # Dict[str, carla.Sensor]
        self.latest_data = {}  # Dict[str, CameraImage|LidarData|RadarData]

        # Thread synchronization for sensor data access
        self.data_lock = threading.Lock()

        # Callback tracking for clean shutdown
        self.active_callbacks = 0
        self.callback_lock = threading.Lock()

        # Synchronization queue - thread safe for sensor callbacks
        self.sensor_queue = Queue()
        self.sensor_count = 0  # Will be set after spawning sensors

        # Load configuration
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "carla_config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Sensor configuration (populated after spawning)
        self.camera_config = {}
        self.lidar_config = {}
        self.radar_config = {}

        # Failover tracking
        self.backup_sensors = {}  # Dict[str, carla.Sensor] - spawned on demand
        self.active_sensors = {}  # Dict[str, str] - maps logical name to active sensor name
        self.sensor_health = {}  # Dict[str, SensorHealth]
        self.frame_count = 0  # Track frames for health monitoring

        # Failover configuration
        self.failover_enabled = self.config.get('sensors', {}).get('failover', {}).get('enabled', True)
        self.timeout_frames = self.config.get('sensors', {}).get('failover', {}).get('timeout_frames', 5)
        failover_validation = self.config.get('sensors', {}).get('failover', {}).get('validation', {})
        self.validate_empty = failover_validation.get('check_empty', True)
        self.validate_nan = failover_validation.get('check_nan', True)
        self.consecutive_failures_threshold = failover_validation.get('consecutive_failures', 5)

    def spawn_sensors(self) -> None:
        """Spawn front camera, LiDAR, and radar sensors from config."""
        # Spawn primary sensors
        self._spawn_single_sensor('camera', 'camera_front')
        self._spawn_single_sensor('lidar', 'lidar_front')
        self._spawn_single_sensor('radar', 'radar_front')

        # Calculate sensor count dynamically based on what was actually spawned
        self.sensor_count = len(self.sensors)

        # Initialize health tracking and active sensor mappings
        for sensor_name in self.sensors.keys():
            self.sensor_health[sensor_name] = SensorHealth(
                state='HEALTHY',
                last_frame=0,
                last_update_time=time.time()
            )
            self.active_sensors[sensor_name] = sensor_name  # Initially, active = primary

        if self.failover_enabled:
            logger.info(f"Sensor failover enabled: timeout={self.timeout_frames} frames, validation={'ON' if self.validate_empty or self.validate_nan else 'OFF'}")

    def _spawn_single_sensor(self, sensor_type: str, sensor_name: str) -> None:
        """
        Spawn a single sensor with configuration from config file.

        Args:
            sensor_type: Type of sensor ('camera', 'lidar', 'radar')
            sensor_name: Unique name for this sensor instance (e.g., 'camera_front', 'camera_front_backup')
        """
        blueprint_library = self.world.get_blueprint_library()
        is_backup = '_backup' in sensor_name
        target_dict = self.backup_sensors if is_backup else self.sensors

        if sensor_type == 'camera':
            cam_cfg = self.config['sensors']['camera']
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(cam_cfg['image_size_x']))
            camera_bp.set_attribute('image_size_y', str(cam_cfg['image_size_y']))
            camera_bp.set_attribute('fov', str(cam_cfg['fov']))

            camera_transform = carla.Transform(
                carla.Location(x=cam_cfg['location']['x'], y=cam_cfg['location']['y'], z=cam_cfg['location']['z']),
                carla.Rotation(pitch=cam_cfg['rotation']['pitch'], yaw=cam_cfg['rotation']['yaw'], roll=cam_cfg['rotation']['roll'])
            )
            camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            target_dict[sensor_name] = camera
            camera.listen(lambda data: self._sensor_callback(data, sensor_name))

            # Store camera configuration for calibration (only once for primary)
            if not is_backup:
                self.camera_config = {
                    'fov': float(camera_bp.get_attribute('fov').as_float()),
                    'location': (camera_transform.location.x, camera_transform.location.y, camera_transform.location.z),
                    'pitch': camera_transform.rotation.pitch
                }

            logger.info(f"Spawned camera sensor: {sensor_name}")

        elif sensor_type == 'lidar':
            lidar_cfg = self.config['sensors']['lidar']
            lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('channels', str(lidar_cfg['channels']))
            lidar_bp.set_attribute('range', str(lidar_cfg['range']))
            lidar_bp.set_attribute('points_per_second', str(lidar_cfg['points_per_second']))
            lidar_bp.set_attribute('rotation_frequency', str(lidar_cfg['rotation_frequency']))
            lidar_bp.set_attribute('horizontal_fov', str(lidar_cfg['horizontal_fov']))
            lidar_bp.set_attribute('upper_fov', str(lidar_cfg['upper_fov']))
            lidar_bp.set_attribute('lower_fov', str(lidar_cfg['lower_fov']))
            lidar_bp.set_attribute('sensor_tick', str(lidar_cfg['sensor_tick']))

            lidar_transform = carla.Transform(
                carla.Location(x=lidar_cfg['location']['x'], y=lidar_cfg['location']['y'], z=lidar_cfg['location']['z'])
            )
            lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
            target_dict[sensor_name] = lidar
            lidar.listen(lambda data: self._sensor_callback(data, sensor_name))

            # Store LiDAR configuration (only once for primary)
            if not is_backup:
                self.lidar_config = {
                    'location': (lidar_transform.location.x, lidar_transform.location.y, lidar_transform.location.z)
                }

            logger.info(f"Spawned LiDAR sensor: {sensor_name}")

        elif sensor_type == 'radar':
            radar_cfg = self.config['sensors']['radar']
            radar_bp = blueprint_library.find('sensor.other.radar')
            radar_bp.set_attribute('horizontal_fov', str(radar_cfg['horizontal_fov']))
            radar_bp.set_attribute('vertical_fov', str(radar_cfg['vertical_fov']))
            radar_bp.set_attribute('range', str(radar_cfg['range']))
            radar_bp.set_attribute('points_per_second', str(radar_cfg['points_per_second']))
            radar_bp.set_attribute('sensor_tick', str(radar_cfg['sensor_tick']))

            radar_transform = carla.Transform(
                carla.Location(x=radar_cfg['location']['x'], y=radar_cfg['location']['y'], z=radar_cfg['location']['z'])
            )
            radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.vehicle)
            target_dict[sensor_name] = radar
            radar.listen(lambda data: self._sensor_callback(data, sensor_name))

            # Store radar configuration (only once for primary)
            if not is_backup:
                self.radar_config = {
                    'location': (radar_transform.location.x, radar_transform.location.y, radar_transform.location.z)
                }

            logger.info(f"Spawned radar sensor: {sensor_name}")

    def _sensor_callback(self, sensor_data, sensor_name: str):
        """
        Unified sensor callback for synchronization.
        Stores data and pushes frame info to queue.
        """
        # Track active callback
        with self.callback_lock:
            self.active_callbacks += 1

        try:
            # Store data based on sensor type
            if sensor_name == 'camera_front' or sensor_name == 'camera_front_backup':
                self._store_camera_data(sensor_data, sensor_name)
            elif sensor_name == 'lidar_front' or sensor_name == 'lidar_front_backup':
                self._store_lidar_data(sensor_data, sensor_name)
            elif sensor_name == 'radar_front' or sensor_name == 'radar_front_backup':
                self._store_radar_data(sensor_data, sensor_name)

            # Update health tracking (track primary sensor name, not backup)
            primary_name = sensor_name.replace('_backup', '')
            if primary_name in self.sensor_health:
                self.sensor_health[primary_name].last_frame = self.frame_count
                self.sensor_health[primary_name].last_update_time = time.time()

            # Push frame and sensor name to queue for synchronization
            self.sensor_queue.put((sensor_data.frame, sensor_name))
        finally:
            # Always decrement, even if error occurs
            with self.callback_lock:
                self.active_callbacks -= 1

    def _store_camera_data(self, image: 'carla.Image', sensor_name: str):
        """Convert and store camera data (thread-safe, drops old frames)."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]  # BGRA to RGB

        with self.data_lock:
            # Store with sensor-specific key (supports both primary and backup)
            self.latest_data[sensor_name] = CameraImage(
                image_data=array,
                width=image.width,
                height=image.height
            )

    def _store_lidar_data(self, lidar_measurement: 'carla.LidarMeasurement', sensor_name: str):
        """Convert and store LiDAR point cloud data (thread-safe, drops old frames)."""
        point_count = len(lidar_measurement)
        points = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32)
        points = np.reshape(points, (point_count, 4))  # [x, y, z, intensity]
        points_xyz = points[:, :3]  # Extract only XYZ coordinates

        with self.data_lock:
            # Store with sensor-specific key (supports both primary and backup)
            self.latest_data[sensor_name] = LidarData(points=points_xyz)

    def _store_radar_data(self, radar_measurement: 'carla.RadarMeasurement', sensor_name: str):
        """Convert and store radar data (thread-safe, drops old frames)."""
        point_count = len(radar_measurement)
        points = np.frombuffer(radar_measurement.raw_data, dtype=np.float32)
        points = np.reshape(points, (point_count, 4))  # [velocity, azimuth, altitude, depth]

        with self.data_lock:
            # Store with sensor-specific key (supports both primary and backup)
            self.latest_data[sensor_name] = RadarData(points=points)

    def wait_for_sensors(self, timeout: float = 1.0) -> bool:
        """
        Wait for all sensors to receive data for the current frame.
        Ensures all sensors are synchronized to the same frame.

        Args:
            timeout: Timeout in seconds for each sensor

        Returns:
            True if all sensors received, False if timeout
        """
        import time

        deadline = time.time() + timeout
        sensors_for_frame = {}

        while time.time() < deadline:
            try:
                frame, sensor_name = self.sensor_queue.get(True, 0.1)

                # Track sensors per frame
                if frame not in sensors_for_frame:
                    sensors_for_frame[frame] = set()
                sensors_for_frame[frame].add(sensor_name)

                # Check if we have all sensors for any frame
                if len(sensors_for_frame[frame]) == self.sensor_count:
                    logger.debug(f"Frame sync OK: frame={frame}, sensors={sorted(sensors_for_frame[frame])}")
                    return True

            except Empty:
                continue

        # Timeout - log what we received
        logger.warning(f"Timeout: frames_received={sensors_for_frame}")
        return False

    def get_camera_image(self, sensor_id: str = 'front') -> Optional[CameraImage]:
        """
        Get the latest camera image from specified sensor (thread-safe).

        Always returns the newest frame, old frames are automatically dropped.
        Routes through active sensor (primary or backup after failover).
        """
        sensor_name = f'camera_{sensor_id}'
        active_sensor = self.active_sensors.get(sensor_name, sensor_name)

        with self.data_lock:
            return self.latest_data.get(active_sensor)

    def get_lidar_data(self, sensor_id: str = 'front') -> Optional[LidarData]:
        """
        Get the latest LiDAR point cloud from specified sensor (thread-safe).

        Always returns the newest frame, old frames are automatically dropped.
        Routes through active sensor (primary or backup after failover).
        """
        sensor_name = f'lidar_{sensor_id}'
        active_sensor = self.active_sensors.get(sensor_name, sensor_name)

        with self.data_lock:
            return self.latest_data.get(active_sensor)

    def get_radar_data(self, sensor_id: str = 'front') -> Optional[RadarData]:
        """
        Get the latest radar data from specified sensor (thread-safe).

        Always returns the newest frame, old frames are automatically dropped.
        Routes through active sensor (primary or backup after failover).
        """
        sensor_name = f'radar_{sensor_id}'
        active_sensor = self.active_sensors.get(sensor_name, sensor_name)

        with self.data_lock:
            return self.latest_data.get(active_sensor)

    def check_sensor_health(self) -> None:
        """
        Check health of all sensors and trigger failover if needed.
        Should be called once per frame before processing sensor data.
        """
        if not self.failover_enabled:
            return

        self.frame_count += 1

        for sensor_name in ['camera_front', 'lidar_front', 'radar_front']:
            if sensor_name not in self.sensor_health:
                continue

            health = self.sensor_health[sensor_name]

            # Skip if already failed over
            if health.state == 'FAILED':
                continue

            # Check 1: Timeout (no data for N frames)
            frames_since_update = self.frame_count - health.last_frame
            if frames_since_update >= self.timeout_frames:
                logger.warning(f"Sensor {sensor_name} timeout: {frames_since_update} frames without data")
                self._trigger_failover(sensor_name, "timeout")
                continue

            # Check 2: Data validation
            active_sensor = self.active_sensors.get(sensor_name, sensor_name)
            with self.data_lock:
                data = self.latest_data.get(active_sensor)

            if not self._validate_sensor_data(sensor_name, data):
                health.consecutive_failures += 1
                if health.consecutive_failures >= self.consecutive_failures_threshold:
                    logger.warning(f"Sensor {sensor_name} validation failed {health.consecutive_failures} times")
                    self._trigger_failover(sensor_name, "invalid_data")
            else:
                health.consecutive_failures = 0

    def _validate_sensor_data(self, sensor_name: str, data) -> bool:
        """
        Validate sensor data for health monitoring.

        Args:
            sensor_name: Name of the sensor (e.g., 'camera_front')
            data: Sensor data to validate

        Returns:
            True if data is valid, False otherwise
        """
        # Check for None or empty data
        if self.validate_empty:
            if data is None:
                return False

        if 'camera' in sensor_name:
            if self.validate_empty:
                if data.image_data is None or data.image_data.size == 0:
                    return False
            if self.validate_nan:
                if np.any(np.isnan(data.image_data)):
                    return False

        elif 'lidar' in sensor_name:
            if self.validate_empty:
                if data.points is None or len(data.points) == 0:
                    return False
            if self.validate_nan:
                if not np.all(np.isfinite(data.points)):
                    return False

        elif 'radar' in sensor_name:
            if self.validate_empty:
                if data.points is None or len(data.points) == 0:
                    return False
            if self.validate_nan:
                if np.any(np.isnan(data.points)):
                    return False

        return True

    def _trigger_failover(self, sensor_name: str, reason: str) -> None:
        """
        Trigger failover from primary to backup sensor.

        Args:
            sensor_name: Primary sensor name (e.g., 'camera_front')
            reason: Reason for failover (for logging)
        """
        logger.error(f"FAILOVER TRIGGERED: {sensor_name} ({reason}) - spawning backup sensor")

        # Mark sensor as failed
        self.sensor_health[sensor_name].state = 'FAILED'

        # Stop primary sensor listener if still alive
        if sensor_name in self.sensors:
            try:
                self.sensors[sensor_name].stop()
                logger.info(f"Stopped primary sensor: {sensor_name}")
            except Exception as e:
                logger.warning(f"Failed to stop primary sensor {sensor_name}: {e}")

        # Spawn backup sensor
        backup_name = f"{sensor_name}_backup"
        try:
            self._spawn_backup_sensor(sensor_name)
            logger.info(f"Successfully spawned backup sensor: {backup_name}")

            # Update active sensor routing
            self.active_sensors[sensor_name] = backup_name

            # Reset health tracking for the backup
            self.sensor_health[sensor_name].last_frame = self.frame_count
            self.sensor_health[sensor_name].last_update_time = time.time()
            self.sensor_health[sensor_name].consecutive_failures = 0

            logger.info(f"Failover complete: {sensor_name} -> {backup_name}")

        except Exception as e:
            logger.error(f"CRITICAL: Backup sensor spawn failed for {sensor_name}: {e}")
            raise

    def _spawn_backup_sensor(self, sensor_name: str) -> None:
        """
        Spawn a backup sensor with identical configuration to primary.

        Args:
            sensor_name: Primary sensor name (e.g., 'camera_front')
        """
        # Extract sensor type from name
        sensor_type = sensor_name.split('_')[0]  # 'camera_front' -> 'camera'
        backup_name = f"{sensor_name}_backup"

        # Use shared spawning logic
        self._spawn_single_sensor(sensor_type, backup_name)

    def destroy_sensors(self) -> None:
        """
        Minimal cleanup - keep sensor references alive.

        Server destroys the actual CARLA actors. Client must NOT clear sensor
        references because that triggers Python __del__ which tries to communicate
        with already-destroyed actors, causing socket errors.

        Keep references alive and let Python process exit naturally.
        """
        # Do NOT clear self.sensors - would trigger __del__ on destroyed actors
        # Only clear data
        self.latest_data.clear()