"""
Perception Pipeline - Main perception pipeline for vehicle detection and tracking.

This module orchestrates the perception pipeline using dependency injection.
Coordinates detector execution, manages frame-level state, and preprocesses sensor data.
"""

import logging
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from lane_change.gateway.sensor_manager import ISensorManager
from lane_change.gateway.velocity_estimator import IVelocityEstimator
from lane_change.gateway.detector import Detector, ILaneDetector, ITrafficLightClassifier
from lane_change.gateway.data_types import DetectedObject, VelocityEstimate, Position3D, BoundingBox
from lane_change.perception.tracked_objects import (
    TrackedObject, TrackedMovingObject, TrackedVehicle, TrackedPedestrian,
    TrackedStaticObject, TrackedTrafficLight
)
from lane_change.perception.hysteresis import HysteresisFilter
from lane_change.config.perception_config import PerceptionConfig

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """
    Orchestrates perception pipeline using dependency injection.

    Responsibilities:
    - Coordinate detector execution order
    - Manage frame-level state (hysteresis, caching)
    - Preprocess sensor data once per frame
    - Orchestrate object creation and metric computation

    Uses injected interfaces directly (NO service wrappers).
    Domain logic lives on domain objects (TrackedVehicle.assign_lane, TrackedObject.update_metrics).
    """

    def __init__(
        self,
        sensor_manager: ISensorManager,
        detector: Detector,
        lane_detector: ILaneDetector,
        velocity_estimator: IVelocityEstimator,
        traffic_light_classifier: ITrafficLightClassifier,
        ego_vehicle: 'CarlaEgoVehicle',
        config: PerceptionConfig
    ):
        """
        Initialize perception pipeline with dependency injection.

        Args:
            sensor_manager: Sensor manager interface for accessing sensor data
            detector: Object detector interface (e.g., YoloDetector)
            lane_detector: Lane detector interface (e.g., UFLDLaneDetector)
            velocity_estimator: Velocity estimator interface for multi-sensor fusion
            traffic_light_classifier: Traffic light classifier interface
            ego_vehicle: Ego vehicle reference for velocity and metrics
            config: Perception configuration
        """
        # Dependency injection - all detectors via interfaces
        self.sensor_manager = sensor_manager
        self.detector = detector
        self.lane_detector = lane_detector
        self.velocity_estimator = velocity_estimator
        self.traffic_light_classifier = traffic_light_classifier
        self.ego_vehicle = ego_vehicle
        self.config = config

        # Ego vehicle mask polygon in NORMALIZED coordinates (0.0-1.0 range)
        # These coordinates are resolution-independent and will be scaled to actual camera dimensions
        # Used for: 1) Filtering LiDAR points, 2) Filtering radar points, 3) Blacking out region in YOLO input
        #
        # Original polygon was defined for 640x360 resolution:
        # (538, 359), (400, 237), (382, 236), (374, 228), (273, 228), (258, 236), (234, 238), (98, 359)
        self.ego_vehicle_mask_normalized = np.array([
            (0.840625, 0.997222),  # (538/640, 359/360)
            (0.625000, 0.658333),  # (400/640, 237/360)
            (0.596875, 0.655556),  # (382/640, 236/360)
            (0.584375, 0.633333),  # (374/640, 228/360)
            (0.426563, 0.633333),  # (273/640, 228/360)
            (0.403125, 0.655556),  # (258/640, 236/360)
            (0.365625, 0.661111),  # (234/640, 238/360)
            (0.153125, 0.997222),  # (98/640, 359/360)
        ], dtype=np.float32)

        # Scaled polygon (will be initialized on first frame when we know actual camera dimensions)
        self.ego_vehicle_mask = None

        # Pre-compute binary mask image for efficient point filtering (O(1) lookup vs O(M) polygon test)
        # Will be initialized on first frame when we know camera dimensions
        self.ego_mask_image = None
        self.ego_mask_image_gpu = None  # GPU version for full-GPU filtering pipeline

        # Pipeline-level state
        self.tracked_vehicles: Dict[int, TrackedVehicle] = {}
        self.frame_count = 0

        # Hysteresis filters for temporal smoothing
        self.lane_hysteresis = HysteresisFilter(threshold_frames=self.config.lane_hysteresis_frames)
        self.color_hysteresis = HysteresisFilter(threshold_frames=self.config.color_hysteresis_frames)
        self.high_security_mode = True
        self.current_lanes: List[List[Tuple[int, int]]] = []
        self.sorted_lanes: List[List[Tuple[int, int]]] = []  # Cached sorted lanes (by Y coord)
        self.sorted_lanes_gpu = []  # GPU tensors for sorted lanes (for fast Strategy 1)
        self.ego_lane_idx = 1  # 1-based from leftmost
        self.lidar_points = None  # Filtered LiDAR points for current frame (for visualization)
        self.radar_points = None  # Filtered radar points for current frame (for visualization)

        # Frame-level cached sensor data (pre-transformed and projected)
        self.lidar_camera_frame = None  # LiDAR in camera frame
        self.lidar_vehicle_frame = None  # LiDAR in vehicle frame (filtered)
        self.lidar_image_uv = None  # LiDAR projected to image (u, v)
        self.radar_xyz_all = None  # Radar in Cartesian coords
        self.radar_camera_frame = None  # Radar in camera frame
        self.radar_image_uv = None  # Radar projected to image (u, v)
        self.radar_spherical = None  # Original radar measurements
        self.camera_K = None  # Cached camera intrinsics

        # GPU tensors (set by preprocessing if GPU available)
        self.lidar_vehicle_frame_gpu = None
        self.lidar_camera_frame_gpu = None
        self.lidar_image_uv_gpu = None
        self.radar_xyz_all_gpu = None
        self.radar_image_uv_gpu = None

        # Persistent ThreadPoolExecutor for parallel vehicle processing
        # Created once and reused every frame to avoid creation/destruction overhead
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix="VehicleProcessor")
        logger.info(f"Initialized persistent ThreadPoolExecutor with {self.config.max_workers} workers")

        # Lane detection optimization: don't detect every frame (lanes don't move!)
        self.lane_detection_interval = self.config.lane_detection_interval
        self.lane_frame_counter = 0

    def __del__(self):
        """Cleanup: shutdown thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            logger.info("ThreadPoolExecutor shut down")

    def process_frame(self) -> Tuple[List[TrackedMovingObject], Optional[TrackedTrafficLight]]:
        """
        Process one frame of camera data and return tracked moving objects (vehicles and pedestrians) with lane assignments,
        plus the closest traffic light ahead.

        Returns:
            Tuple of (tracked moving objects, closest traffic light ahead or None)
        """
        frame_start = time.perf_counter()

        # Check sensor health and trigger failover if needed
        if hasattr(self.sensor_manager, 'check_sensor_health'):
            self.sensor_manager.check_sensor_health()

        # Get camera image
        camera_image = self.sensor_manager.get_camera_image('front')
        if camera_image is None:
            return [], None

        # Detect lanes and cache sorted version (for fast lane lookups)
        # Optimization: Only detect lanes every N frames (lanes don't move much!)
        t0 = time.perf_counter()
        if self.lane_frame_counter % self.lane_detection_interval == 0:
            # Full lane detection
            self.current_lanes = self.lane_detector.detect_lanes(camera_image)
            self.sorted_lanes = [sorted(lane, key=lambda p: p[1]) for lane in self.current_lanes]
        # else: reuse previous lanes (self.current_lanes, self.sorted_lanes unchanged)

        lane_time = (time.perf_counter() - t0) * 1000
        self.lane_frame_counter += 1

        # Cache GPU version of sorted lanes for GPU-accelerated voting (if GPU available)
        try:
            import torch
            if torch.cuda.is_available():
                self.sorted_lanes_gpu = [
                    torch.tensor(lane, dtype=torch.float32).cuda()
                    for lane in self.sorted_lanes
                ]
        except (ImportError, RuntimeError):
            self.sorted_lanes_gpu = []

        # Calculate ego lane
        self.ego_lane_idx = self._calculate_ego_lane(camera_image)

        # Run YOLO detection with tracking
        t0 = time.perf_counter()
        detections = self.detector.detect_objects(camera_image)
        yolo_time = (time.perf_counter() - t0) * 1000

        # Separate traffic lights from moving objects
        traffic_light_detections = [d for d in detections if d.class_name == "traffic_light"]
        moving_object_detections = [d for d in detections if d.class_name != "traffic_light"]

        # Get sensor data for velocity estimation
        lidar_data = None
        radar_data = None
        ego_velocity = np.zeros(3)
        preprocess_time = 0.0

        if self.velocity_estimator is not None:
            lidar_data = self.sensor_manager.get_lidar_data('front')
            radar_data = self.sensor_manager.get_radar_data('front')

            # Preprocess ALL sensor data ONCE per frame
            t0 = time.perf_counter()
            self._preprocess_sensors_once_per_frame(camera_image, lidar_data, radar_data)
            preprocess_time = (time.perf_counter() - t0) * 1000

            # Filter out ego vehicle sensor points using preprocessed projections
            if lidar_data is not None and radar_data is not None:
                # Initialize ego mask image on first use
                self._initialize_ego_mask_image(camera_image)

                # LiDAR ego filtering - GPU path if available (faster, no CPU↔GPU transfers)
                if self.lidar_vehicle_frame_gpu is not None and self.ego_mask_image_gpu is not None:
                    import torch
                    # Full GPU pipeline - everything stays on GPU
                    u_gpu = torch.clamp(self.lidar_image_uv_gpu[:, 0], 0, camera_image.width - 1).long()
                    v_gpu = torch.clamp(self.lidar_image_uv_gpu[:, 1], 0, camera_image.height - 1).long()

                    # Lookup in GPU mask
                    lidar_ego_mask_gpu = self.ego_mask_image_gpu[v_gpu, u_gpu] > 0
                    non_ego_mask_gpu = ~lidar_ego_mask_gpu

                    # Filter on GPU
                    self.lidar_vehicle_frame_gpu = self.lidar_vehicle_frame_gpu[non_ego_mask_gpu]
                    self.lidar_camera_frame_gpu = self.lidar_camera_frame_gpu[non_ego_mask_gpu]
                    self.lidar_image_uv_gpu = self.lidar_image_uv_gpu[non_ego_mask_gpu]

                    # Download to CPU only once (for compatibility)
                    self.lidar_vehicle_frame = self.lidar_vehicle_frame_gpu.cpu().numpy()
                    self.lidar_camera_frame = self.lidar_camera_frame_gpu.cpu().numpy()
                    self.lidar_image_uv = self.lidar_image_uv_gpu.cpu().numpy()

                    logger.debug(f"Ego filter (GPU): removed {lidar_ego_mask_gpu.sum().item()} / {len(lidar_ego_mask_gpu)} LiDAR points")
                else:
                    # CPU fallback
                    u_int = np.clip(self.lidar_image_uv[:, 0], 0, camera_image.width - 1).astype(np.int32)
                    v_int = np.clip(self.lidar_image_uv[:, 1], 0, camera_image.height - 1).astype(np.int32)
                    lidar_ego_mask = self.ego_mask_image[v_int, u_int] > 0

                    # Apply ego filter to cached preprocessed data
                    non_ego_mask = ~lidar_ego_mask
                    self.lidar_vehicle_frame = self.lidar_vehicle_frame[non_ego_mask]
                    self.lidar_camera_frame = self.lidar_camera_frame[non_ego_mask]
                    self.lidar_image_uv = self.lidar_image_uv[non_ego_mask]

                    logger.debug(f"Ego filter (CPU): removed {lidar_ego_mask.sum()} / {len(lidar_ego_mask)} LiDAR points")

                # Store filtered data for visualization and update raw data
                self.lidar_points = self.lidar_vehicle_frame
                lidar_data.points = self.lidar_vehicle_frame

                # Radar ego filtering - GPU path if available (faster, no CPU↔GPU transfers)
                if self.radar_image_uv_gpu is not None and self.ego_mask_image_gpu is not None:
                    import torch
                    # Full GPU pipeline - everything stays on GPU
                    u_gpu = torch.clamp(self.radar_image_uv_gpu[:, 0], 0, camera_image.width - 1).long()
                    v_gpu = torch.clamp(self.radar_image_uv_gpu[:, 1], 0, camera_image.height - 1).long()

                    # Lookup in GPU mask
                    radar_ego_mask_gpu = self.ego_mask_image_gpu[v_gpu, u_gpu] > 0
                    non_radar_ego_mask_gpu = ~radar_ego_mask_gpu

                    # Filter on GPU
                    self.radar_image_uv_gpu = self.radar_image_uv_gpu[non_radar_ego_mask_gpu]
                    self.radar_xyz_all_gpu = self.radar_xyz_all_gpu[non_radar_ego_mask_gpu]

                    # Download to CPU (for compatibility with CPU arrays)
                    self.radar_image_uv = self.radar_image_uv_gpu.cpu().numpy()
                    self.radar_xyz_all = self.radar_xyz_all_gpu.cpu().numpy()

                    # Apply same mask to CPU-only arrays (spherical coordinates)
                    non_radar_ego_mask = non_radar_ego_mask_gpu.cpu().numpy()
                    self.radar_points = self.radar_spherical[non_radar_ego_mask]
                    self.radar_spherical = self.radar_spherical[non_radar_ego_mask]

                    logger.debug(f"Ego filter (GPU): removed {radar_ego_mask_gpu.sum().item()} / {len(radar_ego_mask_gpu)} radar points")
                else:
                    # CPU fallback
                    u_int = np.clip(self.radar_image_uv[:, 0], 0, camera_image.width - 1).astype(np.int32)
                    v_int = np.clip(self.radar_image_uv[:, 1], 0, camera_image.height - 1).astype(np.int32)
                    radar_ego_mask = self.ego_mask_image[v_int, u_int] > 0

                    # Apply ego filter to radar and update cached preprocessed data
                    non_radar_ego_mask = ~radar_ego_mask
                    self.radar_points = self.radar_spherical[non_radar_ego_mask]
                    self.radar_xyz_all = self.radar_xyz_all[non_radar_ego_mask]
                    self.radar_image_uv = self.radar_image_uv[non_radar_ego_mask]
                    self.radar_spherical = self.radar_spherical[non_radar_ego_mask]

                    logger.debug(f"Ego filter (CPU): removed {radar_ego_mask.sum()} / {len(radar_ego_mask)} radar points")

                radar_data.points = self.radar_spherical
            else:
                self.lidar_points = None
                self.radar_points = None

            # Get ego velocity and rotation if available
            ego_velocity_world = np.zeros(3)
            ego_yaw = 0.0
            if self.ego_vehicle is not None and hasattr(self.ego_vehicle, 'carla_actor'):
                velocity_carla = self.ego_vehicle.carla_actor.get_velocity()
                ego_velocity_world = np.array([velocity_carla.x, velocity_carla.y, velocity_carla.z])

                # Get vehicle rotation (yaw angle)
                transform = self.ego_vehicle.carla_actor.get_transform()
                ego_yaw = np.deg2rad(transform.rotation.yaw)

        # Process moving objects (vehicles, pedestrians) in parallel: velocity estimation → lane assignment → filtering
        current_tracks = []
        current_time = time.time()

        # Parallel processing with persistent ThreadPoolExecutor
        # Each vehicle's DBSCAN clustering runs on separate thread → ~3-4x speedup
        t0 = time.perf_counter()
        if len(moving_object_detections) > 1:
            # Submit all vehicle processing tasks to persistent executor
            future_to_detection = {
                self.executor.submit(
                    self._process_single_vehicle,
                    detection,
                    camera_image,
                    lidar_data,
                    radar_data,
                    ego_velocity_world,
                    ego_yaw,
                    current_time
                ): detection
                for detection in moving_object_detections
            }

            # Collect results as they complete with error handling
            for future in as_completed(future_to_detection):
                detection = future_to_detection[future]
                try:
                    tracked = future.result()
                    if tracked is not None:
                        self.tracked_vehicles[tracked.track_id] = tracked
                        current_tracks.append(tracked)
                except Exception as e:
                    # Log error but continue processing other vehicles (graceful degradation)
                    bbox = detection.bounding_box
                    logger.error(
                        f"Vehicle processing failed for detection at ({bbox.x:.0f}, {bbox.y:.0f}): {e}",
                        exc_info=True
                    )
                    # Continue with other vehicles - one failure doesn't crash the frame
        else:
            # Single vehicle: skip thread overhead, process directly
            for detection in moving_object_detections:
                try:
                    tracked = self._process_single_vehicle(
                        detection, camera_image, lidar_data, radar_data,
                        ego_velocity_world, ego_yaw, current_time
                    )
                    if tracked is not None:
                        self.tracked_vehicles[tracked.track_id] = tracked
                        current_tracks.append(tracked)
                except Exception as e:
                    # Log error but continue (in case of multiple single-vehicle frames)
                    bbox = detection.bounding_box
                    logger.error(
                        f"Vehicle processing failed for detection at ({bbox.x:.0f}, {bbox.y:.0f}): {e}",
                        exc_info=True
                    )

        vehicle_processing_time = (time.perf_counter() - t0) * 1000

        # Process traffic lights
        t0 = time.perf_counter()
        closest_traffic_light = None

        if traffic_light_detections:
            tracked_traffic_lights = []

            # Calculate ego speed for TTC calculation
            ego_speed_ms = np.linalg.norm(ego_velocity_world) if self.velocity_estimator is not None else 0.0

            # Collect and expand all bboxes for batch inference (GPU efficient)
            # Expansion: 20% horizontal, 0% vertical (same for color and LiDAR)
            bboxes_expanded = []
            for detection in traffic_light_detections:
                bbox = detection.bounding_box
                x1 = int(bbox.x)
                y1 = int(bbox.y)
                x2 = int(bbox.x + bbox.width)
                y2 = int(bbox.y + bbox.height)

                # Expand bbox by 20% horizontally
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                expand_x = int(bbox_width * 0.2)
                expand_x = 0
                expand_y = 0  # No vertical expansion

                x1_expanded = max(0, x1 - expand_x)
                y1_expanded = max(0, y1 - expand_y)
                x2_expanded = min(camera_image.width, x2 + expand_x)
                y2_expanded = min(camera_image.height, y2 + expand_y)

                bboxes_expanded.append((x1_expanded, y1_expanded, x2_expanded, y2_expanded))

            # Batch color extraction (single GPU call for all traffic lights)
            colors = self.traffic_light_classifier.extract_colors_batch(
                camera_image.image_data,
                bboxes_expanded
            )

            # Process each detection with its predicted color
            for i, detection in enumerate(traffic_light_detections):
                x1, y1, x2, y2 = bboxes_expanded[i]
                color = colors[i]

                # Extract track ID from class_label (e.g., "traffic_light_123" -> 123, "traffic_light_new" -> -1)
                bbox = detection.bounding_box
                label_parts = bbox.class_label.split('_')
                if len(label_parts) > 1 and label_parts[-1].isdigit():
                    track_id = int(label_parts[-1])
                else:
                    track_id = -1

                # Apply color hysteresis to reduce flickering
                if color is not None:
                    stable_color = self.color_hysteresis.apply(track_id, color, field_name="color")
                else:
                    stable_color = None

                # Calculate 3D position using LiDAR (if available)
                position_3d = None

                if lidar_data is not None and self.lidar_image_uv is not None and self.lidar_vehicle_frame is not None:
                    # Find LiDAR points within expanded traffic light bounding box
                    in_bbox_x = (self.lidar_image_uv[:, 0] >= x1) & (self.lidar_image_uv[:, 0] <= x2)
                    in_bbox_y = (self.lidar_image_uv[:, 1] >= y1) & (self.lidar_image_uv[:, 1] <= y2)
                    in_bbox = in_bbox_x & in_bbox_y

                    if in_bbox.any():
                        # Get all 3D points in bounding box (LiDAR frame)
                        bbox_points_3d = self.lidar_vehicle_frame[in_bbox]

                        # Find closest point (minimum X in forward direction) in LiDAR frame
                        closest_idx = np.argmin(bbox_points_3d[:, 0])
                        closest_pos_lidar = bbox_points_3d[closest_idx]

                        # Store position in LiDAR frame (consistent with vehicles)
                        position_3d = Position3D(
                            lidar_points=bbox_points_3d,
                            center_3d=closest_pos_lidar
                        )

                # Skip traffic lights with failed classification
                if stable_color is not None:
                    tracked_light = TrackedTrafficLight(
                        track_id=track_id,
                        detection=detection,
                        position_3d=position_3d,
                        color=stable_color
                    )
                    # Compute distance and TTC using polymorphic method
                    tracked_light.update_metrics(self.ego_vehicle, self.sensor_manager)
                    tracked_traffic_lights.append(tracked_light)

            # Find closest traffic light ahead
            if tracked_traffic_lights:
                # Filter traffic lights that are ahead and not irrelevant
                ahead_lights = [
                    light for light in tracked_traffic_lights
                    if light.position_3d is not None
                    and light.position_3d.center_3d[0] > 0
                    and light.color != "irrelevant"
                ]

                if ahead_lights:
                    # Sort by distance and pick closest
                    closest_traffic_light = min(ahead_lights, key=lambda l: l.distance)

        traffic_light_time = (time.perf_counter() - t0) * 1000

        # Log performance breakdown every 30 frames
        if self.frame_count % 30 == 0:
            total_time = (time.perf_counter() - frame_start) * 1000
            lane_status = "DETECTED" if (self.lane_frame_counter - 1) % self.lane_detection_interval == 0 else "REUSED"
            logger.debug(
                f"PERFORMANCE | Frame {self.frame_count} | "
                f"Total: {total_time:.1f}ms | "
                f"Lane({lane_status}): {lane_time:.1f}ms | "
                f"YOLO: {yolo_time:.1f}ms | "
                f"Preprocess: {preprocess_time:.1f}ms | "
                f"Vehicles({len(moving_object_detections)}): {vehicle_processing_time:.1f}ms | "
                f"TrafficLights({len(traffic_light_detections)}): {traffic_light_time:.1f}ms | "
                f"Tracked: {len(current_tracks)}"
            )

        # Cleanup lane hysteresis state for track_ids no longer active (prevent memory leak)
        active_track_ids = {tracked.track_id for tracked in current_tracks if tracked.track_id >= 0}
        self.lane_hysteresis.cleanup(active_track_ids)

        # Cleanup color hysteresis state for traffic light track_ids no longer active (prevent memory leak)
        active_traffic_light_track_ids = set()
        if traffic_light_detections:
            for detection in traffic_light_detections:
                label_parts = detection.bounding_box.class_label.split('_')
                if len(label_parts) > 1 and label_parts[-1].isdigit():
                    active_traffic_light_track_ids.add(int(label_parts[-1]))

        self.color_hysteresis.cleanup(active_traffic_light_track_ids)

        self.frame_count += 1

        return current_tracks, closest_traffic_light

    def _preprocess_sensors_once_per_frame(self, camera_image, lidar_data, radar_data):
        """
        Transform and project ALL sensor data once per frame.

        This preprocessing step transforms all LiDAR and radar points to camera frame
        and projects them to image coordinates. Per-vehicle processing then uses
        boolean indexing to extract relevant points.

        Uses GPU acceleration when available to minimize CPU↔GPU transfers.

        Args:
            camera_image: Camera image for projection
            lidar_data: LiDAR point cloud data
            radar_data: Radar measurement data
        """
        # Build camera intrinsics (uses caching internally)
        self.camera_K = self.velocity_estimator._build_camera_intrinsics(camera_image)

        # Preprocess sensors - tries GPU first, falls back to CPU automatically
        self._preprocess_sensors(camera_image, lidar_data, radar_data)

    def _preprocess_sensors(self, camera_image, lidar_data, radar_data):
        """
        Transform and project sensor data - tries GPU first, falls back to CPU automatically.

        Always populates CPU arrays for compatibility. GPU tensors populated when available.
        """
        try:
            import torch
            if torch.cuda.is_available():
                # GPU path - keeps data on GPU to minimize transfers
                R_lidar_gpu = torch.from_numpy(self.velocity_estimator.R_lidar_to_camera).float().cuda()
                t_lidar_gpu = torch.from_numpy(self.velocity_estimator.t_lidar_to_camera).float().cuda()
                R_radar_gpu = torch.from_numpy(self.velocity_estimator.R_radar_to_camera).float().cuda()
                t_radar_gpu = torch.from_numpy(self.velocity_estimator.t_radar_to_camera).float().cuda()
                K_gpu = torch.from_numpy(self.camera_K).float().cuda()

                # Process LiDAR on GPU
                lidar_gpu = torch.from_numpy(lidar_data.points.copy()).float().cuda()
                lidar_camera_gpu = (R_lidar_gpu @ lidar_gpu.T).T + t_lidar_gpu
                valid_lidar = lidar_camera_gpu[:, 2] > 0.1
                lidar_camera_gpu = lidar_camera_gpu[valid_lidar]
                lidar_vehicle_gpu = lidar_gpu[valid_lidar]

                # Store GPU tensors
                self.lidar_camera_frame_gpu = lidar_camera_gpu
                self.lidar_vehicle_frame_gpu = lidar_vehicle_gpu

                # Project to image on GPU
                projected = K_gpu @ lidar_camera_gpu.T
                depths = projected[2, :]
                self.lidar_image_uv_gpu = torch.stack([
                    projected[0, :] / depths,
                    projected[1, :] / depths
                ], dim=1)

                # Also keep CPU versions for compatibility
                self.lidar_camera_frame = lidar_camera_gpu.cpu().numpy()
                self.lidar_vehicle_frame = lidar_vehicle_gpu.cpu().numpy()
                self.lidar_image_uv = self.lidar_image_uv_gpu.cpu().numpy()

                # Process Radar on GPU
                valid_radar_initial = np.all(np.isfinite(radar_data.points), axis=1)
                radar_spherical_clean = radar_data.points[valid_radar_initial]

                # Convert spherical to Cartesian (tries GPU internally)
                radar_xyz = self.velocity_estimator._radar_spherical_to_cartesian(radar_spherical_clean)
                radar_xyz_gpu = torch.from_numpy(radar_xyz).float().cuda()
                radar_camera_gpu = (R_radar_gpu @ radar_xyz_gpu.T).T + t_radar_gpu

                # Filter points in front of camera
                valid_depth = radar_camera_gpu[:, 2] > 0.1
                radar_camera_gpu = radar_camera_gpu[valid_depth]
                radar_xyz_gpu = radar_xyz_gpu[valid_depth]

                # Store GPU tensors
                self.radar_xyz_all_gpu = radar_xyz_gpu

                # Project to image on GPU
                projected_radar = K_gpu @ radar_camera_gpu.T
                depths_radar = projected_radar[2, :]
                self.radar_image_uv_gpu = torch.stack([
                    projected_radar[0, :] / depths_radar,
                    projected_radar[1, :] / depths_radar
                ], dim=1)

                # Also keep CPU versions
                self.radar_camera_frame = radar_camera_gpu.cpu().numpy()
                self.radar_xyz_all = radar_xyz_gpu.cpu().numpy()
                self.radar_spherical = radar_spherical_clean[valid_depth.cpu().numpy()]
                self.radar_image_uv = self.radar_image_uv_gpu.cpu().numpy()

                logger.debug("Sensor preprocessing using GPU acceleration")
                return

        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"GPU preprocessing unavailable ({type(e).__name__}), using CPU fallback")

        # CPU fallback - original implementation
        lidar_points = lidar_data.points
        self.lidar_camera_frame = (
            self.velocity_estimator.R_lidar_to_camera @ lidar_points.T
        ).T + self.velocity_estimator.t_lidar_to_camera

        # Filter points in front of camera
        valid_lidar = self.lidar_camera_frame[:, 2] > 0.1
        self.lidar_camera_frame = self.lidar_camera_frame[valid_lidar]
        self.lidar_vehicle_frame = lidar_points[valid_lidar]

        # Project ALL LiDAR to image
        projected_lidar = self.camera_K @ self.lidar_camera_frame.T
        depths = projected_lidar[2, :]
        self.lidar_image_uv = np.column_stack([
            projected_lidar[0, :] / depths,
            projected_lidar[1, :] / depths
        ])

        # Set GPU tensors to None (not available)
        self.lidar_camera_frame_gpu = None
        self.lidar_vehicle_frame_gpu = None
        self.lidar_image_uv_gpu = None

        # Filter out invalid radar measurements first
        valid_radar_initial = np.all(np.isfinite(radar_data.points), axis=1)
        radar_spherical_clean = radar_data.points[valid_radar_initial]

        # Convert ALL radar from spherical to Cartesian (tries GPU internally)
        self.radar_xyz_all = self.velocity_estimator._radar_spherical_to_cartesian(radar_spherical_clean)

        # Transform ALL radar to camera frame
        self.radar_camera_frame = (
            self.velocity_estimator.R_radar_to_camera @ self.radar_xyz_all.T
        ).T + self.velocity_estimator.t_radar_to_camera

        # Filter points in front of camera
        valid_depth = self.radar_camera_frame[:, 2] > 0.1
        self.radar_camera_frame = self.radar_camera_frame[valid_depth]
        self.radar_xyz_all = self.radar_xyz_all[valid_depth]
        self.radar_spherical = radar_spherical_clean[valid_depth]

        # Project ALL radar to image
        projected_radar = self.camera_K @ self.radar_camera_frame.T
        depths_radar = projected_radar[2, :]
        self.radar_image_uv = np.column_stack([
            projected_radar[0, :] / depths_radar,
            projected_radar[1, :] / depths_radar
        ])

        # Set GPU tensors to None (not available)
        self.radar_xyz_all_gpu = None
        self.radar_image_uv_gpu = None

    def _calculate_ego_lane(self, camera_image) -> int:
        """
        Calculate ego vehicle's lane using image center X position.

        Args:
            camera_image: Camera image for dimensions

        Returns:
            1-based lane index from leftmost lane
        """
        if not self.current_lanes:
            return 1  # Default to lane 1 if no lanes detected

        ego_x = camera_image.width / 2  # Center X position

        # Find which lane corridor contains ego position
        for i in range(len(self.current_lanes) - 1):
            left_lane = self.current_lanes[i]
            right_lane = self.current_lanes[i + 1]

            # Get X coordinates at a fixed Y position (bottom of image)
            y_check = camera_image.height - 100
            left_x = self.lane_detector.interpolate_lane_x(left_lane, y_check, camera_image.width)
            right_x = self.lane_detector.interpolate_lane_x(right_lane, y_check, camera_image.width)
            if left_x is None or right_x is None:
                continue
            if left_x > right_x:
                left_x, right_x = right_x, left_x
            if left_x <= ego_x <= right_x:
                return i + 1  # 1-based lane index

        # If not found between lanes, return closest lane
        return 1

    def _process_single_vehicle(
        self,
        detection: DetectedObject,
        camera_image,
        lidar_data,
        radar_data,
        ego_velocity_world: np.ndarray,
        ego_yaw: float,
        current_time: float
    ) -> Optional[TrackedVehicle]:
        """
        Process a single detected vehicle: position estimation, lane assignment, velocity estimation.

        This method is thread-safe and can be executed in parallel for multiple vehicles.

        Args:
            detection: YOLO detection with bounding box
            camera_image: Camera image (read-only)
            lidar_data: LiDAR data (read-only)
            radar_data: Radar data (read-only)
            ego_velocity_world: Ego velocity in world frame (read-only)
            ego_yaw: Ego yaw angle (read-only)
            current_time: Frame timestamp

        Returns:
            TrackedVehicle if successful, None if vehicle is filtered out
        """
        # Extract track ID from class label
        track_id = None
        if '_' in detection.bounding_box.class_label:
            parts = detection.bounding_box.class_label.split('_')
            if len(parts) > 1 and parts[1] != 'new':
                track_id = int(parts[1])

        if track_id is None:
            track_id = -self.frame_count  # Use negative frame count as temporary ID

        # Step 1: Estimate 3D position using preprocessed LiDAR data
        position_3d = None
        if self.velocity_estimator and lidar_data:
            position_3d = self.velocity_estimator.estimate_3d_position(
                detection, camera_image, lidar_data,
                lidar_vehicle_frame=self.lidar_vehicle_frame,
                lidar_image_uv=self.lidar_image_uv,
                lidar_vehicle_frame_gpu=self.lidar_vehicle_frame_gpu,
                lidar_image_uv_gpu=self.lidar_image_uv_gpu
            )

        # Step 2: Estimate velocity using preprocessed radar data
        velocity_estimate = None
        if self.velocity_estimator and radar_data:
            velocity_estimate = self.velocity_estimator.estimate_velocity(
                detection, camera_image, radar_data, ego_velocity_world, ego_yaw, track_id=track_id,
                radar_xyz_all=self.radar_xyz_all,
                radar_image_uv=self.radar_image_uv
            )

        # Step 3: Create tracked object based on class type (without lane assignment yet)
        if detection.class_name == "person":
            tracked = TrackedPedestrian(
                track_id=track_id,
                detection=detection,
                absolute_lane_idx=0,  # Will be assigned below
                position_3d=position_3d,
                velocity_estimate=velocity_estimate
            )
        else:
            # Vehicle (car, motorcycle, bus, truck)
            tracked = TrackedVehicle(
                track_id=track_id,
                detection=detection,
                absolute_lane_idx=0,  # Will be assigned below
                position_3d=position_3d,
                velocity_estimate=velocity_estimate
            )

        # Step 4: Lane assignment (DOMAIN LOGIC ON DOMAIN OBJECT)
        vehicle_lane_idx = tracked.assign_lane(self)

        # Step 5: Filter by lane proximity
        if vehicle_lane_idx is None:
            return None

        # Step 6: Apply lane hysteresis (PIPELINE-LEVEL STATE)
        # Pedestrians can legitimately cross lanes quickly, but vehicles benefit from hysteresis
        if detection.class_name == "person":
            tracked.absolute_lane_idx = vehicle_lane_idx  # No hysteresis for pedestrians
        else:
            # Apply hysteresis only for tracked vehicles (track_id >= 0)
            tracked.absolute_lane_idx = self.lane_hysteresis.apply(track_id, vehicle_lane_idx, field_name="lane")

        # Step 7: Compute distance and TTC using polymorphic method
        tracked.update_metrics(self.ego_vehicle, self.sensor_manager)

        return tracked

    def _calculate_vehicle_lane(
        self,
        detection: DetectedObject,
        camera_image,
        position_3d: Optional[Position3D] = None,
        track_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Calculate which lane a detected vehicle is in using 3D LiDAR cluster voting.

        Uses real 3D sensor data across vehicle width with voting for maximum accuracy.
        If strategy fails, returns last known lane for this track.

        Args:
            detection: Vehicle detection with bounding box
            camera_image: Camera image for reference
            position_3d: Optional 3D position with LiDAR points
            track_id: Optional track ID for last known lane fallback

        Returns:
            1-based lane index from leftmost lane, or None if not determinable
        """
        if not self.current_lanes:
            return None

        # Strategy 1: 3D LiDAR centroid - uses pre-calculated cluster center from DBSCAN
        if position_3d and position_3d.center_3d is not None:
            lane_idx = self._calculate_lane_from_centroid(
                position_3d.center_3d, camera_image
            )
            if lane_idx is not None:
                logger.debug(f"[LANE CALC] Strategy 1 (3D centroid) succeeded: lane {lane_idx}")
                return lane_idx
            else:
                logger.debug(f"[LANE CALC] Strategy 1 (3D centroid) failed")

        if not self.high_security_mode:

            # Strategy 2: Use last known lane from previous frame
            if track_id is not None and track_id in self.tracked_vehicles:
                last_lane = self.tracked_vehicles[track_id].absolute_lane_idx
                logger.debug(f"[LANE CALC] Strategy 2 (track history) succeeded: lane {last_lane} for track {track_id}")
                return last_lane

            # Strategy 3: 2D bbox mid-bottom fallback - no LiDAR or track history
            lane_idx = self._calculate_lane_from_bbox_center(detection, camera_image)
            if lane_idx is not None:
                logger.debug(f"[LANE CALC] Strategy 3 (2D bbox center) succeeded: lane {lane_idx}")
                return lane_idx
            else:
                logger.debug(f"[LANE CALC] Strategy 3 (2D bbox center) failed")

            # Strategy 4: 3D LiDAR cluster voting - uses real sensor data across full vehicle width
            if position_3d and position_3d.lidar_points is not None and len(position_3d.lidar_points) >= 5:
                lane_idx, confidence = self._calculate_lane_from_lidar_cluster(
                    position_3d.lidar_points, detection, camera_image,
                    lidar_points_2d=position_3d.lidar_points_2d
                )
                if lane_idx is not None:
                    logger.debug(f"[LANE CALC] Strategy 4 (3D cluster voting) succeeded: lane {lane_idx} (confidence {confidence:.1%})")
                    return lane_idx
                else:
                    logger.debug(f"[LANE CALC] Strategy 4 (3D cluster voting) rejected (confidence {confidence:.1%})")

        logger.debug(f"[LANE CALC] All strategies failed")
        return None

    def _gpu_interp(self, x_query, x_data, y_data):
        """
        GPU-accelerated linear interpolation (like np.interp).

        Args:
            x_query: Query points (N,) tensor on GPU
            x_data: X coordinates of data points (M,) tensor on GPU (must be sorted)
            y_data: Y coordinates of data points (M,) tensor on GPU

        Returns:
            Interpolated values (N,) tensor on GPU, with NaN for out-of-range queries
        """
        import torch

        # Find indices for interpolation using searchsorted
        # For each query point, find where it would fit in x_data
        indices = torch.searchsorted(x_data, x_query, right=False)

        # Clamp indices to valid range [1, len(x_data)-1]
        # This ensures we always have valid left and right points
        indices = torch.clamp(indices, 1, len(x_data) - 1)

        # Get left and right points for interpolation
        x_left = x_data[indices - 1]
        x_right = x_data[indices]
        y_left = y_data[indices - 1]
        y_right = y_data[indices]

        # Linear interpolation: y = y_left + (x - x_left) * (y_right - y_left) / (x_right - x_left)
        weights = (x_query - x_left) / (x_right - x_left + 1e-8)  # Avoid division by zero
        y_interp = y_left + weights * (y_right - y_left)

        # Mark out-of-range values as NaN
        out_of_range = (x_query < x_data[0]) | (x_query > x_data[-1])
        y_interp[out_of_range] = float('nan')

        return y_interp

    def _calculate_lane_from_lidar_cluster(
        self,
        lidar_points: np.ndarray,
        detection: DetectedObject,
        camera_image,
        min_points: int = 5,
        confidence_threshold: float = 0.5,
        lidar_points_2d: np.ndarray = None
    ) -> Tuple[Optional[int], float]:
        """
        Calculate lane using LiDAR cluster voting - uses all cluster points
        and votes on lane assignment.

        Args:
            lidar_points: (N, 3) LiDAR points hitting the vehicle
            detection: Detection with bounding box for filtering
            camera_image: Camera image for projection
            min_points: Minimum points required for voting
            confidence_threshold: Minimum confidence to accept result

        Returns:
            (lane_idx, confidence) where confidence is vote agreement percentage
        """
        if len(lidar_points) < min_points:
            return None, 0.0

        # Use cached 2D projections if available (avoids re-projection)
        if lidar_points_2d is not None:
            all_projected_2d = lidar_points_2d
            logger.debug(f"Using cached 2D projections: {len(all_projected_2d)} points")
        else:
            all_projected_2d = self.velocity_estimator.project_points_to_image(
                lidar_points, camera_image
            )

        if len(all_projected_2d) == 0:
            return None, 0.0

        # Filter by shrunk bbox to focus on vehicle interior
        bbox = detection.bounding_box
        shrink = 0
        x1 = bbox.x + bbox.width * shrink
        y1 = bbox.y + bbox.height * shrink
        x2 = bbox.x + bbox.width * (1 - shrink)
        y2 = bbox.y + bbox.height * (1 - shrink)

        # Vectorized bbox filtering
        all_projected_2d_np = np.array(all_projected_2d)
        inside = (
            (all_projected_2d_np[:, 0] >= x1) & (all_projected_2d_np[:, 0] <= x2) &
            (all_projected_2d_np[:, 1] >= y1) & (all_projected_2d_np[:, 1] <= y2) &
            (all_projected_2d_np[:, 0] >= 0) & (all_projected_2d_np[:, 0] < camera_image.width) &
            (all_projected_2d_np[:, 1] >= 0) & (all_projected_2d_np[:, 1] < camera_image.height)
        )
        valid_points = all_projected_2d_np[inside]

        if len(valid_points) < min_points:
            return None, 0.0

        # GPU-accelerated voting: check which lane each point belongs to
        votes = {}

        # Use GPU if available (check if GPU tensors exist)
        if self.sorted_lanes_gpu and len(self.sorted_lanes_gpu) > 0:
            import torch

            # Move valid points to GPU once
            valid_points_gpu = torch.from_numpy(valid_points).float().cuda()

            # For each lane (space between two boundaries), check which points fall inside
            for i in range(len(self.sorted_lanes_gpu) - 1):
                left_lane_gpu = self.sorted_lanes_gpu[i]
                right_lane_gpu = self.sorted_lanes_gpu[i + 1]

                if len(left_lane_gpu) < 2 or len(right_lane_gpu) < 2:
                    continue

                # GPU interpolation: interpolate left and right boundaries at ALL point Y coordinates
                left_x_at_points = self._gpu_interp(
                    valid_points_gpu[:, 1], left_lane_gpu[:, 1], left_lane_gpu[:, 0]
                )
                right_x_at_points = self._gpu_interp(
                    valid_points_gpu[:, 1], right_lane_gpu[:, 1], right_lane_gpu[:, 0]
                )

                # Skip points outside lane Y range (NaN check on GPU)
                valid_interp = ~(torch.isnan(left_x_at_points) | torch.isnan(right_x_at_points))
                if not torch.any(valid_interp):
                    continue

                # Handle swapped lanes (ensure left < right) - on GPU
                min_x = torch.minimum(left_x_at_points[valid_interp], right_x_at_points[valid_interp])
                max_x = torch.maximum(left_x_at_points[valid_interp], right_x_at_points[valid_interp])

                # GPU: check which points are in this lane
                points_x = valid_points_gpu[valid_interp, 0]
                inside_lane = (points_x >= min_x) & (points_x <= max_x)

                # Transfer only the vote count (tiny!)
                vote_count = inside_lane.sum().item()
                if vote_count > 0:
                    votes[i + 1] = vote_count  # 1-based lane index
        else:
            # CPU fallback (original implementation)
            for i in range(len(self.sorted_lanes) - 1):
                left_lane = self.sorted_lanes[i]
                right_lane = self.sorted_lanes[i + 1]

                if len(left_lane) < 2 or len(right_lane) < 2:
                    continue

                # Convert to arrays for vectorized interpolation
                left_y, left_x = zip(*[(p[1], p[0]) for p in left_lane])
                right_y, right_x = zip(*[(p[1], p[0]) for p in right_lane])

                # Vectorized: interpolate left and right boundaries at ALL point Y coordinates
                left_x_at_points = np.interp(valid_points[:, 1], left_y, left_x, left=np.nan, right=np.nan)
                right_x_at_points = np.interp(valid_points[:, 1], right_y, right_x, left=np.nan, right=np.nan)

                # Skip points outside lane Y range
                valid_interp = ~(np.isnan(left_x_at_points) | np.isnan(right_x_at_points))
                if not np.any(valid_interp):
                    continue

                # Handle swapped lanes (ensure left < right)
                min_x = np.minimum(left_x_at_points[valid_interp], right_x_at_points[valid_interp])
                max_x = np.maximum(left_x_at_points[valid_interp], right_x_at_points[valid_interp])

                # Vectorized: check which points are in this lane
                points_x = valid_points[valid_interp, 0]
                inside_lane = (points_x >= min_x) & (points_x <= max_x)

                vote_count = inside_lane.sum()
                if vote_count > 0:
                    votes[i + 1] = vote_count  # 1-based lane index

        if not votes:
            return None, 0.0

        # Find winner and calculate confidence
        best_lane = max(votes, key=votes.get)
        confidence = votes[best_lane] / len(valid_points)

        if confidence >= confidence_threshold:
            logger.debug(f"[3D CLUSTER] Lane {best_lane} with {confidence:.1%} confidence ({votes[best_lane]}/{len(valid_points)} points)")
            return best_lane, confidence
        else:
            logger.debug(f"[3D CLUSTER] Low confidence {confidence:.1%}, rejecting")
            return None, confidence

    def _find_lane_at_position(self, x_center: float, y_position: float, image_width: int = 640, use_cached_sorted: bool = True) -> Optional[int]:
        """
        Find which lane contains a given (x, y) position in image coordinates.

        Args:
            x_center: X coordinate in image
            y_position: Y coordinate in image
            image_width: Image width for interpolation clipping
            use_cached_sorted: If True, use pre-sorted lanes (faster)

        Returns:
            Lane index from leftmost lane:
            - 0: LEFT overflow (before first boundary)
            - 1 to N-1: Normal lanes between boundaries
            - N: RIGHT overflow (after last boundary)
            - None: Could not determine (interpolation failed)
        """
        lanes = self.sorted_lanes if use_cached_sorted and self.sorted_lanes else self.current_lanes

        if len(lanes) < 2:
            return None

        # Check if vehicle is LEFT of the leftmost boundary (overflow)
        leftmost_x = self.lane_detector.interpolate_lane_x_presorted(lanes[0], y_position, image_width)
        if leftmost_x is not None and x_center < leftmost_x:
            logger.debug(f"Vehicle at X={x_center:.0f} is LEFT of leftmost boundary at X={leftmost_x:.0f}, assigning to lane 0 (overflow)")
            return 0

        # Check if vehicle is RIGHT of the rightmost boundary (overflow)
        rightmost_x = self.lane_detector.interpolate_lane_x_presorted(lanes[-1], y_position, image_width)
        if rightmost_x is not None and x_center > rightmost_x:
            logger.debug(f"Vehicle at X={x_center:.0f} is RIGHT of rightmost boundary at X={rightmost_x:.0f}, assigning to lane {len(lanes)} (overflow)")
            return len(lanes)

        for i in range(len(lanes) - 1):
            left_lane = lanes[i]
            right_lane = lanes[i + 1]

            # Get X coordinates at the specified Y position (using pre-sorted lanes!)
            left_x = self.lane_detector.interpolate_lane_x_presorted(left_lane, y_position, image_width)
            right_x = self.lane_detector.interpolate_lane_x_presorted(right_lane, y_position, image_width)
            if left_x is None or right_x is None:
                logger.debug(f"Interpolation failed for lane pair {i},{i+1} at Y={y_position}")
                continue

            # Handle swapped lanes
            if left_x > right_x:
                logger.debug(f"Lane crossing detected at Y={y_position}: left_x={left_x:.0f}, right_x={right_x:.0f}, swapping")
                left_x, right_x = right_x, left_x

            if left_x <= x_center <= right_x:
                return i + 1  # 1-based lane index from left

        return None

    def _calculate_lane_from_median(
        self,
        lidar_points: np.ndarray,
        detection: DetectedObject,
        camera_image,
        lidar_points_2d: np.ndarray = None
    ) -> Optional[int]:
        """
        Calculate lane using median of LiDAR cluster.

        Computes the 3D median of all LiDAR points hitting the vehicle,
        projects it to the image, and determines which lane it falls in.

        Args:
            lidar_points: (N, 3) LiDAR points hitting the vehicle
            detection: Detection with bounding box for filtering
            camera_image: Camera image for projection

        Returns:
            Lane index or None
        """
        if len(lidar_points) == 0:
            return None

        # Use cached 2D projections if available (avoids re-projection)
        if lidar_points_2d is not None:
            all_projected_2d = lidar_points_2d
            logger.debug(f"Using cached 2D projections: {len(all_projected_2d)} points")
        else:
            all_projected_2d = self.velocity_estimator.project_points_to_image(
                lidar_points, camera_image
            )

        if len(all_projected_2d) == 0:
            return None

        # Filter by bbox
        bbox = detection.bounding_box
        x1 = bbox.x
        y1 = bbox.y
        x2 = bbox.x + bbox.width
        y2 = bbox.y + bbox.height

        # Vectorized bbox filtering
        all_projected_2d_np = np.array(all_projected_2d)
        inside = (
            (all_projected_2d_np[:, 0] >= x1) & (all_projected_2d_np[:, 0] <= x2) &
            (all_projected_2d_np[:, 1] >= y1) & (all_projected_2d_np[:, 1] <= y2) &
            (all_projected_2d_np[:, 0] >= 0) & (all_projected_2d_np[:, 0] < camera_image.width) &
            (all_projected_2d_np[:, 1] >= 0) & (all_projected_2d_np[:, 1] < camera_image.height)
        )

        if not np.any(inside):
            return None

        interior_lidar = lidar_points[inside]

        # Compute median of ALL interior points
        median_point_3d = np.median(interior_lidar, axis=0)

        # Project median point to image
        projected = self.velocity_estimator.project_points_to_image(
            np.array([median_point_3d]), camera_image
        )

        if len(projected) == 0:
            return None

        x_img, y_img = projected[0]
        if not (0 <= x_img < camera_image.width and 0 <= y_img < camera_image.height):
            return None

        # Find lane at median position
        lane_idx = self._find_lane_at_position(x_img, y_img, camera_image.width)

        if lane_idx is not None:
            logger.debug(f"[3D MEDIAN] Lane {lane_idx} from median of {len(interior_lidar)} points")

        return lane_idx

    def _calculate_lane_from_centroid(
        self,
        centroid_3d: np.ndarray,
        camera_image
    ) -> Optional[int]:
        """
        Calculate lane using pre-calculated centroid of LiDAR cluster.

        Uses the centroid (mean) of the DBSCAN cluster, which is already
        calculated in Position3D.center_3d during clustering.

        Args:
            centroid_3d: (3,) Centroid position in vehicle frame
            camera_image: Camera image for projection

        Returns:
            Lane index or None
        """
        # Project centroid to image
        projected_centroid = self.velocity_estimator.project_points_to_image(
            np.array([centroid_3d]), camera_image
        )

        if len(projected_centroid) == 0:
            return None

        x_center, y_pos = projected_centroid[0]

        # Check if projection is within image bounds
        if not (0 <= x_center < camera_image.width and 0 <= y_pos < camera_image.height):
            return None

        lane_idx = self._find_lane_at_position(x_center, y_pos, camera_image.width)

        if lane_idx is not None:
            logger.debug(f"[3D CENTROID] Lane {lane_idx} at ({x_center:.0f}, {y_pos:.0f})")

        return lane_idx

    def _calculate_lane_from_bbox_center(
        self,
        detection: DetectedObject,
        camera_image
    ) -> Optional[int]:
        """
        Calculate lane using 2D bbox center point at mid-bottom Y position.

        Args:
            detection: Detection with bounding box
            camera_image: Camera image for reference

        Returns:
            lane_idx or None
        """
        bbox = detection.bounding_box

        x_center = bbox.x + bbox.width * 0.5
        y_bottom = bbox.y + bbox.height * 0.85

        lane_idx = self._find_lane_at_position(x_center, y_bottom, camera_image.width)

        if lane_idx is not None:
            logger.debug(f"[2D BBOX CENTER] Lane {lane_idx} at ({x_center:.0f}, {y_bottom:.0f})")

        return lane_idx

    def get_detected_lanes(self) -> List[List[Tuple[int, int]]]:
        """
        Get current detected lane boundaries.

        Returns:
            List of lanes, each as list of (x, y) points
        """
        return self.current_lanes

    def get_ego_lane(self) -> int:
        """
        Get ego vehicle's current lane index.

        Returns:
            1-based lane index from leftmost lane
        """
        return self.ego_lane_idx

    def _initialize_ego_mask_image(self, camera_image):
        """
        Initialize binary mask image for efficient ego vehicle filtering.

        Creates a pre-computed mask where ego polygon pixels = 255, others = 0.
        This allows O(1) lookup instead of O(M) polygon test per point.

        Args:
            camera_image: Camera image to get dimensions
        """
        if self.ego_mask_image is None:
            # Scale normalized polygon to actual camera dimensions
            width, height = camera_image.width, camera_image.height
            self.ego_vehicle_mask = np.array([
                (int(x * width), int(y * height))
                for x, y in self.ego_vehicle_mask_normalized
            ], dtype=np.int32)

            logger.info(f"Scaled ego vehicle polygon from normalized coords to {width}x{height} resolution")
            logger.debug(f"Scaled polygon: {self.ego_vehicle_mask}")

            # Create binary mask image
            self.ego_mask_image = np.zeros((camera_image.height, camera_image.width), dtype=np.uint8)
            cv2.fillPoly(self.ego_mask_image, [self.ego_vehicle_mask], 255)
            logger.info(f"Initialized ego mask image: {camera_image.width}x{camera_image.height}")

            # Also create GPU version for full-GPU filtering pipeline
            try:
                import torch
                if torch.cuda.is_available():
                    self.ego_mask_image_gpu = torch.from_numpy(self.ego_mask_image).cuda()
                    logger.info("Initialized ego mask on GPU for accelerated filtering")
            except (ImportError, RuntimeError):
                pass

    def _compute_ego_mask_for_points(
        self,
        sensor_points: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        camera_image,
        sensor_name: str
    ) -> np.ndarray:
        """
        Compute boolean mask indicating which sensor points hit the ego vehicle.

        Args:
            sensor_points: (N, 3) XYZ points in sensor frame
            R: (3, 3) rotation matrix from sensor to camera frame
            t: (3,) translation vector from sensor to camera frame
            camera_image: Camera image for projection
            sensor_name: Name for logging (e.g., "LiDAR", "radar")

        Returns:
            (N,) boolean array - True = ego vehicle point, False = keep
        """
        if len(sensor_points) == 0:
            return np.array([], dtype=bool)

        # GPU path: transform + projection on GPU (if available)
        try:
            import torch

            # Move to GPU
            points_gpu = torch.from_numpy(sensor_points).float().cuda()
            R_gpu = torch.from_numpy(R).float().cuda()
            t_gpu = torch.from_numpy(t).float().cuda()

            # Transform to camera frame on GPU
            points_camera_gpu = (R_gpu @ points_gpu.T).T + t_gpu

            # Filter points in front of camera
            valid_depth_gpu = points_camera_gpu[:, 2] > 0.1
            if not torch.any(valid_depth_gpu):
                return np.zeros(len(sensor_points), dtype=bool)

            valid_indices = torch.where(valid_depth_gpu)[0]
            points_camera_front_gpu = points_camera_gpu[valid_depth_gpu]

            # Project to image on GPU
            K = self.velocity_estimator._build_camera_intrinsics(camera_image)
            K_gpu = torch.from_numpy(K).float().cuda()
            projected_gpu = K_gpu @ points_camera_front_gpu.T
            u_gpu = projected_gpu[0, :] / projected_gpu[2, :]
            v_gpu = projected_gpu[1, :] / projected_gpu[2, :]

            # Clamp to image bounds on GPU
            u_clamped = torch.clamp(u_gpu, 0, camera_image.width - 1).long()
            v_clamped = torch.clamp(v_gpu, 0, camera_image.height - 1).long()

            # Transfer only the integer coordinates back to CPU
            u_int = u_clamped.cpu().numpy()
            v_int = v_clamped.cpu().numpy()
            valid_indices_cpu = valid_indices.cpu().numpy()

            # CPU: O(1) mask lookup
            inside_ego = self.ego_mask_image[v_int, u_int] > 0

            # Create full ego mask
            ego_mask = np.zeros(len(sensor_points), dtype=bool)
            ego_mask[valid_indices_cpu[inside_ego]] = True

            logger.debug(f"Ego filter (GPU): identified {np.sum(ego_mask)} / {len(sensor_points)} {sensor_name} points")

            return ego_mask

        except (ImportError, RuntimeError, AttributeError):
            pass  # Fall through to CPU fallback

        # CPU fallback
        # Transform to camera frame
        points_camera = (R @ sensor_points.T).T + t

        # Filter points in front of camera
        valid_depth = points_camera[:, 2] > 0.1
        if not np.any(valid_depth):
            return np.zeros(len(sensor_points), dtype=bool)

        # Get indices of valid points
        valid_indices = np.where(valid_depth)[0]
        points_camera_front = points_camera[valid_depth]

        # Project to image
        K = self.velocity_estimator._build_camera_intrinsics(camera_image)
        projected = K @ points_camera_front.T
        u = projected[0, :] / projected[2, :]
        v = projected[1, :] / projected[2, :]

        # Clamp to image bounds and convert to integer indices
        u_int = np.clip(u, 0, camera_image.width - 1).astype(np.int32)
        v_int = np.clip(v, 0, camera_image.height - 1).astype(np.int32)

        # Efficient O(1) mask lookup instead of O(M) polygon test
        inside_ego = self.ego_mask_image[v_int, u_int] > 0

        # Create full ego mask
        ego_mask = np.zeros(len(sensor_points), dtype=bool)
        ego_mask[valid_indices[inside_ego]] = True

        logger.debug(f"Ego filter (CPU): identified {np.sum(ego_mask)} / {len(sensor_points)} {sensor_name} points")

        return ego_mask

    def _filter_ego_points(
        self,
        lidar_points: np.ndarray,
        radar_points: np.ndarray,
        camera_image
    ) -> tuple:
        """
        Filter out LiDAR and radar points that hit the ego vehicle.

        Projects both sensor point clouds to image and removes those inside the ego vehicle polygon mask.
        Uses efficient O(1) binary mask lookup instead of O(M) polygon test per point.

        Args:
            lidar_points: (N, 3) LiDAR points in vehicle frame [x, y, z]
            radar_points: (M, 4) radar points [velocity, azimuth, altitude, depth]
            camera_image: Camera image for projection

        Returns:
            filtered_lidar: (N', 3) filtered LiDAR points excluding ego (N' <= N)
            filtered_radar: (M', 4) filtered radar points excluding ego (M' <= M)
        """
        # Initialize binary mask image on first use
        self._initialize_ego_mask_image(camera_image)

        # Filter LiDAR points using shared helper
        lidar_ego_mask = self._compute_ego_mask_for_points(
            lidar_points,
            self.velocity_estimator.R_lidar_to_camera,
            self.velocity_estimator.t_lidar_to_camera,
            camera_image,
            "LiDAR"
        )
        filtered_lidar = lidar_points[~lidar_ego_mask] if len(lidar_ego_mask) > 0 else lidar_points

        # Filter radar points (convert to Cartesian first, then apply mask to original)
        if len(radar_points) == 0:
            filtered_radar = radar_points
        else:
            # Pre-filter invalid points to ensure mask size matches after conversion
            valid_mask = np.all(np.isfinite(radar_points), axis=1)
            radar_valid = radar_points[valid_mask]

            radar_xyz = self.velocity_estimator._radar_spherical_to_cartesian(radar_valid)
            radar_ego_mask = self._compute_ego_mask_for_points(
                radar_xyz,
                self.velocity_estimator.R_radar_to_camera,
                self.velocity_estimator.t_radar_to_camera,
                camera_image,
                "radar"
            )
            filtered_radar = radar_valid[~radar_ego_mask] if len(radar_ego_mask) > 0 else radar_valid

        return filtered_lidar, filtered_radar

    def reset(self):
        """Reset perception state for new scenario."""
        self.detector.reset_tracker()
        self.tracked_vehicles.clear()
        self.lane_hysteresis_state.clear()
        self.color_hysteresis_state.clear()
        self.frame_count = 0
        self.current_lanes = []
        self.sorted_lanes = []
        self.sorted_lanes_gpu = []
        self.ego_lane_idx = 1
        self.lidar_points = None
        self.radar_points = None
        # Keep ego_mask_image - it's camera-specific and doesn't change between scenarios