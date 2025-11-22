"""
CARLA Velocity Estimator - Multi-sensor fusion for vehicle velocity estimation.

Implements sensor fusion using camera (YOLO 2D boxes), LiDAR (3D points),
and radar (Doppler velocity) to estimate absolute vehicle velocities.
"""

import numpy as np
import logging
from typing import Optional, List, Tuple
from scipy.optimize import least_squares
from scipy import stats

# Import sklearn DBSCAN for clustering
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

from lane_change.gateway.velocity_estimator import IVelocityEstimator
from lane_change.gateway.data_types import (
    DetectedObject, CameraImage, LidarData, RadarData, VelocityEstimate, Position3D
)


class VelocityKalmanTracker:
    """Kalman filter for vehicle velocity with constant velocity model (smoothing + prediction)."""

    def __init__(self, process_noise: float = 2.0, measurement_noise: float = 1.0):
        """
        Initialize velocity Kalman tracker with constant velocity model.

        Args:
            process_noise: Process noise (Q) - how much velocity can change per frame
            measurement_noise: Measurement noise (R) - how much to trust velocity measurements
        """
        self.Q = process_noise
        self.R = measurement_noise
        self.tracks = {}  # track_id -> {'x': (3,), 'P': (3,3)}

    def update(self, track_id: int, velocity_world: np.ndarray, has_measurement: bool = True) -> np.ndarray:
        """
        Update and smooth velocity using Kalman filter with constant velocity model.

        State: [vx, vy, vz]
        Prediction: velocity stays constant (v_new = v_old)
        Measurement: [vx, vy, vz]

        Args:
            track_id: Track ID
            velocity_world: (3,) velocity in world frame [vx, vy, vz]
            has_measurement: Whether we have a measurement this frame

        Returns:
            Smoothed velocity in world frame (3,)
        """
        # Initialize new track
        if track_id not in self.tracks:
            x = velocity_world.copy()
            P = np.eye(3) * 1.0
            self.tracks[track_id] = {'x': x, 'P': P}
            return velocity_world  # First frame: return raw measurement

        state = self.tracks[track_id]

        # Predict step: constant velocity model (velocity stays the same)
        x_pred = state['x']  # v_new = v_old
        P_pred = state['P'] + self.Q * np.eye(3)  # Uncertainty grows

        if has_measurement:
            # Kalman gain
            K = P_pred @ np.linalg.inv(P_pred + self.R * np.eye(3))

            # Update state with measurement
            state['x'] = x_pred + K @ (velocity_world - x_pred)
            state['P'] = (np.eye(3) - K) @ P_pred
        else:
            # No measurement - use prediction (velocity stays constant)
            state['x'] = x_pred
            state['P'] = P_pred
            logger.debug(f"[VELOCITY KALMAN] Track {track_id}: No measurement, using prediction")

        self.tracks[track_id] = state

        return state['x']


class CarlaVelocityEstimator(IVelocityEstimator):
    """
    CARLA-specific velocity estimator using camera-LiDAR-radar fusion.

    Pipeline:
    1. Extract LiDAR points that project inside detection bbox
    2. Cluster points using DBSCAN to isolate vehicle
    3. Compute center position (mean of cluster)
    4. Convert radar from spherical to Cartesian
    5. Filter radar points (2D bbox → clustering → static filtering)
    6. Aggregate velocities and compensate for ego motion
    """

    def __init__(
        self,
        camera_fov: float = 90.0,
        camera_location: tuple = (-1.2, 0.0, 2.0),
        camera_pitch: float = -5.0,
        lidar_location: tuple = (-1.0, 0.0, 2.5),
        radar_location: tuple = (-1.0, 0.0, 2.5),
        cluster_eps: float = 0.5,
        cluster_min_samples: int = 3,
        radar_cluster_min_samples: int = 3,
        min_radar_points: int = 3,
        calib_factor: float = 1.03,
        use_kalman: bool = True,
        kalman_dt: float = 0.05,
        kalman_process_noise: float = 0.5,
        kalman_measurement_noise: float = 1.0
    ):
        """
        Initialize velocity estimator with sensor configuration.

        Args:
            camera_fov: Camera field of view in degrees
            camera_location: Camera sensor position (x, y, z) relative to vehicle
            camera_pitch: Camera pitch angle in degrees (negative = looking down)
            lidar_location: LiDAR sensor position (x, y, z) relative to vehicle
            radar_location: Radar sensor position (x, y, z) relative to vehicle
            cluster_eps: DBSCAN epsilon parameter (meters)
            cluster_min_samples: DBSCAN minimum samples per cluster for LiDAR
            radar_cluster_min_samples: DBSCAN minimum samples per cluster for radar
            min_radar_points: Minimum radar points required for estimate
            calib_factor: Multiplicative calibration factor for final velocity
            use_kalman: Whether to use Kalman filtering for temporal smoothing
            kalman_dt: Time step between frames (seconds)
            kalman_process_noise: Kalman process noise (Q)
            kalman_measurement_noise: Kalman measurement noise (R)
        """
        self.camera_fov = camera_fov
        self.camera_location = np.array(camera_location)
        self.camera_pitch = camera_pitch
        self.lidar_location = np.array(lidar_location)
        self.radar_location = np.array(radar_location)
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.radar_cluster_min_samples = radar_cluster_min_samples
        self.min_radar_points = min_radar_points
        self.calib_factor = calib_factor

        # Cache for camera intrinsics matrix
        self.camera_K = None
        self.camera_width = None
        self.camera_height = None

        # Initialize Kalman tracker for velocity smoothing
        self.use_kalman = use_kalman
        if self.use_kalman:
            self.velocity_kalman = VelocityKalmanTracker(
                process_noise=kalman_process_noise,
                measurement_noise=kalman_measurement_noise
            )

        # Precompute sensor to camera transforms
        self.R_lidar_to_camera, self.t_lidar_to_camera = self._build_sensor_to_camera_transform(self.lidar_location)
        self.R_radar_to_camera, self.t_radar_to_camera = self._build_sensor_to_camera_transform(self.radar_location)

    def _build_sensor_to_camera_transform(self, sensor_location: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build transformation from sensor frame to camera frame.

        Sensor and camera are both attached to vehicle but at different positions.

        Args:
            sensor_location: Sensor position (x, y, z) in vehicle frame

        Returns:
            Tuple of (rotation_matrix, translation_vector)
        """
        # Relative translation from sensor to camera (in vehicle frame)
        translation = self.camera_location - sensor_location

        # Camera pitch rotation (around Y-axis)
        # In camera frame: X=right, Y=down, Z=forward (OpenCV convention)
        # But CARLA vehicle frame: X=forward, Y=right, Z=up
        # We need to account for both the camera rotation AND the coord system change

        # Camera attached with pitch=-5 means rotated -5 degrees from vehicle frame
        pitch_rad = np.deg2rad(self.camera_pitch)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)

        # Rotation matrix: pitch around Y-axis (vehicle frame)
        R_pitch = np.array([
            [cos_p,  0, sin_p],
            [0,      1,     0],
            [-sin_p, 0, cos_p]
        ])

        # Additional rotation to convert from vehicle frame (X=fwd,Y=right,Z=up)
        # to camera frame (X=right, Y=down, Z=forward) - OpenCV convention
        R_vehicle_to_camera = np.array([
            [0,  1,  0],  # Camera X (right) = Vehicle Y (right)
            [0,  0, -1],  # Camera Y (down) = -Vehicle Z (up)
            [1,  0,  0]   # Camera Z (forward) = Vehicle X (forward)
        ])

        # Combined rotation
        rotation = R_vehicle_to_camera @ R_pitch
        translation_transformed = R_vehicle_to_camera @ translation

        return rotation, translation_transformed

    def estimate_3d_position(
        self,
        detection: DetectedObject,
        camera_image: CameraImage,
        lidar_data: LidarData,
        lidar_vehicle_frame: np.ndarray = None,
        lidar_image_uv: np.ndarray = None,
        lidar_vehicle_frame_gpu = None,
        lidar_image_uv_gpu = None
    ) -> Optional[Position3D]:
        """
        Estimate 3D position for a detected vehicle using LiDAR.

        Pipeline:
        1. Filter preprocessed LiDAR by 2D bbox (boolean indexing)
        2. Cluster points using DBSCAN to isolate vehicle
        3. Compute center position (mean of cluster)

        Args:
            detection: YOLO 2D detection with bounding box
            camera_image: Camera image for calibration
            lidar_data: LiDAR point cloud (Nx3 in vehicle frame)
            lidar_vehicle_frame: Preprocessed LiDAR in vehicle frame (optional, faster)
            lidar_image_uv: Preprocessed LiDAR projected to image (optional, faster)
            lidar_vehicle_frame_gpu: Preprocessed LiDAR on GPU (optional, fastest)

        Returns:
            Position3D with LiDAR points and center position, or None if insufficient data
        """
        bbox = detection.bounding_box
        logger.debug(f"=== Position Estimation for bbox at ({bbox.x:.0f}, {bbox.y:.0f}) ===")

        inside_bbox = None
        inside_bbox_gpu = None

        # Use preprocessed data if available (FAST PATH)
        if lidar_vehicle_frame is not None and lidar_image_uv is not None:
            logger.debug(f"Using preprocessed LiDAR data: {len(lidar_vehicle_frame)} points")

            # Filter by bbox - use GPU if available (avoids CPU→GPU mask transfer)
            x1, y1 = bbox.x, bbox.y
            x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height

            if lidar_image_uv_gpu is not None:
                # GPU path: do bbox comparison on GPU (no mask transfer!)
                inside_bbox_gpu = (
                    (lidar_image_uv_gpu[:, 0] >= x1) & (lidar_image_uv_gpu[:, 0] <= x2) &
                    (lidar_image_uv_gpu[:, 1] >= y1) & (lidar_image_uv_gpu[:, 1] <= y2)
                )
                inside_bbox = inside_bbox_gpu.cpu().numpy()
                lidar_in_bbox = lidar_vehicle_frame[inside_bbox]
            else:
                # CPU path: do bbox comparison on CPU
                inside_bbox = (
                    (lidar_image_uv[:, 0] >= x1) & (lidar_image_uv[:, 0] <= x2) &
                    (lidar_image_uv[:, 1] >= y1) & (lidar_image_uv[:, 1] <= y2)
                )
                lidar_in_bbox = lidar_vehicle_frame[inside_bbox]
        else:
            # Fallback to original processing (SLOW PATH)
            logger.debug(f"Total LiDAR points: {len(lidar_data.points)}")
            K = self._build_camera_intrinsics(camera_image)
            lidar_in_bbox = self._extract_bbox_lidar_points(
                lidar_data.points, detection.bounding_box, K, camera_image
            )

        logger.debug(f"LiDAR points in bbox: {len(lidar_in_bbox)}")

        if len(lidar_in_bbox) < self.cluster_min_samples:
            logger.debug(f"[NO POSITION] Insufficient LiDAR in bbox: {len(lidar_in_bbox)} < {self.cluster_min_samples}")
            return None

        # Get 2D projections for bbox-filtered points
        if lidar_vehicle_frame is not None and lidar_image_uv is not None:
            lidar_in_bbox_2d = lidar_image_uv[inside_bbox]
        else:
            lidar_in_bbox_2d = None

        # Cluster to isolate vehicle (use GPU tensor if available, return mask + downsampled 2D)
        if lidar_vehicle_frame_gpu is not None and inside_bbox_gpu is not None:
            # GPU path: use GPU boolean mask (no CPU→GPU transfer!)
            lidar_in_bbox_gpu = lidar_vehicle_frame_gpu[inside_bbox_gpu]
            result = self._cluster_lidar_points(
                lidar_in_bbox,
                points_gpu=lidar_in_bbox_gpu,
                points_2d=lidar_in_bbox_2d,
                return_mask=True
            )
            if result[0] is None:
                logger.debug(f"[NO POSITION] Clustering failed")
                return None
            vehicle_points, cluster_mask, vehicle_points_2d = result
        else:
            result = self._cluster_lidar_points(
                lidar_in_bbox,
                points_2d=lidar_in_bbox_2d,
                return_mask=True
            )
            if result[0] is None:
                logger.debug(f"[NO POSITION] Clustering failed")
                return None
            vehicle_points, cluster_mask, vehicle_points_2d = result

        if vehicle_points is None or len(vehicle_points) < 1:
            logger.debug(f"[NO POSITION] Clustering failed or no points: {len(vehicle_points) if vehicle_points is not None else 0}")
            return None

        logger.debug(f"Clustered vehicle points: {len(vehicle_points)}")

        # Use downsampled 2D projections if available, otherwise apply mask to original
        if vehicle_points_2d is None and lidar_in_bbox_2d is not None and cluster_mask is not None:
            vehicle_points_2d = lidar_in_bbox_2d[cluster_mask]
            logger.debug(f"Applied mask to {len(vehicle_points_2d)} 2D projections for lane calculation")
        elif vehicle_points_2d is not None:
            logger.debug(f"Using {len(vehicle_points_2d)} downsampled 2D projections for lane calculation")

        # Compute center position (mean of cluster points)
        center_3d = np.mean(vehicle_points, axis=0)
        logger.debug(f"Center 3D: {center_3d}")

        return Position3D(
            lidar_points=vehicle_points,
            center_3d=center_3d,
            lidar_points_2d=vehicle_points_2d
        )

    def estimate_velocity(
        self,
        detection: DetectedObject,
        camera_image: CameraImage,
        radar_data: RadarData,
        ego_velocity_world: np.ndarray,
        ego_rotation: float = 0.0,
        track_id: Optional[int] = None,
        radar_xyz_all: np.ndarray = None,
        radar_image_uv: np.ndarray = None
    ) -> Optional[VelocityEstimate]:
        """
        Estimate 3D velocity for a detected vehicle using radar fusion with Kalman smoothing.

        Args:
            detection: YOLO detection with 2D bounding box
            camera_image: Camera image for radar projection
            radar_data: Radar measurements (Nx4 [vel, az, alt, depth])
            ego_velocity_world: Ego velocity vector (3,) in m/s in WORLD frame
            ego_rotation: Ego vehicle yaw rotation in radians
            track_id: Optional track ID for Kalman temporal smoothing
            radar_xyz_all: Preprocessed radar XYZ (optional, faster)
            radar_image_uv: Preprocessed radar projections (optional, faster)

        Returns:
            VelocityEstimate with smoothed velocities, or None if insufficient data
        """
        logger.debug(f"=== Velocity Estimation Track {track_id} ===")
        logger.debug(f"Total radar points BEFORE filtering: {len(radar_data.points)}")

        # Use preprocessed radar data if available (FAST PATH)
        if radar_xyz_all is not None:
            radar_xyz = radar_xyz_all
            logger.debug(f"Using preprocessed radar XYZ: {len(radar_xyz)} points")
        else:
            # Fallback: convert radar to Cartesian (SLOW PATH)
            valid_radar_mask = np.all(np.isfinite(radar_data.points), axis=1)
            radar_data.points = radar_data.points[valid_radar_mask]
            radar_xyz = self._radar_spherical_to_cartesian(radar_data.points)

        if len(radar_data.points) > 0:
            logger.debug(f"Total radar points: {len(radar_data.points)}")
            logger.debug(f"Radar XYZ range: X[{radar_xyz[:, 0].min():.1f}, {radar_xyz[:, 0].max():.1f}], "
                         f"Y[{radar_xyz[:, 1].min():.1f}, {radar_xyz[:, 1].max():.1f}], "
                         f"Z[{radar_xyz[:, 2].min():.1f}, {radar_xyz[:, 2].max():.1f}]")

        # Step 6: Filter radar points using 2D bbox, clustering, and static filtering
        radar_filtered, radar_xyz_filtered = self._filter_radar(
            radar_data.points, radar_xyz,
            ego_velocity_world, ego_rotation, detection, camera_image,
            radar_image_uv=radar_image_uv
        )

        logger.debug(f"Radar points after filtering: {len(radar_filtered)}")

        # Check if we have sufficient radar data for measurement
        has_radar_measurement = len(radar_filtered) >= self.min_radar_points

        if not has_radar_measurement:
            logger.debug(f"[NO RADAR] Insufficient radar: {len(radar_filtered)} < {self.min_radar_points}")
            # Try Kalman prediction if we have track history
            if self.use_kalman and track_id is not None and track_id in self.velocity_kalman.tracks:
                # Use Kalman prediction (no measurement)
                predicted_velocity_world = self.velocity_kalman.update(
                    track_id, np.zeros(3), has_measurement=False
                )
                velocity_relative_world = predicted_velocity_world - ego_velocity_world
                speed_mps = np.linalg.norm(predicted_velocity_world)
                speed_kmh = speed_mps * 3.6

                logger.debug(f"[VELOCITY KALMAN] Track {track_id}: Using prediction (no radar)")

                return VelocityEstimate(
                    velocity_vector=predicted_velocity_world,
                    velocity_relative=velocity_relative_world,
                    speed_kmh=speed_kmh,
                    confidence=0.5,  # Lower confidence for predictions
                    num_radar_points=0,
                    radar_xyz=None
                )
            return None

        # Step 7: Solve for 3D relative velocity using robust median approach
        velocity_relative_vehicle = self._solve_3d_velocity(radar_filtered)

        if velocity_relative_vehicle is None:
            logger.debug("[NO RADAR] Failed to solve 3D velocity")
            # Try Kalman prediction if we have track history
            if self.use_kalman and track_id is not None and track_id in self.velocity_kalman.tracks:
                # Use Kalman prediction (no measurement)
                predicted_velocity_world = self.velocity_kalman.update(
                    track_id, np.zeros(3), has_measurement=False
                )
                velocity_relative_world = predicted_velocity_world - ego_velocity_world
                speed_mps = np.linalg.norm(predicted_velocity_world)
                speed_kmh = speed_mps * 3.6

                logger.debug(f"[VELOCITY KALMAN] Track {track_id}: Using prediction (solve failed)")

                return VelocityEstimate(
                    velocity_vector=predicted_velocity_world,
                    velocity_relative=velocity_relative_world,
                    speed_kmh=speed_kmh,
                    confidence=0.5,  # Lower confidence for predictions
                    num_radar_points=0,
                    radar_xyz=None
                )
            return None

        # Step 8: Transform ego velocity from world to vehicle frame
        ego_velocity_vehicle = self._transform_world_to_vehicle(ego_velocity_world, ego_rotation)

        # DIAGNOSTIC: Log velocity transform pipeline
        logger.debug(f"[VELOCITY TRANSFORM] ego_rotation={np.rad2deg(ego_rotation):.1f}°, "
                     f"ego_vel_world={np.linalg.norm(ego_velocity_world)*3.6:.1f}km/h, "
                     f"ego_vel_vehicle={np.linalg.norm(ego_velocity_vehicle)*3.6:.1f}km/h, "
                     f"relative_vehicle={np.linalg.norm(velocity_relative_vehicle)*3.6:.1f}km/h")

        # Step 9: Compute absolute velocity in vehicle frame, then world frame
        velocity_absolute_vehicle = velocity_relative_vehicle + ego_velocity_vehicle
        velocity_absolute_world = self._transform_vehicle_to_world(velocity_absolute_vehicle, ego_rotation)

        logger.debug(f"[VELOCITY TRANSFORM] absolute_vehicle={np.linalg.norm(velocity_absolute_vehicle)*3.6:.1f}km/h, "
                     f"absolute_world={np.linalg.norm(velocity_absolute_world)*3.6:.1f}km/h")

        # Step 10: Apply Kalman smoothing if enabled and track_id provided
        if self.use_kalman and track_id is not None:
            velocity_absolute_world = self.velocity_kalman.update(
                track_id, velocity_absolute_world, has_measurement=True
            )
            logger.debug(f"[VELOCITY KALMAN] Track {track_id}: Applied smoothing")

        # Apply calibration factor to final absolute velocity
        velocity_absolute_world = velocity_absolute_world * self.calib_factor

        # Compute relative velocity from smoothed absolute velocity
        velocity_relative_world = velocity_absolute_world - ego_velocity_world

        speed_mps = np.linalg.norm(velocity_absolute_world)
        speed_kmh = speed_mps * 3.6

        confidence = min(1.0, len(radar_filtered) / 10.0)

        logger.debug(f"[FINAL ESTIMATE] Speed: {speed_kmh:.1f} km/h (absolute), "
                    f"Relative: {np.linalg.norm(velocity_relative_world)*3.6:.1f} km/h, "
                    f"Confidence: {confidence:.2f}, Radar points used: {len(radar_filtered)}")

        return VelocityEstimate(
            velocity_vector=velocity_absolute_world,
            velocity_relative=velocity_relative_world,
            speed_kmh=speed_kmh,
            confidence=confidence,
            num_radar_points=len(radar_filtered),
            radar_xyz=radar_xyz_filtered
        )

    def _build_camera_intrinsics(self, camera_image: CameraImage) -> np.ndarray:
        """
        Build camera intrinsic matrix K from image dimensions and FOV.
        Uses caching to avoid rebuilding for same camera resolution.

        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        width = camera_image.width
        height = camera_image.height

        # Return cached K if camera resolution hasn't changed
        if (self.camera_K is not None and
            self.camera_width == width and
            self.camera_height == height):
            return self.camera_K

        # Build new K matrix
        fov_rad = np.deg2rad(self.camera_fov)

        # Focal length from FOV
        fx = width / (2.0 * np.tan(fov_rad / 2.0))
        fy = fx

        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0

        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])

        # Cache for future use
        self.camera_K = K
        self.camera_width = width
        self.camera_height = height

        return K

    def _extract_bbox_lidar_points(
        self,
        lidar_points: np.ndarray,
        bbox: 'BoundingBox',
        K: np.ndarray,
        camera_image: CameraImage
    ) -> np.ndarray:
        """
        Extract LiDAR points that project inside a 2D bounding box.

        Args:
            lidar_points: (N, 3) XYZ in LiDAR frame
            bbox: 2D bounding box in image
            K: Camera intrinsic matrix
            camera_image: Camera image for bounds checking

        Returns:
            (M, 3) LiDAR points (in LiDAR frame) that project inside bbox
        """
        if len(lidar_points) == 0:
            return np.array([])

        # Transform LiDAR points to camera frame
        # p_camera = R @ p_lidar + t
        points_camera = (self.R_lidar_to_camera @ lidar_points.T).T + self.t_lidar_to_camera

        # Filter points in front of camera (positive Z in camera frame)
        valid_depth = points_camera[:, 2] > 0.1
        points_camera_front = points_camera[valid_depth]
        points_lidar_front = lidar_points[valid_depth]

        if len(points_camera_front) == 0:
            return np.array([])

        # Project to image: [u, v, w] = K @ [X, Y, Z]
        points_homogeneous = points_camera_front.T  # (3, M)
        projected = K @ points_homogeneous  # (3, M)

        depths = projected[2, :]
        u = projected[0, :] / depths
        v = projected[1, :] / depths

        shrink = 0
        x1 = bbox.x + bbox.width * shrink
        y1 = bbox.y + bbox.height * shrink
        x2 = bbox.x + bbox.width * (1 - shrink)
        y2 = bbox.y + bbox.height * (1 - shrink)

        inside_bbox = (
            (u >= x1) & (u <= x2) &
            (v >= y1) & (v <= y2) &
            (u >= 0) & (u < camera_image.width) &
            (v >= 0) & (v < camera_image.height)
        )

        # Return points in LiDAR frame (for consistency with rest of pipeline)
        return points_lidar_front[inside_bbox]

    def _voxel_downsample_gpu(self, points_gpu, voxel_size=0.15):
        """
        Downsample point cloud using voxel grid filter - FULLY VECTORIZED GPU.

        Uses scatter_add for parallel voxel centroid computation. No Python loops!

        Args:
            points_gpu: (N, 3) points on GPU (torch tensor)
            voxel_size: Size of voxel cube in meters

        Returns:
            (M, 3) downsampled points on GPU, M << N
        """
        import torch

        # Compute voxel indices for each point (VECTORIZED)
        voxel_indices = torch.floor(points_gpu / voxel_size).long()

        # Create unique voxel keys using hash (VECTORIZED)
        # Hash: x + y*1M + z*1B (assumes coords in reasonable range)
        voxel_keys = (voxel_indices[:, 0] +
                      voxel_indices[:, 1] * 1000000 +
                      voxel_indices[:, 2] * 1000000000)

        # Get unique voxels and inverse mapping
        unique_keys, inverse_indices = torch.unique(voxel_keys, return_inverse=True)
        num_voxels = len(unique_keys)

        # VECTORIZED: Sum all points per voxel using scatter_add (PARALLEL!)
        voxel_sums = torch.zeros((num_voxels, 3), device=points_gpu.device, dtype=points_gpu.dtype)
        voxel_sums.scatter_add_(
            0,
            inverse_indices.unsqueeze(1).expand(-1, 3),  # Broadcast index to 3D
            points_gpu
        )

        # VECTORIZED: Count points per voxel (PARALLEL!)
        voxel_counts = torch.zeros(num_voxels, device=points_gpu.device, dtype=points_gpu.dtype)
        voxel_counts.scatter_add_(
            0,
            inverse_indices,
            torch.ones(len(points_gpu), device=points_gpu.device, dtype=points_gpu.dtype)
        )

        # VECTORIZED: Compute centroids (parallel element-wise division)
        downsampled = voxel_sums / voxel_counts.unsqueeze(1)

        logger.debug(f"[VOXEL GPU] Downsampled {len(points_gpu)} → {num_voxels} points (voxel={voxel_size}m)")

        return downsampled

    def _voxel_downsample_joint(self, points_3d_gpu, points_2d_gpu, voxel_size=0.15):
        """
        Downsample 3D points AND their 2D projections together using same voxel grid.

        Uses scatter_add for parallel computation. Both outputs have matching dimensions.

        Args:
            points_3d_gpu: (N, 3) 3D points on GPU
            points_2d_gpu: (N, 2) 2D projections on GPU
            voxel_size: Size of voxel cube in meters

        Returns:
            (downsampled_3d, downsampled_2d) - both (M, ...) on GPU
        """
        import torch

        # Compute voxel indices based on 3D points (VECTORIZED)
        voxel_indices = torch.floor(points_3d_gpu / voxel_size).long()
        voxel_keys = (voxel_indices[:, 0] +
                      voxel_indices[:, 1] * 1000000 +
                      voxel_indices[:, 2] * 1000000000)

        unique_keys, inverse_indices = torch.unique(voxel_keys, return_inverse=True)
        num_voxels = len(unique_keys)

        # PARALLEL: Downsample 3D points
        voxel_sums_3d = torch.zeros((num_voxels, 3), device=points_3d_gpu.device, dtype=points_3d_gpu.dtype)
        voxel_sums_3d.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), points_3d_gpu)

        # PARALLEL: Downsample 2D projections using SAME voxel grouping
        voxel_sums_2d = torch.zeros((num_voxels, 2), device=points_2d_gpu.device, dtype=points_2d_gpu.dtype)
        voxel_sums_2d.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 2), points_2d_gpu)

        # PARALLEL: Count points per voxel (same for both)
        voxel_counts = torch.zeros(num_voxels, device=points_3d_gpu.device, dtype=points_3d_gpu.dtype)
        voxel_counts.scatter_add_(
            0,
            inverse_indices,
            torch.ones(len(points_3d_gpu), device=points_3d_gpu.device, dtype=points_3d_gpu.dtype)
        )

        # PARALLEL: Compute centroids for both
        downsampled_3d = voxel_sums_3d / voxel_counts.unsqueeze(1)
        downsampled_2d = voxel_sums_2d / voxel_counts.unsqueeze(1)

        logger.debug(f"[VOXEL JOINT GPU] Downsampled {len(points_3d_gpu)} → {num_voxels} points (3D+2D)")

        return downsampled_3d, downsampled_2d

    def _cluster_lidar_points(self, points, points_gpu=None, points_2d=None, return_mask: bool = False):
        """
        Use DBSCAN to cluster LiDAR points and return closest cluster (vehicle).

        Applies GPU voxel downsampling for >500 points to speed up DBSCAN.

        Args:
            points: (N, 3) LiDAR XYZ points in vehicle frame
            points_gpu: Optional GPU tensor (avoids transfer if already on GPU)
            points_2d: Optional (N, 2) 2D projections for joint downsampling
            return_mask: If True, return (points, mask, downsampled_2d) tuple

        Returns:
            (M, 3) points of closest cluster, or (points, mask, downsampled_2d) if return_mask=True, or None
        """
        # Handle both numpy and torch inputs
        points_np = points if isinstance(points, np.ndarray) else points.cpu().numpy()

        if len(points_np) < self.cluster_min_samples:
            return (None, None, None) if return_mask else None

        # Smart reduction: Downsample if too many points (DBSCAN is O(n²))
        downsample_threshold = 200
        downsampled_2d = None

        if len(points_np) > downsample_threshold:
            try:
                import torch
                if torch.cuda.is_available():
                    if points_2d is not None:
                        # GPU path: joint voxel downsample for 3D+2D (fully vectorized)
                        if points_gpu is None:
                            points_gpu = torch.from_numpy(points_np).float().cuda()
                        points_2d_gpu = torch.from_numpy(points_2d).float().cuda()

                        points_gpu_down, points_2d_gpu_down = self._voxel_downsample_joint(
                            points_gpu, points_2d_gpu, voxel_size=0.15
                        )
                        points_np = points_gpu_down.cpu().numpy()
                        downsampled_2d = points_2d_gpu_down.cpu().numpy()
                        logger.debug(f"[LIDAR DOWNSAMPLE] {len(points)} → {len(points_np)} points before clustering")
                    else:
                        # GPU path: 3D only downsampling (no 2D projections available)
                        if points_gpu is None:
                            points_gpu = torch.from_numpy(points_np).float().cuda()
                        points_gpu_downsampled = self._voxel_downsample_gpu(points_gpu, voxel_size=0.15)
                        points_np = points_gpu_downsampled.cpu().numpy()
                        logger.debug(f"[LIDAR DOWNSAMPLE] {len(points)} → {len(points_np)} points (3D only)")
            except (ImportError, RuntimeError, AttributeError):
                # CPU fallback: skip downsampling for now
                logger.debug(f"[LIDAR] {len(points_np)} points (no GPU for downsampling)")

        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(points_np)
        labels = clustering.labels_

        # Find closest cluster (ignore noise label -1)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if len(unique_labels) == 0:
            return (None, None) if return_mask else None

        # GPU-accelerated cluster distance calculation (if available)
        try:
            import torch
            if torch.cuda.is_available():
                # Compute all distances on GPU in parallel
                points_gpu = torch.from_numpy(points_np).float().cuda()
                distances_gpu = torch.norm(points_gpu, dim=1)  # Parallel norm calculation
                distances = distances_gpu.cpu().numpy()

                # Compute mean distance per cluster (CPU - fast for small number of clusters)
                cluster_means = {}
                for label in unique_labels:
                    cluster_mask = labels == label
                    cluster_means[label] = distances[cluster_mask].mean()

                closest_cluster_label = min(cluster_means, key=cluster_means.get)
                closest_mean_distance = cluster_means[closest_cluster_label]

                logger.debug(f"[CLUSTERING GPU] Selected closest cluster (label {closest_cluster_label}) with mean distance: {closest_mean_distance:.2f}m")

                mask = labels == closest_cluster_label
                vehicle_points = points_np[mask]

                if return_mask:
                    # Apply mask to downsampled 2D projections if they exist
                    vehicle_points_2d = downsampled_2d[mask] if downsampled_2d is not None else None
                    return (vehicle_points, mask, vehicle_points_2d)
                else:
                    return vehicle_points

        except (ImportError, RuntimeError, AttributeError):
            pass  # Fall through to CPU fallback

        # CPU fallback
        def cluster_distance(label):
            cluster_points = points_np[labels == label]
            distances = np.linalg.norm(cluster_points, axis=1)
            return np.mean(distances)

        closest_cluster_label = min(unique_labels, key=cluster_distance)

        logger.debug(f"[CLUSTERING] Selected closest cluster (label {closest_cluster_label}) with {len(points_np[labels == closest_cluster_label])} points, "
                     f"mean distance: {cluster_distance(closest_cluster_label):.2f}m")

        mask = labels == closest_cluster_label
        vehicle_points = points_np[mask]

        if return_mask:
            # Apply mask to downsampled 2D projections if they exist
            vehicle_points_2d = downsampled_2d[mask] if downsampled_2d is not None else None
            return (vehicle_points, mask, vehicle_points_2d)
        else:
            return vehicle_points

    def _cluster_radar_points(self, points: np.ndarray, xyz: np.ndarray) -> tuple:
        """
        Use DBSCAN to cluster radar points and return closest cluster.

        Applies GPU voxel downsampling for >150 points to speed up DBSCAN.

        Args:
            points: (N, 4) radar measurements [velocity, azimuth, altitude, depth]
            xyz: (N, 3) XYZ points in vehicle frame (for clustering)

        Returns:
            Tuple of (clustered_radar_points, clustered_xyz) or (all_points, all_xyz) if clustering fails
        """
        if len(xyz) < self.radar_cluster_min_samples:
            logger.debug(f"[RADAR CLUSTERING] Insufficient points for clustering: {len(xyz)} < {self.radar_cluster_min_samples}, using all points")
            return points, xyz

        # Smart reduction: Downsample if too many points (DBSCAN is O(n²))
        downsample_threshold = 200
        original_xyz = xyz
        original_points = points

        if len(xyz) > downsample_threshold:
            try:
                import torch
                if torch.cuda.is_available():
                    # GPU path: voxel downsample on GPU (fully vectorized)
                    xyz_gpu = torch.from_numpy(xyz).float().cuda()
                    xyz_gpu_downsampled = self._voxel_downsample_gpu(xyz_gpu, voxel_size=0.2)  # Larger voxel for radar
                    xyz = xyz_gpu_downsampled.cpu().numpy()

                    # Find nearest original points to downsampled centroids (to preserve radar measurements)
                    # For each downsampled point, find closest original point
                    from scipy.spatial import cKDTree
                    tree = cKDTree(original_xyz)
                    distances, indices = tree.query(xyz)
                    points = original_points[indices]

                    logger.debug(f"[RADAR DOWNSAMPLE] {len(original_xyz)} → {len(xyz)} points before clustering")
            except (ImportError, RuntimeError, AttributeError):
                # CPU fallback: skip downsampling for now
                logger.debug(f"[RADAR] {len(xyz)} points (no GPU for downsampling)")

        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.radar_cluster_min_samples).fit(xyz)
        labels = clustering.labels_

        # Find closest cluster (ignore noise label -1)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if len(unique_labels) == 0:
            logger.debug(f"[RADAR CLUSTERING] No clusters found, using all {len(points)} points")
            return points, xyz

        # GPU-accelerated cluster distance calculation (if available)
        try:
            import torch
            if torch.cuda.is_available():
                # Compute all distances on GPU in parallel
                xyz_gpu = torch.from_numpy(xyz).float().cuda()
                distances_gpu = torch.norm(xyz_gpu, dim=1)  # Parallel norm calculation
                distances = distances_gpu.cpu().numpy()

                # Compute mean distance per cluster (CPU - fast for small number of clusters)
                cluster_means = {}
                for label in unique_labels:
                    cluster_mask = labels == label
                    cluster_means[label] = distances[cluster_mask].mean()

                closest_cluster_label = min(cluster_means, key=cluster_means.get)
                closest_mean_distance = cluster_means[closest_cluster_label]

                logger.debug(f"[RADAR CLUSTERING GPU] Selected closest cluster (label {closest_cluster_label}) with mean distance: {closest_mean_distance:.2f}m")

                cluster_mask = labels == closest_cluster_label
                clustered_points = points[cluster_mask]
                clustered_xyz = xyz[cluster_mask]

                return clustered_points, clustered_xyz

        except (ImportError, RuntimeError, AttributeError):
            pass  # Fall through to CPU fallback

        # CPU fallback
        def cluster_distance(label):
            cluster_xyz = xyz[labels == label]
            distances = np.linalg.norm(cluster_xyz, axis=1)
            return np.mean(distances)

        closest_cluster_label = min(unique_labels, key=cluster_distance)

        logger.debug(f"[RADAR CLUSTERING] Selected closest cluster (label {closest_cluster_label}) with {len(xyz[labels == closest_cluster_label])}/{len(points)} points, "
                     f"mean distance: {cluster_distance(closest_cluster_label):.2f}m")

        cluster_mask = labels == closest_cluster_label
        clustered_points = points[cluster_mask]
        clustered_xyz = xyz[cluster_mask]

        return clustered_points, clustered_xyz

    def _radar_spherical_to_cartesian(self, radar_points: np.ndarray) -> np.ndarray:
        """
        Convert CARLA radar from spherical [velocity, azimuth, altitude, depth]
        to Cartesian XYZ positions.

        Tries GPU acceleration first, falls back to CPU automatically.

        Args:
            radar_points: (N, 4) [velocity, azimuth, altitude, depth]

        Returns:
            (N, 3) XYZ positions in vehicle frame
        """
        if len(radar_points) == 0:
            return np.array([]).reshape(0, 3)

        # Filter out invalid radar measurements (NaN or inf values)
        valid_mask = np.all(np.isfinite(radar_points), axis=1)
        radar_points = radar_points[valid_mask]

        if len(radar_points) == 0:
            return np.array([]).reshape(0, 3)

        # Try GPU path first: parallel trigonometric operations
        try:
            import torch
            if torch.cuda.is_available():
                # Move to GPU
                radar_gpu = torch.from_numpy(radar_points).float().cuda()

                velocity = radar_gpu[:, 0]  # Not used for position
                azimuth = radar_gpu[:, 1]   # Radians
                altitude = radar_gpu[:, 2]  # Radians
                depth = radar_gpu[:, 3]     # Meters

                # Spherical to Cartesian on GPU (parallel cos/sin)
                # CARLA: X=forward, Y=right, Z=up
                x = depth * torch.cos(altitude) * torch.cos(azimuth)
                y = depth * torch.cos(altitude) * torch.sin(azimuth)
                z = depth * torch.sin(altitude)

                xyz = torch.stack([x, y, z], dim=1)

                logger.debug(f"[RADAR GPU] Converted {len(radar_points)} points from spherical to Cartesian")

                return xyz.cpu().numpy()

        except (ImportError, RuntimeError, AttributeError):
            pass  # Fall through to CPU fallback

        # CPU fallback
        velocity = radar_points[:, 0]  # Not used for position
        azimuth = radar_points[:, 1]   # Radians
        altitude = radar_points[:, 2]  # Radians
        depth = radar_points[:, 3]     # Meters

        # Spherical to Cartesian
        # CARLA: X=forward, Y=right, Z=up
        x = depth * np.cos(altitude) * np.cos(azimuth)
        y = depth * np.cos(altitude) * np.sin(azimuth)
        z = depth * np.sin(altitude)

        xyz = np.stack([x, y, z], axis=1)

        return xyz

    def _filter_radar(
        self,
        radar_points: np.ndarray,
        radar_xyz: np.ndarray,
        ego_velocity_world: np.ndarray,
        ego_rotation: float,
        detection: DetectedObject,
        camera_image: CameraImage,
        radar_image_uv: np.ndarray = None
    ) -> tuple:
        """
        Filter radar points using 2D bbox projection, clustering, and static object removal.

        Three-stage filtering:
        1. 2D bbox: Keep points that project inside YOLO 2D bounding box
        2. Clustering: Use DBSCAN to isolate closest cluster (vehicle)
        3. Static: Remove ground/barrier returns using Doppler analysis

        Args:
            radar_points: (N, 4) [velocity, azimuth, altitude, depth]
            radar_xyz: (N, 3) Cartesian positions
            ego_velocity_world: (3,) ego velocity in world frame
            ego_rotation: ego yaw rotation in radians
            detection: YOLO detection with 2D bounding box
            camera_image: Camera image for radar projection

        Returns:
            filtered_points: (M, 4) radar points inside 2D bbox, clustered, and moving
            radar_xyz_filtered: (M, 3) XYZ positions of filtered radar points (for visualization)
        """

        # Stage 1: 2D bounding box filtering
        bbox = detection.bounding_box
        x1 = bbox.x
        y1 = bbox.y
        x2 = bbox.x + bbox.width
        y2 = bbox.y + bbox.height

        # Use preprocessed projections if available (FAST PATH)
        if radar_image_uv is not None:
            logger.debug(f"Using preprocessed radar projections: {len(radar_image_uv)} points")

            # Filter by bbox using boolean indexing on preprocessed projections
            inside_2d_bbox = (
                (radar_image_uv[:, 0] >= x1) & (radar_image_uv[:, 0] <= x2) &
                (radar_image_uv[:, 1] >= y1) & (radar_image_uv[:, 1] <= y2)
            )

            filtered_points = radar_points[inside_2d_bbox]
            radar_xyz_filtered = radar_xyz[inside_2d_bbox]
        else:
            # Fallback: transform and project (SLOW PATH)
            radar_camera = (self.R_radar_to_camera @ radar_xyz.T).T + self.t_radar_to_camera
            valid_depth = radar_camera[:, 2] > 0.1

            if not np.any(valid_depth):
                return np.array([]).reshape(0, 4), np.array([]).reshape(0, 3)

            K = self._build_camera_intrinsics(camera_image)
            projected = K @ radar_camera[valid_depth].T
            u = projected[0, :] / projected[2, :]
            v = projected[1, :] / projected[2, :]

            inside_2d_bbox = (
                (u >= x1) & (u <= x2) &
                (v >= y1) & (v <= y2) &
                (u >= 0) & (u < camera_image.width) &
                (v >= 0) & (v < camera_image.height)
            )

            full_2d_mask = np.zeros(len(radar_points), dtype=bool)
            full_2d_mask[valid_depth] = inside_2d_bbox

            filtered_points = radar_points[full_2d_mask]
            radar_xyz_filtered = radar_xyz[full_2d_mask]

        logger.debug(f"[RADAR FILTER] After 2D bbox filter: {len(filtered_points)}/{len(radar_points)} points "
                    f"({100*len(filtered_points)/max(len(radar_points),1):.1f}%)")

        if len(filtered_points) == 0:
            return filtered_points, np.array([]).reshape(0, 3)

        # Stage 2: Clustering - Use DBSCAN to isolate closest cluster (vehicle)
        before_clustering = len(filtered_points)
        filtered_points, radar_xyz_filtered = self._cluster_radar_points(filtered_points, radar_xyz_filtered)

        logger.debug(f"[RADAR FILTER] After clustering: {len(filtered_points)}/{before_clustering} points "
                    f"({100*len(filtered_points)/max(before_clustering,1):.1f}%)")

        if len(filtered_points) == 0:
            return filtered_points, np.array([]).reshape(0, 3)

        # Stage 3: Static object filtering - Doppler analysis
        # Transform ego velocity to vehicle frame (same frame as radar measurements)
        v_ego_vehicle = self._transform_world_to_vehicle(ego_velocity_world, ego_rotation)

        # Line-of-sight unit vectors from radar to target
        los_distances = np.linalg.norm(radar_xyz_filtered, axis=1, keepdims=True)
        los_unit = radar_xyz_filtered / los_distances

        # Expected Doppler for static objects
        # Static object has same velocity as ego in world frame
        # In radar measurement: doppler_static = -dot(los, v_ego_vehicle)
        # Negative sign: approaching (radar moving toward static object) = negative Doppler
        doppler_static = -np.dot(los_unit, v_ego_vehicle)

        # Actual Doppler measurements
        doppler_actual = filtered_points[:, 0]

        # Doppler difference: if small, likely static (ground, barrier)
        doppler_difference = np.abs(doppler_actual - doppler_static)
        # Filter: keep only points with significant difference (moving objects)
        moving_threshold = 0.3  
        moving_mask = doppler_difference > moving_threshold
        before_static = len(filtered_points)
        filtered_points = filtered_points[moving_mask]
        radar_xyz_filtered = radar_xyz_filtered[moving_mask]

        logger.debug(f"[RADAR FILTER] After static filter: {len(filtered_points)}/{before_static} points "
                    f"({100*len(filtered_points)/max(before_static,1):.1f}%) - removed {np.sum(~moving_mask)} static")

        # DIAGNOSTIC: Log radar velocities
        if len(filtered_points) > 0:
            velocities = filtered_points[:, 0]
            logger.debug(f"[RADAR VELOCITIES] min={velocities.min():.1f} max={velocities.max():.1f} "
                         f"median={np.median(velocities):.1f} mean={np.mean(velocities):.1f} m/s")

        return filtered_points, radar_xyz_filtered

    def _least_squares(
        self,
        radar_points: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        delta: float = 2.0,
        max_iter: int = 50
    ) -> Optional[np.ndarray]:
        """
        Estimate 3D relative velocity using robust least squares (Huber loss).

        Solves the system: v_radial_i = dot(beam_i, v_vehicle) for all radar points.
        Uses Huber loss to handle outliers while leveraging all measurements.

        Tries GPU acceleration first, falls back to scipy automatically.

        Args:
            radar_points: (N, 4) radar measurements [velocity, azimuth, altitude, depth]
            initial_guess: (3,) initial velocity guess (uses median approach if None)
            delta: Huber loss threshold
            max_iter: Maximum IRLS iterations for GPU solver

        Returns:
            (3,) 3D velocity in vehicle frame or None if solving fails
        """
        if len(radar_points) < 3:  # Need at least 3 points for 3 unknowns
            return None

        # Extract measurements
        doppler_velocities = radar_points[:, 0]  # (N,) radial velocities
        azimuth = radar_points[:, 1]
        altitude = radar_points[:, 2]

        # Compute beam directions (N, 3)
        beam_directions = np.stack([
            np.cos(altitude) * np.cos(azimuth),
            np.cos(altitude) * np.sin(azimuth),
            np.sin(altitude)
        ], axis=1)

        # Initial guess: use median velocity with its beam direction
        if initial_guess is None:
            n = len(doppler_velocities)
            median_idx = np.argpartition(doppler_velocities, n // 2)[n // 2]
            v_radial_med = doppler_velocities[median_idx]
            beam_med = beam_directions[median_idx]
            initial_guess = v_radial_med * beam_med

        # Try GPU solver first
        try:
            import torch
            if torch.cuda.is_available():
                # GPU implementation using PyTorch with IRLS
                A = torch.from_numpy(beam_directions).float().cuda()
                b = torch.from_numpy(doppler_velocities).float().cuda()
                x = torch.from_numpy(initial_guess).float().cuda()

                # IRLS iterations
                for iteration in range(max_iter):
                    # Compute residuals
                    residuals = A @ x - b

                    # Huber weights
                    abs_res = torch.abs(residuals)
                    weights = torch.where(abs_res <= delta, torch.ones_like(abs_res), delta / abs_res)

                    # Weighted least squares
                    W_sqrt = torch.sqrt(weights).unsqueeze(-1)
                    A_weighted = W_sqrt * A
                    b_weighted = (W_sqrt.squeeze() * b).unsqueeze(-1)

                    # Solve using PyTorch
                    try:
                        x_new = torch.linalg.lstsq(A_weighted, b_weighted).solution.squeeze()
                    except:
                        # Fallback to current solution on failure
                        break

                    # Check convergence
                    if torch.norm(x_new - x) < 1e-6:
                        x = x_new
                        break

                    x = x_new

                # Move back to CPU
                result = x.cpu().numpy()
                logger.debug(f"[ROBUST LS GPU] Solved with {len(doppler_velocities)} points")
                return result

        except (ImportError, RuntimeError, AttributeError):
            pass  # Fall through to CPU fallback

        # CPU fallback using scipy
        def residuals(v_vehicle):
            predicted = beam_directions @ v_vehicle
            return doppler_velocities - predicted

        # Solve using Huber loss (robust to outliers)
        result = least_squares(
            residuals,
            x0=initial_guess,
            loss='huber',
            f_scale=delta,
            method='trf'
        )

        if not result.success:
            logger.debug(f"[ROBUST LS] Failed to converge: {result.message}")
            return None

        logger.debug(f"[ROBUST LS CPU] Solved with {len(doppler_velocities)} points, "
                     f"residual RMS: {np.sqrt(np.mean(result.fun**2)):.2f} m/s")

        return result.x

    def _solve_3d_velocity(
        self,
        radar_points: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Estimate 3D relative velocity from radar Doppler measurements.

        Strategy:
        1. Try robust least squares (Huber loss) - uses all measurements, handles beam divergence
        2. Fall back to trimmed mean + median beam - uses ~80% of data, robust to outliers
        3. Fall back to median + median beam - most conservative, uses median measurement

        Args:
            radar_points: (N, 4) radar measurements [velocity, azimuth, altitude, depth]

        Returns:
            (3,) 3D velocity in vehicle frame or None
        """
        if len(radar_points) == 0:
            return None
        
        # Extract Doppler velocities and angles
        doppler_velocities = radar_points[:, 0]
        azimuth = radar_points[:, 1]
        altitude = radar_points[:, 2]

        # Strategy 1: Try robust least squares first (uses all measurements optimally)
        v_relative_ls = self._least_squares(radar_points)
        if v_relative_ls is not None:
            logger.debug(f"[VELOCITY] Using robust least squares with {len(radar_points)} points")
            return v_relative_ls

        # Strategy 2: Fall back to trimmed mean (drops 10% extremes, uses ~80% of data)
        logger.debug(f"[VELOCITY] Robust LS failed, trying trimmed mean")

        if len(doppler_velocities) >= 5:  # Need enough points for trimming to be meaningful
            # Trim 10% from each tail (drops 20% total outliers)
            v_radial_trimmed = stats.trim_mean(doppler_velocities, proportiontocut=0.1)

            # Use beam direction of the median velocity point (our robust approach)
            n = len(doppler_velocities)
            median_idx = np.argpartition(doppler_velocities, n // 2)[n // 2]
            az_median = azimuth[median_idx]
            alt_median = altitude[median_idx]

            beam_unit = np.array([
                np.cos(alt_median) * np.cos(az_median),
                np.cos(alt_median) * np.sin(az_median),
                np.sin(alt_median)
            ])

            v_relative_trimmed = v_radial_trimmed * beam_unit

            logger.debug(f"[VELOCITY TRIMMED] {len(radar_points)} points, v_radial={v_radial_trimmed:.1f} m/s, "
                         f"v_rel={np.linalg.norm(v_relative_trimmed)*3.6:.1f} km/h")

            return v_relative_trimmed 

        # Strategy 3: Final fallback to median (most conservative)
        logger.debug(f"[VELOCITY] Using median fallback")

        # Use median velocity with median point's beam direction
        n = len(doppler_velocities)
        median_idx = np.argpartition(doppler_velocities, n // 2)[n // 2]
        v_radial_median = doppler_velocities[median_idx]
        az_median = azimuth[median_idx]
        alt_median = altitude[median_idx]

        beam_unit = np.array([
            np.cos(alt_median) * np.cos(az_median),
            np.cos(alt_median) * np.sin(az_median),
            np.sin(alt_median)
        ])

        v_relative_median = v_radial_median * beam_unit

        logger.debug(f"[VELOCITY MEDIAN] {len(radar_points)} points, v_radial={v_radial_median:.1f} m/s, "
                     f"v_rel={np.linalg.norm(v_relative_median)*3.6:.1f} km/h")

        return v_relative_median

    def _transform_world_to_vehicle(
        self,
        velocity_world: np.ndarray,
        yaw_radians: float
    ) -> np.ndarray:
        """
        Transform velocity from world frame to vehicle frame.

        Args:
            velocity_world: (3,) velocity in world frame [vx_world, vy_world, vz_world]
            yaw_radians: Vehicle yaw rotation in radians (CCW from X-axis)

        Returns:
            (3,) velocity in vehicle frame [vx_vehicle, vy_vehicle, vz_vehicle]
        """
        # Rotation matrix from world to vehicle (inverse rotation R(-yaw))
        # When vehicle is rotated by yaw in world frame, we need inverse transform
        cos_yaw = np.cos(yaw_radians)
        sin_yaw = np.sin(yaw_radians)

        R_world_to_vehicle = np.array([
            [cos_yaw,  sin_yaw, 0],
            [-sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        return R_world_to_vehicle @ velocity_world

    def _transform_vehicle_to_world(
        self,
        velocity_vehicle: np.ndarray,
        yaw_radians: float
    ) -> np.ndarray:
        """
        Transform velocity from vehicle frame to world frame.

        Args:
            velocity_vehicle: (3,) velocity in vehicle frame
            yaw_radians: Vehicle yaw rotation in radians (CCW from X-axis)

        Returns:
            (3,) velocity in world frame
        """
        cos_yaw = np.cos(yaw_radians)
        sin_yaw = np.sin(yaw_radians)

        R_vehicle_to_world = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        return R_vehicle_to_world @ velocity_vehicle

    def project_points_to_image(
        self,
        points_3d: np.ndarray,
        camera_image: 'CameraImage'
    ) -> np.ndarray:
        """
        Project 3D points in LiDAR frame to 2D image coordinates.

        Args:
            points_3d: (N, 3) points in LiDAR/vehicle frame
            camera_image: Camera image for dimensions

        Returns:
            (N, 2) [u, v] image coordinates
        """
        if len(points_3d) == 0:
            return np.array([])

        # Transform to camera frame
        points_camera = (self.R_lidar_to_camera @ points_3d.T).T + self.t_lidar_to_camera

        # Filter points in front
        valid = points_camera[:, 2] > 0.1
        points_camera = points_camera[valid]

        if len(points_camera) == 0:
            return np.array([])

        # Build camera intrinsics
        K = self._build_camera_intrinsics(camera_image)

        # Project
        projected = K @ points_camera.T
        u = projected[0, :] / projected[2, :]
        v = projected[1, :] / projected[2, :]

        return np.stack([u, v], axis=1)
