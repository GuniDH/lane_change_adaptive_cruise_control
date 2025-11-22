"""
Rich domain model for tracked objects with polymorphic behavior.

This module defines the hierarchy of tracked objects (vehicles, pedestrians, traffic lights)
with proper polymorphism for distance and TTC calculations. Each object type implements
its own calculation strategy based on its characteristics.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from lane_change.gateway.data_types import DetectedObject, VelocityEstimate, Position3D
from lane_change.perception.geometry_utils import compute_bbox_from_lidar_cluster, compute_min_distance_between_aabb

if TYPE_CHECKING:
    from lane_change.gateway.sensor_manager import ISensorManager

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject(ABC):
    """Base class for all tracked objects with polymorphic behavior."""
    track_id: int
    detection: DetectedObject
    position_3d: Optional[Position3D] = None
    distance: Optional[float] = None
    ttc: Optional[float] = None

    @abstractmethod
    def compute_distance(self, ego_vehicle: 'CarlaEgoVehicle', sensor_manager: 'ISensorManager') -> Optional[float]:
        """
        Compute distance from ego vehicle. Implementation varies by object type.

        Args:
            ego_vehicle: Reference to ego vehicle
            sensor_manager: Sensor manager for accessing sensor configuration

        Returns:
            Distance in meters, or None if cannot be computed
        """
        pass

    @abstractmethod
    def compute_ttc(self, distance: float, ego_vehicle: 'CarlaEgoVehicle') -> Optional[float]:
        """
        Compute time-to-collision. Implementation varies by object type.

        Args:
            distance: Distance to object in meters
            ego_vehicle: Reference to ego vehicle

        Returns:
            TTC in seconds, or None if not approaching
        """
        pass

    def update_metrics(self, ego_vehicle: 'CarlaEgoVehicle', sensor_manager: 'ISensorManager'):
        """
        Template method - computes distance and TTC for any object type.

        This is the same for all objects - polymorphism is in the abstract methods.

        Args:
            ego_vehicle: Reference to ego vehicle
            sensor_manager: Sensor manager for accessing sensor configuration
        """
        self.distance = self.compute_distance(ego_vehicle, sensor_manager)
        if self.distance is not None:
            self.ttc = self.compute_ttc(self.distance, ego_vehicle)


@dataclass
class TrackedMovingObject(TrackedObject):
    """Base class for moving objects (vehicles, pedestrians) with velocity and lane assignment."""
    absolute_lane_idx: int = 0
    velocity_estimate: Optional[VelocityEstimate] = None
    gap: Optional[float] = None

    def get_lane_position(self, ego_lane_idx: int) -> str:
        """
        Get relative lane position compared to ego vehicle.

        Args:
            ego_lane_idx: Ego vehicle's 1-based lane index

        Returns:
            Relative lane position: "ego", "left", or "right"
        """
        if self.absolute_lane_idx == ego_lane_idx:
            return "ego"
        elif self.absolute_lane_idx < ego_lane_idx:
            return "left"
        else:
            return "right"

    @staticmethod
    def _transform_world_to_vehicle(velocity_world: np.ndarray, yaw_radians: float) -> np.ndarray:
        """
        Transform velocity from world frame to vehicle frame.

        Args:
            velocity_world: Velocity in world frame [vx, vy, vz]
            yaw_radians: Vehicle yaw angle in radians

        Returns:
            Velocity in vehicle frame [vx, vy, vz]
        """
        cos_yaw = np.cos(yaw_radians)
        sin_yaw = np.sin(yaw_radians)

        rotation_matrix = np.array([
            [cos_yaw, sin_yaw, 0],
            [-sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        return rotation_matrix @ velocity_world


@dataclass
class TrackedVehicle(TrackedMovingObject):
    """Tracked vehicle with bbox-based distance calculation."""

    def assign_lane(
        self,
        pipeline: 'PerceptionPipeline'
    ) -> Optional[int]:
        """
        Calculate which lane this vehicle is in using perception pipeline logic.

        Delegates to pipeline's _calculate_vehicle_lane() which implements multi-strategy
        lane assignment (centroid, track history, bbox center, cluster voting).

        Args:
            pipeline: PerceptionPipeline instance with lane calculation methods

        Returns:
            1-based lane index from leftmost lane, or None if not determinable
        """
        return pipeline._calculate_vehicle_lane(
            self.detection,
            pipeline.sensor_manager.get_camera_image('front'),
            self.position_3d,
            self.track_id
        )

    def compute_distance(self, ego_vehicle, sensor_manager) -> Optional[float]:
        """
        Compute distance using axis-aligned bounding box method.

        This is the most accurate method for vehicles, using LiDAR clusters to
        estimate bounding boxes and computing minimum separation distance.

        Args:
            ego_vehicle: Reference to ego vehicle
            sensor_manager: Sensor manager for LiDAR offset

        Returns:
            Minimum distance between bounding boxes in meters, or None
        """
        if self.position_3d is None:
            return None

        # Get ego vehicle bounding box from CARLA
        ego_bbox = ego_vehicle.carla_actor.bounding_box
        ego_half_length = ego_bbox.extent.x
        ego_half_width = ego_bbox.extent.y
        ego_bbox_location = ego_bbox.location

        # Ego bbox center in vehicle frame (accounts for offset from vehicle reference point)
        ego_center = np.array([ego_bbox_location.x, ego_bbox_location.y])

        # Get LiDAR sensor offset from sensor manager
        lidar_config = sensor_manager.lidar_config if hasattr(sensor_manager, 'lidar_config') else None
        if lidar_config:
            lidar_offset = np.array([lidar_config['location'][0], lidar_config['location'][1]])
            logger.debug(f"LiDAR offset: x={lidar_offset[0]:.3f}, y={lidar_offset[1]:.3f}")
        else:
            # Fallback to known CARLA position if config not available
            lidar_offset = np.array([-1.0, 0.0])
            logger.warning("LiDAR config not found, using hardcoded offset [-1.0, 0.0]")

        # Transform object position from LiDAR frame to vehicle frame
        obj_center_lidar = self.position_3d.center_3d[:2]
        obj_center_vehicle = obj_center_lidar + lidar_offset

        # Compute object bounding box from LiDAR cluster
        obj_length, obj_width = compute_bbox_from_lidar_cluster(self.position_3d.lidar_points)
        obj_half_length = obj_length / 2
        obj_half_width = obj_width / 2

        # Compute minimum distance between axis-aligned boxes
        distance_2d = compute_min_distance_between_aabb(
            ego_center, ego_half_length, ego_half_width,
            obj_center_vehicle, obj_half_length, obj_half_width
        )

        logger.debug(
            f"Bbox distance: {distance_2d:.2f}m | "
            f"Ego: center=[{ego_center[0]:.2f},{ego_center[1]:.2f}], size=[{2*ego_half_length:.1f}x{2*ego_half_width:.1f}]m | "
            f"Obj: center=[{obj_center_vehicle[0]:.2f},{obj_center_vehicle[1]:.2f}], size=[{obj_length:.1f}x{obj_width:.1f}]m"
        )

        # Compute longitudinal gap (ego front bumper to vehicle back bumper)
        if self.position_3d.lidar_points is not None and len(self.position_3d.lidar_points) > 0:
            # Get ego front bumper position in vehicle frame
            ego_front_x = ego_bbox_location.x + ego_half_length

            # Get vehicle back bumper from LiDAR (minimum X in LiDAR frame)
            vehicle_back_x_lidar = np.min(self.position_3d.lidar_points[:, 0])

            # Transform from LiDAR frame to vehicle frame
            vehicle_back_x_vehicle = vehicle_back_x_lidar + lidar_offset[0]

            # Gap = vehicle back - ego front (positive = space ahead, negative = overlap)
            self.gap = float(vehicle_back_x_vehicle - ego_front_x)

        return distance_2d

    def compute_ttc(self, distance: float, ego_vehicle) -> Optional[float]:
        """
        Compute TTC using relative velocity and range rate.

        For vehicles, we use the relative velocity to compute how fast the distance
        is changing (range rate) and calculate TTC from that.

        Args:
            distance: Distance to vehicle in meters
            ego_vehicle: Reference to ego vehicle

        Returns:
            TTC in seconds, or None if not approaching
        """
        if distance is None or self.velocity_estimate is None:
            return None

        # Get ego vehicle bounding box
        ego_bbox = ego_vehicle.carla_actor.bounding_box
        ego_bbox_location = ego_bbox.location
        ego_center = np.array([ego_bbox_location.x, ego_bbox_location.y])

        # Get LiDAR offset
        lidar_config = ego_vehicle.sensor_manager.lidar_config if hasattr(ego_vehicle, 'sensor_manager') and hasattr(ego_vehicle.sensor_manager, 'lidar_config') else None
        if lidar_config:
            lidar_offset = np.array([lidar_config['location'][0], lidar_config['location'][1]])
        else:
            lidar_offset = np.array([-1.0, 0.0])

        # Transform object position from LiDAR frame to vehicle frame
        obj_center_lidar = self.position_3d.center_3d[:2]
        obj_center_vehicle = obj_center_lidar + lidar_offset

        # Transform relative velocity from world frame to vehicle frame
        velocity_relative_world = self.velocity_estimate.velocity_relative
        # Get ego yaw angle from CARLA actor
        transform = ego_vehicle.carla_actor.get_transform()
        ego_yaw = np.deg2rad(transform.rotation.yaw)
        velocity_relative_vehicle = self._transform_world_to_vehicle(velocity_relative_world, ego_yaw)

        # Compute range rate: project relative velocity onto distance vector
        distance_vector = obj_center_vehicle - ego_center

        # Handle very small distances to avoid division issues
        if distance < 0.1:
            return 0.0

        distance_unit = distance_vector / distance

        # Range rate = rate of change of distance
        # Positive = moving apart, Negative = approaching
        range_rate = np.dot(velocity_relative_vehicle[:2], distance_unit)

        # TTC only defined when approaching (negative range rate = distance decreasing)
        if range_rate < -0.1:  # -0.1 m/s threshold to avoid division by near-zero
            return distance / (-range_rate)

        return None


@dataclass
class TrackedPedestrian(TrackedMovingObject):
    """Tracked pedestrian with bbox-based distance calculation."""

    def assign_lane(
        self,
        pipeline: 'PerceptionPipeline'
    ) -> Optional[int]:
        """
        Calculate which lane this pedestrian is in.

        Uses same logic as TrackedVehicle - delegates to pipeline's lane calculation.

        Args:
            pipeline: PerceptionPipeline instance with lane calculation methods

        Returns:
            1-based lane index from leftmost lane, or None if not determinable
        """
        return pipeline._calculate_vehicle_lane(
            self.detection,
            pipeline.sensor_manager.get_camera_image('front'),
            self.position_3d,
            self.track_id
        )

    def compute_distance(self, ego_vehicle, sensor_manager) -> Optional[float]:
        """
        Compute distance using axis-aligned bounding box method.

        Pedestrians use the same algorithm as vehicles.

        Args:
            ego_vehicle: Reference to ego vehicle
            sensor_manager: Sensor manager for LiDAR offset

        Returns:
            Minimum distance between bounding boxes in meters, or None
        """
        if self.position_3d is None:
            return None

        # Get ego vehicle bounding box from CARLA
        ego_bbox = ego_vehicle.carla_actor.bounding_box
        ego_half_length = ego_bbox.extent.x
        ego_half_width = ego_bbox.extent.y
        ego_bbox_location = ego_bbox.location

        # Ego bbox center in vehicle frame
        ego_center = np.array([ego_bbox_location.x, ego_bbox_location.y])

        # Get LiDAR sensor offset
        lidar_config = sensor_manager.lidar_config if hasattr(sensor_manager, 'lidar_config') else None
        if lidar_config:
            lidar_offset = np.array([lidar_config['location'][0], lidar_config['location'][1]])
        else:
            lidar_offset = np.array([-1.0, 0.0])

        # Transform object position from LiDAR frame to vehicle frame
        obj_center_lidar = self.position_3d.center_3d[:2]
        obj_center_vehicle = obj_center_lidar + lidar_offset

        # Compute object bounding box from LiDAR cluster
        obj_length, obj_width = compute_bbox_from_lidar_cluster(self.position_3d.lidar_points)
        obj_half_length = obj_length / 2
        obj_half_width = obj_width / 2

        # Compute minimum distance
        distance_2d = compute_min_distance_between_aabb(
            ego_center, ego_half_length, ego_half_width,
            obj_center_vehicle, obj_half_length, obj_half_width
        )

        # Compute longitudinal gap
        if self.position_3d.lidar_points is not None and len(self.position_3d.lidar_points) > 0:
            ego_front_x = ego_bbox_location.x + ego_half_length
            pedestrian_back_x_lidar = np.min(self.position_3d.lidar_points[:, 0])
            pedestrian_back_x_vehicle = pedestrian_back_x_lidar + lidar_offset[0]
            self.gap = float(pedestrian_back_x_vehicle - ego_front_x)

        return distance_2d

    def compute_ttc(self, distance: float, ego_vehicle) -> Optional[float]:
        """
        Compute TTC using relative velocity.

        Args:
            distance: Distance to pedestrian in meters
            ego_vehicle: Reference to ego vehicle

        Returns:
            TTC in seconds, or None if not approaching
        """
        if distance is None or self.velocity_estimate is None:
            return None

        # Get ego vehicle bounding box
        ego_bbox = ego_vehicle.carla_actor.bounding_box
        ego_bbox_location = ego_bbox.location
        ego_center = np.array([ego_bbox_location.x, ego_bbox_location.y])

        # Get LiDAR offset
        lidar_config = ego_vehicle.sensor_manager.lidar_config if hasattr(ego_vehicle, 'sensor_manager') and hasattr(ego_vehicle.sensor_manager, 'lidar_config') else None
        if lidar_config:
            lidar_offset = np.array([lidar_config['location'][0], lidar_config['location'][1]])
        else:
            lidar_offset = np.array([-1.0, 0.0])

        # Transform object position
        obj_center_lidar = self.position_3d.center_3d[:2]
        obj_center_vehicle = obj_center_lidar + lidar_offset

        # Transform relative velocity
        velocity_relative_world = self.velocity_estimate.velocity_relative
        ego_yaw = ego_vehicle.get_yaw()
        velocity_relative_vehicle = self._transform_world_to_vehicle(velocity_relative_world, ego_yaw)

        # Compute range rate
        distance_vector = obj_center_vehicle - ego_center

        if distance < 0.1:
            return 0.0

        distance_unit = distance_vector / distance
        range_rate = np.dot(velocity_relative_vehicle[:2], distance_unit)

        if range_rate < -0.1:
            return distance / (-range_rate)

        return None


@dataclass
class TrackedStaticObject(TrackedObject):
    """Base class for static objects (traffic lights, signs, cones)."""
    pass


@dataclass
class TrackedTrafficLight(TrackedStaticObject):
    """Tracked traffic light with simple distance calculation and color detection."""
    color: Optional[str] = None  # YOLO classes: "red", "green", "yellow", "irrelevant" | None on error

    def compute_distance(self, ego_vehicle, sensor_manager) -> Optional[float]:
        """
        Compute distance using simple front distance from LiDAR.

        Traffic lights are static and typically directly ahead, so we use
        a simpler distance calculation based on the front of the object.

        Args:
            ego_vehicle: Reference to ego vehicle
            sensor_manager: Sensor manager for LiDAR offset

        Returns:
            Distance from ego front bumper to traffic light in meters, or None
        """
        if self.position_3d is None:
            return None

        # Get traffic light front X in LiDAR frame
        traffic_light_front_x_lidar = np.min(self.position_3d.lidar_points[:, 0])

        # Get LiDAR offset
        lidar_config = sensor_manager.lidar_config if hasattr(sensor_manager, 'lidar_config') else None
        if lidar_config:
            lidar_offset_x = lidar_config['location'][0]
        else:
            lidar_offset_x = -1.0
            logger.warning("LiDAR config not found for traffic light, using hardcoded offset x=-1.0")

        # Transform from LiDAR frame to vehicle frame
        traffic_light_front_x_vehicle = traffic_light_front_x_lidar + lidar_offset_x

        # Distance from ego front bumper to traffic light
        if hasattr(ego_vehicle, 'carla_actor'):
            ego_bbox = ego_vehicle.carla_actor.bounding_box
            ego_front_x = ego_bbox.location.x + ego_bbox.extent.x
            return float(traffic_light_front_x_vehicle - ego_front_x)
        else:
            # Fallback: distance from vehicle center
            return float(traffic_light_front_x_vehicle)

    def compute_ttc(self, distance: float, ego_vehicle) -> Optional[float]:
        """
        Compute TTC using ego speed.

        For static objects, we use only the ego vehicle's speed since the
        object is not moving.

        Args:
            distance: Distance to traffic light in meters
            ego_vehicle: Reference to ego vehicle

        Returns:
            TTC in seconds, or inf if ego is not moving
        """
        if distance is None:
            return None

        ego_speed_ms = ego_vehicle.get_speed_ms()
        if ego_speed_ms > 0.1:
            return distance / ego_speed_ms

        return float('inf')
