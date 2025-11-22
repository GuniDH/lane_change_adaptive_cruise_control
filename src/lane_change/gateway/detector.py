"""
Detector - Interface for camera-based object detection methods.

This module defines the abstract interface that any detection method
(YOLO, SSD, Faster R-CNN, etc.) must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from .data_types import CameraImage, DetectedObject


class Detector(ABC):
    """Abstract interface for camera-based object detection methods."""

    @abstractmethod
    def detect_objects(self, camera_image: CameraImage) -> List[DetectedObject]:
        """
        Detect objects in camera image and return bounding boxes.

        Args:
            camera_image: Camera image data

        Returns:
            List of detected objects with bounding boxes and object types

        Note:
            This method will be implemented by specific detection adapters
            (e.g., YoloDetectionAdapter, SsdDetectionAdapter, etc.). The actual
            implementation will be part of the perception module.
        """
        pass


class ILaneDetector(ABC):
    """Abstract interface for lane detection algorithms."""

    @abstractmethod
    def detect_lanes(self, camera_image: CameraImage) -> List[List[Tuple[int, int]]]:
        """
        Detect lane boundaries in camera image.

        Args:
            camera_image: Camera image data

        Returns:
            List of detected lanes, where each lane is a list of (x, y) pixel coordinates.
            Lanes are returned in left-to-right order.
        """
        pass


class ITrafficLightClassifier(ABC):
    """Abstract interface for traffic light state classification."""

    @abstractmethod
    def extract_colors_batch(self, image, bboxes: List[Tuple[int, int, int, int]]) -> List[Optional[str]]:
        """
        Extract traffic light colors from multiple bounding boxes.

        Args:
            image: RGB image data (H, W, 3)
            bboxes: List of bounding boxes [(x1, y1, x2, y2), ...]

        Returns:
            List of color strings: ["red", "green", "yellow", "irrelevant", None, ...]
            None for bboxes that failed classification.
        """
        pass