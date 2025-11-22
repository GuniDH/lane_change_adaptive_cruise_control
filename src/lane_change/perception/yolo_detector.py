"""
YOLO Detector - Vehicle detection and tracking using YOLO.

This module implements the Detector interface using YOLO from ultralytics,
with built-in tracking for maintaining consistent vehicle IDs across frames.
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional
from ultralytics import YOLO
from lane_change.gateway.detector import Detector
from lane_change.gateway.data_types import CameraImage, DetectedObject, BoundingBox

logger = logging.getLogger(__name__)


class YoloDetector(Detector):
    """YOLO-based vehicle detector with built-in tracking."""

    # COCO class ID to name mapping for detectable objects
    DETECTION_CLASSES = {
        0: "person",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic_light",
    }

    def __init__(self, model_path: str = 'yolo12m.pt', ego_vehicle_mask_normalized: Optional[np.ndarray] = None):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model weights (defaults to nano version)
            ego_vehicle_mask_normalized: Optional polygon mask in normalized (0.0-1.0) coordinates
        """
        self.model = YOLO(model_path)
        self.model.to('cuda')
        self.ego_vehicle_mask_normalized = ego_vehicle_mask_normalized

        # Pre-allocate buffer for masked image (reused every frame to avoid allocation overhead)
        self.masked_image_buffer = None

        logger.info(f"YOLO model loaded: {model_path} on device: cuda")

    def detect_objects(self, camera_image: CameraImage) -> List[DetectedObject]:
        """
        Detect and track vehicles in camera image.

        Args:
            camera_image: Camera image data

        Returns:
            List of detected objects with bounding boxes, object types, and track IDs
        """
        # Preprocess image: black out ego vehicle region if mask is provided
        image_for_yolo = camera_image.image_data

        if self.ego_vehicle_mask_normalized is not None:
            # Scale normalized mask to current image dimensions
            width, height = camera_image.width, camera_image.height
            ego_mask_scaled = np.array([
                (int(x * width), int(y * height))
                for x, y in self.ego_vehicle_mask_normalized
            ], dtype=np.int32)

            # Reuse pre-allocated buffer instead of allocating new memory every frame
            if self.masked_image_buffer is None or self.masked_image_buffer.shape != camera_image.image_data.shape:
                self.masked_image_buffer = np.empty_like(camera_image.image_data)

            # Copy to buffer (faster than .copy() which allocates)
            np.copyto(self.masked_image_buffer, camera_image.image_data)

            # Fill polygon with black (0, 0, 0)
            cv2.fillPoly(self.masked_image_buffer, [ego_mask_scaled], (0, 0, 0))
            image_for_yolo = self.masked_image_buffer

        # Get ByteTrack config path from environment or use default
        bytetrack_config = os.environ.get('BYTETRACK_CONFIG_PATH')
        if not bytetrack_config:
            # Use relative path from project root
            project_root = Path(__file__).parent.parent.parent.parent
            bytetrack_config = str(project_root / "config\\my_bytetrack.yaml")

        # Run YOLO with tracking
        results = self.model.track(
            image_for_yolo,
            persist=True,  # Maintain tracks across frames
            conf=0.40,       # Confidence threshold
            iou=0.5,
            tracker=bytetrack_config,  # Use ByteTrack
            verbose=False  # Suppress YOLO logging
        )

        detected_objects = []

        for result in results:
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    # Get class ID
                    cls_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls[0])

                    # Check if this is a detectable object
                    if cls_id not in self.DETECTION_CLASSES:
                        continue  # Skip unknown classes

                    class_name = self.DETECTION_CLASSES[cls_id]

                    # Extract confidence
                    confidence = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf[0])

                    # Extract tracking ID
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id.item()) if hasattr(box.id, 'item') else int(box.id[0])

                    # Extract bounding box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    # Create bounding box with track ID in label
                    bbox = BoundingBox(
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1),
                        class_label=f"vehicle_{track_id}" if track_id is not None else "vehicle_new"
                    )

                    # Create detected object with metadata
                    detected_obj = DetectedObject(
                        bounding_box=bbox,
                        confidence=confidence,
                        class_id=cls_id,
                        class_name=class_name
                    )

                    detected_objects.append(detected_obj)

        logger.debug(f"Detected {len(detected_objects)} objects in frame")
        return detected_objects

    def reset_tracker(self):
        """Reset the tracker state for new sequences."""
        # This will clear the tracker's memory
        self.model.predictor = None