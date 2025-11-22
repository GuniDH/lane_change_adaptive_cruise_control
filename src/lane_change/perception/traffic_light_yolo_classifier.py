"""
Traffic Light YOLO Classifier - Deep learning-based traffic light state classification.

YOLO outputs 4 classes:
- red: Red traffic light
- green: Green traffic light
- yellow: Yellow traffic light
- irrelevant: Traffic light seen from side (not relevant to ego)

Returns None on classification failure (inference error, invalid bbox, etc.)
"""

import numpy as np
import logging
import torch
import yaml
from typing import Tuple, List, Optional
from pathlib import Path
from ultralytics import YOLO
from lane_change.gateway.detector import ITrafficLightClassifier

logger = logging.getLogger(__name__)


class TrafficLightYoloClassifier(ITrafficLightClassifier):
    """Classify traffic light state using trained YOLO11n-cls model."""

    def __init__(self, model_path: str, imgsz: Optional[int] = None, config_path: Optional[str] = None):
        """
        Initialize YOLO classifier.

        Args:
            model_path: Path to trained YOLO11n-cls weights (.pt file)
            imgsz: Image size for inference (must match training size). If None, reads from config.
            config_path: Path to carla_config.yaml (optional, defaults to config/carla_config.yaml)
        """
        self.model_path = Path(model_path)

        # Load configuration if not provided
        if imgsz is None:
            if config_path is None:
                project_root = Path(__file__).parent.parent.parent.parent
                config_path = project_root / "config" / "carla_config.yaml"

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            self.imgsz = config['perception']['traffic_light_classifier']['image_size']
            logger.info(f"Loaded image size from config: {self.imgsz}")
        else:
            self.imgsz = imgsz

        # Validate model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO classifier weights not found at {model_path}. "
                f"Please train the model first using scripts/traffic_light_classifier/train.py"
            )

        # Detect GPU availability
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Load model once (reused for all frames)
        logger.info(f"Loading YOLO traffic light classifier from {model_path}")
        self.model = YOLO(str(self.model_path))

        # Explicitly move model to GPU if available
        if self.device != 'cpu':
            self.model.to(self.device)
            logger.info(f"YOLO classifier loaded successfully on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("YOLO classifier loaded on CPU - GPU not available!")
            logger.warning("Performance will be significantly degraded!")

    def extract_colors_batch(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Optional[str]]:
        """
        Extract traffic light colors from multiple bounding boxes (batched GPU inference).

        This is the primary method for efficiency - processes all traffic lights in frame
        with a single GPU call instead of multiple sequential inferences.

        Args:
            image: RGB image (H, W, 3) - will be converted to BGR for YOLO
            bboxes: List of bounding boxes [(x1, y1, x2, y2), ...]

        Returns:
            List of color strings: ["red", "green", "yellow", "irrelevant", None, ...]
            None for bboxes that failed classification
        """
        if not bboxes:
            return []

        try:
            # Get image dimensions for bbox clamping
            img_height, img_width = image.shape[:2]

            # Extract crops from image (bboxes are already expanded by caller)
            crops = []
            for x1, y1, x2, y2 in bboxes:
                # Validate bbox coordinates
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid bbox dimensions: ({x1}, {y1}, {x2}, {y2})")
                    crops.append(None)
                    continue

                # Clamp to image boundaries
                x1_clamped = max(0, x1)
                y1_clamped = max(0, y1)
                x2_clamped = min(img_width, x2)
                y2_clamped = min(img_height, y2)

                # Extract crop
                crop = image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

                # Skip empty or tiny crops (threshold lowered to 3px for low-res scenarios)
                if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
                    logger.warning(f"Crop too small: {crop.shape}")
                    crops.append(None)
                    continue

                # Convert RGB to BGR because YOLO assumes numpy arrays are in BGR format
                # CARLA gives us RGB, but YOLO expects BGR (OpenCV's default color space)
                # YOLO will then convert BGR→RGB internally to match training format
                crop_bgr = crop[:, :, ::-1]
                crops.append(crop_bgr)

            # Filter out None crops for batch inference
            valid_crops = [c for c in crops if c is not None]
            valid_indices = [i for i, c in enumerate(crops) if c is not None]

            if not valid_crops:
                # All crops were invalid
                return [None] * len(bboxes)

            # Batch inference on GPU (YOLO handles letterboxing automatically)
            # Use same image size as training
            results = self.model(valid_crops, verbose=False, device=self.device, imgsz=self.imgsz)

            # Map predictions to color names
            colors = [None] * len(bboxes)  # Default to None for failed classifications

            for i, result_idx in enumerate(valid_indices):
                pred_class_idx = results[i].probs.top1
                pred_class_name = results[i].names[pred_class_idx]
                confidence = results[i].probs.top1conf.item()

                colors[result_idx] = pred_class_name

                logger.debug(
                    f"Traffic light classified: {pred_class_name} "
                    f"(confidence: {confidence:.3f})"
                )

            return colors

        except Exception as e:
            # Graceful degradation - return None for all lights on error
            logger.error(f"YOLO classifier inference failed: {e}", exc_info=True)
            return [None] * len(bboxes)
