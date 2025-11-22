"""
Lane Detector - Ultra-Fast Lane Detection v2 integration.

This module provides lane detection capabilities using the pretrained UFLD-v2 model
to detect lane boundaries in camera images. Returns lanes with 1-based indexing
from leftmost lane.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
import sys
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from scipy.interpolate import CubicSpline
from lane_change.gateway.data_types import CameraImage
from lane_change.gateway.detector import ILaneDetector

# Add UFLD path to import their modules (dynamic path)
# lane_detector.py is in src/lane_change/perception/
# Go up: perception/ -> lane_change/ -> src/ -> lane_change/ -> lane_change_final_proj/
project_root = Path(__file__).parent.parent.parent.parent.parent
ufld_path = str(project_root / "Ultra-Fast-Lane-Detection-v2")
if ufld_path not in sys.path:
    sys.path.insert(0, ufld_path)

# Import the model directly to avoid data dependencies
from utils.config import Config
from utils.common import get_model

logger = logging.getLogger(__name__)


class LaneKalmanTracker:
    """Kalman filter-based tracker for lane polynomial coefficients (smoothing only)."""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        """
        Initialize the Kalman tracker for lanes.

        Args:
            process_noise: How much lanes can change between frames (Q)
            measurement_noise: How much we trust new measurements (R) - higher = more smoothing
        """
        self.Q = process_noise
        self.R = measurement_noise

        self.tracks = {}  # lane_id -> filter state
        self.next_id = 0

    def update(self, lane_id: Optional[int], coeffs: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """
        Update and smooth lane coefficients using Kalman filter.

        Args:
            lane_id: ID of the lane (None for new lanes)
            coeffs: Polynomial coefficients from detection

        Returns:
            Smoothed coefficients and lane ID
        """
        # Handle new lane - initialize without smoothing
        if lane_id is None:
            lane_id = self.next_id
            self.next_id += 1
            self.tracks[lane_id] = {
                'x': coeffs,
                'P': np.eye(len(coeffs)) * 0.1,
                'degree': len(coeffs) - 1
            }
            return coeffs, lane_id

        # Handle active track update with Kalman smoothing
        if lane_id in self.tracks:
            state = self.tracks[lane_id]

            # Check if degree changed - reset if so
            if len(coeffs) != len(state['x']):
                state['x'] = coeffs
                state['P'] = np.eye(len(coeffs)) * 0.1
                state['degree'] = len(coeffs) - 1
            else:
                # Kalman predict step
                x_pred = state['x']
                P_pred = state['P'] + self.Q * np.eye(len(coeffs))

                # Kalman update step (blend prediction with measurement)
                K = P_pred @ np.linalg.inv(P_pred + self.R * np.eye(len(coeffs)))
                state['x'] = x_pred + K @ (coeffs - x_pred)
                state['P'] = (np.eye(len(coeffs)) - K) @ P_pred

            self.tracks[lane_id] = state
            return state['x'], lane_id

        return None, -1


class UFLDLaneDetector(ILaneDetector):
    """Lane detector using Ultra-Fast Lane Detection v2 with Kalman tracking."""

    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None,
                 use_kalman: bool = True, adaptive_confidence: bool = True, use_fp16: bool = True):
        """
        Initialize UFLD lane detector.

        Args:
            model_path: Path to pretrained UFLD model
            config_path: Path to UFLD config file
            use_kalman: Whether to use Kalman filtering for temporal smoothing
            adaptive_confidence: Whether to use adaptive confidence thresholds
            use_fp16: Whether to use FP16 (half precision) for inference
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_kalman = use_kalman
        self.adaptive_confidence = adaptive_confidence
        self.use_fp16 = use_fp16

        # Initialize Kalman tracker (for coefficient smoothing only)
        if self.use_kalman:
            self.kalman_tracker = LaneKalmanTracker(
                process_noise=0.005,
                measurement_noise=0.45
            )

        # Lane ID association (for tracking consistency)
        self.previous_lane_positions = {}  # lane_id -> X position at reference Y
        self.next_lane_id = 0

        # Profiling
        self.frame_count = 0
        self.profiling_interval = 30  # Log detailed timing every N frames
        self.min_required_lanes = 1  # Minimum lanes we expect to see

        # Temporal smoothing for lane count changes
        self.previous_lanes = None
        self.previous_lanes_with_ids = None  # Store lanes with their lane_ids for Kalman tracking
        self.expected_lane_count = 3  # Typical 3 lane boundaries (LATER UPDATED BY THE ACTUAL NUMBER OF LANES DETECTED)
        self.count_change_frames = 0
        self.pending_count = None
        self.count_change_threshold = 5 # consecutive frames to accept change

        # Default paths (dynamic)
        if config_path is None:
            config_path = os.path.join(ufld_path, "configs", "culane_res34.py")
        if model_path is None:
            # ufld_weights is at project root
            model_path = str(project_root / "ufld_weights" / "culane_res34.pth")

        self.cfg = Config.fromfile(config_path)
        self.cfg.test_model = model_path

        # Set anchors like merge_config() does
        if self.cfg.dataset == 'CULane':
            self.cfg.row_anchor = np.linspace(0.42, 1, self.cfg.num_row)  # Back to original - anchors define Y positions, not range
            self.cfg.col_anchor = np.linspace(0, 1, self.cfg.num_col)
        elif self.cfg.dataset == 'Tusimple':
            self.cfg.row_anchor = np.linspace(160, 710, self.cfg.num_row) / 720
            self.cfg.col_anchor = np.linspace(0, 1, self.cfg.num_col)
        elif self.cfg.dataset == 'CurveLanes':
            self.cfg.row_anchor = np.linspace(0.4, 1, self.cfg.num_row)
            self.cfg.col_anchor = np.linspace(0, 1, self.cfg.num_col)

        # Pre-create normalization tensors on GPU for preprocessing optimization
        norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Convert to FP16 if enabled
        if self.use_fp16:
            self.norm_mean = norm_mean.half()
            self.norm_std = norm_std.half()
        else:
            self.norm_mean = norm_mean
            self.norm_std = norm_std

        self.crop_size = self.cfg.train_height

        # Pre-create small index tensors for confidence extraction (local_width=1 means windows of 1-3 elements)
        # These are reused hundreds of times per frame to avoid recreating tensors in loops
        self.index_cache = {
            1: torch.tensor([0], dtype=torch.long),
            2: torch.tensor([0, 1], dtype=torch.long),
            3: torch.tensor([0, 1, 2], dtype=torch.long)
        }

        # Pre-create Sobel kernel for GPU edge detection
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        self.sobel_kernel = sobel_x

        # RGB to grayscale weights (standard luminance formula)
        self.rgb_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=self.device)

        self.model = self._load_model()
        logger.info(f"UFLD lane detector initialized with config: {config_path}")

    def _load_model(self):
        """Load the UFLD model."""
        model = get_model(self.cfg)

        state_dict = torch.load(self.cfg.test_model, map_location=self.device)['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        model.load_state_dict(compatible_state_dict, strict=False)
        model.eval()
        model.to(self.device)

        # Convert to FP16 if enabled
        if self.use_fp16:
            model = model.half()
            logger.info(f"UFLD model loaded on device: {self.device} (FP16)")
        else:
            logger.info(f"UFLD model loaded on device: {self.device} (FP32)")

        return model

    def detect_lanes(self, camera_image: CameraImage) -> List[List[Tuple[int, int]]]:
        """
        Detect lane boundaries in camera image.

        Args:
            camera_image: Camera image data

        Returns:
            List of lanes, each lane is list of (x, y) points from bottom to top.
            Lanes are numbered 1-based from leftmost lane.
        """
        try:
            frame_start = time.perf_counter()
            profiling = (self.frame_count % self.profiling_interval == 0)

            # Convert camera image to tensor
            t0 = time.perf_counter()
            img_tensor, preprocess_timing = self._preprocess_image(camera_image, profiling=profiling)
            preprocess_time = (time.perf_counter() - t0) * 1000

            # Run UFLD inference
            t0 = time.perf_counter()
            inference_timing = {}

            # GPU synchronization before inference (ensure preprocessing is complete)
            t_sync = time.perf_counter()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_timing['pre_sync'] = (time.perf_counter() - t_sync) * 1000

            # Actual model forward pass
            t_forward = time.perf_counter()
            with torch.no_grad():
                pred = self.model(img_tensor)

            # GPU synchronization after inference (ensure completion)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_timing['forward'] = (time.perf_counter() - t_forward) * 1000

            # Total inference time
            inference_time = (time.perf_counter() - t0) * 1000
            inference_timing['total'] = inference_time

            # Convert predictions to lane coordinates (this calls the complex pipeline)
            t0 = time.perf_counter()
            lanes = self._pred2coords(pred, camera_image.width, camera_image.height, camera_image, profiling=profiling)
            pred2coords_time = (time.perf_counter() - t0) * 1000

            total_time = (time.perf_counter() - frame_start) * 1000

            if profiling:
                # Log high-level summary
                logger.debug(
                    f"LANE_DETECTION | Frame {self.frame_count} | "
                    f"Total: {total_time:.1f}ms | "
                    f"Preprocess: {preprocess_time:.1f}ms | "
                    f"Inference: {inference_time:.1f}ms | "
                    f"Pred2Coords: {pred2coords_time:.1f}ms"
                )
                # Log detailed breakdowns if available
                if preprocess_timing:
                    self._log_preprocess_profiling(preprocess_timing)
                if inference_timing:
                    self._log_inference_profiling(inference_timing)

            self.frame_count += 1
            return lanes
        except Exception as e:
            logger.error(f"UFLD lane detection failed: {e}")
            logger.exception("Full traceback:")
            self.frame_count += 1
            return []

    def _preprocess_image(self, camera_image: CameraImage, profiling=False):
        """Preprocess camera image for UFLD model using GPU operations."""
        preprocess_timing = {}

        # Convert numpy array to proper format and transfer to GPU immediately
        t0 = time.perf_counter()
        img_array = camera_image.image_data

        # Combined operation: remove alpha channel + ensure contiguous in one copy
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = np.ascontiguousarray(img_array[:, :, :3])
        elif not img_array.flags['C_CONTIGUOUS']:
            img_array = np.ascontiguousarray(img_array)

        # Convert to tensor, permute, and transfer to GPU as uint8 (smaller transfer)
        # Then convert to float on GPU (faster than converting on CPU)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(self.device, dtype=torch.float32) / 255.0
        preprocess_timing['format_convert'] = (time.perf_counter() - t0) * 1000

        # Apply transforms on GPU: Resize + Normalize
        t0 = time.perf_counter()
        # Resize using bilinear interpolation to match PIL behavior
        # Add batch dimension for F.interpolate
        img_tensor = img_tensor.unsqueeze(0)
        target_height = int(self.cfg.train_height / self.cfg.crop_ratio)
        target_width = self.cfg.train_width
        img_tensor = F.interpolate(
            img_tensor,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )

        # Normalize: (x - mean) / std using pre-created tensors
        img_tensor = (img_tensor - self.norm_mean) / self.norm_std

        # Convert to FP16 if enabled
        if self.use_fp16:
            img_tensor = img_tensor.half()

        preprocess_timing['transforms'] = (time.perf_counter() - t0) * 1000

        # Crop bottom like dataset.py: img[:,:,-self.crop_size:,:]
        t0 = time.perf_counter()
        img_tensor = img_tensor[:, :, -self.crop_size:, :]
        preprocess_timing['crop'] = (time.perf_counter() - t0) * 1000

        # GPU transfer time is now included in format_convert, so this is essentially free
        preprocess_timing['gpu_transfer'] = 0.0

        if profiling:
            return img_tensor, preprocess_timing
        else:
            return img_tensor, None

    def _pred2coords(self, pred, original_width, original_height, camera_image, profiling=False):
        """Convert UFLD predictions to lane coordinates with adaptive confidence."""
        # Timing for profiling
        timing = {}
        pipeline_start = time.perf_counter()

        # Extract predictions
        t0 = time.perf_counter()
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

        max_indices_row = pred['loc_row'].argmax(1).cpu()
        valid_row = pred['exist_row'].argmax(1).cpu()

        max_indices_col = pred['loc_col'].argmax(1).cpu()
        valid_col = pred['exist_col'].argmax(1).cpu()

        pred['loc_row'] = pred['loc_row'].cpu()
        pred['loc_col'] = pred['loc_col'].cpu()
        timing['extract'] = (time.perf_counter() - t0) * 1000

        lanes = []
        local_width = 1

        # Adaptive confidence thresholds based on detection count
        t0 = time.perf_counter()
        if self.adaptive_confidence:
            # Count how many lanes pass the default threshold
            default_row_thresh = num_cls_row / 2
            default_col_thresh = num_cls_col / 4

            detected_count = 0
            for i in [1, 2]:
                if valid_row[0, :, i].sum() > default_row_thresh:
                    detected_count += 1
            for i in [0, 3]:
                if valid_col[0, :, i].sum() > default_col_thresh:
                    detected_count += 1

            # Lower thresholds if too few lanes detected
            if detected_count < self.min_required_lanes:
                row_confidence_threshold = num_cls_row / 4
                col_confidence_threshold = num_cls_col / 6
            else:
                row_confidence_threshold = default_row_thresh
                col_confidence_threshold = default_col_thresh
        else:
            row_confidence_threshold = num_cls_row / 2
            col_confidence_threshold = num_cls_col / 4

        # Row-based lanes (vertical lanes) - extract points with confidence weights
        row_lane_idx = [1, 2]
        for i in row_lane_idx:
            lane_data = []
            if valid_row[0, :, i].sum() > row_confidence_threshold:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        # Compute window bounds
                        max_idx = max_indices_row[0, k, i].item()
                        start = max(0, max_idx - local_width)
                        end = min(num_grid_row - 1, max_idx + local_width)

                        # Use pre-cached index tensor (offset to start position)
                        window_size = end - start + 1
                        indices = self.index_cache[window_size] + start

                        # Get softmax probabilities as confidence weights
                        probs = pred['loc_row'][0, indices, k, i].softmax(0)
                        out_tmp = (probs * indices.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row-1) * original_width

                        # Extract confidence as max probability
                        confidence = float(probs.max())

                        x = int(out_tmp)
                        y = int(self.cfg.row_anchor[k] * original_height)
                        lane_data.append((x, y, confidence))

                if lane_data:
                    lanes.append(lane_data)

        # Column-based lanes (road boundaries) - extract points with confidence weights
        col_lane_idx = [0, 3]
        for i in col_lane_idx:
            lane_data = []
            if valid_col[0, :, i].sum() > col_confidence_threshold:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        # Compute window bounds
                        max_idx = max_indices_col[0, k, i].item()
                        start = max(0, max_idx - local_width)
                        end = min(num_grid_col - 1, max_idx + local_width)

                        # Use pre-cached index tensor (offset to start position)
                        window_size = end - start + 1
                        indices = self.index_cache[window_size] + start

                        # Get softmax probabilities as confidence weights
                        probs = pred['loc_col'][0, indices, k, i].softmax(0)
                        out_tmp = (probs * indices.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_col-1) * original_height

                        # Extract confidence as max probability
                        confidence = float(probs.max())

                        x = int(self.cfg.col_anchor[k] * original_width)
                        y = int(out_tmp)
                        lane_data.append((x, y, confidence))

                if lane_data:
                    lanes.append(lane_data)

        timing['confidence+extraction'] = (time.perf_counter() - t0) * 1000

        # Create edge image for evidence-based gating
        t0 = time.perf_counter()
        edge_img = self._make_edge_image(camera_image.image_data)
        timing['edge_image'] = (time.perf_counter() - t0) * 1000

        # Apply weighted refinement pipeline with spline continuation and collect coefficients
        lanes_with_coeffs = []

        # Collect refined lanes first for Kalman reference_y computation
        t0 = time.perf_counter()
        refined_lanes_before_kalman = []
        for i, lane_data in enumerate(lanes):
            refined, coeffs = self._refine_lane_weighted_spline_with_coeffs(lane_data, original_height, original_width, edge_img)
            if refined:
                refined_lanes_before_kalman.append((refined, coeffs))
        timing['spline_refinement'] = (time.perf_counter() - t0) * 1000

        # Compute reference_y from all refined lanes BEFORE Kalman/trimming
        # This ensures no extrapolation during Kalman matching
        t0 = time.perf_counter()
        if refined_lanes_before_kalman:
            lanes_only = [lane for lane, _ in refined_lanes_before_kalman]
            kalman_reference_y = self._compute_common_reference_y(lanes_only, original_height)
        else:
            kalman_reference_y = int(0.8 * original_height)
        timing['reference_y_1'] = (time.perf_counter() - t0) * 1000

        # Apply Kalman filtering with consistent reference_y
        t0 = time.perf_counter()
        for refined, coeffs in refined_lanes_before_kalman:
            # Compute X position at reference Y for initial spatial matching
            x_at_ref_initial = self._get_lane_x_at_y(refined, kalman_reference_y, original_width)

            # Apply Kalman filtering if enabled
            if self.use_kalman:
                # Match to previous lane based on spatial proximity (using pre-Kalman X)
                matched_id = self._match_lane_to_previous(x_at_ref_initial, proximity_threshold=50)
                smoothed_coeffs, lane_id = self.kalman_tracker.update(matched_id, coeffs)
                if smoothed_coeffs is not None:
                    # Regenerate points from smoothed coefficients
                    refined = self._coeffs_to_points(smoothed_coeffs, original_height, original_width)
                    lanes_with_coeffs.append((refined, smoothed_coeffs, lane_id))
                else:
                    lanes_with_coeffs.append((refined, coeffs, None))
            else:
                lanes_with_coeffs.append((refined, coeffs, None))

        timing['kalman_filtering'] = (time.perf_counter() - t0) * 1000

        # Compute lane positions from UNTRIMMED lanes for Kalman tracking
        # These positions are guaranteed to not require extrapolation since kalman_reference_y
        # was computed from these same untrimmed lanes
        untrimmed_lane_positions = {}
        if self.use_kalman:
            for lane, _, lane_id in lanes_with_coeffs:
                if lane_id is not None:
                    x_pos = self._get_lane_x_at_y(lane, kalman_reference_y, original_width)
                    untrimmed_lane_positions[lane_id] = x_pos

        # Lane prediction/hallucination removed - temporal smoothing handles lane count changes instead

        # Check for low intersections - reject detection if lanes cross in lower 35%
        t0 = time.perf_counter()
        if lanes_with_coeffs and self._has_low_intersection(lanes_with_coeffs, original_height):
            # Bad detection - return previous lanes if available
            if self.previous_lanes is not None:
                logger.debug("Rejecting new lanes due to low intersection, keeping previous lanes")
                if profiling:
                    self._log_profiling(timing)
                return self.previous_lanes
            else:
                # First frame with bad detection - return empty
                logger.debug("First frame has low intersection, returning empty")
                if profiling:
                    self._log_profiling(timing)
                return []

        timing['intersection_check'] = (time.perf_counter() - t0) * 1000

        # Trim lanes at intersection points (fixed to ignore spurious intersections near vehicle)
        t0 = time.perf_counter()
        if lanes_with_coeffs:
            refined_lanes_with_ids = self._trim_lanes_at_intersections(lanes_with_coeffs, original_height)
        else:
            refined_lanes_with_ids = []
        timing['trimming'] = (time.perf_counter() - t0) * 1000

        # Filter out degenerate lanes that cause wrong sorting:
        # 1. < 2 points: Cannot interpolate, fallback uses closest point
        # 2. < 30 pixels vertical range: Causes common range gaps, leads to extrapolation
        t0 = time.perf_counter()
        min_lane_height = 30  # pixels
        valid_lanes_with_ids = [(lane, lid) for lane, lid in refined_lanes_with_ids
                                if lane and len(lane) >= 2 and
                                (max(p[1] for p in lane) - min(p[1] for p in lane)) >= min_lane_height]
        timing['filtering'] = (time.perf_counter() - t0) * 1000

        # Compute reference_y from VALID TRIMMED lanes for sorting to avoid extrapolation
        # This ensures we use Y coordinate where all valid lanes actually exist
        t0 = time.perf_counter()
        if valid_lanes_with_ids:
            trimmed_lanes_only = [lane for lane, _ in valid_lanes_with_ids]
            reference_y = self._compute_common_reference_y(trimmed_lanes_only, original_height)
        else:
            reference_y = int(0.7 * original_height)
        timing['reference_y_2'] = (time.perf_counter() - t0) * 1000

        # Sort lanes by X position at reference Y to maintain left-to-right order
        # This ensures consistent 1-based indexing for vehicle assignment
        t0 = time.perf_counter()
        sorted_lanes_with_ids = self._sort_lanes_left_to_right(valid_lanes_with_ids, original_height, original_width, reference_y)
        timing['sorting'] = (time.perf_counter() - t0) * 1000

        # Temporal smoothing: Validate lane count changes BEFORE updating positions
        t0 = time.perf_counter()
        validated_lanes_with_ids = self._validate_lane_count_change(sorted_lanes_with_ids)
        timing['validation'] = (time.perf_counter() - t0) * 1000

        # Update previous lane positions ONLY if validation ACCEPTED new lanes
        # Uses positions from UNTRIMMED lanes (computed before trimming) to avoid extrapolation
        # Identity check: validation returns same object if accepted, different object if rejected
        t0 = time.perf_counter()
        if self.use_kalman and validated_lanes_with_ids is sorted_lanes_with_ids:
            validated_lane_data = []
            for lane, lane_id in validated_lanes_with_ids:
                if lane_id in untrimmed_lane_positions:
                    x_validated = untrimmed_lane_positions[lane_id]
                    validated_lane_data.append((x_validated, lane_id))
            self._update_previous_positions(validated_lane_data)
        timing['position_update'] = (time.perf_counter() - t0) * 1000

        # Final extraction (extract lanes without IDs)
        t0 = time.perf_counter()
        refined_lanes = [lane for lane, _ in validated_lanes_with_ids]
        timing['final_extraction'] = (time.perf_counter() - t0) * 1000

        # Compute overhead (time not captured by specific stages)
        total_elapsed = (time.perf_counter() - pipeline_start) * 1000
        sum_of_stages = sum(timing.values())
        timing['overhead'] = total_elapsed - sum_of_stages

        # Log detailed profiling if enabled
        if profiling:
            self._log_profiling(timing)

        return refined_lanes

    def _log_preprocess_profiling(self, timing: Dict[str, float]):
        """Log detailed profiling breakdown for preprocessing stage."""
        total = sum(timing.values())
        logger.debug("")
        logger.debug("=" * 80)
        logger.debug(f"PREPROCESSING PROFILING (Frame {self.frame_count})")
        logger.debug("=" * 80)
        logger.debug(f"{'Stage':<30} {'Time (ms)':>10} {'%':>7}")
        logger.debug("-" * 80)

        stage_order = ['format_convert', 'transforms', 'crop', 'gpu_transfer']

        for stage in stage_order:
            if stage in timing:
                time_ms = timing[stage]
                percentage = (time_ms / total * 100) if total > 0 else 0
                logger.debug(f"{stage:<30} {time_ms:>10.2f} {percentage:>6.1f}%")

        logger.debug("-" * 80)
        logger.debug(f"{'TOTAL':<30} {total:>10.2f} {100.0:>6.1f}%")
        logger.debug("=" * 80)

    def _log_inference_profiling(self, timing: Dict[str, float]):
        """Log detailed profiling breakdown for inference stage."""
        total = timing.get('total', sum(timing.values()))
        precision = "FP16" if self.use_fp16 else "FP32"
        logger.debug("")
        logger.debug("=" * 80)
        logger.debug(f"INFERENCE PROFILING (Frame {self.frame_count}) - {precision}")
        logger.debug("=" * 80)
        logger.debug(f"{'Stage':<30} {'Time (ms)':>10} {'%':>7}")
        logger.debug("-" * 80)

        stage_order = ['pre_sync', 'forward']

        for stage in stage_order:
            if stage in timing:
                time_ms = timing[stage]
                percentage = (time_ms / total * 100) if total > 0 else 0
                logger.debug(f"{stage:<30} {time_ms:>10.2f} {percentage:>6.1f}%")

        logger.debug("-" * 80)
        logger.debug(f"{'TOTAL':<30} {total:>10.2f} {100.0:>6.1f}%")
        logger.debug("=" * 80)

    def _log_profiling(self, timing: Dict[str, float]):
        """Log detailed profiling breakdown for lane detection pipeline."""
        total = sum(timing.values())
        logger.debug("")
        logger.debug("=" * 80)
        logger.debug(f"PRED2COORDS PIPELINE PROFILING (Frame {self.frame_count})")
        logger.debug("=" * 80)
        logger.debug(f"{'Stage':<30} {'Time (ms)':>10} {'%':>7}")
        logger.debug("-" * 80)

        # Order stages by typical execution order
        stage_order = [
            'extract',
            'confidence+extraction',
            'edge_image',
            'spline_refinement',
            'reference_y_1',
            'kalman_filtering',
            'intersection_check',
            'trimming',
            'filtering',
            'reference_y_2',
            'sorting',
            'validation',
            'position_update',
            'final_extraction',
            'overhead'
        ]

        for stage in stage_order:
            if stage in timing:
                time_ms = timing[stage]
                percentage = (time_ms / total * 100) if total > 0 else 0
                logger.debug(f"{stage:<30} {time_ms:>10.2f} {percentage:>6.1f}%")

        logger.debug("-" * 80)
        logger.debug(f"{'TOTAL':<30} {total:>10.2f} {100.0:>6.1f}%")
        logger.debug("=" * 80)

    def _clean_lane_points(self, pts: List[Tuple[int, int]], max_dx: int = 80, min_pts: int = 6) -> List[Tuple[int, int]]:
        """Remove spikes to reduce zig-zag lines."""
        if len(pts) < min_pts:
            return []

        # Convert to numpy for vectorized coordinate access
        pts_array = np.array(pts)
        x_coords = pts_array[:, 0]

        # Sequential filtering with vectorized coordinate access
        kept_indices = [0]
        last_kept_x = x_coords[0]

        for i in range(1, len(pts)):
            if abs(x_coords[i] - last_kept_x) <= max_dx:
                kept_indices.append(i)
                last_kept_x = x_coords[i]

        if len(kept_indices) < min_pts:
            return []

        return [pts[i] for i in kept_indices]

    def _make_edge_image(self, bgr_image: np.ndarray) -> np.ndarray:
        """Create edge image for evidence-based lane gating using GPU acceleration."""
        # Ensure contiguous array for torch conversion (CARLA arrays may have negative strides)
        if not bgr_image.flags['C_CONTIGUOUS']:
            bgr_image = np.ascontiguousarray(bgr_image)

        # Convert to tensor and transfer to GPU
        img_tensor = torch.from_numpy(bgr_image).float().to(self.device)

        # Convert RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        gray = (img_tensor * self.rgb_weights).sum(dim=2)

        # Add batch and channel dimensions for conv2d: [H, W] -> [1, 1, H, W]
        gray = gray.unsqueeze(0).unsqueeze(0)

        # Apply Sobel-x filter
        edge = F.conv2d(gray, self.sobel_kernel, padding=1)

        # Take absolute value and remove batch/channel dimensions
        edge = edge.abs().squeeze(0).squeeze(0)

        # Convert back to numpy uint8 for downstream processing
        edge = edge.cpu().numpy().astype(np.uint8)

        return edge

    def _detect_curve_from_points(self, points: List[Tuple[float, float]]) -> bool:
        """Detect if lane points represent a curve using vectorized operations."""
        if len(points) < 3:
            return False

        # Convert to numpy array for vectorized operations
        pts = np.array(points)
        x, y = pts[:, 0], pts[:, 1]

        # Compute segment vectors: [p_i -> p_{i+1}]
        dx1 = x[1:-1] - x[:-2]
        dy1 = y[1:-1] - y[:-2]
        dx2 = x[2:] - x[1:-1]
        dy2 = y[2:] - y[1:-1]

        # Cross product magnitude (vectorized)
        cross = np.abs(dx1 * dy2 - dy1 * dx2)

        # Norms (vectorized)
        norm1 = np.sqrt(dx1**2 + dy1**2)
        norm2 = np.sqrt(dx2**2 + dy2**2)
        norm_product = norm1 * norm2

        # Compute angles where norm > 0
        valid_mask = norm_product > 0
        if not np.any(valid_mask):
            return False

        angles = cross[valid_mask] / norm_product[valid_mask]

        # Average angle change
        avg_angle = np.mean(angles)
        return avg_angle > 0.05

    def _snap_and_gate(self, samples: List[Tuple[float, float]], edge_img: np.ndarray,
                      radius: int = 10, peak_thresh: float = 15, max_no: int = 20,
                      W: int = 640, is_curve: bool = False):
        """Snap to edge peaks and gate by evidence with curve-aware thresholds (vectorized)."""
        if not samples:
            return []

        H = edge_img.shape[0]

        # Adjust thresholds for curves
        if is_curve:
            peak_thresh = 10
            max_no = 30

        # Convert samples to numpy arrays for vectorized operations
        samples_arr = np.array(samples)
        x_vals, y_vals = samples_arr[:, 0], samples_arr[:, 1]

        # Clip coordinates
        yy = np.clip(np.round(y_vals).astype(int), 0, H-1)
        xx = np.clip(np.round(x_vals).astype(int), 0, W-1)

        # Process in vectorized chunks where possible, but maintain early termination
        kept = []
        no_cnt = 0

        for i in range(len(samples)):
            xl = max(0, xx[i] - radius)
            xr = min(W-1, xx[i] + radius)

            roi = edge_img[yy[i], xl:xr+1]
            j = int(np.argmax(roi))
            peak = float(roi[j])

            x_snap = (xl + j) if peak >= peak_thresh else xx[i]
            kept.append((x_snap, y_vals[i]))

            if peak < peak_thresh:
                no_cnt += 1
                if no_cnt >= max_no:
                    return kept[:-max_no]
            else:
                no_cnt = 0

        return kept

    def _refine_lane_weighted_spline_with_coeffs(self, lane_data: List[Tuple[int, int, float]], H: int, W: int, edge_img: np.ndarray,
                                               max_dx: int = 80, min_pts: int = 6, start_frac: float = 0.70,
                                               stop_frac: float = 0.40) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Weighted spline-based lane refinement that also returns polynomial coefficients."""
        if not lane_data:
            return [], np.array([])

        # Extract points and weights
        points = [(x, y) for x, y, _ in lane_data]
        weights = np.array([w for _, _, w in lane_data])

        # Sort by y coordinate and clean spikes
        sorted_data = sorted(zip(points, weights), key=lambda x: x[0][1])
        points = [p for p, _ in sorted_data]
        weights = np.array([w for _, w in sorted_data])

        points = self._clean_lane_points(points, max_dx=max_dx, min_pts=min_pts)
        if len(points) < min_pts:
            return [], np.array([])

        # Recompute weights for cleaned points (simple approach: uniform if cleaning removed points)
        if len(points) != len(weights):
            weights = np.ones(len(points))

        ys = np.array([p[1] for p in points], dtype=float)
        xs = np.array([p[0] for p in points], dtype=float)

        # Weighted polynomial fit (degree 2 or 3 based on fitting quality)
        coeffs = self._fit_weighted_polynomial(ys, xs, weights)

        # Dense resampling for base section
        y_bot = max(float(ys.max()), start_frac * H)
        y_top = float(stop_frac * H)

        if y_bot <= y_top:
            return [], coeffs

        y_dense = np.linspace(y_bot, y_top, int((y_bot - y_top) / 4))
        x_dense = np.polyval(coeffs, y_dense)
        x_dense = np.clip(x_dense, 0, W-1)

        # Spline continuation toward horizon with evidence gating
        y_extend = np.linspace(y_top, 0, int(y_top / 6))
        if len(y_extend) > 0:
            # Create cubic spline for smooth continuation
            try:
                spline = CubicSpline(y_dense, x_dense, bc_type='natural')
                x_extend = spline(y_extend)
                x_extend = np.clip(x_extend, 0, W-1)

                # Combine base and extension
                y_all = np.concatenate([y_dense, y_extend])
                x_all = np.concatenate([x_dense, x_extend])

                # Apply evidence gating to extension only
                base_pts = list(zip(x_dense, y_dense))
                ext_pts = list(zip(x_extend, y_extend))

                # Check if this is a curve
                is_curve = self._detect_curve_from_points(base_pts)
                ext_pts = self._snap_and_gate(ext_pts, edge_img, radius=10, W=W, is_curve=is_curve)

                all_pts = base_pts + ext_pts

            except Exception:
                # Fallback to polynomial extrapolation
                all_pts = list(zip(x_dense, y_dense))
        else:
            all_pts = list(zip(x_dense, y_dense))

        return [(int(x), int(y)) for x, y in all_pts], coeffs

    def _estimate_curvature(self, ys: np.ndarray, xs: np.ndarray) -> float:
        """Estimate curvature from lane points."""
        if len(ys) < 3:
            return 0.0

        # Fit a preliminary polynomial to estimate curvature
        try:
            coeffs = np.polyfit(ys, xs, 2)

            # Calculate second derivative (curvature indicator)
            # For x = ay^2 + by + c, d2x/dy2 = 2a
            curvature = abs(2 * coeffs[0])

            # Also check point-to-point variation
            if len(ys) > 5:
                # Calculate local curvatures
                local_curves = []
                for i in range(2, len(ys) - 2):
                    # Five-point numerical derivative
                    d2x = xs[i-2] - 2*xs[i-1] + 2*xs[i+1] - xs[i+2]
                    dy = ys[i+1] - ys[i-1]
                    if dy > 0:
                        local_curves.append(abs(d2x) / (dy * dy))

                if local_curves:
                    curvature = max(curvature, np.median(local_curves))

            return curvature

        except Exception:
            return 0.0

    def _fit_weighted_polynomial(self, ys: np.ndarray, xs: np.ndarray, weights: np.ndarray,
                                 adaptive_degree: bool = True) -> np.ndarray:
        """Fit weighted polynomial with curvature-based degree selection."""
        try:
            if adaptive_degree:
                # Estimate curvature to choose degree
                curvature = self._estimate_curvature(ys, xs)

                if curvature > 0.002:  # High curvature
                    max_degree = min(4, len(ys) - 1)  # Use degree 4 for curves
                elif curvature > 0.001:  # Moderate curvature
                    max_degree = min(3, len(ys) - 1)
                else:  # Low curvature
                    max_degree = min(2, len(ys) - 1)
            else:
                max_degree = min(3, len(ys) - 1)
            max_degree=2
            # Try fitting with selected degree
            if max_degree >= 2:
                coeffs = np.polyfit(ys, xs, max_degree, w=weights)
                return coeffs

            # Fallback to degree 2
            return np.polyfit(ys, xs, 2, w=weights)

        except Exception:
            # Fallback to simple unweighted fit
            return np.polyfit(ys, xs, 2)

    def _find_polynomial_intersection(self, coeffs_i: np.ndarray, coeffs_j: np.ndarray,
                                      y_min: float, y_max: float, filter_spurious: bool = True) -> Optional[float]:
        """
        Find the y-coordinate where two polynomial lanes intersect.

        Args:
            coeffs_i: Polynomial coefficients for lane i (x = poly(y))
            coeffs_j: Polynomial coefficients for lane j (x = poly(y))
            y_min: Minimum y bound (top of image/horizon)
            y_max: Maximum y bound (bottom of image)
            filter_spurious: Whether to filter spurious intersections in lower 40%

        Returns:
            Y-coordinate of intersection, or None if no valid intersection
        """
        # Subtract polynomials to get difference polynomial
        # Pad coefficients to same length
        max_len = max(len(coeffs_i), len(coeffs_j))
        coeffs_i_pad = np.pad(coeffs_i, (max_len - len(coeffs_i), 0), 'constant')
        coeffs_j_pad = np.pad(coeffs_j, (max_len - len(coeffs_j), 0), 'constant')

        # Difference polynomial: poly_i(y) - poly_j(y) = 0
        diff_coeffs = coeffs_i_pad - coeffs_j_pad

        # Find roots of the difference polynomial
        roots = np.roots(diff_coeffs)

        # Filter for real roots within the valid y range
        valid_roots = []
        for root in roots:
            if np.isreal(root):
                y = float(np.real(root))
                if y_min <= y <= y_max:
                    valid_roots.append(y)

        if not valid_roots:
            return None

        # Find highest Y value
        highest_y = max(valid_roots)

        # Optionally filter spurious intersections in lower 40%
        if filter_spurious:
            H = y_max
            if highest_y > 3 * H / 5 and len(valid_roots) > 1:
                # Find second highest
                second_highest = max(y for y in valid_roots if y != highest_y)
                return second_highest

        return highest_y

    def _has_low_intersection(self, lanes_with_coeffs: List[Tuple[List[Tuple[int, int]], np.ndarray, Optional[int]]],
                              image_height: int) -> bool:
        """
        Check if any lane pairs intersect in the lower 35% of the image.

        This indicates a bad detection where lanes cross near the vehicle,
        which shouldn't happen in normal driving scenarios.

        Args:
            lanes_with_coeffs: List of (lane_points, polynomial_coefficients, lane_id) tuples
            image_height: Image height

        Returns:
            True if any pair intersects at Y >= 0.65*image_height, False otherwise
        """
        if len(lanes_with_coeffs) < 2:
            return False

        low_threshold = 0.65 * image_height

        # Check all pairs of lanes
        for i in range(len(lanes_with_coeffs)):
            for j in range(i + 1, len(lanes_with_coeffs)):
                _, coeffs_i, _ = lanes_with_coeffs[i]
                _, coeffs_j, _ = lanes_with_coeffs[j]

                # Find intersection without filtering spurious ones
                intersection_y = self._find_polynomial_intersection(
                    coeffs_i, coeffs_j, 0, image_height, filter_spurious=False
                )

                # If intersection exists in lower 35%, reject
                if intersection_y is not None and intersection_y >= low_threshold:
                    logger.debug(f"Low intersection detected at Y={intersection_y:.1f} (threshold={low_threshold:.1f})")
                    return True

        return False

    def _trim_lanes_at_intersections(self, lanes_with_coeffs: List[Tuple[List[Tuple[int, int]], np.ndarray, Optional[int]]],
                                     H: int) -> List[Tuple[List[Tuple[int, int]], Optional[int]]]:
        """
        Trim lanes at their intersection points to handle merging/splitting.

        Args:
            lanes_with_coeffs: List of (lane_points, polynomial_coefficients, lane_id) tuples
            H: Image height

        Returns:
            List of (trimmed_lane_points, lane_id) tuples
        """
        if len(lanes_with_coeffs) < 2:
            return [(lane, lane_id) for lane, _, lane_id in lanes_with_coeffs]

        # Find all triplets (lane_i, lane_j, intersection_y)
        triplets = []
        for i, (lane_i, coeffs_i, _) in enumerate(lanes_with_coeffs):
            best_j = None
            best_intersection_y = -1

            # Find lane j with highest intersection y
            for j, (lane_j, coeffs_j, _) in enumerate(lanes_with_coeffs):
                if i == j:
                    continue

                intersection_y = self._find_polynomial_intersection(
                    coeffs_i, coeffs_j, 0, H
                )

                if intersection_y is not None and intersection_y > best_intersection_y:
                    best_intersection_y = intersection_y
                    best_j = j

            if best_j is not None:
                triplets.append((i, best_j, best_intersection_y))

        # Create a copy of lanes for trimming (preserve lane_ids)
        trimmed_lanes = [list(lane) for lane, _, _ in lanes_with_coeffs]
        lane_ids = [lane_id for _, _, lane_id in lanes_with_coeffs]

        # Process each triplet and trim lanes
        for i, j, y_intersection in triplets:
            # Trim lane i: remove points where y < y_intersection
            trimmed_lanes[i] = [
                (x, y) for x, y in trimmed_lanes[i]
                if y >= y_intersection
            ]

            # Trim lane j: remove points where y < y_intersection
            trimmed_lanes[j] = [
                (x, y) for x, y in trimmed_lanes[j]
                if y >= y_intersection
            ]

        # Filter out empty lanes and return with lane_ids
        return [(lane, lane_ids[i]) for i, lane in enumerate(trimmed_lanes) if lane]

    def _coeffs_to_points(self, coeffs: np.ndarray, H: int, W: int,
                         start_frac: float = 0.95, stop_frac: float = 0.35) -> List[Tuple[int, int]]:
        """Convert polynomial coefficients back to lane points.
        Extended range to ensure coverage for vehicle assignment."""
        # Extended range: from very bottom (0.95) to upper region (0.35)
        # This ensures we have points where vehicles are checked
        y_bot = start_frac * H  # Near bottom of image
        y_top = stop_frac * H   # Upper portion

        if y_bot <= y_top:
            return []

        # Generate dense points along the polynomial
        num_points = int((y_bot - y_top) / 3)  # More points for better coverage
        y_values = np.linspace(y_bot, y_top, num_points)
        x_values = np.polyval(coeffs, y_values)
        x_values = np.clip(x_values, 0, W - 1)

        return [(int(x), int(y)) for x, y in zip(x_values, y_values)]

    def _match_lane_to_previous(self, current_x: float, proximity_threshold: float = 50) -> Optional[int]:
        """
        Match current lane to previous frame's lane based on spatial proximity.

        Args:
            current_x: X position of current lane at reference Y
            proximity_threshold: Maximum distance in pixels to consider a match

        Returns:
            Lane ID of matched previous lane, or None for new lane
        """
        if not self.previous_lane_positions:
            return None

        best_match_id = None
        best_distance = proximity_threshold

        for lane_id, prev_x in self.previous_lane_positions.items():
            distance = abs(current_x - prev_x)
            if distance < best_distance:
                best_distance = distance
                best_match_id = lane_id

        return best_match_id

    def _update_previous_positions(self, current_lane_data: List[Tuple[float, Optional[int]]]):
        """
        Update stored lane positions for next frame's matching.

        Args:
            current_lane_data: List of (x_position, lane_id) tuples for current lanes
        """
        self.previous_lane_positions.clear()

        for x_pos, lane_id in current_lane_data:
            if lane_id is not None:
                self.previous_lane_positions[lane_id] = x_pos

    def _compute_common_reference_y(self, lanes: List[List[Tuple[int, int]]], image_height: int) -> int:
        """
        Compute a reference Y coordinate that minimizes extrapolation.

        Finds the middle point between the highest min_y and lowest max_y among all lanes.
        This works optimally for both common ranges and gaps, minimizing maximum extrapolation.

        Args:
            lanes: List of lane point lists
            image_height: Image height for fallback

        Returns:
            Y coordinate that minimizes extrapolation for all lanes
        """
        if not lanes:
            return int(0.7 * image_height)

        max_min_y = 0  # Highest minimum Y among all lanes
        min_max_y = image_height  # Lowest maximum Y among all lanes

        for lane in lanes:
            if lane and len(lane) >= 2:
                y_values = [p[1] for p in lane]
                lane_min_y = min(y_values)
                lane_max_y = max(y_values)
                max_min_y = max(max_min_y, lane_min_y)
                min_max_y = min(min_max_y, lane_max_y)

        # Always use middle formula - works for both common range and gaps
        # Common range: returns middle of overlap (no extrapolation)
        # Gap: returns middle of gap (minimizes maximum extrapolation distance)
        return int((max_min_y + min_max_y) / 2)

    def _get_lane_x_at_y(self, lane_points: List[Tuple[int, int]], target_y: int, image_width: int) -> float:
        """
        Get X coordinate of lane at a specific Y position.

        Args:
            lane_points: List of (x, y) tuples
            target_y: Y coordinate to find X at
            image_width: Image width for fallback

        Returns:
            X coordinate at target_y, using interpolation or closest point
        """
        if not lane_points:
            return image_width / 2

        x_at_y = self.interpolate_lane_x(lane_points, target_y, image_width)

        if x_at_y is not None:
            return float(x_at_y)

        # If interpolation fails, use X from point closest to target Y
        # This ensures consistent spatial-based sorting
        closest_point = min(lane_points, key=lambda p: abs(p[1] - target_y))
        return float(closest_point[0])

    def _sort_lanes_left_to_right(self, lanes_with_ids: List[Tuple[List[Tuple[int, int]], Optional[int]]],
                                   image_height: int, image_width: int, reference_y: int) -> List[Tuple[List[Tuple[int, int]], Optional[int]]]:
        """Sort lanes from left to right based on X position at provided reference Y."""
        if not lanes_with_ids:
            return lanes_with_ids

        lanes_with_x = []
        for lane, lane_id in lanes_with_ids:
            if lane:
                x_at_ref = self._get_lane_x_at_y(lane, reference_y, image_width)
                lanes_with_x.append((lane, lane_id, x_at_ref))

        lanes_with_x.sort(key=lambda x: x[2])

        return [(lane, lane_id) for lane, lane_id, _ in lanes_with_x]

    def _validate_lane_count_change(self, new_lanes_with_ids: List[Tuple[List[Tuple[int, int]], Optional[int]]]) -> List[Tuple[List[Tuple[int, int]], Optional[int]]]:
        """
        Validate lane count changes using temporal smoothing.

        Prevents rapid jitter in lane count by requiring consistent detection
        over multiple consecutive frames before accepting a change.

        Args:
            new_lanes_with_ids: Newly detected lanes with their lane_ids

        Returns:
            Validated lanes with ids (either new lanes if accepted, or previous lanes if rejected)
        """
        current_count = len(new_lanes_with_ids)

        # First frame - accept immediately
        if self.previous_lanes_with_ids is None:
            self.previous_lanes_with_ids = new_lanes_with_ids
            self.previous_lanes = [lane for lane, _ in new_lanes_with_ids]
            self.expected_lane_count = current_count
            self.count_change_frames = 0
            self.pending_count = None
            return new_lanes_with_ids

        # Same count as expected - accept immediately
        if current_count == self.expected_lane_count:
            self.previous_lanes_with_ids = new_lanes_with_ids
            self.previous_lanes = [lane for lane, _ in new_lanes_with_ids]
            self.count_change_frames = 0
            self.pending_count = None
            return new_lanes_with_ids

        # Count changed - start or continue tracking
        if current_count != self.expected_lane_count:
            if self.pending_count == current_count:
                # Same new count as before - increment counter
                self.count_change_frames += 1

                if self.count_change_frames >= self.count_change_threshold:
                    # Accept the change!
                    logger.debug(f"Lane count change accepted: {self.expected_lane_count} → {current_count} (after {self.count_change_frames} frames)")
                    self.expected_lane_count = current_count
                    self.previous_lanes_with_ids = new_lanes_with_ids
                    self.previous_lanes = [lane for lane, _ in new_lanes_with_ids]
                    self.count_change_frames = 0
                    self.pending_count = None
                    return new_lanes_with_ids
                else:
                    # Still waiting - return previous lanes
                    logger.debug(f"Lane count change pending: {current_count} lanes ({self.count_change_frames}/{self.count_change_threshold} frames)")
                    return self.previous_lanes_with_ids
            else:
                # Different new count - restart counter
                self.pending_count = current_count
                self.count_change_frames = 1
                logger.debug(f"Lane count change detected: {self.expected_lane_count} → {current_count}, tracking...")
                return self.previous_lanes_with_ids

        # Shouldn't reach here, but return new lanes as fallback
        return new_lanes_with_ids

    def interpolate_lane_x_presorted(self, sorted_lane_points: List[Tuple[int, int]], y_target: int, image_width: int = 640) -> Optional[int]:
        """
        Interpolate X coordinate of lane at given Y position using pre-sorted points.

        This is the optimized version that assumes lane_points are already sorted by Y.
        Use this when lane points have been sorted once and cached.

        Args:
            sorted_lane_points: List of (x, y) points already sorted by Y coordinate
            y_target: Y coordinate to interpolate X at
            image_width: Image width for clipping extrapolated values

        Returns:
            X coordinate at y_target, or None if not possible
        """
        if not sorted_lane_points or len(sorted_lane_points) < 2:
            return None

        # Find two points that bracket the target Y
        for i in range(len(sorted_lane_points) - 1):
            y1, y2 = sorted_lane_points[i][1], sorted_lane_points[i + 1][1]

            # Check if y_target is between these points
            if min(y1, y2) <= y_target <= max(y1, y2):
                x1, x2 = sorted_lane_points[i][0], sorted_lane_points[i + 1][0]

                # Linear interpolation
                if y2 == y1:
                    return x1

                t = (y_target - y1) / (y2 - y1)
                x_interpolated = x1 + t * (x2 - x1)
                return int(x_interpolated)

        return None

    def interpolate_lane_x(self, lane_points: List[Tuple[int, int]], y_target: int, image_width: int = 640) -> Optional[int]:
        """
        Interpolate X coordinate of lane at given Y position.
        Enhanced to handle trimmed lanes and edge cases better.

        Args:
            lane_points: List of (x, y) points defining the lane
            y_target: Y coordinate to interpolate X at
            image_width: Image width for clipping extrapolated values

        Returns:
            X coordinate at y_target, or None if not possible
        """
        if not lane_points or len(lane_points) < 2:
            return None

        # Sort points by Y coordinate to ensure proper ordering
        sorted_points = sorted(lane_points, key=lambda p: p[1])

        # Find two points that bracket the target Y
        for i in range(len(sorted_points) - 1):
            y1, y2 = sorted_points[i][1], sorted_points[i + 1][1]

            # Check if y_target is between these points
            if min(y1, y2) <= y_target <= max(y1, y2):
                x1, x2 = sorted_points[i][0], sorted_points[i + 1][0]

                # Linear interpolation
                if y2 == y1:
                    return x1

                t = (y_target - y1) / (y2 - y1)
                x_interpolated = x1 + t * (x2 - x1)
                return int(x_interpolated)

        # If y_target is outside the range, extrapolate or return closest
        min_y = min(p[1] for p in sorted_points)
        max_y = max(p[1] for p in sorted_points)

        if y_target < min_y:
            # Extrapolate upward (toward horizon) using first two points
            if len(sorted_points) >= 2:
                y1, y2 = sorted_points[0][1], sorted_points[1][1]
                x1, x2 = sorted_points[0][0], sorted_points[1][0]
                if y2 != y1:
                    slope = (x2 - x1) / (y2 - y1)
                    x_extrapolated = x1 + slope * (y_target - y1)
                    return int(np.clip(x_extrapolated, 0, image_width - 1))
            return sorted_points[0][0]
        elif y_target > max_y:
            # Extrapolate downward (toward vehicle) using last two points
            if len(sorted_points) >= 2:
                y1, y2 = sorted_points[-2][1], sorted_points[-1][1]
                x1, x2 = sorted_points[-2][0], sorted_points[-1][0]
                if y2 != y1:
                    slope = (x2 - x1) / (y2 - y1)
                    x_extrapolated = x2 + slope * (y_target - y2)
                    return int(np.clip(x_extrapolated, 0, image_width - 1))
            return sorted_points[-1][0]
        return None