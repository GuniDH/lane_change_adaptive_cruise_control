import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

# Load CARLA configuration for image size (single source of truth)
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "config" / "carla_config.yaml"
with open(config_path, 'r') as f:
    carla_config = yaml.safe_load(f)

# Extract image size from config - must match inference size!
IMAGE_SIZE = carla_config['perception']['traffic_light_classifier']['image_size']

# Dataset root (created by create_dataset.py)
DATASET_ROOT = r"F:\traffic_light_data"

# Where to save trained model weights
OUTPUT_DIR = str(project_root / "models" / "traffic_light_classifier")

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 64  # Reduced batch size for larger image size
LEARNING_RATE = 0.01
PATIENCE = 20  # Early stopping patience

# Model architecture
MODEL_ARCH = "yolo11s-cls.pt"  # YOLO11 small classification (better for tiny crops)

# Device
DEVICE = 0  # GPU 0, use 'cpu' for CPU training

# ============================================================================
# END CONFIGURATION
# ============================================================================


def train_classifier():
    """Train YOLO12n-cls model on traffic light dataset"""

    print("=" * 60)
    print("  Traffic Light Classifier Training")
    print("=" * 60)
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Model: {MODEL_ARCH}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Verify dataset exists
    train_path = Path(DATASET_ROOT) / "train"
    val_path = Path(DATASET_ROOT) / "val"

    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        print("Please run create_dataset.py first!")
        return

    if not val_path.exists():
        print(f"Error: Validation data not found at {val_path}")
        print("Please run create_dataset.py first!")
        return

    # Count images per class
    print("\nDataset Statistics:")
    for split in ['train', 'val']:
        split_path = Path(DATASET_ROOT) / split
        print(f"\n{split.upper()} SET:")
        for class_name in ['red', 'green', 'yellow', 'irrelevant']:
            class_path = split_path / class_name
            if class_path.exists():
                num_images = len(list(class_path.glob("*.png")))
                print(f"  {class_name:12s}: {num_images:>6} images")

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Extract model name for unique output directory (e.g., "yolo11s-cls.pt" -> "yolo11s")
    model_name = MODEL_ARCH.replace("-cls.pt", "").replace(".pt", "")

    # Add timestamp to make each training run unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"traffic_light_cls_{model_name}_img{IMAGE_SIZE}_{timestamp}"

    print("\n" + "=" * 60)
    print("  Starting Training...")
    print("=" * 60 + "\n")
    print(f"Model: {MODEL_ARCH}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Output name: {output_name}")
    print("=" * 60 + "\n")

    # Load YOLO classification model
    model = YOLO(MODEL_ARCH)

    # Train the model (for classification, pass dataset root directly - no yaml needed!)
    results = model.train(
        data=DATASET_ROOT,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        lr0=LEARNING_RATE,
        patience=PATIENCE,
        device=DEVICE,
        workers=8,  # Parallel data loading for GPU efficiency
        project=OUTPUT_DIR,
        name=output_name,
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        plots=True,

        # Augmentation settings
        hsv_h=0.015,  # Hue augmentation (small, traffic lights have consistent colors)
        hsv_s=0.3,    # Saturation augmentation
        hsv_v=0.3,    # Value/brightness augmentation
        degrees=10,   # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.2,    # Scale augmentation
        fliplr=0.5,   # No horizontal flip (traffic lights have orientation)
        flipud=0.0,   # No vertical flip
        mosaic=0.0,   # No mosaic for classification
    )

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"Best weights saved at: {OUTPUT_DIR}/{output_name}/weights/best.pt")
    print(f"Last weights saved at: {OUTPUT_DIR}/{output_name}/weights/last.pt")
    print("=" * 60)

    # Validate the best model
    print("\nValidating best model...")
    best_model = YOLO(f"{OUTPUT_DIR}/{output_name}/weights/best.pt")
    metrics = best_model.val()

    print("\nValidation Metrics:")
    print(f"  Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"  Top-5 Accuracy: {metrics.top5:.4f}")

    return best_model, results


if __name__ == "__main__":
    train_classifier()
