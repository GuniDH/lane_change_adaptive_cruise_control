"""
Prepare an existing traffic light dataset for YOLO training.

This script performs two operations on an existing dataset:
1. Converts all JPG images to PNG (to prevent format artifact learning)
2. Upscales images smaller than 10px to meet YOLO's minimum requirement

Use this on datasets that already have train/val/class folder structure.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset root (must have train/ and val/ folders with class subfolders)
DATASET_ROOT = r"F:\traffic_light_data"

# Minimum dimension (YOLO requires >= 10, we use 12 for safety margin)
MIN_DIMENSION = 12

# Interpolation method for upscaling (INTER_CUBIC or INTER_LANCZOS4 for best quality)
INTERPOLATION = cv2.INTER_LANCZOS4

# ============================================================================
# END CONFIGURATION
# ============================================================================


def convert_jpg_to_png(jpg_path: Path):
    """
    Convert a JPG image to PNG.

    Args:
        jpg_path: Path to JPG file

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        img = cv2.imread(str(jpg_path))
        if img is None:
            return False

        png_path = jpg_path.with_suffix('.png')
        cv2.imwrite(str(png_path), img)
        jpg_path.unlink()  # Delete original JPG
        return True

    except Exception as e:
        print(f"Error converting {jpg_path}: {e}")
        return False


def upscale_image(image_path: Path, min_dim: int = MIN_DIMENSION):
    """
    Upscale an image if it's smaller than min_dim in any dimension.

    Args:
        image_path: Path to the image
        min_dim: Minimum dimension required

    Returns:
        True if image was upscaled, False otherwise
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    height, width = img.shape[:2]

    # Check if upscaling is needed
    if width >= min_dim and height >= min_dim:
        return False

    # Calculate scale factor to make smallest dimension = min_dim
    scale_factor = min_dim / min(width, height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Upscale using high-quality interpolation
    upscaled = cv2.resize(img, (new_width, new_height), interpolation=INTERPOLATION)
    cv2.imwrite(str(image_path), upscaled)

    return True


def prepare_dataset():
    """
    Prepare dataset for YOLO training:
    1. Convert all JPG to PNG
    2. Upscale images < min_dimension
    """

    print("=" * 80)
    print("  PREPARING EXISTING DATASET FOR YOLO TRAINING")
    print("=" * 80)
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Minimum dimension: {MIN_DIMENSION} pixels")
    print("=" * 80 + "\n")

    dataset_path = Path(DATASET_ROOT)

    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset not found at {DATASET_ROOT}")
        return

    # Check structure
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"

    if not train_path.exists() or not val_path.exists():
        print(f"❌ ERROR: Dataset must have train/ and val/ folders")
        print(f"   Found train/: {train_path.exists()}")
        print(f"   Found val/:   {val_path.exists()}")
        return

    total_converted = 0
    total_upscaled = 0
    total_images = 0

    # Process train and val splits
    for split in ['train', 'val']:
        split_path = dataset_path / split
        print(f"\n{'='*80}")
        print(f"  Processing {split.upper()} set")
        print(f"{'='*80}")

        class_folders = [d for d in split_path.iterdir() if d.is_dir()]

        if not class_folders:
            print(f"⚠️  WARNING: No class folders found in {split}/")
            continue

        for class_folder in class_folders:
            class_name = class_folder.name
            print(f"\n{class_name}:")

            # Step 1: Convert JPG to PNG
            jpg_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg"))

            if jpg_files:
                print(f"  Step 1: Converting {len(jpg_files)} JPG files to PNG...")
                converted = 0

                for jpg_file in tqdm(jpg_files, desc=f"    Converting", leave=False):
                    if convert_jpg_to_png(jpg_file):
                        converted += 1

                total_converted += converted
                print(f"  ✓ Converted {converted}/{len(jpg_files)} files")

            # Step 2: Upscale tiny images
            png_files = list(class_folder.glob("*.png"))
            total_images += len(png_files)

            if png_files:
                print(f"  Step 2: Checking {len(png_files)} PNG files for upscaling...")
                upscaled = 0

                for png_file in tqdm(png_files, desc=f"    Upscaling", leave=False):
                    if upscale_image(png_file, MIN_DIMENSION):
                        upscaled += 1

                total_upscaled += upscaled

                if upscaled > 0:
                    print(f"  ✓ Upscaled {upscaled}/{len(png_files)} images")
                else:
                    print(f"  ✓ All images already meet minimum size")

    # Summary
    print("\n" + "=" * 80)
    print("  PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Total images processed:      {total_images}")
    print(f"Images converted (JPG→PNG):  {total_converted}")
    print(f"Images upscaled (< {MIN_DIMENSION}px):      {total_upscaled}")
    print("=" * 80)

    if total_converted > 0 or total_upscaled > 0:
        print("\n✓ Dataset is now ready for YOLO training!")
        print("\nIMPORTANT: Delete YOLO cache files before training:")
        print(f"  rm \"{DATASET_ROOT}/train.cache\"")
        print(f"  rm \"{DATASET_ROOT}/val.cache\"")
        print("\nThen train with:")
        print("  python scripts/traffic_light_classifier/train.py")
    else:
        print("\n✓ Dataset was already properly formatted!")
        print("  Ready to train: python scripts/traffic_light_classifier/train.py")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    prepare_dataset()
