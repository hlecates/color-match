#!/usr/bin/env python3
"""
Global configuration file for color-match project.
All command line arguments are converted to constants here.
"""

from pathlib import Path

# Base paths (relative to project root)
BASE = Path("../data/polyvore_outfits")
DATA_ROOT = Path("../data/polyvore_outfits")
IMAGES_DIR = DATA_ROOT / "images"
DISJOINT_DIR = DATA_ROOT / "disjoint"
CATEGORIES_CSV = DATA_ROOT / "categories.csv"
META_PATH = DATA_ROOT / "polyvore_item_metadata.json"

# Output paths (relative to project root)
OUTPUT_DIR = Path("../outputs")
INTERMEDIATE_PAIRS_DIR = Path("../data/intermediate_pairs")
SEG_ARTIFACTS_DIR = Path("../artifacts/seg")

# ============================================================================
# make_pairs.py configuration
# ============================================================================
MAKE_PAIRS_CONFIG = {
    "data_root": str(DATA_ROOT),
    "split": "train",           # choices: ["train", "valid", "test"]
    "neg_per_pos": 1.0,         # Number of negative pairs per positive pair
    "allowed_pairs": "",        # e.g. 'top:bottom,top:shoes,dress:shoes' (optional)
    "hard_negatives": False,    # Prefer color-similar cross-outfit negatives (needs Pillow)
    "seed": 42,                 # Random seed
    "items_out": "",            # Optional path to write unique items list (txt)
}

# ============================================================================
# prepare_masks.py configuration
# ============================================================================
PREPARE_MASKS_CONFIG = {
    "images": "",               # ModaNet images dir
    "ann": "",                  # instances_[train|val].json
    "out_images": "",           # Output images directory
    "out_masks": "",            # Output masks directory
}

# ============================================================================
# preprocess_wb_shading.py configuration
# ============================================================================
PREPROCESS_WB_SHADING_CONFIG = {
    "items_list": "",           # Path to items list file
    "masks_dir": "",            # Directory containing masks
    "out_dir": "",              # Output directory
    "size": 256,                # Output image size
    "wb": "grayworld",          # choices: ["grayworld", "off"]
    "shading": "on",            # choices: ["on", "off"]
}

# ============================================================================
# segment_pairs.py configuration
# ============================================================================
SEGMENT_PAIRS_CONFIG = {
    "items_list": "",           # Paths to item images (one per line)
    "onnx": "",                 # SEGMENTATION.onnx path
    "out_dir": "",              # Output directory
    "size": 384,                # Input size for segmentation model
}

# ============================================================================
# export_to_onnx.py configuration
# ============================================================================
EXPORT_TO_ONNX_CONFIG = {
    "ckpt": "",                 # Checkpoint path
    "out": str(SEG_ARTIFACTS_DIR / "SEGMENTATION.onnx"),  # Output ONNX path
    "size": 384,                # Model input size
}

# ============================================================================
# segment/train.py configuration
# ============================================================================
SEGMENT_TRAIN_CONFIG = {
    "images": "",                 # Path to images directory
    "masks": "",                  # Path to masks directory
    "epochs": 25,                 # Number of training epochs
    "batch": 16,                  # Batch size
    "size": 384,                  # Input image size
    "lr": 2e-4,                   # Learning rate
    "out": str(SEG_ARTIFACTS_DIR), # Output directory for checkpoints
}
