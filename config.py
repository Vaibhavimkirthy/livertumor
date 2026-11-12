import torch
import os

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Weight
MODEL_PATH = "weights/medsam.pth"

# Upload Folder
UPLOAD_FOLDER = "static/uploads"

# Output Folder
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Image Size
IMAGE_SIZE = 1024

# Number of Classes
NUM_CLASSES = 2


# =========================
# Dataset Paths
# =========================

TRAIN_IMAGE_DIR = "datasets/train/images"
TRAIN_MASK_DIR = "datasets/train/masks"

VAL_IMAGE_DIR = "datasets/val/images"
VAL_MASK_DIR = "datasets/val/masks"

TEST_IMAGE_DIR = "datasets/test/images"
TEST_MASK_DIR = "datasets/test/masks"