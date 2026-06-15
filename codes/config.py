"""
AgroVision Configuration
All hyperparameters centralized here. Tuned for Apple Silicon (M1/M2/M3/M4).
"""

#  Reproducibility 
SEED = 42

#  Device 
# MPS = Metal Performance Shaders (Apple Silicon GPU). Falls back to CPU.
DEVICE = "mps"

#  Data 
from pathlib import Path as Path

PROJECT_ROOT   = Path(__file__).resolve().parent.parent   # repo root (codes/ is one level down)
DATA_DIR       = PROJECT_ROOT / "cassava-leaf-dataset"
IMAGE_DIR      = DATA_DIR / "train_images"
MODELS_DIR     = PROJECT_ROOT / "models"
RESULTS_DIR    = PROJECT_ROOT / "results"

NUM_CLASSES    = 5
INPUT_SIZE     = 224

TRAIN_SPLIT    = 0.80
VAL_SPLIT      = 0.10
TEST_SPLIT     = 0.10

#  Training 
BATCH_SIZE     = 32          # MPS performs well at 32; raise to 64 if RAM allows
NUM_EPOCHS     = 50
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-5

# num_workers > 0 can hang on macOS with some DataLoader configs; keep at 0.
NUM_WORKERS    = 0

#  Model 
PRETRAINED     = True
# Choices: "resnet50" | "efficientnet_v2_s" | "convnext_tiny"
MODEL_ARCHITECTURE = "efficientnet_v2_s"

#  Augmentation 
AUGMENTATION_INTENSITY = 0.5
USE_MIXUP      = False

#  Early stopping 
PATIENCE       = 5
MIN_DELTA      = 1e-4