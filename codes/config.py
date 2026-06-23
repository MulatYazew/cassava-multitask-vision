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
BATCH_SIZE     = 64          # MPS performs well at 
NUM_EPOCHS     = 30
WARMUP_LR      = 1e-3        # Phase 1: head-only warm-up (backbone frozen)
LEARNING_RATE  = 1e-4        # Phase 2: full fine-tune — must be 10× lower than warm-up
WEIGHT_DECAY   = 1e-5

# num_workers > 0 can hang on macOS with some DataLoader configs; keep at 0.
NUM_WORKERS    = 4

#  Model
PRETRAINED     = True
# Choices: "cassava_cnn" | "efficientnet_v2_s" | "swin_tiny"
MODEL_ARCHITECTURE = "efficientnet_v2_s"

#  Augmentation
AUGMENTATION_INTENSITY = 0.5

# MixUp / CutMix — randomly chooses one per batch during training
# Set either to 0.0 to disable it; both > 0 → random choice each batch.
USE_MIXUP      = True
MIXUP_ALPHA    = 0.4
CUTMIX_ALPHA   = 1.0

#  Loss function
# gamma=3 focuses more aggressively on hard minority cases (CBB, CBSD, CGM)
# than the default gamma=2. Appropriate when WeightedRandomSampler is active.
FOCAL_GAMMA    = 2.0

# Label smoothing: helps regularize against noisy or mislabeled samples.
LABEL_SMOOTHING = 0.1

#  Test-Time Augmentation (TTA)
# At inference, run TTA_STEPS different augmented views and average softmax outputs.
USE_TTA        = True
TTA_STEPS      = 5

#  Early stopping (on val macro-F1, not val loss)
PATIENCE       = 10          # longer patience lets cosine annealing finish its job
MIN_DELTA      = 1e-4