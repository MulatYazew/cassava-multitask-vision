"""
Configuration file for AgroVision project.
All hyperparameters and seed settings are centralized here.
"""

# Seed for reproducibility
SEED = 42

# Model hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-5

# Model architecture
NUM_CLASSES = 5  # Cassava leaf disease classes
INPUT_SIZE = 224
PRETRAINED = True
MODEL_ARCHITECTURE = "efficientnet_b0"

# Data augmentation
AUGMENTATION_INTENSITY = 0.5
USE_MIXUP = False

# Training settings
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Paths
DATA_DIR = "cassava-leaf-dataset"
MODEL_SAVE_DIR = "models"
RESULTS_DIR = "results"

# Device
DEVICE = "mps"  # or "cpu"

# Early stopping
PATIENCE = 5
MIN_DELTA = 1e-4
