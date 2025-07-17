"""
Configuration and default hyper-parameters for ShipHullGAN.
"""

import os
import torch

# -------------------------------------------------------------------
# 1) Data paths
# -------------------------------------------------------------------

# Directory containing your raw .stl files (override with env var)
ROOT = os.path.dirname(os.path.abspath(__file__))

# Replace your existing lines with:

DEFAULT_STL_DIR = os.getenv(
    "SHIPHULL_STL_DIR",
    os.path.join(ROOT, "ShipHullDataSet")
)




CHECKPOINT_DIR = os.getenv(
    "SHIPHULL_CHECKPOINT_DIR",
    os.path.join(ROOT, "checkpoints")
)
LOG_DIR = os.getenv(
    "SHIPHULL_LOG_DIR",
    os.path.join(ROOT, "logs")
)


# -------------------------------------------------------------------
# 2) SST geometry constants
# -------------------------------------------------------------------

# Number of points per cross-section
POINTS_PER_CS = 25

# Number of cross-sections (stations) along the hull
NUM_SECTIONS  = 56

# If you pad/crop SSTs to 64×64:
SST_TARGET_SIZE = (64, 64)
SST_PAD = (4, 3, 20, 19)  # matches F.pad(x, (4,3,20,19))

# -------------------------------------------------------------------
# 3) Training hyper-parameters
# -------------------------------------------------------------------

BATCH_SIZE    = 16
LATENT_DIM    = 20
EPOCHS        = 300
LEARNING_RATE = 2e-4
BETAS         = (0.5, 0.999)

# DataLoader settings
NUM_WORKERS = 2
PIN_MEMORY  = True

# -------------------------------------------------------------------
# 4) Miscellaneous
# -------------------------------------------------------------------

# Seed for reproducibility
SEED = 42

# Device for training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# def ensure_dirs():
#     """Call this at startup to create missing folders."""
#     for path in (CHECKPOINT_DIR, LOG_DIR):
#         os.makedirs(path, exist_ok=True)


# # If you want to auto-create on import:
# ensure_dirs()
