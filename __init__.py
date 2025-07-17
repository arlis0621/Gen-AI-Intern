"""
shiphullgan: A generic parametric modeller for ship hull design using GANs.

Submodules
----------
config   — Project configuration and hyper‑parameters
sst      — Shape‑Signature‑Tensor pipeline
dataset  — PyTorch Dataset wrapper for SSTs
models   — Generator & Discriminator architectures
utils    — Helper functions and losses
train    — Training entry‑point
"""

__version__ = "0.1.0"

from .config import *
from .sst import create_sst, pad_sst_to_64x64
from .dataset import ShipSSTDataset
from .models import Generator, Discriminator, weights_init_normal
from .utils import repulsion_loss
from .train import train_shiphullgan
