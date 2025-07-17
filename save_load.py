"""
save_load.py

Utility functions to save and load the Generator model.
"""

import os
import torch
from models import Generator
from config import CHECKPOINT_DIR

def save_generator(G: torch.nn.Module, filename: str = "generator.pth"):
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(G.state_dict(), save_path)
    print(f"✅ Generator weights saved to {save_path}")

def load_generator(latent_dim: int = 20,filename: str = "generator.pth",device: str = None) -> torch.nn.Module:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    path = os.path.join(CHECKPOINT_DIR, filename)
    G = Generator(latent_dim=latent_dim).to(device)
    state = torch.load(path, map_location=device)
    G.load_state_dict(state)
    G.eval()
    print(f"✅ Generator loaded from {path}")
    return G
