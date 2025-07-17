# run_training.py
#!/usr/bin/env python3

import os
from config import CHECKPOINT_DIR, LOG_DIR, DEFAULT_STL_DIR, DEVICE
from train import main as train_main
from inference import main as inference_main
from nurbs import reconstruct_and_export
from save_load import load_generator

GEN_FILENAME = "generator.pth"  # must match the filename in save_generator
N_SAMPLES = 5  # number of GAN outputs to convert into NURBS surfaces

def prepare_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == "__main__":
    print("🔧 Preparing directories...")
    prepare_dirs()

    print("🚀 Starting training...")
    train_main()

    print("📉 Running inference t-SNE visualization...")
    inference_main()

    print("📐 Generating NURBS hulls...")
    gen = load_generator(20,filename=GEN_FILENAME, device=DEVICE)
    reconstruct_and_export(gen, n_samples=N_SAMPLES, visualize=False)
