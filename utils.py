
"""
utils.py

Miscellaneous helper functions for training ShipHullGAN,
including custom loss terms and memory monitoring.
"""

import torch
import psutil


def repulsion_loss(samples, epsilon=1e-6):
    B = samples.size(0)
    # Option A: make contiguous then view
    # vectors = samples.contiguous().view(B, -1)

    # Option B: just reshape
    vectors = samples.reshape(B, -1)

    dist2 = torch.cdist(vectors, vectors, p=2).pow(2) + epsilon
    inv2  = 1.0 / dist2
    eye   = torch.eye(B, device=samples.device).bool()
    inv2  = inv2.masked_fill(eye, 0.0)
    return inv2.sum() / (B*(B-1))


def print_cpu_ram():
    vmem = psutil.virtual_memory()
    print(f"🧠 CPU RAM Used: {vmem.used / 1e9:.2f} GB / {vmem.total / 1e9:.2f} GB")

def print_gpu_mem():
    if torch.cuda.is_available():
        print(f"🔥 GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"🔥 GPU Memory Reserved : {torch.cuda.memory_reserved() / 1e6:.2f} MB")
