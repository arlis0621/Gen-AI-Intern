"""
train.py

Defines the GAN training loop for ShipHullGAN.
Handles batchwise GPU loading, training logic, optimizer updates, logging and checkpointing.
"""

import os
import argparse

import pandas as pd

from save_load import save_generator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config   import (
    DEFAULT_STL_DIR, BATCH_SIZE, LATENT_DIM, EPOCHS,
    LEARNING_RATE, BETAS, DEVICE, CHECKPOINT_DIR, LOG_DIR,
    NUM_WORKERS, PIN_MEMORY
)
from models    import Generator, Discriminator, weights_init_normal
from dataset   import ShipSSTDataset
from sst       import pad_sst_to_64x64
from utils     import repulsion_loss

def train_shiphullgan(
    stl_dir: str = DEFAULT_STL_DIR,
    batch_size: int = BATCH_SIZE,
    latent_dim: int = LATENT_DIM,
    epochs: int = 300,
    lr: float = LEARNING_RATE,
    betas: tuple = BETAS,
    device: torch.device = DEVICE
):
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # os.makedirs(LOG_DIR, exist_ok=True)

    writer = SummaryWriter(LOG_DIR)

    ds = ShipSSTDataset(stl_dir)
    print("CWD:", os.getcwd())
    print("Using stl_dir:", stl_dir)
    print("Found STL files:", ds.files)
    print("Dataset length:", len(ds))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )

    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    adv_loss = nn.BCELoss()
    optG     = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optD     = optim.Adam(D.parameters(), lr=lr, betas=betas)

    all_d_losses = []
    all_g_losses = []
    global_step = 0
    for epoch in range(1, epochs+1):
        d_loss_accum = []
        g_loss_accum = []

        for real_sst in loader:
            B = real_sst.size(0)
            real_sst = real_sst.to(device)
            real     = pad_sst_to_64x64(real_sst)

            optD.zero_grad()
            labels_real = torch.ones(B, device=device)
            out_real    = D(real)
            loss_real   = adv_loss(out_real, labels_real)

            z          = torch.randn(B, latent_dim, device=device)
            fake_sst   = G(z).detach()
            fake       = pad_sst_to_64x64(fake_sst)
            labels_fake= torch.zeros(B, device=device)
            out_fake   = D(fake)
            loss_fake  = adv_loss(out_fake, labels_fake)

            d_loss = (loss_real + loss_fake) * 0.5
            d_loss.backward()
            optD.step()

            optG.zero_grad()
            z         = torch.randn(B, latent_dim, device=device)
            gen_sst   = G(z)
            gen_img   = pad_sst_to_64x64(gen_sst)
            out_gen   = D(gen_img)
            g_adv     = adv_loss(out_gen, torch.ones(B, device=device))
            g_rep     = repulsion_loss(gen_sst)
            gamma     = min(1.0, epoch / epochs)
            g_loss    = g_adv + gamma * g_rep
            g_loss.backward()
            optG.step()

            d_loss_accum.append(d_loss.item())
            g_loss_accum.append(g_loss.item())
            writer.add_scalar("D_loss_batch", d_loss.item(), global_step)
            writer.add_scalar("G_loss_batch", g_loss.item(), global_step)
            global_step += 1

        d_avg = np.mean(d_loss_accum)
        g_avg = np.mean(g_loss_accum)
        print(f"Epoch {epoch:03d}/{epochs:03d} | D: {d_avg:.4f} | G: {g_avg:.4f}")

        writer.add_scalar("D_loss_epoch", d_avg, epoch)
        writer.add_scalar("G_loss_epoch", g_avg, epoch)

        all_d_losses.append(d_avg)
        all_g_losses.append(g_avg)

        if epoch % 10 == 0 or epoch == epochs:
            ckpt = {
                "epoch": epoch,
                "G_state": G.state_dict(),
                "D_state": D.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
            }
            
            print("[DEBUG] cwd:", os.getcwd())
            print("[DEBUG] checkpoint_dir:", CHECKPOINT_DIR, "exists?", os.path.isdir(CHECKPOINT_DIR))
            print("[DEBUG] save_path:", os.path.join(CHECKPOINT_DIR, f"shiphullgan_epoch{epoch}.pt"))

    

    writer.close()
    save_generator(G, filename=f"generator.pth")

    # Convergence plot
    

    # Optional: Save CSV
    df = pd.DataFrame({"epoch": list(range(1, epochs+1)), "D_loss": all_d_losses, "G_loss": all_g_losses})
    df.to_csv(os.path.join(LOG_DIR, "losses.csv"), index=False)

    return G, D, optG, optD

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stl_dir",    default=DEFAULT_STL_DIR)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    p.add_argument("--epochs",     type=int, default=EPOCHS)
    p.add_argument("--lr",         type=float, default=LEARNING_RATE)
    p.add_argument("--betas",      nargs=2,  type=float, default=list(BETAS))
    p.add_argument("--device",     default=str(DEVICE))
    args = p.parse_args()

    train_shiphullgan(
        stl_dir=args.stl_dir,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        betas=tuple(args.betas),
        device=torch.device(args.device)
    )

if __name__ == "__main__":
    main()
