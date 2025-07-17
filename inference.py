# inference.py
# Load a trained generator, sample new SST tensors, compare to training SSTs via t-SNE

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from save_load import load_generator
from dataset import ShipSSTDataset
from config import CHECKPOINT_DIR, DEFAULT_STL_DIR, DEVICE

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
GEN_FILENAME     = "generator.pth"  # or other checkpoint name
NUM_GEN_SAMPLES  = 30  # if None, match size of training dataset
TSNE_PERPLEXITY  = 30
TSNE_ITER        = 200
TSNE_RANDOM_STATE= 42
PCA_COMPONENTS   = 50

# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("[1] Loading generator...")
    device = torch.device(DEVICE)
    G = load_generator(filename=GEN_FILENAME, device=str(device))

    print("[2] Loading training SST dataset...")
    ds = ShipSSTDataset(DEFAULT_STL_DIR)
    print(f"✅ Loaded {len(ds)} training SST samples.")

    print("[3] Flattening training SST tensors...")
    train_vecs = []
    for idx in range(len(ds)):
        sst = ds[idx].numpy()            # shape: (3,25,57)
        train_vecs.append(sst.flatten())  # shape: (4275,)
    train_vecs = np.stack(train_vecs, axis=0)
    n_train = train_vecs.shape[0]

    print("[4] Sampling SSTs from generator...")
    if NUM_GEN_SAMPLES is None:
        NUM = n_train
    else:
        NUM = NUM_GEN_SAMPLES

    G.to(device)
    G.eval()
    with torch.no_grad():
        z = torch.randn(NUM, G.fc.in_features, device=device)
        gen_out = G(z)                   # (B,3,25,57)
        gen_vecs = gen_out.cpu().numpy().reshape(NUM, -1)

    print("[5] Concatenating all vectors...")
    all_vecs = np.vstack([train_vecs, gen_vecs])
    labels   = np.array([0]*n_train + [1]*NUM)

    print("[6] Reducing dimensionality with PCA...")
    pca = PCA(n_components=PCA_COMPONENTS)
    all_vecs_pca = pca.fit_transform(all_vecs)

    print("[7] Running t-SNE on PCA-reduced vectors...")
    tsne = TSNE(n_components=2,perplexity=TSNE_PERPLEXITY,max_iter=500,random_state=TSNE_RANDOM_STATE,init="random",learning_rate="auto")
    emb = tsne.fit_transform(all_vecs_pca)

    print("[8] Plotting and saving result...")
    plt.figure(figsize=(8,6))
    idx_train = labels == 0
    idx_gen   = labels == 1
    plt.scatter(emb[idx_train,0], emb[idx_train,1], s=15, alpha=0.6, label="Train SST")
    plt.scatter(emb[idx_gen,0],   emb[idx_gen,1],   s=15, alpha=0.6, label="GAN SST")
    plt.legend()
    plt.title("t-SNE of Training vs GAN-generated SSTs")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.tight_layout()

    out_path = os.path.join(CHECKPOINT_DIR, "inference_tsne.png")
    plt.savefig(out_path)
    print(f"✅ Saved t-SNE plot to {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
