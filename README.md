# Gen-AI-Intern
ShipHullGAN: AI‑Driven Ship Hull Generation:
“Reimagining naval design with GAN‑powered geometry matrices and NURBS reconstruction for ultra‑realistic hull forms.” About I spearheaded the core GAN architecture (models.py) that learns from 3D ship‑hull SST tensors—each a 3×25×57 geometry grid—and produces entirely new hull shapes. I engineered the custom SST pipeline (sst.py) to sample cross‑sections, compute invariant features, and normalize data for robust model training.

This project tackles the time‑intensive challenge of parametric hull design by automating shape synthesis and CAD‑ready reconstruction, marrying machine learning, 3D geometry processing, and NURBS‑Python. The result: “first‑pass” hull forms that naval architects can immediately review and iterate.

As a Generative AI (GANs) Intern, I authored end‑to‑end training scripts (train.py, run_training.py), optimized data loading with PyTorch DataLoader, and integrated advanced t‑SNE diagnostics (inference.py) for model convergence insights—driving a seamless CI/CD‑style workflow on GPU clusters.

Outcome & Impact Statement • Accelerated ideation: Reduced manual hull‑form prototyping from weeks to minutes. • Educational value: Delivered a hands‑on GAN framework and dataset for ship‑design curricula, enabling future engineers to explore ML‑driven CAD.
