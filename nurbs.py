# nurbs.py

import os
import numpy as np
import torch
import trimesh

from geomdl import fitting, exchange
from geomdl.visualization import VisMPL
from save_load import load_generator
from config import CHECKPOINT_DIR, DEVICE

# ─── Configuration ───────────────────────────────────────────────────────────
GEN_FILENAME      = "generator.pth"
N_SAMPLES         = 5
EXPORT_DIR        = os.path.join(CHECKPOINT_DIR, "nurbs_exports")
SAMPLE_RESOLUTION = (50, 50)  # (num_u, num_v) for STL sampling

# ─── Helpers ─────────────────────────────────────────────────────────────────
def strip_gmi_row(grid: np.ndarray) -> np.ndarray:
    """Drop the last (GMI) row: from 25→24 samples."""
    return grid[:-1, :]

def fit_cross_section(points: np.ndarray, degree: int = 3):
    """
    Fit a NURBS curve to a single cross-section (24×3 array).
    Uses least‐squares approximation with chord‐length parameterization.
    """
    ctrlpts = points.tolist()
    curve = fitting.approximate_curve(ctrlpts, degree)
    return curve

# ─── Main Pipeline ────────────────────────────────────────────────────────────
def reconstruct_and_export(generator, n_samples=N_SAMPLES, visualize=False):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    generator.eval().to(DEVICE)

    with torch.no_grad():
        z = torch.randn(n_samples, generator.fc.in_features, device=DEVICE)
        outs = generator(z).cpu().numpy()  # shape: (B, 3, 25, 57)

    for idx in range(n_samples):
        # 1) Extract and strip GMI row
        xg, yg, zg = outs[idx]
        xu, yu, zu = strip_gmi_row(xg), strip_gmi_row(yg), strip_gmi_row(zg)
        num_u, num_v = xu.shape  # 24 × 57

        # 2) Fit each of the 57 cross‐section curves
        curves = []
        for v in range(num_v):
            pts = np.stack([xu[:, v], yu[:, v], zu[:, v]], axis=1)  # (24,3)
            crv = fit_cross_section(pts, degree=3)
            curves.append(crv)

        # 3) Sample each curve at a uniform parameter grid
        #    to build a full surface point–grid of size num_u_sections × samples_per_curve
        curve_samples = len(curves[0].evalpts)
        grid_pts = []
        for u in range(len(curves)):
            # we sample curve u at its evalpts (already uniform for approx. curves)
            for pt in curves[u].evalpts:
                grid_pts.append(pt)  # a flat list of length num_u_sections × curve_samples

        # 4) Fit a bicubic NURBS surface through that grid
        surf = fitting.approximate_surface(
            grid_pts,
            size_u=len(curves),
            size_v=curve_samples,
            degree_u=3,
            degree_v=3,
            centripetal=True
        )

        prefix = f"hull_{idx:03d}"
        # 5) Export as OBJ
        obj_path = os.path.join(EXPORT_DIR, f"{prefix}.obj")
        exchange.export_obj(surf, obj_path)
        print(f"[✓] OBJ exported: {obj_path}")

        # 6) Export as STL via uniform sampling
        surf.sample_size_u, surf.sample_size_v = SAMPLE_RESOLUTION
        evalpts = np.array(surf.evalpts)
        pts_grid = evalpts.reshape(SAMPLE_RESOLUTION[0], SAMPLE_RESOLUTION[1], 3)
        verts = pts_grid.reshape(-1, 3)

        faces = []
        u_res, v_res = SAMPLE_RESOLUTION
        for iu in range(u_res - 1):
            for iv in range(v_res - 1):
                a = iu * v_res + iv
                b = a + 1
                c = (iu + 1) * v_res + iv
                d = c + 1
                faces.append([a, b, c])
                faces.append([b, d, c])

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        stl_path = os.path.join(EXPORT_DIR, f"{prefix}.stl")
        mesh.export(stl_path)
        print(f"[✓] STL exported: {stl_path}")

        # 7) Optional: visualize
        if visualize:
            surf.vis = VisMPL.VisSurface()
            surf.render(colormap=None)
            import matplotlib.pyplot as plt
            plt.title(prefix)

# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[1] Loading generator…")
    gen = load_generator(20,filename=GEN_FILENAME, device=DEVICE)

    print("[2] Reconstructing & exporting hulls…")
    reconstruct_and_export(gen, n_samples=N_SAMPLES, visualize=True)