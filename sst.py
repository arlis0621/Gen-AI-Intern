

# ── Imports 
import os
POINTS_PER_CS = 25
import numpy as np
import torch

import torch.nn.functional as F
from math import comb
import trimesh
from collections import defaultdict, deque
from typing import Tuple
def load_and_normalize_mesh(stl_path: str) -> trimesh.Trimesh:
    
    mesh = trimesh.load_mesh(stl_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load a Trimesh from '{stl_path}'.")
    bbox_min, bbox_max = mesh.bounds
    dims = bbox_max - bbox_min
    L_bar = dims[0]
    if L_bar <= 0:
        raise ValueError(f"Invalid bounding-box length L̄={L_bar}. Must be > 0.")
    mesh.apply_translation(-bbox_min)
    scale_factor = 1.0 / L_bar
    mesh.apply_scale(scale_factor)
    return mesh

def compute_station_positions(num_sections: int = 56) -> np.ndarray:
    if num_sections % 4 != 0:
        raise ValueError(f"num_sections must be divisible by 4, got {num_sections}")

    # The four region boundaries (from Fig. 8(c) in the paper):
    regions = [
        (0.0, 0.1),
        (0.1, 0.3),
        (0.3, 0.8),
        (0.8, 1.0)
    ]
    sections_per_region = num_sections // 4  

    x_positions = []
    for (a, b) in regions:
        Δ = (b - a) / (sections_per_region - 1)
        for j in range(sections_per_region):
            x_val = a + j * Δ
            x_positions.append(x_val)

    stations = np.array(x_positions, dtype=float)

    # Sanity check: should be exactly 56 values
    if stations.shape[0] != num_sections:
        raise RuntimeError(
            f"Expected {num_sections} stations, but obtained {stations.shape[0]}"
        )

    return stations
def get_hinge_point(mesh: trimesh.Trimesh) -> np.ndarray:
    #print("\n[Hinge] Slicing at x = 0.1 to find hinge point")

    # 1) Transverse slice at x = 0.1
    section = mesh.section(plane_origin=[0.1, 0.0, 0.0], plane_normal=[1.0, 0.0, 0.0])
    if section is None:
        print("  → No transverse slice at x=0.1; cannot compute hinge.")
        return None

    # 2) Convert Path3D → Path2D + transformation matrix
    section2d, to_3d_transform = section.to_planar()
    #print(f"  → Received 2D path with {len(section2d.vertices)} vertices")
    #print(f"  → to_3d_transform matrix:\n{np.round(to_3d_transform,4)}")

    planar_pts = section2d.vertices  # shape (K, 2)

    # 3) Find the planar vertex with minimal |u| (u ≈ y in 3D)
    abs_u = np.abs(planar_pts[:, 0])
    idx = int(np.argmin(abs_u))
    chosen_u, chosen_v = planar_pts[idx]
    #print(f"  → Chosen planar index: {idx} (u={chosen_u:.6f}, v={chosen_v:.6f})")

    # 4) Manually map (u,v) → (x,y,z) using the 4×4 transform
    uv_homog = np.array([chosen_u, chosen_v, 0.0, 1.0])
    hinge_3d_homog = to_3d_transform @ uv_homog
    hinge_3d = hinge_3d_homog[:3]
    #print(f"    → Mapped hinge 3D = {np.round(hinge_3d,6)}")

    # 5) Sanity: print the centroid of this slice as well
    centroid_2d = planar_pts.mean(axis=0)
    cent2d_homog = np.array([centroid_2d[0], centroid_2d[1], 0.0, 1.0])
    centroid_3d = (to_3d_transform @ cent2d_homog)[:3]
    #print(f"  → Slice centroid 2D: ({centroid_2d[0]:.6f}, {centroid_2d[1]:.6f})")
    #print(f"    → Slice centroid 3D: {np.round(centroid_3d,6)}")

    return hinge_3d

def get_deck_curve_points(mesh: trimesh.Trimesh, num_points: int = 14) -> np.ndarray:
    
    #print(f"\n[Deck] Finding {num_points} deck-curve points in x ∈ [0, 0.1]")

    vertices = mesh.vertices  # (V, 3)
    V_p1 = vertices[vertices[:, 0] <= 0.1]
    #print(f"  → Total vertices with x ≤ 0.1: {len(V_p1)}")

    x_targets = np.linspace(0.0, 0.1, num_points)
    deck_pts = []

    for i, x_t in enumerate(x_targets):
        tol = 1e-3
        candidates = V_p1[np.abs(V_p1[:, 0] - x_t) < tol]
        #print(f"    • x_target[{i}] = {x_t:.6f}: {len(candidates)} candidates within tol={tol}")
        if len(candidates) > 0:
            idx_maxz = int(np.argmax(candidates[:, 2]))
            chosen = candidates[idx_maxz]
            #print(f"      → Chose candidate with max z={chosen[2]:.6f} → {np.round(chosen,6)}")
        else:
            idx_near = int(np.argmin(np.abs(V_p1[:, 0] - x_t)))
            chosen = V_p1[idx_near]
            #print(f"      → No exact candidate; chose nearest in x → {np.round(chosen,6)}")
        deck_pts.append(chosen)

    deck_pts = np.vstack(deck_pts)
    #print(f"  → Final deck_pts shape: {deck_pts.shape}")
    return deck_pts





def reconstruct_chain(section):
    verts = section.vertices
    # 1) adjacency list
    adj = defaultdict(list)
    for ent in section.entities:
        pts = ent.points
        for a,b in zip(pts[:-1], pts[1:]):
            adj[a].append(b)
            adj[b].append(a)
    if not adj:
        return np.array([],dtype=int)

    # 2) find largest component
    visited = set()
    comps = []
    for v in adj:
        if v not in visited:
            comp = []
            dq = deque([v])
            visited.add(v)
            while dq:
                u = dq.popleft()
                comp.append(u)
                for w in adj[u]:
                    if w not in visited:
                        visited.add(w)
                        dq.append(w)
            comps.append(comp)
    comp = max(comps, key=len)

    # 3) find endpoint or start loop
    ends = [v for v in comp if len(adj[v])==1]
    start = ends[0] if len(ends)==2 else comp[0]

    # 4) traverse
    chain = []
    prev = None
    curr = start
    while True:
        chain.append(curr)
        nbrs = [n for n in adj[curr] if n!=prev]
        if not nbrs:
            break
        prev, curr = curr, nbrs[0]
        if curr == start:
            break

    return np.array(chain, dtype=int)


def sample_along_chain(chain3d, num_samples=POINTS_PER_CS):
    K = len(chain3d)
    if K < 2:
        return np.zeros((num_samples,3)) if K==0 else np.tile(chain3d[0], (num_samples,1))
    # segment lengths
    deltas = np.diff(chain3d, axis=0)
    seglen = np.linalg.norm(deltas, axis=1)
    total = seglen.sum()
    if total == 0:
        return np.tile(chain3d[0], (num_samples,1))
    cum = np.concatenate([[0], np.cumsum(seglen)])
    targets = np.linspace(0, total, num_samples)
    sampled = []
    for t in targets:
        i = np.searchsorted(cum, t) - 1
        i = np.clip(i, 0, K-2)
        s0, L = cum[i], seglen[i]
        alpha = 0 if L==0 else (t-s0)/L
        P, Q = chain3d[i], chain3d[i+1]
        sampled.append(P + alpha*(Q-P))
    return np.vstack(sampled)


def sample_no_deck(section, num_samples=POINTS_PER_CS, z_tol_frac=0.02):
    verts = section.vertices
    idx_chain = reconstruct_chain(section)
    if len(idx_chain) < 2:
        return np.zeros((num_samples,3))

    chain3d = verts[idx_chain]
    z = chain3d[:,2]
    zmin, zmax = z.min(), z.max()
    tol = (zmax - zmin) * z_tol_frac

    # mask: keep only points strictly below deck plateau
    mask = z < (zmax - tol)
    # find contiguous True runs
    runs = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i==len(mask)-1) and start is not None:
            end = i if m else i-1
            runs.append((start, end+1))
            start = None
    if not runs:
        # nothing left below deck: fall back to full chain
        return sample_along_chain(chain3d, num_samples)

    # pick longest run
    run = max(runs, key=lambda r: r[1]-r[0])
    subchain3d = chain3d[run[0]:run[1]]
    if len(subchain3d) < 2:
        return sample_along_chain(chain3d, num_samples)
    return sample_along_chain(subchain3d, num_samples)
def extract_oblique_section_entity(mesh, deck_pt, hinge_pt):
    p1, p2 = deck_pt, hinge_pt
    p3 = p1 + np.array([0,1e-3,0])
    n  = np.cross(p2-p1, p3-p1)
    if np.linalg.norm(n)==0: return None
    n /= np.linalg.norm(n)
    sec = mesh.section(plane_origin=p1.tolist(),plane_normal=n.tolist())
    return None if (sec is None) else sample_no_deck(sec)

def extract_transverse_section_entity(mesh, x_val, eps=1e-5):
    for x in (x_val, x_val+eps, x_val-eps):
        sec = mesh.section(plane_origin=[x,0,0],plane_normal=[1,0,0])
        if sec is not None:
            return sample_no_deck(sec)
    return None
import numpy as np

# constants (make sure these match your notebook)
POINTS_PER_CS = 25
NUM_SECTIONS  = 56


# 1) build_geometry_matrices

def build_geometry_matrices(mesh: trimesh.Trimesh,stations: np.ndarray,num_samples: int = POINTS_PER_CS):
    # 1) Prepare P1 helpers
    hinge_pt = get_hinge_point(mesh)
    deck_pts = get_deck_curve_points(mesh)  # default returns 14 deck points

    # 2) Allocate
    Nsec = len(stations)
    X = np.zeros((num_samples, Nsec), dtype=float)
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    valid = np.zeros(Nsec, dtype=bool)

    # 3) Loop stations
    for i, x in enumerate(stations):
        if x <= 0.1:
            # bulb region → oblique slice
            j = int(round((x / 0.1)*(len(deck_pts)-1)))
            samp = extract_oblique_section_entity(mesh, deck_pts[j], hinge_pt)
        else:
            samp = extract_transverse_section_entity(mesh, x)

        # 4) Check shape and non‐zero
        if samp is None or samp.shape != (num_samples, 3):
            continue
        if np.allclose(samp, 0.0):
            continue

        # 5) Fill in
        valid[i] = True
        X[:, i], Y[:, i], Z[:, i] = samp[:, 0], samp[:, 1], samp[:, 2]

    return X, Y, Z, valid
import numpy as np
import trimesh


# Constants
INV_ORDER      = 4                # Compute moments up to total order 4
NUM_INVARIANTS = 35               # Number of (p, q, r) with p+q+r ≤ 4
POINTS_PER_CS  = 25               # Number of points per cross‐section (unchanged)
NUM_SECTIONS   = 56               # Number of longitudinal stations
AUG_COLS       = NUM_SECTIONS + 1 # 56 raw stations + 1 extra column for invariants


# ---------------------------------------------------
# 8.1. Computing 4th‐Order Geometric Moment Invariants
# ---------------------------------------------------

def compute_moment_invariants(mesh: trimesh.Trimesh, order: int = INV_ORDER) -> np.ndarray:
    # ————————————
    # 1) Volume and centroid
    volume = mesh.volume
    if volume is None or volume <= 0:
        try:
            ch = mesh.convex_hull
            volume = ch.volume if ch.is_watertight else 0.0
        except Exception:
            volume = 0.0

    try:
        centroid = mesh.center_mass
    except Exception:
        bmin, bmax = mesh.bounds  # (2×3)
        centroid = (bmin + bmax) * 0.5

    #print(f"[Moments] Using volume (μ000) = {volume:.6e}")
    #print(f"[Moments] Using centroid = {np.round(centroid,6)}")

    # ————————————
    # 2) Raw moments
    raw_moments = _compute_raw_polyhedral_moments(mesh, order)
    # raw_moments[(p,q,r)] = ∭ x^p y^q z^r dV

    # ————————————
    # 3) Central moments
    central_moments = _raw_to_central(raw_moments, centroid, order)
    # central_moments[(p,q,r)] = ∭ (x - Cx)^p (y - Cy)^q (z - Cz)^r dV

    # ————————————
    # 4) Scale‐normalized invariants
    invariants = []
    for total in range(order + 1):        # total = p+q+r = 0..4
        for p in range(total + 1):
            for q in range(total - p + 1):
                r = total - p - q
                mu_val = central_moments.get((p, q, r), 0.0)
                expo = 1.0 + (p + q + r) / 3.0
                if volume <= 0:
                    #print("  ⚠️ Non‐positive volume → setting MI=0.")
                    mi = 0.0
                else:
                    mi = mu_val / (volume ** expo)
                invariants.append(mi)
                #print(f"   ► (p,q,r)=({p},{q},{r}): μ={mu_val:.6e}, MI={mi:.6e}")

    invariants = np.array(invariants, dtype=float)
    if invariants.size != NUM_INVARIANTS:
        raise RuntimeError(f"Expected {NUM_INVARIANTS} invariants but got {invariants.size}")

    #print(f"[Moments] Computed {len(invariants)} invariants.")
    return invariants


def _compute_raw_polyhedral_moments(mesh: trimesh.Trimesh, order: int) -> dict:
    # Initialize raw dictionary
    raw = {
        (p, q, r): 0.0
        for total in range(order + 1)
        for p in range(total + 1)
        for q in range(total - p + 1)
        for r in [total - p - q]
    }

    vertices = mesh.vertices  # (V,3)
    faces    = mesh.faces     # (F,3)

    for f in faces:
        v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        face_dict  = _compute_face_raw_moments(v0, v1, v2, order)
        # Add each face’s contribution
        for key, val in face_dict.items():
            raw[key] += val

    return raw


def _compute_face_raw_moments(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, order: int) -> dict:
    # 1) Face normal (not normalized) and area
    n = np.cross(v1 - v0, v2 - v0)
    area2 = np.linalg.norm(n)
    if area2 == 0:
        # Degenerate face → no contribution
        return {
            (p, q, r): 0.0
            for total in range(order + 1)
            for p in range(total + 1)
            for q in range(total - p + 1)
            for r in [total - p - q]
        }

    # 2) Unit normal
    unit_n = n / area2

    # 3) Determine dominant axis (largest absolute component of unit_n)
    dom = int(np.argmax(np.abs(unit_n)))  # 0→x, 1→y, 2→z

    # 4) Project vertices onto the plane orthogonal to dom
    if dom == 0:
        # Projection onto y–z: (a_i = y_i, b_i = z_i)
        a0, b0 = v0[1], v0[2]
        a1, b1 = v1[1], v1[2]
        a2, b2 = v2[1], v2[2]
        nf = unit_n[0]
    elif dom == 1:
        # Projection onto x–z: (a_i = x_i, b_i = z_i)
        a0, b0 = v0[0], v0[2]
        a1, b1 = v1[0], v1[2]
        a2, b2 = v2[0], v2[2]
        nf = unit_n[1]
    else:
        # dom == 2 → project onto x–y: (a_i = x_i, b_i = y_i)
        a0, b0 = v0[0], v0[1]
        a1, b1 = v1[0], v1[1]
        a2, b2 = v2[0], v2[1]
        nf = unit_n[2]

    # 5) Evaluate F_{p,q,r} for each monomial and scale by nf/(p+q+r+3)
    face_contrib = {}
    for total in range(order + 1):
        for p in range(total + 1):
            for q in range(total - p + 1):
                r = total - p - q
                F = _evaluate_F_pqr(a0, b0, a1, b1, a2, b2, p, q, r, dom)
                face_contrib[(p, q, r)] = (nf / (total + 3)) * F

    return face_contrib


def _evaluate_F_pqr(a0, b0, a1, b1, a2, b2, p, q, r, dom):
    # ——— Common shorthand sums ———
    # sums over the three vertices in “a_i” domain:
    s1 = a0 + a1 + a2
    s2 = a0*a0 + a1*a1 + a2*a2 + a0*a1 + a1*a2 + a2*a0
    s3 = (
        a0*a0*a0
        + a1*a1*a1
        + a2*a2*a2
        + a0*a0*(a1 + a2)
        + a1*a1*(a2 + a0)
        + a2*a2*(a0 + a1)
        + a0*a1*a2
    )
    s4 = (
        a0**4
        + a1**4
        + a2**4
        + a0**2*(a1*a1 + a2*a2 + a1*a2)
        + a1**2*(a2*a2 + a2*a0 + a0*a0)
        + a2**2*(a0*a0 + a0*a1 + a1*a1)
        + a0*a1*a1*a2
        + a0*a1*a2*a2
        + a0*a2*a2*a1
    )
    # And for “b_i” domain:
    t1 = b0 + b1 + b2
    t2 = b0*b0 + b1*b1 + b2*b2 + b0*b1 + b1*b2 + b2*b0
    t3 = (
        b0*b0*b0
        + b1*b1*b1
        + b2*b2*b2
        + b0*b0*(b1 + b2)
        + b1*b1*(b2 + b0)
        + b2*b2*(b0 + b1)
        + b0*b1*b2
    )
    t4 = (
        b0**4
        + b1**4
        + b2**4
        + b0**2*(b1*b1 + b2*b2 + b1*b2)
        + b1**2*(b2*b2 + b2*b0 + b0*b0)
        + b2**2*(b0*b0 + b0*b1 + b1*b1)
        + b0*b1*b1*b2
        + b0*b1*b2*b2
        + b0*b2*b2*b1
    )

    # Now branch on dom
    if dom == 0:
        # —— Projection onto y–z plane. “a_i = y_i, b_i = z_i,” so “x” is omitted.
        # We must return F_{p,q,r} for each (p,q,r). Summarize cases:
        if (p, q, r) == (0, 0, 0):
            return 0.5 * ((b0 - b2)*(a0*b1 - a1*b0)+ (b1 - b0)*(a1*b2 - a2*b1)+ (b2 - b1)*(a2*b0 - a0*b2))

        elif (p, q, r) == (1, 0, 0):
            # F_{1,0,0}
            term0 = s1 * (b0 - b2)*(a0*b1 - a1*b0)
            term1 = s1 * (b1 - b0)*(a1*b2 - a2*b1)
            term2 = s1 * (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 6.0) * (term0 + term1 + term2)

        elif (p, q, r) == (0, 1, 0):
            # F_{0,1,0} = 1/6 * [ (b0+b1+b2)*(b0 - b2)*(a0*b1 - a1*b0) + … cyclic ]
            S_b012 = b0 + b1 + b2
            term0 = S_b012 * (b0 - b2)*(a0*b1 - a1*b0)
            term1 = S_b012 * (b1 - b0)*(a1*b2 - a2*b1)
            term2 = S_b012 * (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 6.0) * (term0 + term1 + term2)

        elif (p, q, r) == (0, 0, 1):
            # F_{0,0,1} = 1/6 * [ (b0 - b2)*(a0*b1 - a1*b0) + … cyclic ] * (a0+a1+a2)
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (s1 / 6.0) * (term0 + term1 + term2)

        elif (p, q, r) == (2, 0, 0):
            # F_{2,0,0} = 1/12 * [ (a0+a1+a2)^2 + (a0^2+a1^2+a2^2) ] * [ (b0-b2)*(a0*b1 - a1*b0) + … ]
            A = s1*s1 + (a0*a0 + a1*a1 + a2*a2 + a0*a1 + a1*a2 + a2*a0)
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 12.0) * A * (term0 + term1 + term2)

        elif (p, q, r) == (1, 1, 0):
            # F_{1,1,0}
            T = s2
            term0 = (b0 - b2)*(a0*b1 - a1*b0) * (a0 + a1 + a2)
            term1 = (b1 - b0)*(a1*b2 - a2*b1) * (a1 + a2 + a0)
            term2 = (b2 - b1)*(a2*b0 - a0*b2) * (a2 + a0 + a1)
            return (1.0 / 12.0) * (T * (term0 + term1 + term2))

        elif (p, q, r) == (1, 0, 1):
            # F_{1,0,1}
            T = s2
            t01 = (b0 - b2)*(a0*b1 - a1*b0)
            t11 = (b1 - b0)*(a1*b2 - a2*b1)
            t21 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 12.0) * (s1 * (t01 + t11 + t21) * (b0 + b1 + b2))

        elif (p, q, r) == (0, 2, 0):
            # F_{0,2,0}
            B = t1*t1 + (b0*b0 + b1*b1 + b2*b2 + b0*b1 + b1*b2 + b2*b0)
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 12.0) * B * (term0 + term1 + term2)

        elif (p, q, r) == (0, 1, 1):
            # F_{0,1,1}
            T = t2
            term0 = (b0 - b2)*(a0*b1 - a1*b0) * (a0 + a1 + a2)
            term1 = (b1 - b0)*(a1*b2 - a2*b1) * (a1 + a2 + a0)
            term2 = (b2 - b1)*(a2*b0 - a0*b2) * (a2 + a0 + a1)
            return (1.0 / 12.0) * (T * (term0 + term1 + term2))

        elif (p, q, r) == (0, 0, 2):
            # F_{0,0,2}
            # = (1/12) * ( (b0 - b2)*(a0*b1 - a1*b0) + … ) * (b0 + b1 + b2)^2 + ...
            C = t1*t1 + (b0*b0 + b1*b1 + b2*b2 + b0*b1 + b1*b2 + b2*b0)
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 12.0) * C * (term0 + term1 + term2)

        elif (p, q, r) == (3, 0, 0):
            # F_{3,0,0}
            P = s3
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 20.0) * P * (term0 + term1 + term2)

        elif (p, q, r) == (2, 1, 0):
            # F_{2,1,0}
            P = s3
            term0 = (b0 - b2)*(a0*b1 - a1*b0) * (a0 + a1 + a2)
            term1 = (b1 - b0)*(a1*b2 - a2*b1) * (a1 + a2 + a0)
            term2 = (b2 - b1)*(a2*b0 - a0*b2) * (a2 + a0 + a1)
            return (1.0 / 20.0) * P * (term0 + term1 + term2)

        elif (p, q, r) == (2, 0, 1):
            # F_{2,0,1}
            P = s3
            Q = t1
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 20.0) * (P * (term0 + term1 + term2) * Q)

        elif (p, q, r) == (1, 2, 0):
            # F_{1,2,0}
            Q = t3
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 20.0) * (Q * (term0 + term1 + term2))

        elif (p, q, r) == (1, 1, 1):
            # F_{1,1,1}
            P = s3
            Q = t3
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 20.0) * (P * Q * (term0 + term1 + term2))

        elif (p, q, r) == (1, 0, 2):
            # F_{1,0,2}
            Q = t3
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 20.0) * (s1 * (term0 + term1 + term2) * Q)

        elif (p, q, r) == (0, 3, 0):
            # F_{0,3,0}
            R = t3
            return (1.0 / 20.0) * R * ( (b0 - b2)*(a0*b1 - a1*b0)+ (b1 - b0)*(a1*b2 - a2*b1)+ (b2 - b1)*(a2*b0 - a0*b2) )

        elif (p, q, r) == (0, 2, 1):
            # F_{0,2,1}
            R = t3
            term0 = (b0 - b2)*(a0*b1 - a1*b0) * (a0 + a1 + a2)
            term1 = (b1 - b0)*(a1*b2 - a2*b1) * (a1 + a2 + a0)
            term2 = (b2 - b1)*(a2*b0 - a0*b2) * (a2 + a0 + a1)
            return (1.0 / 20.0) * (R * (term0 + term1 + term2))

        elif (p, q, r) == (0, 1, 2):
            # F_{0,1,2}
            R = t3
            S = s1
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 20.0) * (R * S * (term0 + term1 + term2))

        elif (p, q, r) == (0, 0, 3):
            # F_{0,0,3}
            R = t3
            return (1.0 / 20.0) * R * ( (b0 - b2)*(a0*b1 - a1*b0)+ (b1 - b0)*(a1*b2 - a2*b1)+ (b2 - b1)*(a2*b0 - a0*b2) )

        elif (p, q, r) == (4, 0, 0):
            S4 = s4
            C = (b0 - b2)*(a0*b1 - a1*b0) + (b1 - b0)*(a1*b2 - a2*b1) + (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 30.0) * S4 * C

        elif (p, q, r) == (3, 1, 0):
            S4 = s4
            term0 = (b0 - b2)*(a0*b1 - a1*b0) * s1
            term1 = (b1 - b0)*(a1*b2 - a2*b1) * s1
            term2 = (b2 - b1)*(a2*b0 - a0*b2) * s1
            return (1.0 / 30.0) * (S4 * (term0 + term1 + term2))

        elif (p, q, r) == (3, 0, 1):
            S4 = s4
            T = t1
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 30.0) * (S4 * T * (term0 + term1 + term2))

        elif (p, q, r) == (2, 2, 0):
            S4 = s4
            T = t2
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 30.0) * (S4 * T * (term0 + term1 + term2))

        elif (p, q, r) == (2, 1, 1):
            S4 = s4
            T = t2
            term0 = (b0 - b2)*(a0*b1 - a1*b0) * s1
            term1 = (b1 - b0)*(a1*b2 - a2*b1) * s1
            term2 = (b2 - b1)*(a2*b0 - a0*b2) * s1
            return (1.0 / 30.0) * (S4 * T * (term0 + term1 + term2))

        elif (p, q, r) == (2, 0, 2):
            S4 = s4
            T = t2
            return (1.0 / 30.0) * (S4 * T * ( (b0 - b2)*(a0*b1 - a1*b0)+ (b1 - b0)*(a1*b2 - a2*b1)+ (b2 - b1)*(a2*b0 - a0*b2) ))

        elif (p, q, r) == (1, 3, 0):
            S4 = s4
            U = t3
            term0 = (b0 - b2)*(a0*b1 - a1*b0)
            term1 = (b1 - b0)*(a1*b2 - a2*b1)
            term2 = (b2 - b1)*(a2*b0 - a0*b2)
            return (1.0 / 30.0) * (S4 * U * (term0 + term1 + term2))

        elif (p, q, r) == (1, 2, 1):
            S4 = s4
            U = t3
            term0 = (b0 - b2)*(a0*b1 - a1*b0) * s1
            term1 = (b1 - b0)*(a1*b2 - a2*b1) * s1
            term2 = (b2 - b1)*(a2*b0 - a0*b2) * s1
            return (1.0 / 30.0) * (S4 * U * (term0 + term1 + term2))

        elif (p, q, r) == (1, 1, 2):
            S4 = s4
            U = t3
            return (1.0 / 30.0) * (S4 * U * ( (b0 - b2)*(a0*b1 - a1*b0) + (b1 - b0)*(a1*b2 - a2*b1)+ (b2 - b1)*(a2*b0 - a0*b2) ))

        elif (p, q, r) == (1, 0, 3):
            S4 = s4
            U = t3
            return (1.0 / 30.0) * (S4 * U * ( (b0 - b2)*(a0*b1 - a1*b0) + (b1 - b0)*(a1*b2 - a2*b1)+ (b2 - b1)*(a2*b0 - a0*b2) ))

        elif (p, q, r) == (0, 4, 0):
            L = t4
            return (1.0 / 30.0) * (L * ( (b0 - b2)*(a0*b1 - a1*b0)+ (b1 - b0)*(a1*b2 - a2*b1)+ (b2 - b1)*(a2*b0 - a0*b2) ))

        elif (p, q, r) == (0, 3, 1):
            L = t4
            term0 = (b0 - b2)*(a0*b1 - a1*b0) * (a0 + a1 + a2)
            term1 = (b1 - b0)*(a1*b2 - a2*b1) * (a1 + a2 + a0)
            term2 = (b2 - b1)*(a2*b0 - a0*b2) * (a2 + a0 + a1)
            return (1.0 / 30.0) * (L * (term0 + term1 + term2))

        elif (p, q, r) == (0, 2, 2):
            L = t4
            term0 = (b0 - b2)*(a0*b1 - a1*b0) * (a0 + a1 + a2)
            term1 = (b1 - b0)*(a1*b2 - a2*b1) * (a1 + a2 + a0)
            term2 = (b2 - b1)*(a2*b0 - a0*b2) * (a2 + a0 + a1)
            return (1.0 / 30.0) * (L * (term0 + term1 + term2))

        elif (p, q, r) == (0, 1, 3):
            L = t4
            return (1.0 / 30.0) * (L * ( (b0 - b2)*(a0*b1 - a1*b0)
                                    + (b1 - b0)*(a1*b2 - a2*b1)
                                    + (b2 - b1)*(a2*b0 - a0*b2) ))

        elif (p, q, r) == (0, 0, 4):
            L = t4
            return (1.0 / 30.0) * (L * ( (b0 - b2)*(a0*b1 - a1*b0)
                                    + (b1 - b0)*(a1*b2 - a2*b1)
                                    + (b2 - b1)*(a2*b0 - a0*b2) ))
        else:
            return 0.0

    elif dom == 1:
        # —— Projection onto x–z plane. “a_i = x_i, b_i = z_i”
        # We rename: A0=x0, B0=z0; A1=x1, B1=z1; A2=x2, B2=z2
        # Then replicate exactly the same 35 formulas as above, but using (A,B) instead of (a,b).
        A0, B0 = a0, b0  # (but actually here a0==x0, b0==z0 since dom==1)
        A1, B1 = a1, b1
        A2, B2 = a2, b2
        # Re‐compute sums:
        S1 = A0 + A1 + A2
        S2 = (A0*A0 + A1*A1 + A2*A2 + A0*A1 + A1*A2 + A2*A0)
        S3 = (
            A0*A0*A0
            + A1*A1*A1
            + A2*A2*A2
            + A0*A0*(A1 + A2)
            + A1*A1*(A2 + A0)
            + A2*A2*(A0 + A1)
            + A0*A1*A2
        )
        S4 = (
            A0**4
            + A1**4
            + A2**4
            + A0**2*(A1*A1 + A2*A2 + A1*A2)
            + A1**2*(A2*A2 + A2*A0 + A0*A0)
            + A2**2*(A0*A0 + A0*A1 + A1*A1)
            + A0*A1*A1*A2
            + A0*A1*A2*A2
            + A0*A2*A2*A1
        )
        T1 = B0 + B1 + B2
        T2 = (B0*B0 + B1*B1 + B2*B2 + B0*B1 + B1*B2 + B2*B0)
        T3 = (
            B0*B0*B0
            + B1*B1*B1
            + B2*B2*B2
            + B0*B0*(B1 + B2)
            + B1*B1*(B2 + B0)
            + B2*B2*(B0 + B1)
            + B0*B1*B2
        )
        T4 = (
            B0**4
            + B1**4
            + B2**4
            + B0**2*(B1*B1 + B2*B2 + B1*B2)
            + B1**2*(B2*B2 + B2*B0 + B0*B0)
            + B2**2*(B0*B0 + B0*B1 + B1*B1)
            + B0*B1*B1*B2
            + B0*B1*B2*B2
            + B0*B2*B2*B1
        )

        if (p, q, r) == (0, 0, 0):
            return 0.5 * ((B0 - B2)*(A0*B1 - A1*B0)+ (B1 - B0)*(A1*B2 - A2*B1)+ (B2 - B1)*(A2*B0 - A0*B2))

        elif (p, q, r) == (1, 0, 0):
            term0 = S1 * (B0 - B2)*(A0*B1 - A1*B0)
            term1 = S1 * (B1 - B0)*(A1*B2 - A2*B1)
            term2 = S1 * (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 6.0) * (term0 + term1 + term2)

        elif (p, q, r) == (0, 1, 0):
            return 0.0  # because q=1 in “x–z” projection means a y‐component doesn’t appear
        elif (p, q, r) == (0, 0, 1):
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (S1 / 6.0) * (term0 + term1 + term2)

        elif (p, q, r) == (2, 0, 0):
            A = S1*S1 + (A0*A0 + A1*A1 + A2*A2 + A0*A1 + A1*A2 + A2*A0)
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 12.0) * A * (term0 + term1 + term2)

        elif (p, q, r) == (1, 1, 0):
            # In x–z projection, “q” refers to y‐power; we’re missing a y factor,
            # so F_{1,1,0} = 0 because none of (b0,b1,b2) refer to y.
            return 0.0

        elif (p, q, r) == (1, 0, 1):
            T = T1
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 12.0) * (S1 * (term0 + term1 + term2) * T)

        elif (p, q, r) == (0, 2, 0):
            # q=2 means y^2; in x–z projection, y disappears → 0
            return 0.0

        elif (p, q, r) == (0, 1, 1):
            # y^1 z^1: again y^1 cannot appear → 0
            return 0.0

        elif (p, q, r) == (0, 0, 2):
            C = T1*T1 + (B0*B0 + B1*B1 + B2*B2 + B0*B1 + B1*B2 + B2*B0)
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 12.0) * C * (term0 + term1 + term2)

        elif (p, q, r) == (3, 0, 0):
            P = S3
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 20.0) * P * (term0 + term1 + term2)

        elif (p, q, r) == (2, 1, 0):
            return 0.0  # includes a y‐power, so 0

        elif (p, q, r) == (2, 0, 1):
            P = S3
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 20.0) * (P * (term0 + term1 + term2) * T1)

        elif (p, q, r) == (1, 2, 0):
            return 0.0

        elif (p, q, r) == (1, 1, 1):
            return 0.0

        elif (p, q, r) == (1, 0, 2):
            P = S3
            T = T1
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 20.0) * (P * (term0 + term1 + term2) * T)

        elif (p, q, r) == (0, 3, 0):
            return 0.0

        elif (p, q, r) == (0, 2, 1):
            return 0.0

        elif (p, q, r) == (0, 1, 2):
            return 0.0

        elif (p, q, r) == (0, 0, 3):
            R = T3
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 20.0) * R * (term0 + term1 + term2)

        elif (p, q, r) == (4, 0, 0):
            S4_ = S4
            C = (B0 - B2)*(A0*B1 - A1*B0) + (B1 - B0)*(A1*B2 - A2*B1) + (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 30.0) * S4_ * C

        elif (p, q, r) == (3, 1, 0):
            return 0.0

        elif (p, q, r) == (3, 0, 1):
            S4_ = S4
            T = T1
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 30.0) * (S4_ * T * (term0 + term1 + term2))

        elif (p, q, r) == (2, 2, 0):
            return 0.0

        elif (p, q, r) == (2, 1, 1):
            return 0.0

        elif (p, q, r) == (2, 0, 2):
            S4_ = S4
            T = T1
            C = (B0 - B2)*(A0*B1 - A1*B0) + (B1 - B0)*(A1*B2 - A2*B1) + (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 30.0) * (S4_ * T * C)

        elif (p, q, r) == (1, 3, 0):
            return 0.0

        elif (p, q, r) == (1, 2, 1):
            return 0.0

        elif (p, q, r) == (1, 1, 2):
            return 0.0

        elif (p, q, r) == (1, 0, 3):
            return 0.0

        elif (p, q, r) == (0, 4, 0):
            return 0.0

        elif (p, q, r) == (0, 3, 1):
            return 0.0

        elif (p, q, r) == (0, 2, 2):
            return 0.0

        elif (p, q, r) == (0, 1, 3):
            return 0.0

        elif (p, q, r) == (0, 0, 4):
            W = T4
            C = (B0 - B2)*(A0*B1 - A1*B0) + (B1 - B0)*(A1*B2 - A2*B1) + (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 30.0) * (W * C)

        else:
            return 0.0

    else:
        # dom == 2 → projection onto x–y plane (“a_i = x_i, b_i = y_i”)
        A0, B0 = a0, b0  # but here a0 == x0, b0 == y0, etc.
        A1, B1 = a1, b1
        A2, B2 = a2, b2
        # Recompute sums:
        S1 = A0 + A1 + A2
        S2 = (A0*A0 + A1*A1 + A2*A2 + A0*A1 + A1*A2 + A2*A0)
        S3 = (
            A0**3
            + A1**3
            + A2**3
            + A0**2*(A1 + A2)
            + A1**2*(A2 + A0)
            + A2**2*(A0 + A1)
            + A0*A1*A2
        )
        S4 = (
            A0**4
            + A1**4
            + A2**4
            + A0**2*(A1*A1 + A2*A2 + A1*A2)
            + A1**2*(A2*A2 + A2*A0 + A0*A0)
            + A2**2*(A0*A0 + A0*A1 + A1*A1)
            + A0*A1*A1*A2
            + A0*A1*A2*A2
            + A0*A2*A2*A1
        )
        T1 = B0 + B1 + B2
        T2 = (B0*B0 + B1*B1 + B2*B2 + B0*B1 + B1*B2 + B2*B0)
        T3 = (
            B0**3
            + B1**3
            + B2**3
            + B0**2*(B1 + B2)
            + B1**2*(B2 + B0)
            + B2**2*(B0 + B1)
            + B0*B1*B2
        )
        T4 = (
            B0**4
            + B1**4
            + B2**4
            + B0**2*(B1*B1 + B2*B2 + B1*B2)
            + B1**2*(B2*B2 + B2*B0 + B0*B0)
            + B2**2*(B0*B0 + B0*B1 + B1*B1)
            + B0*B1*B1*B2
            + B0*B1*B2*B2
            + B0*B2*B2*B1
        )

        if (p, q, r) == (0, 0, 0):
            return 0.5 * ((B0 - B2)*(A0*B1 - A1*B0)+ (B1 - B0)*(A1*B2 - A2*B1)+ (B2 - B1)*(A2*B0 - A0*B2))

        elif (p, q, r) == (1, 0, 0):
            term0 = S1 * (B0 - B2)*(A0*B1 - A1*B0)
            term1 = S1 * (B1 - B0)*(A1*B2 - A2*B1)
            term2 = S1 * (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 6.0) * (term0 + term1 + term2)

        elif (p, q, r) == (0, 1, 0):
            term0 = S1 * (B0 - B2)*(A0*B1 - A1*B0)
            term1 = S1 * (B1 - B0)*(A1*B2 - A2*B1)
            term2 = S1 * (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 6.0) * (term0 + term1 + term2)

        elif (p, q, r) == (0, 0, 1):
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 6.0) * (term0 + term1 + term2)

        elif (p, q, r) == (2, 0, 0):
            A = S1*S1 + (A0*A0 + A1*A1 + A2*A2 + A0*A1 + A1*A2 + A2*A0)
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 12.0) * A * (term0 + term1 + term2)

        elif (p, q, r) == (1, 1, 0):
            T = S2
            term0 = (B0 - B2)*(A0*B1 - A1*B0) * (A0 + A1 + A2)
            term1 = (B1 - B0)*(A1*B2 - A2*B1) * (A1 + A2 + A0)
            term2 = (B2 - B1)*(A2*B0 - A0*B2) * (A2 + A0 + A1)
            return (1.0 / 12.0) * (T * (term0 + term1 + term2))

        elif (p, q, r) == (1, 0, 1):
            T = S2
            U = T1
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 12.0) * (T * (term0 + term1 + term2) * U)

        elif (p, q, r) == (0, 2, 0):
            U = T2
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 12.0) * (U * (term0 + term1 + term2))

        elif (p, q, r) == (0, 1, 1):
            U = T2
            term0 = (B0 - B2)*(A0*B1 - A1*B0) * (A0 + A1 + A2)
            term1 = (B1 - B0)*(A1*B2 - A2*B1) * (A1 + A2 + A0)
            term2 = (B2 - B1)*(A2*B0 - A0*B2) * (A2 + A0 + A1)
            return (1.0 / 12.0) * (U * (term0 + term1 + term2))

        elif (p, q, r) == (0, 0, 2):
            U = T2
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 12.0) * (U * (term0 + term1 + term2))

        elif (p, q, r) == (3, 0, 0):
            P = S3
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 20.0) * P * (term0 + term1 + term2)

        elif (p, q, r) == (2, 1, 0):
            Q = S3
            term0 = (B0 - B2)*(A0*B1 - A1*B0) * (A0 + A1 + A2)
            term1 = (B1 - B0)*(A1*B2 - A2*B1) * (A1 + A2 + A0)
            term2 = (B2 - B1)*(A2*B0 - A0*B2) * (A2 + A0 + A1)
            return (1.0 / 20.0) * (Q * (term0 + term1 + term2))

        elif (p, q, r) == (2, 0, 1):
            Q = S3
            U = T1
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 20.0) * (Q * (term0 + term1 + term2) * U)

        elif (p, q, r) == (1, 2, 0):
            Q = T3
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 20.0) * (Q * (term0 + term1 + term2))

        elif (p, q, r) == (1, 1, 1):
            Q = S3
            R = T3
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 20.0) * (Q * R * (term0 + term1 + term2))

        elif (p, q, r) == (1, 0, 2):
            Q = S3
            R = T3
            return (1.0 / 20.0) * (Q * R * ( (B0 - B2)*(A0*B1 - A1*B0)
                                        + (B1 - B0)*(A1*B2 - A2*B1)
                                        + (B2 - B1)*(A2*B0 - A0*B2) ))

        elif (p, q, r) == (0, 3, 0):
            R = T3
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 20.0) * (R * (term0 + term1 + term2))

        elif (p, q, r) == (0, 2, 1):
            R = T3
            S = S1
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 20.0) * (R * S * (term0 + term1 + term2))

        elif (p, q, r) == (0, 1, 2):
            R = T3
            S = S1
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 20.0) * (R * S * (term0 + term1 + term2))

        elif (p, q, r) == (0, 0, 3):
            R = T3
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 20.0) * (R * (term0 + term1 + term2))

        elif (p, q, r) == (4, 0, 0):
            S4_ = S4
            C = (B0 - B2)*(A0*B1 - A1*B0) + (B1 - B0)*(A1*B2 - A2*B1) + (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 30.0) * S4_ * C

        elif (p, q, r) == (3, 1, 0):
            S4_ = S4
            term0 = (B0 - B2)*(A0*B1 - A1*B0) * S1
            term1 = (B1 - B0)*(A1*B2 - A2*B1) * S1
            term2 = (B2 - B1)*(A2*B0 - A0*B2) * S1
            return (1.0 / 30.0) * (S4_ * (term0 + term1 + term2))

        elif (p, q, r) == (3, 0, 1):
            S4_ = S4
            T = T1
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 30.0) * (S4_ * T * (term0 + term1 + term2))

        elif (p, q, r) == (2, 2, 0):
            S4_ = S4
            T2_ = T2
            term0 = (B0 - B2)*(A0*B1 - A1*B0)
            term1 = (B1 - B0)*(A1*B2 - A2*B1)
            term2 = (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 30.0) * (S4_ * T2_ * (term0 + term1 + term2))

        elif (p, q, r) == (2, 1, 1):
            S4_ = S4
            T2_ = T2
            term0 = (B0 - B2)*(A0*B1 - A1*B0) * S1
            term1 = (B1 - B0)*(A1*B2 - A2*B1) * S1
            term2 = (B2 - B1)*(A2*B0 - A0*B2) * S1
            return (1.0 / 30.0) * (S4_ * T2_ * (term0 + term1 + term2))

        elif (p, q, r) == (2, 0, 2):
            S4_ = S4
            T2_ = T2
            C = (B0 - B2)*(A0*B1 - A1*B0) + (B1 - B0)*(A1*B2 - A2*B1) + (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 30.0) * (S4_ * T2_ * C)

        elif (p, q, r) == (1, 3, 0):
            return 0.0  # includes y^3, which is omitted in x–y projection

        elif (p, q, r) == (1, 2, 1):
            return 0.0

        elif (p, q, r) == (1, 1, 2):
            return 0.0

        elif (p, q, r) == (1, 0, 3):
            return 0.0

        elif (p, q, r) == (0, 4, 0):
            return 0.0

        elif (p, q, r) == (0, 3, 1):
            return 0.0

        elif (p, q, r) == (0, 2, 2):
            return 0.0

        elif (p, q, r) == (0, 1, 3):
            return 0.0

        elif (p, q, r) == (0, 0, 4):
            W = T4
            C = (B0 - B2)*(A0*B1 - A1*B0) + (B1 - B0)*(A1*B2 - A2*B1) + (B2 - B1)*(A2*B0 - A0*B2)
            return (1.0 / 30.0) * (W * C)

        else:
            return 0.0


def _raw_to_central(raw_moments: dict, centroid: np.ndarray, order: int) -> dict:
    Cx, Cy, Cz = centroid
    central = {}

    for total in range(order + 1):
        for p in range(total + 1):
            for q in range(total - p + 1):
                r = total - p - q
                mu_val = 0.0
                for i in range(p + 1):
                    for j in range(q + 1):
                        for k in range(r + 1):
                            M_ijk = raw_moments.get((i, j, k), 0.0)
                            coeff = (
                                comb(p, i)
                                * comb(q, j)
                                * comb(r, k)
                                * ((-Cx) ** (p - i))
                                * ((-Cy) ** (q - j))
                                * ((-Cz) ** (r - k))
                            )
                            mu_val += coeff * M_ijk
                central[(p, q, r)] = mu_val
                #print(f"  μ_{{{p}{q}{r}}} = {mu_val:.6e}")
    return central



# ---------------------------------------------------
# 9.1. Assembling Augmented Matrices with Invariants
# ---------------------------------------------------

def assemble_augmented_matrices(X: np.ndarray,Y: np.ndarray,Z: np.ndarray,invariants: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # print(invariants)
    assert X.shape == (POINTS_PER_CS, NUM_SECTIONS)
    assert Y.shape == (POINTS_PER_CS, NUM_SECTIONS)
    assert Z.shape == (POINTS_PER_CS, NUM_SECTIONS)
    assert invariants.shape == (NUM_INVARIANTS,)

    # Allocate zero‐padded arrays of shape [25 × 57]
    X_aug = np.zeros((POINTS_PER_CS, AUG_COLS))
    Y_aug = np.zeros((POINTS_PER_CS, AUG_COLS))
    Z_aug = np.zeros((POINTS_PER_CS, AUG_COLS))

    # 1) Copy rows 0–23 (i.e. first 24 rows) from columns 0–55
    X_aug[:POINTS_PER_CS-1, :NUM_SECTIONS] = X[:POINTS_PER_CS-1, :]
    Y_aug[:POINTS_PER_CS-1, :NUM_SECTIONS] = Y[:POINTS_PER_CS-1, :]
    Z_aug[:POINTS_PER_CS-1, :NUM_SECTIONS] = Z[:POINTS_PER_CS-1, :]

    # 2) Place the 35 invariants into row 24 (the last row), columns 0–34
    X_aug[POINTS_PER_CS-1, :NUM_INVARIANTS] = invariants
    Y_aug[POINTS_PER_CS-1, :NUM_INVARIANTS] = invariants
    Z_aug[POINTS_PER_CS-1, :NUM_INVARIANTS] = invariants

    # 3) All other entries remain zero by default
    #print(f"[Assemble] Built augmented matrices of shape {X_aug.shape}")
    return X_aug, Y_aug, Z_aug


# -----------------------------------------------------------------------------
# 2) create_sst
# -----------------------------------------------------------------------------
def create_sst(stl_path: str, N: int = POINTS_PER_CS) -> np.ndarray:
    # --- 1) load & normalize
    mesh = load_and_normalize_mesh(stl_path)

    # --- 2) 56 stations
    stations = compute_station_positions()  # np.ndarray of shape (56,)

    # --- 3) geometry matrices
    X, Y, Z, valid = build_geometry_matrices(mesh, stations, num_samples=N)

    # --- 4) invariants
    M = compute_moment_invariants(mesh)

    # --- 5) augment
    X_aug, Y_aug, Z_aug =assemble_augmented_matrices(X,Y,Z, M)
    

    # --- 6) stack
    SST = np.stack([X_aug, Y_aug, Z_aug], axis=0).astype(np.float32)

    # --- 7) min–max to [–1,1]
    mn, mx = SST.min(), SST.max()
    if mx > mn:
        SST = 2 * (SST - mn) / (mx - mn) - 1.0
    return SST

def pad_sst_to_64x64(x: torch.Tensor) -> torch.Tensor:
    # x.shape == (B, 3, 25, 57)
    return F.pad(x, (4, 3, 20, 19))  # → (B, 3, 64, 64)
