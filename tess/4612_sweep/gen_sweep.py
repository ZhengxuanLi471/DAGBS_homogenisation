#!/usr/bin/env python3
"""Family of generalized truncated-trihexagonal (4,6,12) tilings with
varying small-grain size.

One-parameter truncation t of the kagome lattice (gen_hex_seeds.py
machinery): rectangles (small grains, sides t*Lk and sqrt(3)*t*Lk) on
kagome vertices, hexagons on kagome triangles, dodecagons (large grains)
on kagome hexagons. Lk is fixed by the average face area
(sqrt(3)/3)*Lk^2 = 1/50, which is independent of t, so the 2x2-cell
domain is fixed.

Perimeters:
    rectangle  P_r = 2*t*Lk*(1+sqrt(3))
    dodecagon  P_d = 6*Lk*(1 - 2t + sqrt(3)*t)
    ratio r = P_r/P_d  =>  t = 3r / (1 + sqrt(3) + 3r*(2 - sqrt(3)))

t = 1/3 is the generator's uniform tiling, where r = 1/3 exactly.
We sweep 10 log-spaced ratios from 1/3 down to 1/5000.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gen_hex_seeds as g
from cut_paste_rect import cut_and_paste_axis

# ============================================================
N_GEOM = 10
# NB: the generator's uniform tiling (t=1/3, r=1/3) is NOT included: at
# t=1/3 the rectangles' x-intervals exactly tile the period (tangent
# bands), so every axis-aligned cut slices a rectangle — verified
# numerically; the original tth_4612_cut.ply silently sliced 2 of 12.
# r=1/4 is the largest ratio with a clean cut corridor (clearance ~0.011).
R_MAX = 0.25
R_MIN = 1.0 / 5000.0
INDEX_RADIUS = 14
OUT_JSON = "tessellation_output.json"
# ============================================================

sqrt3 = math.sqrt(3.0)
Lk = g.Lk
domain_x = g.CELLS_X * 2.0 * Lk
domain_y = g.CELLS_Y * sqrt3 * Lk


def find_cut(V, F, axis, dom, small_n):
    """Cut position maximizing min distance to any vertex coordinate, among
    lines that do NOT cross any small grain (faces with small_n vertices).

    Candidates are clamped to [eps_above, dom - eps_below], where eps_* are
    the face protrusions beyond the [0, dom] window: cut_and_paste_axis is a
    plain translation (no periodic wrap), so a cut outside this interval
    strands protruding content outside the final box (gaps on one edge,
    overhang on the other)."""
    eps_below = max(0.0, -float(V[:, axis].min()))
    eps_above = max(0.0, float(V[:, axis].max()) - dom)
    lo = max(0.05 * dom, eps_above + 1e-9)
    hi = min(0.95 * dom, dom - eps_below - 1e-9)
    if lo >= hi:
        raise RuntimeError("no admissible cut range (protrusions too large)")
    coords = np.unique(np.round(V[:, axis], 12))
    cand = np.linspace(lo, hi, 4001)
    dmin = np.min(np.abs(cand[:, None] - coords[None, :]), axis=1)
    ok = np.ones(len(cand), dtype=bool)
    for f in F:
        if len(f) != small_n:
            continue
        flo, fhi = V[f, axis].min(), V[f, axis].max()
        # wrap modulo the domain (the cut acts on the periodic cell, and the
        # patch contains protruding duplicates of edge faces)
        ok &= ~(((cand - flo + 1e-9) % dom) < (fhi - flo + 2e-9))
    if not ok.any():
        raise RuntimeError("no admissible cut position")
    dmin[~ok] = -1.0
    i = int(np.argmax(dmin))
    return float(cand[i]), float(dmin[i])


def cut_paste(V, F, cut_x, cut_y):
    verts = np.column_stack([V, np.zeros(len(V))])
    verts, faces = cut_and_paste_axis(verts, F, axis=0, cut_value=cut_x, domain_size=domain_x)
    verts, faces = cut_and_paste_axis(verts, faces, axis=1, cut_value=cut_y, domain_size=domain_y)
    seen = {}
    keep = []
    for i, f in enumerate(faces):
        c = verts[f].mean(axis=0)
        key = (round(c[0] / 1e-6), round(c[1] / 1e-6))
        if key not in seen:
            seen[key] = i
            keep.append(i)
    faces = [faces[i] for i in keep]
    verts = verts.copy()
    verts[:, 0] -= domain_x / 2
    verts[:, 1] -= domain_y / 2
    return verts, faces


def check_box_tiling(pts, regions, dx, dy, n_sample=2000):
    """All vertices inside the centered box, and every interior sample point
    covered by exactly one grain (catches protrusion/gap defects that area
    and boundary-pairing checks miss)."""
    from matplotlib.path import Path
    pts_a = np.asarray(pts)
    if (np.abs(pts_a[:, 0]) > dx / 2 + 1e-6).any() or \
       (np.abs(pts_a[:, 1]) > dy / 2 + 1e-6).any():
        raise RuntimeError("vertices protrude outside the domain box")
    paths = [Path(pts_a[np.array(r) - 1]) for r in regions]
    rng = np.random.default_rng(0)
    sample = rng.uniform([-dx/2 + 1e-4, -dy/2 + 1e-4],
                         [dx/2 - 1e-4, dy/2 - 1e-4], size=(n_sample, 2))
    cover = np.zeros(n_sample, dtype=int)
    for p in paths:
        cover += p.contains_points(sample)
    n_bad = int((cover != 1).sum())
    if n_bad:
        raise RuntimeError(f"{n_bad}/{n_sample} sample points not covered "
                           f"exactly once (gaps or overlaps)")


def faces_to_regions(verts, faces):
    used = sorted(set(v for f in faces for v in f))
    old2new = {old: i for i, old in enumerate(used)}
    pts = [(verts[v][0], verts[v][1]) for v in used]
    regions = []
    for f in faces:
        idx = [old2new[v] for v in f]
        p = np.array([pts[i] for i in idx])
        cx, cy = p.mean(axis=0)
        order = np.argsort(np.arctan2(p[:, 1] - cy, p[:, 0] - cx))
        regions.append([idx[k] + 1 for k in order])
    return pts, regions


def main():
    ratios = np.geomspace(R_MAX, R_MIN, N_GEOM)
    data = {}
    print(f"Lk = {Lk:.8f}, domain {domain_x:.6f} x {domain_y:.6f}")

    print("Building kagome lattice (t-independent)...")
    kpts = g.build_kagome_points(INDEX_RADIUS)
    k_edges = g.build_nn_edges(kpts)
    print(f"  {len(kpts)} points, {len(k_edges)} edges")

    for r in ratios:
        t = 3.0 * r / (1.0 + sqrt3 + 3.0 * r * (2.0 - sqrt3))
        g.t = t  # truncate_kagome reads the module global at call time
        label = f"ratio_{r:.3e}"
        vtx, edges = g.truncate_kagome(kpts, k_edges)
        faces = g.extract_faces(vtx, edges)
        V, F = g.select_rectangular_faces(vtx, faces, g.CELLS_X, g.CELLS_Y)
        cut_x, clear_x = find_cut(V, F, 0, domain_x, small_n=4)
        cut_y, clear_y = find_cut(V, F, 1, domain_y, small_n=4)
        verts, faces = cut_paste(V, F, cut_x, cut_y)
        pts, regions = faces_to_regions(verts, faces)
        check_box_tiling(pts, regions, domain_x, domain_y)
        data[label] = [pts, regions]

        areas, perims, nvtx = [], [], []
        for reg in regions:
            p = np.array([pts[i - 1] for i in reg])
            x, y = p[:, 0], p[:, 1]
            areas.append(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            perims.append(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1).sum())
            nvtx.append(len(reg))
        areas, perims, nvtx = map(np.array, (areas, perims, nvtx))
        P_rect = 2.0 * t * Lk * (1.0 + sqrt3)
        P_dod = 6.0 * Lk * (1.0 - 2.0 * t + sqrt3 * t)
        intact = (nvtx == 4) & (np.abs(perims - P_rect) < 0.02 * P_rect)
        n_rect = int(intact.sum())
        if n_rect != 3 * g.CELLS_X * g.CELLS_Y:
            raise RuntimeError(f"{label}: expected {3*g.CELLS_X*g.CELLS_Y} intact "
                               f"rectangles, found {n_rect}")
        achieved = float(np.median(perims[intact])) / P_dod
        print(f"{label}: t={t:.4e} | {len(regions)} grains ({n_rect} rectangles intact), "
              f"area={areas.sum():.8f}, CCW={bool((areas > 0).all())}, "
              f"cut=({cut_x:.4f},{cut_y:.4f}) clear=({clear_x:.4f},{clear_y:.4f}) | "
              f"achieved r={achieved:.4e} (target {r:.4e})")

    with open(OUT_JSON, "w") as fh:
        json.dump(data, fh)
    print(f"\n{len(data)} geometries written to {OUT_JSON}")


if __name__ == "__main__":
    main()
