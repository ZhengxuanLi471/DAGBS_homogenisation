#!/usr/bin/env python3
"""Family of generalized truncated-square (4,8,8) tilings with varying
small-grain size.

The regular (4,8,8) tiling is the t-midpoint of a one-parameter truncation
family of the square lattice: octagons (large grains) on cell centers and
tilted squares (small grains) on lattice points. With lattice spacing
a = 0.2 the average face area is a^2/2 = 1/50 and the 5x5-cell domain is
1 x 1, independent of truncation.

Parametrization (h = a/2):
    u = h - q            half-diagonal of the small square
    small-square edge    u*sqrt(2),  perimeter P_s = 4*sqrt(2)*u
    octagon perimeter    P_o = 8*q + 4*sqrt(2)*u
    ratio r = P_s/P_o    =>  u = 2*r*h / (sqrt(2) + 2*r - sqrt(2)*r)

r = 1/2 reproduces the regular Archimedean tiling (all edges equal);
r -> 0 shrinks the small squares to points. We sweep 10 log-spaced ratios
from 1/2 down to 1/5000 and collect all cut-paste geometries into a single
tessellation_output.json keyed by the ratio.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cut_paste_rect import cut_and_paste_axis

# ============================================================
CELLS_X = 5
CELLS_Y = 5
AVG_AREA = 1.0 / 50.0
N_GEOM = 10
R_MAX = 0.5          # regular (4,8,8)
R_MIN = 1.0 / 5000.0
CUT = 0.1            # octagon centerline, clear of vertices family-wide
OUT_JSON = "tessellation_output.json"
# ============================================================

sqrt2 = math.sqrt(2.0)
a = math.sqrt(2.0 * AVG_AREA)   # 0.2
h = a / 2.0
domain_x = CELLS_X * a
domain_y = CELLS_Y * a


def build_tiling(u, margin=3):
    """Generalized truncated-square tiling with small-square half-diagonal u."""
    q = h - u
    vtx_map = {}
    vertices = []
    faces = []
    tol = u * 1e-6

    def get_vtx(x, y):
        key = (round(x / tol) * tol, round(y / tol) * tol)
        if key not in vtx_map:
            vtx_map[key] = len(vertices)
            vertices.append(np.array([x, y]))
        return vtx_map[key]

    lo = -margin
    hi_x = CELLS_X + margin
    hi_y = CELLS_Y + margin

    # Octagons centred at ((i+0.5)a, (j+0.5)a)
    for j in range(lo, hi_y):
        for i in range(lo, hi_x):
            cx = (i + 0.5) * a
            cy = (j + 0.5) * a
            faces.append([
                get_vtx(cx + q, cy + h),
                get_vtx(cx + h, cy + q),
                get_vtx(cx + h, cy - q),
                get_vtx(cx + q, cy - h),
                get_vtx(cx - q, cy - h),
                get_vtx(cx - h, cy - q),
                get_vtx(cx - h, cy + q),
                get_vtx(cx - q, cy + h),
            ])

    # Small tilted squares centred at (I*a, J*a)
    for J in range(lo, hi_y + 1):
        for I in range(lo, hi_x + 1):
            sx = I * a
            sy = J * a
            faces.append([
                get_vtx(sx,     sy + u),
                get_vtx(sx + u, sy),
                get_vtx(sx,     sy - u),
                get_vtx(sx - u, sy),
            ])

    return np.array(vertices), faces


def select_faces(vtx, faces):
    pad = 0.3 * a
    cents = np.array([vtx[f].mean(axis=0) for f in faces])
    chosen = [fi for fi, c in enumerate(cents)
              if -pad <= c[0] <= domain_x + pad and -pad <= c[1] <= domain_y + pad]
    kept = [faces[i] for i in chosen]
    used = sorted(set(v for f in kept for v in f))
    old2new = {old: i for i, old in enumerate(used)}
    V = vtx[used].copy()
    F = [[old2new[v] for v in f] for f in kept]
    return V, F


def cut_paste(V, F):
    """In-memory equivalent of cut_paste_rectangular (cut x, cut y, dedup, center)."""
    verts = np.column_stack([V, np.zeros(len(V))])
    faces = F
    verts, faces = cut_and_paste_axis(verts, faces, axis=0, cut_value=CUT, domain_size=domain_x)
    verts, faces = cut_and_paste_axis(verts, faces, axis=1, cut_value=CUT, domain_size=domain_y)

    # dedup faces by centroid (periodic duplicates from the selection pad)
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


def faces_to_regions(verts, faces):
    """pts + 1-based CCW regions in tessellation_output.json format."""
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


def region_perimeters(pts, regions):
    pts = np.asarray(pts)
    out = []
    for r in regions:
        p = pts[np.array(r) - 1]
        out.append(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1).sum())
    return np.array(out)


def main():
    ratios = np.geomspace(R_MAX, R_MIN, N_GEOM)
    data = {}
    print(f"a = {a}, h = {h}, domain {domain_x} x {domain_y}, cut at {CUT}")
    for r in ratios:
        u = 2.0 * r * h / (sqrt2 + 2.0 * r - sqrt2 * r)
        q = h - u
        label = f"ratio_{r:.3e}"
        V, F = build_tiling(u)
        V, F = select_faces(V, F)
        verts, faces = cut_paste(V, F)
        pts, regions = faces_to_regions(verts, faces)
        data[label] = [pts, regions]

        # verify: total area + achieved perimeter ratio (intact grains only)
        areas = []
        for reg in regions:
            p = np.array([pts[i - 1] for i in reg])
            x, y = p[:, 0], p[:, 1]
            areas.append(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        areas = np.array(areas)
        perims = region_perimeters(pts, regions)
        small = perims.min()
        octagon = 8.0 * q + 4.0 * sqrt2 * u
        print(f"{label}: u={u:.4e} q={q:.6f} | {len(regions)} grains, "
              f"area={areas.sum():.8f}, all CCW={bool((areas > 0).all())} | "
              f"P_small={small:.4e}, P_oct={octagon:.4e}, "
              f"achieved r={small / octagon:.4e} (target {r:.4e})")

    with open(OUT_JSON, "w") as fh:
        json.dump(data, fh)
    print(f"\n{len(data)} geometries written to {OUT_JSON}")


if __name__ == "__main__":
    main()
