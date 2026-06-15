#!/usr/bin/env python3
"""Family of generalized truncated-hexagonal (3,12,12) tilings with varying
small-grain size.

One-parameter truncation t of the regular hexagonal tiling: dodecagons
(large grains) on hexagon centers, triangles (small grains) on honeycomb
vertices. With base-hexagon edge L fixed the average face area is
(sqrt(3)/2)*L^2 = 1/50 and the 2x2-supercell domain is fixed, independent
of t.

Perimeters:
    triangle   P_t = 3*sqrt(3)*t*L
    dodecagon  P_d = 6*L*(1 - 2t + sqrt(3)*t)
    ratio r = P_t/P_d  =>  t = 2r / (sqrt(3) + 4r - sqrt(3)*2r)

t = 1/(2+sqrt(3)) is the Archimedean tiling, where r = 1/4 exactly.
We sweep 10 log-spaced ratios from 1/4 down to 1/5000.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cut_paste_rect import cut_and_paste_axis

# ============================================================
CELLS_X = 2
CELLS_Y = 2
AVG_AREA = 1.0 / 50.0
N_GEOM = 10
R_MAX = 0.25         # Archimedean (3,12,12)
R_MIN = 1.0 / 5000.0
OUT_JSON = "tessellation_output.json"
# ============================================================

sqrt3 = math.sqrt(3.0)
L = math.sqrt(2.0 * AVG_AREA / sqrt3)
a = sqrt3 * L
Ax = 3.0 * L
Ay = sqrt3 * L
domain_x = CELLS_X * Ax
domain_y = CELLS_Y * Ay
R_hex = L


def build_tiling(t, margin=3):
    vtx_map = {}
    vertices = []
    faces = []
    tol = L * 1e-9

    def key_xy(x, y):
        return (round(x / tol) * tol, round(y / tol) * tol)

    def get_vtx(x, y):
        key = key_xy(x, y)
        if key not in vtx_map:
            vtx_map[key] = len(vertices)
            vertices.append(np.array([x, y]))
        return vtx_map[key]

    def add_dodecagon(cx, cy):
        hex_v = [np.array([cx + R_hex * math.cos(math.pi / 3.0 * k),
                           cy + R_hex * math.sin(math.pi / 3.0 * k)]) for k in range(6)]
        dodec = []
        for k in range(6):
            v0, v1 = hex_v[k], hex_v[(k + 1) % 6]
            p0 = v0 + t * (v1 - v0)
            p1 = v1 + t * (v0 - v1)
            dodec.append(get_vtx(p0[0], p0[1]))
            dodec.append(get_vtx(p1[0], p1[1]))
        faces.append(dodec)

    def add_triangle(tx, ty, base_angle):
        circumradius = t * L
        tri = []
        for k in range(3):
            ang = base_angle + k * (2.0 * math.pi / 3.0)
            tri.append(get_vtx(tx + circumradius * math.cos(ang),
                               ty + circumradius * math.sin(ang)))
        faces.append(tri)

    lo = -margin
    hi_x = CELLS_X + margin
    hi_y = CELLS_Y + margin

    for J in range(lo, hi_y):
        for I in range(lo, hi_x):
            ox, oy = I * Ax, J * Ay
            add_dodecagon(ox, oy)
            add_dodecagon(ox + Ax / 2.0, oy + Ay / 2.0)

    tri_done = set()
    for J in range(lo, hi_y):
        for I in range(lo, hi_x):
            ox, oy = I * Ax, J * Ay
            for cx, cy in [(ox, oy), (ox + Ax / 2.0, oy + Ay / 2.0)]:
                for k in range(6):
                    ang = math.pi / 3.0 * k
                    tx = cx + R_hex * math.cos(ang)
                    ty = cy + R_hex * math.sin(ang)
                    tkey = key_xy(tx, ty)
                    if tkey not in tri_done:
                        tri_done.add(tkey)
                        add_triangle(tx, ty, ang)

    return np.array(vertices), faces


def select_faces(vtx, faces):
    """Half-open centroid window [0,dx) x [0,dy): one copy per periodic cell."""
    eps = L * 1e-6
    cents = np.array([vtx[f].mean(axis=0) for f in faces])
    chosen = [fi for fi, c in enumerate(cents)
              if -eps <= c[0] < domain_x - eps and -eps <= c[1] < domain_y - eps]
    kept = [faces[i] for i in chosen]
    used = sorted(set(v for f in kept for v in f))
    old2new = {old: i for i, old in enumerate(used)}
    V = vtx[used].copy()
    F = [[old2new[v] for v in f] for f in kept]
    return V, F


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
    print(f"L = {L:.8f}, domain {domain_x:.6f} x {domain_y:.6f}")
    for r in ratios:
        t = 2.0 * r / (sqrt3 + 4.0 * r - 2.0 * sqrt3 * r)
        label = f"ratio_{r:.3e}"
        V, F = build_tiling(t)
        V, F = select_faces(V, F)
        cut_x, clear_x = find_cut(V, F, 0, domain_x, small_n=3)
        cut_y, clear_y = find_cut(V, F, 1, domain_y, small_n=3)
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
        P_tri = 3.0 * sqrt3 * t * L
        P_dod = 6.0 * L * (1.0 - 2.0 * t + sqrt3 * t)
        intact = (nvtx == 3) & (np.abs(perims - P_tri) < 0.02 * P_tri)
        n_tri = int(intact.sum())
        if n_tri != 4 * CELLS_X * CELLS_Y:
            raise RuntimeError(f"{label}: expected {4*CELLS_X*CELLS_Y} intact "
                               f"triangles, found {n_tri}")
        achieved = float(np.median(perims[intact])) / P_dod
        print(f"{label}: t={t:.4e} | {len(regions)} grains ({n_tri} triangles intact), "
              f"area={areas.sum():.8f}, CCW={bool((areas > 0).all())}, "
              f"cut=({cut_x:.4f},{cut_y:.4f}) clear=({clear_x:.4f},{clear_y:.4f}) | "
              f"achieved r={achieved:.4e} (target {r:.4e})")

    with open(OUT_JSON, "w") as fh:
        json.dump(data, fh)
    print(f"\n{len(data)} geometries written to {OUT_JSON}")


if __name__ == "__main__":
    main()
