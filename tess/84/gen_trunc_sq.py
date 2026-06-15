#!/usr/bin/env python3
"""
Generate a truncated square tiling (4,8,8) — regular octagons + squares.

The tiling has a SQUARE Bravais lattice with cell spacing  a = s(1+√2),
where s is the common edge length.  Each unit cell contains exactly
2 faces (1 octagon + 1 square), so:

    avg face area = a² / 2

Given a target average area we solve for s and a.

Produces a PLY file ready for cut_paste_rect.py.
"""
import math
import numpy as np

# ============================================================
# User knobs
# ============================================================
CELLS_X = 5               # unit cells in x
CELLS_Y = 5               # unit cells in y
AVG_AREA = 1.0 / 50.0     # desired average face area
OUT = "trunc_sq_5x5.ply"
# ============================================================

sqrt2 = math.sqrt(2.0)

# Derived geometry ------------------------------------------------
# avg face area = a²/2  =>  a = sqrt(2 * AVG_AREA)
a = math.sqrt(2.0 * AVG_AREA)
s = a / (1.0 + sqrt2)          # edge length (all edges equal)
h = a / 2.0                    # half-cell = octagon apothem
q = s / 2.0                    # half edge length
d = s * sqrt2 / 2.0            # = h - q, half-diagonal of square face


# -----------------------------------------------------------------
def build_tiling(cells_x, cells_y, margin=3):
    """
    Build a truncated square tiling over a grid with `margin` extra cells
    on every side so that the target rectangle is fully covered.

    Octagon centres sit at  ((i+½)a, (j+½)a).
    Square  centres sit at  (i·a,   j·a).
    """
    vtx_map = {}          # rounded-coord key -> vertex index
    vertices = []
    faces = []
    tol = s * 1e-9

    def get_vtx(x, y):
        kx = round(x / tol) * tol
        ky = round(y / tol) * tol
        key = (kx, ky)
        if key not in vtx_map:
            vtx_map[key] = len(vertices)
            vertices.append(np.array([x, y]))
        return vtx_map[key]

    lo = -margin
    hi_x = cells_x + margin
    hi_y = cells_y + margin

    # --- Octagons centred at ((i+0.5)*a, (j+0.5)*a) ---
    for j in range(lo, hi_y):
        for i in range(lo, hi_x):
            cx = (i + 0.5) * a
            cy = (j + 0.5) * a
            # 8 vertices, CW from top-right  (v0 .. v7)
            v = [
                get_vtx(cx + q, cy + h),    # v0  top-right
                get_vtx(cx + h, cy + q),    # v1  right-top
                get_vtx(cx + h, cy - q),    # v2  right-bottom
                get_vtx(cx + q, cy - h),    # v3  bottom-right
                get_vtx(cx - q, cy - h),    # v4  bottom-left
                get_vtx(cx - h, cy - q),    # v5  left-bottom
                get_vtx(cx - h, cy + q),    # v6  left-top
                get_vtx(cx - q, cy + h),    # v7  top-left
            ]
            faces.append(v)

    # --- Small squares centred at (I*a, J*a) ---
    for J in range(lo, hi_y + 1):
        for I in range(lo, hi_x + 1):
            sx = I * a
            sy = J * a
            # 4 vertices, CW from north
            v = [
                get_vtx(sx,     sy + d),    # north
                get_vtx(sx + d, sy),        # east
                get_vtx(sx,     sy - d),    # south
                get_vtx(sx - d, sy),        # west
            ]
            faces.append(v)

    return np.array(vertices), faces


# -----------------------------------------------------------------
def select_rectangular_faces(vtx, faces, cells_x, cells_y):
    """
    Keep faces whose centroids lie within the periodic domain
    [0, domain_x] × [0, domain_y]  plus a small margin so that
    boundary faces are included for the cut-paste step.
    """
    domain_x = cells_x * a
    domain_y = cells_y * a
    pad = 0.6 * s

    cents = np.array([vtx[f].mean(axis=0) for f in faces])

    chosen = [
        fi for fi, c in enumerate(cents)
        if -pad <= c[0] <= domain_x + pad
        and -pad <= c[1] <= domain_y + pad
    ]
    print(f"  rectangular selection: {len(chosen)} faces")

    kept = [faces[i] for i in chosen]

    # reindex vertices
    used = sorted(set(v for f in kept for v in f))
    old2new = {old: i for i, old in enumerate(used)}
    V = vtx[used].copy()
    F = [[old2new[v] for v in f] for f in kept]

    # shift min corner to (0, 0)
    mins = V.min(axis=0)
    V -= mins

    return V, F


# -----------------------------------------------------------------
def write_ply(path, V, F):
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(V)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write(f"element face {len(F)}\n")
        fh.write("property list uchar int vertex_indices\n")
        fh.write("end_header\n")
        for x, y in V:
            fh.write(f"{x:.12e} {y:.12e} 0.000000000000e+00\n")
        for face in F:
            fh.write(f"{len(face)} " + " ".join(map(str, face)) + "\n")


# -----------------------------------------------------------------
def main():
    domain_x = CELLS_X * a
    domain_y = CELLS_Y * a
    n_faces = 2 * CELLS_X * CELLS_Y

    print(f"Truncated square tiling (4,8,8)")
    print(f"  edge length  s = {s:.8f}")
    print(f"  cell spacing a = {a:.8f}")
    print(f"  domain: {domain_x:.6f} x {domain_y:.6f}  ({n_faces} faces)")
    print(f"  avg face area: {AVG_AREA:.6f}")

    print("Building tiling grid...")
    vtx, faces = build_tiling(CELLS_X, CELLS_Y)
    print(f"  total: {len(vtx)} vertices, {len(faces)} faces")

    print("Selecting rectangular patch...")
    V, F = select_rectangular_faces(vtx, faces, CELLS_X, CELLS_Y)
    print(f"  selected: {len(V)} vertices, {len(F)} faces")

    write_ply(OUT, V, F)
    print(f"Written: {OUT}")

    cut_x = domain_x / 6.0
    cut_y = domain_y / 6.0
    print(f"\nNext step:")
    print(f"  python cut_paste_rect.py {OUT} trunc_sq_cut.ply "
          f"--domain-x {domain_x:.6f} --domain-y {domain_y:.6f} "
          f"--cut-x {cut_x:.4f} --cut-y {cut_y:.4f}")


if __name__ == "__main__":
    main()
