#!/usr/bin/env python3
"""
Generate the truncated hexagonal tiling (3,12,12) — regular dodecagons + triangles.

This is the *uniform truncation* of the regular hexagonal tiling (6^3).
Properties:
  - Only triple junctions (3 faces meet at every vertex)
  - The largest tiles (dodecagons) touch each other by edges (there are 12–12 edges)

Geometry setup (all edges equal to s):
  Start from a regular hexagon of edge length L, then truncate each vertex at distance tL
  along each incident edge. Uniformity requires:
      (1 - 2t)L  ==  (sqrt(3) t) L   =>  t = 1 / (2 + sqrt(3))

We tile using a rectangular supercell compatible with a triangular (hex) lattice:
  Ax = a
  Ay = sqrt(3) * a
where a is nearest-neighbour spacing of hexagon centers:
  a = sqrt(3) * L

Each rectangular cell contains:
  2 dodecagons + 4 triangles = 6 faces
So:
  avg face area = (Ax * Ay) / 6 = (sqrt(3)/2) * L^2
=>  L = sqrt( 2 * AVG_AREA / sqrt(3) )

Produces a PLY file ready for cut_paste_rect.py.
"""
import math
import numpy as np

# ============================================================
# User knobs
# ============================================================
CELLS_X = 2               # rectangular supercells in x
CELLS_Y = 2               # rectangular supercells in y
AVG_AREA = 1.0 / 50.0     # desired average face area
OUT = "trunc_hex_5x5.ply"
# ============================================================

sqrt3 = math.sqrt(3.0)

# ---------------------------------------------------------------------
# Solve for base-hexagon edge length L from avg face area
# avg face area = (sqrt3/2) * L^2
L = math.sqrt(2.0 * AVG_AREA / sqrt3)

# Uniform truncation parameter for 3.12.12
t = 1.0 / (2.0 + sqrt3)

# All edges in the truncated tiling have equal length s
s = (1.0 - 2.0 * t) * L   # also equals sqrt(3)*t*L

# Hexagon-center spacing (triangular lattice constant)
a = sqrt3 * L

# Rectangular supercell dimensions (periodic by pure x/y translations)
# For pointy-right hexagons (vertices at 0°, 60°, ...), the lattice
# neighbours are at 30°, 90°, ... so:
#   Ax = sqrt3 * a = 3L   (two lattice spacings projected onto x)
#   Ay = a         = sqrt3*L
Ax = sqrt3 * a   # = 3L
Ay = a            # = sqrt3 * L

# Circumradius of the *original* hexagon is L (for a regular hexagon)
R = L

# Tolerances for vertex dedup
tol = s * 1e-9 if s != 0 else 1e-12


# ---------------------------------------------------------------------
def build_tiling(cells_x, cells_y, margin=3):
    """
    Build truncated hexagonal tiling over a grid with `margin` extra cells
    so the target rectangle is fully covered.

    We place hexagon centers (which become dodecagon centers) on a triangular lattice,
    but we use a rectangular supercell:
        cell origin = (I*Ax, J*Ay)
        centers at origin + (0, 0) and origin + (Ax/2, Ay/2)
    """
    vtx_map = {}          # rounded-coord key -> vertex index
    vertices = []
    faces = []            # list of faces (each face is list of vertex indices)

    def key_xy(x, y):
        kx = round(x / tol) * tol
        ky = round(y / tol) * tol
        return (kx, ky)

    def get_vtx(x, y):
        key = key_xy(x, y)
        if key not in vtx_map:
            vtx_map[key] = len(vertices)
            vertices.append(np.array([x, y], dtype=float))
        return vtx_map[key]

    def add_dodecagon(cx, cy):
        """
        Create one dodecagon by truncating the original hexagon at center (cx, cy).
        """
        # Original hexagon vertices CCW
        hex_v = []
        for k in range(6):
            ang = (math.pi / 3.0) * k  # 0, 60, 120, ...
            hex_v.append(np.array([cx + R * math.cos(ang), cy + R * math.sin(ang)]))

        # Dodecagon vertices: for each edge (k -> k+1), take two points near each end
        dodec = []
        for k in range(6):
            v0 = hex_v[k]
            v1 = hex_v[(k + 1) % 6]

            p0 = v0 + t * (v1 - v0)   # near v0
            p1 = v1 + t * (v0 - v1)   # near v1

            dodec.append(get_vtx(p0[0], p0[1]))
            dodec.append(get_vtx(p1[0], p1[1]))

        faces.append(dodec)

    def add_triangle(tx, ty, base_angle):
        """
        Create one equilateral triangle centred at honeycomb vertex (tx, ty).

        Each original hexagon vertex becomes a triangle after truncation.
        The 3 triangle vertices sit at distance t*L from the centre along the
        3 hexagonal edges that met at that vertex, separated by 120°.
        base_angle is the angle (from positive x) of the first edge direction.
        Vertices are listed CCW.
        """
        circumradius = t * L
        tri = []
        for k in range(3):
            ang = base_angle + k * (2.0 * math.pi / 3.0)
            vx = tx + circumradius * math.cos(ang)
            vy = ty + circumradius * math.sin(ang)
            tri.append(get_vtx(vx, vy))
        faces.append(tri)

    lo = -margin
    hi_x = cells_x + margin
    hi_y = cells_y + margin

    # --- Dodecagons (centres on triangular lattice, packed into rectangular supercells) ---
    for J in range(lo, hi_y):
        for I in range(lo, hi_x):
            ox = I * Ax
            oy = J * Ay

            # Two centres per rectangular supercell
            add_dodecagon(ox + 0.0,      oy + 0.0)
            add_dodecagon(ox + Ax / 2.0, oy + Ay / 2.0)

    # --- Triangles (one at each original honeycomb vertex) ---
    # Each hex centre has 6 vertices; each vertex is shared by 3 hexagons.
    # We deduplicate by tracking which positions have already been drawn.
    tri_done = set()
    for J in range(lo, hi_y):
        for I in range(lo, hi_x):
            ox = I * Ax
            oy = J * Ay
            for cx, cy in [(ox, oy), (ox + Ax / 2.0, oy + Ay / 2.0)]:
                for k in range(6):
                    ang = (math.pi / 3.0) * k
                    tx = cx + R * math.cos(ang)
                    ty = cy + R * math.sin(ang)
                    tkey = key_xy(tx, ty)
                    if tkey not in tri_done:
                        tri_done.add(tkey)
                        add_triangle(tx, ty, ang)

    return np.array(vertices), faces


# ---------------------------------------------------------------------
def select_rectangular_faces(vtx, faces, cells_x, cells_y):
    """
    Keep faces whose centroids lie within [0, domain_x) × [0, domain_y).
    Half-open intervals guarantee exactly one copy per periodic cell.
    """
    domain_x = cells_x * Ax
    domain_y = cells_y * Ay
    eps = s * 1e-6

    cents = np.array([vtx[f].mean(axis=0) for f in faces])

    chosen = [
        fi for fi, c in enumerate(cents)
        if -eps <= c[0] < domain_x - eps
        and -eps <= c[1] < domain_y - eps
    ]
    print(f"  rectangular selection: {len(chosen)} faces")

    kept = [faces[i] for i in chosen]

    # reindex vertices
    used = sorted(set(v for f in kept for v in f))
    old2new = {old: i for i, old in enumerate(used)}
    V = vtx[used].copy()
    F = [[old2new[v] for v in f] for f in kept]

    # Do NOT shift min corner to (0,0) — the domain must stay at
    # [0, domain_x) × [0, domain_y) so cut_paste_rect can clip correctly.

    return V, F, domain_x, domain_y


# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
def main():
    # Expected faces per rectangular supercell: 6 = 2 dodecagons + 4 triangles
    n_faces_expected = 6 * CELLS_X * CELLS_Y

    print("Truncated hexagonal tiling (3,12,12)")
    print(f"  base hex edge   L  = {L:.8f}")
    print(f"  trunc param     t  = {t:.10f}")
    print(f"  edge length     s  = {s:.8f}")
    print(f"  center spacing  a  = {a:.8f}")
    print(f"  supercell:  Ax = {Ax:.8f},  Ay = {Ay:.8f}")
    print(f"  avg face area target: {AVG_AREA:.8e}")
    print(f"  expected faces in domain (ideal): {n_faces_expected}")

    print("Building tiling grid...")
    vtx, faces = build_tiling(CELLS_X, CELLS_Y)
    print(f"  total: {len(vtx)} vertices, {len(faces)} faces (before selection)")

    print("Selecting rectangular patch...")
    V, F, domain_x, domain_y = select_rectangular_faces(vtx, faces, CELLS_X, CELLS_Y)
    print(f"  selected: {len(V)} vertices, {len(F)} faces")

    write_ply(OUT, V, F)
    print(f"Written: {OUT}")

    # Cut positions: L/2 in x and a/6 in y — guaranteed to be inside
    # a dodecagon, not near any vertex or edge of the tiling.
    cut_x = L / 2.0
    cut_y = a / 6.0
    print("\nNext step:")
    print(
        f"  python cut_paste_rect.py {OUT} trunc_hex_cut.ply "
        f"--domain-x {domain_x:.6f} --domain-y {domain_y:.6f} "
        f"--cut-x {cut_x:.4f} --cut-y {cut_y:.4f}"
    )


if __name__ == "__main__":
    main()
