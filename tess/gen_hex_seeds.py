#!/usr/bin/env python3
import math
import numpy as np

# ============================================================
# User knobs
# ============================================================
CELLS_X = 2            # 2x2 unit-cell patch
CELLS_Y = 2
AVG_AREA = 1.0 / 50.0  # desired average face area
OUT = "tth_4612_2x2.ply"
# ============================================================

sqrt3 = math.sqrt(3.0)

# (4,6,12) tiling: 6 faces per unit cell, avg face area = 3*sqrt3 * s^2
# Solve for s from AVG_AREA = 3*sqrt3 * s^2
s = math.sqrt(AVG_AREA / (3.0 * sqrt3))

# Uniform truncation parameter that yields the Archimedean (4,6,12) tiling
t = 1.0 / 3.0

# Kagome nearest-neighbour length Lk. For t=1/3, all new edges have length Lk/3 = s.
Lk = 3.0 * s

# IMPORTANT: Kagome convenient triangular Bravais lattice with |a1| = 2*Lk
# so that within-cell basis distances are exactly Lk.
a1 = np.array([2.0 * Lk, 0.0])
a2 = np.array([1.0 * Lk, sqrt3 * Lk])

# Kagome basis (3 sites per triangular cell)
b0 = np.array([0.0, 0.0])
b1 = 0.5 * a1                    # (Lk, 0)
b2 = 0.5 * a2                    # (0.5Lk, 0.5*sqrt3*Lk)
basis = [b0, b1, b2]


def build_kagome_points(index_radius: int):
    """
    Build kagome vertices on a triangular lattice for i,j in [-R,R], with 3 basis points per cell.
    Uses exact floats (multiples of s), no rounding needed.
    """
    pts = []
    for j in range(-index_radius, index_radius + 1):
        for i in range(-index_radius, index_radius + 1):
            R = i * a1 + j * a2
            for b in basis:
                pts.append(R + b)
    return np.asarray(pts)


def build_nn_edges(points: np.ndarray):
    """
    Build nearest-neighbour edges at distance Lk using spatial hashing.
    """
    cell = 1.10 * Lk
    grid = {}
    for idx, p in enumerate(points):
        gx = int(math.floor(p[0] / cell))
        gy = int(math.floor(p[1] / cell))
        grid.setdefault((gx, gy), []).append(idx)

    def candidates(p):
        gx = int(math.floor(p[0] / cell))
        gy = int(math.floor(p[1] / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield from grid.get((gx + dx, gy + dy), [])

    edges = set()
    tol = 1e-6 * Lk
    for i, pi in enumerate(points):
        for j in candidates(pi):
            if j <= i:
                continue
            d = np.linalg.norm(points[j] - pi)
            if abs(d - Lk) <= tol:
                edges.add((i, j))
    return list(edges)


def truncate_kagome(kpts: np.ndarray, k_edges):
    """
    Uniform truncation: for each directed kagome edge u->v create cutpoint near u.
    New graph edges:
      - middle segment on each kagome edge
      - square edges around each kagome vertex (requires degree 4)
    """
    adj = [[] for _ in range(len(kpts))]
    for u, v in k_edges:
        adj[u].append(v)
        adj[v].append(u)

    vtx = []
    cutpt = {}  # (u,v) -> new vertex index in vtx

    def new_vertex(p):
        vtx.append(p)
        return len(vtx) - 1

    # create cutpoints for each directed edge (u->v)
    for u, v in k_edges:
        pu, pv = kpts[u], kpts[v]
        iu = new_vertex(pu + t * (pv - pu))
        iv = new_vertex(pv + t * (pu - pv))
        cutpt[(u, v)] = iu
        cutpt[(v, u)] = iv

    edges = set()

    # middle segments (between the two cutpoints on the same original edge)
    for u, v in k_edges:
        a = cutpt[(u, v)]
        b = cutpt[(v, u)]
        if a != b:
            edges.add((min(a, b), max(a, b)))

    # squares around each kagome vertex (kagome degree should be 4 in the interior)
    for u, nbrs in enumerate(adj):
        if len(nbrs) != 4:
            continue
        pu = kpts[u]
        pts = []
        for v in nbrs:
            pid = cutpt[(u, v)]
            p = vtx[pid]
            ang = math.atan2(p[1] - pu[1], p[0] - pu[0])
            pts.append((ang, pid))
        pts.sort()
        cyc = [pid for _, pid in pts]
        for k in range(4):
            a = cyc[k]
            b = cyc[(k + 1) % 4]
            if a != b:
                edges.add((min(a, b), max(a, b)))

    return np.asarray(vtx), edges


def extract_faces(vtx: np.ndarray, edges):
    """
    Planar face-walk via 'keep-left' half-edge traversal.
    Returns bounded faces as vertex index cycles.
    """
    nbr = {i: [] for i in range(len(vtx))}
    for a, b in edges:
        nbr[a].append(b)
        nbr[b].append(a)

    # sort neighbors CCW by angle
    nbr_sorted = {}
    for i in range(len(vtx)):
        pi = vtx[i]
        lst = []
        for j in nbr[i]:
            pj = vtx[j]
            ang = math.atan2(pj[1] - pi[1], pj[0] - pi[0])
            lst.append((ang, j))
        lst.sort()
        nbr_sorted[i] = [j for _, j in lst]

    def next_halfedge(u, v):
        neigh = nbr_sorted[v]
        k = neigh.index(u)
        w = neigh[(k - 1) % len(neigh)]  # predecessor in CCW order => keep-left
        return v, w

    def signed_area(face):
        pts = vtx[face]
        area2 = 0.0
        for i in range(len(face)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(face)]
            area2 += x1 * y2 - x2 * y1
        return 0.5 * area2

    # walk all half-edges
    visited = set()
    faces = []
    for u in range(len(vtx)):
        for v in nbr_sorted[u]:
            he = (u, v)
            if he in visited:
                continue
            poly = []
            curr = he
            while True:
                if curr in visited:
                    break
                visited.add(curr)
                a, b = curr
                poly.append(a)
                curr = next_halfedge(a, b)
                if curr == he:
                    break
            if len(poly) >= 3:
                A = signed_area(poly)
                if abs(A) > 1e-14:
                    faces.append(poly)

    # deduplicate cycles by canonical form
    def canon_cycle(cyc):
        n = len(cyc)
        a = min(range(n), key=lambda i: cyc[i])
        r1 = cyc[a:] + cyc[:a]
        r2 = list(reversed(cyc))
        b = min(range(n), key=lambda i: r2[i])
        r2 = r2[b:] + r2[:b]
        return tuple(r1) if tuple(r1) < tuple(r2) else tuple(r2)

    uniq = {}
    for f in faces:
        uniq[canon_cycle(f)] = f
    faces = list(uniq.values())

    # remove outer (unbounded) face: largest |area|
    areas = [signed_area(f) for f in faces]
    outer = int(np.argmax([abs(a) for a in areas]))
    faces.pop(outer)

    return faces


def select_rectangular_faces(vtx: np.ndarray, faces, cells_x: int, cells_y: int):
    """
    Select all faces whose centroids fall inside a rectangle large enough to
    tile CELLS_X × CELLS_Y unit cells of the (4,6,12) tiling.

    The unit-cell dimensions for the (4,6,12) Archimedean tiling are:
        Ux = 2 * Lk   (= |a1|)
        Uy = sqrt(3) * Lk   (= |a2_y|)
    We select a rectangle of size cells_x * Ux  ×  cells_y * Uy centred on the
    mean centroid, with a small margin (half an edge length) so edge faces are
    not accidentally excluded.
    """
    Ux = 2.0 * Lk          # unit-cell width  (= |a1|)
    Uy = sqrt3 * Lk         # unit-cell height (= a2_y)
    margin = 0.0            # no pad; strict upper bound excludes periodic duplicates

    rect_w = cells_x * Ux + 2.0 * margin
    rect_h = cells_y * Uy + 2.0 * margin

    cents = np.array([vtx[f].mean(axis=0) for f in faces])
    c0 = cents.mean(axis=0)          # center of the whole lattice

    xlo = c0[0] - rect_w / 2.0
    xhi = c0[0] + rect_w / 2.0
    ylo = c0[1] - rect_h / 2.0
    yhi = c0[1] + rect_h / 2.0

    chosen = [
        fi for fi, c in enumerate(cents)
        if xlo <= c[0] < xhi and ylo <= c[1] < yhi
    ]
    expected = 6 * cells_x * cells_y
    print(f"  rectangular selection: {len(chosen)} faces (expected {expected}) "
          f"in [{xlo:.4f},{xhi:.4f}] x [{ylo:.4f},{yhi:.4f}]")

    if len(chosen) == 0:
        raise RuntimeError("No faces selected — increase index_radius.")
    if len(chosen) != expected:
        print(f"  WARNING: selected {len(chosen)} faces but expected {expected}")

    kept = [faces[i] for i in chosen]

    # reindex vertices
    used = sorted(set(i for f in kept for i in f))
    old2new = {old: i for i, old in enumerate(used)}
    V = vtx[used].copy()
    F = [[old2new[i] for i in f] for f in kept]

    # shift so min corner at (0,0)
    mins = V.min(axis=0)
    V -= mins

    return V, F


def write_ply_faces(path, V: np.ndarray, F):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(V)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(F)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for (x, y) in V:
            f.write(f"{x:.12e} {y:.12e} 0.000000000000e+00\n")
        for face in F:
            f.write(str(len(face)) + " " + " ".join(map(str, face)) + "\n")


def main():
    # Choose index_radius big enough so the interior has plenty of degree-4 kagome vertices
    # For a 3x3 patch (54 faces), radius ~ 10 is plenty; use 14 to be safe.
    index_radius = 14

    Ux = 2.0 * Lk       # = 6s
    Uy = sqrt3 * Lk      # = 3*sqrt3*s
    domain_x = CELLS_X * Ux
    domain_y = CELLS_Y * Uy
    n_faces = 6 * CELLS_X * CELLS_Y
    print(f"Edge length s = {s:.8f}")
    print(f"Unit cell: {Ux:.6f} x {Uy:.6f}")
    print(f"Domain: {domain_x:.6f} x {domain_y:.6f}  ({n_faces} faces)")
    print(f"Avg face area: {domain_x*domain_y/n_faces:.6f} (target {AVG_AREA:.6f})")

    print("Building kagome points...")
    kpts = build_kagome_points(index_radius)
    print("  points:", len(kpts))

    print("Building kagome NN edges...")
    k_edges = build_nn_edges(kpts)
    print("  edges:", len(k_edges))

    print("Truncating -> (4,6,12) graph...")
    vtx, edges = truncate_kagome(kpts, k_edges)
    print("  vertices:", len(vtx))
    print("  edges:", len(edges))

    print("Extracting bounded faces...")
    faces = extract_faces(vtx, edges)
    print("  faces extracted:", len(faces))

    print(f"Selecting rectangular {CELLS_X}x{CELLS_Y} patch...")
    V, F = select_rectangular_faces(vtx, faces, CELLS_X, CELLS_Y)
    print("  selected faces:", len(F))
    print("  selected vertices:", len(V))

    print("Writing:", OUT)
    write_ply_faces(OUT, V, F)
    print("Done.")


if __name__ == "__main__":
    main()
