"""Adapted read.py for the cut tiling PLYs (tess workflow).

The generator + cut_paste_rect outputs declare no `element cell` in the
header (each face IS a grain), so the original read.py — which builds
regions from cell records — would emit empty regions. Here each face is
treated as its own cell. Output format matches tessellation_output.json:
{key: [pts, regions]} with pts = list of (x, y) and regions = tuples of
1-based vertex indices in CCW order.
"""
import json

import numpy as np


def read_ply_faces_as_cells(filename):
    with open(filename, "r") as f:
        lines = f.read().splitlines()

    n_vertices = n_faces = 0
    i = 0
    while not lines[i].startswith("end_header"):
        if lines[i].startswith("element vertex"):
            n_vertices = int(lines[i].split()[2])
        elif lines[i].startswith("element face"):
            n_faces = int(lines[i].split()[2])
        i += 1
    header_end = i + 1

    pts = []
    for j in range(n_vertices):
        x, y = map(float, lines[header_end + j].split()[:2])
        pts.append((x, y))

    faces_start = header_end + n_vertices
    regions = []
    for j in range(n_faces):
        parts = list(map(int, map(float, lines[faces_start + j].split())))
        k = parts[0]
        verts = parts[1 : 1 + k]

        # enforce CCW (faces here are convex, so angle sort is safe)
        cell_pts = np.array([pts[v] for v in verts])
        cx, cy = cell_pts.mean(axis=0)
        angles = np.arctan2(cell_pts[:, 1] - cy, cell_pts[:, 0] - cx)
        order = np.argsort(angles)
        regions.append(tuple(verts[k] + 1 for k in order))  # 1-based

    return pts, regions


FILES = {
    "tth_4612": "tth_4612_cut.ply",
    "trunc_hex_31212": "12123/trunc_hex_cut.ply",
    "trunc_sq_488": "84/trunc_sq_cut.ply",
}

if __name__ == "__main__":
    data = {}
    for key, fname in FILES.items():
        pts, regions = read_ply_faces_as_cells(fname)
        data[key] = [pts, regions]
        print(f"Processed {fname}: points={len(pts)}, regions={len(regions)}")

    out = "tessellation_output.json"
    with open(out, "w") as fh:
        json.dump(data, fh)
    print(f"{len(data)} tessellations written to {out}")

    with open(out) as fh:
        data = json.load(fh)
    for key in data:
        pts, regions = data[key]
        print(f"\n{key}: {len(pts)} pts, {len(regions)} regions")
        print("  first 3 regions:", [tuple(r) for r in regions[:3]])
