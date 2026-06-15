#!/usr/bin/env python3
"""
Cut and Paste Rectangular Domain Generator for Hexagonal Tessellations
=======================================================================

This script transforms a periodic hexagonal tessellation into a rectangular
periodic domain where NO grain boundaries coincide with the domain boundaries.

Background
----------
A periodic hexagonal tessellation from Neper has domain boundaries that may
align with hexagon edges (grain boundaries). For simulations, it's often
desirable to have domain boundaries that cut THROUGH hexagons, so that:
1. The domain has clean rectangular edges
2. No grain boundary lies exactly on the domain boundary
3. Cut hexagons wrap around periodically to the opposite edge

Method
------
The script cuts the tessellation along vertical (x) and horizontal (y) lines,
then pastes the cut strips to the opposite sides:

    Original:                    After cut-paste:
    ┌─────────────┐              ┌─────────────┐
    │A│    B      │              │    B    │A' │
    │ │           │   ──────►    │         │   │
    │ │           │              │         │   │
    └─────────────┘              └─────────────┘

    Strip A (x < cut_x) moves to the right edge as A'
    The cut line becomes the new periodic boundary

Default Parameters
------------------
For hexagonal tessellations with area = 1/50 per hexagon:
- Hexagon side length: s = 0.087738
- Hexagon pitch: a = s√3 = 0.151967

Default cut positions (chosen to avoid grain boundaries):
- cut_x = 3√3/4 * s ≈ 0.114 (cuts through hexagon interior, not edges)
- cut_y = 0.15 (cuts through hexagon interior)

Usage
-----
Basic usage with defaults:
    python cut_paste_rect.py hex_regular.ply hex_cut.ply \\
        --domain-x 0.911803 --domain-y 1.052859

Custom cut positions:
    python cut_paste_rect.py hex_regular.ply hex_cut.ply \\
        --domain-x 0.911803 --domain-y 1.052859 \\
        --cut-x 0.114 --cut-y 0.15

Workflow
--------
1. Generate periodic hexagonal tessellation with gen_hex_seeds.py
2. Run this script to create rectangular domain with no grain boundaries at edges
3. Use the output PLY for simulations

Output
------
- Rectangular domain with same dimensions as input
- Hexagons crossing cut lines are split into two faces
- Split faces connect periodically across domain boundaries
- Total area preserved (same number of hexagons worth of area)

Author: Generated with Claude Code
"""
import argparse
from typing import List, Tuple, Dict

import numpy as np

EPS = 1e-9

# Default values for hexagons with area = 1/50
DEFAULT_HEX_SIDE = 0.087738  # s = sqrt(2 * (1/50) / (3 * sqrt(3)))
DEFAULT_CUT_X = 0.113975     # 3*sqrt(3)/4 * s - cuts through hexagon interior
DEFAULT_CUT_Y = 0.15         # Cuts through hexagon interior


def read_neper_tess_ply(filename: str) -> Tuple[List[str], np.ndarray, List[List[int]]]:
    """Read a Neper-format PLY file.

    Returns:
        header: List of header lines
        verts: Nx3 array of vertex coordinates
        faces: List of faces, each face is a list of vertex indices
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    header: List[str] = []
    i = 0
    n_vertices = n_faces = 0

    while True:
        line = lines[i]
        header.append(line)
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[2])
        elif line.startswith("element face"):
            n_faces = int(line.split()[2])
        elif line.startswith("end_header"):
            i += 1
            break
        i += 1

    verts = np.array([
        list(map(float, lines[i + j].split()[:3])) for j in range(n_vertices)
    ])

    face_start = i + n_vertices
    faces: List[List[int]] = []
    for j in range(n_faces):
        parts = lines[face_start + j].split()
        k = int(parts[0])
        faces.append(list(map(int, parts[1 : 1 + k])))

    return header, verts, faces


def write_neper_tess_ply(filename: str, header: List[str], verts: np.ndarray, faces: List[List[int]]) -> None:
    """Write a Neper-format PLY file."""
    header_out = list(header)
    n_vertices = verts.shape[0]
    n_faces = len(faces)

    for idx, line in enumerate(header_out):
        if line.startswith("element vertex"):
            header_out[idx] = f"element vertex {n_vertices}"
        elif line.startswith("element face"):
            header_out[idx] = f"element face {n_faces}"
        elif line.startswith("element cell"):
            header_out[idx] = f"element cell {n_faces}"
        elif line.startswith("end_header"):
            break

    with open(filename, "w", encoding="utf-8") as f:
        for line in header_out:
            f.write(line + "\n")

        for x, y, z in verts:
            f.write(f"{x:.12f} {y:.12f} {z:.12f}\n")

        for face in faces:
            f.write(str(len(face)) + " " + " ".join(map(str, face)) + "\n")

        for idx in range(n_faces):
            f.write(f"1 {idx}\n")


def get_or_create_vertex(verts_list: List[List[float]], vert_map: Dict[Tuple, int],
                         x: float, y: float, z: float, eps: float = 1e-6) -> int:
    """Get existing vertex index or create new one.

    Uses rounded coordinates as key to merge nearby vertices.
    Note: eps=1e-6 is needed because coordinate shifts during cut-paste
    can introduce floating-point discrepancies of ~1.8e-7, which the
    previous eps=1e-7 failed to merge.
    """
    key = (round(x / eps), round(y / eps), round(z / eps))
    if key in vert_map:
        return vert_map[key]
    idx = len(verts_list)
    verts_list.append([x, y, z])
    vert_map[key] = idx
    return idx


def cut_and_paste_axis(
    verts: np.ndarray,
    faces: List[List[int]],
    axis: int,
    cut_value: float,
    domain_size: float,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Cut tessellation at axis=cut_value and paste strip to opposite side.

    Transformation:
    - Strip [0, cut_value] moves to [domain_size - cut_value, domain_size]
    - Part [cut_value, domain_size] shifts to [0, domain_size - cut_value]
    - Polygons crossing the cut line are split into two parts

    Args:
        verts: Vertex coordinates
        faces: Face definitions (list of vertex indices per face)
        axis: 0 for x-axis, 1 for y-axis
        cut_value: Position of the cut line
        domain_size: Size of domain along this axis

    Returns:
        new_verts: Updated vertex coordinates
        new_faces: Updated face definitions
    """
    new_verts_list: List[List[float]] = []
    vert_map: Dict[Tuple, int] = {}
    new_faces: List[List[int]] = []

    other_axis = 1 - axis  # For 2D: if axis=0, other=1 and vice versa

    for face in faces:
        poly_verts = [verts[v] for v in face]
        n = len(poly_verts)

        # Check bounds
        coords = [v[axis] for v in poly_verts]
        min_c, max_c = min(coords), max(coords)

        if max_c <= cut_value + EPS:
            # Entire polygon in left strip -> move to right
            # New coord = old_coord + (domain_size - cut_value)
            face_indices = []
            for v in poly_verts:
                new_coord = v[axis] + domain_size - cut_value
                new_v = [0, 0, v[2]]
                new_v[axis] = new_coord
                new_v[other_axis] = v[other_axis]
                idx = get_or_create_vertex(new_verts_list, vert_map, *new_v)
                face_indices.append(idx)
            if len(face_indices) >= 3:
                new_faces.append(face_indices)

        elif min_c >= cut_value - EPS:
            # Entire polygon in right part -> shift left
            # New coord = old_coord - cut_value
            face_indices = []
            for v in poly_verts:
                new_coord = v[axis] - cut_value
                new_v = [0, 0, v[2]]
                new_v[axis] = new_coord
                new_v[other_axis] = v[other_axis]
                idx = get_or_create_vertex(new_verts_list, vert_map, *new_v)
                face_indices.append(idx)
            if len(face_indices) >= 3:
                new_faces.append(face_indices)

        else:
            # Polygon crosses the cut line -> split into two parts
            left_part = []  # coords < cut_value (will move to right edge)
            right_part = []  # coords >= cut_value (will shift to left)

            for i in range(n):
                curr = poly_verts[i]
                next_v = poly_verts[(i + 1) % n]

                curr_left = curr[axis] < cut_value
                next_left = next_v[axis] < cut_value

                if curr_left:
                    left_part.append(curr)
                else:
                    right_part.append(curr)

                # If edge crosses cut line, add intersection to both parts
                if curr_left != next_left:
                    t = (cut_value - curr[axis]) / (next_v[axis] - curr[axis])
                    inter = np.array([
                        curr[0] + t * (next_v[0] - curr[0]),
                        curr[1] + t * (next_v[1] - curr[1]),
                        curr[2] + t * (next_v[2] - curr[2]),
                    ])
                    inter[axis] = cut_value  # Exact position at cut line
                    left_part.append(inter)
                    right_part.append(inter)

            # Create face for left part (moves to right edge)
            if len(left_part) >= 3:
                face_indices = []
                for v in left_part:
                    new_coord = v[axis] + domain_size - cut_value
                    new_v = [0, 0, v[2]]
                    new_v[axis] = new_coord
                    new_v[other_axis] = v[other_axis]
                    idx = get_or_create_vertex(new_verts_list, vert_map, *new_v)
                    face_indices.append(idx)
                # Remove consecutive duplicates
                cleaned = []
                for idx in face_indices:
                    if not cleaned or cleaned[-1] != idx:
                        cleaned.append(idx)
                if len(cleaned) >= 3:
                    new_faces.append(cleaned)

            # Create face for right part (shifts to left)
            if len(right_part) >= 3:
                face_indices = []
                for v in right_part:
                    new_coord = v[axis] - cut_value
                    new_v = [0, 0, v[2]]
                    new_v[axis] = new_coord
                    new_v[other_axis] = v[other_axis]
                    idx = get_or_create_vertex(new_verts_list, vert_map, *new_v)
                    face_indices.append(idx)
                # Remove consecutive duplicates
                cleaned = []
                for idx in face_indices:
                    if not cleaned or cleaned[-1] != idx:
                        cleaned.append(idx)
                if len(cleaned) >= 3:
                    new_faces.append(cleaned)

    new_verts = np.array(new_verts_list) if new_verts_list else np.zeros((0, 3))
    return new_verts, new_faces


def cut_paste_rectangular(
    input_ply: str,
    output_ply: str,
    domain_x: float,
    domain_y: float,
    cut_x: float,
    cut_y: float,
) -> None:
    """Cut and paste to create rectangular periodic domain.

    Args:
        input_ply: Input periodic PLY file from Neper
        output_ply: Output PLY file with rectangular domain
        domain_x: Domain width (Lx)
        domain_y: Domain height (Ly)
        cut_x: Vertical cut position (strip [0, cut_x] moves to right)
        cut_y: Horizontal cut position (strip [0, cut_y] moves to top)
    """
    print(f"Original domain: {domain_x:.6f} × {domain_y:.6f}")
    print(f"Cut lines: x = {cut_x:.6f}, y = {cut_y:.6f}")

    header, verts, faces = read_neper_tess_ply(input_ply)
    print(f"\nInput: {verts.shape[0]} vertices, {len(faces)} faces")

    # Cut and paste along x-axis
    if cut_x > EPS:
        verts, faces = cut_and_paste_axis(verts, faces, axis=0,
                                          cut_value=cut_x, domain_size=domain_x)
        print(f"After x-cut: {verts.shape[0]} vertices, {len(faces)} faces")

    # Cut and paste along y-axis
    if cut_y > EPS:
        verts, faces = cut_and_paste_axis(verts, faces, axis=1,
                                          cut_value=cut_y, domain_size=domain_y)
        print(f"After y-cut: {verts.shape[0]} vertices, {len(faces)} faces")

    # ------------------------------------------------------------------
    # Deduplicate faces that share the same centroid (within tolerance).
    # The cut-paste can create duplicate copies of small faces (squares)
    # that sit exactly on a cut line, or at intersections of x/y cuts.
    # ------------------------------------------------------------------
    if len(faces) > 0:
        cents = []
        for f in faces:
            c = verts[f].mean(axis=0)
            cents.append(c)
        cents = np.array(cents)

        keep = []
        seen = {}
        dedup_eps = 1e-6
        for i, c in enumerate(cents):
            key = (round(c[0] / dedup_eps), round(c[1] / dedup_eps))
            if key not in seen:
                seen[key] = i
                keep.append(i)
        n_removed = len(faces) - len(keep)
        if n_removed > 0:
            faces = [faces[i] for i in keep]
            print(f"Dedup: removed {n_removed} duplicate faces, {len(faces)} remaining")

    # Center the geometry at the origin
    if len(verts) > 0:
        # Shift so center is at (0, 0)
        verts[:, 0] -= domain_x / 2
        verts[:, 1] -= domain_y / 2

        # Report bounds
        x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
        y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
        print(f"\nFinal bounds: x=[{x_min:.6f}, {x_max:.6f}], y=[{y_min:.6f}, {y_max:.6f}]")
        print(f"Final size: {x_max - x_min:.6f} × {y_max - y_min:.6f}")
        print(f"Center: (0, 0)")

    write_neper_tess_ply(output_ply, header, verts, faces)
    print(f"\nOutput: {output_ply}")


def main():
    parser = argparse.ArgumentParser(
        description="Cut tessellation along straight lines and paste strips to opposite side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using defaults for hexagons with area = 1/50:
  python cut_paste_rect.py hex_regular.ply hex_cut.ply --domain-x 0.911803 --domain-y 1.052859

  # Custom cut positions:
  python cut_paste_rect.py hex_regular.ply hex_cut.ply --domain-x 0.911803 --domain-y 1.052859 --cut-x 0.12 --cut-y 0.16

Notes:
  - Default cut_x = 0.113975 (= 3√3/4 × s, where s = hexagon side length)
  - Default cut_y = 0.15
  - These defaults ensure no grain boundary coincides with domain boundary
        """
    )
    parser.add_argument("input", help="Input periodic PLY file")
    parser.add_argument("output", help="Output PLY file")
    parser.add_argument("--domain-x", type=float, required=True, help="Domain width (Lx)")
    parser.add_argument("--domain-y", type=float, required=True, help="Domain height (Ly)")
    parser.add_argument("--cut-x", type=float, default=DEFAULT_CUT_X,
                        help=f"Vertical cut position (default: {DEFAULT_CUT_X:.6f})")
    parser.add_argument("--cut-y", type=float, default=DEFAULT_CUT_Y,
                        help=f"Horizontal cut position (default: {DEFAULT_CUT_Y:.6f})")

    args = parser.parse_args()
    cut_paste_rectangular(args.input, args.output, args.domain_x, args.domain_y,
                          args.cut_x, args.cut_y)


if __name__ == "__main__":
    main()
