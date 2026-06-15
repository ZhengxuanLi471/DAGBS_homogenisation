#!/usr/bin/env python3
"""Visualize a PLY tessellation as PNG with transparent background."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def read_ply(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    i = 0
    n_verts = n_faces = 0
    while True:
        line = lines[i]
        if line.startswith("element vertex"):
            n_verts = int(line.split()[2])
        elif line.startswith("element face"):
            n_faces = int(line.split()[2])
        elif line.startswith("end_header"):
            i += 1
            break
        i += 1
    verts = np.array([list(map(float, lines[i+j].split()[:3])) for j in range(n_verts)])
    faces = []
    for j in range(n_faces):
        parts = lines[i + n_verts + j].split()
        k = int(parts[0])
        faces.append(list(map(int, parts[1:1+k])))
    return verts, faces

def plot_tessellation(verts, faces, output, title=""):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Color faces by number of sides
    color_map = {4: "#ff9999", 6: "#99ccff", 12: "#99ff99"}
    patches = []
    colors = []
    for face in faces:
        poly = Polygon(verts[face, :2], closed=True)
        patches.append(poly)
        n = len(face)
        colors.append(color_map.get(n, "#dddddd"))

    pc = PatchCollection(patches, facecolors=colors, edgecolors="black", linewidths=0.8, alpha=0.6)
    ax.add_collection(pc)

    # Draw domain boundary
    xmin, xmax = verts[:, 0].min(), verts[:, 0].max()
    ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                          fill=False, edgecolor="red", linewidth=2, linestyle="--")
    ax.add_patch(rect)

    ax.set_xlim(xmin - 0.02, xmax + 0.02)
    ax.set_ylim(ymin - 0.02, ymax + 0.02)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ff9999", edgecolor="black", alpha=0.6, label="Square (4)"),
        Patch(facecolor="#99ccff", edgecolor="black", alpha=0.6, label="Hexagon (6)"),
        Patch(facecolor="#99ff99", edgecolor="black", alpha=0.6, label="Dodecagon (12)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    fig.savefig(output, dpi=150, transparent=True, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)

if __name__ == "__main__":
    ply_file = sys.argv[1] if len(sys.argv) > 1 else "tth_4612_3x3.ply"
    out_file = sys.argv[2] if len(sys.argv) > 2 else ply_file.replace(".ply", ".png")
    title = sys.argv[3] if len(sys.argv) > 3 else ply_file
    verts, faces = read_ply(ply_file)
    print(f"Loaded: {len(verts)} vertices, {len(faces)} faces")
    print(f"  x: [{verts[:,0].min():.5f}, {verts[:,0].max():.5f}]")
    print(f"  y: [{verts[:,1].min():.5f}, {verts[:,1].max():.5f}]")
    plot_tessellation(verts, faces, out_file, title=title)
