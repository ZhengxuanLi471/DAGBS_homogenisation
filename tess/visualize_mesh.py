#!/usr/bin/env python3
"""
Mesh Visualization Script
-------------------------
Visualizes the polycrystalline mesh structure with grain boundaries,
triple junctions, and boundary labels.
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from meshes import MakeMesh
from mpi4py import MPI
from ngsolve import *

def _point_in_polygon(point, polygon):
    """Simple ray casting algorithm for point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Load tessellation data: visualize_mesh.py [json] [key] [out_prefix]
json_path = sys.argv[1] if len(sys.argv) > 1 else "tessellation_output.json"
with open(json_path, "r") as f:
    data = json.load(f)

key = sys.argv[2] if len(sys.argv) > 2 else list(data.keys())[0]
out_prefix = sys.argv[3] if len(sys.argv) > 3 else key
pts = data[key][0]
regions = data[key][1]
num_grains = len(regions)

print(f"Number of grains: {num_grains}")
print(f"Number of vertices: {len(pts)}")

# Create the mesh
print("\nGenerating mesh...")
try:
    (
        shape, geo, mesh, faces,
        contact_pairs,
        outer_contact_pairs,
        corner_label,
        outer_core_labels,
        junction_incidence,
    ) = MakeMesh(
        pts,
        regions,
        maxh=0.1,
        comm=MPI.COMM_WORLD,
        core_frac=0.01,
    )
    print(f"Mesh generated successfully!")
    print(f"Number of elements: {mesh.ne}")
    print(f"Number of vertices: {mesh.nv}")
    print(f"Number of edges: {mesh.nedge}")
except Exception as e:
    print(f"Error generating mesh: {e}")
    mesh = None

# ============================================================================
# Visualization 1: Tessellation with grain boundaries
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# --- Left plot: Grain regions ---
ax1 = axes[0]
patches = []
colors = []

for region_idx, region in enumerate(regions):
    # Get polygon vertices (convert 1-based to 0-based indexing)
    poly_pts = [pts[i-1] for i in region]
    polygon = Polygon(poly_pts, closed=True)
    patches.append(polygon)
    colors.append(region_idx)

# Create collection with unique colors for each grain
p = PatchCollection(patches, alpha=0.6, edgecolors='black', linewidths=1.5, cmap='tab20')
p.set_array(np.array(colors))
ax1.add_collection(p)

# Plot vertices
pts_array = np.array(pts)
ax1.scatter(pts_array[:, 0], pts_array[:, 1], c='red', s=20, zorder=5, alpha=0.7, label='Vertices')

# Label grain regions
for region_idx, region in enumerate(regions):
    poly_pts = np.array([pts[i-1] for i in region])
    centroid = poly_pts.mean(axis=0)
    ax1.text(centroid[0], centroid[1], f'{region_idx+1}',
             ha='center', va='center', fontsize=8, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax1.set_xlim(pts_array[:, 0].min() - 0.05, pts_array[:, 0].max() + 0.05)
ax1.set_ylim(pts_array[:, 1].min() - 0.05, pts_array[:, 1].max() + 0.05)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_title(f'Grain Structure ({num_grains} grains)', fontsize=14, fontweight='bold')
ax1.legend()

# --- Right plot: Grain boundaries and triple junctions ---
ax2 = axes[1]

# Draw grain regions (lighter colors)
p2 = PatchCollection(patches, alpha=0.3, edgecolors='gray', linewidths=0.5, cmap='tab20')
p2.set_array(np.array(colors))
ax2.add_collection(p2)

# Find and highlight grain boundaries (shared edges)
edge_count = {}
for region in regions:
    n = len(region)
    for k in range(n):
        edge = tuple(sorted([region[k], region[(k+1) % n]]))
        edge_count[edge] = edge_count.get(edge, 0) + 1

# Draw grain boundaries (edges shared by 2 grains) in blue
gb_lines = []
for edge, count in edge_count.items():
    if count == 2:  # Internal grain boundary
        p0, p1 = pts[edge[0]-1], pts[edge[1]-1]
        gb_lines.append([p0, p1])

gb_collection = LineCollection(gb_lines, colors='blue', linewidths=2, label='Grain Boundaries')
ax2.add_collection(gb_collection)

# Draw external boundaries in red
external_lines = []
for edge, count in edge_count.items():
    if count == 1:  # External boundary
        p0, p1 = pts[edge[0]-1], pts[edge[1]-1]
        external_lines.append([p0, p1])

ext_collection = LineCollection(external_lines, colors='red', linewidths=2.5, label='External Boundaries')
ax2.add_collection(ext_collection)

# Find and mark triple junctions (vertices shared by 3+ grains)
vertex_grain_count = {i: set() for i in range(len(pts))}
for region_idx, region in enumerate(regions):
    for vertex_idx in region:
        vertex_grain_count[vertex_idx-1].add(region_idx)

triple_junctions = []
for vertex_idx, grain_set in vertex_grain_count.items():
    if len(grain_set) >= 3:
        triple_junctions.append(pts[vertex_idx])

if triple_junctions:
    triple_junctions = np.array(triple_junctions)
    ax2.scatter(triple_junctions[:, 0], triple_junctions[:, 1],
               c='green', s=100, marker='*', zorder=10,
               edgecolors='black', linewidths=1,
               label=f'Triple Junctions ({len(triple_junctions)})')

ax2.set_xlim(pts_array[:, 0].min() - 0.05, pts_array[:, 0].max() + 0.05)
ax2.set_ylim(pts_array[:, 1].min() - 0.05, pts_array[:, 1].max() + 0.05)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('Y', fontsize=12)
ax2.set_title('Grain Boundaries & Triple Junctions', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'{out_prefix}_mesh_tessellation.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {out_prefix}_mesh_tessellation.png")

# ============================================================================
# Visualization 2: Finite element mesh (if successfully generated)
# ============================================================================
if mesh is not None:
    fig2, ax3 = plt.subplots(1, 1, figsize=(12, 10))

    # Extract mesh vertices and elements
    mesh_pts = []
    for i in range(mesh.nv):
        pt = mesh[NodeId(VERTEX, i)].point
        mesh_pts.append([pt[0], pt[1]])
    mesh_pts = np.array(mesh_pts)

    # Extract triangular elements
    triangles = []
    for el in mesh.Elements():
        vertices = el.vertices
        if len(vertices) == 3:  # Triangle
            triangles.append([v.nr for v in vertices])

    # Draw mesh elements
    for tri in triangles:
        pts_tri = mesh_pts[tri]
        triangle = Polygon(pts_tri, closed=True, fill=False,
                          edgecolor='gray', linewidth=0.3, alpha=0.5)
        ax3.add_patch(triangle)

    # Color by grain region
    element_colors = []
    element_patches = []
    for el in mesh.Elements():
        vertices = el.vertices
        if len(vertices) >= 3:
            pts_tri = mesh_pts[[v.nr for v in vertices]]
            # Determine which grain this element belongs to
            centroid = pts_tri.mean(axis=0)
            region_id = 0  # Default
            for idx, region in enumerate(regions):
                poly_pts = np.array([pts[i-1] for i in region])
                # Simple point-in-polygon test
                if _point_in_polygon(centroid, poly_pts):
                    region_id = idx
                    break
            element_patches.append(Polygon(pts_tri, closed=True))
            element_colors.append(region_id)

    pc = PatchCollection(element_patches, alpha=0.4, edgecolors='none', cmap='tab20')
    pc.set_array(np.array(element_colors))
    ax3.add_collection(pc)

    # Overlay grain boundaries
    gb_collection2 = LineCollection(gb_lines, colors='blue', linewidths=1.5,
                                    label='Grain Boundaries', zorder=5)
    ax3.add_collection(gb_collection2)

    ax3.set_xlim(mesh_pts[:, 0].min() - 0.05, mesh_pts[:, 0].max() + 0.05)
    ax3.set_ylim(mesh_pts[:, 1].min() - 0.05, mesh_pts[:, 1].max() + 0.05)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('X', fontsize=12)
    ax3.set_ylabel('Y', fontsize=12)
    ax3.set_title(f'Finite Element Mesh ({mesh.ne} elements, {mesh.nv} vertices)',
                 fontsize=14, fontweight='bold')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f'{out_prefix}_mesh_finite_elements.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {out_prefix}_mesh_finite_elements.png")

    # ========================================================================
    # Visualization 3: Boundary labels
    # ========================================================================
    fig3, ax4 = plt.subplots(1, 1, figsize=(12, 10))

    # Draw grain regions
    p3 = PatchCollection([Polygon([pts[i-1] for i in region], closed=True)
                         for region in regions],
                        alpha=0.2, edgecolors='gray', linewidths=0.5)
    ax4.add_collection(p3)

    # Draw and label boundaries by type
    boundary_types = {}
    for i, bnd_name in enumerate(mesh.GetBoundaries()):
        boundary_types[bnd_name] = []

    # Sample boundary edge midpoints
    for el in mesh.Elements(BND):
        edge_pts = [mesh[v].point for v in el.vertices]
        if len(edge_pts) == 2:
            midpoint = np.array([(edge_pts[0][0] + edge_pts[1][0])/2,
                                (edge_pts[0][1] + edge_pts[1][1])/2])
            boundary_types.setdefault(el.mat, []).append(midpoint)

    # Color code different boundary types
    colors_map = {
        'core': 'green',
        'slide': 'blue',
        'left': 'red',
        'right': 'red',
        'top': 'orange',
        'bottom': 'orange',
        'LB': 'purple',
    }

    for bnd_name, points in boundary_types.items():
        if not points:
            continue
        points = np.array(points)
        # Determine color based on boundary name
        color = 'gray'
        for key, col in colors_map.items():
            if key in bnd_name.lower():
                color = col
                break
        ax4.scatter(points[:, 0], points[:, 1], c=color, s=5, alpha=0.6, label=bnd_name[:20])

    ax4.set_xlim(pts_array[:, 0].min() - 0.05, pts_array[:, 0].max() + 0.05)
    ax4.set_ylim(pts_array[:, 1].min() - 0.05, pts_array[:, 1].max() + 0.05)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('X', fontsize=12)
    ax4.set_ylabel('Y', fontsize=12)
    ax4.set_title('Boundary Labels', fontsize=14, fontweight='bold')
    # Limit legend entries
    handles, labels = ax4.get_legend_handles_labels()
    if len(handles) > 20:
        ax4.legend(handles[:20], labels[:20], loc='upper right', fontsize=8, ncol=2)
    else:
        ax4.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(f'{out_prefix}_mesh_boundaries.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {out_prefix}_mesh_boundaries.png")

print("\nVisualization complete!")
