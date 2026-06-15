import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
from meshes import MakeMesh

a = np.sqrt(3)
pts0 = [
    (0, 0), (3/4, 0), (1/2, a/4), (0, a/4),
    (9/4, 0), (5/2, a/4), (2, 3*a/4), (1, 3*a/4),
    (3, 0), (3, a/4), (3, a), (9/4, a),
    (3/4, a), (0, a)
]
pts1 = [(x - 1.5, y - 0.5 * a) for (x, y) in pts0]
area_per_grain = 3 * np.sqrt(3) / 2
scale_factor = np.sqrt(1 / 50 / area_per_grain)
pts = [(x * scale_factor, y * scale_factor) for (x, y) in pts1]

regions = [
    (1, 2, 3, 4),
    (2, 5, 6, 7, 8, 3),
    (5, 9, 10, 6),
    (6, 10, 11, 12, 7),
    (8, 7, 12, 13),
    (4, 3, 8, 13, 14)
]

_, _, mesh, _, _, _, _, _, _ = MakeMesh(
    pts, regions,
    maxh=0.1 * scale_factor,
    comm=MPI.COMM_WORLD,
    core_frac=0.01 * scale_factor,
)

# Extract vertices and triangles from the NGSolve mesh
verts = np.array([[v.point[0], v.point[1]] for v in mesh.vertices])
tris  = np.array([[v.nr for v in el.vertices] for el in mesh.Elements()])

# Color by grain (material region)
grain_colors = ['#AED6F1', '#A9DFBF', '#F9E79F', '#F1948A', '#D2B4DE', '#FAD7A0']
mat_names = [el.mat for el in mesh.Elements()]
name_to_color = {f"region_{i+1}": grain_colors[i] for i in range(6)}

fig, ax = plt.subplots(figsize=(8, 6))

# Draw each grain's triangles in its color, then overlay edges
from matplotlib.colors import to_rgba
from matplotlib.collections import PolyCollection

# Build face color array, one entry per triangle
face_colors = np.zeros((len(tris), 4))
for grain_idx in range(1, 7):
    name = f"region_{grain_idx}"
    mask = np.array([m == name for m in mat_names])
    face_colors[mask] = to_rgba(grain_colors[grain_idx - 1])

xy = verts[tris]  # (ntri, 3, 2)
coll = PolyCollection(xy, facecolors=face_colors, edgecolors='k', linewidths=0.4)
ax.add_collection(coll)
ax.autoscale_view()

ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Hexagonal RVE mesh  (ne = {len(tris)}, nv = {len(verts)})')
plt.tight_layout()
plt.savefig('hex_mesh.png', dpi=200)
print(f"Saved hex_mesh.png  ({len(tris)} elements, {len(verts)} vertices)")
