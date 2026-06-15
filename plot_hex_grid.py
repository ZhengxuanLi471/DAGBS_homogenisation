import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

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

colors = ['#AED6F1', '#A9DFBF', '#F9E79F', '#F1948A', '#D2B4DE', '#FAD7A0']

fig, ax = plt.subplots(figsize=(7, 6))

patches = []
for i, reg in enumerate(regions):
    verts = [pts[v - 1] for v in reg]
    poly = Polygon(verts, closed=True)
    patches.append(poly)

coll = PatchCollection(patches, facecolors=colors, edgecolors='k', linewidths=1.5, zorder=2)
ax.add_collection(coll)

# Label each grain at its centroid
for i, reg in enumerate(regions):
    verts = np.array([pts[v - 1] for v in reg])
    cx, cy = verts.mean(axis=0)
    ax.text(cx, cy, f"G{i+1}", ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)

# Label vertices with 1-based index
for idx, (x, y) in enumerate(pts, start=1):
    ax.plot(x, y, 'ko', markersize=4, zorder=4)
    ax.text(x, y, f' {idx}', fontsize=8, color='#333333', zorder=5,
            ha='left', va='bottom')

# Draw the periodic bounding box
xmax = 1.5 * scale_factor
ymax = 0.5 * a * scale_factor
box_x = [-xmax, xmax, xmax, -xmax, -xmax]
box_y = [-ymax, -ymax, ymax, ymax, -ymax]
ax.plot(box_x, box_y, 'k--', linewidth=1.0, zorder=1, label='Periodic RVE boundary')

ax.set_aspect('equal')
ax.margins(0.12)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Hexagonal RVE — 6-grain periodic grid\n(vertex labels = 1-based index used in meshes.py)')
ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('hex_grid.png', dpi=150)
print("Saved hex_grid.png")
