"""Line-profile frequency movie -- storage & dissipation density vs distance along
three probe segments, animated as frequency sweeps. Pure Python (meshio + numpy +
matplotlib, headless); reuses the existing per-frequency VTU frames in vtu_out/.

Probe lines (ALL are interior grain-boundary edges -- shared by 2 grains, neither
endpoint on the periodic box -- pointing in roughly the same direction so the
directional shear response is comparable):
  col 0  hex full-length GB edge                       (sampled on the hex VTUs)
  col 1  LONG interior GB edge bordering LARGE grains   (seed 24)
  col 2  SHORT interior GB edge bordering SMALL grains  (seed 24)
Rows: storage_energy_density (top), dissipation_energy_density (bottom). 2x3 grid.

Dissipation is a GB field (baked onto the grain-boundary shell); since every probe
runs ALONG a real GB, all three dissipation profiles are the meaningful
along-boundary ones. Storage is sampled just at the boundary line.

    python line_profiles_movie.py                 # both GIFs (fixed + autoscale)
    python line_profiles_movie.py --yscale fixed  # only line_profiles_fixed.gif
    python line_profiles_movie.py --stride 40      # quick preview
"""
import os
import json
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import meshio
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_JSON = os.path.normpath(os.path.join(HERE, "..", "sigmas", "sigma_0.45",
                                          "tessellation_output.json"))
NPTS = 200
INSET = 0.02


# ----------------------------------------------------------------------- geometry
def hex_full_edge():
    from collections import Counter
    a = np.sqrt(3)
    pts0 = [(0, 0), (3/4, 0), (1/2, a/4), (0, a/4), (9/4, 0), (5/2, a/4),
            (2, 3*a/4), (1, 3*a/4), (3, 0), (3, a/4), (3, a), (9/4, a),
            (3/4, a), (0, a)]
    pts1 = [(x - 1.5, y - 0.5 * a) for (x, y) in pts0]
    s = np.sqrt(1 / 50 / (3 * np.sqrt(3) / 2))
    pts = [(x * s, y * s) for (x, y) in pts1]
    regions = [(1, 2, 3, 4), (2, 5, 6, 7, 8, 3), (5, 9, 10, 6),
               (6, 10, 11, 12, 7), (8, 7, 12, 13), (4, 3, 8, 13, 14)]
    ec = Counter()
    for r in regions:
        n = len(r)
        for i in range(n):
            ec[tuple(sorted((r[i], r[(i + 1) % n])))] += 1
    shared = [k for k, c in ec.items() if c == 2]
    k = max(shared, key=lambda k: np.hypot(*(np.subtract(pts[k[1]-1], pts[k[0]-1]))))
    return np.array(pts[k[0]-1]), np.array(pts[k[1]-1])


def seed_gb_edges(target_ang, window=15.0, lfloor=0.02):
    """Pick two INTERIOR grain-boundary edges (shared by 2 grains, neither endpoint on
    the periodic box) whose orientation is within `window` deg of `target_ang` (so all
    three probes point roughly the same way -- shear is directional) and whose length
    >= `lfloor` (samplable). Returns:
      A = LONG edge bordering LARGE grains   (max  length * mean adjacent grain area)
      B = SHORT edge bordering SMALL grains  (min  mean adjacent grain area)
    plus per-edge metadata (length, angle, mean grain area)."""
    from collections import defaultdict
    pts, regs = json.load(open(SEED_JSON))["seeds_24"]
    pts = np.array(pts)
    bx = [pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()]

    def on_box(p, t=1e-6):
        return (abs(p[0]-bx[0]) < t or abs(p[0]-bx[1]) < t or
                abs(p[1]-bx[2]) < t or abs(p[1]-bx[3]) < t)

    def area(reg):
        P = pts[np.array(reg) - 1]
        x, y = P[:, 0], P[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    areas = [area(r) for r in regs]
    eg = defaultdict(list)
    for gi, r in enumerate(regs):
        n = len(r)
        for i in range(n):
            eg[tuple(sorted((r[i], r[(i + 1) % n])))].append(gi)

    cand = []
    for e, gs in eg.items():
        if len(gs) != 2:                              # interior = shared by 2 grains
            continue
        a, b = pts[e[0] - 1], pts[e[1] - 1]
        if on_box(a) or on_box(b):                    # neither endpoint on the box
            continue
        L = np.hypot(*(b - a))
        ang = np.degrees(np.arctan2((b - a)[1], (b - a)[0])) % 180
        if L < lfloor or abs(((ang - target_ang + 90) % 180) - 90) > window:
            continue
        cand.append((a, b, L, ang, float(np.mean([areas[g] for g in gs]))))
    A = max(cand, key=lambda x: x[2] * x[4])          # long + large grains
    B = min(cand, key=lambda x: x[4])                 # smallest adjacent grains
    return A, B


def sample(A, B, n=NPTS, inset=INSET):
    t = np.linspace(inset, 1 - inset, n)
    P = A[None, :] + t[:, None] * (B - A)[None, :]
    dist = t * np.hypot(*(B - A))
    return P[:, 0], P[:, 1], dist


# ------------------------------------------------------------------ interpolation
def read_pvd(pvd):
    root = ET.parse(pvd).getroot()
    items = sorted((float(ds.get("timestep")), ds.get("file"))
                   for ds in root.iter("DataSet"))
    base = os.path.dirname(pvd)
    return np.array([t for t, _ in items]), [os.path.join(base, f) for _, f in items]


def load_geom(vtu):
    """The VTU writes disconnected per-triangle nodes (duplicate coords at shared
    vertices). Merge duplicates and build a scipy Delaunay for robust point location
    (matplotlib's TriFinder rejects this FE mesh). Return (Delaunay, inv, nuniq);
    Delaunay vertices are the unique points in order, matching field_u's indexing."""
    m = meshio.read(vtu)
    uniq, inv = np.unique(np.round(m.points[:, :2], 9), axis=0, return_inverse=True)
    return Delaunay(uniq), inv.ravel(), len(uniq)


def field_u(m, k, inv, nuniq):
    f = np.asarray(m.point_data[k]).ravel()
    fu = np.empty(nuniq)
    fu[inv] = f                                  # duplicates share a value -> safe
    return fu


def bary_weights(dt, xq, yq):
    """Locate each query point in the Delaunay and return (vertex_idx, weights,
    valid_mask) for barycentric interpolation."""
    Q = np.c_[xq, yq]
    s = dt.find_simplex(Q)
    s_clip = np.clip(s, 0, None)
    T = dt.simplices[s_clip]
    b = np.einsum("ijk,ik->ij", dt.transform[s_clip, :2], Q - dt.transform[s_clip, 2])
    W = np.c_[b, 1 - b.sum(axis=1)]
    return T, W, s >= 0


def interp(values, T, W, valid):
    out = (W * values[T]).sum(axis=1)
    out[~valid] = np.nan
    return out


# ------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yscale", choices=["fixed", "auto", "both"], default="both")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    hexA, hexB = hex_full_edge()
    hang = np.degrees(np.arctan2((hexB - hexA)[1], (hexB - hexA)[0])) % 180
    A, B = seed_gb_edges(hang)                           # interior GB edges near hang
    lgA, lgB = A[0], A[1]
    smA, smB = B[0], B[1]
    print(f"probe GBs (all interior, off-box): hex {hang:.0f}deg | "
          f"long/large L={A[2]:.3f} {A[3]:.0f}deg meanA={A[4]:.1e} | "
          f"short/small L={B[2]:.3f} {B[3]:.0f}deg meanA={B[4]:.1e}")

    hx, hy, hd = sample(hexA, hexB)
    lx, ly, ld = sample(lgA, lgB)
    sx, sy, sd = sample(smA, smB)

    hex_pvd = os.path.join(HERE, "vtu_out", "hex", "hex.pvd")
    seed_pvd = os.path.join(HERE, "vtu_out", "seed24", "seed24.pvd")
    htimes, hfiles = read_pvd(hex_pvd)
    stimes, sfiles = read_pvd(seed_pvd)
    idx = np.arange(0, len(hfiles), args.stride)
    times = htimes[idx]
    hfiles = [hfiles[i] for i in idx]
    sfiles = [sfiles[i] for i in idx]
    nfr = len(times)
    print(f"{nfr} frames, ln(w*tau) in [{times.min():.2f},{times.max():.2f}]")

    htri, hinv, hn = load_geom(hfiles[0])
    stri, sinv, sn = load_geom(sfiles[0])
    hT, hW, hV = bary_weights(htri, hx, hy)                      # hex edge on hex mesh
    lT, lW, lV = bary_weights(stri, lx, ly)                      # largest grain on seed
    sT, sW, sV = bary_weights(stri, sx, sy)                      # smallest grain on seed

    # one caching pass: profiles[col][field][frame] -> array(NPTS)
    cols = [f"hex GB edge ({hang:.0f}°)",
            f"long GB / large grains ({A[3]:.0f}°)",
            f"short GB / small grains ({B[3]:.0f}°)"]
    dists = [hd, ld, sd]
    store = {c: {"sto": [], "dis": []} for c in cols}
    for i in range(nfr):
        mh = meshio.read(hfiles[i])
        ms = meshio.read(sfiles[i])
        h_sto = field_u(mh, "storage_energy_density", hinv, hn)
        h_dis = field_u(mh, "dissipation_energy_density", hinv, hn)
        s_sto = field_u(ms, "storage_energy_density", sinv, sn)
        s_dis = field_u(ms, "dissipation_energy_density", sinv, sn)
        store[cols[0]]["sto"].append(interp(h_sto, hT, hW, hV))
        store[cols[0]]["dis"].append(interp(h_dis, hT, hW, hV))
        store[cols[1]]["sto"].append(interp(s_sto, lT, lW, lV))
        store[cols[1]]["dis"].append(interp(s_dis, lT, lW, lV))
        store[cols[2]]["sto"].append(interp(s_sto, sT, sW, sV))
        store[cols[2]]["dis"].append(interp(s_dis, sT, sW, sV))
        if (i + 1) % 25 == 0 or i == nfr - 1:
            print(f"  cached {i+1}/{nfr}")
    for c in cols:
        for k in ("sto", "dis"):
            store[c][k] = np.array(store[c][k])                  # (nfr, NPTS)

    # global y-limits per row (field), across the 3 columns
    def glim(kind):
        allv = np.concatenate([store[c][kind].ravel() for c in cols])
        allv = allv[np.isfinite(allv) & (allv > 0)]
        vmax = allv.max() if allv.size else 1.0
        return vmax / 1e8, vmax
    ylim = {"sto": glim("sto"), "dis": glim("dis")}
    rowname = {"sto": "storage energy density", "dis": "dissipation energy density"}
    rowcol = {"sto": "tab:blue", "dis": "tab:red"}

    def render(yscale, outfile):
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        lines = {}
        for r, kind in enumerate(("sto", "dis")):
            for cI, c in enumerate(cols):
                a = ax[r, cI]
                ln, = a.plot(dists[cI], store[c][kind][0], color=rowcol[kind], lw=1.8)
                lines[(r, cI)] = ln
                a.set_yscale("log")
                if yscale == "fixed":
                    a.set_ylim(*ylim[kind])
                a.grid(True, which="both", alpha=0.2)
                if r == 0:
                    a.set_title(c, fontsize=11)
                if r == 1:
                    a.set_xlabel("distance along line")
                if cI == 0:
                    a.set_ylabel(rowname[kind], color=rowcol[kind])
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        def draw(i):
            for r, kind in enumerate(("sto", "dis")):
                for cI, c in enumerate(cols):
                    y = store[c][kind][i]
                    lines[(r, cI)].set_ydata(y)
                    if yscale == "auto":
                        a = ax[r, cI]
                        pos = y[np.isfinite(y) & (y > 0)]
                        if pos.size:
                            vmax = pos.max()
                            a.set_ylim(vmax / 1e4, vmax * 1.5)
            fig.suptitle(f"line profiles vs distance -- shear, "
                         f"$\\ln(\\omega\\tau_M)$ = {times[i]:+.2f}  "
                         f"(frame {i+1}/{nfr}, y={yscale})", fontsize=13)
            return list(lines.values())

        anim = FuncAnimation(fig, draw, frames=nfr, blit=False)
        anim.save(outfile, writer=PillowWriter(fps=args.fps))
        plt.close(fig)
        print(f"saved {outfile}")

    if args.yscale in ("fixed", "both"):
        render("fixed", os.path.join(HERE, "line_profiles_fixed.gif"))
    if args.yscale in ("auto", "both"):
        render("auto", os.path.join(HERE, "line_profiles_auto.gif"))


if __name__ == "__main__":
    main()
