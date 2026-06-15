"""Assemble a frequency movie (GIF) from the per-frequency VTU frames -- pure Python,
no ParaView. Reads the <name>.pvd collection (which carries the ln(omega*tau_M)
timeline) with meshio, renders a 2x2 matplotlib panel per frame, and writes a GIF
via PillowWriter (the ngsolve venv has no ffmpeg, so GIF not mp4).

Panels:
  (a) |displacement| = sqrt(|uR|^2+|uI|^2)   (b) storage energy density
  (c) dissipation energy density (lit on GBs) (d) Q^-1 spectrum with a moving marker

Color scales AUTOSCALE per frame (each frame's colorbar is rescaled to that frame's
own min/max, LogNorm with a 4-decade floor) so the spatial pattern is visible at every
frequency rather than washing out against a fixed global scale. (Trade-off: absolute
magnitudes are NOT comparable frame-to-frame; read the colorbar each frame.)

Run locally after syncing vtu_out/ back from CSD3:
    python vtu_movie.py --target hex                 # hex_movie.gif
    python vtu_movie.py --target seed                # seed24_movie.gif
    python vtu_movie.py --target hex --stride 20     # quick preview (every 20th frame)
"""
import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import meshio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable

HERE = os.path.dirname(os.path.abspath(__file__))
HEX_CSV = os.path.normpath(os.path.join(HERE, "..", "sigmas", "sigma_0.05",
                                        "hex_reference_shear.csv"))
SEED_CSV = os.path.normpath(os.path.join(HERE, "..", "sigmas", "sigma_0.45",
                                         "Seed_seeds_24_energy_real_im_data_shear.csv"))


def read_pvd(pvd):
    root = ET.parse(pvd).getroot()
    items = [(float(ds.get("timestep")), ds.get("file"))
             for ds in root.iter("DataSet")]
    items.sort()
    base = os.path.dirname(pvd)
    times = np.array([t for t, _ in items])
    files = [os.path.join(base, f) for _, f in items]
    return times, files


def field(m, key):
    return np.asarray(m.point_data[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["hex", "seed"], required=True)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    name = "hex" if args.target == "hex" else "seed24"
    pvd = os.path.join(HERE, "vtu_out", name, f"{name}.pvd")
    csv = HEX_CSV if args.target == "hex" else SEED_CSV
    times, files = read_pvd(pvd)
    idx = np.arange(0, len(files), args.stride)
    times, files = times[idx], [files[i] for i in idx]
    print(f"{name}: {len(files)} frames, ln(w*tau) in [{times.min():.2f},{times.max():.2f}]")

    # fixed triangulation from frame 0 (mesh constant across frames)
    m0 = meshio.read(files[0])
    x, y = m0.points[:, 0], m0.points[:, 1]
    tris = np.vstack([c.data for c in m0.cells if c.type == "triangle"])
    triang = Triangulation(x, y, tris)

    def disp_mag(m):
        uR, uI = field(m, "disp_real"), field(m, "disp_imag")
        return np.sqrt((uR ** 2).sum(1) + (uI ** 2).sum(1))

    def scalar(m, k):
        return field(m, k).ravel()

    cmaps = dict(disp="viridis", sto="inferno", dis="magma")

    qdf = pd.read_csv(csv).sort_values("ln_omega")
    q_ln = qdf["ln_omega"].to_numpy()
    q_val = (qdf["Cxyxy_imag"] / qdf["Cxyxy_real"]).to_numpy()

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    (a_d, a_s), (a_x, a_q) = ax
    titles = {a_d: ("disp", r"$|u|=\sqrt{|u_R|^2+|u_I|^2}$"),
              a_s: ("sto", "storage energy density"),
              a_x: ("dis", "dissipation energy density (GB)")}
    # a dedicated, attached colorbar axis per field panel (cleared + redrawn per frame)
    caxes = {a: make_axes_locatable(a).append_axes("right", size="4%", pad=0.05)
             for a in titles}
    # Q^-1 panel (static curve + moving marker)
    a_q.semilogy(q_ln, q_val, color="0.4", lw=1.5)
    a_q.set_xlabel(r"$\ln(\omega\tau_M)$"); a_q.set_ylabel(r"$Q^{-1}$")
    a_q.set_title("spectrum (marker = current frame)", fontsize=11)
    a_q.grid(True, which="both", alpha=0.2)
    marker, = a_q.plot([], [], "o", color="crimson", ms=10, zorder=5)
    vline = a_q.axvline(q_ln[0], color="crimson", lw=1, alpha=0.5)

    def draw(i):
        m = meshio.read(files[i])
        vals = dict(disp=disp_mag(m),
                    sto=scalar(m, "storage_energy_density"),
                    dis=scalar(m, "dissipation_energy_density"))
        for a, (k, ttl) in titles.items():
            v = vals[k]
            vmax = max(float(np.nanmax(v)), 1e-300)
            floor = vmax / 1e4                       # per-frame 4-decade window
            a.clear(); a.set_aspect("equal"); a.axis("off"); a.set_title(ttl, fontsize=11)
            tpc = a.tripcolor(triang, np.maximum(v, floor), shading="gouraud",
                              norm=LogNorm(floor, vmax), cmap=cmaps[k])
            caxes[a].clear()
            fig.colorbar(tpc, cax=caxes[a])
        ln = times[i]
        marker.set_data([ln], [np.interp(ln, q_ln, q_val)])
        vline.set_xdata([ln, ln])
        fig.suptitle(f"{name}: shear, $\\ln(\\omega\\tau_M)$ = {ln:+.2f}  "
                     f"(frame {i+1}/{len(files)}) -- colorbars autoscale per frame",
                     fontsize=13)
        return []

    anim = FuncAnimation(fig, draw, frames=len(files), blit=False)
    out = os.path.join(HERE, f"{name}_movie.gif")
    anim.save(out, writer=PillowWriter(fps=args.fps))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
