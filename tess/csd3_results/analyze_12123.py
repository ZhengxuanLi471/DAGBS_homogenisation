#!/usr/bin/env python3
"""Secondary-peak scaling analysis for the (3,12,12) ratio sweep.

Overlays C''(omega*tau_M) and Q^-1 for the 9 swept geometries, detects the
secondary (small-triangle) loss peak, and fits ln(omega_peak) vs ln(L_facet)
against the L^-2 / L^-3 diffusion-scaling guides.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "12123_sweep")
TESS = os.path.normpath(os.path.join(HERE, "..", "12123_sweep", "tessellation_output.json"))

# facet length per key (triangle perimeter / 3, measured from the geometry)
with open(TESS) as f:
    data = json.load(f)
facet = {}
for key, (pts, regions) in data.items():
    pts_a = np.array(pts)
    perims = [np.linalg.norm(pts_a[np.array(r) - 1] - np.roll(pts_a[np.array(r) - 1], -1, axis=0), axis=1).sum()
              for r in regions if len(r) == 3]
    facet[key] = min(perims) / 3.0

rows = []
curves = {}
for path in sorted(glob.glob(os.path.join(SWEEP, "refine_ratio_*_frac50_shear.csv"))):
    key = re.search(r"refine_(ratio_[0-9.e-]+)_frac", os.path.basename(path)).group(1)
    df = pd.read_csv(path)
    meta = dict(re.findall(r"(\w+)=([\d.e+-]+)",
                open(path.replace(".csv", "_meta.txt")).read()))
    L = facet[key]
    qinv = df.Cxyxy_imag / df.Cxyxy_real
    curves[key] = (df, qinv, L, float(meta["tau_M"]))

    # secondary peak: local max of C'' above the main relaxation (ln > 1.5)
    ln = df.ln_omega.values
    logc2 = np.log(df.Cxyxy_imag.values)
    sel = ln > 1.5
    pk, props = find_peaks(logc2[sel], prominence=0.02)
    peak_ln = peak_c2 = peak_prom = np.nan
    if len(pk):
        best = pk[np.argmax(props["prominences"])]
        peak_ln = ln[sel][best]
        peak_c2 = df.Cxyxy_imag.values[sel][best]
        peak_prom = props["prominences"].max()
    rows.append(dict(key=key, L=L, tau_M=float(meta["tau_M"]), G_U=float(meta["G_U"]),
                     peak_ln=peak_ln, peak_c2=peak_c2, prominence=peak_prom))

tab = pd.DataFrame(rows).sort_values("L", ascending=False)
print(tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
order = tab.key.tolist()
colors = plt.cm.viridis(np.linspace(0, 0.95, len(order)))

for c, key in zip(colors, order):
    df, qinv, L, tau = curves[key]
    lab = f"L={L:.2e}"
    axes[0].semilogy(df.ln_omega, df.Cxyxy_imag, color=c, lw=1.4, label=lab)
    axes[1].semilogy(df.ln_omega, qinv, color=c, lw=1.4, label=lab)
    pk = tab[tab.key == key].peak_ln.iloc[0]
    if np.isfinite(pk):
        axes[0].axvline(pk, color=c, ls=":", alpha=0.5)

axes[0].set_xlabel(r"$\ln(\omega\tau_M)$"); axes[0].set_ylabel(r"$C''$")
axes[0].set_title("(3,12,12) sweep: loss modulus (dotted = detected secondary peak)")
axes[0].legend(fontsize=8, loc="lower left")
axes[1].set_xlabel(r"$\ln(\omega\tau_M)$"); axes[1].set_ylabel(r"$Q^{-1}$")
axes[1].set_title("attenuation")
axes[1].legend(fontsize=8, loc="lower left")

# Genuine triangle peaks: only members whose L^-3-extrapolated peak lies
# safely inside the band (the smallest-L members' true peaks fall beyond
# ln=18; their detected bumps are the pinned artifact, not the relaxation).
ok = tab.dropna(subset=["peak_ln"])
genuine = ok[ok.L >= 1e-3]
pinned = ok[ok.L < 1e-3]
x, y = np.log(genuine.L.values), genuine.peak_ln.values
axes[2].plot(x, y, "o", ms=8, color="crimson", label="triangle peak (in band)")
if len(pinned):
    axes[2].plot(np.log(pinned.L.values), pinned.peak_ln.values, "s", ms=8,
                 mfc="none", mec="gray", label="pinned bump (true peak > band)")
for _, r in ok.iterrows():
    axes[2].annotate(f"{r.key.split('_')[1]}", (np.log(r.L), r.peak_ln),
                     fontsize=7, xytext=(4, 4), textcoords="offset points")
A = np.polyfit(x, y, 1)
print(f"\nscaling fit over {len(x)} genuine peaks: "
      f"ln(omega_peak*tau_M) = {A[0]:.3f}*ln L + {A[1]:.3f}")
xx = np.linspace(np.log(ok.L.min()) - 0.3, x.max() + 0.3, 10)
axes[2].plot(xx, np.polyval(A, xx), "k-", lw=1,
             label=f"fit slope = {A[0]:.2f}")
for s, ls in ((-2, "--"), (-3, ":")):
    axes[2].plot(xx, y.mean() + s * (xx - x.mean()), color="gray", ls=ls, lw=1,
                 label=f"slope {s}")
axes[2].axhline(18, color="k", lw=0.5, ls="--")
axes[2].text(xx[0], 18.15, "band edge", fontsize=8)
axes[2].set_xlabel(r"$\ln L_{\rm facet}$"); axes[2].set_ylabel(r"$\ln(\omega_{\rm peak}\tau_M)$")
axes[2].set_title("secondary-peak frequency vs facet length")
axes[2].legend()

fig.tight_layout()
out = os.path.join(HERE, "scaling_12123.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
