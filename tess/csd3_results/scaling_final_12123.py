#!/usr/bin/env python3
"""Consolidated (3,12,12) scaling: two distinct GB relaxations.

 - triangle's own diffusion mode: ln(omega_peak tau_M) ~ -3 ln L   (Coble L^-3)
 - network "RC" mode through the triangle resistors: slope ~ -1

Triangle peaks: production members 2-6 (band [-3,18]) + wide members 7,9.
Bump peaks: wide members 8,9 (+ member 7 shoulder if resolvable).
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
SW = os.path.join(HERE, "12123_sweep")
with open(os.path.normpath(os.path.join(HERE, "..", "12123_sweep",
                                         "tessellation_output.json"))) as f:
    data = json.load(f)
facet = {}
for key, (pts, regions) in data.items():
    pa = np.array(pts)
    perims = [np.linalg.norm(pa[np.array(r) - 1] - np.roll(pa[np.array(r) - 1], -1, axis=0),
                             axis=1).sum() for r in regions if len(r) == 3]
    facet[key] = min(perims) / 3.0


def all_peaks(df, lncut=1.5, prom=0.02):
    ln = df.ln_omega.values
    lc = np.log(df.Cxyxy_imag.values)
    sel = ln > lncut
    pk, pr = find_peaks(lc[sel], prominence=prom)
    return sorted(zip(ln[sel][pk], pr["prominences"]))


# NOTE: member 8's (ratio_9.755e-04) WIDE re-run had a bad eta_ss calibration
# (tau_M ~110x off) and is excluded entirely. Its bump is taken from its clean
# PRODUCTION run instead. Member 8's triangle peak (predicted ln~21) was never
# in any clean window, so the triangle mode simply does not use member 8.
tri = []   # (L, ln_peak)  -- triangle Coble mode (L^-3)
bump = []  # (L, ln_peak)  -- network RC mode (L^-1)

# triangle mode: production members 2-6 (peak inside [-3,18]) ...
prod_keys = ["ratio_1.132e-01", "ratio_5.126e-02", "ratio_2.321e-02",
             "ratio_1.051e-02", "ratio_4.758e-03"]
for key in prod_keys:
    df = pd.read_csv(os.path.join(SW, f"refine_{key}_frac50_shear.csv"))
    pks = all_peaks(df)
    if pks:
        tri.append((facet[key], max(pks, key=lambda p: p[1])[0]))
# ... plus the genuine triangle peaks that only entered the WIDE band (members 7, 9)
df7 = pd.read_csv(os.path.join(SW, "refine_ratio_2.154e-03_frac50_wide_shear.csv"))
tri.append((facet["ratio_2.154e-03"], all_peaks(df7)[-1][0]))     # highest ln = triangle
df9w = pd.read_csv(os.path.join(SW, "refine_ratio_4.417e-04_frac50_wide_shear.csv"))
tri.append((facet["ratio_4.417e-04"], all_peaks(df9w)[-1][0]))    # highest ln = triangle

# bump mode: from PRODUCTION runs only (members 8 and 9, both clean)
for key in ("ratio_9.755e-04", "ratio_4.417e-04"):
    df = pd.read_csv(os.path.join(SW, f"refine_{key}_frac50_shear.csv"))
    bump.append((facet[key], all_peaks(df)[0][0]))               # lowest ln = bump

tri = np.array(sorted(tri, reverse=True))
bump = np.array(sorted(bump, reverse=True))

# fits
xt, yt = np.log(tri[:, 0]), tri[:, 1]
At = np.polyfit(xt, yt, 1)
xb, yb = np.log(bump[:, 0]), bump[:, 1]
Ab = np.polyfit(xb, yb, 1) if len(xb) >= 2 else (np.nan, np.nan)

print(f"TRIANGLE mode ({len(tri)} pts, L {tri[:,0].min():.1e}-{tri[:,0].max():.1e}): "
      f"slope = {At[0]:.3f}  intercept = {At[1]:.3f}")
for L, p in tri:
    print(f"   L={L:.3e}  ln_peak={p:.2f}")
print(f"BUMP mode ({len(bump)} pts): slope = {Ab[0]:.3f}  intercept = {Ab[1]:.3f}")
for L, p in bump:
    print(f"   L={L:.3e}  ln_peak={p:.2f}")

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(xt, yt, "o", ms=9, color="crimson", label=f"triangle mode (slope {At[0]:.2f})")
ax.plot(xb, yb, "s", ms=9, color="seagreen", label=f"RC/network mode (slope {Ab[0]:.2f})")
xx = np.linspace(min(xt.min(), xb.min()) - 0.3, max(xt.max(), xb.max()) + 0.3, 10)
ax.plot(xx, np.polyval(At, xx), "-", color="crimson", lw=1)
ax.plot(xx, np.polyval(Ab, xx), "-", color="seagreen", lw=1)
# guides anchored at the triangle cloud mean
ax.plot(xx, yt.mean() - 3 * (xx - xt.mean()), "k--", lw=0.8, label="slope -3 (Coble)")
ax.plot(xx, yb.mean() - 1 * (xx - xb.mean()), "k:", lw=0.8, label="slope -1")
ax.set_xlabel(r"$\ln L_{\rm facet}$"); ax.set_ylabel(r"$\ln(\omega_{\rm peak}\tau_M)$")
ax.set_title("(3,12,12): two GB relaxations vs triangle facet length")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "scaling_final_12123.png"), dpi=130, bbox_inches="tight")
print("saved scaling_final_12123.png")
