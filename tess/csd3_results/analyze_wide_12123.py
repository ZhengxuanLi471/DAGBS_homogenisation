#!/usr/bin/env python3
"""Tests B & C analysis for the (3,12,12) members-7/8/9 wide-band re-run.

B: do members 7-9 show TWO modes at once -- the slope~-1 "bump" at ln~13-15
   AND the genuine triangle (L^-3.245) peak entering the extended window
   (predicted 17.6 / 20.0 / 22.4 for L = 6.5e-4 / 3.0e-4 / 1.3e-4)?
C: member 9 at den=100 vs den=50 -- are bump and triangle peak mesh-independent?

tau_M correction: the wide run of member 8 (ratio_9.755e-04) had its eta_ss
calibration solve stagnate to a garbage tau_M (1.59e-2 vs production 1.44e-4).
Since tau_M enters ONLY as the diffusion coefficient C_d and the loss prefactor,
a wrong tau_M is exactly a rigid shift of the master-curve frequency axis (the
prefactor factor cancels the C_d/omega field shift). We therefore plot every
curve on the corrected axis  x = ln_omega + ln(tau_ref / tau_used)  with tau_ref
the trusted production tau_M, and validate the shift by overlap with production.
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

# triangle facet length per key
with open(os.path.normpath(os.path.join(HERE, "..", "12123_sweep",
                                         "tessellation_output.json"))) as f:
    data = json.load(f)
facet = {}
for key, (pts, regions) in data.items():
    pa = np.array(pts)
    perims = [np.linalg.norm(pa[np.array(r) - 1] - np.roll(pa[np.array(r) - 1], -1, axis=0),
                             axis=1).sum() for r in regions if len(r) == 3]
    facet[key] = min(perims) / 3.0


def tau_of(path):
    return float(re.search(r"tau_M=([\d.e+-]+)",
                           open(path.replace(".csv", "_meta.txt")).read()).group(1))


def prod_tau(key):
    return tau_of(os.path.join(SW, f"refine_{key}_frac50_shear.csv"))


# consolidated triangle-mode law (scaling_final_12123.py, 7 pts): -3.09 ln L -4.45
LAW_SLOPE, LAW_INT = -3.092, -4.449
# member 8 (ratio_9.755e-04) wide run had a bad eta_ss calibration (tau_M ~110x
# off) -> excluded from the wide-band figure; its bump comes from production.
members = [("7", "ratio_2.154e-03"), ("9", "ratio_4.417e-04")]


def load(path):
    df = pd.read_csv(path)
    return df


def peaks_above(df, lncut):
    ln = df.ln_omega.values
    lc = np.log(df.Cxyxy_imag.values)
    sel = ln > lncut
    pk, pr = find_peaks(lc[sel], prominence=0.02)
    return [(ln[sel][p], df.Cxyxy_imag.values[sel][p], pr["prominences"][n])
            for n, p in enumerate(pk)]


# ---------------------------------------------------------------- Test B figure
fig, axes = plt.subplots(1, len(members), figsize=(7 * len(members), 6), sharey=True)
axes = np.atleast_1d(axes)
print("=== wide-band members 7/9 (member 8 excluded: bad wide calibration) ===")
for ax, (mlabel, key) in zip(axes, members):
    L = facet[key]
    tref = prod_tau(key)
    wide = load(os.path.join(SW, f"refine_{key}_frac50_wide_shear.csv"))
    tw = tau_of(os.path.join(SW, f"refine_{key}_frac50_wide_shear.csv"))
    shift = np.log(tref / tw)                       # 0 for clean, +/- for corrupted
    xw = wide.ln_omega.values + shift
    prod = load(os.path.join(SW, f"refine_{key}_frac50_shear.csv"))

    ax.semilogy(prod.ln_omega, prod.Cxyxy_imag, color="0.6", lw=3, alpha=0.6,
                label="production [-3,18]")
    ax.semilogy(xw, wide.Cxyxy_imag, color="navy", lw=1.3,
                label=f"wide [-3,24]{' (shift %+.2f)' % shift if abs(shift) > .01 else ''}")
    pred = LAW_SLOPE * np.log(L) + LAW_INT
    ax.axvline(pred, color="crimson", ls="--", lw=1.2)
    ax.text(pred, 1.1 * wide.Cxyxy_imag.max(), f"tri pred {pred:.1f}",
            color="crimson", fontsize=9, ha="center")
    pk = peaks_above(pd.DataFrame({"ln_omega": xw, "Cxyxy_imag": wide.Cxyxy_imag.values}), 1.5)
    txt = "  ".join(f"{p[0]:.1f}" for p in pk)
    ax.set_title(f"member {mlabel}: L={L:.2e}\ndetected peaks (ln>1.5): {txt}", fontsize=10)
    ax.set_xlabel(r"$\ln(\omega\tau_M)$"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    print(f" member {mlabel} L={L:.3e} shift={shift:+.3f} | peaks: " +
          ", ".join(f"ln={p[0]:.2f}(prom {p[2]:.3f})" for p in pk) +
          f" | tri law predicts {pred:.2f}")
axes[0].set_ylabel(r"$C''$")
fig.suptitle("(3,12,12) wide band: network RC bump (ln~14) + genuine triangle Coble peak (L^-3)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "wideB_12123.png"), dpi=130, bbox_inches="tight")
print("saved wideB_12123.png")

# ---------------------------------------------------------------- Test C figure
fig2, ax = plt.subplots(figsize=(9, 6))
key = "ratio_4.417e-04"; L = facet[key]
d50 = load(os.path.join(SW, f"refine_{key}_frac50_wide_shear.csv"))
d100 = load(os.path.join(SW, f"refine_{key}_frac100_wide_shear.csv"))
ax.semilogy(d50.ln_omega, d50.Cxyxy_imag, color="navy", lw=1.6, label="den=50 (L/50)")
ax.semilogy(d100.ln_omega, d100.Cxyxy_imag, color="orange", lw=1.6, ls="--", label="den=100 (L/100)")
pred = LAW_SLOPE * np.log(L) + LAW_INT
ax.axvline(pred, color="crimson", ls=":", label=f"tri law {pred:.1f}")
ax.set_xlabel(r"$\ln(\omega\tau_M)$"); ax.set_ylabel(r"$C''$")
ax.set_title(f"Test C: member 9 (L={L:.2e}) mesh independence")
ax.legend(); ax.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(HERE, "wideC_12123.png"), dpi=130, bbox_inches="tight")
print("\n=== Test C: member 9 den=50 vs den=100 ===")
for tag, d in (("den50", d50), ("den100", d100)):
    pk = peaks_above(d, 1.5)
    print(f" {tag}: peaks " + ", ".join(f"ln={p[0]:.2f}(prom {p[2]:.3f})" for p in pk))
print("saved wideC_12123.png")
