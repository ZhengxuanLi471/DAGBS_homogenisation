"""Overlay the ensemble-mean viscoelastic response of every sigma vs the hex model.

Reads each sigma's per-ensemble mean (fig2_ensemble_mean.csv, written by
make_ensemble_figs.py) and overlays the means colored by sigma, with the regular
hexagon reference (black) on top. Produces two 4-panel figures, one per frequency
normalization:
    overlay_all_sigma_indivtau.png   (each sigma's mean vs its own tau_M)
    overlay_all_sigma_hextau.png     (each sigma's mean vs the hex tau_M)

Panels: (a) |G*|/G0  (b) Q^-1  (c) scaled compliances sJ1(solid)/sJ2(dashed)
        (d) Cole-Cole sJ1 vs sJ2.  Panels (c)/(d) are resolution-limited (Delta).

Run: /home/zl471/miniconda3/envs/ngsolve/bin/python overlay_all_sigma.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
HEX_DIR = os.path.join(ROOT, "sigma_0.05")
NU = 0.35
SIGMA_LABELS = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5"]

# --- shared hexagon reference quantities (same derive() math) ---------------
with open(os.path.join(HEX_DIR, "hex_reference_tau.txt")) as fh:
    TAU_HEX = float(fh.read().strip())
hex_df = pd.read_csv(os.path.join(HEX_DIR, "hex_reference_shear.csv"))


def derive(df):
    G1 = df["Cxyxy_real"].to_numpy()
    G2 = df["Cxyxy_imag"].to_numpy()
    x = df["omega"].to_numpy()
    Gc = G1 + 1j * G2
    G0 = G1[-1]
    low = x < 0.1
    if low.sum() < 3:
        low = np.zeros_like(x, dtype=bool); low[:3] = True
    G_ss = float(np.median(G2[low] ** 2 / G1[low]))
    Delta = G0 / G_ss - 1.0
    jn = G0 / Gc
    return (x, np.abs(Gc) / G0, G2 / G1,
            (jn.real - 1.0) / Delta, (-jn.imag - 1.0 / x) / Delta, Delta)


hx, h_absG, h_Qinv, h_sJ1, h_sJ2, h_Delta = derive(hex_df)

# --- load per-sigma means ---------------------------------------------------
means = {}
for lab in SIGMA_LABELS:
    p = os.path.join(ROOT, f"sigma_{lab}", "fig2_ensemble_mean.csv")
    if os.path.exists(p):
        means[lab] = pd.read_csv(p)
    else:
        print(f"WARN: missing {p}")

cmap = cm.viridis
norm_c = Normalize(vmin=float(SIGMA_LABELS[0]), vmax=float(SIGMA_LABELS[-1]))
HEXST = dict(color="k", lw=2.6, zorder=10)


def style_axes(ax):
    ax[0, 0].set(xscale="log", xlabel=r"$\omega\tau_M$", ylabel=r"$|G^*|/G_0$",
                 ylim=(0, 1.05))
    ax[0, 1].set(xscale="log", yscale="log", xlabel=r"$\omega\tau_M$",
                 ylabel=r"$Q^{-1}$")
    ax[1, 0].set(xscale="log", xlabel=r"$\omega\tau_M$",
                 ylabel="Scaled compliance", ylim=(0, 1.05))
    ax[1, 1].set(xlabel=r"$(G_0 J_1-1)/\Delta$",
                 ylabel=r"$(G_0 J_2-1/\omega\tau_M)/\Delta$",
                 xlim=(0, 1), ylim=(0, 0.35))
    for a_ in ax.ravel():
        a_.grid(True, which="both", alpha=0.15)


def make_overlay(norm):
    suf = "indiv" if norm == "indiv" else "hex"
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
    for lab in SIGMA_LABELS:
        if lab not in means:
            continue
        d = means[lab]
        x = d["omega_tauM"].to_numpy()
        col = cmap(norm_c(float(lab)))
        absG = d[f"mean_absG_over_G0_{suf}"].to_numpy()
        Qinv = d[f"mean_Qinv_{suf}"].to_numpy()
        sJ1 = d[f"mean_sJ1_{suf}"].to_numpy()
        sJ2 = d[f"mean_sJ2_{suf}"].to_numpy()
        ax[0, 0].plot(x, absG, color=col, lw=1.8, label=fr"$\sigma$={lab}")
        ax[0, 1].plot(x, Qinv, color=col, lw=1.8)
        ax[1, 0].plot(x, sJ1, color=col, lw=1.6)
        ax[1, 0].plot(x, sJ2, color=col, lw=1.6, ls="--")
        ax[1, 1].plot(sJ1, sJ2, color=col, lw=1.6)

    # hexagon reference on top
    ax[0, 0].plot(hx, h_absG, label="hex", **HEXST)
    ax[0, 1].plot(hx, h_Qinv, **HEXST)
    ax[1, 0].plot(hx, h_sJ1, **HEXST)
    ax[1, 0].plot(hx, h_sJ2, ls="--", **HEXST)
    ax[1, 1].plot(h_sJ1, h_sJ2, **HEXST)

    style_axes(ax)
    ax[0, 0].legend(fontsize=8, ncol=2, title=r"seed mean", title_fontsize=8)
    ax[1, 0].plot([], [], color="0.4", lw=1.6, label=r"$sJ_1$ (solid)")
    ax[1, 0].plot([], [], color="0.4", lw=1.6, ls="--", label=r"$sJ_2$ (dashed)")
    ax[1, 0].legend(fontsize=8)
    nl = r"individual $\tau_M$" if norm == "indiv" else r"hex $\tau_M$"
    fig.suptitle(f"DAGBS ensemble means across disorder sigma, nu={NU} "
                 f"— freq normalized by {nl}  (hex tau_M={TAU_HEX:.3e})",
                 fontsize=11)
    fig.text(0.5, 0.005,
             "panels (c)/(d) resolution-limited: seed grid reaches only "
             r"$\omega\tau_M\approx0.05$, so $\Delta$ is poorly constrained",
             ha="center", fontsize=7, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    out = os.path.join(ROOT, f"overlay_all_sigma_{suf}tau.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", os.path.basename(out))


make_overlay("indiv")
make_overlay("hex")
