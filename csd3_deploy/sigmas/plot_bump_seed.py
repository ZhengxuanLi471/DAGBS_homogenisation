"""Isolate one seed's viscoelastic response and overlay it on the hex reference,
fig2-style (4 panels). Used to showcase the most prominent bump-and-return Q^-1
anomaly: sigma=0.05 seed 89, whose Q^-1 hump peaks at ln(omega*tau_M) ~ 13.9 --
entirely inside the high-omega extension (it was invisible at the old ln=10 cutoff).

    /home/zl471/miniconda3/envs/ngsolve/bin/python plot_bump_seed.py [sigma seed]
defaults to 0.05 89. Writes bump_seed_<sigma>_<seed>.png.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
HEX_DIR = os.path.join(ROOT, "sigma_0.05")

SIGMA = sys.argv[1] if len(sys.argv) > 1 else "0.05"
SEED = sys.argv[2] if len(sys.argv) > 2 else "89"


def derive(df):
    """(x, |G*|/G0, Q^-1, sJ1, sJ2, Delta) -- identical to make_ensemble_figs."""
    G1 = df["Cxyxy_real"].to_numpy()
    G2 = df["Cxyxy_imag"].to_numpy()
    x = df["omega"].to_numpy()
    Gc = G1 + 1j * G2
    G0 = G1[-1]
    low = x < 0.1
    if low.sum() < 3:
        low = np.zeros_like(x, dtype=bool)
        low[:3] = True
    G_ss = float(np.median(G2[low] ** 2 / G1[low]))
    Delta = G0 / G_ss - 1.0
    jn = G0 / Gc
    return x, np.abs(Gc) / G0, G2 / G1, (jn.real - 1.0) / Delta, \
        (-jn.imag - 1.0 / x) / Delta, Delta


seed_csv = os.path.join(ROOT, f"sigma_{SIGMA}",
                        f"Seed_seeds_{SEED}_energy_real_im_data_shear.csv")
sdf = pd.read_csv(seed_csv).sort_values("ln_omega")
hdf = pd.read_csv(os.path.join(HEX_DIR, "hex_reference_shear.csv")).sort_values("ln_omega")

sx, s_absG, s_Q, s_J1, s_J2, s_D = derive(sdf)
hx, h_absG, h_Q, h_J1, h_J2, h_D = derive(hdf)

SEEDST = dict(color="crimson", lw=2.0, zorder=3)
HEXST = dict(color="black", lw=2.0, ls="--", zorder=2)

fig, ax = plt.subplots(2, 2, figsize=(11, 9))

ax[0, 0].plot(sx, s_absG, label=f"seed {SEED} (sigma={SIGMA})", **SEEDST)
ax[0, 0].plot(hx, h_absG, label="hex ref", **HEXST)
ax[0, 0].set(xscale="log", xlabel=r"$\omega\tau_M$", ylabel=r"$|G^*|/G_0$", ylim=(0, 1.05))
ax[0, 0].legend(fontsize=9)

ax[0, 1].plot(sx, s_Q, **SEEDST)
ax[0, 1].plot(hx, h_Q, **HEXST)
ax[0, 1].set(xscale="log", yscale="log", xlabel=r"$\omega\tau_M$", ylabel=r"$Q^{-1}$")
# mark the bump peak in the high-omega region
hi = sx > np.exp(10.0)
if hi.any():
    ipk = np.where(hi)[0][np.argmax(s_Q[hi])]
    ax[0, 1].plot(sx[ipk], s_Q[ipk], "o", color="crimson", ms=8, zorder=4)
    ax[0, 1].annotate(f"bump peak\n$\\ln\\,\\omega\\tau_M$={np.log(sx[ipk]):.1f}",
                      (sx[ipk], s_Q[ipk]), textcoords="offset points",
                      xytext=(-10, 14), fontsize=8, ha="right", color="crimson")

ax[1, 0].plot(sx, s_J1, color="crimson", lw=2.0, label=r"$sJ_1$ seed")
ax[1, 0].plot(sx, s_J2, color="crimson", lw=2.0, ls=":", label=r"$sJ_2$ seed")
ax[1, 0].plot(hx, h_J1, color="black", lw=1.6, label=r"$sJ_1$ hex")
ax[1, 0].plot(hx, h_J2, color="black", lw=1.6, ls=":", label=r"$sJ_2$ hex")
ax[1, 0].set(xscale="log", xlabel=r"$\omega\tau_M$", ylabel="Scaled compliance", ylim=(0, 1.05))
ax[1, 0].legend(fontsize=8)

ax[1, 1].plot(s_J1, s_J2, **SEEDST)
ax[1, 1].plot(h_J1, h_J2, **HEXST)
ax[1, 1].set(xlabel=r"$(G_0 J_1-1)/\Delta$",
             ylabel=r"$(G_0 J_2-1/\omega\tau_M)/\Delta$", xlim=(0, 1), ylim=(0, 0.35))

for a_ in ax.ravel():
    a_.grid(True, which="both", alpha=0.15)

fig.suptitle(f"Bump-and-return showcase: sigma={SIGMA} seed {SEED} vs hex "
             f"(nu=0.35) -- Delta_seed={s_D:.3f}, Delta_hex={h_D:.3f}",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = os.path.join(ROOT, f"bump_seed_{SIGMA}_{SEED}.png")
fig.savefig(out, dpi=140)
print(f"saved {out}")
print(f"seed {SEED}: Delta={s_D:.4f}, grid ln_omega[{np.log(sx.min()):.2f},{np.log(sx.max()):.2f}]")
print(f"hex: Delta={h_D:.4f}")
