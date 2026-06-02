# Reproduce Rudge (2025) Figure 2: viscoelastic response of the hexagon model at
# Poisson's ratio nu = 0.3, vs dimensionless frequency omega*tau_M.
#
#   (a) |G*|/G0     (b) Q^-1 = G2/G1
#   (c) scaled compliances (G0 J1 - 1)/Delta , (G0 J2 - 1/wt)/Delta
#   (d) Cole-Cole: (G0 J2 - 1/wt)/Delta  vs  (G0 J1 - 1)/Delta
#
# FEM solid curves come from our DAGBS solver (the paper's exact equations);
# Maxwell / Andrade / extended-Burgers overlays use the parameters MEASURED here
# (G0, Delta, tau_M, tau_A, alpha). Run:  python reproduce_fig2.py [refine_h]

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gamma as Gamma, hyp2f1
from mpi4py import MPI

from ngsolve import SetNumThreads
from meshes import MakeMesh
from main import build_spaces
from main import solve_rve
from calibrate_tau import (measure_tau_M, storage_modulus, diffusional_loss,
                           fit_andrade_time, SHEAR)

SetNumThreads(32)

NU = 0.3                      # paper Figure 2
MU = 1.0
ALPHA = 0.3672092            # Andrade exponent for the 120 deg triple junction
REFINE_H = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-5
X = np.logspace(-2.0, 4.0, 61)   # omega*tau_M (code-omega with diff_coeff=tau_M)


# --- hexagon geometry (matches real_im_hex_energy.py) ----------------------
a = np.sqrt(3)
pts0 = [(0, 0), (3/4, 0), (1/2, a/4), (0, a/4),
        (9/4, 0), (5/2, a/4), (2, 3*a/4), (1, 3*a/4),
        (3, 0), (3, a/4), (3, a), (9/4, a), (3/4, a), (0, a)]
pts1 = [(x - 1.5, y - 0.5 * a) for (x, y) in pts0]
sf = np.sqrt(1/50 / (3 * np.sqrt(3) / 2))
pts = [(x * sf, y * sf) for (x, y) in pts1]
regions = [(1, 2, 3, 4), (2, 5, 6, 7, 8, 3), (5, 9, 10, 6),
           (6, 10, 11, 12, 7), (8, 7, 12, 13), (4, 3, 8, 13, 14)]
NG = len(regions)

(_, _, mesh, _, contact_pairs, outer_contact_pairs,
 _cbl, _ocl, junction_incidence) = MakeMesh(
    pts, regions, maxh=0.1 * sf, comm=MPI.COMM_WORLD,
    core_frac=0.01 * sf, refine_h=REFINE_H)
print(f"mesh: ne={mesh.ne} nv={mesh.nv} (refine_h={REFINE_H})")

spaces = build_spaces(mesh, contact_pairs, outer_contact_pairs,
                      order_bulk=2, order_gb=1, junction_incidence=junction_incidence)
gni = spaces[4]

tau_M, info = measure_tau_M(spaces, mesh, contact_pairs, outer_contact_pairs,
                            junction_incidence, num_grains=NG, nu=NU, mu=MU)
G0 = info["G_U"]
print(f"tau_M={tau_M:.6e}  G0={G0:.6f}  eta_ss={info['eta_ss']:.6e}")


# --- FEM frequency sweep (diff_coeff=tau_M => code-omega == omega*tau_M) ----
G1 = np.empty_like(X); G2 = np.empty_like(X)
for i, x in enumerate(X):
    gfu, mesh_, conv = solve_rve(spaces, mesh, contact_pairs, outer_contact_pairs,
                                 SHEAR, nu=NU, mu=MU, omega=float(x),
                                 solver='cg', rtol=1e-7,
                                 junction_incidence=junction_incidence, diff_coeff=tau_M)
    G1[i] = storage_modulus(gfu, mesh, NU, MU, NG)
    G2[i] = diffusional_loss(gfu, mesh, contact_pairs, gni, float(x), diff_coeff=tau_M)
    if not conv:
        print(f"  (warn) x={x:.3e} not converged")

Gc_fem = G1 + 1j * G2

# steady-state modulus and relaxation strength from the low-omega plateau
low = X < 0.05
G_ss = float(np.median(G2[low]**2 / G1[low]))
Delta = G0 / G_ss - 1.0

# Andrade time: fit the full high-omega Andrade form over the resolved power-law
# window (alpha fixed by the 120 deg triple junction). The leading-order Q^-1
# asymptote is biased; the clean power law sits at high omega*tau_M (the x~10
# region is the transition shoulder, NOT the Andrade tail).
rA = fit_andrade_time(X, Gc_fem, G0, ALPHA, window=(300.0, 5000.0))   # tau_A/tau_M
tau_H_over_M = rA * (Gamma(1 - ALPHA) * Delta) ** (1.0 / ALPHA)
print(f"G_ss={G_ss:.6f}  Delta={Delta:.4f}  tau_A/tau_M={rA:.4f}  "
      f"tau_H/tau_M={tau_H_over_M:.4f}")
print("paper targets (nu=0.3): G0/mu~0.81  Delta~0.33  tau_A/tau_M~1.31")


# --- response-function helpers (unify FEM + analytic via complex modulus) ---
def derive(Gc, x):
    jn = G0 / Gc                 # = G0 J* = G0 J1 - i G0 J2
    return (np.abs(Gc) / G0, (-jn.imag) / jn.real,
            (jn.real - 1) / Delta, (-jn.imag - 1.0 / x) / Delta)

x = X
iw = 1j * x
# Maxwell = dashpot (viscosity eta) + spring of the *steady-state* compliance
# J_ss in series (paper sec 3): G0 J* = (1+Delta) + 1/(i x), so |G*|/G0 -> 1/(1+Delta)
# = G_ss/G0 at high omega (not 1).
jn_max = (1 + Delta) + 1/iw
jn_and = 1 + Gamma(1 + ALPHA) / (iw * rA) ** ALPHA + 1/iw
# extended Burgers: sJ̃ = J0(1 + Delta 2F1(1,a,1+a,-s*tau_H) + 1/(s*tau_M)) (H13)
jn_bur = 1 + Delta * hyp2f1(1, ALPHA, 1 + ALPHA, -iw * tau_H_over_M) + 1/iw
models = {
    "FEM": (Gc_fem, dict(color="C0", lw=2)),
    "Extended Burgers": (G0 / jn_bur, dict(color="C0", ls="--", lw=1)),
    "Andrade": (G0 / jn_and, dict(color="C0", ls=":", lw=1)),
    "Maxwell": (G0 / jn_max, dict(color="C0", ls="-.", lw=1)),
}


# --- plot 4 panels ---------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for label, (Gc, st) in models.items():
    absG, Qi, sJ1, sJ2 = derive(Gc, x)
    ax[0, 0].plot(x, absG, label=label, **st)
    ax[0, 1].plot(x, Qi, **st)
    if label == "FEM":
        ax[1, 0].plot(x, sJ1, color="C0", lw=2, label=r"$(G_0 J_1-1)/\Delta$")
        ax[1, 0].plot(x, sJ2, color="C1", lw=2, label=r"$(G_0 J_2-1/\omega\tau_M)/\Delta$")
        ax[1, 1].plot(sJ1, sJ2, color="C0", lw=2)
    else:
        ax[1, 0].plot(x, sJ1, **st); ax[1, 0].plot(x, sJ2, **{**st, "color": "C1"})
        ax[1, 1].plot(sJ1, sJ2, **st)

ax[0, 0].set(xscale="log", xlabel=r"$\omega\tau_M$", ylabel=r"$|G^*|/G_0$", ylim=(0, 1.05))
ax[0, 0].legend(fontsize=8)
ax[0, 1].set(xscale="log", yscale="log", xlabel=r"$\omega\tau_M$", ylabel=r"$Q^{-1}$")
ax[0, 1].legend(["FEM", "Extended Burgers", "Andrade", "Maxwell"], fontsize=8)
ax[1, 0].set(xscale="log", xlabel=r"$\omega\tau_M$", ylabel="Scaled compliance", ylim=(0, 1.05))
ax[1, 0].legend(fontsize=8)
ax[1, 1].set(xlabel=r"$(G_0 J_1-1)/\Delta$", ylabel=r"$(G_0 J_2-1/\omega\tau_M)/\Delta$",
             xlim=(0, 1), ylim=(0, 0.35))
for a_ in ("a", "b", "c", "d"):
    pass
fig.suptitle(f"Rudge (2025) Fig. 2 reproduction — hexagon, nu={NU}  "
             f"(G0={G0:.3f}, Delta={Delta:.3f}, tau_A/tau_M={rA:.2f}; "
             f"omega*tau_M>~1e3 resolution-limited)", fontsize=10)
fig.tight_layout()
fig.savefig("reproduce_fig2.png", dpi=140)
print("saved reproduce_fig2.png")

# dump CSV
import pandas as pd
pd.DataFrame({"omega_tauM": X, "G1": G1, "G2": G2,
              "absG_over_G0": np.abs(Gc_fem)/G0, "Qinv": G2/G1}).to_csv(
    "reproduce_fig2.csv", index=False)
print("saved reproduce_fig2.csv")
