# Reproduce Rudge (2025) Figure 1: hexagon-model properties vs Poisson's ratio nu.
#
#   (a) G0/mu (unrelaxed) and G_ss/mu (steady-state) shear moduli
#   (b) relaxation strength Delta = G0/G_ss - 1
#   (c) scaled Andrade time tau_A/tau_M
#
# The mesh/spaces are nu-independent, so they are built ONCE and nu is swept via
# solve_rve(nu=...). Per nu we calibrate tau_M (so code-omega == omega*tau_M),
# read G0 from the diff_coeff=0 limit, G_ss = G2^2/G1 on the low-omega plateau,
# and tau_A from the high-omega Q^-1 power-law tail (alpha fixed by the junction
# angle). Paper Table-2 rational fits are overlaid as a reference.
# Run:  python reproduce_fig1.py [refine_h]

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gamma as Gamma
from mpi4py import MPI

from ngsolve import SetNumThreads
from meshes import MakeMesh
from main import build_spaces, solve_rve
from calibrate_tau import (measure_tau_M, storage_modulus, diffusional_loss,
                           fit_andrade_time, SHEAR)

SetNumThreads(32)

MU = 1.0
ALPHA = 0.3672092
REFINE_H = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-5
NUS = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.45])
import os
if os.environ.get("FIG1_NUS"):   # quick-test subset, e.g. FIG1_NUS=0.0,0.2,0.45
    NUS = np.array([float(v) for v in os.environ["FIG1_NUS"].split(",")])
X_LOW = 1e-2                      # omega*tau_M on the steady-creep plateau
X_HIGH = np.logspace(2.5, 3.5, 6)   # resolved Andrade power-law window (~316-3162)


# --- hexagon geometry ------------------------------------------------------
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


def G12(nu, x, diff_coeff, solver='cg'):
    gfu, _, _ = solve_rve(spaces, mesh, contact_pairs, outer_contact_pairs, SHEAR,
                          nu=nu, mu=MU, omega=float(x), solver=solver, rtol=1e-7,
                          junction_incidence=junction_incidence, diff_coeff=diff_coeff)
    return (storage_modulus(gfu, mesh, nu, MU, NG),
            diffusional_loss(gfu, mesh, contact_pairs, gni, float(x), diff_coeff))


G0s, Gss, Deltas, rAs = [], [], [], []
for nu in NUS:
    tau_M, info = measure_tau_M(spaces, mesh, contact_pairs, outer_contact_pairs,
                                junction_incidence, num_grains=NG, nu=nu, mu=MU)
    G0 = info["G_U"]
    # steady-state modulus from the low-omega plateau (code-omega == omega*tau_M)
    g1, g2 = G12(nu, X_LOW, tau_M)
    G_ss = g2**2 / g1
    Delta = G0 / G_ss - 1.0
    # Andrade time: global LSQ fit of the full high-omega Andrade form over the
    # resolved power-law window (alpha fixed by the 120 deg triple junction).
    Gc_high = np.array([complex(*G12(nu, xh, tau_M)) for xh in X_HIGH])
    rA = fit_andrade_time(X_HIGH, Gc_high, G0, ALPHA,
                          window=(X_HIGH.min(), X_HIGH.max()))
    G0s.append(G0); Gss.append(G_ss); Deltas.append(Delta); rAs.append(rA)
    print(f"nu={nu:+.2f}  G0/mu={G0:.4f}  G_ss/mu={G_ss:.4f}  Delta={Delta:.4f}  tau_A/tau_M={rA:.4f}")

G0s = np.array(G0s); Gss = np.array(Gss); Deltas = np.array(Deltas); rAs = np.array(rAs)


# --- paper Table-2 rational fits (nodes nu1=-1, nu2=1/2, w1=1) --------------
def rat(nu, f1, f2, w2):
    n = 1.0 * f1 / (nu + 1.0) + w2 * f2 / (nu - 0.5)
    d = 1.0 / (nu + 1.0) + w2 / (nu - 0.5)
    return n / d

nf = np.linspace(-0.99, 0.49, 200)
G0_fit = rat(nf, 0.599973, 0.857133, -0.700034)
Gss_fit = rat(nf, 0.353374, 0.686123, -0.515031)
Delta_fit = G0_fit / Gss_fit - 1.0
rA_fit = rat(nf, 0.68953, 1.319737, -0.301846) ** (1.0 / ALPHA)


# --- plot ------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(6.2, 10), sharex=True)
ax[0].plot(nf, G0_fit, "C0--", lw=1, label="paper fit $G_0/\\mu$")
ax[0].plot(nf, Gss_fit, "C1--", lw=1, label="paper fit $G_{ss}/\\mu$")
ax[0].plot(NUS, G0s, "C0o", label="FEM $G_0/\\mu$")
ax[0].plot(NUS, Gss, "C1s", label="FEM $G_{ss}/\\mu$")
ax[0].set(ylabel="Scaled modulus", ylim=(0, 1.0)); ax[0].legend(fontsize=8)

ax[1].plot(nf, Delta_fit, "C0--", lw=1, label="paper fit")
ax[1].plot(NUS, Deltas, "C0o", label="FEM")
ax[1].set(ylabel=r"Relaxation strength $\Delta$", ylim=(0, 0.8)); ax[1].legend(fontsize=8)

ax[2].plot(nf, rA_fit, "C0--", lw=1, label="paper fit")
ax[2].plot(NUS, rAs, "C0o", label="FEM")
ax[2].set(xlabel=r"Poisson's ratio $\nu$", ylabel=r"$\tau_A/\tau_M$", ylim=(0, 2.5))
ax[2].legend(fontsize=8)

fig.suptitle(f"Rudge (2025) Fig. 1 reproduction — hexagon model (refine_h={REFINE_H})", fontsize=10)
fig.tight_layout()
fig.savefig("reproduce_fig1.png", dpi=140)
print("saved reproduce_fig1.png")

import pandas as pd
pd.DataFrame({"nu": NUS, "G0_over_mu": G0s, "Gss_over_mu": Gss,
              "Delta": Deltas, "tauA_over_tauM": rAs}).to_csv("reproduce_fig1.csv", index=False)
print("saved reproduce_fig1.csv")
