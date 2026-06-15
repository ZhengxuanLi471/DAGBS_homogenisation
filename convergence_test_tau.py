# Convergence test: confirm the Maxwell-time calibration plateau for sigma=0.5.
#
# For each of 10 seeds we compute tau_M(omega) = [C''(omega)/omega] / G_U across
# ln(omega) in [-3, 3] with diff_coeff=1. If the calibration point omega=1
# (ln omega=0) is deep in the creep plateau, tau_M(omega) should be flat across
# the whole range. Run with:
#   /home/zl471/miniconda3/envs/ngsolve/bin/python convergence_test_tau.py

from main import solve_rve, build_spaces
from meshes import MakeMesh
from calibrate_tau import storage_modulus, diffusional_loss, SHEAR
from ngsolve import SetNumThreads
from mpi4py import MPI
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SetNumThreads(8)

# ── settings ────────────────────────────────────────────────────────────────
TESS_JSON = "sigmas/sigma_0.5/tessellation_output.json"
SEED_KEYS = [f"seeds_{1 + 10*i}" for i in range(10)]   # seeds_1,11,21,...,91
LN_OMEGAS = np.linspace(-3, 3, 7)
NU, MU = 0.35, 1.0
OUT_PNG   = "convergence_test_tau.png"
# ────────────────────────────────────────────────────────────────────────────

with open(TESS_JSON) as f:
    data = json.load(f)

results = {}   # seed_key -> array of tau_M(omega)

for seed_key in SEED_KEYS:
    print(f"\n{'='*70}")
    print(f"Seed: {seed_key}")
    print(f"{'='*70}", flush=True)

    pts, regions = data[seed_key]
    num_grains = len(regions)

    (_, _, mesh, _,
     contact_pairs, outer_contact_pairs,
     corner_bnd, _outer_core,
     junction_incidence) = MakeMesh(
        pts, regions,
        maxh=0.1,
        comm=MPI.COMM_WORLD,
        core_frac=0.01,
    )

    spaces = build_spaces(
        mesh, contact_pairs, outer_contact_pairs,
        order_bulk=2, order_gb=1,
        junction_incidence=junction_incidence,
    )
    gb_normal_indices = spaces[4]

    # G_U: normal-locked limit (omega-independent)
    print(f"[{seed_key}] solving G_U (diff_coeff=0) ...", flush=True)
    gfu, mesh = solve_rve(
        spaces, mesh, contact_pairs, outer_contact_pairs,
        SHEAR, nu=NU, mu=MU, omega=1.0,
        solver='cg', rtol=1e-8,
        junction_incidence=junction_incidence,
        diff_coeff=0.0,
    )
    G_U = storage_modulus(gfu, mesh, NU, MU, num_grains)
    print(f"[{seed_key}] G_U = {G_U:.6e}", flush=True)

    # Sweep: measure eta_ss = C''/omega at each ln(omega)
    tau_M_arr = []
    for ln_w in LN_OMEGAS:
        omega = float(np.exp(ln_w))
        print(f"[{seed_key}] ln(omega)={ln_w:+.1f}  omega={omega:.4f} ...",
              end=" ", flush=True)
        gfu, mesh = solve_rve(
            spaces, mesh, contact_pairs, outer_contact_pairs,
            SHEAR, nu=NU, mu=MU, omega=omega,
            solver='cg', rtol=1e-8,
            junction_incidence=junction_incidence,
            diff_coeff=1.0,
        )
        Cpp = diffusional_loss(gfu, mesh, contact_pairs, gb_normal_indices,
                               omega, diff_coeff=1.0)
        eta_ss = Cpp / omega
        tau_M_w = eta_ss / G_U
        print(f"tau_M = {tau_M_w:.4e}", flush=True)
        tau_M_arr.append(tau_M_w)

    results[seed_key] = np.array(tau_M_arr)
    print(f"[{seed_key}] done. tau_M range: "
          f"[{results[seed_key].min():.4e}, {results[seed_key].max():.4e}]",
          flush=True)

# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = plt.cm.tab10(np.linspace(0, 0.9, len(SEED_KEYS)))

for (seed_key, tau_arr), col in zip(results.items(), colors):
    ax.plot(LN_OMEGAS, tau_arr, color=col, lw=1.5,
            label=seed_key.replace("seeds_", "seed "))

ax.axvline(0, color='k', ls='--', lw=1.0, label=r"calibration point ($\omega=1$)")

ax.set_xlabel(r"$\ln\,\omega$")
ax.set_ylabel(r"$\tau_M(\omega) = \eta_{ss}(\omega)\,/\,G_U$")
ax.set_yscale("log")
ax.set_title(r"Maxwell-time plateau — $\sigma=0.5$, 10 seeds")
ax.legend(fontsize=7, ncol=2, loc="lower left")
ax.set_xlim(-3, 3)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"\nSaved {OUT_PNG}")
