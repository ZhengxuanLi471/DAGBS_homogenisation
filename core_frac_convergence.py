# Convergence of Q^-1 vs omega*tau_M with respect to core_frac.
# refine_h=None throughout.  solver='direct' (needed for correct low-omega tail).
# Usage: python core_frac_convergence.py

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
from ngsolve import SetNumThreads

from meshes import MakeMesh
from main import build_spaces, solve_rve
from calibrate_tau import measure_tau_M, storage_modulus, diffusional_loss, SHEAR

SetNumThreads(8)

NU = 0.3
MU = 1.0

# omega*tau_M axis: 1e-1 to 1e8
X = np.logspace(-1, 8, 91)   # 10 pts/decade

CORE_FRACS = [0.1, 0.05, 0.01, 0.005, 0.001]

# --- hexagon geometry (identical to reproduce_fig2.py / real_im_hex_energy.py) ---
a = np.sqrt(3)
pts0 = [(0,0),(3/4,0),(1/2,a/4),(0,a/4),
        (9/4,0),(5/2,a/4),(2,3*a/4),(1,3*a/4),
        (3,0),(3,a/4),(3,a),(9/4,a),(3/4,a),(0,a)]
pts1 = [(x-1.5, y-0.5*a) for x,y in pts0]
sf = np.sqrt(1/50 / (3*np.sqrt(3)/2))
pts = [(x*sf, y*sf) for x,y in pts1]
regions = [(1,2,3,4),(2,5,6,7,8,3),(5,9,10,6),
           (6,10,11,12,7),(8,7,12,13),(4,3,8,13,14)]
NG = len(regions)

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(CORE_FRACS)))

fig, ax = plt.subplots(figsize=(7, 5))

for cf, color in zip(CORE_FRACS, colors):
    print(f"\n=== core_frac = {cf} ===")

    (_, _, mesh, _, contact_pairs, outer_contact_pairs,
     _cbl, _ocl, junction_incidence) = MakeMesh(
        pts, regions,
        maxh=0.1*sf, comm=MPI.COMM_WORLD,
        core_frac=cf*sf,
        refine_h=None)
    print(f"  mesh: ne={mesh.ne}  nv={mesh.nv}")

    spaces = build_spaces(mesh, contact_pairs, outer_contact_pairs,
                          order_bulk=2, order_gb=1,
                          junction_incidence=junction_incidence)
    gni = spaces[4]

    tau_M, info = measure_tau_M(spaces, mesh, contact_pairs, outer_contact_pairs,
                                junction_incidence, num_grains=NG, nu=NU, mu=MU)
    G0 = info["G_U"]
    print(f"  tau_M={tau_M:.6e}  G0={G0:.6f}  eta_ss={info['eta_ss']:.6e}")

    G1 = np.empty_like(X)
    G2 = np.empty_like(X)
    for i, x in enumerate(X):
        gfu, _ = solve_rve(spaces, mesh, contact_pairs, outer_contact_pairs,
                           SHEAR, nu=NU, mu=MU, omega=float(x),
                           solver='cg', rtol=1e-7,
                           junction_incidence=junction_incidence,
                           diff_coeff=tau_M)
        G1[i] = storage_modulus(gfu, mesh, NU, MU, NG)
        G2[i] = diffusional_loss(gfu, mesh, contact_pairs, gni, float(x), diff_coeff=tau_M)
        print(f"    omega*tauM={x:.3e}  G1={G1[i]:.6f}  G2={G2[i]:.6e}  Q-1={G2[i]/G1[i]:.6e}")

    Qinv = G2 / G1
    np.savetxt(f"core_frac_conv_{cf}.csv",
               np.column_stack([X, G1, G2, Qinv]),
               header="omega_tauM,G1,G2,Qinv", delimiter=",", comments="")
    print(f"  saved core_frac_conv_{cf}.csv")

    ax.loglog(X, Qinv, color=color, lw=1.5, label=f"cf={cf}")

# reference slope -1 line
xref = np.array([1e-1, 1e1])
ax.loglog(xref, 0.25 / xref, 'k--', lw=0.8, label=r"$\propto (\omega\tau_M)^{-1}$")

ax.set_xlabel(r"$\omega \tau_M$")
ax.set_ylabel(r"$Q^{-1}$")
ax.set_title(r"$Q^{-1}$ convergence vs core\_frac  ($\nu=0.3$, refine\_h=None, direct solver)")
ax.legend(fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.4)
fig.tight_layout()
fig.savefig("core_frac_convergence.png", dpi=150)
print("\nSaved core_frac_convergence.png")
