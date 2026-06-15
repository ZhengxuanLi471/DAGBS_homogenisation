# Generate the nu=0.35 hexagon reference (shear only) on the SAME (extended) omega
# grid as the seed sweep, for overlaying on the fig2-style ensemble plots. Trimmed
# from real_im_hex_energy.py: one branch, same per-geometry tau_M calibration.
# Writes hex_reference_shear.csv and prints the calibrated hex tau_M (the normalizer
# for the "hex tau_M" plot set).
#
# Grid matches the fully-extended seeds: the base linspace(-3,10,100) PLUS the
# low-omega tail ln(omega*tau_M) in [-6,-3) (23 pts, solver='direct' -- CG floors
# the tiny C'~omega^2) and the high-omega tail in (10,15] (39 pts, solver='cg' --
# well-conditioned plateau), all on the same step h=13/99. Total 162 points on
# ln(omega*tau_M) in [-6.02, 15.12]. Solver is chosen per point: direct below -3,
# CG at/above -3.

from main import solve_rve, build_spaces
from meshes import MakeMesh
from physics import *
from calibrate_tau import measure_tau_M
from ngsolve import *
import pandas as pd
import numpy as np
from mpi4py import MPI

SetNumThreads(8)

NU = 0.35                      # match the seed sweep (real_im_energy.py)
MU = 1.0
MACRO_SCALE = 1e-3
LN_OMEGA_MIN = -3.0            # match seed base grid
LN_OMEGA_MAX = 10.0
OMEGA_SAMPLES = 100

# Composite extended grid (same construction as the low/high-omega tail drivers).
_H = 13.0 / 99.0                                              # base step
_LN_BASE = np.linspace(LN_OMEGA_MIN, LN_OMEGA_MAX, OMEGA_SAMPLES)
_LN_LOW = (-3.0 - _H * np.arange(1, int(np.ceil(3.0 / _H)) + 1))[::-1]   # [-6,-3)
_LN_HIGH = 10.0 + _H * np.arange(1, int(np.ceil(5.0 / _H)) + 1)          # (10,15]
LN_OMEGA_FULL = np.concatenate([_LN_LOW, _LN_BASE, _LN_HIGH])            # 162 pts


def compute_energy_metrics(gfu, mesh, contact_pairs, gb_normal_indices, omega,
                           area, diff_coeff, num_grains=6):
    """Bulk storage + GB diffusional dissipation for one solution (shear)."""
    uR = gfu.components[0]
    uI = gfu.components[1]
    lam = 2 * MU * NU / (1 - 2 * NU)
    epsR, epsI = strain(uR), strain(uI)
    sigR, sigI = stress(uR, lam, MU), stress(uI, lam, MU)
    storage_density = InnerProduct(sigR, epsR) + InnerProduct(sigI, epsI)

    bulk_storage_total = 0.0
    for grain_id in range(1, num_grains + 1):
        grain_region = mesh.Materials(f"region_{grain_id}")
        bulk_storage_total += float(
            Integrate(storage_density, mesh, VOL, definedon=grain_region))
    bulk_storage = 0.5 * bulk_storage_total / area

    gb_energy = 0.0
    for (a, b), (_, right_name) in contact_pairs.items():
        region = mesh.Boundaries(f"core_{right_name}.*|slide_{right_name}")
        idx_re, idx_im = gb_normal_indices[(a, b)]
        t_n_re = gfu.components[idx_re]
        t_n_im = gfu.components[idx_im]
        gtr, gti = Grad(t_n_re).Trace(), Grad(t_n_im).Trace()
        flux_sq = InnerProduct(gtr, gtr) + InnerProduct(gti, gti)
        gb_energy += float(Integrate(flux_sq, mesh, BND, definedon=region))
    gb_diss = 0.5 * diff_coeff * gb_energy / area / omega
    return bulk_storage, gb_diss


# --- hexagon geometry (identical to real_im_hex_energy.py) ------------------
a = np.sqrt(3)
pts0 = [(0, 0), (3/4, 0), (1/2, a/4), (0, a/4),
        (9/4, 0), (5/2, a/4), (2, 3*a/4), (1, 3*a/4),
        (3, 0), (3, a/4), (3, a), (9/4, a), (3/4, a), (0, a)]
pts1 = [(x - 1.5, y - 0.5 * a) for (x, y) in pts0]
area_per_grain = 3 * np.sqrt(3) / 2
scale_factor = np.sqrt(1 / 50 / area_per_grain)
pts = [(x * scale_factor, y * scale_factor) for (x, y) in pts1]
regions = [(1, 2, 3, 4), (2, 5, 6, 7, 8, 3), (5, 9, 10, 6),
           (6, 10, 11, 12, 7), (8, 7, 12, 13), (4, 3, 8, 13, 14)]

(_, _, mesh, _, contact_pairs, outer_contact_pairs,
 corner_bnd_label, _ocl, junction_incidence) = MakeMesh(
    pts, regions, maxh=0.1 * scale_factor, comm=MPI.COMM_WORLD,
    core_frac=0.01 * scale_factor, refine_h=None)

spaces = build_spaces(mesh, contact_pairs, outer_contact_pairs,
                      order_bulk=2, order_gb=1,
                      junction_incidence=junction_incidence)
gb_normal_indices = spaces[4]

tau_M, info = measure_tau_M(spaces, mesh, contact_pairs, outer_contact_pairs,
                            junction_incidence, num_grains=len(regions),
                            nu=NU, mu=MU)
print(f"HEX tau_M = {tau_M:.6e}  (eta_ss={info['eta_ss']:.4e}, "
      f"G_U={info['G_U']:.4e})", flush=True)

# --- shear sweep on the full extended grid ---------------------------------
Gamma = ((0, 1), (0, 0))
ln_omega = LN_OMEGA_FULL
storage_vals, diss_vals = [], []
for lw in ln_omega:
    omegai = float(np.exp(lw))
    solver = 'direct' if lw < -3.0 else 'cg'   # low-omega tail needs Pardiso
    print(f"hex omega: {omegai} ({solver})", flush=True)
    gfu, mesh = solve_rve(spaces, mesh, contact_pairs, outer_contact_pairs,
                          Gamma, nu=NU, mu=MU, omega=omegai,
                          solver=solver, rtol=1e-8, corner_bnd=None,
                          junction_incidence=junction_incidence,
                          diff_coeff=tau_M)
    area = float(Integrate(1, mesh, VOL))
    storage, diss = compute_energy_metrics(
        gfu, mesh, contact_pairs, gb_normal_indices, omegai, area, tau_M)
    storage_vals.append(storage)
    diss_vals.append(diss)

modulus_scale = 2.0 / (MACRO_SCALE ** 2)
df = pd.DataFrame({
    'ln_omega': ln_omega,
    'omega': np.exp(ln_omega),
    'E_storage': storage_vals,
    'E_diss_total': diss_vals,
    'Cxyxy_real': modulus_scale * np.array(storage_vals),
    'Cxyxy_imag': modulus_scale * np.array(diss_vals),
})
df.to_csv('hex_reference_shear.csv', index=False)
# stash tau_M alongside for the plotter
with open('hex_reference_tau.txt', 'w') as fh:
    fh.write(f"{tau_M:.10e}\n")
print("Saved hex_reference_shear.csv and hex_reference_tau.txt", flush=True)
