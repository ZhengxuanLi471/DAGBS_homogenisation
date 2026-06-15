# DAGBS LOW-OMEGA TAIL driver. Companion to real_im_energy.py: it sweeps only the
# extra decades ln(omega*tau_M) in [-6, -3) so the existing [-3, 10] CSVs can be
# extended down into the creep plateau WITHOUT recomputing them. Output goes to a
# separate "_lowomega" CSV; concatenate later (drop nothing — the grids don't
# overlap; the last low point + step == -3, the first existing point).
#
# Deliberate difference from real_im_energy.py:
#   Grid: 23 points on the SAME spacing as linspace(-3,10,100) (h=13/99),
#   running from -6.0202 up to -3.1313 (strictly below -3, so no duplicate).
# NOTE on the solver: this driver now uses solver='cg' (standing directive: CG
# everywhere, never direct). At low omega the storage modulus C' ~ omega^2 is
# tiny and CG re-floors it at the residual level (CLAUDE.md), so G_ss = G2^2/G1
# and hence the relaxation strength Delta from this tail are APPROXIMATE. This is
# accepted; the historical direct (Pardiso) tail was more accurate but too slow.
# Shear branch only (matches the per-seed *_shear.csv we analyse).


from main import solve_rve, build_spaces
from meshes import MakeMesh
from physics import *
from calibrate_tau import measure_tau_M
from ngsolve import *
import os
import pandas as pd
import numpy as np
from mpi4py import MPI
import json
import argparse

SetNumThreads(8)


parser = argparse.ArgumentParser(
    description="Low-omega tail (ln omega*tau_M in [-6,-3)) for a range of seeds"
)
parser.add_argument("seed_start", type=int, nargs="?",
                    help="First seed id to process (inclusive)")
parser.add_argument("seed_end", type=int, nargs="?",
                    help="Upper bound (exclusive) for seeds")
parser.add_argument("--refine_h", type=float, default=None,
                    help="optional target element size at triple junctions/corners")
args = parser.parse_args()

NU = 0.35
MU = 1.0
MACRO_SCALE = 1e-3

# Low-omega grid on the SAME spacing as the original linspace(-3, 10, 100).
_H = 13.0 / 99.0                                  # original step
_N_LOW = int(np.ceil(3.0 / _H))                  # 23 points to reach -6
LN_OMEGA_LOW = (-3.0 - _H * np.arange(1, _N_LOW + 1))[::-1]   # ascending, < -3


def compute_energy_metrics(gfu, mesh, contact_pairs, gb_normal_indices, omega, area, diff_coeff, num_grains):
    """Return bulk storage and GB diffusional dissipation for one solution."""
    uR = gfu.components[0]
    uI = gfu.components[1]

    lam = 2 * MU * NU / (1 - 2 * NU)

    epsR = strain(uR)
    epsI = strain(uI)
    sigR = stress(uR, lam, MU)
    sigI = stress(uI, lam, MU)

    storage_density = InnerProduct(sigR, epsR) + InnerProduct(sigI, epsI)

    bulk_storage_total = 0.0
    for grain_id in range(1, num_grains + 1):
        grain_region = mesh.Materials(f"region_{grain_id}")
        grain_storage = float(Integrate(storage_density, mesh, VOL, definedon=grain_region))
        bulk_storage_total += grain_storage

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
    total_diss = gb_diss

    return bulk_storage, total_diss


def run_branch(Gamma, gamma_tag, mesh, spaces, contact_pairs, outer_contact_pairs,
               corner_penalty_label, gb_normal_indices, num_grains, seedname,
               diff_coeff, junction_incidence=None):
    """Sweep the low-omega grid for one macro loading tensor and dump energy curves."""

    print("Starting with single Gamma:", Gamma)
    ln_omega = LN_OMEGA_LOW

    storage_vals = []
    diss_total_vals = []

    for j in range(len(ln_omega)):
        omegai = np.exp(ln_omega[j])
        print("Current omega: ", omegai)
        gfu, mesh = solve_rve(
            spaces,
            mesh,
            contact_pairs,
            outer_contact_pairs,
            Gamma,
            nu=NU,
            mu=MU,
            omega=omegai,
            solver='cg',              # CG default everywhere (standing directive); low-omega C'/Delta is approximate
            rtol=1e-8,
            corner_bnd=corner_penalty_label,
            junction_incidence=junction_incidence,
            diff_coeff=diff_coeff,
        )

        area = float(Integrate(1, mesh, VOL))
        storage, total_diss = compute_energy_metrics(
            gfu, mesh, contact_pairs, gb_normal_indices, omegai, area,
            diff_coeff, num_grains)
        storage_vals.append(storage)
        diss_total_vals.append(total_diss)

    omega_vals = np.exp(ln_omega)
    modulus_scale = 2.0 / (MACRO_SCALE ** 2)
    comp_name = "Cxyxy" if gamma_tag == "shear" else (
        "Cxxxx" if gamma_tag == "normal_x" else "Cyyyy")

    df = pd.DataFrame({
        'ln_omega': ln_omega,
        'omega': omega_vals,
        'E_storage': storage_vals,
        'E_diss_total': diss_total_vals,
        f'{comp_name}_real': modulus_scale * np.array(storage_vals),
        f'{comp_name}_imag': modulus_scale * np.array(diss_total_vals),
    })
    out_path = 'Seed_{}_energy_real_im_data_{}_lowomega.csv'.format(seedname, gamma_tag)
    df.to_csv(out_path, index=False)
    print(f"Saved low-omega energy curves to {out_path}")

    return mesh


with open("tessellation_output.json", "r") as f:
    data = json.load(f)

seed_keys = [key for key in data if key.startswith("seeds_")]
if seed_keys:
    seeds = sorted(int(key.split("_")[1]) for key in seed_keys)
else:
    seeds = list(range(1, len(data) + 1))

if args.seed_start is not None:
    if args.seed_end is None:
        parser.error("seed_end is required when specifying seed_start")
    if args.seed_start >= args.seed_end:
        parser.error("seed_start must be less than seed_end")
    seeds = list(range(args.seed_start, args.seed_end))

print(seeds)
for seed in seeds:
    seedname = "seeds_{}".format(seed)
    print("Processing ", seedname)
    pts, regions = data[seedname]
    num_grains = len(regions)

    try:
        (
            _, _, mesh, _,
            contact_pairs,
            outer_contact_pairs,
            corner_bnd_label,
            _outer_core_labels,
            junction_incidence,
        ) = MakeMesh(
            pts, regions, maxh=0.1, comm=MPI.COMM_WORLD,
            core_frac=0.01, refine_h=args.refine_h,
        )
    except Exception as e:
        print(f"MakeMesh failed for {seedname}: {e}")
        continue

    penalty_boundaries = []
    if corner_bnd_label:
        if isinstance(corner_bnd_label, (list, tuple, set)):
            penalty_boundaries.extend(name for name in corner_bnd_label if name)
        else:
            penalty_boundaries.append(corner_bnd_label)
    penalty_boundaries = list(dict.fromkeys(penalty_boundaries))
    corner_penalty_label = "|".join(penalty_boundaries) if penalty_boundaries else None

    spaces = build_spaces(
        mesh, contact_pairs, outer_contact_pairs,
        order_bulk=2, order_gb=1, junction_incidence=junction_incidence,
    )
    gb_normal_indices = spaces[4]

    # Same per-seed tau_M calibration as real_im_energy.py, so the low-omega axis
    # reads omega*tau_M on the identical normalization as the existing CSV.
    tau_M, tau_info = measure_tau_M(
        spaces, mesh, contact_pairs, outer_contact_pairs,
        junction_incidence, num_grains=num_grains, nu=NU, mu=MU)
    print(f"{seedname}: calibrated tau_M = {tau_M:.6e} "
          f"(eta_ss={tau_info['eta_ss']:.4e}, G_U={tau_info['G_U']:.4e})")

    mesh = run_branch(((0, 1), (0, 0)), "shear", mesh, spaces, contact_pairs,
                      outer_contact_pairs, corner_penalty_label, gb_normal_indices,
                      num_grains, seedname, tau_M, junction_incidence=junction_incidence)
