# DAGBS frequency sweep driver: computes complex shear and normal stiffness for
# each tessellation seed and saves per-seed CSVs. Dissipation comes from the
# grain-boundary diffusion channel (∫|∂ₛt_n|²).


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

SetNumThreads(32)


parser = argparse.ArgumentParser(
    description="Compute real/imag stiffness sweep for a range of seeds (energy-based)"
)
parser.add_argument(
    "seed_start",
    type=int,
    nargs="?",
    help="First seed id to process (inclusive)"
)
parser.add_argument(
    "seed_end",
    type=int,
    nargs="?",
    help="Upper bound (exclusive) for seeds"
)
args = parser.parse_args()

NU = 0.35
MU = 1.0
# The GB diffusion coefficient C_d is the per-geometry Maxwell time tau_M,
# measured at runtime by measure_tau_M (one call per seed). It is BOTH the
# diff_coeff passed to solve_rve AND the output-loss prefactor, so the omega
# axis reads omega*tau_M and the crossover sits at omega = 1.
MACRO_SCALE = 1e-3  # Gamma scaling applied in _setup_material_properties
LN_OMEGA_MIN = -3.0
LN_OMEGA_MAX = 10.0
OMEGA_SAMPLES = 100


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

    # Integrate grain by grain and sum
    bulk_storage_total = 0.0
    for grain_id in range(1, num_grains + 1):
        grain_region = mesh.Materials(f"region_{grain_id}")
        grain_storage = float(Integrate(storage_density, mesh, VOL, definedon=grain_region))
        bulk_storage_total += grain_storage

    bulk_storage = 0.5 * bulk_storage_total / area

    # GB diffusional dissipation ∝ ∫|∂ₛt_n|² over the whole boundary (core∪slide)
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
    """Sweep omega for one macro loading tensor and dump energy curves.

    diff_coeff is this seed's Maxwell time tau_M (folded into the form and the
    loss prefactor), so the omega axis reads omega*tau_M.
    """

    print("Starting with single Gamma:", Gamma)
    ln_omega = np.linspace(LN_OMEGA_MIN, LN_OMEGA_MAX, OMEGA_SAMPLES)

    storage_vals = []
    diss_total_vals = []

    for j in range(len(ln_omega)):
        omegai = np.exp(ln_omega[j])
        print("Current omega: ", omegai)
        gfu, mesh, convergence = solve_rve(
            spaces,
            mesh,
            contact_pairs,
            outer_contact_pairs,
            Gamma,
            nu=NU,
            mu=MU,
            omega=omegai,
            solver='cg',
            rtol=1e-8,
            corner_bnd=corner_penalty_label,
            junction_incidence=junction_incidence,
            diff_coeff=diff_coeff,
        )
        print(convergence)
        while not convergence:
            ln_omega[j] += 0.01
            omegai = np.exp(ln_omega[j])
            print("Adjusting omega to ", omegai)
            gfu, mesh, convergence = solve_rve(
                spaces,
                mesh,
                contact_pairs,
                outer_contact_pairs,
                Gamma,
                nu=NU,
                mu=MU,
                omega=omegai,
                solver='cg',
                rtol=1e-8,
                corner_bnd=corner_penalty_label,
                junction_incidence=junction_incidence,
                diff_coeff=diff_coeff,
            )
            print(convergence)

        area = float(Integrate(1, mesh, VOL))
        storage, total_diss = compute_energy_metrics(
            gfu,
            mesh,
            contact_pairs,
            gb_normal_indices,
            omegai,
            area,
            diff_coeff,
            num_grains,
        )
        storage_vals.append(storage)
        diss_total_vals.append(total_diss)

    omega_vals = np.exp(ln_omega)
    modulus_scale = 2.0 / (MACRO_SCALE ** 2)
    if gamma_tag == "shear":
        comp_name = "Cxyxy"
    elif gamma_tag == "normal_x":
        comp_name = "Cxxxx"
    else:
        comp_name = "Cyyyy"

    df = pd.DataFrame({
        'ln_omega': ln_omega,
        'omega': omega_vals,
        'E_storage': storage_vals,
        'E_diss_total': diss_total_vals,
        f'{comp_name}_real': modulus_scale * np.array(storage_vals),
        f'{comp_name}_imag': modulus_scale * np.array(diss_total_vals),
    })
    out_path = 'Seed_{}_energy_real_im_data_{}.csv'.format(seedname, gamma_tag)
    df.to_csv(out_path, index=False)
    print(f"Saved energy curves to {out_path}")

    return mesh  # Return potentially updated mesh


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
            pts,
            regions,
            maxh=0.1,
            comm=MPI.COMM_WORLD,
            core_frac=0.01,
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
        mesh,
        contact_pairs,
        outer_contact_pairs,
        order_bulk=2,
        order_gb=1,
        junction_incidence=junction_incidence,
    )
    gb_normal_indices = spaces[4]

    # Calibrate this seed's Maxwell time tau_M = eta_ss / G_U (2 solves), then
    # fold it into every branch so the omega axis reads omega*tau_M.
    tau_M, tau_info = measure_tau_M(
        spaces, mesh, contact_pairs, outer_contact_pairs,
        junction_incidence, num_grains=num_grains, nu=NU, mu=MU)
    print(f"{seedname}: calibrated tau_M = {tau_M:.6e} "
          f"(eta_ss={tau_info['eta_ss']:.4e}, G_U={tau_info['G_U']:.4e})")

    mesh = run_branch(((0, 1), (0, 0)), "shear", mesh, spaces, contact_pairs,
                      outer_contact_pairs, corner_penalty_label, gb_normal_indices,
                      num_grains, seedname, tau_M, junction_incidence=junction_incidence)
    mesh = run_branch(((1, 0), (0, 0)), "normal_x", mesh, spaces, contact_pairs,
                      outer_contact_pairs, corner_penalty_label, gb_normal_indices,
                      num_grains, seedname, tau_M, junction_incidence=junction_incidence)
    mesh = run_branch(((0, 0), (0, 1)), "normal_y", mesh, spaces, contact_pairs,
                      outer_contact_pairs, corner_penalty_label, gb_normal_indices,
                      num_grains, seedname, tau_M, junction_incidence=junction_incidence)
