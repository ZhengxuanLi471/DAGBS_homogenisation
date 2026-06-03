# DAGBS hexagonal benchmark: regular 6-grain RVE swept over frequency. The
# reference shape used to cross-check the diffusion-accommodated GB sliding model.


from main import solve_rve,build_spaces
from meshes import MakeMesh
from physics import *
from calibrate_tau import measure_tau_M
from ngsolve import *
import pandas as pd
import numpy as np
from mpi4py import MPI
import sys

SetNumThreads(8)

NU = 0.35
MU = 1.0
# The GB diffusion coefficient C_d is the per-geometry Maxwell time tau_M,
# measured at runtime by measure_tau_M (see below). It is BOTH the diff_coeff
# passed to solve_rve AND the output-loss prefactor, so code-omega == omega*tau_M
# and the fluid<->solid crossover sits at omega = 1.
MACRO_SCALE = 1e-3  # Gamma scaling applied in _setup_material_properties
LN_OMEGA_MIN = -10.0
LN_OMEGA_MAX = 10.0
OMEGA_SAMPLES = 100


def compute_energy_metrics(gfu, mesh, contact_pairs, gb_normal_indices, omega, area, diff_coeff, num_grains=6):
    """Return bulk storage plus GB diffusional dissipation for one solution."""
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

    gb_diss = 0.5 * diff_coeff * gb_energy / area/omega
    total_diss = gb_diss

    return bulk_storage, total_diss


def run_branch(Gamma, gamma_tag, diff_coeff):
    """Sweep omega for one macro loading tensor and dump energy curves.

    diff_coeff is the geometry's Maxwell time tau_M (folded into the form and
    the loss prefactor), so the omega axis reads omega*tau_M.
    """
    global mesh

    print("Starting with single Gamma:", Gamma)
    ln_omega = np.linspace(LN_OMEGA_MIN, LN_OMEGA_MAX, OMEGA_SAMPLES)

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
            solver='cg',
            rtol=1e-8,
            corner_bnd=corner_penalty_label,
            junction_incidence=junction_incidence,
            diff_coeff=diff_coeff,
        )

        area = float(Integrate(1, mesh, VOL))
        storage, total_diss = compute_energy_metrics(
            gfu,
            mesh,
            contact_pairs,
            gb_normal_indices,
            omegai,
            area,
            diff_coeff,
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
    out_path = '{}_Seed_{}_modulus_data_{}.csv'.format(core_frac_tag, seedname, gamma_tag)
    df.to_csv(out_path, index=False)
    print(f"Saved energy curves to {out_path}")
 
a = np.sqrt(3)
pts0 = [
    (0, 0), (3/4, 0), (1/2, a/4), (0, a/4),
    (9/4, 0), (5/2, a/4), (2, 3*a/4), (1, 3*a/4),
    (3, 0), (3, a/4), (3, a), (9/4, a),
    (3/4, a), (0, a)
]
# Shift to center domain around origin
pts1 = [(x - 1.5, y - 0.5 * a) for (x, y) in pts0]
#scale to have average grain area of 1/50
area_per_grain = 3*np.sqrt(3)/2
scale_factor = np.sqrt(1/50 / area_per_grain)
pts = [(x * scale_factor, y * scale_factor) for (x, y) in pts1]

# Define polygonal regions (each grain) by vertex indices
regions = [
    (1, 2, 3, 4),
    (2, 5, 6, 7, 8, 3),
    (5, 9, 10, 6),
    (6, 10, 11, 12, 7),
    (8, 7, 12, 13),
    (4, 3, 8, 13, 14)
]

Gamma = ((0,1), (0, 0))
seedname = 'hex'

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
    maxh=0.1*scale_factor,
    comm=MPI.COMM_WORLD,
     core_frac=(float(sys.argv[1]) if len(sys.argv) > 1 else 0.01) * scale_factor,
     refine_h=(float(sys.argv[2]) if len(sys.argv) > 2 else None),
)

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

core_frac_tag = (sys.argv[1] if len(sys.argv) > 1 else "0.01")

# Calibrate the Maxwell time tau_M = eta_ss / G_U for THIS geometry (2 solves),
# then fold it into every branch so the omega axis reads omega*tau_M.
tau_M, tau_info = measure_tau_M(
    spaces, mesh, contact_pairs, outer_contact_pairs,
    junction_incidence, num_grains=len(regions), nu=NU, mu=MU)
print(f"Calibrated tau_M = {tau_M:.6e} "
      f"(eta_ss={tau_info['eta_ss']:.4e}, G_U={tau_info['G_U']:.4e})")

run_branch(((0,1), (0, 0)), "shear", tau_M)
run_branch(((1,0), (0, 0)), "normal_x", tau_M)
run_branch(((0,0), (0, 1)), "normal_y", tau_M)
