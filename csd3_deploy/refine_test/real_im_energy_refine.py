# MESH-REFINEMENT CONVERGENCE TEST for the high-omega (sliver) Q^-1 peak.
#
# Question: is the high-frequency peak the genuine DAGBS relaxation of a short
# grain-boundary sliver (then it should PERSIST and climb toward its true L^-3
# frequency as we resolve the sliver), or an under-resolution/degenerate-junction
# artifact (then it should weaken/move when properly resolved)?
#
# This driver re-sweeps ONE seed's high-omega band on a mesh whose SHORT facets
# are refined PROPORTIONALLY to their own length (element size = L*refine_frac on
# facets with L < refine_cut), via the local edited `meshes_refine.MakeMesh`. The
# junction-coupling physics (core_frac) is left UNCHANGED -- only the mesh size,
# so the test isolates resolution. solver='cg' throughout (user directive: never
# direct). Shear branch only. CSV is written after every frequency so a walltime
# kill still leaves a usable partial curve.
#
# Solver modules (main/physics/calibrate_tau) are imported unmodified from a
# sigma dir; only MakeMesh comes from the edited copy here. Tessellation is read
# by path for the requested seed/sigma.
#
#   python real_im_energy_refine.py --seed 24 --sigma 0.45 --den 100
#   python real_im_energy_refine.py --seed 24 --sigma 0.45 --den 0   # baseline (no proportional refine)

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SIGMA_MODULES = os.path.normpath(os.path.join(HERE, "..", "sigmas", "sigma_0.45"))
sys.path.insert(0, SIGMA_MODULES)        # main, physics, calibrate_tau (unmodified, identical across sigmas)
sys.path.insert(0, HERE)                 # meshes_refine FIRST so our MakeMesh wins

from meshes_refine import MakeMesh       # noqa: E402  (edited copy)
from main import solve_rve, build_spaces  # noqa: E402
from physics import *                    # noqa: E402
from calibrate_tau import measure_tau_M  # noqa: E402
from ngsolve import *                    # noqa: E402
from mpi4py import MPI                    # noqa: E402

SetNumThreads(8)

NU = 0.35
MU = 1.0
MACRO_SCALE = 1e-3
GAMMA = ((0, 1), (0, 0))                  # shear


def compute_energy_metrics(gfu, mesh, contact_pairs, gb_normal_indices, omega,
                           area, diff_coeff, num_grains):
    uR, uI = gfu.components[0], gfu.components[1]
    lam = 2 * MU * NU / (1 - 2 * NU)
    storage_density = (InnerProduct(stress(uR, lam, MU), strain(uR)) +
                       InnerProduct(stress(uI, lam, MU), strain(uI)))
    bulk = 0.0
    for g in range(1, num_grains + 1):
        bulk += float(Integrate(storage_density, mesh, VOL,
                                definedon=mesh.Materials(f"region_{g}")))
    bulk_storage = 0.5 * bulk / area
    gb_energy = 0.0
    for (a, b), (_, right_name) in contact_pairs.items():
        region = mesh.Boundaries(f"core_{right_name}.*|slide_{right_name}")
        idx_re, idx_im = gb_normal_indices[(a, b)]
        gtr = Grad(gfu.components[idx_re]).Trace()
        gti = Grad(gfu.components[idx_im]).Trace()
        flux_sq = InnerProduct(gtr, gtr) + InnerProduct(gti, gti)
        gb_energy += float(Integrate(flux_sq, mesh, BND, definedon=region))
    gb_diss = 0.5 * diff_coeff * gb_energy / area / omega
    return bulk_storage, gb_diss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sigma", type=str, required=True, help="sigma dir tag, e.g. 0.45")
    ap.add_argument("--den", type=float, required=True,
                    help="refine denominator: element size = L/den on short facets; <=0 => baseline (no proportional refine)")
    ap.add_argument("--refine_cut", type=float, default=0.02,
                    help="only facets shorter than this are proportionally refined")
    ap.add_argument("--maxh", type=float, default=0.1)
    ap.add_argument("--core_frac", type=float, default=0.01)
    ap.add_argument("--lnmin", type=float, default=4.0)
    ap.add_argument("--lnmax", type=float, default=18.0)
    ap.add_argument("--npts", type=int, default=85)
    args = ap.parse_args()

    refine_frac = None if args.den <= 0 else 1.0 / args.den
    tag = "base" if refine_frac is None else f"{int(round(args.den))}"
    tess_path = os.path.normpath(os.path.join(
        HERE, "..", "sigmas", f"sigma_{args.sigma}", "tessellation_output.json"))
    seedname = f"seeds_{args.seed}"
    out_path = os.path.join(
        HERE, f"refine_seed{args.seed}_sigma{args.sigma}_frac{tag}_shear.csv")

    print(f"=== seed {args.seed} (sigma {args.sigma}), refine_frac={refine_frac} "
          f"(L/{tag}), cut={args.refine_cut} ===", flush=True)
    with open(tess_path) as f:
        pts, regions = json.load(f)[seedname]
    num_grains = len(regions)

    (_, _, mesh, _, contact_pairs, outer_contact_pairs, corner_bnd_label,
     _ocl, junction_incidence) = MakeMesh(
        pts, regions, maxh=args.maxh, comm=MPI.COMM_WORLD,
        core_frac=args.core_frac, refine_h=None,
        refine_frac=refine_frac, refine_cut=args.refine_cut)
    print(f"  mesh: nv={mesh.nv}  ne={mesh.ne}", flush=True)

    pen = []
    if corner_bnd_label:
        pen = list(corner_bnd_label) if isinstance(corner_bnd_label, (list, tuple, set)) \
            else [corner_bnd_label]
    pen = [p for p in dict.fromkeys(pen) if p]
    corner_penalty_label = "|".join(pen) if pen else None

    spaces = build_spaces(mesh, contact_pairs, outer_contact_pairs,
                          order_bulk=2, order_gb=1,
                          junction_incidence=junction_incidence)
    gb_normal_indices = spaces[4]
    print(f"  ndof={spaces[0].ndof}", flush=True)

    tau_M, info = measure_tau_M(spaces, mesh, contact_pairs, outer_contact_pairs,
                                junction_incidence, num_grains=num_grains, nu=NU, mu=MU)
    print(f"  tau_M={tau_M:.6e} (eta_ss={info['eta_ss']:.4e}, G_U={info['G_U']:.4e})",
          flush=True)

    ln_grid = np.linspace(args.lnmin, args.lnmax, args.npts)
    modulus_scale = 2.0 / (MACRO_SCALE ** 2)
    rows = []
    for k, lw in enumerate(ln_grid):
        omegai = float(np.exp(lw))
        gfu, mesh2 = solve_rve(spaces, mesh, contact_pairs, outer_contact_pairs,
                               GAMMA, nu=NU, mu=MU, omega=omegai, solver='cg',
                               rtol=1e-8, corner_bnd=corner_penalty_label,
                               junction_incidence=junction_incidence, diff_coeff=tau_M)
        area = float(Integrate(1, mesh2, VOL))
        storage, diss = compute_energy_metrics(
            gfu, mesh2, contact_pairs, gb_normal_indices, omegai, area, tau_M, num_grains)
        rows.append(dict(ln_omega=lw, omega=omegai, E_storage=storage,
                         E_diss_total=diss,
                         Cxyxy_real=modulus_scale * storage,
                         Cxyxy_imag=modulus_scale * diss))
        pd.DataFrame(rows).to_csv(out_path, index=False)     # incremental
        q = diss / storage if storage > 0 else float("nan")
        print(f"  [{k+1}/{len(ln_grid)}] ln={lw:+.2f}  C'={modulus_scale*storage:.4e}"
              f"  C''={modulus_scale*diss:.4e}  Q^-1={q:.4e}", flush=True)

    # record tau_M / mesh metadata alongside
    with open(out_path.replace(".csv", "_meta.txt"), "w") as f:
        f.write(f"seed={args.seed} sigma={args.sigma} den={tag} refine_frac={refine_frac} "
                f"refine_cut={args.refine_cut}\n")
        f.write(f"tau_M={tau_M:.8e} eta_ss={info['eta_ss']:.6e} G_U={info['G_U']:.6e}\n")
        f.write(f"nv={mesh.nv} ne={mesh.ne} ndof={spaces[0].ndof}\n")
    print(f"  saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
