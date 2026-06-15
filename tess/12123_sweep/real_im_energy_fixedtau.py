# FIXED-tau_M frequency sweep — calibration-free variant of
# real_im_energy_refine.py.
#
# Why: on tiny-facet meshes the eta_ss calibration solve (code-omega=1) can
# stagnate under CG and, mildly non-deterministically, return a garbage tau_M
# (the 12123 member-8 wide run came out ~110x off, while the SAME geometry's
# production run was fine). tau_M is a per-geometry constant that only sets the
# diffusion coefficient + loss prefactor, so the robust fix is to skip the
# calibration entirely and pass the trusted value measured in a clean run.
#
# This driver is identical to real_im_energy_refine.py EXCEPT it takes a
# required --tau_M and never calls measure_tau_M. Use the tau_M from the clean
# production _meta.txt for the same geometry.
#
#   # member 8 (ratio_9.755e-04) wide re-run with the production tau_M:
#   python real_im_energy_fixedtau.py --idx 7 --den 50 \
#       --tau_M 1.44237781e-04 --lnmin -3 --lnmax 24 --npts 207 --outtag _wide
#
# solver='cg' throughout. Shear branch only. CSV written incrementally.

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Self-contained folder: local main.py (maxiter=400), physics.py,
# meshes_refine.py shadow any repo-root versions. (calibrate_tau not needed.)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from meshes_refine import MakeMesh        # noqa: E402
from main import solve_rve, build_spaces  # noqa: E402
from physics import *                     # noqa: E402
from ngsolve import *                     # noqa: E402
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
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--idx", type=int, help="geometry index in tessellation_output.json (0 = largest ratio)")
    sel.add_argument("--key", type=str, help="geometry key, e.g. ratio_9.755e-04")
    ap.add_argument("--tau_M", type=float, required=True,
                    help="hardcoded Maxwell time (use the clean production _meta.txt value); "
                         "calibration is SKIPPED")
    ap.add_argument("--den", type=float, required=True,
                    help="refine denominator: element size = L/den on short facets; <=0 => baseline")
    ap.add_argument("--refine_cut", type=float, default=0.02)
    ap.add_argument("--maxh", type=float, default=0.1)
    ap.add_argument("--core_frac", type=float, default=0.01)
    ap.add_argument("--lnmin", type=float, default=-3.0)
    ap.add_argument("--lnmax", type=float, default=24.0)
    ap.add_argument("--npts", type=int, default=207)
    ap.add_argument("--outtag", type=str, default="",
                    help="suffix inserted in the output filename, e.g. _wide")
    # array-slicing: build one global linspace(lnmin,lnmax,ntotal) and let each
    # SLURM array task compute only its contiguous slice (chunk*chunk_pts ..).
    # Each task writes refine_<key>_frac<den><outtag>_part<NN>_shear.csv; merge
    # the parts afterwards (merge_member8_parts.py). If --ntotal<=0 the old
    # single-run --npts behaviour is used.
    ap.add_argument("--ntotal", type=int, default=0,
                    help="global grid size for array runs (>0 enables slicing)")
    ap.add_argument("--chunk", type=int, default=-1,
                    help="0-based array task index (which slice to compute)")
    ap.add_argument("--chunk_pts", type=int, default=20,
                    help="grid points per array task")
    args = ap.parse_args()

    refine_frac = None if args.den <= 0 else 1.0 / args.den
    tag = "base" if refine_frac is None else f"{int(round(args.den))}"
    tess_path = os.path.join(HERE, "tessellation_output.json")
    with open(tess_path) as f:
        data = json.load(f)
    key = args.key if args.key is not None else list(data.keys())[args.idx]
    pts, regions = data[key]
    num_grains = len(regions)

    # build the frequency grid (full, or this task's slice for array runs)
    if args.ntotal > 0 and args.chunk >= 0:
        full = np.linspace(args.lnmin, args.lnmax, args.ntotal)
        ln_grid = full[args.chunk * args.chunk_pts:(args.chunk + 1) * args.chunk_pts]
        part = f"_part{args.chunk:02d}"
        if len(ln_grid) == 0:
            print(f"chunk {args.chunk} is empty for ntotal={args.ntotal}, "
                  f"chunk_pts={args.chunk_pts} -- nothing to do", flush=True)
            return
        print(f"  array slice: chunk {args.chunk} -> grid pts "
              f"[{args.chunk*args.chunk_pts}:{args.chunk*args.chunk_pts+len(ln_grid)}] "
              f"of {args.ntotal}, ln in [{ln_grid[0]:.3f}, {ln_grid[-1]:.3f}]", flush=True)
    else:
        ln_grid = np.linspace(args.lnmin, args.lnmax, args.npts)
        part = ""
    out_path = os.path.join(HERE, f"refine_{key}_frac{tag}{args.outtag}{part}_shear.csv")

    tau_M = args.tau_M
    print(f"=== {key} ({num_grains} grains), refine_frac={refine_frac} "
          f"(L/{tag}), cut={args.refine_cut} | FIXED tau_M={tau_M:.8e} "
          f"(calibration SKIPPED) ===", flush=True)

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

    with open(out_path.replace(".csv", "_meta.txt"), "w") as f:
        f.write(f"key={key} den={tag} refine_frac={refine_frac} "
                f"refine_cut={args.refine_cut}\n")
        f.write(f"tau_M={tau_M:.8e} (HARDCODED, calibration skipped)\n")
        f.write(f"nv={mesh.nv} ne={mesh.ne} ndof={spaces[0].ndof}\n")
    print(f"  saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
