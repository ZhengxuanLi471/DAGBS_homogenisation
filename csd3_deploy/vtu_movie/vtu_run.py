# Per-frequency VTU field export for one seed (+ the hex reference), for a frequency
# movie. Re-runs the FULL extended shear sweep (ln(omega*tau_M) in [-6.02, 15.12],
# 162 pts, same grid as the analysed CSVs) and, at every omega,
# writes the spatial fields to a VTK series:
#   disp_real                  -- real displacement uR (vector)
#   disp_imag                  -- imaginary displacement uI (vector)
#   storage_energy_density     -- 0.5*(sigR:epsR + sigI:epsI), bulk scalar field
#   dissipation_energy_density -- 0.5*(C_d/omega)|d_s t_n|^2, a GRAIN-BOUNDARY field
#                                 baked into a zero VOLUME field (interiors 0,
#                                 dissipation on the GB shell) so all four live in one
#                                 VOL series.
# One .pvd per geometry carries the ln(omega*tau_M) timeline (the movie axis).
# Shear branch only. solver='cg' for ALL frequencies (user directive: never direct,
# always CG -- direct is too slow). Caveat: at the deepest low-w frames CG floors the
# tiny storage C'~w^2, so those storage panels are approximate (accepted tradeoff;
# displacement / GB-dissipation fields are fine for the movie).
#
# This script lives in csd3_deploy/vtu_movie/ (kept clean) but reuses the solver
# modules + seed data from ../sigmas/sigma_0.45/ via sys.path -- no code duplication.
#
#   python vtu_run.py --target both            # seed 24 + hex (full 162 frames each)
#   python vtu_run.py --target hex --nmax 4    # quick smoke (4 frames spanning range)

import os
import sys
import argparse
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIGMA_DIR = os.path.normpath(os.path.join(HERE, "..", "sigmas", "sigma_0.45"))
sys.path.insert(0, SIGMA_DIR)                    # solver modules live here

from main import solve_rve, build_spaces        # noqa: E402
from meshes import MakeMesh                      # noqa: E402
from physics import *                            # noqa: E402
from calibrate_tau import measure_tau_M          # noqa: E402
from ngsolve import *                            # noqa: E402
from mpi4py import MPI                           # noqa: E402

SetNumThreads(8)

NU = 0.35
MU = 1.0
MACRO_SCALE = 1e-3
LAM = 2 * MU * NU / (1 - 2 * NU)
GAMMA = ((0, 1), (0, 0))                          # shear
SEED_NAME = "seeds_24"
SUBDIV = 0                                        # plain triangles (P1); clean + small
OUT_ROOT = os.path.join(HERE, "vtu_out")
SEED_JSON = os.path.join(SIGMA_DIR, "tessellation_output.json")

# Full extended grid (same construction as the low/high-omega tail drivers).
_H = 13.0 / 99.0
_LN_LOW = (-3.0 - _H * np.arange(1, int(np.ceil(3.0 / _H)) + 1))[::-1]    # [-6,-3)
_LN_BASE = np.linspace(-3.0, 10.0, 100)
_LN_HIGH = 10.0 + _H * np.arange(1, int(np.ceil(5.0 / _H)) + 1)          # (10,15]
LN_OMEGA_FULL = np.concatenate([_LN_LOW, _LN_BASE, _LN_HIGH])            # 162 pts


def hex_geometry():
    a = np.sqrt(3)
    pts0 = [(0, 0), (3/4, 0), (1/2, a/4), (0, a/4),
            (9/4, 0), (5/2, a/4), (2, 3*a/4), (1, 3*a/4),
            (3, 0), (3, a/4), (3, a), (9/4, a), (3/4, a), (0, a)]
    pts1 = [(x - 1.5, y - 0.5 * a) for (x, y) in pts0]
    area_per_grain = 3 * np.sqrt(3) / 2
    s = np.sqrt(1 / 50 / area_per_grain)
    pts = [(x * s, y * s) for (x, y) in pts1]
    regions = [(1, 2, 3, 4), (2, 5, 6, 7, 8, 3), (5, 9, 10, 6),
               (6, 10, 11, 12, 7), (8, 7, 12, 13), (4, 3, 8, 13, 14)]
    return pts, regions, dict(maxh=0.1 * s, core_frac=0.01 * s)


def seed_geometry():
    with open(SEED_JSON) as f:
        data = json.load(f)
    pts, regions = data[SEED_NAME]
    return pts, regions, dict(maxh=0.1, core_frac=0.01)


def run(name, pts, regions, meshkw, ln_grid):
    outdir = os.path.join(OUT_ROOT, name)
    os.makedirs(outdir, exist_ok=True)
    num_grains = len(regions)
    print(f"=== {name}: {num_grains} grains, {len(ln_grid)} frames ===", flush=True)

    (_, _, mesh, _, contact_pairs, outer_contact_pairs,
     corner_bnd_label, _ocl, junction_incidence) = MakeMesh(
        pts, regions, comm=MPI.COMM_WORLD, refine_h=None, **meshkw)

    pen = []
    if corner_bnd_label:
        pen = list(corner_bnd_label) if isinstance(corner_bnd_label, (list, tuple, set)) \
            else [corner_bnd_label]
    pen = [p for p in dict.fromkeys(pen) if p]
    corner_penalty_label = "|".join(pen) if pen else None

    spaces = build_spaces(mesh, contact_pairs, outer_contact_pairs,
                          order_bulk=2, order_gb=1,
                          junction_incidence=junction_incidence)
    fes, gb_normal_indices = spaces[0], spaces[4]

    tau_M, info = measure_tau_M(spaces, mesh, contact_pairs, outer_contact_pairs,
                               junction_incidence, num_grains=num_grains, nu=NU, mu=MU)
    print(f"  tau_M={tau_M:.6e} (eta_ss={info['eta_ss']:.4e}, G_U={info['G_U']:.4e})",
          flush=True)

    # Persistent state so one VTKOutput series stays valid across frames.
    gfu_p = GridFunction(fes)
    uR, uI = gfu_p.components[0], gfu_p.components[1]
    storage_cf = 0.5 * (InnerProduct(stress(uR, LAM, MU), strain(uR)) +
                        InnerProduct(stress(uI, LAM, MU), strain(uI)))

    omega_par = Parameter(1.0)
    flux_sum = None
    for (a, b) in contact_pairs:
        idx_re, idx_im = gb_normal_indices[(a, b)]
        gtr = Grad(gfu_p.components[idx_re]).Trace()
        gti = Grad(gfu_p.components[idx_im]).Trace()
        term = InnerProduct(gtr, gtr) + InnerProduct(gti, gti)
        flux_sum = term if flux_sum is None else flux_sum + term
    diss_cf = 0.5 * (tau_M / omega_par) * flux_sum
    gb_region = mesh.Boundaries(
        "|".join(f"core_{rn}.*|slide_{rn}" for (_, rn) in contact_pairs.values()))
    diss_vol = GridFunction(H1(mesh, order=1))

    vtk = VTKOutput(mesh, coefs=[uR, uI, storage_cf, diss_vol],
                    names=["disp_real", "disp_imag", "storage_energy_density",
                           "dissipation_energy_density"],
                    filename=os.path.join(outdir, name), subdivision=SUBDIV)

    for k, lw in enumerate(ln_grid):
        omegai = float(np.exp(lw))
        solver = 'cg'                              # always CG (user directive)
        gfu, _ = solve_rve(spaces, mesh, contact_pairs, outer_contact_pairs,
                           GAMMA, nu=NU, mu=MU, omega=omegai, solver=solver,
                           rtol=1e-8, corner_bnd=corner_penalty_label,
                           junction_incidence=junction_incidence, diff_coeff=tau_M)
        gfu_p.vec.data = gfu.vec
        omega_par.Set(omegai)
        diss_vol.vec[:] = 0.0
        diss_vol.Set(diss_cf, definedon=gb_region)          # bake GB loss into volume
        vtk.Do(time=float(lw))
        print(f"  [{k+1}/{len(ln_grid)}] ln(w*tau)={lw:+.3f} ({solver})", flush=True)

    print(f"  wrote {outdir}/{name}.pvd (+frames)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-frequency VTU field export")
    ap.add_argument("--target", choices=["seed", "hex", "both"], default="both")
    ap.add_argument("--nmax", type=int, default=None,
                    help="cap frames to ~nmax evenly-spaced points (smoke test)")
    args = ap.parse_args()

    grid = LN_OMEGA_FULL
    if args.nmax and args.nmax < len(grid):
        grid = grid[np.linspace(0, len(grid) - 1, args.nmax).round().astype(int)]

    if args.target in ("hex", "both"):
        pts, regions, mk = hex_geometry()
        run("hex", pts, regions, mk, grid)
    if args.target in ("seed", "both"):
        pts, regions, mk = seed_geometry()
        run("seed24", pts, regions, mk, grid)
