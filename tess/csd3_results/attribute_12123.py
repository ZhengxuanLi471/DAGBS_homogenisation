#!/usr/bin/env python3
"""Per-facet dissipation attribution for the (3,12,12) members-8/9 bump.

Where does the GB dissipation live at the bump frequency (member 9, ln=14.46)?
Controls: member 9 at the main peak (ln=0, expected delocalized) and member 5
at its genuine triangle peak (ln=13.80, expected localized on triangle edges).

tau_M is taken from the pulled CSD3 _meta.txt files (no recalibration -> the
stagnating eta_ss solve is never run); all solves are well-conditioned band
solves (CG, den=50).
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, "..", "12123_sweep"))
sys.path.insert(0, PKG)

from meshes_refine import MakeMesh          # noqa: E402
from main import build_spaces, solve_rve    # noqa: E402
from ngsolve import (Grad, InnerProduct, Integrate, BND, SetNumThreads)  # noqa: E402
from mpi4py import MPI                       # noqa: E402

SetNumThreads(32)   # local box has 32 cores; CSD3 scripts stay at 8

NU, MU = 0.35, 1.0
GAMMA = ((0, 1), (0, 0))

# (label, key, idx, ln_target)
# member 9's three relaxations (main / bump / triangle Coble), ordered by
# frequency, plus a same-frequency genuine-triangle-peak control (member 5).
CASES = [
    ("m9 @ main ln=0",          "ratio_4.417e-04", 8, 0.0),
    ("m9 @ bump ln=14.4",       "ratio_4.417e-04", 8, 14.43),
    ("m9 @ triangle ln=23.0",   "ratio_4.417e-04", 8, 22.95),
    ("m5 @ triangle ln=13.8 (control)", "ratio_1.051e-02", 4, 13.80),
]


def meta_tau(key):
    txt = open(os.path.join(HERE, "12123_sweep",
                            f"refine_{key}_frac50_shear_meta.txt")).read()
    return float(re.search(r"tau_M=([\d.e+-]+)", txt).group(1))


def facet_info(pts, regions):
    """(i,j) -> facet length; set of triangle region ids; triangle vertex ids."""
    pts_a = np.array(pts)
    owners = defaultdict(list)
    for ri, reg in enumerate(regions, start=1):
        m = len(reg)
        for k in range(m):
            owners[tuple(sorted((reg[k], reg[(k + 1) % m])))].append(ri)
    fac_len, fac_verts = {}, {}
    for (a, b), regs in owners.items():
        if len(regs) == 2:
            pair = tuple(sorted(regs))
            fac_len[pair] = float(np.linalg.norm(pts_a[a - 1] - pts_a[b - 1]))
            fac_verts[pair] = (a, b)
    # triangles = 3-gons with perimeter within 1.5x of the smallest
    perims = {ri: np.linalg.norm(pts_a[np.array(r) - 1] -
                                 np.roll(pts_a[np.array(r) - 1], -1, axis=0), axis=1).sum()
              for ri, r in enumerate(regions, start=1) if len(r) == 3}
    pmin = min(perims.values())
    tri_regions = {ri for ri, p in perims.items() if p < 1.5 * pmin}
    tri_vertices = set()
    for ri in tri_regions:
        tri_vertices.update(regions[ri - 1])
    return fac_len, fac_verts, tri_regions, tri_vertices


def classify(pair, fac_verts, tri_regions, tri_vertices):
    if pair[0] in tri_regions or pair[1] in tri_regions:
        return "triangle edge"
    a, b = fac_verts[pair]
    if a in tri_vertices or b in tri_vertices:
        return "adjacent long facet"
    return "other facet"


def main():
    with open(os.path.join(PKG, "tessellation_output.json")) as f:
        data = json.load(f)

    results = []
    built = {}
    for label, key, idx, ln_t in CASES:
        pts, regions = data[key]
        if key not in built:
            print(f"=== building mesh for {key} ===", flush=True)
            (_, _, mesh, _, cp, ocp, cb, _o, ji) = MakeMesh(
                pts, regions, maxh=0.1, comm=MPI.COMM_WORLD, core_frac=0.01,
                refine_h=None, refine_frac=1.0 / 50.0, refine_cut=0.02)
            spaces = build_spaces(mesh, cp, ocp, order_bulk=2, order_gb=1,
                                  junction_incidence=ji)
            pen = [p for p in dict.fromkeys(
                (list(cb) if isinstance(cb, (list, tuple, set)) else [cb])) if p]
            built[key] = (mesh, cp, ocp, "|".join(pen) or None, ji, spaces)
        mesh, cp, ocp, corner, ji, spaces = built[key]
        gb_idx = spaces[4]
        tau = meta_tau(key)

        omega = float(np.exp(ln_t))
        print(f"=== {label} (tau_M={tau:.4e}) ===", flush=True)
        gfu, mesh2 = solve_rve(spaces, mesh, cp, ocp, GAMMA, nu=NU, mu=MU,
                               omega=omega, solver='cg', rtol=1e-8,
                               corner_bnd=corner, junction_incidence=ji,
                               diff_coeff=tau)

        fac_len, fac_verts, tri_regions, tri_vertices = facet_info(pts, regions)
        per = {}
        for (a, b), (_, right_name) in cp.items():
            region = mesh2.Boundaries(f"core_{right_name}.*|slide_{right_name}")
            i_re, i_im = gb_idx[(a, b)]
            gtr = Grad(gfu.components[i_re]).Trace()
            gti = Grad(gfu.components[i_im]).Trace()
            per[(a, b)] = float(Integrate(InnerProduct(gtr, gtr) + InnerProduct(gti, gti),
                                          mesh2, BND, definedon=region))
        tot = sum(per.values())
        shares = {k: v / tot for k, v in per.items()}
        d = np.array(list(shares.values()))
        prat = (d.sum() ** 2) / (d ** 2).sum()
        by_class = defaultdict(float)
        for pair, s in shares.items():
            by_class[classify(tuple(sorted(pair)), fac_verts, tri_regions, tri_vertices)] += s
        top = sorted(shares.items(), key=lambda kv: -kv[1])[:3]
        print(f"  participation ratio = {prat:.1f}")
        for cls, s in sorted(by_class.items()):
            print(f"  {cls:>22}: {100*s:5.1f}%")
        for pair, s in top:
            L = fac_len.get(tuple(sorted(pair)), np.nan)
            print(f"  top facet {pair}: {100*s:5.1f}%  L={L:.3e}  "
                  f"[{classify(tuple(sorted(pair)), fac_verts, tri_regions, tri_vertices)}]")
        results.append((label, shares, fac_len, fac_verts, tri_regions, tri_vertices, prat, by_class))

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, len(results), figsize=(6.3 * len(results), 5.5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    cls_color = {"triangle edge": "crimson", "adjacent long facet": "darkorange",
                 "other facet": "steelblue"}
    for ax, (label, shares, fac_len, fac_verts, tri_r, tri_v, prat, by_class) in zip(axes, results):
        for pair, s in shares.items():
            p = tuple(sorted(pair))
            L = fac_len.get(p)
            if L is None:
                continue
            ax.semilogx(L, 100 * s, "o", ms=7,
                        color=cls_color[classify(p, fac_verts, tri_r, tri_v)], alpha=0.8)
        ax.set_title(f"{label}\nparticipation={prat:.1f} | " +
                     " ".join(f"{c.split()[0]}:{100*by_class[c]:.0f}%" for c in cls_color),
                     fontsize=10)
        ax.set_xlabel("facet length L"); ax.set_ylabel("dissipation share (%)")
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k)
               for k, c in cls_color.items()]
    axes[0].legend(handles=handles, fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, "attribution_12123.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
