# DAGBS_homogenisation

Frequency-domain finite-element homogenisation of 2D polycrystalline RVEs with
diffusion-accommodated grain-boundary sliding. Computes the complex effective
stiffness `C*(ω) = C'(ω) + i C''(ω)` of a periodic tessellation by sweeping ω
over many decades.

> **Model.** **DAGBS** — diffusion-accommodated grain-boundary sliding (Coble
> regime; Raj–Ashby / Rudge 2025). Interior grain boundaries slide **freely** in
> tangent (`σ_ns = 0`) and open **diffusively** in normal, with the
> grain-boundary diffusion field connected **through triple junctions**. That
> connection produces steady-state Coble creep: at low frequency the aggregate
> flows, so `C' → 0` (`∝ ω²`) and the attenuation diverges,
> `Q⁻¹ = C''/C' ∝ 1/ω`. (The earlier elastically-accommodated EAGBS
> viscous-sliding model lives in git history.) Physics derivation:
> `docs/DAGBS_physics.md`; per-session working notes: `CLAUDE.md`.

## Method in one paragraph

The displacement is solved as a complex `(u_Re, u_Im)` pair on the RVE
(linear-elastic grains). On every interior grain boundary tangential sliding is
**free** (no tangential multiplier), and the normal opening is governed by a
single scalar field `t_n` that does double duty: it is the normal-traction
Lagrange multiplier **and** the diffusion potential (chemical potential
`μ = −Ω σ_nn`, Herring). Its weak form carries the Coble/Raj–Ashby
surface-diffusion channel `(i·C_d/ω) ∫ ∂ₛt_n ∂ₛr_n` — the only place ω enters the
bilinear form (`C_d` is the GB diffusion coefficient; see Maxwell-time
calibration below). The per-edge `t_n` fields are stitched into one connected network
at every triple junction by an **exact-Lagrange junction coupling** (a reservoir
potential `μ_J` plus per-edge flux multipliers `λ_{e,J}`) that enforces
chemical-potential continuity (`t_n = μ_J`) and Kirchhoff flux balance
(`Σ λ = 0`) — this is what allows mass to circulate through junctions
(Coble creep). The outer box is closed with periodic `ContactBoundary` pairs and
the macro strain `Γ` enters as a traction jump. Rigid-body modes are removed by
**stress-free global integral constraints** (2 translations + 1 rotation, on Re
and Im) — no point pinning, so the relaxed modulus is not artificially floored.
Per ω, storage `= ½ ∫ σ:ε / |Ω|` and the diffusional loss
`= ½ C_d/(ω|Ω|) ∫ |∂ₛt_n|²`, where `C_d` is the same diffusion coefficient used
in the form (the per-geometry Maxwell time `τ_M`).

## Maxwell-time calibration

The natural timescale is the Maxwell time `τ_M = η_ss / G_U` (steady-state Coble
viscosity / unrelaxed shear modulus) — both are **emergent** properties of the
homogenised RVE, so `τ_M` is **geometry-dependent and measured at runtime, never
hardcoded**. `calibrate_tau.measure_tau_M` does it in 2 shear solves: `G_U` from
the `diff_coeff=0` normal-locked limit, and `η_ss = C''/ω` on the steady-creep
plateau (measured at the well-conditioned `code-ω = 1` — pushing ω very low makes
the `1/ω` diffusion term stiff and CG diverges). Running the sweep with
`diff_coeff = τ_M` folds the Maxwell time into the form, so code-ω reads `ω·τ_M`
and the fluid↔solid crossover sits at ω ≈ 1. Hex benchmark (ν=0.3): `G_U/μ≈0.81`,
`τ_M≈2.98e-5`.

## Files

| File | Purpose |
| --- | --- |
| `physics.py` | `strain`, `stress`, `complexify` / `complexify_multi` (split `Re(a·conj(b))` into real/imag saddle blocks), and `gb_diffusion_block` — the `(i·C_d/ω) ∂ₛt_n·∂ₛr_n` surface-diffusion channel (`C_d=diff_coeff`, default `DIFF_COEFF=1.0`; uses `Grad(t_n).Trace()` for the tangential gradient). |
| `meshes.py` | Tessellation → Netgen OCC geometry. Labels shared edges `slide_i_j_{left,right}` and per-junction core stubs `core_i_j_{left,right}_jJID`; records `junction_incidence` (which interior edges meet at each triple junction, with their right-side core-stub names); classifies outer edges into periodic pairs (`outer_<n>_minus/plus`). `MakeMesh(..., refine_h=, refine_frac=, refine_cut=0.02)` returns `(shape, geo, mesh, faces, contact_pairs, outer_contact_pairs, corner_label, outer_core_labels, junction_incidence)`. `refine_h` refines element size at corners + triple junctions (set on the `core_*` edge stubs). **`refine_frac` (optional, backward-compatible — `None` = off): per-facet *proportional* refinement** — interior GB facets shorter than `refine_cut` get element size `L*refine_frac` on their `core_`+`slide_` edges, so sub-`maxh` slivers are actually resolved. |
| `main.py` | `build_spaces` (mixed FE space: bulk `VectorH1²`, per-edge GB `t_n`, junction reservoir/flux multipliers, and 6 rigid-body-mode multipliers); assembly = elasticity + GB normal coupling + GB diffusion + junction coupling + periodic + RBM constraints; `solve_rve(..., diff_coeff=)` does one ω solve. **Default `solver='cg'`** (multigrid, `maxiter=400`); legacy Pardiso `direct` still reachable with `solver='direct'`. `diff_coeff=0` gives the ω→∞ normal-locked limit. |
| `calibrate_tau.py` | Per-geometry Maxwell-time calibration. `measure_tau_M(...)` → `(τ_M, info)` from 2 shear solves (`G_U` from `diff_coeff=0`; `η_ss=C''/ω` at `code-ω=1`). Also `storage_modulus`, `diffusional_loss`, and `fit_andrade_time` (high-ω Andrade fit → `τ_A/τ_M`). Builds no geometry of its own. |
| `real_im_energy.py` | **Master frequency-sweep driver** (the `_refine` driver promoted to canonical). **Shear branch only** (`Cxyxy`), `solver='cg'`, with optional per-facet mesh refinement via `MakeMesh(refine_frac=, refine_cut=)`. Selects one geometry from `tessellation_output.json` by `--idx N` (0-based) or `--key K`; `--den D` sets `refine_frac = 1/D` (`--den 0` = no refinement); `--outtag` suffixes the CSV name. Calibrates `τ_M` (`measure_tau_M`), runs at `diff_coeff=τ_M`, and writes `refine_<key>_frac<den><outtag>_shear.csv` incrementally (one row per ω, so a walltime kill leaves a usable partial curve). NB the old 3-branch version (which also wrote uniaxial `Cxxxx`/`Cyyyy`) is gone; recover uniaxial branches from git history if needed. |
| `real_im_energy_lowomega.py` | Low-ω **tail** companion to `real_im_energy.py`: sweeps only `ln(ω·τ_M) ∈ [-6, -3)` (23 pts, no overlap) with `solver='cg'`, writing `*_shear_lowomega.csv`. Extends a main sweep below `ln(ω·τ_M) = -3` into the creep plateau without recomputing it — concatenate the two CSVs afterwards. (Its grid step was matched to the historical `[-3, 10]` σ-ensemble grid, not the current `[-3, 25]` default — re-match steps before attaching to fresh runs.) **CG floors the tiny low-ω `C'`, so the relaxation strength Δ from this tail is approximate** (accepted under the CG-everywhere directive). |
| `real_im_energy_highomega.py` | High-ω **tail** companion: sweeps only `ln(ω·τ_M) ∈ (10, 15]` (39 pts on the same grid step, no overlap) with `solver='cg'` (the unrelaxed plateau is well-conditioned and direct is far too slow), writing `*_shear_highomega.csv`. Extends the sweep up the elastic plateau for a cleaner unrelaxed `G0` and the high-ω `Q⁻¹` asymptote — concatenate afterwards. |
| `real_im_hex_energy.py` | Same sweep on the regular hexagonal 6-grain RVE — reference geometry (same per-geometry `measure_tau_M` calibration). |
| `reproduce_fig1.py`, `reproduce_fig2.py` | Reproduce Rudge (2025) Figs 1 (hexagon properties vs Poisson's ratio ν) and 2 (ν=0.3 complex-modulus spectrum vs ω·τ_M). Output `reproduce_fig{1,2}.png` + CSVs. |

## Geometry toolkit (`tess/`)

`tess/` generates structured Archimedean-tiling RVEs as controlled alternatives
to the random Neper seeds. Three families — truncated square **(4,8,8)**
(`tess/84*`), truncated hexagonal **(3,12,12)** (`tess/12123*`), and truncated
trihexagonal **(4,6,12)** (`tess/4612_sweep`, generator `tess/gen_hex_seeds.py`)
— each pass through the same pipeline: generate the periodic tiling →
`cut_paste_rect.py` (re-cut to a rectangular periodic box with no grain boundary
on the box edge) → faces-as-cells JSON in the solver's `tessellation_output.json`
format.

Each `tess/<family>_sweep/` folder holds a one-parameter **small-grain-size
sweep**: `gen_sweep.py` produces ~10 geometries whose small-grain/large-grain
boundary-length ratio is log-spaced from the Archimedean value down to 1/5000
(keys `ratio_<r>`), with built-in verification (exact area, periodic pairing,
intact small grains, exact-once point coverage). `84_sweep/` and `12123_sweep/`
are additionally self-contained sweep packages: `real_im_energy_refine.py`
(frequency sweep with per-facet proportional mesh refinement from
`meshes_refine.py`) plus `run_refine.sh` (SLURM array, one task per geometry)
and local copies of the solver modules.

## Dependencies

- NGSolve (with Netgen-OCC), MPI build for parallel assembly
- NumPy, pandas, mpi4py
- Pardiso (bundled with NGSolve) for `solver='direct'`

## Running

The driver expects a `tessellation_output.json` in the working directory and
runs **one geometry at a time** (shear branch, `solver='cg'`):

```bash
python real_im_energy.py --idx 0 --den 0            # geometry index 0, no mesh refinement
python real_im_energy.py --idx 0 --den 50           # refine_frac = 1/50 on short facets
python real_im_energy.py --key seeds_3 --den 100    # select a geometry by key
```

`tessellation_output.json` holds entries keyed by name (e.g. `seeds_<n>` or
`ratio_<r>`), each a `[points, regions]` pair: `points` a list of `(x, y)`,
`regions` a list of polygons as 1-based CCW vertex indices. `--idx` selects the
N-th key; `--key` selects by name; `--den D` sets the proportional-refinement
denominator (`refine_frac = 1/D`, `--den 0` = none); `--outtag` suffixes the CSV.
The default frequency window is `ln(ω·τ_M) ∈ [-3, 25]` with 140 points
(`--lnmin`/`--lnmax`/`--npts` override it).

Hexagonal benchmark:

```bash
python real_im_hex_energy.py            # default core_frac = 0.01
python real_im_hex_energy.py 0.005      # override core_frac
```

The hex benchmark sweeps the same default window `ln(ω·τ_M) ∈ [-3, 25]`, 140
points (module constants `LN_OMEGA_MIN/MAX`, `OMEGA_SAMPLES`).

Outputs are CSVs with columns `ln_omega, omega, E_storage, E_diss_total,
<comp>_real, <comp>_imag`. Each driver calibrates the Maxwell time `τ_M` once per
geometry/seed (2 extra solves) and runs at `diff_coeff=τ_M`, so the ω axis reads
`ω·τ_M`.

Reproduce the Rudge (2025) hexagon figures:

```bash
python reproduce_fig2.py 1e-5   # nu=0.3 spectrum (refine_h=1e-5) -> reproduce_fig2.png
python reproduce_fig1.py 1e-5   # properties vs nu               -> reproduce_fig1.png
```

## Key parameters

| Knob | Where | Meaning |
| --- | --- | --- |
| `NU`, `MU` | driver scripts | bulk Poisson ratio, shear modulus |
| `diff_coeff` (`C_d`) | `solve_rve` arg / `physics.DIFF_COEFF` | nondimensional GB diffusion coefficient. Set to the per-geometry Maxwell time `τ_M` (measured by `calibrate_tau.measure_tau_M`), which rescales code-ω to `ω·τ_M` so the fluid↔solid crossover (`C'≈C''`, `Q⁻¹≈1`) sits at ω≈1. The **same** `C_d` scales the output-loss prefactor. Default `1.0` = uncalibrated reference. |
| `MACRO_SCALE` | driver scripts | applied macro strain magnitude (`Γ` scaled by `1e-3` in `_setup_material_properties`); modulus recovered with `2 / MACRO_SCALE²` |
| `LN_OMEGA_MIN/MAX`, `OMEGA_SAMPLES` (hex) / `--lnmin`/`--lnmax`/`--npts` (seed driver) | driver scripts / CLI | log-ω sweep grid. **Default `ln(ω·τ_M) ∈ [-3, 25]`, 140 points** for both `real_im_energy.py` and `real_im_hex_energy.py`. |
| `maxh`, `core_frac`, `refine_h` | `MakeMesh` call | global mesh size; length fraction of the per-junction `core` stub (junction-coupling region); `refine_h` sets element size at corners + triple junctions (must be `< core_frac×edge` to bite) |
| `refine_frac`, `refine_cut` | `MakeMesh` call (driver `--den`) | optional per-facet proportional refinement: interior GB facets with `L < refine_cut` get element size `L*refine_frac`. `None`/`--den 0` = off (backward-compatible). Resolves sub-`maxh` slivers. |
| `order_bulk`, `order_gb` | `build_spaces` call | polynomial order of bulk displacement and GB multiplier spaces |
| `remove_rbm` | `build_spaces` / `solve_rve` | `True` (default) = stress-free RBM removal; `False` = legacy Nitsche corner pin (injects a storage floor — for comparison only) |

## Solver note (CG everywhere)

The toolkit uses **`solver='cg'` everywhere by default** (multigrid, fast at
scale). The legacy `solver='direct'` (Pardiso) path is still reachable but is not
used by any driver. Caveat: at low frequency the storage relaxes to `C' ~ 1e-18`
(the flowing limit), and CG (`rtol=1e-8`) cannot resolve a storage that small —
it re-floors `C'`, so the `Q⁻¹ ∝ 1/ω` divergence and the relaxation strength Δ
measured from the low-ω tail (`real_im_energy_lowomega.py`) are **approximate**.
This is accepted: direct (Pardiso) is more accurate at low ω but far too slow at
production scale. Pass `solver='direct'` manually only for a one-off
high-accuracy low-ω check.

## License

See `LICENSE`.
