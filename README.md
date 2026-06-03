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
| `meshes.py` | Tessellation → Netgen OCC geometry. Labels shared edges `slide_i_j_{left,right}` and per-junction core stubs `core_i_j_{left,right}_jJID`; records `junction_incidence` (which interior edges meet at each triple junction, with their right-side core-stub names); classifies outer edges into periodic pairs (`outer_<n>_minus/plus`). `MakeMesh(..., refine_h=)` returns `(shape, geo, mesh, faces, contact_pairs, outer_contact_pairs, corner_label, outer_core_labels, junction_incidence)`. `refine_h` refines element size at corners + triple junctions (set on the `core_*` edge stubs). |
| `main.py` | `build_spaces` (mixed FE space: bulk `VectorH1²`, per-edge GB `t_n`, junction reservoir/flux multipliers, and 6 rigid-body-mode multipliers); assembly = elasticity + GB normal coupling + GB diffusion + junction coupling + periodic + RBM constraints; `solve_rve(..., diff_coeff=)` does one ω solve (Pardiso `direct` or multigrid `cg`); `diff_coeff=0` gives the ω→∞ normal-locked limit. |
| `calibrate_tau.py` | Per-geometry Maxwell-time calibration. `measure_tau_M(...)` → `(τ_M, info)` from 2 shear solves (`G_U` from `diff_coeff=0`; `η_ss=C''/ω` at `code-ω=1`). Also `storage_modulus`, `diffusional_loss`, and `fit_andrade_time` (high-ω Andrade fit → `τ_A/τ_M`). Builds no geometry of its own. |
| `real_im_energy.py` | Frequency-sweep driver over `tessellation_output.json` seeds; calibrates `τ_M` per seed (`measure_tau_M`) and runs at `diff_coeff=τ_M`; writes one CSV per (seed, loading branch): shear `Cxyxy`, uniaxial-x `Cxxxx`, uniaxial-y `Cyyyy`. |
| `real_im_hex_energy.py` | Same sweep on the regular hexagonal 6-grain RVE — reference geometry (same per-geometry `measure_tau_M` calibration). |
| `reproduce_fig1.py`, `reproduce_fig2.py` | Reproduce Rudge (2025) Figs 1 (hexagon properties vs Poisson's ratio ν) and 2 (ν=0.3 complex-modulus spectrum vs ω·τ_M). Output `reproduce_fig{1,2}.png` + CSVs. |

## Dependencies

- NGSolve (with Netgen-OCC), MPI build for parallel assembly
- NumPy, pandas, mpi4py
- Pardiso (bundled with NGSolve) for `solver='direct'`

## Running

The seed driver expects a `tessellation_output.json` in the working directory:

```bash
python real_im_energy.py            # all seeds in tessellation_output.json
python real_im_energy.py 0 10       # seeds 0..9 (start inclusive, end exclusive)
```

`tessellation_output.json` holds entries keyed `seeds_<n>`, each a
`[points, regions]` pair: `points` a list of `(x, y)`, `regions` a list of
polygons as 1-based CCW vertex indices.

Hexagonal benchmark:

```bash
python real_im_hex_energy.py            # default core_frac = 0.01
python real_im_hex_energy.py 0.005      # override core_frac
```

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
| `LN_OMEGA_MIN/MAX`, `OMEGA_SAMPLES` | driver scripts | log-ω sweep grid |
| `maxh`, `core_frac`, `refine_h` | `MakeMesh` call | global mesh size; length fraction of the per-junction `core` stub (junction-coupling region); `refine_h` sets element size at corners + triple junctions (must be `< core_frac×edge` to bite) |
| `order_bulk`, `order_gb` | `build_spaces` call | polynomial order of bulk displacement and GB multiplier spaces |
| `remove_rbm` | `build_spaces` / `solve_rve` | `True` (default) = stress-free RBM removal; `False` = legacy Nitsche corner pin (injects a storage floor — for comparison only) |

## Solver note (important for the low-ω divergence)

At low frequency the storage relaxes to `C' ~ 1e-18` (the flowing limit). Use
`solver='direct'` (Pardiso) to see the `Q⁻¹ ∝ 1/ω` divergence: the multigrid-CG
path (`rtol=1e-8`) cannot resolve a storage that small and will re-floor `C'`,
masking the divergence. The drivers default to `solver='cg'`; pass
`solver='direct'` for trustworthy low-ω results.

## License

See `LICENSE`.
