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
surface-diffusion channel `(i/ω) ∫ ∂ₛt_n ∂ₛr_n` — the only place ω enters the
bilinear form. The per-edge `t_n` fields are stitched into one connected network
at every triple junction by an **exact-Lagrange junction coupling** (a reservoir
potential `μ_J` plus per-edge flux multipliers `λ_{e,J}`) that enforces
chemical-potential continuity (`t_n = μ_J`) and Kirchhoff flux balance
(`Σ λ = 0`) — this is what allows mass to circulate through junctions
(Coble creep). The outer box is closed with periodic `ContactBoundary` pairs and
the macro strain `Γ` enters as a traction jump. Rigid-body modes are removed by
**stress-free global integral constraints** (2 translations + 1 rotation, on Re
and Im) — no point pinning, so the relaxed modulus is not artificially floored.
Per ω, storage `= ½ ∫ σ:ε / |Ω|` and the diffusional loss
`= ½ DIFF_RATIO/(ω|Ω|) ∫ |∂ₛt_n|²`.

## Files

| File | Purpose |
| --- | --- |
| `physics.py` | `strain`, `stress`, `complexify` / `complexify_multi` (split `Re(a·conj(b))` into real/imag saddle blocks), and `gb_diffusion_block` — the `(i/ω) ∂ₛt_n·∂ₛr_n` surface-diffusion channel (uses `Grad(t_n).Trace()` for the tangential gradient). |
| `meshes.py` | Tessellation → Netgen OCC geometry. Labels shared edges `slide_i_j_{left,right}` and per-junction core stubs `core_i_j_{left,right}_jJID`; records `junction_incidence` (which interior edges meet at each triple junction, with their right-side core-stub names); classifies outer edges into periodic pairs (`outer_<n>_minus/plus`). `MakeMesh(...)` returns `(shape, geo, mesh, faces, contact_pairs, outer_contact_pairs, corner_label, outer_core_labels, junction_incidence)`. |
| `main.py` | `build_spaces` (mixed FE space: bulk `VectorH1²`, per-edge GB `t_n`, junction reservoir/flux multipliers, and 6 rigid-body-mode multipliers); assembly = elasticity + GB normal coupling + GB diffusion + junction coupling + periodic + RBM constraints; `solve_rve` does one ω solve (Pardiso `direct` or multigrid `cg`). |
| `real_im_energy.py` | Frequency-sweep driver over `tessellation_output.json` seeds; writes one CSV per (seed, loading branch): shear `Cxyxy`, uniaxial-x `Cxxxx`, uniaxial-y `Cyyyy`. |
| `real_im_hex_energy.py` | Same sweep on the regular hexagonal 6-grain RVE — reference geometry. |

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
<comp>_real, <comp>_imag`.

## Key parameters

| Knob | Where | Meaning |
| --- | --- | --- |
| `NU`, `MU` | driver scripts | bulk Poisson ratio, shear modulus |
| `DIFF_RATIO` | driver scripts | nondimensional GB diffusion coefficient. **Note:** currently only scales the *output* loss (`=1`, inert); the bilinear diffusion prefactor is hardwired to `1/ω` in `gb_diffusion_block`. The relaxation is therefore *uncalibrated* — the fluid↔solid crossover (`C'≈C''`, `Q⁻¹≈1`) sits near `ln ω ≈ 10`, i.e. implied `τ_M ≈ e⁻¹⁰` in code units. |
| `MACRO_SCALE` | driver scripts | applied macro strain magnitude (`Γ` scaled by `1e-3` in `_setup_material_properties`); modulus recovered with `2 / MACRO_SCALE²` |
| `LN_OMEGA_MIN/MAX`, `OMEGA_SAMPLES` | driver scripts | log-ω sweep grid |
| `maxh`, `core_frac` | `MakeMesh` call | global mesh size; length fraction of the per-junction `core` stub (now the junction-coupling region + local mesh refinement) |
| `order_bulk`, `order_gb` | `build_spaces` call | polynomial order of bulk displacement and GB multiplier spaces |
| `remove_rbm` | `build_spaces` / `solve_rve` | `True` (default) = stress-free RBM removal; `False` = legacy Nitsche corner pin (injects a storage floor — for comparison only) |

## Solver note (important for the low-ω divergence)

At low frequency the storage relaxes to `C' ~ 1e-18` (the flowing limit). Use
`solver='direct'` (Pardiso) to see the `Q⁻¹ ∝ 1/ω` divergence: the multigrid-CG
path (`rtol=1e-8`) cannot resolve a storage that small and will re-floor `C'`,
masking the divergence. The hex driver currently uses CG with a forced
`convergence = True` override (testing scaffolding); switch to `direct` for
trustworthy low-ω results.

## License

See `LICENSE`.
