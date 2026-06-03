# CLAUDE.md

Working notes for future Claude sessions on this repo. Read this before
touching the solver.

## Project goal

Build a frequency-domain FE homogeniser for 2D polycrystals with grain-boundary
sliding, producing `C*(ω)`. The code was forked-by-hand from a prior EAGBS
solver (https://github.com/ZhengxuanLi471/EAGBS_homogenisation). The solver now
implements **DAGBS** (diffusion-accommodated GB sliding, Coble regime;
Raj–Ashby / Rudge 2025). EAGBS was replaced outright — recover it from git
history if the elastically-accommodated benchmark is needed. The physics
derivation lives in `docs/DAGBS_physics.md` (cross-checked against Rudge 2025
Appendix A).

## Physics model currently in the code

Mixed saddle problem, complex-valued, split into real and imaginary unknowns:

- Bulk: linear elasticity on every grain (`region_<i>`), `VectorH1`, order 2.
- Interior GB segments — every shared edge slides **freely** in tangent
  (`σ_ns = 0`, no tangential multiplier) and opens **diffusively** in normal:
  - Normal coupling `−∫[v_n]t_n* −∫[u_n]*r_n` is applied uniformly over the whole
    edge (both `core_i_j_{left,right}_jJID` stubs and `slide_i_j_{left,right}`).
  - The diffusion channel `(i/ω)∫ ∂ₛt_n·∂ₛr_n` (Coble/Raj–Ashby — the
    surface-Laplacian of the normal traction = Laplace–Beltrami of `t_n`) is
    added over `core_i_j ∪ slide_i_j`. **This is the only place ω enters the
    bilinear form** (coefficient `C_d/ω`, where `C_d` is the GB diffusion
    coefficient = the per-geometry Maxwell time `τ_M`; see "Maxwell-time
    calibration" below).
  - **Triple-junction coupling (`_add_gb_junction_coupling`)** connects the
    per-edge `t_n` fields: a reservoir potential `μ_J` per junction + a flux
    multiplier `λ_{e,J}` per incident edge (complex `NumberSpace` scalars,
    **`definedon`-restricted to the junction's stub region** — see the
    "Assembly performance" gotcha; a plain global `NumberSpace` is catastrophic).
    A mortar constraint over each edge's `core` stub at `J` enforces
    `t_n^{(e)} = μ_J` (continuity); stationarity in `μ_J` gives `Σ λ = 0` (flux
    balance). **This is what enables Coble creep** — without it the boundaries are
    diffusively isolated (zero-flux natural BC) → SLS, no low-ω divergence.
  - The `core` stubs are now the **junction-coupling region** (and local mesh
    refinement); physically core and slide are treated identically.
- Outer box: classified into periodic pairs `outer_<n>_{minus,plus}` with a
  displacement vector. **Both** tangential and normal multiplier blocks enforce
  `u(x + d) − u(x) = Γ · d` (the periodic box is not a real GB, so no free-slide /
  diffusion there). Macro strain `Γ` enters as a traction jump on the RHS.
- **Rigid-body-mode removal (`_add_rbm_constraints`, default `remove_rbm=True`)** —
  3 stress-free global integral constraints (2 translations + 1 rotation, on Re
  and Im) instead of a corner pin. The rotation target matches the affine field's
  rotation (nonzero for simple shear). The legacy Nitsche corner pin
  (`_add_corner_penalty`) is still reachable with `remove_rbm=False` but **floors
  the storage modulus** (injects non-relaxing corner stress) — for comparison
  only.

The "i·C_d/ω" coupling is implemented in `physics.gb_diffusion_block` as
`(C_d/ω) · complexify((-Grad(t_n_Im).Trace(), Grad(t_n_Re).Trace()), (...))`
(`C_d = diff_coeff`, defaulting to `physics.DIFF_COEFF=1.0`) — the
swap+sign is multiplication by `i` in real/imag blocks, and `.Trace()` is
**required** to get the *tangential* (surface) gradient of a boundary-restricted
H1 field (plain `Grad` raises "Trialfunction does not support BND-forms").

## File map

- `physics.py` — `strain`, `stress(w, lam, mu)`, `complexify` (saddle real/imag
  block for inner products), `complexify_multi` (scalar version),
  `gb_diffusion_block(t_n_Re, t_n_Im, r_n_Re, r_n_Im, omega, diff_coeff=None)`
  (the DAGBS surface-diffusion `(i·C_d/ω)∂ₛt_n·∂ₛr_n` term; `C_d=diff_coeff` or
  `DIFF_COEFF=1.0`), and the module constant `DIFF_COEFF` (uncalibrated
  reference; the real value is the per-geometry `τ_M` passed at runtime).
- `meshes.py` — geometry + boundary labelling. Entry point: `MakeMesh(pts,
  regions, maxh, comm, core_frac=, corner_core_frac=, refine_h=)`. Returns
  `(shape, geo, mesh, faces, contact_pairs, outer_contact_pairs, corner_label,
  outer_core_labels, junction_incidence)` (**9-tuple** — `junction_incidence` is
  new). `contact_pairs[(i,j)] = ('i_j_left', 'i_j_right')` feeds the GB
  integrator. Core stubs are named **per junction** `core_i_j_lr_jJID`;
  `junction_incidence[jid] = [((i,j), right_stub_name), ...]` lists the interior
  edges meeting at triple junction `jid`. `outer_contact_pairs` keys like
  `"outer_pair_1"` carry `displacement` + `minus/plus` prefixes and core-name
  lists.
- `main.py` — `build_spaces(mesh, contact_pairs, outer_contact_pairs, order_bulk,
  order_gb, junction_incidence=, remove_rbm=True)` returns
  `(fes, V_Re, V_Im, sym, gb_normal_indices)` (the `t_n` Re/Im trial indices per
  interior GB, for the loss post-processing). The mixed `fes` is built in groups
  of ~√N spaces. **GB/outer multiplier spaces are wrapped in `Compress(...)`** and
  **junction `NumberSpace`s are `definedon`-restricted to their stub region** —
  both are *load-bearing for assembly speed* (see "Assembly performance" gotcha):
  without them a ~70-grain seed builds an ~8M-dof system that takes >15 min/solve
  (effectively hangs); with them it's ~0.27M dofs and ~60 s/solve. Assembly:
  `_add_gb_normal_coupling` (term4+term5 via `ContactBoundary`, for `'slide'` and
  `'core'`, `intorder=4`), `_add_gb_diffusion` (`ds` self-term over `core∪slide`),
  `_add_gb_junction_coupling` (reservoir/flux Lagrange at junctions),
  `_add_outer_terms` (periodic), and either `_add_rbm_constraints` (default) or
  `_add_corner_penalty` (legacy).
  `solve_rve(spaces, mesh, contact_pairs, outer_contact_pairs, gamma, nu, mu,
  omega, solver, rtol, corner_bnd, junction_incidence=, remove_rbm=True,
  diff_coeff=None)` is the single-frequency solve; it **returns `(gfu, mesh)`**
  (the old `convergence` flag and the drivers' ω-retry loop were removed — see
  "Convergence gate"). `diff_coeff` overrides the GB
  diffusion coefficient `C_d` (default `physics.DIFF_COEFF`; pass `0.0` for the
  ω→∞ normal-locked unrelaxed limit used to measure `G_U`). **`build_spaces` and `solve_rve` must be called with the
  same `junction_incidence` and `remove_rbm`** (the FES must match the assembly).
- `real_im_energy.py` — frequency-sweep driver over seeds from
  `tessellation_output.json`. Computes storage from `½ ∫ σ:ε / |Ω|` (grain by
  grain to avoid integrating across GBs) and **diffusional** dissipation from
  `½ C_d / (ω|Ω|) ∫_{core∪slide} |∂ₛt_n|²`. **Per seed it calls
  `calibrate_tau.measure_tau_M` to get `τ_M` and passes `diff_coeff=τ_M`** to
  both `solve_rve` and the loss prefactor (the same `C_d` must scale form and
  loss). The old module-level `DIFF_RATIO=1.0` is gone.
- `real_im_hex_energy.py` — hexagonal benchmark RVE swept over frequency (same
  per-geometry `measure_tau_M` calibration as the seed driver).
- `calibrate_tau.py` — per-geometry Maxwell-time calibration. `measure_tau_M(
  spaces, mesh, contact_pairs, outer_contact_pairs, junction_incidence,
  num_grains, nu, mu)` returns `(τ_M, info)` from **2 shear solves**: `G_U` from
  the `diff_coeff=0` normal-locked limit, `η_ss=C''/ω` at **`code-ω=1`**
  (`LOW_OMEGA=1.0`; see the Maxwell-time section for why NOT a tiny ω),
  `τ_M=η_ss/G_U`. Builds no geometry of its own. Also exposes `storage_modulus`,
  `diffusional_loss`, and `fit_andrade_time` (high-ω Andrade fit → `τ_A/τ_M`).
- `reproduce_fig1.py`, `reproduce_fig2.py` — reproduce Rudge (2025) Figs 1 (vs ν)
  and 2 (ν=0.3 spectrum) for the hexagon; match the paper. Reuse `measure_tau_M`
  + the `calibrate_tau` helpers; pass a small `refine_h` for high-ω fidelity.
  Extended-Burgers overlay uses `scipy.special.hyp2f1` (complex z).

## Conventions / gotchas

- All `slide_*` and `core_*` boundary names are constructed in `meshes.py` and
  consumed in `main.py` — keep them in lockstep. Core stubs carry a per-junction
  suffix (`core_i_j_lr_jJID`); consumers select them with a `core_i_j_lr.*`
  **regex** (NGSolve `mesh.Boundaries` is *full-match* regex — `core_1_2` matches
  nothing, `core_1_2.*` matches both stubs; the `_lr` anchor avoids multi-digit
  collisions like `1_2` vs `1_20`). `slide` is a single segment per side (no
  suffix). The exact stub name (for the junction mortar `ds`) comes from
  `junction_incidence`.
- The macro strain tensor is passed as a 2×2 nested tuple to `solve_rve`, e.g.
  `((0,1),(0,0))` for pure shear. Inside `_setup_material_properties` it is
  multiplied by `1e-3` (`MACRO_SCALE`). Effective modulus is recovered with
  `modulus_scale = 2 / MACRO_SCALE²`.
- `core_frac` plays two roles: (i) length fraction of the per-junction `core`
  stub, which is the **junction-coupling region** (the mortar constraint
  `∫_stub (t_n − μ_J)` is integrated over it — smaller `core_frac` → more
  point-like continuity), and (ii) the default local mesh-size target
  (`smallh = min(maxh, core_frac * 0.5)`) when `refine_h` is not given.
- `refine_h` (new) sets the target element size at the outer corners AND interior
  triple junctions, **decoupled** from `core_frac`. Mechanism (see meshes.py): set
  `.maxh = refine_h` on every `core_*` named **edge** (the corner + junction
  stubs). Gotchas learned the hard way: `OCCGeometry` has **no `SetLocalH`** (the
  old corner-label call was a silent no-op in a try/except); vertex `.maxh` only
  bites on *boundary* vertices, not interior junction vertices; and `refine_h`
  must be **< the stub length** (`core_frac×edge ≈ 9e-5` for the hex) or it won't
  subdivide. With `refine_h` below that, the mesh grades out from each junction
  (hex: 1e-5→ne≈6k, 3e-6→ne≈12k). Needed for high-`ω·τ_M` attenuation fidelity.
- `solver='direct'` uses Pardiso. `solver='cg'` uses multigrid preconditioning
  with `maxiter=50` and stops at relative residual `rtol` (now `1e-8`). On the
  fixed (`Compress`+`definedon`) system CG converges in ~2 iterations even for
  ~70-grain seeds.
- **`rtol` is used ONLY by the CG solver** (`CGSolver(tol=rtol)`); `solver='direct'`
  ignores it. There is no longer any post-hoc residual gate — see "Convergence gate".
- **Assembly performance (the big one).** NGSolve compound-FESpace assembly cost
  scales hard with the *number of component spaces*, and a **global `NumberSpace`
  is treated as present on every element** — so ~800 global junction scalars (a
  ~70-grain seed: 100 junctions × (μ_J + per-edge λ)) turn a 4 s elastic assemble
  into >14 min, *independent of thread count* (measured). Two fixes, both in
  `build_spaces`, keep it cheap: (1) `Compress(...)` the GB/outer `H1` multiplier
  spaces — a plain `definedon=bdry` H1 still allocates FULL-mesh ndof (only flags
  off-region dofs unused), so ~340 of them stack to millions of dead dofs (8M
  total → the solve froze); `Compress` drops them to ~17 each. (2)
  `definedon=<stub region>` on the junction `NumberSpace`s so their dof is local
  to the stub BND elements (absent on all VOL elements). Net: ~70-grain seed
  ~0.27M dofs, calibration (2 solves) ~2.2 min. Do **not** reintroduce a bare
  `NumberSpace(mesh)` for per-junction unknowns.
- **Convergence gate (removed).** `_solve` returns just `gfu`; `solve_rve` returns
  `(gfu, mesh)`. The old post-hoc `rel_residual < rtol` boolean, the drivers'
  `while not convergence:` ω-nudging retry, and the hex driver's `convergence=True`
  override are all gone. `_solve` still *prints* the relative residual as a
  diagnostic. (Callers must unpack the 2-tuple: `gfu, mesh = solve_rve(...)`.)
- DOF count grows roughly as `2 · (#interior GB segments)` (normal multiplier
  only — free sliding dropped the tangential one) plus `4 · (#outer pairs)` plus
  the junction scalars (`2 · #junctions` reservoirs + `2 · #edge-incidences`
  fluxes) plus `6` RBM scalars, on top of the bulk H1². With `Compress`+`definedon`
  the multiplier blocks are tiny and the bulk H1² dominates (seed_1: 0.26M of 0.27M).
- `build_spaces` checks `Trial arity == name_order_trial`; if you add a new
  multiplier block you must update **both** the space list and the name list,
  or this assertion fires.

## DAGBS implementation notes

DAGBS is implemented **and the low-ω attenuation divergence is achieved**
(hex shear, direct solver: `C' ∝ ω²`, `C'' ∝ ω`, `Q⁻¹ ∝ 1/ω` slope −1.000 over
ln ω ∈ [−10, 10]). Facts a future session needs:

- `t_n` *is* the diffusion variable (the chemical potential `μ = −Ω σ_nn` folded
  in); the weak form `(i/ω)∫∂ₛt_n·∂ₛr_n` encodes the surface-Laplacian law
  directly. No separate `μ` field; the interior tangential multiplier was removed.
- **The divergence needed two pieces** (see `docs/DAGBS_physics.md` §6.2/§6.3):
  (1) **junction coupling** to connect the per-edge `t_n` through triple junctions
  (else diffusively isolated → SLS, `Q⁻¹ → 0`); (2) **stress-free RBM removal**
  (`remove_rbm=True`) — the old Nitsche corner pin floored `C'` at ~5.6e-8
  (confirmed: floor ∝ `gammaN`, `C''` invariant), masking the divergence. With
  both, `C'` relaxes to ~1e-18 (flow) and `Q⁻¹ ∝ 1/ω` natively.
- **i/ω sign is correct** (no swap flip): `C'' > 0`, `C' → 0`, divergence in the
  flowing direction.
- **Nondimensionalisation / Maxwell time IS now calibrated (per geometry).** The
  diffusion coefficient `C_d` is a `diff_coeff` argument (default
  `physics.DIFF_COEFF=1.0`, uncalibrated). `calibrate_tau.measure_tau_M` measures
  `τ_M = η_ss/G_U` for each geometry — `G_U` from the `diff_coeff=0` normal-locked
  limit, `η_ss=C''/ω` at `code-ω=1` (creep plateau) — and the drivers run with
  `diff_coeff=τ_M`, which rescales code-ω to `ω·τ_M` so the fluid↔solid crossover
  sits at ω≈1 (`Q⁻¹=1` near `ln ω≈-0.3`, slightly off 1 because the spectrum is
  distributed not single-Debye). The same `τ_M` also scales the output-loss
  prefactor. Hex benchmark: `η_ss=2.418e-5`, `G_U=0.8211`, `τ_M=2.944e-5`.
- **Solver caveat:** at low ω `C'` is ~1e-18; only `solver='direct'` resolves it.
  CG (`rtol=1e-8`) re-floors `C'` at the residual level and re-hides the
  divergence — use `solver='direct'` for trustworthy low-ω points. (The drivers'
  forced `convergence=True` override and ω-retry loop have been removed; the
  drivers still default to `solver='cg'`, so flip to direct for the `1/ω` tail.)

References: Raj & Ashby (1971) coupled model; Rudge (2025, App. A) the GB
diffusion law actually used; Ghahremani (1980) hex EAGBS limit; Lee & Morris
(2010) / Jackson et al. for diffusional creep in mantle olivine.

## Build/run

Run under the `ngsolve` conda env:
`/home/zl471/miniconda3/envs/ngsolve/bin/python`. Single sweep:

```bash
python real_im_energy.py            # all seeds in tessellation_output.json
python real_im_energy.py 0 10       # seeds 0..9
python real_im_hex_energy.py 0.01   # hex benchmark with core_frac=0.01
```

There is no Makefile, no test suite, and no CI. Outputs are CSVs in the cwd.
Both drivers call `calibrate_tau.measure_tau_M` once per geometry/seed before
sweeping (2 extra solves) and run with `diff_coeff=τ_M`.

## Maxwell-time calibration

`τ_M = η_ss/G_U` (steady-state Coble viscosity / unrelaxed shear modulus), both
emergent → geometry-dependent, measured at runtime, **never hardcoded**.
`calibrate_tau.measure_tau_M` does 2 shear solves: `G_U` = `diff_coeff=0`
normal-locked limit (ω-independent); `η_ss` = `C''/ω` at **`code-ω=1`**
(`LOW_OMEGA=1.0`). **Measure η_ss at a moderate ω, NOT a tiny one**: the creep
plateau is flat for `ω·τ_M ≪ 1` (at diff_coeff=1, code-ω ≪ 1/τ_M ≈ 3e4), so
code-ω=1 is already deep in creep AND well-conditioned. Pushing ω low (e.g.
`ln ω=-10`) makes the diffusion form-coeff `1/ω≈e^10≈22000` stiff and the
bulk-only multigrid **CG fails for near-incompressible ν** → garbage η_ss/τ_M
(silently broke per-ν τ_M / Fig-1 τ_A before this fix). At code-ω=1, η_ss comes
out `2.418e-5` for ALL ν (geometry-only Coble viscosity, ν-independent).
Running with `diff_coeff=τ_M` rescales code-ω to `ω·τ_M`. The same `τ_M` must
scale both the form and the output-loss prefactor (drivers handle this). NB:
after calibration the seed driver's band `ln ω ∈ [-3,10]` sits mostly on the
solid side of the ω≈1 crossover — widen the low end if the creep tail is wanted.

## Things to fix later (low priority)

- The hex driver uses module-level globals (`global mesh` inside `run_branch`)
  while the seed driver passes them explicitly. Worth unifying.
- `build_spaces` builds the product space via repeated `*` in a Python loop;
  the grouping by `√n` is a workaround for old NGSolve eval-string limits and
  may no longer be necessary on recent NGSolve.
- `corner_core_frac` is accepted by `MakeMesh` but forced to `0.0` inside
  `build_geometry_with_region_labels`; dead parameter, can be removed.
- **Maxwell time is calibrated** (`τ_M=η_ss/G_U`, per geometry — see the section
  above); `C_d` is still a *nondimensional* group, not anchored to a physical
  `η_Coble=(kT/ΩδD^gb)·L³`. Anchoring to lab units is the remaining step.
- **Drivers default to `solver='cg'`** (the forced `convergence=True` override and
  ω-retry loop are now removed — see "Convergence gate"). CG re-floors `C'` at low ω
  and hides the `1/ω` divergence; a production sweep that wants the tail must pass
  `solver='direct'`. (Calibration is robust to this — `η_ss`/`G_U` come from
  well-conditioned CG points.)
- The legacy corner pin (`_add_corner_penalty`, `remove_rbm=False`) is kept only
  for comparison; it floors `C'`. Default is the stress-free RBM removal.
