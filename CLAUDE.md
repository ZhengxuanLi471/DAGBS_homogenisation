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
- `meshes.py` — geometry + boundary labelling. **This is the former
  `meshes_refine.py` promoted to canonical** (strict backward-compatible superset
  of the old `meshes.py`). Entry point: `MakeMesh(pts,
  regions, maxh, comm, core_frac=, corner_core_frac=, refine_h=, refine_frac=None,
  refine_cut=0.02)`. `refine_frac` (optional, `None`=off) does per-facet
  proportional refinement: interior GB facets with `L < refine_cut` get element
  size `L*refine_frac` on their `core_`+`slide_` edges (resolves sub-`maxh`
  slivers). Returns
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
- `real_im_energy.py` — **master frequency-sweep driver** (the former
  `real_im_energy_refine.py` general fork, promoted to canonical 2026-06-15).
  **Shear branch only** (`Cxyxy`), `solver='cg'`, runs ONE geometry per
  invocation: `--idx N` or `--key K` selects a geometry from
  `tessellation_output.json`, `--den D` sets `refine_frac=1/D` (`--den 0` = no
  refinement) forwarded to `MakeMesh`, `--outtag` suffixes the CSV. Writes
  `refine_<key>_frac<den><outtag>_shear.csv` incrementally (row per ω). Computes
  storage from `½ ∫ σ:ε / |Ω|` (grain by grain) and **diffusional** dissipation
  from `½ C_d / (ω|Ω|) ∫_{core∪slide} |∂ₛt_n|²`; calls `measure_tau_M` and runs
  at `diff_coeff=τ_M`. **The old 3-branch version (shear + uniaxial `Cxxxx`/`Cyyyy`,
  CLI `real_im_energy.py 0 10`) was REPLACED — recover uniaxial branches from git
  history if needed.** It imports `from meshes import MakeMesh` (the master mesh).
  **Default frequency window `ln(ω·τ_M) ∈ [-3, 25]`, 140 pts** (`--lnmin -3`,
  `--lnmax 25`, `--npts 140`; all CLI-overridable).
- `real_im_energy_lowomega.py` — low-ω **tail** companion to `real_im_energy.py`:
  sweeps only `ln(ω·τ_M) ∈ [−6,−3)` (23 pts on the same grid step as the original
  `linspace(-3,10,100)`, no overlap) with **`solver='cg'`** (CG-everywhere
  directive; CG floors `C'` at low ω so `G_ss=G2²/G1` and Δ from this tail are
  **approximate** — accepted). Writes
  `Seed_*_..._shear_lowomega.csv` to be concatenated onto the existing `[−3,10]`
  CSVs — see the σ-ensemble pipeline under `csd3_deploy/sigmas/`
  (`merge_lowomega.py`, `make_ensemble_figs.py`, `overlay_all_sigma.py`).
  **NB (2026-06-15):** this companion's grid step (`h=13/99≈0.131`, matched to the
  OLD `linspace(-3,10,100)` σ-ensemble production grid) **no longer matches the new
  master-driver default** (`[-3,25]`, 140 pts, `h≈0.201`). The σ-ensemble CSVs and
  this tail pipeline were NOT regenerated; if you re-run the new master driver and
  want to attach low/high-ω tails, re-match the steps first.
- `real_im_energy_highomega.py` — high-ω **tail** companion: sweeps only
  `ln(ω·τ_M) ∈ (10,15]` (39 pts on the same grid step, no overlap) with
  **`solver='cg'`** (NOT direct — the unrelaxed plateau is well-conditioned and
  direct is far too slow for 39 solves/seed; CG floors only the *tiny low-ω* `C'`,
  which is irrelevant here). Writes `Seed_*_..._shear_highomega.csv`, merged onto
  the canonical CSV by `csd3_deploy/sigmas/merge_highomega.py` (junction-continuity
  guard on `C'`; `.prehigh` backup, independent of the low merge's `.premerge`).
  The fully-extended seed grid is then `ln(ω·τ_M) ∈ [−6.02, 15.12]` (162 pts).
- `real_im_hex_energy.py` — hexagonal benchmark RVE swept over frequency (same
  per-geometry `measure_tau_M` calibration as the seed driver). **Default window
  `ln(ω·τ_M) ∈ [-3, 25]`, 140 pts** (module constants `LN_OMEGA_MIN=-3`,
  `LN_OMEGA_MAX=25`, `OMEGA_SAMPLES=140`), matching the seed driver.
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
- `tess/` — Archimedean-tiling geometry toolkit + small-grain ratio-sweep
  packages (`84_sweep`, `12123_sweep`, `4612_sweep`). See the dedicated
  "tess/" section below.

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
- **Solver caveat:** the whole toolkit is now **`solver='cg'` everywhere by
  default** (CG-everywhere directive; `solve_rve` default flipped to `cg`,
  `maxiter=400`). At low ω `C'` is ~1e-18 and CG (`rtol=1e-8`) re-floors it at the
  residual level, so the `1/ω` divergence and Δ from the low-ω tail are
  **approximate**. Direct (Pardiso) is more accurate there but far too slow at
  production scale — pass `solver='direct'` manually only for a one-off low-ω
  accuracy check. (The drivers' forced `convergence=True` override and ω-retry
  loop were already removed.)

References: Raj & Ashby (1971) coupled model; Rudge (2025, App. A) the GB
diffusion law actually used; Ghahremani (1980) hex EAGBS limit; Lee & Morris
(2010) / Jackson et al. for diffusional creep in mantle olivine.

## Build/run

Run under the `ngsolve` conda env:
`/home/zl471/miniconda3/envs/ngsolve/bin/python`. Single sweep:

```bash
python real_im_energy.py --idx 0 --den 0   # geometry 0, no mesh refinement (shear, cg)
python real_im_energy.py --idx 0 --den 50  # refine_frac=1/50 on short facets
python real_im_energy.py --key seeds_3 --den 100   # select by key
python real_im_hex_energy.py 0.01          # hex benchmark with core_frac=0.01
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
after calibration the seed driver's default band `ln ω ∈ [-3,25]` (140 pts) sits
mostly on the solid side of the ω≈1 crossover and now reaches well up the high-ω
plateau — widen the low end (`--lnmin`) below -3 if the creep tail is wanted.
On tiny-facet geometries the η_ss solve stagnates under CG but its VALUE is
still correct — see "Calibration conditioning" in the tess/ section below.

## tess/ — Archimedean-tiling ratio sweeps (2026-06-10)

Controlled-geometry counterpart to the random σ-ensemble: three Archimedean
families with a tunable small grain, built to test the facet-length scaling of
the secondary DAGBS relaxation (see "high-ω peak mechanism" above). Pipeline per
family: tiling generator → `tess/cut_paste_rect.py` (rectangular periodic box,
no GB on the box edge) → faces-as-cells solver JSON (`tess/read_cut.py` pattern).
`tess/visualize_mesh.py` renders tessellation/FE-mesh/boundary-label figures
(fixed 2026-06-10: canonical 9-tuple `MakeMesh` unpack, `mesh.Elements(BND)`
+ `el.mat` for boundary sampling, CLI `[json] [key] [out_prefix]`).

**Sweep folders** `tess/{84,12123,4612}_sweep/`: `gen_sweep.py` writes ~10
geometries per family into one `tessellation_output.json` keyed `ratio_<r>` =
small/large boundary-length ratio, log-spaced down to 1/5000. Domain and avg
face area (1/50) are invariant along each sweep. Verification built into the
generator (raises on failure): exact target ratio measured on intact small
grains, exact area, CCW, periodic boundary pairing, all-vertices-in-box, and
exact-once point coverage (`check_box_tiling`).

Geometry gotchas learned the hard way:
- **`cut_and_paste_axis` is a plain translation, no periodic wrap** → the cut
  must lie in `[eps_above, dom − eps_below]` (the face protrusions beyond the
  `[0,dom]` window), else protruding content is stranded outside the box (gaps
  on one edge, overhang on the other). Area + boundary-pairing checks are BLIND
  to this defect (equal areas, no segments on the line); only point-coverage
  catches it. `find_cut` clamps to the valid window.
- Small-grain avoidance intervals must be wrapped **mod the domain** (the patch
  contains protruding duplicate faces that otherwise break periodicity of the
  forbidden set).
- **(4,6,12) at t=1/3 (r=1/3, the uniform tiling) cannot be box-cut cleanly**:
  the rectangles' x-intervals tile the period exactly (tangent bands), so every
  axis-aligned line slices one. The original `tess/tth_4612_cut.ply` silently
  sliced 2 of its 12 rectangles. The 4612 sweep therefore starts at r=1/4.
  (NB the generator's "squares" are 1:√3 rectangles — uniform kagome truncation,
  not the true Archimedean 4.6.12.)
- Smallest members (facet < 1e-4: 84 idx 8-9, 12123 idx 9) are EXCLUDED from
  production (mesh/conditioning cost); arrays run 0-7 (84) and 0-8 (12123).

**Status (2026-06-15):** 12123 sweep run on CSD3, pulled to
`tess/csd3_results/12123_sweep/`, and analysed (two-mode result above). 84
sweep NOT yet submitted (its physics was settled by calibration alone). 4612 is
geometry-only. The driver gained `--outtag` (CSV-name suffix) so follow-up runs
don't clobber production CSVs.

**Refine sweep packages** (refine_test machinery applied to 84_sweep and
12123_sweep only; 4612_sweep has just the geometries): each package folder is
self-contained — `meshes_refine.py` (verbatim from `csd3_deploy/refine_test/`;
per-facet proportional refinement `refine_frac`/`refine_cut`), local `main.py`
(**CG `maxiter=400`**, only diff from canonical besides the solver default),
`physics.py`, `calibrate_tau.py`, driver `real_im_energy_refine.py` (`--idx|--key`,
`--den`, CG-only, shear, incremental CSV `refine_<key>_frac<den>_shear.csv`),
and `run_refine.sh` (RUDGE-**SL2**-CPU, 8 cpus, 16 h, `den=50`,
`ln(ω·τ_M) ∈ [−3,18]`, 161 pts). Status 2026-06-10: locally validated; rsync +
sbatch commands handed to user, **submission not yet confirmed**.

**Calibration conditioning (important).** On tiny-facet meshes the η_ss
calibration solve (`code-ω=1`, diffusion coefficient exactly 1) is the ONE solve
where CG struggles: multigrid doesn't treat the GB multiplier blocks, whose
surface-Laplacian entries on facets with element size h ~ 1e-6 sit ~5 orders
above the elastic scale → CG on the indefinite saddle plateaus (residual
1e-3-1e-4; on the worst case it returned residual 2.29 = noise). `G_U`
(diff=0) and ALL band sweep solves (coeff τ_M/ω ≪ 1) converge to 1e-13.
**The stagnated calibration VALUES are nevertheless correct** — validated
three ways: cross-mesh CG agreement (12123 idx8 baseline vs refined: 4 digits),
Pardiso ground truth (12123: 0.17%; 84 idx7 across meshes: 2.4%), and the 84
η(L) law below. Even Pardiso residual-floors at ~1e-4 there (κ~1e12, machine-ε
limit) with a correct solution. Don't "fix" the alarming residual prints; do NOT
trust a calibration only when its value is unstable across meshes (the one real
failure was 84 idx7 BASELINE CG: residual 2.29, value 4× off). Pardiso was used
ONCE for this diagnostic with explicit user authorization (2026-06-10) —
production stays CG.

**Physics finding — the two families probe complementary regimes:**
- **(4,8,8) slip highways**: the long octagon edges are collinear across cells →
  free-sliding planes interrupted only by the diamonds → steady-state creep is
  rate-limited by diffusion around the diamonds alone:
  **η_ss = 0.039·L³** (L = diamond facet; exponent 3.000 over 5 calibrations
  spanning L = 8.3e-2 → 2.3e-4, η = 2.2e-5 → 4.6e-13, each within ~2% of the
  law). τ_M collapses with the diamonds (1.9e-5 → 2.0e-12) and G_U softens
  1.19 → 0.231. So in ω·τ_M units the 84 spectrum is diamond-controlled around
  the main crossover — no separated secondary peak.
- **(3,12,12)**: hex-like topology, no highways → τ_M set by the big grains and
  ~constant along the sweep (η_ss 2.4e-5 → 1.2e-4, τ_M ~1.5e-4 at idx8, G_U
  0.58 → 0.82 toward the hex value) → the shrinking triangles produce genuine
  secondary relaxations marching UP through the band.

**(3,12,12) RESULT (2026-06-15).** The production sweep (`run_refine.sh`, 9
geoms) + a wide-band follow-up (`run_wide.sh`: members 7-9 over ln∈[−3,24] +
member 9 at den=100) reveal **TWO distinct GB relaxations** above the main peak,
both mesh-converged (den=50 vs den=100 identical):
1. **Triangle's own Coble mode**: `ln(ω_peak·τ_M) = −3.09·ln L − 4.45`, i.e.
   ω_peak ∝ L^−3 over 7 points / 2.4 decades (L = 1.3e-4 → 3.3e-2). The
   controlled confirmation of the ensemble secondary peak (whose cross-seed
   slope was floor-compressed to −0.70 by unresolved slivers).
2. **Network "RC" mode**: slope **−0.98 ≈ −1** — a collective relaxation of the
   long-facet diffusion network charging *through* the tiny triangles as
   diffusive bottleneck resistors (R∝L, fixed C → ω∝L^−1). Per-facet
   attribution: 81% of its dissipation on the triangle facets, participation
   ratio 59 (collective). Only unmasks in members 8-9 once the L^−3 mode leaves
   the [−3,18] band; member 9 wide shows both at once (ln≈14.4 + ln≈23.0).
Scripts/figs in `tess/csd3_results/` (`analyze_12123.py`,
`analyze_wide_12123.py`, `attribute_12123.py`, `scaling_final_12123.py`;
`scaling_final_12123.png` has both laws). **Production caveat hit:** the
member-8 wide run's η_ss calibration stagnated to a τ_M ×110 off (the
documented CG stagnation is *non-deterministic* — same mesh, fine in
production). Recoverable by an x-axis shift `ln(τ_ref/τ_used)` since τ_M is a
pure frequency rescaling; always check each `_meta.txt` τ_M vs production before
trusting a tail's x-axis.

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
- **CG is the default everywhere** (`solve_rve` default = `cg`, `maxiter=400`;
  CG-everywhere directive). CG re-floors `C'` at low ω and hides the `1/ω`
  divergence, so the low-ω tail (`real_im_energy_lowomega.py`, also CG now) gives
  an **approximate** Δ — accepted (direct is too slow at scale). Calibration is
  robust to this — `η_ss`/`G_U` come from well-conditioned CG points. For a one-off
  high-accuracy low-ω check, pass `solver='direct'` manually.
- The legacy corner pin (`_add_corner_penalty`, `remove_rbm=False`) is kept only
  for comparison; it floors `C'`. Default is the stress-free RBM removal.
- **High-ω Q⁻¹ tails are not all asymptotic (σ-ensemble, open question).** In the
  per-σ `fig2_ens_allcurves_*.png` panel (b), most seeds settle to a log-log
  `Q⁻¹` asymptote of slope ≈ −1/3, but a minority depart at the top of the swept
  band (`ω·τ_M ≳ 100`). `make_ensemble_figs.py` now detects and colours two
  mutually-exclusive kinds (constants `TAIL_SLOPE_THR=-0.25`, `BUMP_PEAK_THR=0.10`,
  `BUMP_WT_MIN=100`): **crimson** = *flatten-and-stay* (last-4-pt slope > −0.25,
  tail flattens toward 0 and never recovers) and **forest green** = *bump-and-
  return* (a pronounced hump above the ensemble median that then re-steepens back
  to the asymptote). Both are dumped per σ to `qinv_nonasymptotic_seeds_<σ>.csv`
  (cols `seed,category,tail_slope,peak_excess`). Both worsen at high σ → likely a
  high-ω **mesh-resolution limit** (the diffusion boundary layer ~√(D/ω) goes
  unresolved at sharp/distorted junctions; cf. `refine_h`). **Working hypothesis
  (untested):** the crimson flatten-and-stay tails are just a bump-and-return hump
  whose recovery falls beyond `ln(ω·τ_M)=10` — i.e. the same phenomenon, an
  "incomplete hump".
  **RESOLVED (band extended to ln(ω·τ_M)=15.12 via `real_im_energy_highomega.py`
  + `merge_highomega.py`).** Tested directly: of the 62 seeds (across all σ) that
  "flatten-and-stay" by ln=10 (slope > −0.25 over ln∈[9.3,10]), **61 (98%) turn
  back down in the new (10,15] decades** (new-band slope < −0.10, recovering toward
  the −1/3 asymptote; e.g. σ=0.5 seeds steepen from ≈−0.1 to ≈−0.4). So the crimson
  flatten-and-stay and green bump-and-return are **the same phenomenon** — a hump
  whose recovery simply fell beyond the old ln=10 cutoff. It was a
  **frequency-window artifact, NOT a mesh-resolution failure**. The high-ω tail
  used `solver='cg'` (fine on the well-conditioned plateau). The two-class
  detection in `make_ensemble_figs.py` is now mostly moot on the extended grid (the
  classifier's last-4-pt slope is measured at ln≈15, where nearly all seeds are
  asymptotic).
  **MECHANISM (identified 2026-06-09).** The high-ω hump is a *second DAGBS
  relaxation: the diffusion-accommodated sliding of the single shortest grain-
  boundary facet* (a "sliver" at a near-degenerate triple junction). Relaxation
  frequency is set by facet length L (diffusion path): long facets → the main peak
  at ω·τ_M≈1, the shortest sliver → the high-ω peak (ω_peak ∝ L^{-2..-3}). It is a
  real peak in the **loss modulus C''**, not just the `Q⁻¹` ratio, and the regular
  hex (single facet length) has **no** second peak. The two peaks have opposite
  character (seed 24, σ=0.45): the main peak is *collective* (≈41 facets
  participate, top facet 8%); the secondary peak is *localized* (participation ≈1,
  **96% of its dissipation on one facet**, L≈1e-3 vs mean 7e-2). Generalizes across
  the 988-seed ensemble: **28 seeds (3%) show the peak, and every one has an extreme
  sliver** (median L_min 5.6e-4 vs 1.5e-3 for non-peaked); corr(ln L_min, peak_ln)
  = −0.53. The cross-seed slope is shallow (−0.70) only because sub-`maxh` slivers
  hit the **mesh-resolution floor** (peak_ln pins ~12–14 regardless of true L_min).
  Convergence test of this (does the peak follow L^{-3} when the sliver is resolved
  = physical, vs vanish = artifact) lives in **`csd3_deploy/refine_test/`**
  (`meshes_refine.py` adds per-facet proportional refinement `refine_frac`/
  `refine_cut`; `real_im_energy_refine.py` + `run_refine.sh` re-sweep ln∈[4,18] for
  seed 24 + σ=0.30/seed 96 at L/30,L/100,L/150, CG, SL3).
  **Convergence RESULT (2026-06-09).** Seed 24's strong peak (Q⁻¹≈0.0335) is
  **mesh-CONVERGED** (baseline + all 3 refined curves indistinguishable, peak fixed at
  ln≈11.5, height unchanged from 1→165 elements on the sliver) → **physical**; it does
  NOT climb because ω_peak∝L⁻³ is set by the *geometric* facet length, captured at any
  resolution.
  **Seed 96 was a MERGE ARTIFACT, not a relaxation (corrected).** Its "peak" sat exactly
  at the base|high-ω **glue point ln=10.13**: the production CSV's high-ω tail is stitched
  on ~**+15% high** relative to where the base sweep ends (Q⁻¹ 0.0112→0.0129 across the
  join — a step that slipped through `merge_highomega.py`'s continuity guard, ratio 1.15 <
  1.25 tol), and the local-max detector misread the step as a peak. The continuous refined
  sweep (no merge) is monotonic. So this is a **CSV-stitching discontinuity**, NOT
  under-resolution (earlier note here was wrong). The step is NOT universal — most
  tails are already continuous; only a minority (e.g. seed96 σ0.3) had a real
  base↔high τ_M-mismatch step.
  **FIXED (2026-06-09): `merge_highomega.py` now re-levels each high-ω tail.** It
  log-linearly extrapolates the base's last 5 points of C' and C'' to the junction and
  multiplies the whole tail to land on that trend (uniform scale preserves genuine bumps;
  removes the glue step). Median re-level factor across all σ = 1.001 (range mostly
  [0.99,1.01], a few to 1.33; seed96 σ0.3 = 0.87). A **gross** factor (outside [0.5,2.0])
  = broken dissipation solve → refused/quarantined as `.bad` (σ0.3 **seed61**, factor
  8.17 — reverts to base-only). Backups: `.prehigh`; the merge restores from it so it is
  re-runnable. **Re-leveled census: 26 genuine high-ω peaks** (the 2 ln=10.13 junction
  artifacts gone), peaked-seed median L_min 5.6e-4 vs 1.5e-3 non-peaked, corr(ln L_min,
  peak_ln)=−0.60. Per-σ figs + ensemble overlay + survey re-run on the corrected CSVs
  (`make_ensemble_figs.py`, `overlay_all_sigma.py`; survey table `/tmp/survey_relevel.csv`).
  Figs `refine_test/refine_convergence.png`, `refine_test/merge_artifact_census.png`.
  Mechanism figures in
  `csd3_deploy/vtu_movie/` (`highf_mechanism.png`, `per_gb_dissipation.png`,
  `sliver_geometry_dissipation.png`, `peaked_seeds_gallery.png`,
  `peaked_seeds_scaling.png`).
