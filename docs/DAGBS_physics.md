# DAGBS physics — key equations and reflections

Working note distilling the **diffusion-accommodated grain-boundary sliding
(DAGBS)** model that the solver is being extended toward. This is the physics
companion to `CLAUDE.md` (which covers the code map). Read both before touching
`_add_gb_terms`.

> **Status (2026-06-01):** **DAGBS is fully implemented and validated.** Free
> tangential sliding + diffusive normal channel, with the per-edge pressure field
> `t_n` connected **through triple junctions** (exact-Lagrange reservoir coupling,
> §6.2) and rigid-body modes removed **stress-free** (§7). The hex benchmark now
> shows the steady-state Coble-creep signature: `C' ∝ ω²`, `C'' ∝ ω`, and
> `Q⁻¹ = C''/C' ∝ 1/ω` diverging at low frequency. The derivation below is the
> model in the code; a few places where the original target note differs from the
> final implementation are called out inline.

---

## 1. Regime and the core idea

We are in the **Coble-creep** regime: matter diffuses *along* the grain boundary,
driven by gradients of the normal stress, and this diffusion lets neighbouring
grains separate/interpenetrate (normal opening) without bulk diffusion.

Two separable timescales are assumed:

- **Tangential sliding** of grain boundaries is *fast* (effectively
  instantaneous on the diffusion timescale) → boundaries slide **freely**.
- **Normal opening** is *slow*, rate-limited by grain-boundary diffusion.

These are orders of magnitude apart, so we decouple them: free shear + diffusive
normal channel. The grain interior stays **linear-elastic** (complex modulus in
the frequency domain).

### Contrast with EAGBS (what actually changes)

| Channel | EAGBS (in code now) | DAGBS (target) |
|---|---|---|
| Tangential | viscous: `Δu_t = t_s / (iωη)` | **free**: `σ_ns = t_s = 0` (no constraint) |
| Normal | **locked**: `Δu_n = 0` (multiplier `t_n`) | **diffusive**: `∂²σ_nn/∂s² ∝ ∂u_n/∂t` |

So the two constraint roles essentially **swap**: EAGBS locks the normal and
relaxes the tangent; DAGBS frees the tangent and relaxes the normal. The
`core`/`slide` stub machinery and the "multiply-by-`i`" real/imag trick carry
over, but the *content* of the GB block is different.

---

## 2. Governing equations

**Grain interior** (each `region_i`): nondimensional linear elasticity
(eq. `nondim ela` in the manuscript), complex-valued.

**On every grain-boundary segment `S`:**

Free tangential sliding:
```
σ_ns = 0                                                      (shear)
```

Diffusive normal channel (Rudge 2025, Appendix A; cf. Raj–Ashby). The primitive
statement is the mass balance for material plated onto the boundary by GB
diffusion:
```
[u̇ · n] + Ω δ D^gb_v ∇²⊥ c = 0                               (A4)
```
`[·]` is the jump across the boundary, `u̇ = ∂u/∂t` the velocity, `n` the unit
normal, so `[u̇·n]` is the **rate at which new material is plated out** on the
boundary. `D^gb_v` is the *vacancy* diffusivity along the boundary, `δ` the GB
width, `Ω` the atomic volume, `c` the vacancy concentration, and `∇²⊥` is the
**surface Laplacian** (gradient/Laplacian restricted to the boundary).

Vacancy concentration is tied to the normal traction by **Herring's relation**:
```
c = c₀ (1 + Ω p / kT),     p ≡ n · σ · n   on S              (A5,A6)
```
`c₀` is the equilibrium vacancy concentration, `k` Boltzmann's constant, `p` the
normal traction (`= σ_nn = t_n`). Substituting (A5) into (A4) and defining the
**self-diffusion coefficient** `D^gb = D^gb_v c₀` gives the working form:
```
[u̇ · n] + (Ω δ D^gb / kT) ∇²⊥ p = 0                          (A7)
```
i.e.
```
∇²⊥ σ_nn = − (kT)/(Ω δ D^gb) · [∂u_n/∂t]                      (normal)
```

Free tangential sliding has already been imposed above (`σ_ns = 0`), so the
boundary carries only this normal diffusive channel plus the shear-free
condition.

> **2D reduction (`s` vs `n`).** Along a grain-boundary segment in 2D the surface
> Laplacian `∇²⊥` is just the second derivative w.r.t. **arc length `s`**:
> `∇²⊥ σ_nn = ∂²σ_nn/∂s²`. The source LaTeX writes `∂²σ_nn/∂n²` and `∂t_n/∂n` —
> these should read `∂²σ_nn/∂s²` and `∂t_n/∂s`. The operator is *tangential*
> (diffusion runs *along* the boundary), the IBP in the weak form is *along*
> `Γ_GB`, and `CLAUDE.md` already says "`t_n` proportional to `∂²μ/∂s²`".
> **Treat every `∂/∂n` on the diffusion term as `∂/∂s`** — and fix it in the
> manuscript. (Using `∇²⊥` directly is cleaner still, and generalises to curved
> boundaries.)

---

## 3. Frequency domain and the `i/ω` factor

With `e^{iωt}`, `∂u_n/∂t → iω u_n`. Substituting into the normal law and
writing `t_n` for the unknown normal traction multiplier:
```
∂²t_n/∂s² = − (kT)/(Ω δ D^gb) · iω · u_n
```
Multiply by a test function `r_n`, integrate over `Γ_GB`, integrate by parts in
`s`:
```
− ∫ ∂t_n/∂s · ∂r_n/∂s  dS  + [boundary terms]
       = − (iω kT)/(Ω δ D^gb) ∫ u_n r_n dS
```
Rearranged (and using `1/i = −i`, with the sign absorbed by the complex-conjugate
test convention `σ*` used throughout):
```
  (i Ω δ D^gb)/(ω kT) ∫ ∂t_n*/∂s · ∂r_n/∂s dS  =  ∫ [u_n]* r_n dS
```
This is exactly the new GB block. Note the **single place ω enters the bilinear
form** is this prefactor `(i Ω δ D^gb)/(ω kT)` — directly analogous to EAGBS's
`1/(iωη)` on the tangential term. Same "stiffness ∝ 1/ω" structure, different
physical channel.

---

## 4. Weak form (full)

Per grain, test elasticity against `v ∈ [H¹]²`, Green's theorem, and assemble
the shared-interface traction terms (`n_j = −n_i`, `[v] = v_i − v_j`,
`t_n = n·σ·n`, `t_s = s·σ·n`):

```
  Σ_i ∫_{Ω_i} σ*(u) : ε(v) dV
    − ∫_{Γ_GB} [v_n] t_n* dS                         ← normal traction work
    − ∫_{Γ_GB} [u_n]* r_n dS                          ← normal kinematic constraint
    + (iΩδD^gb)/(ωkT) ∫_{Γ_GB} ∂t_n*/∂s · ∂r_n/∂s dS  ← diffusion (NEW)
    + (RVE boundary terms)
  = 0
```

**What is absent vs EAGBS:** there is **no `t_s` / `[v_s]` tangential block** on
the interior `Γ_GB` — free sliding means `t_s ≡ 0`, so the tangential jump is
left unconstrained. (Contrast EAGBS, which carries a tangential multiplier with
the `1/(iωη)` viscous law.)

### RVE (periodic) boundaries

Macro/local split `u_total = U + u`. Following Rudge's micropolar model
(`rudge_micropolar_2021`, `rudge_viscoelastic_2025`) but **upscaled to a Cauchy
continuum**: drop granular rotation (`K = 0`), keep `Γ` symmetric (= macroscopic
strain-rate tensor), so the jump across a periodic pair is
```
[U] = Γ · R
```
with `R` joining neighbouring-RVE centroids. The periodic faces
(`Γ_top ∪ Γ_right`) contribute both tangential and normal multiplier blocks
(`t_s, t_n` / `r_s, r_n`) — note **both** components appear here even though the
interior GBs lost their tangential block — and the macro strain enters as the RHS:
```
  RHS = − ∫_{top∪right} (Γ*·R·n) r_n dS  − ∫_{top∪right} (Γ*·R·s) r_s dS
```

---

## 5. Rescaling — Maxwell time

Natural timescale:
```
τ_M = η / G_0
```
`η` steady-state (Coble) viscosity, `G_0` unrelaxed shear modulus. Nondimensional
frequency `ω̃ = ω τ_M`. Because the steady-state Coble viscosity itself scales as
`η ∼ (kT/(Ω δ D^gb)) · L³` (with `L` a grain-size/length scale), the diffusion
prefactor `(Ω δ D^gb)/(ω kT)` collapses to a clean `∼ 1/ω̃` group once lengths are
measured in grain sizes. **Practical upshot:** after nondimensionalisation the
DAGBS GB block should look like `i/ω̃ · ∫ ∂t_n/∂s ∂r_n/∂s`, mirroring the EAGBS
`i/ω̃ · ∫ t_s r_s` block — only the operator changes (surface Laplacian vs mass).

**Implemented (2026-06-02):** rather than assume the prefactor, it is *measured*
per geometry as `τ_M = η_ss/G_U` (`calibrate_tau.measure_tau_M`) and folded into
the form via `diff_coeff=τ_M`, so code-ω becomes `ω̃ = ω·τ_M` directly. See §7.

---

## 6. Discretisation / implementation implications

Consequences for the FE code (`main.py`, `meshes.py`):

1. **`t_n` now needs tangential regularity.** The diffusion term contains
   `∂t_n/∂s`, so the normal-traction multiplier must live in an `H¹`-along-the-
   boundary space *per segment*, not a per-segment L²/constant. This is the
   biggest structural change from EAGBS, where multipliers needed no surface
   derivative. `build_spaces` and the `name_order_trial` arity check must be
   updated in lockstep (see `CLAUDE.md`).
2. **Triple-junction coupling (implemented — see §6.2).** Leaving the segment-end
   terms as natural BCs gives `∂t_n/∂s = 0` = **zero diffusive flux at triple
   junctions** (impermeable junctions). That is the Raj–Ashby *isolated-boundary*
   model, and it is a **standard-linear-solid**: no through-junction transport →
   no steady-state creep → `Q⁻¹ → 0` at low ω (no divergence). To get Coble creep
   we instead **connect the boundaries at every junction** (next section).
3. **Interior tangential constraint dropped (done).** Interior GBs carry no
   `t_s` block (free sliding) — `_add_gb_normal_coupling` adds only the normal
   `t_n` terms. `t_s`/`r_s` survive only on the periodic RVE faces
   (`_add_outer_terms`).
4. **Two parallel relaxation channels.** If a model with *both* viscous shear and
   diffusional normal opening is ever wanted, the two GB blocks simply add — they
   are independent paths. Pure DAGBS = diffusion block only.
5. **Dissipation post-processing.** The driver's loss term is the diffusional
   `∼ (1/ω) ∫ |∂ₛt_n|²` (energy dissipated by GB diffusion ∝ flux² = `(∂ₛp)²`),
   replacing the EAGBS `∼ η/ω ∫_slide |t_s|²`. See the `real_im_energy.py`
   storage/loss split.

### 6.2 Triple-junction coupling (the key to Coble creep)

Per-edge `t_n` fields with zero-flux ends are diffusively **isolated** — matter
cannot move from one boundary to another, so no grain can change shape: a
standard-linear-solid (finite relaxed modulus, no low-ω divergence). The FE mesh
is **non-conforming** at grain boundaries (left/right traces are distinct DOFs;
each triple junction is 3 distinct mesh vertices), so a single `H1` field over the
boundary skeleton does **not** auto-connect at junctions. We therefore connect
them with an **exact-Lagrange junction coupling** (`main._add_gb_junction_coupling`,
multipliers built in `build_spaces`):

- one **reservoir potential** `μ_J` per triple junction (the junction chemical
  potential), and one **flux multiplier** `λ_{e,J}` per incident edge — all
  complex `NumberSpace` scalars;
- a **mortar constraint** over each edge's right-side `core` stub at `J`,
  `∫_stub ξ (t_n^{(e)} − μ_J) = 0`, enforces continuity `t_n^{(e)}|_J = μ_J`
  (stub-averaged), and stationarity w.r.t. `μ_J` gives `Σ_e λ_{e,J} = 0`
  (Kirchhoff flux balance — `λ` is the diffusive flux into the junction).

This makes the per-edge fields behave as one continuous network field, enabling
through-junction transport ⇒ steady-state Coble creep ⇒ `C' → 0`, `Q⁻¹ ∝ 1/ω`.
Implementation detail: `meshes.py` names core stubs per junction
(`core_i_j_lr_jJID`) and returns a `junction_incidence` map; consumers select
cores with a `core_i_j_lr.*` regex (NGSolve `Boundaries` is full-match regex).

### 6.3 Stress-free rigid-body-mode removal

The old Nitsche corner pin (pinning corner `u` to the affine field) injects a
fixed, non-relaxing bulk stress near the corner that **floors the storage
modulus** (`C'` floored at ~5.6e-8 instead of → 0), turning the spectrum into a
Debye peak and hiding the divergence — confirmed by a `gammaN` sweep (the floor
scales with the pin strength; `C''` is invariant). It is replaced
(`main._add_rbm_constraints`, default `remove_rbm=True`) by **three global
integral constraints** removing the rigid-body nullspace without any local pin:

```
∫_Ω u_x dΩ = ∫_Ω (Γx)_x dΩ ,   ∫_Ω u_y dΩ = ∫_Ω (Γx)_y dΩ   (translations)
∫_Ω (∂ₓu_y − ∂_y u_x) dΩ = (Γ_yx − Γ_xy)·|Ω|                  (rotation)
```

applied to both Re and Im fields (6 scalar Lagrange multipliers). **Both
translation and rotation must be removed.** The rotation target matches the
*affine* rotation (simple shear `Γ=((0,1),(0,0))` carries a rotation component —
it is **not** forced to zero). No stress injected ⇒ `C'` relaxes to its true
value (~0 for the flowing limit). The legacy pin is still reachable with
`remove_rbm=False` for comparison.

---

## 7. Status / open questions

**DAGBS is implemented and the low-frequency attenuation divergence is achieved.**
On the hex benchmark (shear, direct solver), `C' ∝ ω²`, `C'' ∝ ω`,
`Q⁻¹ ∝ 1/ω` (slope −1.000 across ln ω ∈ [−10, 10]). Surface gradients use
`Grad(t_n).Trace()` (plain `Grad` on a boundary-restricted H1 field raises
"Trialfunction does not support BND-forms").

- [x] **`s` not `n`** — surface gradient `Grad(...).Trace()` (PSD Laplace–Beltrami,
      constant nullspace). Manuscript still needs the `∂n → ∂s` fix.
- [x] **Through-junction transport** — exact-Lagrange junction coupling (§6.2);
      this, not the zero-flux natural BC, is what enables Coble creep.
- [x] **Stress-free RBM removal** — §6.3; replaces the corner pin (which floored
      `C'`). Both translation and rotation removed.
- [x] **i/ω sign** — confirmed correct: `C'' > 0`, `C' → 0` as `ω → 0`, and
      `Q⁻¹ ∝ 1/ω` (the flowing/Maxwell direction). No swap flip needed.
- [x] **Nondimensional prefactor / Maxwell time** — calibrated per geometry
      (2026-06-02). The diffusion coefficient `C_d` is now a `diff_coeff`
      argument (default `physics.DIFF_COEFF=1.0`). `calibrate_tau.measure_tau_M`
      measures `τ_M = η_ss/G_U` (`G_U` from the `diff_coeff=0` normal-locked limit,
      `η_ss=C''/ω` from the `ln ω=-10` creep plateau) and the drivers sweep with
      `diff_coeff=τ_M`, so code-ω reads `ω·τ_M` and the crossover sits at ω≈1.
      Hex: `η_ss=2.418e-5`, `G_U=0.821`, `τ_M=2.944e-5`. **Still nondimensional** —
      not yet anchored to a physical `η_Coble = (kT/ΩδD^gb)·L³`.
- [ ] **Solver for low-ω points** — `C'` reaches ~1e-18; only `solver='direct'`
      resolves it. The CG path (`rtol=1e-8`) re-floors `C'` and re-hides the
      divergence. The drivers default to `solver='cg'`; pass `solver='direct'`
      for the low-ω tail (the old forced `convergence=True` override was removed).
- [ ] **Benchmark target** — is there a published Raj–Ashby/Coble curve (or
      analytical limit) to cross-check the hex `Q⁻¹ ∝ 1/ω` magnitude against?

## References

- Raj & Ashby (1971) — original coupled GB sliding + diffusion model.
- Ghahremani (1980) — hex-grain analytical limit (current EAGBS benchmark).
- Rudge (2021, 2025) — micropolar / viscoelastic RVE homogenisation. The GB
  diffusion law used here is **Rudge 2025, Appendix A, eqs. (A4)–(A7)**
  (`rspa.2025.0606`); it is also the source of the periodic macro-strain
  decomposition (§4).
- Herring (1950) — relation between vacancy concentration and normal traction
  (eq. A5).
- Lee & Morris (2010); Jackson et al. — diffusional creep in mantle olivine
  (target application).
