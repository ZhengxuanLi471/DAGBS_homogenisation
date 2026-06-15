# refine_test — is the high-ω peak physical?

Convergence test deciding whether the secondary high-ω loss peak found in the
σ-ensemble is a **genuine relaxation of a short grain-boundary facet** or an
**under-resolution artefact**. Criterion: a peak set by the geometric facet
length `L` is insensitive to mesh density once `L` is resolved; an artefact moves
or weakens.

## Files
- `meshes_refine.py` — edited copy of `meshes.py` adding **per-facet proportional
  refinement**: interior facets with `L < refine_cut` get element size
  `L·refine_frac` on their `core`+`slide` edges (resolves sub-`maxh` slivers).
  `refine_frac=None` ⇒ unchanged.
- `real_im_energy_refine.py` — re-sweeps one seed's high-ω band on the refined
  mesh; shear only, `solver='cg'`, CSV written incrementally. Imports
  `main/physics/calibrate_tau` unmodified from a `../sigmas/sigma_*` dir;
  selects the seed by `--seed`/`--sigma`, refinement by `--den` (`refine_frac=1/den`).
- `run_refine.sh` — SLURM array (one task per refinement level), 8 cpus, icelake.

## Run
```bash
sbatch --export=SEED=24,SIGMA=0.45,DEN=30  run_refine.sh   # and DEN=100, 150
```
Compare the baseline (production, re-leveled) curve against L/30, L/100, L/150.

## Result
Seed 24 (σ=0.45): baseline and all refined curves indistinguishable, peak fixed
at `ln(ω·τ_M) ≈ 11.6`, height unchanged from 1 → ~165 elements on the sliver ⇒
**mesh-converged ⇒ physical**. (One apparent peak, seed 96, was instead a
CSV-merge stitching step at the base|high-ω join, fixed by re-levelling in
`../sigmas/merge_highomega.py`.) Figures: `refine_convergence.png`,
`merge_artifact_census.png`.
