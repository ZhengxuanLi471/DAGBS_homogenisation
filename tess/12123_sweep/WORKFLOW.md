# (3,12,12) ratio sweep

Controlled-geometry counterpart to the random σ-ensemble: a truncated-hexagonal
**(3,12,12)** tiling with one tunable shrinking triangle, used to extract the
**facet-length scaling** of the secondary DAGBS relaxation cleanly (random seeds
confound τ_M, topology and facet length together).

## Files
- `gen_sweep.py` → `tessellation_output.json` — ~9 geometries keyed `ratio_<r>`
  (small/large boundary-length ratio, log-spaced down to 1/5000); domain and mean
  grain area held fixed. Self-verifying (raises on bad area/pairing/coverage).
- `meshes_refine.py`, `main.py`, `physics.py`, `calibrate_tau.py` — local solver
  copies (CG, `main.py` uses `maxiter=400` for tiny-facet conditioning).
- `real_im_energy_refine.py` — shear, CG, per-facet refinement; selects a
  geometry by `--idx`/`--key`, refinement by `--den`; incremental CSV.
- `real_im_energy_fixedtau.py` — variant pinning τ_M (x-axis recovery for the
  rare calibration-stagnation case).
- `run_refine.sh` (array `0-8`, den=50, SL2, 16 h), `run_wide.sh` /
  `run_member8_wide.sh` — wide-band follow-ups (`ln ∈ [-3,24]`) for members 7–9.

## Run
```bash
python gen_sweep.py                                   # build the geometries
sbatch --export=DEN=50 run_refine.sh                  # production sweep on CSD3
```
Pull the CSVs to `../csd3_results/` for analysis.

## Result
Two mesh-converged GB relaxations above the main peak: a **triangle Coble mode**
`ω_peak ∝ L⁻³` (slope −3.09) and a collective **network "RC" mode** `∝ L⁻¹`
(slope −0.99). The L⁻³ mode is the controlled confirmation of the ensemble bump.
Check each tail's `_meta.txt` τ_M against production before trusting its x-axis
(CG calibration stagnation is non-deterministic).
