# (4,8,8) ratio sweep

Truncated-square **(4,8,8)** tiling with one tunable shrinking grain — the
structured analog of the `12123_sweep` study, built with the same machinery.

## Files
- `gen_sweep.py` → `tessellation_output.json` — geometries keyed `ratio_<r>`
  (log-spaced ratio, fixed domain and mean grain area; self-verifying).
- `meshes_refine.py`, `main.py`, `physics.py`, `calibrate_tau.py` — local solver
  copies (CG).
- `real_im_energy_refine.py` — shear, CG, per-facet refinement (`--idx`/`--key`,
  `--den`); incremental CSV.
- `run_refine.sh` — SLURM array, 8 cpus, icelake.

## Run
```bash
python gen_sweep.py
sbatch --export=DEN=50 run_refine.sh
```

## Status
Geometry + per-geometry **calibration only** — the full frequency sweep was not
submitted (its physics is settled by the steady-state viscosity vs facet-length
calibration, with no separated secondary peak in `ω·τ_M` units). Locally
validated; figure `sweep_overview.png`.
