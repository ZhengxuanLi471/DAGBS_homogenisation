# tess/ — Archimedean-tiling geometry toolkit & ratio-sweep studies

File map only. The physics and gotchas live in the repo `CLAUDE.md` "tess/"
section; read that before editing generators.

## Tools (root)
- `gen_hex_seeds.py` — (4,6,12) truncated-trihexagonal generator (kagome
  truncation). Imported by `4612_sweep/gen_sweep.py`, so it must stay at root.
- `cut_paste_rect.py` — re-cut a periodic tiling into a rectangular periodic box
  with no grain boundary on the box edge.
- `read.py` / `read_cut.py` — PLY tessellation → solver `tessellation_output.json`
  (faces-as-cells). `read_cut.py` produces the 3-geometry demo json at root.
- `visualize_ply.py` — render a tessellation PLY to PNG.
- `visualize_mesh.py` — build the NGSolve mesh for a geometry and render
  tessellation / FE-mesh / boundary-label figures. Usage:
  `visualize_mesh.py [json] [key] [out_prefix]` (writes `<prefix>_mesh_*.png` to cwd).

## Per-family generators (one-off demo geometries)
- `84/` — (4,8,8) truncated-square: `gen_trunc_sq.py` + cut PLYs.
- `12123/` — (3,12,12) truncated-hexagonal: `gen12123.py` + cut PLYs.
- (4,6,12) is generated at root via `gen_hex_seeds.py` (+ `tth_4612_*.ply`).
- `tessellation_output.json` (root) = the 3-geometry demo set from `read_cut.py`.
- `demo_figures/` — rendered PNGs of the demo geometries and their meshes.

## Ratio-sweep packages (the studies)
- `84_sweep/`, `12123_sweep/`, `4612_sweep/` — each `gen_sweep.py` writes one
  `tessellation_output.json` of ~10 geometries keyed `ratio_<r>` (small/large
  GB-length ratio, log-spaced down to 1/5000).
- `84_sweep/` and `12123_sweep/` are **self-contained CSD3 run packages**: local
  copies of `main.py` (CG maxiter=400), `physics.py`, `calibrate_tau.py`,
  `meshes_refine.py`, the driver `real_im_energy_refine.py`
  (+ `real_im_energy_fixedtau.py` for hardcoded τ_M), and `run_*.sh` SLURM
  scripts. Do not strip these — they are rsynced to CSD3 as a unit.
  (`4612_sweep/` is geometry-only.)

## Analysis & results
- `csd3_results/` — analysis scripts + figures + the report:
  `analyze_12123.py`, `analyze_wide_12123.py`, `scaling_final_12123.py`,
  `attribute_12123.py`, `build_report_12123.py` → `report_12123.html`.
- `csd3_results/<sweep>/` — CSVs pulled back from CSD3 (`refine_ratio_*_shear.csv`
  + `_meta.txt`); `slurm_logs/` holds the job stdout.
