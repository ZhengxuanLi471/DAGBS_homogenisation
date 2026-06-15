# (4,6,12) ratio sweep — geometry only

Truncated-trihexagonal **(4,6,12)** family (kagome truncation; the generator's
"squares" are 1:√3 rectangles). Geometry generation only — no frequency sweep run.

## Files
- `gen_sweep.py` → `tessellation_output.json` — geometries keyed `ratio_<r>`,
  built on `../gen_hex_seeds.py` (kept at `tess/` root) + `../cut_paste_rect.py`.
  Self-verifying.

## Run
```bash
python gen_sweep.py        # → tessellation_output.json + sweep_overview.png
```

## Caveat
The uniform tiling at `t = 1/3` (`r = 1/3`) **cannot be box-cut cleanly** — every
axis-aligned line slices a rectangle — so the sweep starts at `r = 1/4`. To run a
frequency sweep, copy the `12123_sweep/` solver package (`main`, `meshes_refine`,
`physics`, `calibrate_tau`, `real_im_energy_refine`, `run_refine.sh`) alongside
this `tessellation_output.json`.
