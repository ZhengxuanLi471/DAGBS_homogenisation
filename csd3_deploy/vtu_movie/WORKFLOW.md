# vtu_movie — field export, movies & mechanism figures

Visualises the spatial fields behind the secondary high-ω relaxation: where the
dissipation concentrates, and how it differs between the collective main peak and
the localized sliver peak.

## Files
- `vtu_run.py` (CSD3) — per-frequency VTU export of `disp_real`, `disp_imag`,
  `storage_energy_density`, `dissipation_energy_density` over the full composite
  grid for σ=0.45 seed 24 and the hex reference. The GB dissipation field is
  baked into a zero volume field; `subdivision=0`. Imports the solver via
  `sys.path`; reads `tessellation_output.json` from `../sigmas/sigma_0.45/`.
- `run_vtu.sh` — single SLURM job (icelake, 8 cpus).
- `vtu_movie.py` (local) — meshio + matplotlib → GIF (no ParaView).
- `line_profiles_movie.py` — 1-D field profiles across the sliver vs frequency.

## Run
```bash
sbatch run_vtu.sh                     # on CSD3 → vtu_out/{seed24,hex}/
python vtu_movie.py                   # local → *.gif
```

## Outputs (git-ignored)
VTUs in `vtu_out/`, GIFs, and the mechanism figures referenced by the write-up:
`highf_mechanism.png`, `per_gb_dissipation.png`, `sliver_geometry_dissipation.png`,
`peaked_seeds_gallery.png`, `peaked_seeds_scaling.png`. Key finding: ~96% of the
secondary-peak dissipation sits on a single shortest facet, versus the collective
(~40-facet) main peak.
