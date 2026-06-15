# csd3_deploy/ — CSD3 cluster staging tree

Self-contained copies of the DAGBS solver plus the cluster experiments, laid out
to be rsync'd to `~/rds/` on CSD3 (the relative layout here is preserved there,
so cross-directory imports resolve). Solver runs use `solver='cg'` throughout and
the `ngsolve` conda env (`/home/zl471/miniconda3/envs/ngsolve/bin/python`); SLURM
jobs request 8 cpus on `icelake` (RUDGE-SL2/SL3-CPU accounts).

> The canonical, up-to-date solver lives at the **repo root**. The copies under
> here are deployment snapshots and may lag the root.

## Contents
- `DAGBS/` — the solver template (one clean copy of `main/meshes/physics/
  calibrate_tau/real_im_energy*` + reproduce scripts) that the experiment dirs
  are cloned from.
- `sigmas/` — **random σ-ensemble experiment** (the main study). See
  `sigmas/WORKFLOW.md`.
- `refine_test/` — **mesh-refinement convergence test** for the secondary high-ω
  loss peak. See `refine_test/WORKFLOW.md`.
- `vtu_movie/` — **field export + movies / mechanism figures**. See
  `vtu_movie/WORKFLOW.md`.

Outputs (CSVs, VTUs, GIFs, figures, `.premerge`/`.prehigh` backups) are
git-ignored; only code, run scripts, input `tessellation_output.json`, and these
workflow notes are tracked.
