# σ-ensemble experiment

Frequency-domain DAGBS response of **random periodic polycrystals** as a function
of grain-size disorder. Goal: ensemble-average `C*(ω)` per disorder level and
characterise the **secondary high-ω loss peak** ("bump") absent in the regular
hexagon.

## Layout
`sigma_<σ>/` for σ ∈ {0.05, 0.10, …, 0.50} (std-dev of the log-normal grain-size
distribution). Each holds its own solver copy, a `tessellation_output.json`
(≈100 random seeds, keys `seeds_<n>`, generated externally with **Neper**), and
SLURM scripts. Analysis/merge scripts live once at this `sigmas/` level.

## Pipeline (order matters)
1. **Main sweep** — `run_seeds.sh` (SLURM array `0-99`) → `real_im_energy.py` →
   `Seed_*_…_shear.csv` (+ uniaxial), band `ln(ω·τ_M) ∈ [-3,10]`, per-seed τ_M.
2. **Low-ω tail** — `run_seeds_lowomega.sh` → `real_im_energy_lowomega.py` →
   `*_shear_lowomega.csv` (`ln ∈ [-6,-3)`).
3. **High-ω tail** — `run_seeds_highomega.sh` → `real_im_energy_highomega.py` →
   `*_shear_highomega.csv` (`ln ∈ (10,15]`).
4. **Merge** — `merge_lowomega.py` then `merge_highomega.py` (attach tails onto
   each base CSV; `.premerge`/`.prehigh` backups; continuity guard re-levels each
   high-ω tail at the join). Re-runnable. Composite grid → `ln ∈ [-6.02, 15.12]`.
5. **Figures** — `make_ensemble_figs.py` (per-σ fig2-style sets + `fig2_ens_mean`),
   `overlay_all_sigma.py` (all-σ mean overlay vs hex), `plot_bump_seed.py`
   (individual bumped seeds).

`submit_all.sh` fans the SLURM jobs across all σ dirs. The **hex reference**
(`sigma_0.05/hex_reference.py` → `hex_reference_shear.csv`, τ_M≈2.945e-5) is
σ-independent and reused by every figure. Helpers `collect_tau.py`,
`plot_fig2_ensemble.py` live in `sigma_0.05/` only.

## Key result
~3% of seeds show a pronounced secondary C″ peak; each has an extreme short facet
("sliver"). Confirmed physical (not a window/mesh artefact) by the `refine_test/`
and `tess/` studies.
