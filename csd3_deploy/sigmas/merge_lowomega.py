"""Attach the low-omega tail CSVs onto the existing [-3,10] per-seed CSVs.

For each seed in each sigma dir it concatenates
    Seed_<seed>_energy_real_im_data_shear_lowomega.csv   (ln omega*tau_M in [-6,-3))
onto
    Seed_<seed>_energy_real_im_data_shear.csv            (ln omega*tau_M in [-3,10])
sorted by ln_omega, de-duplicated, and writes the result back to the canonical
*_shear.csv after saving a one-time *_shear.csv.premerge backup. The grids were
generated on the same step with no overlap, so concatenation is exact.

Idempotent: a seed whose lowomega rows are already present (min ln_omega < -3.05)
is skipped, and the .premerge backup is never overwritten.

Run AFTER syncing the low-omega results back from CSD3:
    /home/zl471/miniconda3/envs/ngsolve/bin/python merge_lowomega.py [label ...]
With no args it merges every sigma dir in SIGMA_LABELS. Re-run make_ensemble_figs.py
and overlay_all_sigma.py afterwards to regenerate the figures on the extended grid.
"""
import glob
import os
import re
import shutil
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SIGMA_LABELS = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5"]


# Junction-continuity guard: the base and low-omega sweeps each re-measure tau_M
# independently, and the stored `omega` column is omega*tau_M. If the low-omega
# run's calibration diverges from the base (e.g. sigma_0.5 seed 54: 2.6e-8 vs the
# base 3.9e-6), the two CSVs live on different frequency axes and concatenating
# them glues a discontinuity at the ln_omega=-3 junction. In the creep regime C'
# (Cxyxy_real) DECREASES as omega decreases, so the tail point just below -3 must
# be <= the base point at -3. A tail point that is markedly higher is a divergent
# calibration -> refuse the merge.
JUNCTION_MAX_RATIO = 1.5


def _junction_ok(df_low, df_base):
    below = df_low[df_low["ln_omega"] < -3.0].sort_values("ln_omega").tail(1)
    above = df_base[df_base["ln_omega"] >= -3.0].sort_values("ln_omega").head(1)
    if below.empty or above.empty:
        return True, float("nan")
    c_lo = float(below["Cxyxy_real"].iloc[0])
    c_hi = float(above["Cxyxy_real"].iloc[0])
    ratio = c_lo / c_hi if c_hi > 0 else float("inf")
    return ratio <= JUNCTION_MAX_RATIO, ratio


def merge_dir(label):
    sdir = os.path.join(ROOT, f"sigma_{label}")
    base_glob = os.path.join(sdir, "Seed_seeds_*_energy_real_im_data_shear.csv")
    n_merged = n_skip = n_nolow = n_bad = 0
    for base in sorted(glob.glob(base_glob),
                       key=lambda p: int(re.search(r"seeds_(\d+)_", p).group(1))):
        low = base.replace("_shear.csv", "_shear_lowomega.csv")
        if not os.path.exists(low):
            n_nolow += 1
            continue
        df_base = pd.read_csv(base)
        if df_base["ln_omega"].min() < -3.05:        # already extended
            n_skip += 1
            continue
        df_low = pd.read_csv(low)
        ok, ratio = _junction_ok(df_low, df_base)
        if not ok:
            seed = re.search(r"seeds_(\d+)_", base).group(1)
            os.rename(low, low + ".bad")
            print(f"  sigma={label} seed {seed}: REFUSED merge "
                  f"(junction C' ratio {ratio:.1f} > {JUNCTION_MAX_RATIO}; "
                  f"divergent low-omega tau_M) -> quarantined {os.path.basename(low)}.bad")
            n_bad += 1
            continue
        merged = (pd.concat([df_low, df_base], ignore_index=True)
                    .drop_duplicates(subset="ln_omega")
                    .sort_values("ln_omega")
                    .reset_index(drop=True))
        bak = base + ".premerge"
        if not os.path.exists(bak):
            shutil.copy2(base, bak)
        merged.to_csv(base, index=False)
        n_merged += 1
    print(f"sigma={label}: merged {n_merged}, already-extended {n_skip}, "
          f"no-lowomega {n_nolow}, refused-bad {n_bad}")


if __name__ == "__main__":
    for lab in (sys.argv[1:] or SIGMA_LABELS):
        merge_dir(lab)
