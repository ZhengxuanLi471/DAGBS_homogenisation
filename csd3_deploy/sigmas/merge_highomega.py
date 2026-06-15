"""Attach the high-omega tail CSVs onto the existing per-seed CSVs.

For each seed in each sigma dir it concatenates
    Seed_<seed>_energy_real_im_data_shear_highomega.csv   (ln omega*tau_M in (10,15])
onto
    Seed_<seed>_energy_real_im_data_shear.csv             (... up to ln omega*tau_M = 10)
sorted by ln_omega, de-duplicated, and writes the result back to the canonical
*_shear.csv after saving a one-time *_shear.csv.prehigh backup. The grids were
generated on the same step with no overlap (base max = 10.0, first high = 10.131),
so concatenation is exact.

Run this AFTER merge_lowomega.py if you also extended the low end -- it operates on
whatever the current *_shear.csv is (base-only, or already low-extended) and just
appends the high tail, so the two extensions compose. The backups are independent
(.premerge from the low merge, .prehigh from this one).

Junction-continuity guard: like the low merge, the high and base sweeps each
re-measure tau_M independently. At the unrelaxed plateau the storage C'
(Cxyxy_real) is nearly flat and continuous (high_first/base_last ~ 1.00-1.02), so a
gross discontinuity there flags a broken solve / divergent calibration -> the tail
is refused and quarantined as *.bad rather than glued on. (Unlike the low-omega
creep tail, the high-omega tail is intrinsically robust to a tau error -- a wrong
tau just drives the solve toward the elastic limit, where the plateau already sits
-- so refusals here should be rare and indicate a genuinely failed solve.)

Idempotent: a seed whose high rows are already present (max ln_omega > 10.05) is
skipped, and the .prehigh backup is never overwritten.

Run AFTER syncing the high-omega results back from CSD3:
    /home/zl471/miniconda3/envs/ngsolve/bin/python merge_highomega.py [label ...]
With no args it merges every sigma dir in SIGMA_LABELS. Re-run make_ensemble_figs.py
and overlay_all_sigma.py afterwards to regenerate the figures on the extended grid.
"""
import glob
import os
import re
import shutil
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SIGMA_LABELS = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5"]

# Storage-modulus continuity at the ln_omega=10 junction. The plateau is flat, so
# the legitimate spread is ~[0.997, 1.022]; these bounds catch only gross breakage.
JUNCTION_RATIO_LO = 0.8
JUNCTION_RATIO_HI = 1.25

# C'' (dissipation) re-level factor sanity bounds. A small junction step (factor
# ~[0.85,1.35]) is a benign tau_M-mismatch between the base and high runs and is
# corrected by re-leveling. A GROSS factor (e.g. 8x) is not a level offset but a
# broken high-omega solve -> refuse and quarantine rather than force-scale it.
RELEVEL_LO = 0.5
RELEVEL_HI = 2.0


def _junction_ok(df_base, df_high):
    base_last = df_base[df_base["ln_omega"] <= 10.001].sort_values("ln_omega").tail(1)
    high_first = df_high[df_high["ln_omega"] > 10.0].sort_values("ln_omega").head(1)
    if base_last.empty or high_first.empty:
        return True, float("nan")
    c_base = float(base_last["Cxyxy_real"].iloc[0])
    c_high = float(high_first["Cxyxy_real"].iloc[0])
    if not (c_base > 0):
        return True, float("nan")
    ratio = c_high / c_base
    ok = np.isfinite(ratio) and JUNCTION_RATIO_LO <= ratio <= JUNCTION_RATIO_HI
    return ok, ratio


def _relevel_high(df_base, df_high, ncal=5):
    """Re-level the high-omega tail so it is CONTINUOUS with the base sweep at the
    ln=10 junction. The base and high sweeps re-measure tau_M independently, which
    leaves a small *multiplicative* level offset at the glue point — chiefly in the
    dissipation C''=Cxyxy_imag (~+15%; C'=Cxyxy_real is ~continuous, -0.3%). That step
    was misread by the peak detector as a high-omega Q^-1 peak (false positives at
    ln=10.13). Here we log-linearly extrapolate the base's last `ncal` points of each
    modulus to the first high ln and scale the WHOLE tail by expected/first so it
    lands on the base trend. A uniform scale preserves any genuine bump in the tail
    (a local maximum is scale-invariant) — it only removes the glue discontinuity.
    Returns (releveled_high_df, (f_real, f_imag))."""
    out = df_high.copy()
    b = df_base[df_base["ln_omega"] <= 10.001].sort_values("ln_omega").tail(ncal)
    h = out[out["ln_omega"] > 10.0].sort_values("ln_omega")
    if len(b) < 2 or h.empty:
        return out, (1.0, 1.0)
    ln_h0 = float(h["ln_omega"].iloc[0])
    facs = {}
    for col, ecol in (("Cxyxy_real", "E_storage"), ("Cxyxy_imag", "E_diss_total")):
        y = b[col].to_numpy()
        h0 = float(h[col].iloc[0])
        if np.any(y <= 0) or not (h0 > 0):
            facs[col] = 1.0
            continue
        a, c = np.polyfit(b["ln_omega"].to_numpy(), np.log(y), 1)
        f = float(np.exp(a * ln_h0 + c) / h0)
        out[col] = out[col] * f
        if ecol in out.columns:
            out[ecol] = out[ecol] * f
        facs[col] = f
    return out, (facs.get("Cxyxy_real", 1.0), facs.get("Cxyxy_imag", 1.0))


def merge_dir(label):
    sdir = os.path.join(ROOT, f"sigma_{label}")
    base_glob = os.path.join(sdir, "Seed_seeds_*_energy_real_im_data_shear.csv")
    n_merged = n_skip = n_nohigh = n_bad = 0
    relevels = []
    for base in sorted(glob.glob(base_glob),
                       key=lambda p: int(re.search(r"seeds_(\d+)_", p).group(1))):
        high = base.replace("_shear.csv", "_shear_highomega.csv")
        if not os.path.exists(high):
            n_nohigh += 1
            continue
        bak = base + ".prehigh"
        if os.path.exists(bak):
            shutil.copy2(bak, base)                   # restore pre-high state -> re-runnable
        df_base = pd.read_csv(base)
        if df_base["ln_omega"].max() > 10.05:        # merged with no backup; cannot un-merge
            n_skip += 1
            continue
        df_high = pd.read_csv(high)
        ok, ratio = _junction_ok(df_base, df_high)
        if not ok:
            seed = re.search(r"seeds_(\d+)_", base).group(1)
            os.replace(high, high + ".bad")
            print(f"  sigma={label} seed {seed}: REFUSED merge "
                  f"(junction C' ratio {ratio:.2f} outside "
                  f"[{JUNCTION_RATIO_LO},{JUNCTION_RATIO_HI}]; broken high-omega solve) "
                  f"-> quarantined {os.path.basename(high)}.bad")
            n_bad += 1
            continue
        df_high_rl, (fr, fi) = _relevel_high(df_base, df_high)   # remove the C''/C' glue step
        if not (RELEVEL_LO <= fi <= RELEVEL_HI):
            seed = re.search(r"seeds_(\d+)_", base).group(1)
            os.replace(high, high + ".bad")
            print(f"  sigma={label} seed {seed}: REFUSED merge "
                  f"(C'' re-level factor {fi:.2f} outside [{RELEVEL_LO},{RELEVEL_HI}]; "
                  f"broken high-omega dissipation) -> quarantined {os.path.basename(high)}.bad")
            n_bad += 1
            continue
        df_high = df_high_rl
        relevels.append(fi)
        merged = (pd.concat([df_base, df_high], ignore_index=True)
                    .drop_duplicates(subset="ln_omega")
                    .sort_values("ln_omega")
                    .reset_index(drop=True))
        if not os.path.exists(bak):
            shutil.copy2(base, bak)
        merged.to_csv(base, index=False)
        n_merged += 1
    rl = np.array(relevels) if relevels else np.array([1.0])
    print(f"sigma={label}: merged {n_merged}, already-extended {n_skip}, "
          f"no-highomega {n_nohigh}, refused-bad {n_bad}; "
          f"C'' re-level factor median={np.median(rl):.3f} range=[{rl.min():.3f},{rl.max():.3f}]")


if __name__ == "__main__":
    for lab in (sys.argv[1:] or SIGMA_LABELS):
        merge_dir(lab)
