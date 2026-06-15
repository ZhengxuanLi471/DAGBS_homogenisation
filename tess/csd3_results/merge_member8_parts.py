#!/usr/bin/env python3
"""Concatenate the array-split member-8 wide-band parts into one canonical CSV.

The 14-task array (run_member8_wide.sh) writes
  12123_sweep/refine_ratio_9.755e-04_frac50_wide_part00..13_shear.csv
each holding 20 points of a global 280-point linspace over ln in [-3,24].
This stitches them (sorted by ln_omega, duplicate ln dropped) into
  12123_sweep/refine_ratio_9.755e-04_frac50_wide_shear.csv
which analyze_wide_12123.py / scaling_final_12123.py already read.

Usage:  python merge_member8_parts.py
        python merge_member8_parts.py ratio_9.755e-04   # explicit key
"""
import glob
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.join(HERE, "12123_sweep")
KEY = sys.argv[1] if len(sys.argv) > 1 else "ratio_9.755e-04"

pat = os.path.join(SW, f"refine_{KEY}_frac50_wide_part*_shear.csv")
parts = sorted(glob.glob(pat))
if not parts:
    sys.exit(f"no part files matching {pat}")

frames = []
for p in parts:
    df = pd.read_csv(p)
    frames.append(df)
    print(f"  {os.path.basename(p)}: {len(df)} pts "
          f"(ln {df.ln_omega.min():.2f}..{df.ln_omega.max():.2f})")

merged = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset="ln_omega")
            .sort_values("ln_omega")
            .reset_index(drop=True))

out = os.path.join(SW, f"refine_{KEY}_frac50_wide_shear.csv")
merged.to_csv(out, index=False)
print(f"\nmerged {len(parts)} parts -> {len(merged)} pts, "
      f"ln [{merged.ln_omega.min():.2f}, {merged.ln_omega.max():.2f}]")
print(f"wrote {out}")

# carry a tau_M meta forward (from any part's meta) so downstream sees it
metas = sorted(glob.glob(os.path.join(SW, f"refine_{KEY}_frac50_wide_part*_shear_meta.txt")))
if metas:
    txt = open(metas[0]).read()
    tau = re.search(r"tau_M=([\d.eE+-]+)", txt)
    with open(out.replace(".csv", "_meta.txt"), "w") as f:
        f.write(f"key={KEY} den=50 wide MERGED from {len(parts)} array parts\n")
        if tau:
            f.write(f"tau_M={tau.group(1)} (HARDCODED, calibration skipped)\n")
    print(f"wrote {out.replace('.csv', '_meta.txt')}")
