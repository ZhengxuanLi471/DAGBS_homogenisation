#!/usr/bin/env python3
"""Collect calibrated tau_M from the SLURM .out files into a CSV.

Parses the line emitted by calibrate_tau.measure_tau_M, e.g.
    [calibrate_tau] RESULT: tau_M = eta_ss/G_U = 2.620006e-05  (ln(1/tau_M) = 10.5497)
and pairs it with the seed name from the preceding "Processing  seeds_N" line.
"""
import glob
import os
import re
import csv

HERE = os.path.dirname(os.path.abspath(__file__))

seed_re   = re.compile(r"Processing\s+(seeds_\d+)")
result_re = re.compile(
    r"RESULT:\s*tau_M\s*=\s*eta_ss/G_U\s*=\s*([0-9.eE+-]+)\s*"
    r"\(ln\(1/tau_M\)\s*=\s*([0-9.eE+-]+)\)"
)

rows = []
for path in glob.glob(os.path.join(HERE, "slurm-*.out")):
    seed = None
    with open(path) as fh:
        text = fh.read()
    m_seed = seed_re.search(text)
    if m_seed:
        seed = m_seed.group(1)
    m_res = result_re.search(text)
    if not m_res:
        print(f"WARN: no RESULT line in {os.path.basename(path)}")
        continue
    tau_M = float(m_res.group(1))
    ln_inv_tau_M = float(m_res.group(2))
    rows.append((seed, tau_M, ln_inv_tau_M, os.path.basename(path)))

# sort by seed number when available
def seed_key(r):
    if r[0] is None:
        return 1 << 30
    return int(r[0].split("_")[1])

rows.sort(key=seed_key)

out_csv = os.path.join(HERE, "tau_M_sigma_0.05.csv")
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["seed", "tau_M", "ln_inv_tau_M", "source_file"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows -> {out_csv}")
