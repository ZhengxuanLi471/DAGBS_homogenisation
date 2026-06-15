#!/bin/bash
# Submit the seed sweep for every sigma. Run from /home/zl471/rds/sigmas.
set -euo pipefail
BASE="/home/zl471/rds/sigmas"
for d in "$BASE"/sigma_*/; do
  sig=$(basename "$d")
  echo "Submitting $sig ..."
  ( cd "$d" && sbatch run_seeds.sh )
done
echo "All submitted. Check with: squeue -u $USER"
