#!/bin/bash
# Member 8 (ratio_9.755e-04, idx 7) wide-band sweep, SPLIT INTO A 14-TASK ARRAY.
# A global grid of NTOTAL=280 points over ln(omega*tau_M) in [-3,24] is sliced
# into 14 chunks of NPER=20 points; each array task computes one chunk. Tiny
# tasks backfill instantly on SL2. Maxwell time is HARDCODED to the clean
# production value (calibration skipped — the earlier single-shot calibration
# stagnated to a tau_M ~110x off).
#
# Each task writes refine_ratio_9.755e-04_frac50_wide_part<NN>_shear.csv;
# after pulling, concatenate with merge_member8_parts.py to get the canonical
# refine_ratio_9.755e-04_frac50_wide_shear.csv.
#
# Keep --array and NTOTAL/NPER in sync: array size must be NTOTAL/NPER (=14).
# Submit from this dir:  sbatch run_member8_wide.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --partition=icelake
#SBATCH --account=RUDGE-SL2-CPU
#SBATCH --array=0-13
#SBATCH --mail-user=zl471@cam.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module purge
module load miniconda3
source "$HOME/ngsolve/bin/activate"
which python
python -c "import ngsolve; print('ngsolve', ngsolve.__version__)"

cd "$SLURM_SUBMIT_DIR"

# tau_M from the clean production run: refine_ratio_9.755e-04_frac50_shear_meta.txt
TAU_M=1.44237781e-04
NTOTAL=280       # global grid points over [-3,24]
NPER=20          # points per array task  (NTOTAL/NPER = 14 = array size)

echo "member 8 (idx 7) chunk ${SLURM_ARRAY_TASK_ID}/13, fixed tau_M=$TAU_M (cwd=$SLURM_SUBMIT_DIR)"
python -u "$SLURM_SUBMIT_DIR/real_im_energy_fixedtau.py" \
    --idx 7 --den 50 --tau_M "$TAU_M" \
    --lnmin -3 --lnmax 24 --ntotal "$NTOTAL" --chunk "${SLURM_ARRAY_TASK_ID}" \
    --chunk_pts "$NPER" --outtag _wide
