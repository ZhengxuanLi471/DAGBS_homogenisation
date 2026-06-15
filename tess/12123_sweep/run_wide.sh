#!/bin/bash
# Follow-up to the production frac50 sweep: resolve the members-8/9 bump.
#   tasks 0-2: idx 6,7,8 (facet 6.5e-4, 3.0e-4, 1.3e-4) den=50, WIDE band
#              ln(omega*tau_M) in [-3,24] (207 pts) -> the true triangle peaks
#              (predicted at ln 17.6 / 20.0 / 22.4 by the L^-3.245 law) enter
#              the window; the slope~-1 bumps at ln 13-15 should stay put.
#   task 3:    idx 8 at den=100, high band ln in [8,24] -> mesh-independence
#              of both the bump and the triangle peak. (Its eta_ss calibration
#              solve will print a stagnated residual - documented cosmetic.)
# Outputs carry the _wide tag so the production frac50 CSVs are not touched.
# Submit from this dir:  sbatch run_wide.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --partition=icelake
#SBATCH --account=RUDGE-SL2-CPU
#SBATCH --array=0-3
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

i=${SLURM_ARRAY_TASK_ID}
if [ "$i" -lt 3 ]; then
    IDX=$((i + 6)); DEN=50; LNMIN=-3; LNMAX=24; NPTS=207
else
    IDX=8; DEN=100; LNMIN=8; LNMAX=24; NPTS=124
fi

echo "task $i -> idx=$IDX den=$DEN ln[$LNMIN,$LNMAX] npts=$NPTS (cwd=$SLURM_SUBMIT_DIR)"
python -u "$SLURM_SUBMIT_DIR/real_im_energy_refine.py" \
    --idx "$IDX" --den "$DEN" \
    --lnmin "$LNMIN" --lnmax "$LNMAX" --npts "$NPTS" --outtag _wide
