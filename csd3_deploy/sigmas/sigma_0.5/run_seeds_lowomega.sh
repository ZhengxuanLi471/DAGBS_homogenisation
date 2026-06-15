#!/bin/bash
# Low-omega tail array job: ln(omega*tau_M) in [-6,-3) with the DIRECT solver.
# Path-agnostic (uses $SLURM_SUBMIT_DIR), so the identical script works in every
# sigma_* dir -- submit from within the sigma dir you want: `sbatch run_seeds_lowomega.sh`.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=icelake
#SBATCH --account=RUDGE-SL2-CPU

#SBATCH --array=0-99

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
python -c "import ngsolve; print('ngsolve version:', ngsolve.__version__)"

cd "$SLURM_SUBMIT_DIR"

START=$((1 + SLURM_ARRAY_TASK_ID))
END=$((START + 1))

echo "Running real_im_energy_lowomega.py with args: $START $END  (cwd=$SLURM_SUBMIT_DIR)"

python -u "$SLURM_SUBMIT_DIR/real_im_energy_lowomega.py" "$START" "$END"
