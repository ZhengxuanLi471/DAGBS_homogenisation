#!/bin/bash
# Per-frequency VTU field export for sigma_0.45 seed 24 + hex (one non-array job).
# Lives in csd3_deploy/vtu_movie/; vtu_run.py imports the solver modules + seed data
# from ../sigmas/sigma_0.45/ (that dir must already be present on CSD3).
# Submit from this folder on CSD3:  sbatch run_vtu.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=icelake
#SBATCH --account=RUDGE-SL2-CPU

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

echo "Running vtu_run.py --target both  (cwd=$SLURM_SUBMIT_DIR)"
python -u "$SLURM_SUBMIT_DIR/vtu_run.py" --target both
