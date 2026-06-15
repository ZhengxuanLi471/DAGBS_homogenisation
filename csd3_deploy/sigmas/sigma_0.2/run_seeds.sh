#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --partition=icelake
#SBATCH --account=RUDGE-SL3-CPU

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

cd "/home/zl471/rds/sigmas/sigma_0.2"

START=$((1 + SLURM_ARRAY_TASK_ID))
END=$((START + 1))

echo "Running real_im_energy.py with args: $START $END"

python -u /home/zl471/rds/sigmas/sigma_0.2/real_im_energy.py "$START" "$END"
