#!/bin/bash
# Mesh-refinement convergence test for the high-omega sliver peak.
# 6 array tasks = {seed 24 (sigma 0.45), seed 96 (sigma 0.30)} x {L/30, L/100, L/150}.
# Each task re-sweeps ln(omega*tau_M) in [4,18] on a proportionally-refined mesh.
# CG solver; CSV written incrementally so a walltime kill still yields data.
# Submit from this dir:  sbatch run_refine.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=icelake
#SBATCH --account=RUDGE-SL3-CPU
#SBATCH --array=0-2
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

# task id -> (seed, sigma, den)
SEEDS=(96   96   96)
SIGMAS=(0.3 0.3 0.3)
DENS=(30   100  150)

i=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$i]}; SIGMA=${SIGMAS[$i]}; DEN=${DENS[$i]}

echo "task $i -> seed=$SEED sigma=$SIGMA den=$DEN  (cwd=$SLURM_SUBMIT_DIR)"
python -u "$SLURM_SUBMIT_DIR/real_im_energy_refine.py" \
    --seed "$SEED" --sigma "$SIGMA" --den "$DEN" \
    --lnmin 4 --lnmax 18 --npts 85
