#!/bin/bash
# Proportionally-refined frequency sweep over the small-grain ratio
# geometries of this family (tessellation_output.json, keys ratio_<r>).
# Geometries with facet length < 1e-4 (idx 8,9) are EXCLUDED: at that scale
# the graded mesh breaks the multigrid CG (calibration stagnates -> garbage
# tau_M, verified locally). One array task per kept geometry,
# ln(omega*tau_M) in [-3,18], CG solver, element size L/50 on facets
# shorter than 0.02 (meshes_refine.MakeMesh).
# CSV written incrementally so a walltime kill still yields data.
# Submit from this dir:  sbatch run_refine.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --partition=icelake
#SBATCH --account=RUDGE-SL2-CPU
#SBATCH --array=0-7
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

DEN=50

echo "task ${SLURM_ARRAY_TASK_ID} -> geometry idx ${SLURM_ARRAY_TASK_ID}, den=$DEN (cwd=$SLURM_SUBMIT_DIR)"
python -u "$SLURM_SUBMIT_DIR/real_im_energy_refine.py" \
    --idx "${SLURM_ARRAY_TASK_ID}" --den "$DEN" \
    --lnmin -3 --lnmax 18 --npts 161
