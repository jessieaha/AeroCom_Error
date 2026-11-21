#!/bin/bash
#SBATCH -J dashbox
#SBATCH -p rome
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /scratch-shared/jzhang1/logs/dashbox-%j.out
#SBATCH -e /scratch-shared/jzhang1/logs/dashbox-%j.err

# Aggregate one drawn polygon (pd2024, pd2010, ICON) without blocking the dashboard.
#
#   sbatch dashboard/aggregate_box.sbatch <slug>
#
# Uses .venv-1. To run inside an image instead:
#   APPTAINER_IMAGE=/path/to/env.sif sbatch dashboard/aggregate_box.sbatch <slug>
# The image must provide python with xarray, netCDF4, pandas and matplotlib.
# The project and scratch directories are bind-mounted.

set -euo pipefail

PROJECT_ROOT="/home/jzhang1/AeroCom_Error"
VENV="${PROJECT_ROOT}/.venv-1"
SLUG="${1:?polygon slug required}"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  cleaned="$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '/sw/arch/' | paste -sd: - || true)"
  export LD_LIBRARY_PATH="${cleaned}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLBACKEND=Agg
export PYTHONPATH="${PROJECT_ROOT}/py:${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

echo "host=$(hostname)  date=$(date -Is)  slug=${SLUG}"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}MB"

run_py() {
  python -u -m dashboard.aggregate_box "${SLUG}"
}

if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
  echo "apptainer=${APPTAINER_IMAGE}"
  apptainer exec --bind /home/jzhang1,/scratch-shared,/projects "${APPTAINER_IMAGE}" \
    bash -lc "cd ${PROJECT_ROOT} && export PYTHONPATH=${PYTHONPATH} && python -u -m dashboard.aggregate_box ${SLUG}"
else
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  echo "python=$(command -v python)"
  run_py
fi

echo "Finished $(date -Is)"
