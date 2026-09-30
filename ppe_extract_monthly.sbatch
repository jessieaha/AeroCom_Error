#!/bin/bash
#SBATCH -J ppe_monthly
#SBATCH -p rome
#SBATCH -A vuc17235
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /scratch-shared/jzhang1/PPE_monthly/logs/%x-%j.out
#SBATCH -e /scratch-shared/jzhang1/PPE_monthly/logs/%x-%j.err

# PPE monthly ensemble extract for AOD_SS_ERROR inputs.
# Source tree under PPE_Experiments is read-only.
# Do NOT module load 2025 here: EasyBuild HDF5 on LD_LIBRARY_PATH can
# clash with the netCDF4 / h5py wheels in the project .venv.

set -euo pipefail

PROJECT_ROOT="/home/jzhang1/AeroCom_Error"
VENV="${PROJECT_ROOT}/.venv"
SCRATCH="/scratch-shared/jzhang1/PPE_monthly"

mkdir -p "${SCRATCH}/members" "${SCRATCH}/logs" \
         "${PROJECT_ROOT}/Data/PPE_processed_monthly"

# Prefer the venv's own libs over any inherited EasyBuild HDF5.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  cleaned="$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '/sw/arch/' | paste -sd: - || true)"
  export LD_LIBRARY_PATH="${cleaned}"
fi

source "${VENV}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HDF5_USE_FILE_LOCKING=FALSE

cd "${PROJECT_ROOT}"
echo "host=$(hostname)  date=$(date -Is)"
echo "python=$(command -v python)  $($(command -v python) -V)"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}MB"

python -u py/ppe_extract_monthly.py "$@"

echo "Finished $(date -Is)"
