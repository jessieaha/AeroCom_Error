#!/bin/bash
#SBATCH -J ppe_monthly
#SBATCH -p rome
#SBATCH -A vuc17235
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /scratch-shared/jzhang1/PPE/logs/%x-%j.out
#SBATCH -e /scratch-shared/jzhang1/PPE/logs/%x-%j.err

# PPE monthly ensemble extract for one registered PPE run (py/PPE_set.PPE_RUNS).
#   sbatch ppe_extract_monthly.sbatch --ppe pd2010
#   sbatch ppe_extract_monthly.sbatch --ppe pd2024 --renew
# Source trees are read-only; output goes to /scratch-shared/jzhang1/PPE/<ppe_id>/.
# Do NOT module load 2025 here: EasyBuild HDF5 on LD_LIBRARY_PATH can
# clash with the netCDF4 / h5py wheels in the project .venv-1.

set -euo pipefail

PROJECT_ROOT="/home/jzhang1/AeroCom_Error"
VENV="${PROJECT_ROOT}/.venv-1"

mkdir -p /scratch-shared/jzhang1/PPE/logs

# Prefer the venv's own libs over any inherited EasyBuild HDF5.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  cleaned="$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '/sw/arch/' | paste -sd: - || true)"
  export LD_LIBRARY_PATH="${cleaned}"
fi

source "${VENV}/bin/activate"
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "${PROJECT_ROOT}"
echo "host=$(hostname)  date=$(date -Is)"
echo "python=$(command -v python)  $($(command -v python) -V)"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}MB  args=$*"

python -u py/ppe_extract_monthly.py --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"

echo "Finished $(date -Is)"
