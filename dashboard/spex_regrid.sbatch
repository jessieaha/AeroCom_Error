#!/bin/bash
#SBATCH -J spexregrid
#SBATCH -p rome
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /scratch-shared/jzhang1/logs/spexregrid-%A_%a.out
#SBATCH -e /scratch-shared/jzhang1/logs/spexregrid-%A_%a.err
#SBATCH --array=1-12

# Regrid one month of SPEXone L2 to a regular grid (SPEX_RES degrees).
#
#   SPEX_YEAR=2025 SPEX_RES=0.5 sbatch dashboard/spex_regrid.sbatch
#
# Array task id is the month. Output:
#   Data/SPEXone_regridded/SPEXone_L2_regridded_{year}_{month}_{res}deg_v4_0.parquet

set -euo pipefail

PROJECT_ROOT="/home/jzhang1/AeroCom_Error"
VENV="${PROJECT_ROOT}/.venv-1"
YEAR="${SPEX_YEAR:-2025}"
RES="${SPEX_RES:-1.0}"
MONTH="${SLURM_ARRAY_TASK_ID:?array task id is the month}"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  cleaned="$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '/sw/arch/' | paste -sd: - || true)"
  export LD_LIBRARY_PATH="${cleaned}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export SPEX_YEAR="${YEAR}"
export SPEX_RES="${RES}"
export SPEXONE_DIR="${PROJECT_ROOT}/Data/SPEXone"
export SPEX_OUT_DIR="${PROJECT_ROOT}/Data/SPEXone_regridded"
export PYTHONPATH="${PROJECT_ROOT}/py:${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

echo "host=$(hostname)  date=$(date -Is)  year=${YEAR} month=${MONTH} res=${RES}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -u py/SPEXone_regrid_model.py "${MONTH}" "${YEAR}" "${RES}"
echo "Finished $(date -Is)"
