#!/bin/bash
#SBATCH -J aero_run
#SBATCH -p rome
#SBATCH -A vuc17235
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /scratch-shared/jzhang1/logs/%x-%j.out
#SBATCH -e /scratch-shared/jzhang1/logs/%x-%j.err

# Generic runner for project .py scripts or .ipynb notebooks on Snellius.
#
# Usage:
#   sbatch jobs/run_notebook_or_py.sbatch notebooks/AOD_SS_ERROR_PPE.py
#   sbatch jobs/run_notebook_or_py.sbatch notebooks/AOD_SS_ERROR.ipynb
#   TARGET=notebooks/foo.py sbatch jobs/run_notebook_or_py.sbatch
#   sbatch jobs/run_notebook_or_py.sbatch notebooks/AOD_SS_ERROR_PPE.py -- --other-args
#
# Do NOT module load 2025: EasyBuild HDF5 can clash with venv netCDF4/h5py.

set -euo pipefail

PROJECT_ROOT="/home/jzhang1/AeroCom_Error"
VENV="${PROJECT_ROOT}/.venv"
LOG_DIR="/scratch-shared/jzhang1/logs"
mkdir -p "${LOG_DIR}"

# Prefer TARGET env, else first positional after sbatch script name.
if [[ -n "${TARGET:-}" ]]; then
  :
elif [[ $# -ge 1 ]]; then
  TARGET="$1"
  shift
else
  echo "ERROR: pass a .py or .ipynb path, or set TARGET=..." >&2
  exit 1
fi

# Allow "--" then extra args for the python/notebook process.
EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA_ARGS=("$@")
elif [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

# Resolve relative paths against the project root.
if [[ "${TARGET}" != /* ]]; then
  TARGET="${PROJECT_ROOT}/${TARGET}"
fi

if [[ ! -e "${TARGET}" ]]; then
  echo "ERROR: TARGET not found: ${TARGET}" >&2
  exit 1
fi

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  cleaned="$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '/sw/arch/' | paste -sd: - || true)"
  export LD_LIBRARY_PATH="${cleaned}"
fi

source "${VENV}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLBACKEND=Agg

cd "${PROJECT_ROOT}"
echo "host=$(hostname)  date=$(date -Is)"
echo "TARGET=${TARGET}"
echo "python=$(command -v python)  $($(command -v python) -V)"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}MB"

case "${TARGET}" in
  *.py)
    python -u "${TARGET}" "${EXTRA_ARGS[@]}"
    ;;
  *.ipynb)
    if ! python -c "import nbconvert, jupyter_client" 2>/dev/null; then
      echo "Installing nbconvert + ipykernel into .venv ..."
      pip install -q nbconvert ipykernel
    fi
    jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 \
      "${TARGET}"
    ;;
  *)
    echo "ERROR: unsupported TARGET extension (use .py or .ipynb): ${TARGET}" >&2
    exit 1
    ;;
esac

echo "Finished $(date -Is)"
