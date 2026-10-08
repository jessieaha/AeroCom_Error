#!/bin/bash
# Submit PPE AAOD attribution on a compute node (not the login node).
set -euo pipefail
PROJECT_ROOT="/home/jzhang1/AeroCom_Error"
mkdir -p /scratch-shared/jzhang1/logs /scratch-shared/jzhang1/PPE_AAOD
mkdir -p "${PROJECT_ROOT}/Data/PPE_AAOD" "${PROJECT_ROOT}/Data/PPE_AAOD/scratch" \
         "${PROJECT_ROOT}/figure/PPE_AAOD"
cd "${PROJECT_ROOT}"
exec sbatch -J aaod_ppe --mem=64G run_notebook_or_py.sbatch notebooks/AAOD_error_attribution_PPE.py
