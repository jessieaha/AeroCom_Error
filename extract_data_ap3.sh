#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
# Replace with your actual server username and hostname
SERVER="j.zhang3@vu.nl@aerocom-users.met.no"
DEST_DIR="Data/AP3_2026/"
PROCESSED_DIR="Data/AP3_processed_monthly"

# Base paths
PATH_AP3_2019="/metno/aerocom-users-database/AEROCOM-PHASE-III-2019"
PATH_AP3="/metno/aerocom-users-database/AEROCOM-PHASE-III"

# ==========================================
# RENEW MODE
# ==========================================
# RENEW=false (default):
#   - Skip model/var if a non-empty source file already exists locally.
#   - Delete any 0-byte placeholder files and re-download them.
# RENEW=true:
#   - Always re-download all source files (overwrite existing).
#   - Also delete the corresponding processed NetCDF in
#     Data/AP3_processed_monthly/{var}/{model}_{var}_processed.nc
#     so that get_data.py will reprocess them on the next run.
RENEW=FALSE

# ==========================================
# HARDCODED DATA
# ==========================================
# Variables from var.csv
VARIABLES=(
    # "abs550aer" "depbc" "depdust" "depoa" "depso2" "depso4" "depss"
    # "emibc" "emidust" "emioa" "emiso2" "emiss" "loadbc" "loaddust"
    # "loadoa" "loadso2" "loadso4" "loadss" "od440aer" "od550aer"
    # "od870aer" "od865aer" "od550bc" "od550dust" "od550oa" "od550so4" "od550ss",
    "dryss" "wetss" "dryso4" "wetso4"
)

# Updated Models List
MODELS=(
    "CAM5-ATRAS_AP3-CTRL"
    "CAM5.3-Oslo_AP3-CTRL2016-PD"
    "EC-Earth3-AerChem-met2010_AP3-CTRL2019"
    "ECHAM6-HAM2_AP3-CTRL2016-PD"
    "ECHAM6-SALSA_CTRL2016-PD"
    "ECHAM6.3-HAM2.3-met2010_AP3-CTRL"
    "ECHAM6.3-SALSA2.0-met2010_AP3-CTRL"
    "ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD"
    "ECMWF-IFS-CY45R1-CAMS-CTRL-met2010_AP3-CTRL"
    "ECMWF-IFS-CY46R1-CAMS-CTRL-met2010_AP3-CTRL"
    "EMEP_rv4_33_Glob-CTRL"
    "ETHZ-ECHAM6-HAM2_CTRL2015"
    "GEOS-Chem-v11-01_AP3-CTRL2016-PD"
    "GEOS-i33p2-met2010_AP3-CTRL"
    "GFDL-AM4-met2010_AP3-CTRL"
    "GISS-ModelE2p1p1-MATRIX_AP3-CTRL"
    "GISS-ModelE2p1p1-OMA_AP3-CTRL"
    "HadGEM3-GA7.1_AP3-CTRL2016-PD"
    "IMPACT_CTRL2016"
    "INCA_AP3-CTRL"
    "MIROC-SPRINTARS_AP3-CTRL"
    "NorESM2-met2010_AP3-CTRL"
    "NorESM2-met2010_AP3-CTRL-v3"
    "OsloCTM3v1.01-met2010_AP3-CTRL"
    "OsloCTM3v1.02-met2010_AP3-CTRL"
    "SPRINTARS-T213_AP3-CTRL2016-PD"
    "TM5-met2010_AP3-CTRL2019"
    "TM5_AP3-CTRL2016"
)

# ==========================================
# EXECUTION
# ==========================================
echo "========================================"
echo "RENEW mode: $RENEW"
echo "========================================"

# Create destination directory on local HPC
mkdir -p "$DEST_DIR"

for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "Processing model: $model"

    LOCAL_MODEL_DIR="${DEST_DIR}/${model}"
    mkdir -p "$LOCAL_MODEL_DIR"

    for var in "${VARIABLES[@]}"; do

        # ------------------------------------------------------------------
        # RENEW=true: wipe the processed output so get_data.py reprocesses it
        # ------------------------------------------------------------------
        if [ "$RENEW" = "true" ]; then
            processed_file="${PROCESSED_DIR}/${var}/${model}_${var}_processed.nc"
            if [ -f "$processed_file" ]; then
                rm -f "$processed_file"
                echo "    -> [REMOVED] Processed file: $processed_file"
            fi
        fi

        # ------------------------------------------------------------------
        # Handle existing local source files
        # ------------------------------------------------------------------
        # Collect all local source files matching this model/var/2010
        mapfile -t existing_files < <(find "$LOCAL_MODEL_DIR" -maxdepth 1 -name "*${var}*2010*.nc" 2>/dev/null)

        empty_removed=0
        has_valid=false

        for f in "${existing_files[@]}"; do
            if [ -s "$f" ]; then
                # Non-empty file exists
                has_valid=true
            else
                # 0-byte placeholder — delete so we can re-download
                rm -f "$f"
                echo "    -> [REMOVED] Empty placeholder: $(basename "$f")"
                empty_removed=$((empty_removed + 1))
            fi
        done

        # RENEW=false: skip download if a valid (non-empty) file already exists
        if [ "$RENEW" = "false" ] && [ "$has_valid" = "true" ]; then
            echo "    -> [SKIPPED] '$var' already present and non-empty."
            continue
        fi

        # ------------------------------------------------------------------
        # Download from remote server
        # ------------------------------------------------------------------
        SRC_PATH_2019="${SERVER}:${PATH_AP3_2019}/${model}/renamed/*${var}*2010*.nc"
        SRC_PATH_AP3="${SERVER}:${PATH_AP3}/${model}/renamed/*${var}*2010*.nc"

        # 1. Try AP3 2019 path first
        scp -q "$SRC_PATH_2019" "${LOCAL_MODEL_DIR}/" 2>/dev/null

        if [ $? -eq 0 ]; then
            echo "    -> [COPIED] '$var' pulled from AP3 2019."
        else
            # 2. Fall back to AP3 path
            scp -q "$SRC_PATH_AP3" "${LOCAL_MODEL_DIR}/" 2>/dev/null

            if [ $? -eq 0 ]; then
                echo "    -> [COPIED] '$var' pulled from AP3 Fallback."
            else
                echo "    -> [WARNING/MISSING] '$var' (2010) not found in either server path."
            fi
        fi

        # Check if the newly downloaded file is also 0-byte (server sent a placeholder)
        mapfile -t new_files < <(find "$LOCAL_MODEL_DIR" -maxdepth 1 -name "*${var}*2010*.nc" 2>/dev/null)
        for f in "${new_files[@]}"; do
            if [ ! -s "$f" ]; then
                rm -f "$f"
                echo "    -> [WARNING] Downloaded file was empty and has been removed: $(basename "$f")"
            fi
        done

    done
done

echo "========================================"
echo "Data transfer loop completed."
echo "========================================"
