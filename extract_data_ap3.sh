#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
# Replace with your actual server username and hostname
SERVER="j.zhang3@vu.nl@aerocom-users.met.no"
DEST_DIR="Data/AP3_2026/"

# Base paths 
PATH_AP3_2019="/metno/aerocom-users-database/AEROCOM-PHASE-III-2019"
PATH_AP3="/metno/aerocom-users-database/AEROCOM-PHASE-III"

# ==========================================
# HARDCODED DATA
# ==========================================
# Variables from var.csv
VARIABLES=(
    "abs550aer" "depbc" "depdust" "depoa" "depso2" "depso4" "depss"
    "emibc" "emidust" "emioa" "emiso2" "emiss" "loadbc" "loaddust"
    "loadoa" "loadso2" "loadso4" "loadss" "od440aer" "od550aer"
    "od870aer" "od865aer" "od550bc" "od550dust" "od550oa" "od550so4" "od550ss"
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
# Create destination directory on local HPC
mkdir -p "$DEST_DIR"
for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "Processing model: $model"

    # NEW: Define and create the model-specific target subdirectory locally
    LOCAL_MODEL_DIR="${DEST_DIR}/${model}"
    mkdir -p "$LOCAL_MODEL_DIR"

    for var in "${VARIABLES[@]}"; do
        # Build the two potential remote path strings
        SRC_PATH_2019="${SERVER}:${PATH_AP3_2019}/${model}/renamed/*${var}*2010*.nc"
        SRC_PATH_AP3="${SERVER}:${PATH_AP3}/${model}/renamed/*${var}*2010*.nc"

        # 1. Direct attempt from AP3 2019 (Saves directly to model subfolder)
        scp "$SRC_PATH_2019" "${LOCAL_MODEL_DIR}/"

        if [ $? -eq 0 ]; then
            echo "    -> [COPIED] '$var' successfully pulled from AP3 2019."
        else
            # 2. Fall back to old AP3 path directly if first attempt failed
            scp -q "$SRC_PATH_AP3" "${LOCAL_MODEL_DIR}/" 2>/dev/null

            if [ $? -eq 0 ]; then
                echo "    -> [COPIED] '$var' successfully pulled from AP3 Fallback."
            else
                # 3. Warning if completely missing from both target locations
                echo "    -> [WARNING/MISSING] Variable '$var' (2010) failed or path does not exist in either server root."
            fi
        fi
    done
done

echo "========================================"
echo "Data transfer loop completed."
