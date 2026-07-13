#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
# Replace with your actual server username and hostname
SERVER="user@your_server_address"
DEST_DIR="Data/AP32026"

# Base paths 
PATH_AP3_2019="/metno/aerocom-users-database/AEROCOM-PHASE-III-CTRL2018"
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
    echo "Processing model: $model"
    
    remote_dir=""

    # 1. Check AP3 2019 path first
    if ssh -q "$SERVER" "[ -d \"$PATH_AP3_2019/$model/renamed\" ]"; then
        remote_dir="$PATH_AP3_2019/$model/renamed"
        echo " -> Target located in AP3 2019 path."
        
    # 2. Fall back to AP3 path if the first fails
    elif ssh -q "$SERVER" "[ -d \"$PATH_AP3/$model/renamed\" ]"; then
        remote_dir="$PATH_AP3/$model/renamed"
        echo " -> Target located in AP3 fallback path."
        
    # 3. Warning if completely missing
    else
        echo " -> [WARNING] Model folder missing in both AP3 2019 and AP3 paths. Skipping."
        echo "----------------------------------------"
        continue
    fi

    # 4. Transfer Variables
    for var in "${VARIABLES[@]}"; do
        # We use -q for quiet scp, redirect error to /dev/null, and capture the exit status.
        scp -q "${SERVER}:\"${remote_dir}/*${var}*2010*\"" "${DEST_DIR}/" 2>/dev/null
        
        # If scp returns anything other than 0, the transfer failed (usually because the file is missing)
        if [ $? -ne 0 ]; then
            echo "    -> [MISSING] Variable '$var' not found for year 2010."
        else
            echo "    -> [COPIED] Variable '$var' transferred successfully."
        fi
    done
    
    echo "----------------------------------------"
done

echo "Data transfer complete."
