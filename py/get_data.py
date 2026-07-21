""" Script to get the data from the AeroCom files.

Monthly data is written as structured NetCDF files under Data/AP3_processed_monthly/.
The legacy master pickle can still be produced for backward compatibility by setting
SAVE_PICKLE = True. See the MODEL_SELECTION variable to switch between explicit and
auto-discovered model lists.

BEFORE RUNNING THIS FILE, CHECK:
    - WHERE THE FILES ARE BEING SAVED (save_path / save_path_average)
    - WHICH FILES ARE BEING OPENED (path_original / path_regrid)

"""


###################### IMPORT MODULES ######################

import sys
import os
from pathlib import Path
import pickle
# Add parent directory to sys.path to enable proper imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import aerocom_data
import functions

import importlib
importlib.reload(aerocom_data)
importlib.reload(functions)


###################### DEFINE STUFF ######################

# YEAR = 2010
temporal = 'monthly'

# --- Output options ---
# SAVE_PICKLE controls whether the master monthly dictionary is written to
# Data/var_files/original/monthly/monthly_aerocom_data.pickle.
# It is kept for backward compatibility; the default is now False because
# downstream work is moving to the NetCDF-based loader.
SAVE_PICKLE = False
SAVE_NETCDF = True

# --- Monthly model selection ---
# 'explicit' -> use the hard-coded monthly model lists (models_AOD, models_AAOD, ...)
#               and take the union of all monthly models (legacy behavior).
# 'auto'     -> discover all model directories in dir_primary / dir_secondary
#               and load every model that has at least one monthly variable.
MODEL_SELECTION = 'explicit'  # or 'auto'

if temporal == 'monthly':
    print('Getting monthly data')

    # -------------------------------------------------------------------------
    # 1. Explicit monthly model lists (Option A)
    # -------------------------------------------------------------------------
    models_AOD = [
        'CAM5.3-Oslo_AP3-CTRL2016-PD', 'ECHAM6-HAM2_AP3-CTRL2016-PD', 'ECHAM6-SALSA_CTRL2016-PD',
        'ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD', 'TM5_AP3-CTRL2016',
        'CAM5-ATRAS_AP3-CTRL', 'EC-Earth3-AerChem-met2010_AP3-CTRL2019', 'ECHAM6.3-HAM2.3-met2010_AP3-CTRL',
        'ECHAM6.3-SALSA2.0-met2010_AP3-CTRL', 'GEOS-i33p2-met2010_AP3-CTRL', 'GFDL-AM4-met2010_AP3-CTRL',
        'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010', 'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010', 'INCA_AP3-CTRL',
        'NorESM2-met2010_AP3-CTRL', 'MIROC-SPRINTARS_AP3-CTRL', 'TM5-met2010_AP3-CTRL2019'
    ]

    models_AAOD = [
        'CAM5.3-Oslo_AP3-CTRL2016-PD', 'CAM5_CTRL2016', 'CAM5-ATRAS_AP3-CTRL',
        'EC-Earth3-AerChem-met2010_AP3-CTRL2019', 'ECHAM6-HAM2_AP3-CTRL2016-PD',
        'ECHAM6.3-HAM2.3-met2010_AP3-CTRL', 'ECHAM6-SALSA_CTRL2016-PD', 'ECHAM6.3-SALSA2.0-met2010_AP3-CTRL',
        'ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD', 'GEOS-i33p2-met2010_AP3-CTRL', 'GFDL-AM4-met2010_AP3-CTRL',
        'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010', 'INCA_AP3-CTRL', 'SPRINTARS-T213_AP3-CTRL2016-PD',
        'MIROC-SPRINTARS_AP3-CTRL', 'TM5_AP3-CTRL2016', 'TM5-met2010_AP3-CTRL2019'
    ]

    # Optional sanity checks for the explicit lists
    missing_from_aaod = list(set(models_AOD) - set(models_AAOD))
    print("Missing from AAOD:", missing_from_aaod)
    missing_from_aod = list(set(models_AAOD) - set(models_AOD))
    print("Missing from AOD:", missing_from_aod)

    # -------------------------------------------------------------------------
    # 2. Directories and variables
    # -------------------------------------------------------------------------
    dir_primary = "./Data/AP3_2026"
    dir_secondary = "./Data/AEROCOM_III"

    VARIABLES = (
        "abs550aer", "depbc", "depdust", "depoa", "depso2", "depso4", "depss",
        "emibc", "emidust", "emioa", "emiso2", "emiss", "loadbc", "loaddust",
        "loadoa", "loadso2", "loadso4", "loadss", "od440aer", "od550aer",
        "od870aer", "od865aer", "od550bc", "od550dust", "od550oa", "od550so4", "od550ss",
        "precip"
    )

    # -------------------------------------------------------------------------
    # 3. Choose model list: explicit union or auto-discovery (Option B)
    # -------------------------------------------------------------------------
    def discover_models(*dirs):
        """Return sorted model directory names found in any of the given directories."""
        discovered = set()
        for d in dirs:
            if not os.path.isdir(d):
                continue
            discovered.update(
                m for m in os.listdir(d)
                if os.path.isdir(os.path.join(d, m)) and not m.startswith('.')
            )
        return sorted(discovered)

    if MODEL_SELECTION == 'explicit':
        all_models = sorted(set(models_AOD + models_AAOD))
        print(f"\nMODEL_SELECTION='explicit': using {len(all_models)} models from union of models_AOD and models_AAOD")
    elif MODEL_SELECTION == 'auto':
        all_models = discover_models(dir_primary, dir_secondary)
        print(f"\nMODEL_SELECTION='auto': discovered {len(all_models)} model directories")
    else:
        raise ValueError(f"MODEL_SELECTION must be 'explicit' or 'auto', got '{MODEL_SELECTION}'")

    # Split models by availability in primary / secondary directories
    primary_models = [m for m in all_models if os.path.isdir(os.path.join(dir_primary, m))]
    secondary_models = [m for m in all_models if m not in primary_models and os.path.isdir(os.path.join(dir_secondary, m))]
    missing_models = [m for m in all_models if m not in primary_models and m not in secondary_models]
    if missing_models:
        print(f"WARNING: {len(missing_models)} model(s) not found in either directory:")
        for m in missing_models:
            print(f"  - {m}")

    save_path = f'./Data/var_files/original/{temporal}/'
    save_path_average = './Data/var_files/original/average/'

    # -------------------------------------------------------------------------
    # 4. Load and save monthly variables
    # -------------------------------------------------------------------------
    print(f"\nLoading monthly data from AP3_2026 for {len(primary_models)} models...")
    data_primary = aerocom_data.load_all_model_data(
        base_dir=dir_primary,
        models=primary_models,
        variables=VARIABLES,
        temporal='monthly'
    )

    if secondary_models:
        print(f"\nLoading monthly data from AEROCOM_III for {len(secondary_models)} fallback models...")
        data_secondary = aerocom_data.load_all_model_data(
            base_dir=dir_secondary,
            models=secondary_models,
            variables=VARIABLES,
            temporal='monthly'
        )
        all_data = {**data_primary, **data_secondary}
    else:
        all_data = data_primary

    print("\n=============================================")
    print(f" Master dataset created with {len(all_data)} total models.")
    print("=============================================\n")

    if SAVE_PICKLE:
        output_base_dir = './Data/var_files/original/monthly/'
        os.makedirs(output_base_dir, exist_ok=True)
        print(f"\nSaving master dataset to a single PICKLE file in {output_base_dir} ...")
        out_file = os.path.join(output_base_dir, f"{temporal}_aerocom_data.pickle")
        try:
            with open(out_file, 'wb') as f:
                pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved pickle: {out_file}")
        except Exception as e:
            print(f"Error saving pickle file: {e}")

    if SAVE_NETCDF:
        aerocom_data.save_model_data_to_netcdf(all_data, output_base_dir="./Data/AP3_processed_monthly")
elif temporal == '3hourly':
    print('Getting 3hourly data')
    # 3hourly branch keeps its own output settings to preserve existing behaviour
    save_pickle = True
    save_nc = SAVE_NETCDF

    var_3hour = (
    "abs550aer", "od440aer", "od550aer", "od870aer", "od865aer", "od550bc", "od550dust", "od550oa", "od550so4", "od550ss"
)
    models_primary = [
        'ECHAM6-SALSA_CTRL2016-PD',
        'ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD',
        'ECMWF-IFS-CY45R1-CAMS-CTRL-met2010_AP3-CTRL',
        'ECMWF-IFS-CY46R1-CAMS-CTRL-met2010_AP3-CTRL',
        'GEOS-Chem-v11-01_AP3-CTRL2016-PD',
        'HadGEM3-GA7.1_AP3-CTRL2016-PD',
        'IMPACT_CTRL2016',
        'MIROC-SPRINTARS_AP3-CTRL',
        'SPRINTARS-T213_AP3-CTRL2016-PD',
        'TM5_AP3-CTRL2016'
    ]
    dir_primary = "./Data/AP3_2026"

    models_secondary_unique = [
        'CAM5.3-Oslo_AP3-CTRL2016-PD',
        'CAM5_CTRL2016',
        'ECHAM6-HAM2_AP3-CTRL2016-PD',
        'GEOS-i33p2-met2010_AP3-CTRL'
    ]
    dir_secondary = "./Data/AEROCOM_III"

    # 2. Load data from both directories separately
    print("Loading Primary Data (AP3_2026)...")
    data_primary = aerocom_data.load_all_model_data(
        base_dir=dir_primary, 
        models=models_primary, 
        variables=var_3hour,
        temporal='3hourly'
    )

    print("\nLoading Secondary Unique Data (AEROCOM_III)...")
    data_secondary = aerocom_data.load_all_model_data(
        base_dir=dir_secondary, 
        models=models_secondary_unique, 
        variables=var_3hour,
        temporal=temporal
    )

    # 3. Merge the two dictionaries into one master dataset
    # (Using Python dictionary unpacking to combine them instantly)
    all_data = {**data_primary, **data_secondary}
    all_models = models_primary + models_secondary_unique

    print("\n=============================================")
    print(f" Master dataset created with {len(all_models)} total models.")
    print("=============================================\n")

    if save_pickle:
        output_base_dir = './Data/var_files/original/3hourly/'
        os.makedirs(output_base_dir, exist_ok=True)
        print(f"\n📦 Saving entire dataset to a single PICKLE file in {output_base_dir} ...")
        out_file = os.path.join(output_base_dir, f"{temporal}_aerocom_data.pickle")
        
        try:
            with open(out_file, 'wb') as f:
                pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✅ Save Complete: Master dictionary saved to {out_file}")
        except Exception as e:
            print(f"❌ Error saving pickle file: {e}")
    if save_nc:
        aerocom_data.save_model_data_to_netcdf(all_data, output_base_dir="./Data/AP3_processed_3hourly")


# prect = aerocom_data.get_data(path_original, models_AOD, 'prect')
# functions.save_pickle_files(save_path, 'prect.pickle', prect)
# print('prect done')

#???
# colors_AOD = aerocom_data.get_data(path_original, models_AOD, 'AOD550')
# functions.save_pickle_files(save_path, 'colors_AOD.pickle', colors_AOD)
# colors_AAOD = aerocom_data.get_data(path_original, models_AAOD, 'AOD550')
# functions.save_pickle_files(save_path, 'colors_AAOD.pickle', colors_AAOD)


###################### AVERAGE AND SAVE FILES ######################
# Regional means and derived variables (MEC, SSA, lifetime, …) are computed in
# notebooks using functions.create_region_mask() and functions.regional_aggregate()
# on the monthly data (now loaded from Data/AP3_processed_monthly/ NetCDF files,
# or from the backward-compatible pickle when SAVE_PICKLE=True). Uncomment below to
# regenerate the legacy pre-averaged pickles in save_path_average.

# print('starting calculation of the regional means')
# emi_mean = functions.apply_global_mean(emi,'emi_total', False)
# functions.save_pickle_files(save_path_average, 'emi_total.pickle', emi_mean)
# emi_BC_OA_mean = functions.apply_global_mean(emi_BC_OA,'emi_BC_OA', False)
# functions.save_pickle_files(save_path_average, 'emi_BC_OA.pickle', emi_BC_OA_mean)
# emi_BC_mean = functions.apply_global_mean(emi_BC,'emi_bc', False)
# functions.save_pickle_files(save_path_average, 'emi_BC.pickle', emi_BC_mean)
# emi_OA_mean = functions.apply_global_mean(emi_OA,'emi_oa', False)
# functions.save_pickle_files(save_path_average, 'emi_OA.pickle', emi_OA_mean)
# print('emissions done')

# load_mean = functions.apply_global_mean(load,'load_total', False)
# functions.save_pickle_files(save_path_average, 'load_total.pickle', load_mean)
# load_BC_OA_mean = functions.apply_global_mean(load_BC_OA,'load_BC_OA', False)
# functions.save_pickle_files(save_path_average, 'load_BC_OA.pickle', load_BC_OA_mean)
# load_BC_mean = functions.apply_global_mean(load_BC,'load_bc', False)
# functions.save_pickle_files(save_path_average, 'load_BC.pickle', load_BC_mean)
# load_OA_mean = functions.apply_global_mean(load_OA,'load_oa', False)
# functions.save_pickle_files(save_path_average, 'load_OA.pickle', load_OA_mean)
# print('load done')

# od550_mean = functions.apply_global_mean(od550, 'AOD550', False)
# functions.save_pickle_files(save_path_average, 'od550cs.pickle', od550_mean)
# od550_for_SSA = functions.apply_global_mean(aerocom_data.get_data(path_original, models_AAOD, 'AOD550'),
#                                                   'AOD550', False)
# od_other_mean = functions.apply_global_mean(od_other, 'AOD440', False)
# functions.save_pickle_files(save_path_average, 'od_other.pickle', od_other_mean)
# abs550_mean = functions.apply_global_mean(abs550, 'AAOD550', False)
# functions.save_pickle_files(save_path_average, 'abs550.pickle', abs550_mean)
# print('optical depth done')

# prect_mean = functions.apply_global_mean(prect,'prect', False)
# functions.save_pickle_files(save_path_average, 'prect.pickle', prect_mean)

# MEC = aerocom_data.calculate_var(od550_mean, load_mean, 'MEC')
# functions.save_pickle_files(save_path_average, 'MEC.pickle', MEC)
# MAC = aerocom_data.calculate_var(abs550_mean, load_BC_OA_mean, 'MAC')
# functions.save_pickle_files(save_path_average, 'MAC.pickle', MAC)

# lifetime = aerocom_data.calculate_var(load_mean, emi_mean, 'lifetime', False)
# functions.save_pickle_files(save_path_average, 'lifetime.pickle', lifetime)
# lifetime_BC_OA = aerocom_data.calculate_var(load_BC_OA_mean, emi_BC_OA_mean, 'lifetime', False)
# functions.save_pickle_files(save_path_average, 'lifetime_BC_OA.pickle', lifetime_BC_OA)
# lifetime_inv = aerocom_data.calculate_var(load_mean, emi_mean, 'lifetime', True)
# functions.save_pickle_files(save_path_average, 'lifetime_inv.pickle', lifetime_inv)
# lifetime_inv_BC_OA = aerocom_data.calculate_var(load_BC_OA_mean, emi_BC_OA_mean, 'lifetime', True)
# functions.save_pickle_files(save_path_average, 'lifetime_inv_BC_OA.pickle', lifetime_inv_BC_OA)

# AE = aerocom_data.calculate_var(od550_mean, od_other_mean, 'AE')
# functions.save_pickle_files(save_path_average, 'AE.pickle', AE)
# SSA = aerocom_data.calculate_var(abs550_mean, od550_for_SSA, 'SSA')
# functions.save_pickle_files(save_path_average, 'SSA.pickle', SSA)
