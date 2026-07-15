""" Script to get the data from the AeroCom files.
These variables are saved as pickle files, so it is easier and faster to read them.

BEFORE RUNNING THIS FILE, CHECK:
    - WHERE THE FILES ARE BEING SAVED (save_path / save_path_average)
    - WHICH FILES ARE BEING OPENED (path_original / path_regrid)

FRdM, 27th of September 2024 """


###################### IMPORT MODULES ######################

import sys
import os
from pathlib import Path

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
models_AOD = ['CAM5.3-Oslo_AP3-CTRL2016-PD', 'ECHAM6-HAM2_AP3-CTRL2016-PD', 'ECHAM6-SALSA_CTRL2016-PD',
              'ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD', 'TM5_AP3-CTRL2016',
              'CAM5-ATRAS_AP3-CTRL',  'EC-Earth3-AerChem-met2010_AP3-CTRL2019', 'ECHAM6.3-HAM2.3-met2010_AP3-CTRL',
              'ECHAM6.3-SALSA2.0-met2010_AP3-CTRL', 'GEOS-i33p2-met2010_AP3-CTRL', 'GFDL-AM4-met2010_AP3-CTRL',
              'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010', 'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010', 'INCA_AP3-CTRL',
              'NorESM2-met2010_AP3-CTRL', 'MIROC-SPRINTARS_AP3-CTRL', 'TM5-met2010_AP3-CTRL2019']
models_AAOD = ['CAM5.3-Oslo_AP3-CTRL2016-PD', 'CAM5_CTRL2016', 'CAM5-ATRAS_AP3-CTRL',
               'EC-Earth3-AerChem-met2010_AP3-CTRL2019', 'ECHAM6-HAM2_AP3-CTRL2016-PD',
               'ECHAM6.3-HAM2.3-met2010_AP3-CTRL', 'ECHAM6-SALSA_CTRL2016-PD',  'ECHAM6.3-SALSA2.0-met2010_AP3-CTRL',
               'ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD', 'GEOS-i33p2-met2010_AP3-CTRL', 'GFDL-AM4-met2010_AP3-CTRL',
               'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010', 'INCA_AP3-CTRL', 'SPRINTARS-T213_AP3-CTRL2016-PD',
               'MIROC-SPRINTARS_AP3-CTRL', 'TM5_AP3-CTRL2016', 'TM5-met2010_AP3-CTRL2019']
models_3hourly = ['CAM5.3-Oslo_AP3-CTRL2016-PD', 'CAM5_CTRL2016', 'ECHAM6-SALSA_CTRL2016-PD', 'ECHAM6-HAM2_AP3-CTRL2016-PD',
                  'ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD', 'GEOS-i33p2-met2010_AP3-CTRL',
                  'SPRINTARS-T213_AP3-CTRL2016-PD', 'MIROC-SPRINTARS_AP3-CTRL', 'TM5_AP3-CTRL2016']
###double check 
# Models in AOD but missing from AAOD
missing_from_aaod = list(set(models_AOD) - set(models_AAOD))
print("Missing from AAOD:", missing_from_aaod)
# Output: ['GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010', 'NorESM2-met2010_AP3-CTRL']

# Models in AAOD but missing from AOD (Bonus check!)
missing_from_aod = list(set(models_AAOD) - set(models_AOD))
print("Missing from AOD:", missing_from_aod)
###########################################
# Output: ['SPRINTARS-T213_AP3-CTRL2016-PD', 'CAM5_CTRL2016']
# path_regrid = './Data/AEROCOM_III_regrid/{}/aerocom3_{}_{}_{}__{}.nc'
path_original = './Data/AEROCOM_III/{}/aerocom3_{}_{}_{}_2010_{}.nc'

save_path = f'./Data/var_files/original/{temporal}/'
save_path_average = './Data/var_files/original/average/'


###################### CALCULATE VARIABLES AND SAVE FILES ######################

# for every model, get the whole dictionary containing only the data we want

# GET monthly AAOD data
emi = aerocom_data.get_data(path_original, models_AOD, 'emi_total')
functions.save_pickle_files(save_path, 'emi_total.pickle', emi)
emi_BC_OA = aerocom_data.get_data(path_original, models_AAOD, 'emi_BC_OA',od550_freq = temporal, odother_freq = temporal, abs550_freq = temporal)
functions.save_pickle_files(save_path, 'emi_BC_OA.pickle', emi_BC_OA, )
emi_BC = aerocom_data.get_data(path_original, models_AAOD, 'emi_bc',od550_freq = temporal, odother_freq = temporal, abs550_freq = temporal)
functions.save_pickle_files(save_path, 'emi_BC.pickle', emi_BC)
emi_OA = aerocom_data.get_data(path_original, models_AAOD, 'emi_oa',od550_freq = temporal, odother_freq = temporal, abs550_freq = temporal)
functions.save_pickle_files(save_path, 'emi_OA.pickle', emi_OA)
print('emissions done')

# GET monthly AOD data
load = aerocom_data.get_data(path_original, models_AOD, 'load_total')
functions.save_pickle_files(save_path, 'load_total.pickle', load)
load_BC_OA = aerocom_data.get_data(path_original, models_AAOD, 'load_BC_OA')
functions.save_pickle_files(save_path, 'load_BC_OA.pickle', load_BC_OA)
load_BC = aerocom_data.get_data(path_original, models_AAOD, 'load_bc')
functions.save_pickle_files(save_path, 'load_BC.pickle', load_BC)
load_OA = aerocom_data.get_data(path_original, models_AAOD, 'load_oa')
functions.save_pickle_files(save_path, 'load_OA.pickle', load_OA)
print('load done')

# # GET 3hourly AOD data   
# what is optical depth 1
od550 = aerocom_data.get_data(path_original, models_AOD, 'AOD550', od550_freq=temporal)
# od550cs = aerocom_data.get_data(path_original, models_AOD, 'AOD550')
functions.save_pickle_files(save_path, 'od550.pickle', od550)
od_other = aerocom_data.get_data(path_original, models_AOD, 'AOD440', od550_freq=temporal, odother_freq=temporal)
functions.save_pickle_files(save_path, 'od_other.pickle', od_other)
abs550 = aerocom_data.get_data(path_original, models_AAOD, 'AAOD550', od550_freq=temporal, odother_freq=temporal, abs550_freq=temporal)
functions.save_pickle_files(save_path, 'abs550.pickle', abs550)
print('optical depth done')

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
# on the monthly pickles saved above (save_path). Uncomment below to regenerate
# the legacy pre-averaged pickles in save_path_average.

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
