""" Script to get the data from the AeroCom files.
These variables are saved as pickle files, so it is easier and faster to read them.

BEFORE RUNNING THIS FILE, CHECK:
    - WHERE THE FILES ARE BEING SAVED (save_path / save_path_average)
    - WHICH FILES ARE BEING OPENED (path_original / path_regrid)

FRdM, 27th of September 2024 """


###################### IMPORT MODULES ######################

from AeroCom import aerocom_data
from general_scripts import functions

import importlib
importlib.reload(aerocom_data)
importlib.reload(functions)


###################### DEFINE STUFF ######################

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

path_regrid = '/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/AeroCom_data/data_regrid/{}/aerocom3_{}_{}_{}_2010_{}.nc'
path_original = '/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/AeroCom_data/data_original/{}/aerocom3_{}_{}_{}_2010_{}.nc'

save_path = '/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/var_files/original/3hourly/'
save_path_average = '/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/var_files/regrid/0average/'


###################### CALCULATE VARIABLES AND SAVE FILES ######################

# for every model, get the whole dictionary containing only the data we want

emi = aerocom_data.get_data(path_regrid, models_AOD, 'emi_total')
functions.save_pickle_files(save_path, 'emi_total.pickle', emi)
emi_BC_OA = aerocom_data.get_data(path_regrid, models_AAOD, 'emi_BC_OA')
functions.save_pickle_files(save_path, 'emi_BC_OA.pickle', emi_BC_OA)
emi_BC = aerocom_data.get_data(path_regrid, models_AAOD, 'emi_bc')
functions.save_pickle_files(save_path, 'emi_BC.pickle', emi_BC)
emi_OA = aerocom_data.get_data(path_regrid, models_AAOD, 'emi_oa')
functions.save_pickle_files(save_path, 'emi_OA.pickle', emi_OA)
print('emissions done')

load = aerocom_data.get_data(path_regrid, models_AOD, 'load_total')
functions.save_pickle_files(save_path, 'load_total.pickle', load)
load_BC_OA = aerocom_data.get_data(path_regrid, models_AAOD, 'load_BC_OA')
functions.save_pickle_files(save_path, 'load_BC_OA.pickle', load_BC_OA)
load_BC = aerocom_data.get_data(path_regrid, models_AAOD, 'load_bc')
functions.save_pickle_files(save_path, 'load_BC.pickle', load_BC)
load_OA = aerocom_data.get_data(path_regrid, models_AAOD, 'load_oa')
functions.save_pickle_files(save_path, 'load_OA.pickle', load_OA)
print('load done')

od550 = aerocom_data.get_data(path_original, models_3hourly, 'optical_depth_1', od550_freq='3hourly')
# od550cs = aerocom_data.get_data(path_regrid, models_AOD, 'optical_depth_1')
functions.save_pickle_files(save_path, 'od550.pickle', od550)
od_other = aerocom_data.get_data(path_original, models_3hourly, 'optical_depth_2', od550_freq='monthly', odother_freq='3hourly')
functions.save_pickle_files(save_path, 'od_other.pickle', od_other)
abs550 = aerocom_data.get_data(path_original, models_3hourly, 'abs_optical_depth', od550_freq='monthly', odother_freq='monthly', abs550_freq='3hourly')
functions.save_pickle_files(save_path, 'abs550.pickle', abs550)
print('optical depth done')

prect = aerocom_data.get_data(path_regrid, models_AOD, 'prect')
functions.save_pickle_files(save_path, 'prect.pickle', prect)
print('prect done')

colors_AOD = aerocom_data.get_data(path_regrid, models_AOD, 'optical_depth_1')
functions.save_pickle_files(save_path, 'colors_AOD.pickle', colors_AOD)
colors_AAOD = aerocom_data.get_data(path_regrid, models_AAOD, 'optical_depth_1')
functions.save_pickle_files(save_path, 'colors_AAOD.pickle', colors_AAOD)

print('starting calculation of the regional means')


###################### AVERAGE AND SAVE FILES ######################

# for every region, for every model, only one value

emi_mean = functions.apply_global_mean(emi,'emi_total', False)
functions.save_pickle_files(save_path_average, 'emi_total.pickle', emi_mean)
emi_BC_OA_mean = functions.apply_global_mean(emi_BC_OA,'emi_BC_OA', False)
functions.save_pickle_files(save_path_average, 'emi_BC_OA.pickle', emi_BC_OA_mean)
emi_BC_mean = functions.apply_global_mean(emi_BC,'emi_bc', False)
functions.save_pickle_files(save_path_average, 'emi_BC.pickle', emi_BC_mean)
emi_OA_mean = functions.apply_global_mean(emi_OA,'emi_oa', False)
functions.save_pickle_files(save_path_average, 'emi_OA.pickle', emi_OA_mean)
print('emissions done')

load_mean = functions.apply_global_mean(load,'load_total', False)
functions.save_pickle_files(save_path_average, 'load_total.pickle', load_mean)
load_BC_OA_mean = functions.apply_global_mean(load_BC_OA,'load_BC_OA', False)
functions.save_pickle_files(save_path_average, 'load_BC_OA.pickle', load_BC_OA_mean)
load_BC_mean = functions.apply_global_mean(load_BC,'load_bc', False)
functions.save_pickle_files(save_path_average, 'load_BC.pickle', load_BC_mean)
load_OA_mean = functions.apply_global_mean(load_OA,'load_oa', False)
functions.save_pickle_files(save_path_average, 'load_OA.pickle', load_OA_mean)
print('load done')

od550_mean = functions.apply_global_mean(od550, 'optical_depth_1', False)
functions.save_pickle_files(save_path_average, 'od550cs.pickle', od550_mean)
od550_for_SSA = functions.apply_global_mean(aerocom_data.get_data(path_regrid, models_AAOD, 'optical_depth_1'),
                                                  'optical_depth_1', False)
od_other_mean = functions.apply_global_mean(od_other, 'optical_depth_2', False)
functions.save_pickle_files(save_path_average, 'od_other.pickle', od_other_mean)
abs550_mean = functions.apply_global_mean(abs550, 'abs_optical_depth', False)
functions.save_pickle_files(save_path_average, 'abs550.pickle', abs550_mean)
print('optical depth done')

prect_mean = functions.apply_global_mean(prect,'prect', False)
functions.save_pickle_files(save_path_average, 'prect.pickle', prect_mean)

MEC = aerocom_data.calculate_var(od550_mean, load_mean, 'MEC')
functions.save_pickle_files(save_path_average, 'MEC.pickle', MEC)
MAC = aerocom_data.calculate_var(abs550_mean, load_BC_OA_mean, 'MAC')
functions.save_pickle_files(save_path_average, 'MAC.pickle', MAC)

lifetime = aerocom_data.calculate_var(load_mean, emi_mean, 'lifetime', False)
functions.save_pickle_files(save_path_average, 'lifetime.pickle', lifetime)
lifetime_BC_OA = aerocom_data.calculate_var(load_BC_OA_mean, emi_BC_OA_mean, 'lifetime', False)
functions.save_pickle_files(save_path_average, 'lifetime_BC_OA.pickle', lifetime_BC_OA)
lifetime_inv = aerocom_data.calculate_var(load_mean, emi_mean, 'lifetime', True)
functions.save_pickle_files(save_path_average, 'lifetime_inv.pickle', lifetime_inv)
lifetime_inv_BC_OA = aerocom_data.calculate_var(load_BC_OA_mean, emi_BC_OA_mean, 'lifetime', True)
functions.save_pickle_files(save_path_average, 'lifetime_inv_BC_OA.pickle', lifetime_inv_BC_OA)

AE = aerocom_data.calculate_var(od550_mean, od_other_mean, 'AE')
functions.save_pickle_files(save_path_average, 'AE.pickle', AE)
SSA = aerocom_data.calculate_var(abs550_mean, od550_for_SSA, 'SSA')
functions.save_pickle_files(save_path_average, 'SSA.pickle', SSA)
