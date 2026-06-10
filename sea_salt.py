"""
Script to make a scatter plot of the annual globally sea salt of MEC vs AE

FRdM, 17th June 2024
"""

# IMPORT MODULES
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# import files
# import funct_plot
import funct_geo_tools

import importlib
# importlib.reload(funct_plot)
importlib.reload(funct_geo_tools)



###################### DEFINE FUNCTIONS ######################
# (meant for dictionaries)
def average(dict, virtual_var):

    """
        Applies the function 'funct_geo_tools.global_mean1' to every model in the given dictionary.

        Params:
            dict (dictionary): dictionary containing the data
            virtual_var (str): name of the virtual variable to calculate (ex: 'emi_total', see data_ver_regrid.x for the names)

        Returns:
            Dictionary with the corresponding monthly averages (for each region)
    """

    # Initialize the result dictionary
    result1 = {model: {} for model in dict.keys()}

    # Loop through each model and mask to calculate the mean
    for model, model_data in dict.items():
        result1[model] = funct_geo_tools.global_mean1(model_data[virtual_var], model_data['masks']['region'])

    return result1

def calculate_var(dict1, dict2, var='MEC'):

    """
        Calculates MEC (m2/g), 1/lifetime (1/day), lifetime (day) or AE.
        If the units of the inputs change, the conversion factors also need to change!!

        Params:
            dict1, dict2 (dict): dictionaries containing the data. Every element has to be a xarray defined over time.
            var (string): variable to calculate. It has to be MEC, lifetime_inv or AE.

        Returns:
            Dictionary containing the desired variable for every region.
    """

    assert var in ['MEC', 'lifetime_inv', 'AE', 'lifetime'], f'var must be either MEC, lifetime_inv or AE and not {var}'

    # Define the models that use 870nm
    ae_models = {'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010', 'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010', 'ECHAM6.3-HAM2.3-met2010_AP3-CTRL'}

    result = {}

    for model in dict1.keys():
        result[model] = {}

        if var == 'MEC':
            result[model] = dict1[model] / (dict2[model]*1e3)

        elif var == 'lifetime_inv':
            result[model] = (dict1[model]*3600*24) / dict2[model]

        elif var == 'lifetime':
            result[model] = dict1[model] / (dict2[model]*3600*24)

        else:
            if model in ae_models:
                result[model] = - np.log(dict1[model] / dict2[model]) / np.log(550/870)

            else:
                result[model] = - np.log(dict1[model] / dict2[model]) / np.log(550/440)

    return result



###################### OPEN DATA ######################
S = 'Surface'
C = 'Column'
M = 'monthly'
emi_units = 1           # units: kg/m2s ; 1e9*3600*24*365 units: g/km2 yr
load_units = 1          # units: kg/m2 ; 1e3 units: g/m2
pr_units = 3600*24      # units: mm/day

# define dictionary to fill in with the data
x = {
    'CAM5-ATRAS_AP3-CTRL': {'emi':
                                {'emiss': [C, emi_units, M]},
                            'emi_ss': 0,
                            'load':
                                {'loadss': [C, load_units, M]},
                            'load_ss': 0,
                            'od1':
                                {'od550aer': [C, M]},
                            'od2':
                                {'od440aer': [C, M]},
                            'optical_depth_1': 0,
                            'optical_depth_2': 0,
                            'precipitation':
                                {'pr': [S, pr_units, M]},
                            'prect': 0,
                            'masks':
                                {'region': 0}
                            },

    'CAM5.3-Oslo_AP3-CTRL2016-PD': {'emi':
                                        {'emiss': [S, emi_units, M]},
                                    'emi_ss': 0,
                                    'load':
                                        {'loadss': [C, load_units, M]},
                                    'load_ss': 0,
                                    'od1':
                                        {'od550csaer': [C, M]},
                                    'od2':
                                        {'od440csaer': [C, M]},
                                    'optical_depth_1': 0,
                                    'optical_depth_2': 0,
                                    'precipitation':
                                        {'precip': ['ModelLevel', pr_units, M]},
                                    'prect': 0,
                                    'masks':
                                        {'region': 0}
                                    },

    'EC-Earth3-AerChem-met2010_AP3-CTRL2019': {'emi':
                                                   {'emiss': [C, emi_units, M]},
                                               'emi_ss': 0,
                                               'load':
                                                   {'loadss': [C, load_units, M]},
                                               'load_ss': 0,
                                               'od1':
                                                   {'od550aer': [C, M]},
                                               'od2':
                                                   {'od440aer': [C, M]},
                                               'optical_depth_1': 0,
                                               'optical_depth_2': 0,
                                               'precipitation':
                                                   {'pr': [S, pr_units, M]},
                                               'prect': 0,
                                               'masks':
                                                   {'region': 0}
                                               },

     'ECHAM6-HAM2_AP3-CTRL2016-PD': {'emi':
                                         {'emiss': [S, emi_units, M]},
                                     'emi_ss': 0,
                                     'load':
                                         {'loadss': [C, load_units, M]},
                                     'load_ss': 0,
                                     'od1':
                                         {'od550aer': [C, M]},
                                     'od2':
                                         {'od440aer': [C, M]},
                                     'optical_depth_1': 0,
                                     'optical_depth_2': 0,
                                     'precipitation':
                                         {'precip': [S, pr_units, M]},
                                     'prect': 0,
                                     'masks':
                                         {'region': 0}
                                     },

    'ECHAM6.3-HAM2.3-met2010_AP3-CTRL': {'emi':
                                             {'emiss': [S, emi_units, M]},
                                         'emi_ss': 0,
                                         'load':
                                             {'loadss': [C, load_units, M]},
                                         'load_ss': 0,
                                         'od1':
                                             {'od550csaer': [C, M]},
                                         'od2':
                                             {'od870aer': [C, M]},
                                         'optical_depth_1': 0,
                                         'optical_depth_2': 0,
                                         'precipitation':
                                             {'pr': [S, pr_units, M]},
                                         'prect': 0,
                                         'masks':
                                             {'region': 0}
                                         },

    'ECHAM6-SALSA_CTRL2016-PD': {'emi':
                                     {'emiss': [S, emi_units, M]},
                                 'emi_ss': 0,
                                 'load':
                                     {'loadss': [C, load_units, M]},
                                 'load_ss': 0,
                                 'od1':
                                     {'od550aer': [C, M]},
                                 'od2':
                                     {'od440aer': [C, M]},
                                 'optical_depth_1': 0,
                                 'optical_depth_2': 0,
                                 'precipitation':
                                     {'precip': [S, pr_units, M]},
                                 'prect': 0,
                                 'masks':
                                     {'region': 0}
                                 },

    'ECHAM6.3-SALSA2.0-met2010_AP3-CTRL': {'emi':
                                               {'emiss': [S, emi_units, M]},
                                           'emi_ss': 0,
                                           'load':
                                               {'loadss': [C, load_units, M]},
                                           'load_ss': 0,
                                           'od1':
                                               {'od550csaer': [C, M]},
                                           'od2':
                                               {'od440aer': [C, M]},
                                           'optical_depth_1': 0,
                                           'optical_depth_2': 0,
                                           'precipitation':
                                               {'pr': [S, pr_units, M]},
                                           'prect': 0,
                                           'masks':
                                               {'region': 0}
                                           },

    'ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD': {'emi':
                                                          {'emiss': [C, emi_units, M]},
                                                      'emi_ss': 0,
                                                      'load':
                                                          {'loadss': [C, load_units, M]},
                                                      'load_ss': 0,
                                                      'od1':
                                                          {'od550aer': [C, M]},
                                                      'od2':
                                                          {'od440aer': [C, M]},
                                                      'optical_depth_1': 0,
                                                      'optical_depth_2': 0,
                                                      'precipitation':
                                                          {'precip': [S, pr_units, M]},
                                                      'prect': 0,
                                                      'masks':
                                                          {'region': 0}
                                                      },

    'GEOS-i33p2-met2010_AP3-CTRL': {'emi':
                                        {'emiss': [S, emi_units, M]},
                                    'emi_ss': 0,
                                    'load':
                                        {'loadss': [C, load_units, M]},
                                    'load_ss': 0,
                                    'od1':
                                        {'od550aer': [C, M]},
                                    'od2':
                                        {'od440aer': [C, M]},
                                    'optical_depth_1': 0,
                                    'optical_depth_2': 0,
                                    'precipitation':
                                        {'pr': [S, pr_units, M]},
                                    'prect': 0,
                                    'masks':
                                        {'region': 0}
                                    },

    'GFDL-AM4-met2010_AP3-CTRL': {'emi':
                                      {'emiss': [S, emi_units, M]},
                                  'emi_ss': 0,
                                  'load':
                                      {'loadss': [C, load_units, M]},
                                  'load_ss': 0,
                                  'od1':
                                      {'od550aer': [C, M]},
                                  'od2':
                                      {'od440aer': [C, M]},
                                  'optical_depth_1': 0,
                                  'optical_depth_2': 0,
                                  'precipitation':
                                      {'pr': [S, pr_units, M]},
                                  'prect': 0,
                                  'masks':
                                      {'region': 0}
                                  },

    'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010': {'emi':
                                                  {'emiss': [S, emi_units, M]},
                                              'emi_ss': 0,
                                              'load':
                                                  {'loadss': [C, load_units, M]},
                                              'load_ss': 0,
                                              'od1':
                                                  {'od550csaer': [C, M]},
                                              'od2':
                                                  {'od870csaer': [C, M]},
                                              'optical_depth_1': 0,
                                              'optical_depth_2': 0,
                                              'precipitation':
                                                  {'pr': [C, pr_units, M]},
                                              'prect': 0,
                                              'masks':
                                                  {'region': 0}
                                              },

    'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010': {'emi':
                                               {'emiss': [S, emi_units, M]},
                                           'emi_ss': 0,
                                           'load':
                                               {'loadss': [C, load_units, M]},
                                           'load_ss': 0,
                                           'od1':
                                               {'od550csaer': [C, M]},
                                           'od2':
                                               {'od870csaer': [C, M]},
                                           'optical_depth_1': 0,
                                           'optical_depth_2': 0,
                                           'precipitation':
                                               {'pr': [C, pr_units, M]},
                                           'prect': 0,
                                           'masks':
                                               {'region': 0}
                                           },

    'INCA_AP3-CTRL': {'emi':
                          {'emiss': [S, emi_units, M]},
                      'emi_ss': 0,
                      'load':
                          {'loadss': [C, load_units, M]},
                      'load_ss': 0,
                      'od1':
                          {'od550aer': [C, M]},
                      'od2':
                          {'od440aer': [C, M]},
                      'optical_depth_1': 0,
                      'optical_depth_2': 0,
                      'precipitation':
                          {'pr': [S, pr_units, M]},
                      'prect': 0,
                      'masks':
                          {'region': 0}
                      },

    'NorESM2-met2010_AP3-CTRL': {'emi':
                                     {'emiss': [S, emi_units, M]},
                                 'emi_ss': 0,
                                 'load':
                                     {'loadss': [C, load_units, M]},
                                 'load_ss': 0,
                                 'od1':
                                     {'od550csaer': [C, M]},
                                 'od2':
                                     {'od440csaer': [C, M]},
                                 'optical_depth_1': 0,
                                 'optical_depth_2': 0,
                                 'precipitation':
                                     {'precip': ['ModelLevel', pr_units, M]},
                                 'prect': 0,
                                 'masks':
                                     {'region': 0}
                                 },

    'MIROC-SPRINTARS_AP3-CTRL': {'emi':
                                     {'emiss': [C, emi_units, M]},
                                 'emi_ss': 0,
                                 'load':
                                     {'loadss': [C, load_units, M]},
                                 'load_ss': 0,
                                 'od1':
                                     {'od550aer': [C, M]},
                                 'od2':
                                     {'od440aer': [C, M]},
                                 'optical_depth_1': 0,
                                 'optical_depth_2': 0,
                                 'precipitation':
                                     {'pr': [S, pr_units, M]},
                                 'prect': 0,
                                 'masks':
                                     {'region': 0}
                                 },

    'TM5_AP3-CTRL2016': {'emi':
                             {'emiss': [C, emi_units, M]},
                         'emi_ss': 0,
                         'load':
                             {'loadss': [C, load_units, M]},
                         'load_ss': 0,
                         'od1':
                             {'od550aer': [C, M]},
                         'od2':
                             {'od440aer': [C, M]},
                         'optical_depth_1': 0,
                         'optical_depth_2': 0,
                         'precipitation':
                             {'pr': [S, pr_units, M]},
                         'prect': 0,
                         'masks':
                             {'region': 0}
                         },

    'TM5-met2010_AP3-CTRL2019': {'emi':
                                     {'emiss': [C, emi_units, M]},
                                 'emi_ss': 0,
                                 'load':
                                     {'loadss': [C, load_units, M]},
                                 'load_ss': 0,
                                 'od1':
                                     {'od550aer': [C, M]},
                                 'od2':
                                     {'od440aer': [C, M]},
                                 'optical_depth_1': 0,
                                 'optical_depth_2': 0,
                                 'precipitation':
                                     {'pr': [S, pr_units, M]},
                                 'prect': 0,
                                 'masks':
                                     {'region': 0}
                                 }
    }

path = '/Users/fionaromandemiguel/Documents/VU/data/AeroCom_data/data_regrid/{}/aerocom3_{}_{}_{}_2010_{}.nc'

for model in x.keys():
    print('model:', model)

    for var in x[model]['emi']:
        data = funct_geo_tools.shift360(
        xr.open_dataset(path.format(model, model, var, x[model]['emi'][var][0], x[model]['emi'][var][2]) ) )
        x[model]['emi_ss'] = (data[var]*x[model]['emi'][var][1]) #.transpose('lat', 'lon', 'time')

    for var in x[model]['load']:
        data = funct_geo_tools.shift360(
        xr.open_dataset(path.format(model, model, var, x[model]['load'][var][0], x[model]['load'][var][2]) ) )
        x[model]['load_ss'] = (data[var]*x[model]['load'][var][1]) #.transpose('lat', 'lon', 'time')

    for var in x[model]['od1']:
        x[model]['optical_depth_1'] = (funct_geo_tools.shift360(
            xr.open_dataset(path.format(model, model, var, x[model]['od1'][var][0], x[model]['od1'][var][1])))[
            var])  # .transpose('lat', 'lon', 'time')

    for var in x[model]['od2']:
        x[model]['optical_depth_2'] = (funct_geo_tools.shift360(
            xr.open_dataset(path.format(model, model, var, x[model]['od2'][var][0], x[model]['od2'][var][1])))[
            var])  # .transpose('lat', 'lon', 'time')

    for var in x[model]['precipitation']:
        data = funct_geo_tools.shift360(
        xr.open_dataset(path.format(model, model, var, x[model]['precipitation'][var][0], x[model]['precipitation'][var][2]) ) )
        x[model]['prect'] = (data[var]*x[model]['precipitation'][var][1]) #.transpose('lat', 'lon', 'time')

        x[model]['masks']['region'] = xr.ones_like(x[model]['emi_ss'].isel(time=0))



###################### CALCULATE VARIABLES ######################

emi_mean = average(x, 'emi_ss')
burden_mean1 = average(x, 'load_ss')
od550aer_mean1 = average(x, 'optical_depth_1')
od_other_mean1 = average(x, 'optical_depth_2')
prect_mean1 = average(x, 'prect')

# MEC = optical_depth / burden, units: m2/g
MEC = calculate_var(od550aer_mean1, burden_mean1, 'MEC')

# 1/lifetime = emissions / burden, units: 1/day
lifetime = calculate_var(burden_mean1, emi_mean, 'lifetime')
lifetime_inv = calculate_var(emi_mean, burden_mean1, 'lifetime_inv')

# AE = - log(od1/od2) / log(lambda1/lambda2)
AE1 = calculate_var(od550aer_mean1, od_other_mean1, 'AE')

a = []
b = []
for i in range(len(od550aer_mean1['ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD'])):
    a.append((- np.log(od550aer_mean1['ECHAM6-HAM2_AP3-CTRL2016-PD'][i] / od_other_mean1['ECHAM6-HAM2_AP3-CTRL2016-PD'][i]) / np.log(550/440)).values)
    b.append((- np.log(od550aer_mean1['ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD'][i] / od_other_mean1['ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD'][i]) / np.log(550/440)).values)

AE1['ECHAM6-HAM2_AP3-CTRL2016-PD'] = a
AE1['ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD'] = b



###################### MAKE PLOT ######################
from mycolorpy import colorlist as mcp
colors = mcp.gen_color(cmap='tab20', n=17)

plt.figure(figsize=(8,6))
for index, model in enumerate(AE1.keys()):
    # print(model)
    plt.scatter(lifetime_inv[model], prect_mean1[model], color=colors[index], label=model)
plt.xlabel('AE')
plt.ylabel('MEC [m2/g]')
plt.title('Global monthly sea salt')
# plt.legend()

plt.figure()
for index, model in enumerate (emi_mean.keys()):
    plt.plot(emi_mean[model], color=colors[index], label=model)
plt.xlabel('Month in 2010', fontsize=13)
plt.xticks(fontsize=13)
plt.ylabel('Global emissions [$kg / m^2 yr$]', fontsize=13)
plt.yticks(fontsize=13)
plt.title('Time series sea salt', fontsize=16)
# plt.legend(facecolor='white', framealpha=1)

