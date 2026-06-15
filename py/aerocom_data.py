""" Script to get the AeroCom data. This file only contains functions.
All AeroCom variables are obtained at 'get_data.py'.

Direct model output --> get_data()
    - emissions (monthly)                                       units: kg/m2s
    - load (or burden) (monthly)                                units: kg/m2
    - AOD_550, AOD_870, AOD_440 (monthly, daily or 3hourly)     units: none
    - precipitation (monthly)                                   units: mm/day

Other variables --> calculate_var()
    - MEC = AOD_550 / load                                      units: m2/g
    - MAC = AAOD_550 / load_BC_OA                               units: m2/g
    - SSA = 1 - AAOD/AOD                                        units: none
    - lifetime = load / emissions                               units: day
    - AE = - log(AOD_550/AOD_other) / log(550/other)            units: none

For every model, it also provides the masks for the different regions.
    - Amazon: lat -17,-3    lon -73,-43 (287:317)    Jul-Oct
    - Africa: lat -15,0     lon 15,37                Jun-Sep

FRdM, 7th of August 2024 """


###################### IMPORT MODULES ######################

import xarray as xr
import numpy as np
from general_scripts import functions

import importlib
importlib.reload(functions)


###################### DEFINE VARIABLES ######################

S = 'Surface'
C = 'Column'
M = 'monthly'
emi_units = 1               # units: kg/m2s ; 1e9*3600*24*365 units: g/km2 yr
load_units = 1              # units: kg/m2 ; 1e3 units: g/m2
pr_units = 3600*24          # units: mm/day


###################### DEFINE FUNCTIONS ######################

def get_data(path, models_names, var_name,
             od550_freq = 'monthly', odother_freq = 'monthly', abs550_freq = 'monthly'):

    """ Opens the AeroCom files (all times and for everywhere).
    The files are shifted to a longitude range 0,360 and converted to datetime with
    - functions.shift360()
    - functions.convert_cftime_to_datetime()

    :param path: path to the files (string).
    :param models_names: names of the models to open (list).
    :param var_name: name of the variable to get (string).
    :param od550_freq: frequency of the AOD 550 (monthly, daily or 3hourly).
    :param odother_freq: frequency of the other AOD (monthly, daily or 3hourly).
    :param abs550_freq: frequency of the AAOD (monthly, daily or 3hourly).

    :returns: dictionary with the desired data for every model.
              If one model is not requested, it deletes it from the dictionary.
    """

    valid_freq = ['monthly', 'daily', '3hourly']
    freq = {'od550_freq': od550_freq, 'odother_freq': odother_freq, 'abs550_freq': abs550_freq}

    for var_name, freq in freq.items():
        assert freq in valid_freq, f'mean must be either {", ".join(valid_freq)} and not {freq} ({var_name})'

    # assert od550_freq in ['monthly', 'daily', '3hourly'], f'mean must be either monthly, daily or 3hourly and not {od550_freq}'
    # assert odother_freq in ['monthly', 'daily', '3hourly'], f'mean must be either monthly, daily or 3hourly and not {odother_freq}'
    # assert abs550_freq in ['monthly', 'daily', '3hourly'], f'mean must be either monthly, daily or 3hourly and not {abs550_freq}'

    # define dictionary to fill in with the data
    x = {
        'CAM5-ATRAS_AP3-CTRL': {'emi':
                                    {'emibc': [C, emi_units, M],
                                     'emidust': [C, emi_units, M],
                                     'emiss': [C, emi_units, M],
                                     'emioa': [C, emi_units, M],
                                     'emiso2': [C, emi_units, M]},
                                'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                'emi_total': 0, 'emi_BC_OA': 0,
                                'load':
                                    {'loadbc': [C, load_units, M],
                                     'loaddu': [C, load_units, M],
                                     'loadss': [C, load_units, M],
                                     'loadoa': [C, load_units, M],
                                     'loadso4': [C, load_units, M]},
                                'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                'load_total': 0, 'load_BC_OA': 0,
                                'od1': {'od550aer': [C, M]},
                                'od2': {'od440aer': [C, M]},
                                'od3': {'abs550aer': [C, M]},
                                'optical_depth_1': 0,
                                'optical_depth_2': 0,
                                'abs_optical_depth': 0,
                                'precipitation': {'pr': [S, pr_units, M]},
                                'prect': 0,
                                'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                'color': '#1f77b4'},

        'CAM5.3-Oslo_AP3-CTRL2016-PD': {'emi':
                                            {'emibc': [S, emi_units, M],
                                             'emidust': [S, emi_units, M],
                                             'emiss': [S, emi_units, M],
                                             'emioa': [S, emi_units, M],
                                             'emiso2': [S, emi_units, M]},
                                        'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                        'emi_total': 0, 'emi_BC_OA': 0,
                                        'load':
                                            {'loadbc': [C, load_units, M],
                                             'loaddust': [C, load_units, M],
                                             'loadss': [C, load_units, M],
                                             'loadoa': [C, load_units, M],
                                             'loadso4': [C, load_units, M]},
                                        'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                        'load_total': 0, 'load_BC_OA': 0,
                                        'od1': {'od550aer': [C, od550_freq]},     # od550csaer
                                        'od2': {'od440aer': [C, odother_freq]},     # od440csaer
                                        'od3': {'abs550aer': [C, abs550_freq]},
                                        'optical_depth_1': 0,
                                        'optical_depth_2': 0,
                                        'abs_optical_depth': 0,
                                        'precipitation': {'precip': ['ModelLevel', pr_units, M]},
                                        'prect': 0,
                                        'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                        'color': '#aec7e8'},

        'CAM5_CTRL2016': {'emi':
                              {'emibc': [C, emi_units, M],
                               'emidust': [C, emi_units, M],
                               # 'emiss': [S, emi_units, M],
                               'emioa': [C, emi_units, M],
                               'emiso2': [C, emi_units, M]},
                          'emi_bc': 0, 'emi_dust': 0, 'emi_oa': 0, 'emi_so2': 0,
                          'emi_total': 0, 'emi_BC_OA': 0,
                          'load':
                              {'loadbc': [C, load_units, M],
                               'loaddust': [C, load_units, M],
                               'loadss': [C, load_units, M],
                               'loadoa': [C, load_units, M],
                               'loadso4': [C, load_units, M]},
                          'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                          'load_total': 0, 'load_BC_OA': 0,
                          'od1': {'od550aer': [C, od550_freq]},
                          'od2': {'od440aer': [C, odother_freq]},
                          'od3': {'abs550aer': [S, abs550_freq]},
                          'optical_depth_1': 0,
                          'optical_depth_2': 0,
                          'abs_optical_depth': 0,
                          'precipitation': {'precip': [S, pr_units, M]},
                          'prect': 0,
                          'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                          'color': '#d62728'},

        'EC-Earth3-AerChem-met2010_AP3-CTRL2019': {'emi':
                                                       {'emibc': [C, emi_units, M],
                                                        'emidust': [C, emi_units, M],
                                                        'emiss': [C, emi_units, M],
                                                        'emiso2': [C, emi_units, M],
                                                        'emioa': [C, emi_units, M]},
                                                   'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                                   'emi_total': 0, 'emi_BC_OA': 0,
                                                   'load':
                                                       {'loadbc': [C, load_units, M],
                                                        'loaddust': [C, load_units, M],
                                                        'loadss': [C, load_units, M],
                                                        'loadoa': [C, load_units, M],
                                                        'loadso4': [C, load_units, M]},
                                                   'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                                   'load_total': 0, 'load_BC_OA': 0,
                                                   'od1': {'od550aer': [C, M]},
                                                   'od2': {'od440aer': [C, M]},
                                                   'od3': {'abs550aer': [C, M]},
                                                   'optical_depth_1': 0,
                                                   'optical_depth_2': 0,
                                                   'abs_optical_depth': 0,
                                                   'precipitation': {'pr': [S, pr_units, M]},
                                                   'prect': 0,
                                                   'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                                   'color': '#ff7f0e'},

         'ECHAM6-HAM2_AP3-CTRL2016-PD': {'emi':
                                             {'emibc': [S, emi_units, M],
                                              'emidust': [S, emi_units, M],
                                              'emiss': [S, emi_units, M],
                                              'emioa': [S, emi_units, M],
                                              'emiso2': [S, emi_units, M]},
                                         'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                         'emi_total': 0, 'emi_BC_OA': 0,
                                         'load':
                                             {'loadbc': [C, load_units, M],
                                              'loaddust': [C, load_units, M],
                                              'loadss': [C, load_units, M],
                                              'loadoa': [C, load_units, M],
                                              'loadso4': [C, load_units, M]},
                                         'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                         'load_total': 0, 'load_BC_OA': 0,
                                         'od1': {'od550aer': [C, od550_freq]},
                                         'od2': {'od440aer': [C, odother_freq]},
                                         'od3': {'abs550aer': [C, abs550_freq]},
                                         'optical_depth_1': 0,
                                         'optical_depth_2': 0,
                                         'abs_optical_depth': 0,
                                         'precipitation': {'precip': [S, pr_units, M]},
                                         'prect': 0,
                                         'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                         'color': '#ffbb78'},

        'ECHAM6.3-HAM2.3-met2010_AP3-CTRL': {'emi':
                                                 {'emibc': [S, emi_units, M],
                                                  'emidust': [S, emi_units, M],
                                                  'emiss': [S, emi_units, M],
                                                  'emioa': [S, emi_units, M],
                                                  'emiso2': [S, emi_units, M]},
                                             'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                             'emi_total': 0, 'emi_BC_OA': 0,
                                             'load':
                                                 {'loadbc': [C, load_units, M],
                                                  'loaddu': [C, load_units, M],
                                                  'loadss': [C, load_units, M],
                                                  'loadoa': [C, load_units, M],
                                                  'loadso4': [C, load_units, M]},
                                             'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                             'load_total': 0, 'load_BC_OA': 0,
                                             'od1': {'od550aer': [C, M]},     # od550csaer
                                             'od2': {'od440aer': [C, M]},
                                             'od3': {'abs550aer': [C, M]},
                                             'optical_depth_1': 0,
                                             'optical_depth_2': 0,
                                             'abs_optical_depth': 0,
                                             'precipitation': {'pr': [S, pr_units, M]},
                                             'prect': 0,
                                             'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                             'color': '#2ca02c'},

        'ECHAM6-SALSA_CTRL2016-PD': {'emi':
                                         {'emibc': [S, emi_units, M],
                                          'emidust': [S, emi_units, M],
                                          'emiss': [S, emi_units, M],
                                          'emioa': [S, emi_units, M],
                                          'emiso2': [S, emi_units, M]},
                                     'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                     'emi_total': 0, 'emi_BC_OA': 0,
                                     'load':
                                         {'loadbc': [C, load_units, M],
                                          'loaddust': [C, load_units, M],
                                          'loadss': [C, load_units, M],
                                          'loadoa': [C, load_units, M],
                                          'loadso4': [C, load_units, M]},
                                     'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                     'load_total': 0, 'load_BC_OA': 0,
                                     'od1': {'od550aer': [C, od550_freq]},
                                     'od2': {'od440aer': [C, odother_freq]},
                                     'od3': {'abs550aer': [C, abs550_freq]},
                                     'optical_depth_1': 0,
                                     'optical_depth_2': 0,
                                     'abs_optical_depth': 0,
                                     'precipitation': {'precip': [S, pr_units, M]},
                                     'prect': 0,
                                     'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                     'color': '#98df8a'},

        'ECHAM6.3-SALSA2.0-met2010_AP3-CTRL': {'emi':
                                                   {'emibc': [S, emi_units, M],
                                                    'emidust': [S, emi_units, M],
                                                    'emiss': [S, emi_units, M],
                                                    'emioa': [S, emi_units, M],
                                                    'emiso2': [S, emi_units, M]},
                                               'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                               'emi_total': 0, 'emi_BC_OA': 0,
                                               'load':
                                                   {'loadbc': [C, load_units, M],
                                                    'loaddu': [C, load_units, M],
                                                    'loadss': [C, load_units, M],
                                                    'loadoa': [C, load_units, M],
                                                    'loadso4': [C, load_units, M]},
                                               'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                               'load_total': 0, 'load_BC_OA': 0,
                                               'od1': {'od550aer': [C, M]},     # od550csaer
                                               'od2': {'od440aer': [C, M]},
                                               'od3': {'abs550aer': [C, M]},
                                               'optical_depth_1': 0,
                                               'optical_depth_2': 0,
                                               'abs_optical_depth': 0,
                                               'precipitation': {'pr': [S, pr_units, M]},
                                               'prect': 0,
                                               'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                               'color': '#ff9896'},

        'ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD': {'emi':
                                                              {'emibc': [C, emi_units, M],
                                                               'emidust': [C, emi_units, M],
                                                               'emiss': [C, emi_units, M],
                                                               'emioa': [C, emi_units, M],
                                                               'emiso2': [C, emi_units, M]},
                                                          'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                                          'emi_total': 0, 'emi_BC_OA': 0,
                                                          'load':
                                                              {'loadbc': [C, load_units, M],
                                                               'loaddust': [C, load_units, M],
                                                               'loadss': [C, load_units, M],
                                                               'loadoa': [C, load_units, M],
                                                               'loadso4': [C, load_units, M]},
                                                          'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                                          'load_total': 0, 'load_BC_OA': 0,
                                                          'od1': {'od550aer': [C, od550_freq]},
                                                          'od2': {'od440aer': [C, odother_freq]},
                                                          'od3': {'abs550aer': [C, abs550_freq]},
                                                          'optical_depth_1': 0,
                                                          'optical_depth_2': 0,
                                                          'abs_optical_depth': 0,
                                                          'precipitation': {'precip': [S, pr_units, M]},
                                                          'prect': 0,
                                                          'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                                          'color': '#9467bd'},

        'GEOS-i33p2-met2010_AP3-CTRL': {'emi':
                                            {'emibc': [S, emi_units, M],
                                             'emidust': [S, emi_units, M],
                                             'emiss': [S, emi_units, M],
                                             'emioa': [S, emi_units, M],
                                             'emiso2': [S, emi_units, M]},
                                        'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                        'emi_total': 0, 'emi_BC_OA': 0,
                                        'load':
                                            {'loadbc': [C, load_units, M],
                                             'loaddu': [C, load_units, M],
                                             'loadss': [C, load_units, M],
                                             'loadoa': [C, load_units, M],
                                             'loadso4': [C, load_units, M]},
                                        'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0,
                                        'load_total': 0, 'load_BC_OA': 0,
                                        'od1': {'od550aer': [C, od550_freq]},
                                        'od2': {'od440aer': [C, odother_freq]},
                                        'od3': {'abs550aer': [C, abs550_freq]},
                                        'optical_depth_1': 0,
                                        'optical_depth_2': 0,
                                        'abs_optical_depth': 0,
                                        'precipitation': {'pr': [S, pr_units, M]},
                                        'prect': 0,
                                        'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                        'color': '#c5b0d5'},

        'GFDL-AM4-met2010_AP3-CTRL': {'emi':
                                          {'emibc': [S, emi_units, M],
                                           'emidust': [S, emi_units, M],
                                           'emiss': [S, emi_units, M],
                                           'emioa': [S, emi_units, M],
                                           'emiso2': [S, emi_units, M]},
                                      'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                      'emi_total': 0, 'emi_BC_OA': 0,
                                      'load':
                                          {'loadbc': [C, load_units, M],
                                           'loaddu': [C, load_units, M],
                                           'loadss': [C, load_units, M],
                                           'loadoa': [C, load_units, M],
                                           'loadso4': [C, load_units, M]},
                                      'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                      'load_total': 0, 'load_BC_OA': 0,
                                      'od1': {'od550aer': [C, M]},
                                      'od2': {'od440aer': [C, M]},
                                      'od3': {'abs550aer': [C, M]},
                                      'optical_depth_1': 0,
                                      'optical_depth_2': 0,
                                      'abs_optical_depth': 0,
                                      'precipitation': {'pr': [S, pr_units, M]},
                                      'prect': 0,
                                      'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                      'color': '#8c564b'},

        'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010': {'emi':
                                                      {'emibc': [S, emi_units, M],
                                                       'emidust': [S, emi_units, M],
                                                       'emiss': [S, emi_units, M],
                                                       'emioa': [S, emi_units, M],
                                                       'emiso2': [S, emi_units, M]},
                                                  'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                                  'emi_total': 0, 'emi_BC_OA': 0,
                                                  'load':
                                                      {'loadbc': [C, load_units, M],
                                                       'loaddu': [C, load_units, M],
                                                       'loadss': [C, load_units, M],
                                                       'loadoa': [C, load_units, M],
                                                       'loadso4': [C, load_units, M]},
                                                  'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                                  'load_total': 0, 'load_BC_OA': 0,
                                                  'od1': {'od550aer': [C, M]},    # od550csaer
                                                  'od2': {'od870csaer': [C, M]},
                                                  'od3': {'abs550aer': [C, M]},
                                                  'optical_depth_1': 0,
                                                  'optical_depth_2': 0,
                                                  'abs_optical_depth': 0,
                                                  'precipitation': {'pr': [C, pr_units, M]},
                                                  'prect': 0,
                                                  'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                                  'color': '#c49c94'},

        'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010': {'emi':
                                                   {'emibc': [S, emi_units, M],
                                                    'emidust': [S, emi_units, M],
                                                    'emiss': [S, emi_units, M],
                                                    'emioa': [S, emi_units, M],
                                                    'emiso2': [S, emi_units, M]},
                                               'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                               'emi_total': 0, 'emi_BC_OA': 0,
                                               'load':
                                                   {'loadbc': [C, load_units, M],
                                                    'loaddu': [C, load_units, M],
                                                    'loadss': [C, load_units, M],
                                                    'loadoa': [C, load_units, M],
                                                    'loadso4': [C, load_units, M]},
                                               'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                               'load_total': 0, 'load_BC_OA': 0,
                                               'od1': {'od550aer': [C, M]},   # od550csaer
                                               'od2': {'od870csaer': [C, M]},
                                               'od3': {'abs550aer': [C, M]},
                                               'optical_depth_1': 0,
                                               'optical_depth_2': 0,
                                               'abs_optical_depth': 0,
                                               'precipitation': {'pr': [C, pr_units, M]},
                                               'prect': 0,
                                               'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                               'color': '#e377c2'},

        'HadGEM3-GA7.1_AP3-CTRL2016-PD': {'emi':
                                              {'emibc': [S, emi_units, M],
                                               'emidust': [S, emi_units, M],
                                               'emiss': [S, emi_units, M],
                                               'emioa': [S, emi_units, M],
                                               'emiso2': [S, emi_units, M]},
                                          'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                          'emi_total': 0, 'emi_BC_OA': 0,
                                          'load':
                                              {'loadbc': [C, load_units, M],
                                               'loaddust': [C, load_units, M],
                                               'loadss': [C, load_units, M],
                                               'loadoa': [C, load_units, M],
                                               'loadso4': [C, load_units, M]},
                                          'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                          'load_total': 0, 'load_BC_OA': 0,
                                          'od1': {'od550aer': [C, od550_freq]},
                                          'od2': {'od870aer': [C, M]},
                                          'od3': {'abs550aer': [C, M]},
                                          'optical_depth_1': 0,
                                          'optical_depth_2': 0,
                                          'abs_optical_depth': 0,
                                          # 'prect': 0,
                                          'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                          'color': '#7f7f7f'},

        'INCA_AP3-CTRL': {'emi':
                              {'emibc': [S, emi_units, M],
                               'emidust': [S, emi_units, M],
                               'emiss': [S, emi_units, M],
                               'emioa': [S, emi_units, M],
                               'emiso2': [S, emi_units, M]},
                          'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                          'emi_total': 0, 'emi_BC_OA': 0,
                          'load':
                              {'loadbc': [C, load_units, M],
                               'loaddust': [C, load_units, M],
                               'loadss': [C, load_units, M],
                               'loadoa': [C, load_units, M],
                               'loadso4': [C, load_units, M]},
                          'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                          'load_total': 0, 'load_BC_OA': 0,
                          'od1': {'od550aer': [C, M]},
                          'od2': {'od440aer': [C, M]},
                          'od3': {'abs550aer': [C, M]},
                          'optical_depth_1': 0,
                          'optical_depth_2': 0,
                          'abs_optical_depth': 0,
                          'precipitation': {'pr': [S, pr_units, M]},
                          'prect': 0,
                          'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                          'color': '#c7c7c7'},

        'NorESM2-met2010_AP3-CTRL': {'emi':
                                         {'emibc': [S, emi_units, M],
                                          'emidust': [S, emi_units, M],
                                          'emiss': [S, emi_units, M],
                                          'emioa': [S, emi_units, M],
                                          'emiso2': [S, emi_units, M]},
                                     'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                     'emi_total': 0, 'emi_BC_OA': 0,
                                     'load':
                                         {'loadbc': [C, load_units, M],
                                          'loaddust': [C, load_units, M],
                                          'loadss': [C, load_units, M],
                                          'loadoa': [C, load_units, M],
                                          'loadso4': [C, load_units, M]},
                                     'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                     'load_total': 0, 'load_BC_OA': 0,
                                     'od1': {'od550aer': [C, M]},     # od550csaer
                                     'od2': {'od440csaer': [C, M]},
                                     'od3': {'abs550aer': [C, M]},
                                     'optical_depth_1': 0,
                                     'optical_depth_2': 0,
                                     'abs_optical_depth': 0,
                                     'precipitation': {'precip': ['ModelLevel', pr_units, M]},
                                     'prect': 0,
                                     'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                     'color': '#bcbd22'},

        'MIROC-SPRINTARS_AP3-CTRL': {'emi':
                                         {'emibc': [C, emi_units, M],
                                          'emidust': [C, emi_units, M],
                                          'emiss': [C, emi_units, M],
                                          'emioa': [C, emi_units, M],
                                          'emiso2': [C, emi_units, M]},
                                     'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                     'emi_total': 0, 'emi_BC_OA': 0,
                                     'load':
                                         {'loadbc': [C, load_units, M],
                                          'loaddust': [C, load_units, M],
                                          'loadss': [C, load_units, M],
                                          'loadoa': [C, load_units, M],
                                          'loadso4': [C, load_units, M]},
                                     'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                     'load_total': 0, 'load_BC_OA': 0,
                                     'od1': {'od550aer': [C, od550_freq]},
                                     'od2': {'od440aer': [C, odother_freq]},
                                     'od3': {'abs550aer': [C, abs550_freq]},
                                     'optical_depth_1': 0,
                                     'optical_depth_2': 0,
                                     'abs_optical_depth': 0,
                                     'precipitation': {'pr': [S, pr_units, M]},
                                     'prect': 0,
                                     'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                     'color': '#dbdb8d'},

        'SPRINTARS-T213_AP3-CTRL2016-PD': {'emi':
                                               {'emibc': [S, emi_units, M],
                                                'emidust': [S, emi_units, M],
                                                'emiss': [S, emi_units, M],
                                                'emioa': [S, emi_units, M],
                                                'emiso2': [S, emi_units, M]},
                                           'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                           'emi_total': 0, 'emi_BC_OA': 0,
                                           'load':
                                               {'loadbc': [S, load_units, M],
                                                'loaddust': [S, load_units, M],
                                                'loadss': [S, load_units, M],
                                                'loadoa': [S, load_units, M],
                                                'loadso4': [S, load_units, M]},
                                           'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                           'load_total': 0, 'load_BC_OA': 0,
                                           'od1': {'od550aer': [C, od550_freq]},
                                           'od2': {'od440aer': [C, odother_freq]},
                                           'od3': {'abs550aer': [C, abs550_freq]},
                                           'optical_depth_1': 0,
                                           'optical_depth_2': 0,
                                           'abs_optical_depth': 0,
                                           'precipitation': {'pr': [S, pr_units, M]},
                                           'prect': 0,
                                           'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                           'color': '#f7b6d2'},

        'TM5_AP3-CTRL2016': {'emi':
                                 {'emibc': [C, emi_units, M],
                                  'emidust': [C, emi_units, M],
                                  'emiss': [C, emi_units, M],
                                  'emioa': [C, emi_units, M],
                                  'emiso2': [C, emi_units, M]},
                             'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                             'emi_total': 0, 'emi_BC_OA': 0,
                             'load':
                                 {'loadbc': [C, load_units, M],
                                  'loaddust': [C, load_units, M],
                                  'loadss': [C, load_units, M],
                                  'loadoa': [C, load_units, M],
                                  'loadso4': [C, load_units, M]},
                             'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                             'load_total': 0, 'load_BC_OA': 0,
                             'od1': {'od550aer': [C, od550_freq]},
                             'od2': {'od440aer': [C, odother_freq]},
                             'od3': {'abs550aer': [C, abs550_freq]},
                             'optical_depth_1': 0,
                             'optical_depth_2': 0,
                             'abs_optical_depth': 0,
                             'precipitation': {'pr': [S, pr_units, M]},
                             'prect': 0,
                             'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                             'color': '#17becf'},

        'TM5-met2010_AP3-CTRL2019': {'emi':
                                         {'emibc': [C, emi_units, M],
                                          'emidust': [C, emi_units, M],
                                          'emiss': [C, emi_units, M],
                                          'emioa': [C, emi_units, M],
                                          'emiso2': [C, emi_units, M]},
                                     'emi_bc': 0, 'emi_dust': 0, 'emi_ss': 0, 'emi_oa': 0, 'emi_so2': 0,
                                     'emi_total': 0, 'emi_BC_OA': 0,
                                     'load':
                                         {'loadbc': [C, load_units, M],
                                          'loaddust': [C, load_units, M],
                                          'loadss': [C, load_units, M],
                                          'loadoa': [C, load_units, M],
                                          'loadso4': [C, load_units, M]},
                                     'load_bc': 0, 'load_dust': 0, 'load_ss': 0, 'load_oa': 0, 'load_so4': 0,
                                     'load_total': 0, 'load_BC_OA': 0,
                                     'od1': {'od550aer': [C, M]},
                                     'od2': {'od440aer': [C, M]},
                                     'od3': {'abs550aer': [C, M]},
                                     'optical_depth_1': 0,
                                     'optical_depth_2': 0,
                                     'abs_optical_depth': 0,
                                     'precipitation': {'pr': [S, pr_units, M]},
                                     'prect': 0,
                                     'masks': {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0},
                                     'color': '#9edae5'}
        }

    # delete the models not requested
    keys_to_delete = []

    for model in x.keys():
        # print(model)
        if model not in models_names:
            keys_to_delete.append(model)
            continue

        if 'emi_total' == var_name:
            for var in x[model]['emi']:
                data = functions.shift360(functions.convert_cftime_to_datetime(
                    xr.open_dataset(path.format(model, model, var, x[model]['emi'][var][0], x[model]['emi'][var][2]))))
                x[model]['emi_total'] += (data[var]*x[model]['emi'][var][1])

        elif 'emi_BC_OA' == var_name:
            x[model]['emi_BC_OA'] = (functions.shift360(functions.convert_cftime_to_datetime(
                                          xr.open_dataset(path.format(model, model, 'emibc', x[model]['emi']['emibc'][0], M)))['emibc']) +
                                      functions.shift360(functions.convert_cftime_to_datetime(
                                          xr.open_dataset(path.format(model, model, 'emioa', x[model]['emi']['emibc'][0], M)))['emioa']))*x[model]['emi']['emibc'][1]

        elif 'emi_ss' == var_name:
            x[model]['emi_ss'] = functions.shift360(functions.convert_cftime_to_datetime(
                xr.open_dataset(path.format(model, model, 'emiss', x[model]['emi']['emiss'][0], M))))['emiss']*x[model]['emi']['emiss'][1]

        elif 'emi_bc' == var_name:
            x[model]['emi_bc'] = functions.shift360(functions.convert_cftime_to_datetime(
                xr.open_dataset(path.format(model, model, 'emibc', x[model]['emi']['emibc'][0], M))))['emibc']*x[model]['emi']['emibc'][1]

        elif 'emi_oa' == var_name:
            x[model]['emi_oa'] = functions.shift360(functions.convert_cftime_to_datetime(
                xr.open_dataset(path.format(model, model, 'emioa', x[model]['emi']['emioa'][0], M))))['emioa']*x[model]['emi']['emioa'][1]

        elif 'load_total' == var_name:
            for var in x[model]['load']:
                data = functions.shift360(functions.convert_cftime_to_datetime(
                    xr.open_dataset(path.format(model, model, var, x[model]['load'][var][0], x[model]['load'][var][2]))))
                x[model]['load_total'] += (data[var]*x[model]['load'][var][1])

        elif 'load_BC_OA' == var_name:
            x[model]['load_BC_OA'] = (functions.shift360(functions.convert_cftime_to_datetime(
                                          xr.open_dataset(path.format(model, model, 'loadbc', x[model]['load']['loadbc'][0], M)))['loadbc']) +
                                      functions.shift360(functions.convert_cftime_to_datetime(
                                          xr.open_dataset(path.format(model, model, 'loadoa', x[model]['load']['loadbc'][0], M)))['loadoa']))*x[model]['load']['loadbc'][1]

        elif 'load_ss' == var_name:
            x[model]['load_ss'] = functions.shift360(functions.convert_cftime_to_datetime(
                xr.open_dataset(path.format(model, model, 'loadss', x[model]['load']['loadss'][0], M))))['loadss']

        elif 'load_bc' == var_name:
            x[model]['load_bc'] = functions.shift360(functions.convert_cftime_to_datetime(
                xr.open_dataset(path.format(model, model, 'loadbc', x[model]['load']['loadbc'][0], M))))['loadbc']/1e-6

        elif 'load_oa' == var_name:
            x[model]['load_oa'] = functions.shift360(functions.convert_cftime_to_datetime(
                xr.open_dataset(path.format(model, model, 'loadoa', x[model]['load']['loadoa'][0], M))))['loadoa']/1e-6
        # TODO: some models have 'loaddust' and others 'loaddu' --> FIX IT

        elif 'optical_depth_1' == var_name:
            for var in x[model]['od1']:
                x[model]['optical_depth_1'] = (functions.shift360(functions.convert_cftime_to_datetime(
                    xr.open_dataset(path.format(model, model, var, x[model]['od1'][var][0], x[model]['od1'][var][1])))[var]))

        elif 'optical_depth_2' == var_name:
            for var in x[model]['od2']:
                x[model]['optical_depth_2'] = (functions.shift360(functions.convert_cftime_to_datetime(
                    xr.open_dataset(path.format(model, model, var, x[model]['od2'][var][0], x[model]['od2'][var][1])))[var]))

        elif 'abs_optical_depth' == var_name:
            for var in x[model]['od3']:
                x[model]['abs_optical_depth'] = functions.shift360(functions.convert_cftime_to_datetime(
                    xr.open_dataset(path.format(model, model, var, x[model]['od3'][var][0], x[model]['od3'][var][1])))[var])

        elif 'prect' == var_name:
            for var in x[model]['precipitation']:
                data = functions.shift360(functions.convert_cftime_to_datetime(
                    xr.open_dataset(path.format(model, model, var, x[model]['precipitation'][var][0], x[model]['precipitation'][var][2]))))
                x[model]['prect'] = (data[var]*x[model]['precipitation'][var][1])

        x[model]['masks']['global'] = xr.ones_like(x[model][var_name].isel(time=0))
        x[model]['masks']['africa'] = functions.create_mask(x[model][var_name].isel(time=0), 15, 37, -15, 0)
        x[model]['masks']['amazon'] = functions.create_mask(x[model][var_name].isel(time=0), 287, 317, -17, -3)
        # x[model]['masks']'north_am'] = functions.create_mask_old(x[model][var_name].isel(time=0), 232, 265, 55,67) # lon:-128,-95
        # x[model]['masks']['siberia'] = functions.create_mask(x[model]['optical_depth_1'].isel(time=0), 156, 176, 62, 71)
        # x[model]['masks']['se_asia'] = functions.create_mask(x[model]['optical_depth_1'].isel(time=0), 93, 109, 11, 22)
        x[model]['masks']['outflow_af'] = functions.create_mask_weigth(x[model][var_name].isel(time=0), 350, 8, -15,3)

    for key in keys_to_delete:
        del x[key]

    return x

def calculate_var(dict1, dict2, var='MEC', lifetime_inv=True):

    """ Calculates MEC (m2/g), MAC (m2/g), lifetime (day), AE or SSA for every model.
        The input units should be:  - emissions: kg/m2s
                                    - load: kg/m2
        If these units change, the conversion factors also need to change!!

    :param dict1: dictionary containing the data.
    :param dict2: dictionary containing the data.
    :param var: name of the variable to calculate. It has to be MEC, lifetime_inv, AE or SSA (string).
    :param lifetime_inv: False --> lifetime, True --> 1/lifetime

    :returns: dictionary containing the desired variable.
    """

    assert var in ['MEC', 'MAC', 'lifetime', 'AE', 'SSA'], f'var must be either MEC, MAC, lifetime, AE or SSA and not {var}'

    # Define the models that use 870nm
    ae_models = {'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010', 'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010'}

    result = {}

    for mask in dict1.keys():
        result[mask] = {}

        for model in dict1[mask].keys():
            if var == 'MEC' or var == 'MAC':
                result[mask][model] = dict1[mask][model] / (dict2[mask][model]*1e3)
            elif var == 'SSA':
                result[mask][model] = 1 - (dict1[mask][model]/dict2[mask][model])
            elif var == 'AE':
                if model in ae_models:
                    result[mask][model] = - np.log(dict1[mask][model] / dict2[mask][model]) / np.log(550/870)
                else:
                    result[mask][model] = - np.log(dict1[mask][model] / dict2[mask][model]) / np.log(550/440)
            elif var == 'lifetime':
                if lifetime_inv is True:
                    result[mask][model] = (dict2[mask][model]*3600*24) / dict1[mask][model]
                elif lifetime_inv is False:
                    result[mask][model] = dict1[mask][model] / (dict2[mask][model]*3600*24)

    return result