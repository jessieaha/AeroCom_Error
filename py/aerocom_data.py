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

Regional masks and area means are handled in post-processing (see functions.create_region_mask
and functions.regional_aggregate in notebooks).

FRdM, 7th of August 2024 """


###################### IMPORT MODULES ######################

import xarray as xr
import numpy as np
import functions

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
               
    :raises ValueError: If path template doesn't contain placeholders, var_name is invalid, or frequency is invalid.
    :raises FileNotFoundError: If no files match the path pattern.
    :raises KeyError: If requested variable is not found in the model configuration.
    """

    import os
    import glob

    # Validate input parameters
    if not isinstance(models_names, (list, tuple)):
        raise TypeError(f"models_names must be a list or tuple, got {type(models_names).__name__}")
     
    if not isinstance(path, str):
        raise TypeError(f"path must be a string, got {type(path).__name__}")
     
    if not isinstance(var_name, str):
        raise TypeError(f"var_name must be a string, got {type(var_name).__name__}")

    # Validate path template contains format placeholders
    try:
        # Try to see if path has format placeholders
        if '{}' not in path:
            raise ValueError(
                f"❌ ERROR: Path template is invalid.\n"
                f"   Expected format string with {{}} placeholders for model, model, variable, location, frequency.\n"
                f"   Received: {path}\n"
                f"   Example: './Data/AEROCOM_III_regrid/{{}}/aerocom3_{{}}_{{}}_{{}}__{{}}.nc'"
            )
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"❌ ERROR: Failed to validate path template: {str(e)}")

    # Validate frequency parameters
    valid_freq = ['monthly', 'daily', '3hourly']
    freq_params = {'od550_freq': od550_freq, 'odother_freq': odother_freq, 'abs550_freq': abs550_freq}

    for freq_name, freq_value in freq_params.items():
        if freq_value not in valid_freq:
            raise ValueError(
                f"❌ ERROR: Invalid frequency for {freq_name}.\n"
                f"   Valid options: {', '.join(valid_freq)}\n"
                f"   Received: {freq_value}"
            )

    def safe_open_dataset(filepath, model, variable_name, location=None):
        """
        Safely open a netCDF dataset with comprehensive error handling.
         
        Args:
            filepath: Full file path to open
            model: Model name (for error messages)
            variable_name: Variable name being loaded (for error messages)
            location: Location/region info (for error messages)
             
        Returns:
            xr.Dataset: The opened dataset
             
        Raises:
            FileNotFoundError: If file doesn't exist
            KeyError: If variable not found in file
            Exception: For other file opening errors
        """
        import os
         
        # Check if parent directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            raise FileNotFoundError(
                f"❌ ERROR: Data directory does not exist.\n"
                f"   Model: {model}\n"
                f"   Variable: {variable_name}\n"
                f"   Directory: {directory}\n"
                f"   Please check the data path configuration."
            )
         
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"❌ ERROR: Data file not found.\n"
                f"   Model: {model}\n"
                f"   Variable: {variable_name}\n"
                f"   Expected file: {filepath}\n"
                f"   Please verify:\n"
                f"     - File path is correct\n"
                f"     - File has been downloaded/created\n"
                f"     - File permissions allow reading"
            )
         
        # Try to open the file
        try:
            dataset = xr.open_dataset(filepath)
            return dataset
        except KeyError as e:
            raise KeyError(
                f"❌ ERROR: Variable not found in dataset.\n"
                f"   Model: {model}\n"
                f"   Variable: {variable_name}\n"
                f"   File: {filepath}\n"
                f"   Available variables: {list(dataset.data_vars) if 'dataset' in locals() else 'Unknown'}\n"
                f"   Original error: {str(e)}"
            )
        except Exception as e:
            raise Exception(
                f"❌ ERROR: Failed to open data file.\n"
                f"   Model: {model}\n"
                f"   Variable: {variable_name}\n"
                f"   File: {filepath}\n"
                f"   Error type: {type(e).__name__}\n"
                f"   Error message: {str(e)}"
            )

    def get_variable_safely(dataset, variable_name, model, filepath):
        """
        Safely extract variable from dataset with error handling.
         
        Args:
            dataset: xr.Dataset to extract from
            variable_name: Name of variable to extract
            model: Model name (for error messages)
            filepath: File path (for error messages)
             
        Returns:
            xr.DataArray: The extracted variable
             
        Raises:
            KeyError: If variable not found in dataset
        """
        try:
            return dataset[variable_name]
        except KeyError:
            available_vars = list(dataset.data_vars.keys())
            raise KeyError(
                f"❌ ERROR: Variable not found in dataset.\n"
                f"   Model: {model}\n"
                f"   Requested variable: {variable_name}\n"
                f"   File: {filepath}\n"
                f"   Available variables ({len(available_vars)}): {', '.join(available_vars[:5])}{'...' if len(available_vars) > 5 else ''}\n"
                f"   Please check variable name spelling or file contents."
            )

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
                                'od550aer': {'od550aer': [C, M]},
                                'od440aer': {'od440aer': [C, M]},
                                'abs550aer': {'abs550aer': [C, M]},
                                'AOD550': 0,
                                'AOD440': 0,
                                'AAOD550': 0,
                                'precipitation': {'pr': [S, pr_units, M]},
                                'prect': 0,
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
                                        'od550aer': {'od550aer': [C, od550_freq]},     # od550csaer
                                        'od440aer': {'od440aer': [C, odother_freq]},     # od440csaer
                                        'abs550aer': {'abs550aer': [C, abs550_freq]},
                                        'AOD550': 0,
                                        'AOD440': 0,
                                        'AAOD550': 0,
                                        'precipitation': {'precip': ['ModelLevel', pr_units, M]},
                                        'prect': 0,
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
                          'od550aer': {'od550aer': [C, od550_freq]},
                          'od440aer': {'od440aer': [C, odother_freq]},
                          'abs550aer': {'abs550aer': [S, abs550_freq]},
                          'AOD550': 0,
                          'AOD440': 0,
                          'AAOD550': 0,
                          'precipitation': {'precip': [S, pr_units, M]},
                          'prect': 0,
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
                                                   'od550aer': {'od550aer': [C, M]},
                                                   'od440aer': {'od440aer': [C, M]},
                                                   'abs550aer': {'abs550aer': [C, M]},
                                                   'AOD550': 0,
                                                   'AOD440': 0,
                                                   'AAOD550': 0,
                                                   'precipitation': {'pr': [S, pr_units, M]},
                                                   'prect': 0,
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
                                         'od550aer': {'od550aer': [C, od550_freq]},
                                         'od440aer': {'od440aer': [C, odother_freq]},
                                         'abs550aer': {'abs550aer': [C, abs550_freq]},
                                         'AOD550': 0,
                                         'AOD440': 0,
                                         'AAOD550': 0,
                                         'precipitation': {'precip': [S, pr_units, M]},
                                         'prect': 0,
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
                                             'od550aer': {'od550aer': [C, M]},     # od550csaer
                                             'od440aer': {'od440aer': [C, M]},
                                             'abs550aer': {'abs550aer': [C, M]},
                                             'AOD550': 0,
                                             'AOD440': 0,
                                             'AAOD550': 0,
                                             'precipitation': {'pr': [S, pr_units, M]},
                                             'prect': 0,
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
                                     'od550aer': {'od550aer': [C, od550_freq]},
                                     'od440aer': {'od440aer': [C, odother_freq]},
                                     'abs550aer': {'abs550aer': [C, abs550_freq]},
                                     'AOD550': 0,
                                     'AOD440': 0,
                                     'AAOD550': 0,
                                     'precipitation': {'precip': [S, pr_units, M]},
                                     'prect': 0,
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
                                               'od550aer': {'od550aer': [C, M]},     # od550csaer
                                               'od440aer': {'od440aer': [C, M]},
                                               'abs550aer': {'abs550aer': [C, M]},
                                               'AOD550': 0,
                                               'AOD440': 0,
                                               'AAOD550': 0,
                                               'precipitation': {'pr': [S, pr_units, M]},
                                               'prect': 0,
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
                                                          'od550aer': {'od550aer': [C, od550_freq]},
                                                          'od440aer': {'od440aer': [C, odother_freq]},
                                                          'abs550aer': {'abs550aer': [C, abs550_freq]},
                                                          'AOD550': 0,
                                                          'AOD440': 0,
                                                          'AAOD550': 0,
                                                          'precipitation': {'precip': [S, pr_units, M]},
                                                          'prect': 0,
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
                                        'od550aer': {'od550aer': [C, od550_freq]},
                                        'od440aer': {'od440aer': [C, odother_freq]},
                                        'abs550aer': {'abs550aer': [C, abs550_freq]},
                                        'AOD550': 0,
                                        'AOD440': 0,
                                        'AAOD550': 0,
                                        'precipitation': {'pr': [S, pr_units, M]},
                                        'prect': 0,
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
                                      'od550aer': {'od550aer': [C, M]},
                                      'od440aer': {'od440aer': [C, M]},
                                      'abs550aer': {'abs550aer': [C, M]},
                                      'AOD550': 0,
                                      'AOD440': 0,
                                      'AAOD550': 0,
                                      'precipitation': {'pr': [S, pr_units, M]},
                                      'prect': 0,
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
                                                  'od550aer': {'od550aer': [C, M]},    # od550csaer
                                                  'od440aer': {'od870csaer': [C, M]},
                                                  'abs550aer': {'abs550aer': [C, M]},
                                                  'AOD550': 0,
                                                  'AOD440': 0,
                                                  'AAOD550': 0,
                                                  'precipitation': {'pr': [C, pr_units, M]},
                                                  'prect': 0,
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
                                               'od550aer': {'od550aer': [C, M]},   # od550csaer
                                               'od440aer': {'od870csaer': [C, M]},
                                               'abs550aer': {'abs550aer': [C, M]},
                                               'AOD550': 0,
                                               'AOD440': 0,
                                               'AAOD550': 0,
                                               'precipitation': {'pr': [C, pr_units, M]},
                                               'prect': 0,
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
                                          'od550aer': {'od550aer': [C, od550_freq]},
                                          'od440aer': {'od870aer': [C, M]},
                                          'abs550aer': {'abs550aer': [C, M]},
                                          'AOD550': 0,
                                          'AOD440': 0,
                                          'AAOD550': 0,
                                          # 'prect': 0,
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
                          'od550aer': {'od550aer': [C, M]},
                          'od440aer': {'od440aer': [C, M]},
                          'abs550aer': {'abs550aer': [C, M]},
                          'AOD550': 0,
                          'AOD440': 0,
                          'AAOD550': 0,
                          'precipitation': {'pr': [S, pr_units, M]},
                          'prect': 0,
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
                                     'od550aer': {'od550aer': [C, M]},     # od550csaer
                                     'od440aer': {'od440csaer': [C, M]},
                                     'abs550aer': {'abs550aer': [C, M]},
                                     'AOD550': 0,
                                     'AOD440': 0,
                                     'AAOD550': 0,
                                     'precipitation': {'precip': ['ModelLevel', pr_units, M]},
                                     'prect': 0,
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
                                     'od550aer': {'od550aer': [C, od550_freq]},
                                     'od440aer': {'od440aer': [C, odother_freq]},
                                     'abs550aer': {'abs550aer': [C, abs550_freq]},
                                     'AOD550': 0,
                                     'AOD440': 0,
                                     'AAOD550': 0,
                                     'precipitation': {'pr': [S, pr_units, M]},
                                     'prect': 0,
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
                                           'od550aer': {'od550aer': [C, od550_freq]},
                                           'od440aer': {'od440aer': [C, odother_freq]},
                                           'abs550aer': {'abs550aer': [C, abs550_freq]},
                                           'AOD550': 0,
                                           'AOD440': 0,
                                           'AAOD550': 0,
                                           'precipitation': {'pr': [S, pr_units, M]},
                                           'prect': 0,
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
                             'od550aer': {'od550aer': [C, od550_freq]},
                             'od440aer': {'od440aer': [C, odother_freq]},
                             'abs550aer': {'abs550aer': [C, abs550_freq]},
                             'AOD550': 0,
                             'AOD440': 0,
                             'AAOD550': 0,
                             'precipitation': {'pr': [S, pr_units, M]},
                             'prect': 0,
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
                                     'od550aer': {'od550aer': [C, M]},
                                     'od440aer': {'od440aer': [C, M]},
                                     'abs550aer': {'abs550aer': [C, M]},
                                     'AOD550': 0,
                                     'AOD440': 0,
                                     'AAOD550': 0,
                                     'precipitation': {'pr': [S, pr_units, M]},
                                     'prect': 0,
                                     'color': '#9edae5'}
        }

    # delete the models not requested
    keys_to_delete = []
    failed_models = []

    for model in x.keys():
        # print(model)
        if model not in models_names:
            keys_to_delete.append(model)
            continue

        try:
            if 'emi_total' == var_name:
                for var in x[model]['emi']:
                    try:
                        file_path = path.format(model, model, var, x[model]['emi'][var][0], x[model]['emi'][var][2])
                        dataset = safe_open_dataset(file_path, model, var)
                        var_data = get_variable_safely(dataset, var, model, file_path)
                        data = functions.shift360(functions.convert_cftime_to_datetime(dataset))
                        x[model]['emi_total'] += (data[var]*x[model]['emi'][var][1])
                    except (FileNotFoundError, KeyError) as e:
                        print(f"⚠️  WARNING: {str(e)}")
                        failed_models.append((model, var, str(e)))
                        raise

            elif 'emi_BC_OA' == var_name:
                try:
                    # Load emibc
                    file_path_bc = path.format(model, model, 'emibc', x[model]['emi']['emibc'][0], M)
                    dataset_bc = safe_open_dataset(file_path_bc, model, 'emibc')
                    data_bc = functions.shift360(functions.convert_cftime_to_datetime(dataset_bc))
                    emibc_data = get_variable_safely(data_bc, 'emibc', model, file_path_bc)
                     
                    # Load emioa
                    file_path_oa = path.format(model, model, 'emioa', x[model]['emi']['emibc'][0], M)
                    dataset_oa = safe_open_dataset(file_path_oa, model, 'emioa')
                    data_oa = functions.shift360(functions.convert_cftime_to_datetime(dataset_oa))
                    emioa_data = get_variable_safely(data_oa, 'emioa', model, file_path_oa)
                     
                    x[model]['emi_BC_OA'] = (emibc_data + emioa_data) * x[model]['emi']['emibc'][1]
                except (FileNotFoundError, KeyError) as e:
                    print(f"⚠️  WARNING: {str(e)}")
                    failed_models.append((model, 'emi_BC_OA', str(e)))
                    raise

            elif 'emi_ss' == var_name:
                try:
                    file_path = path.format(model, model, 'emiss', x[model]['emi']['emiss'][0], M)
                    dataset = safe_open_dataset(file_path, model, 'emiss')
                    data = functions.shift360(functions.convert_cftime_to_datetime(dataset))
                    x[model]['emi_ss'] = get_variable_safely(data, 'emiss', model, file_path) * x[model]['emi']['emiss'][1]
                except (FileNotFoundError, KeyError) as e:
                    print(f"⚠️  WARNING: {str(e)}")
                    failed_models.append((model, 'emiss', str(e)))
                    raise

            elif 'emi_bc' == var_name:
                try:
                    file_path = path.format(model, model, 'emibc', x[model]['emi']['emibc'][0], M)
                    dataset = safe_open_dataset(file_path, model, 'emibc')
                    data = functions.shift360(functions.convert_cftime_to_datetime(dataset))
                    x[model]['emi_bc'] = get_variable_safely(data, 'emibc', model, file_path) * x[model]['emi']['emibc'][1]
                except (FileNotFoundError, KeyError) as e:
                    print(f"⚠️  WARNING: {str(e)}")
                    failed_models.append((model, 'emibc', str(e)))
                    raise

            elif 'emi_oa' == var_name:
                try:
                    file_path = path.format(model, model, 'emioa', x[model]['emi']['emioa'][0], M)
                    dataset = safe_open_dataset(file_path, model, 'emioa')
                    data = functions.shift360(functions.convert_cftime_to_datetime(dataset))
                    x[model]['emi_oa'] = get_variable_safely(data, 'emioa', model, file_path) * x[model]['emi']['emioa'][1]
                except (FileNotFoundError, KeyError) as e:
                    print(f"⚠️  WARNING: {str(e)}")
                    failed_models.append((model, 'emioa', str(e)))
                    raise

            elif 'load_total' == var_name:
                for var in x[model]['load']:
                    try:
                        file_path = path.format(model, model, var, x[model]['load'][var][0], x[model]['load'][var][2])
                        dataset = safe_open_dataset(file_path, model, var)
                        data = functions.shift360(functions.convert_cftime_to_datetime(dataset))
                        var_data = get_variable_safely(data, var, model, file_path)
                        x[model]['load_total'] += (data[var]*x[model]['load'][var][1])
                    except (FileNotFoundError, KeyError) as e:
                        print(f"⚠️  WARNING: {str(e)}")
                        failed_models.append((model, var, str(e)))
                        raise

            elif 'load_BC_OA' == var_name:
                try:
                    # Load loadbc
                    file_path_bc = path.format(model, model, 'loadbc', x[model]['load']['loadbc'][0], M)
                    dataset_bc = safe_open_dataset(file_path_bc, model, 'loadbc')
                    data_bc = functions.shift360(functions.convert_cftime_to_datetime(dataset_bc))
                    loadbc_data = get_variable_safely(data_bc, 'loadbc', model, file_path_bc)
                     
                    # Load loadoa
                    file_path_oa = path.format(model, model, 'loadoa', x[model]['load']['loadbc'][0], M)
                    dataset_oa = safe_open_dataset(file_path_oa, model, 'loadoa')
                    data_oa = functions.shift360(functions.convert_cftime_to_datetime(dataset_oa))
                    loadoa_data = get_variable_safely(data_oa, 'loadoa', model, file_path_oa)
                     
                    x[model]['load_BC_OA'] = (loadbc_data + loadoa_data) * x[model]['load']['loadbc'][1]
                except (FileNotFoundError, KeyError) as e:
                    print(f"⚠️  WARNING: {str(e)}")
                    failed_models.append((model, 'load_BC_OA', str(e)))
                    raise

            elif 'load_ss' == var_name:
                try:
                    file_path = path.format(model, model, 'loadss', x[model]['load']['loadss'][0], M)
                    dataset = safe_open_dataset(file_path, model, 'loadss')
                    data = functions.shift360(functions.convert_cftime_to_datetime(dataset))
                    x[model]['load_ss'] = get_variable_safely(data, 'loadss', model, file_path)
                except (FileNotFoundError, KeyError) as e:
                    print(f"⚠️  WARNING: {str(e)}")
                    failed_models.append((model, 'loadss', str(e)))
                    raise

            elif 'load_bc' == var_name:
                try:
                    file_path = path.format(model, model, 'loadbc', x[model]['load']['loadbc'][0], M)
                    dataset = safe_open_dataset(file_path, model, 'loadbc')
                    data = functions.shift360(functions.convert_cftime_to_datetime(dataset))
                    x[model]['load_bc'] = get_variable_safely(data, 'loadbc', model, file_path) / 1e-6
                except (FileNotFoundError, KeyError) as e:
                    print(f"⚠️  WARNING: {str(e)}")
                    failed_models.append((model, 'loadbc', str(e)))
                    raise

            elif 'load_oa' == var_name:
                try:
                    file_path = path.format(model, model, 'loadoa', x[model]['load']['loadoa'][0], M)
                    dataset = safe_open_dataset(file_path, model, 'loadoa')
                    data = functions.shift360(functions.convert_cftime_to_datetime(dataset))
                    x[model]['load_oa'] = get_variable_safely(data, 'loadoa', model, file_path) / 1e-6
                except (FileNotFoundError, KeyError) as e:
                    print(f"⚠️  WARNING: {str(e)}")
                    failed_models.append((model, 'loadoa', str(e)))
                    raise
            # TODO: some models have 'loaddust' and others 'loaddu' --> FIX IT

            elif 'AOD550' == var_name:
                for var in x[model]['od550aer']:
                    try:
                        file_path = path.format(model, model, var, x[model]['od550aer'][var][0], x[model]['od550aer'][var][1])
                        dataset = safe_open_dataset(file_path, model, var)
                        data = functions.shift360(functions.convert_cftime_to_datetime(dataset))
                        x[model]['AOD550'] = get_variable_safely(data, var, model, file_path)
                    except (FileNotFoundError, KeyError) as e:
                        print(f"⚠️  WARNING: {str(e)}")
                        failed_models.append((model, var, str(e)))
                        raise

            elif 'AOD440' == var_name:
                for var in x[model]['od440aer']:
                    try:
                        file_path = path.format(model, model, var, x[model]['od440aer'][var][0], x[model]['od440aer'][var][1])
                        dataset = safe_open_dataset(file_path, model, var)
                        data = functions.shift360(functions.convert_cftime_to_datetime(dataset))
                        x[model]['AOD440'] = get_variable_safely(data, var, model, file_path)
                    except (FileNotFoundError, KeyError) as e:
                        print(f"⚠️  WARNING: {str(e)}")
                        failed_models.append((model, var, str(e)))
                        raise

            elif 'AAOD550' == var_name:
                for var in x[model]['abs550aer']:
                    try:
                        file_path = path.format(model, model, var, x[model]['abs550aer'][var][0], x[model]['abs550aer'][var][1])
                        dataset = safe_open_dataset(file_path, model, var)
                        data = functions.shift360(functions.convert_cftime_to_datetime(dataset))
                        x[model]['AAOD550'] = get_variable_safely(data, var, model, file_path)
                    except (FileNotFoundError, KeyError) as e:
                        print(f"⚠️  WARNING: {str(e)}")
                        failed_models.append((model, var, str(e)))
                        raise

            elif 'prect' == var_name:
                for var in x[model]['precipitation']:
                    try:
                        file_path = path.format(model, model, var, x[model]['precipitation'][var][0], x[model]['precipitation'][var][2])
                        dataset = safe_open_dataset(file_path, model, var)
                        data = functions.shift360(functions.convert_cftime_to_datetime(dataset))
                        x[model]['prect'] = get_variable_safely(data, var, model, file_path) * x[model]['precipitation'][var][1]
                    except (FileNotFoundError, KeyError) as e:
                        print(f"⚠️  WARNING: {str(e)}")
                        failed_models.append((model, var, str(e)))
                        raise

            else:
                raise ValueError(
                    f"ERROR: Unknown variable requested.\n"
                    f"   Variable: {var_name}\n"
                    f"   Model: {model}\n"
                    f"   Valid variables: emi_total, emi_BC_OA, emi_ss, emi_bc, emi_oa,\n"
                    f"                    load_total, load_BC_OA, load_ss, load_bc, load_oa,\n"
                    f"                    AOD550, AOD440, AAOD550, prect"
                )

            # Regional masks are created in notebook post-processing (functions.create_region_mask).
        except Exception as e:
            error_msg = f"ERROR: Failed to load data for model '{model}' and variable '{var_name}'.\n{str(e)}"
            print(error_msg)
            keys_to_delete.append(model)
            continue
        # x[model]['masks']'north_am'] = functions.create_mask_old(x[model][var_name].isel(time=0), 232, 265, 55,67) # lon:-128,-95
        # x[model]['masks']['siberia'] = functions.create_mask(x[model]['AOD550'].isel(time=0), 156, 176, 62, 71)
        # x[model]['masks']['se_asia'] = functions.create_mask(x[model]['AOD550'].isel(time=0), 93, 109, 11, 22)
        # x[model]['masks']['outflow_af'] = functions.create_mask_weigth(x[model][var_name].isel(time=0), 350, 8, -15,3)

    # Remove failed models
    for key in keys_to_delete:
        del x[key]

    # Print summary
    successful_models = [m for m in models_names if m in x.keys()]
    if len(models_names) > len(successful_models):
        print(f"\n📊 Data Loading Summary:")
        print(f"   ✓ Successful: {len(successful_models)} / {len(models_names)} models")
        print(f"   ✗ Failed: {len(models_names) - len(successful_models)} models")
        if failed_models:
            print(f"\n   Failed models:")
            for model, var, error in failed_models[:5]:
                print(f"     - {model}: {var}")
            if len(failed_models) > 5:
                print(f"     ... and {len(failed_models) - 5} more")
    else:
        print(f"\n✓ Successfully loaded data for {len(successful_models)} models")

    if not x:
        raise ValueError(
            f"❌ ERROR: No models were successfully loaded.\n"
            f"   Requested variable: {var_name}\n"
            f"   Requested models: {', '.join(models_names[:3])}{'...' if len(models_names) > 3 else ''}\n"
            f"   Path template: {path}\n"
            f"   Please check:\n"
            f"     1. Data files exist in the specified path\n"
            f"     2. Path template is correct\n"
            f"     3. Model names are spelled correctly\n"
            f"     4. Variable name is valid"
        )

    # Normalise the time coordinate of every loaded DataArray to first-of-month.
    # Models differ in their time stamping convention (mid-month, end-of-month,
    # or next-month-start), which would otherwise misalign variables when
    # computing derived quantities such as MEC = AOD / load or SSA = 1 - abs/AOD.
    for model in x:
        if isinstance(x[model].get(var_name), xr.DataArray):
            x[model][var_name] = functions.normalize_monthly_time(x[model][var_name])

    return x

def calculate_var(dict1, dict2, var='MEC', lifetime_inv=False):

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

    # Context manager to handle warnings globally within the loop
    with np.errstate(divide='ignore', invalid='ignore'):
        for mask in dict1.keys():
            result[mask] = {}
            for model in dict1[mask].keys():
                val1 = dict1[mask][model]
                val2 = dict2[mask][model]
                
                if var in ['MEC', 'MAC']:
                    result[mask][model] = val1 / (val2 * 1e3)
                
                elif var == 'SSA':
                    result[mask][model] = 1 - (val1 / val2)
                
                elif var == 'AE':
                    divisor = np.log(550/870) if model in ae_models else np.log(550/440)
                    result[mask][model] = - np.log(val1 / val2) / divisor
                
                elif var == 'lifetime':
                    if lifetime_inv:
                        result[mask][model] = (val2 * 3600 * 24) / val1
                    else:
                        result[mask][model] = val1 / (val2 * 3600 * 24)

    return result