"""
Modernized, automated AeroCom Phase III data extraction engine.
This module dynamically scans model directories, discovers files matching variable patterns,
standardizes coordinates to ('lon', 'lat', 'time'), and returns clean, analysis-ready
lazy xarray Datasets. Missing variables are gracefully marked as None.


"""

import os
import glob
import logging
import xarray as xr
import numpy as np

# Configure standard logging to report missing files cleanly
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ==============================================================================
# GLOBAL CONFIGURATIONS
# ==============================================================================
# VARIABLES = (
#     "abs550aer", "depbc", "depdust", "depoa", "depso2", "depso4", "depss",
#     "emibc", "emidust", "emioa", "emiso2", "emiss", "loadbc", "loaddust",
#     "loadoa", "loadso2", "loadso4", "loadss", "od440aer", "od550aer",
#     "od870aer", "od865aer", "od550bc", "od550dust", "od550oa", "od550so4", "od550ss"
# )

# Define regional models that use 870nm instead of 440nm for Angstrom Exponent
AE_870_MODELS = {
    'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010', 
    'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010'
}
# 440
# ECHAM6-SALSA_CTRL2016-PD/aerocom3_ECHAM6-SALSA_CTRL2016-PD_od440aer_Column_2010_3hourly.nc
# ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD/aerocom3_ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD_od440aer_Column_2010_3hourly.nc
# ECMWF-IFS-CY45R1-CAMS-CTRL-met2010_AP3-CTRL/aerocom3_ECMWF-IFS-CY45R1-CAMS-CTRL-met2010_AP3-CTRL_od440aer_Column_2010_3hourly.nc
# ECMWF-IFS-CY46R1-CAMS-CTRL-met2010_AP3-CTRL/aerocom3_ECMWF-IFS-CY46R1-CAMS-CTRL-met2010_AP3-CTRL_od440aer_Column_2010_3hourly.nc
# GEOS-Chem-v11-01_AP3-CTRL2016-PD/aerocom3_GEOS-Chem-v11-01_AP3-CTRL2016-PD_od440aer_Column_2010_3hourly.nc
# HadGEM3-GA7.1_AP3-CTRL2016-PD/aerocom3_HadGEM3-GA7.1_AP3-CTRL2016-PD_od440aer_Column_2010_3hourly.nc
# IMPACT_CTRL2016/aerocom3_IMPACT_CTRL2016_od440aer_Column_2010_3hourly.nc
# MIROC-SPRINTARS_AP3-CTRL/aerocom3_MIROC-SPRINTARS_AP3-CTRL_od440aer_Column_2010_3hourly.nc
# SPRINTARS-T213_AP3-CTRL2016-PD/aerocom3_SPRINTARS-T213_AP3-CTRL2016-PD_od440aer_Column_2010_3hourly.nc
# TM5_AP3-CTRL2016/aerocom3_TM5_AP3-CTRL2016_od440aer_Column_2010_3hourly.nc
# 870 
# GEOS-Chem-v11-01_AP3-CTRL2016-PD/aerocom3_GEOS-Chem-v11-01_AP3-CTRL2016-PD_od870aer_Column_2010_3hourly.nc
# HadGEM3-GA7.1_AP3-CTRL2016-PD/aerocom3_HadGEM3-GA7.1_AP3-CTRL2016-PD_od870aer_Column_2010_3hourly.nc
# MIROC-SPRINTARS_AP3-CTRL/aerocom3_MIROC-SPRINTARS_AP3-CTRL_od870aer_Column_2010_3hourly.nc
# SPRINTARS-T213_AP3-CTRL2016-PD/aerocom3_SPRINTARS-T213_AP3-CTRL2016-PD_od870aer_Column_2010_3hourly.nc
# TM5_AP3-CTRL2016/aerocom3_TM5_AP3-CTRL2016_od870aer_Column_2010_3hourly.nc

#3 hourly models Data/AP3_2026 :
# AP3_2026/ECHAM6-SALSA_CTRL2016-PD/aerocom3_ECHAM6-SALSA_CTRL2016-PD_od550aer3d_ModelLevel_2010_3hourly.nc
# AP3_2026/ECHAM6-SALSA_CTRL2016-PD/aerocom3_ECHAM6-SALSA_CTRL2016-PD_od550aer_Column_2010_3hourly.nc
# AP3_2026/ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD/aerocom3_ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD_od550aer_Column_2010_3hourly.nc
# AP3_2026/ECMWF-IFS-CY45R1-CAMS-CTRL-met2010_AP3-CTRL/aerocom3_ECMWF-IFS-CY45R1-CAMS-CTRL-met2010_AP3-CTRL_od550aer_Column_2010_3hourly.nc
# AP3_2026/ECMWF-IFS-CY46R1-CAMS-CTRL-met2010_AP3-CTRL/aerocom3_ECMWF-IFS-CY46R1-CAMS-CTRL-met2010_AP3-CTRL_od550aer_Column_2010_3hourly.nc
# AP3_2026/GEOS-Chem-v11-01_AP3-CTRL2016-PD/aerocom3_GEOS-Chem-v11-01_AP3-CTRL2016-PD_od550aer_Column_2010_3hourly.nc
# AP3_2026/HadGEM3-GA7.1_AP3-CTRL2016-PD/aerocom3_HadGEM3-GA7.1_AP3-CTRL2016-PD_od550aer_Column_2010_3hourly.nc
# AP3_2026/IMPACT_CTRL2016/aerocom3_IMPACT_CTRL2016_od550aer3d_ModelLevel_2010_3hourly.nc
# AP3_2026/IMPACT_CTRL2016/aerocom3_IMPACT_CTRL2016_od550aer_Column_2010_3hourly.nc
# AP3_2026/MIROC-SPRINTARS_AP3-CTRL/aerocom3_MIROC-SPRINTARS_AP3-CTRL_od550aer_Column_2010_3hourly.nc
# AP3_2026/SPRINTARS-T213_AP3-CTRL2016-PD/aerocom3_SPRINTARS-T213_AP3-CTRL2016-PD_od550aer_Column_2010_3hourly.nc
# AP3_2026/TM5_AP3-CTRL2016/aerocom3_TM5_AP3-CTRL2016_od550aer3d_ModelLevel_2010_3hourly.nc
# AP3_2026/TM5_AP3-CTRL2016/aerocom3_TM5_AP3-CTRL2016_od550aer_Column_2010_3hourly.nc

# 3 hourly model Data/AEROCOME_III :
# AEROCOM_III/CAM5.3-Oslo_AP3-CTRL2016-PD/aerocom3_CAM5.3-Oslo_AP3-CTRL2016-PD_od550aer_Column_2010_3hourly.nc
# AEROCOM_III/CAM5_CTRL2016/aerocom3_CAM5_CTRL2016_od550aer_Column_2010_3hourly.nc
# AEROCOM_III/ECHAM6-HAM2_AP3-CTRL2016-PD/aerocom3_ECHAM6-HAM2_AP3-CTRL2016-PD_od550aer_Column_2010_3hourly.nc
# AEROCOM_III/ECHAM6-SALSA_CTRL2016-PD/aerocom3_ECHAM6-SALSA_CTRL2016-PD_od550aer_Column_2010_3hourly.nc
# AEROCOM_III/ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD/aerocom3_ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD_od550aer_Column_2010_3hourly.nc
# AEROCOM_III/GEOS-i33p2-met2010_AP3-CTRL/aerocom3_GEOS-i33p2-met2010_AP3-CTRL_od550aer_Column_2010_3hourly.nc
# AEROCOM_III/HadGEM3-GA7.1_AP3-CTRL2016-PD/aerocom3_HadGEM3-GA7.1_AP3-CTRL2016-PD_od550aer_Column_2010_3hourly.nc
# AEROCOM_III/MIROC-SPRINTARS_AP3-CTRL/aerocom3_MIROC-SPRINTARS_AP3-CTRL_od550aer_Column_2010_3hourly.nc
# AEROCOM_III/TM5_AP3-CTRL2016/aerocom3_TM5_AP3-CTRL2016_od550aer_Column_2010_3hourly.nc
# ==============================================================================
# LONGITUDE NORMALIZATION
# ==============================================================================
def normalize_longitude(ds, lon_name='lon'):
    """
    Normalize longitude coordinates to a consistent 0-360 range and sort the
    dataset along the longitude axis so that coordinates are monotonically
    increasing. This is required for correct interpolation against the POLDER
    0-360 grid and avoids gaps at the -180/180 anti-meridian.
    """
    if lon_name not in ds.coords:
        return ds

    lon = ds[lon_name].values
    # Already in 0-360 and sorted -> nothing to do
    if np.all(lon >= 0) and np.all(lon <= 360) and np.all(np.diff(lon) >= 0):
        return ds

    # Wrap any -180..180 values into 0..360
    lon360 = np.mod(lon + 360.0, 360.0)

    ds = ds.assign_coords({lon_name: lon360})
    ds = ds.sortby(lon_name)

    # Drop any duplicate longitudes that can appear after wrapping (e.g. 0 and 360)
    lon_sorted = ds[lon_name].values
    _, unique_idx = np.unique(lon_sorted, return_index=True)
    if len(unique_idx) < len(lon_sorted):
        ds = ds.isel({lon_name: unique_idx})

    return ds


# ==============================================================================
# COORDINATE STANDARDIZATION ENGINE
# ==============================================================================
def standardize_dataset(ds, var_name):
    """
    Standardizes coordinate and dimension names to 'lon', 'lat', and 'time', 
    and slices the dataset to contain only the target variable. Squeezes out 
    unneeded singleton dimensions (e.g. height, levels).
    """
    rename_dict = {}
    
    # 1. Identify and standardize Longitude
    lon_coords = [c for c in ds.coords if c.lower() in ['lon', 'longitude', 'lons']]
    if lon_coords and lon_coords[0] != 'lon':
        rename_dict[lon_coords[0]] = 'lon'
        
    # 2. Identify and standardize Latitude
    lat_coords = [c for c in ds.coords if c.lower() in ['lat', 'latitude', 'lats']]
    if lat_coords and lat_coords[0] != 'lat':
        rename_dict[lat_coords[0]] = 'lat'
        
    # 3. Identify and standardize Time
    time_coords = [c for c in ds.coords if c.lower() in ['time', 't', 'times']]
    if time_coords and time_coords[0] != 'time':
        rename_dict[time_coords[0]] = 'time'
        
    # Rename matching coordinates
    if rename_dict:
        ds = ds.rename(rename_dict)
        
    # Squeeze out singleton dimensions (like level or height) to simplify analytical shapes
    ds = ds.squeeze()

    # Normalize longitude to a consistent 0-360 grid for POLDER interpolation
    ds = normalize_longitude(ds)

    # Keep only standardized dimensions & the variable itself
    dims_to_keep = [d for d in ['lon', 'lat', 'time'] if d in ds.coords]
    
    if var_name in ds.variables:
        # Filter down dataset variables to save system memory
        ds = ds[dims_to_keep + [var_name]]
    else:
        # Sometimes variable name inside NetCDF is slightly different than file pattern
        # Handle known aliases (e.g., precipitation files may use 'pr' internally)
        aliases = {
            'precip': ['pr', 'precipitation', 'PRECT', 'prect'],
            'pr': ['precip', 'precipitation', 'PRECT', 'prect'],
        }

        data_vars = list(ds.data_vars)
        target_var = None
        for alt in aliases.get(var_name, []):
            if alt in data_vars:
                target_var = alt
                break

        if target_var is not None:
            ds = ds.rename({target_var: var_name})
            ds = ds[dims_to_keep + [var_name]]
        elif len(data_vars) == 1:
            ds = ds.rename({data_vars[0]: var_name})
            ds = ds[dims_to_keep + [var_name]]
        else:
            raise KeyError(f"Could not reliably isolate data variable '{var_name}' in dataset variables: {data_vars}")

    return ds


# ==============================================================================
# BULK DYNAMIC DATA LOADER (GLOB PATTERNS)
# ==============================================================================
def load_all_model_data(base_dir, models, variables, temporal, standardize=True):
    """
    Automatically search files, open them lazily via Xarray, and return 
    a structured dictionary map: data[model][variable] = xr.Dataset (or None)
    
    :param base_dir: Path to directory containing model subfolders.
    :param models: List of model directory names to scan.
    :param variables: Iterable containing variables to look up.
    :param temporal: Frequency filter (e.g. '3hourly', 'daily', 'monthly').
    :param standardize: Rename variations of coord names to 'lon', 'lat', 'time'.
    :return: dict structured as {model: {variable: xr.Dataset_or_None}}
    """
    data_dict = {}
    
    for model in models:
        data_dict[model] = {}
        model_dir = os.path.join(base_dir, model)
        
        logging.info(f"Scanning directory for model: {model} ...")
        
        for var in variables:
            # Allow precipitation files to be named either 'precip' or 'pr'
            var_names = [var]
            if var == 'precip':
                var_names.append('pr')
            elif var == 'pr':
                var_names.append('precip')

            patterns = []
            for vn in var_names:
                patterns.extend([
                    os.path.join(model_dir, f"*{vn}_*Column*2010*{temporal}*.nc"),       # Column variables (e.g., loads, optical)
                    os.path.join(model_dir, f"*{vn}_*Surface*2010*{temporal}*.nc"),      # Surface variables (e.g., emissions, deposition)
                    os.path.join(model_dir, f"*{vn}_*ModelLevel*2010*{temporal}*.nc"),  # Model-level variables (e.g., some precipitation)
                    os.path.join(model_dir, f"*{vn}_*2010*{temporal}*.nc"),             # Generic fallback
                    os.path.join(base_dir, f"*{model}*{vn}_*Column*2010*{temporal}*.nc"),  # Flat directory fallback
                    os.path.join(base_dir, f"*{model}*{vn}_*Surface*2010*{temporal}*.nc"), # Flat directory fallback (Surface)
                    os.path.join(base_dir, f"*{model}*{vn}_*ModelLevel*2010*{temporal}*.nc"), # Flat directory fallback (ModelLevel)
                    os.path.join(base_dir, f"*{model}*{vn}_*2010*{temporal}*.nc"),         # Flat directory generic fallback
                ])

            filepath = None
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    filepath = matches[0]  # Pick first match
                    break

            if filepath and os.path.exists(filepath):
                try:
                    # Open dataset lazily without loading actual array bytes into RAM
                    ds = xr.open_dataset(filepath, chunks='auto')

                    if standardize:
                        ds = standardize_dataset(ds, var)

                    data_dict[model][var] = ds
                    logging.debug(f"  ✓ Found & opened: {var} -> {os.path.basename(filepath)}")
                except Exception as e:
                    logging.warning(f"  ❌ Error reading '{filepath}' for variable '{var}': {e}")
                    data_dict[model][var] = None
            else:
                # Variable does not exist on disk, mark as NA
                data_dict[model][var] = None
                
    return data_dict


# ==============================================================================
# VECTORIZED DERIVED VARIABLE CALCULATOR
# ==============================================================================
def _get_dataarray(value, var_name):
    """Extract a DataArray from a Dataset, or pass a DataArray through."""
    if value is None:
        return None
    if isinstance(value, xr.DataArray):
        return value
    if isinstance(value, xr.Dataset):
        if var_name in value.data_vars:
            return value[var_name]
        data_vars = list(value.data_vars)
        if len(data_vars) == 1:
            return value[data_vars[0]]
        raise KeyError(
            f"Could not extract '{var_name}' from dataset with variables {data_vars}"
        )
    raise TypeError(f"Unsupported type for {var_name}: {type(value)}")


def calculate_derived_var(model_data, model_name, derived_var):
    """
    Calculates complex aerosol diagnostics (MEC, MAC, SSA, AE) directly on the 
    lazily loaded xarray Datasets using highly performant, vectorized array math.
    
    :param model_data: Dictionary for a specific model (i.e. data_dict[model_name])
    :param model_name: Name of the model currently processed (used for AE sensor wavelengths)
    :param derived_var: Metric to calculate ('MEC', 'MAC', 'SSA', 'AE')
    :return: Standardized, derived xr.Dataset or None if required dependencies are missing.
    """
    try:
        if derived_var == 'MEC':
            # MEC = AOD_550 / (total_load * 1e3)
            od = _get_dataarray(model_data.get('od550aer'), 'od550aer')
            # Sum up loaded variables dynamically (ignoring missing species cleanly if needed)
            loads = [
                _get_dataarray(model_data.get(k), k)
                for k in ['loadbc', 'loaddust', 'loadoa', 'loadso4', 'loadss']
                if model_data.get(k) is not None
            ]
            if od is None or not loads:
                raise ValueError("Missing 'od550aer' or loading fields.")
            total_load = sum(loads)
            
            mec = od / (total_load * 1e3)
            return mec.to_dataset(name='MEC')
            
        elif derived_var == 'MAC':
            # MAC = AAOD_550 / (load_BC_OA * 1e3)
            abs550 = _get_dataarray(model_data.get('abs550aer'), 'abs550aer')
            bc = _get_dataarray(model_data.get('loadbc'), 'loadbc')
            oa = _get_dataarray(model_data.get('loadoa'), 'loadoa')
            if abs550 is None or bc is None or oa is None:
                raise ValueError("Missing absorption 'abs550aer' or loads (loadbc/loadoa).")
                
            mac = abs550 / ((bc + oa) * 1e3)
            return mac.to_dataset(name='MAC')
            
        elif derived_var == 'SSA':
            # SSA = 1 - (AAOD_550 / AOD_550)
            abs550 = _get_dataarray(model_data.get('abs550aer'), 'abs550aer')
            od550 = _get_dataarray(model_data.get('od550aer'), 'od550aer')
            if abs550 is None or od550 is None:
                raise ValueError("Missing 'abs550aer' or 'od550aer'.")
                
            ssa = 1.0 - (abs550 / od550)
            return ssa.to_dataset(name='SSA')
            
        elif derived_var == 'AE':
            # AE = - log(AOD_550 / AOD_other) / log(550 / other)
            od550 = _get_dataarray(model_data.get('od550aer'), 'od550aer')
            
            if model_name in AE_870_MODELS:
                od870 = _get_dataarray(model_data.get('od870aer'), 'od870aer')
                od865 = _get_dataarray(model_data.get('od865aer'), 'od865aer')
                if od870 is not None:
                    od_other = od870
                    other_wavelength = 870
                elif od865 is not None:
                    od_other = od865
                    other_wavelength = 865
                else:
                    od_other = None
            else:
                od_other = _get_dataarray(model_data.get('od440aer'), 'od440aer')
                other_wavelength = 440
                
            if od550 is None or od_other is None:
                raise ValueError(f"Missing spectral bands to compute Angstrom Exponent for {model_name}.")
            
            var_other = (
                'od870aer' if other_wavelength == 870
                else ('od865aer' if other_wavelength == 865 else 'od440aer')
            )
            divisor = np.log(550.0 / other_wavelength)
            
            ae = - np.log(od550 / od_other) / divisor
            return ae.to_dataset(name='AE')
            
    except Exception as e:
        logging.warning(f"Could not calculate {derived_var} for {model_name}: {e}")
        return None

def save_model_data_to_netcdf(data_dict, output_base_dir="./Data/AEROCOM_Processed/"):
    """
    Saves the extracted xarray Datasets into a structured directory of NetCDF files,
    organized into folders by variable name.
    
    :param data_dict: The master dictionary {model: {variable: xr.Dataset}}
    :param output_base_dir: The root folder where processed data will be saved
    """
    print(f"\n💾 Saving processed data to {output_base_dir} ...")
    
    saved_count = 0
    missing_count = 0

    for model, variables in data_dict.items():
        for var_name, ds in variables.items():
            if ds is not None:
                # Create a specific output folder for this variable
                var_out_dir = os.path.join(output_base_dir, var_name)
                os.makedirs(var_out_dir, exist_ok=True)
                
                # Define the output file name (e.g., ./Data/AEROCOM_Processed/od550aer/GEOS-Chem_od550aer_processed.nc)
                out_file = os.path.join(var_out_dir, f"{model}_{var_name}_processed.nc")
                
                try:
                    # Save the dataset to NetCDF using standard compression and HPC compatibility
                    ds.to_netcdf(out_file, engine='netcdf4')
                    saved_count += 1
                except Exception as e:
                    print(f"  ❌ Error saving {out_file}: {e}")
            else:
                missing_count += 1

    print("=============================================")
    print(f"Save Complete: {saved_count} files saved successfully.")
    print(f"Skipped {missing_count} missing variables.")
    print("=============================================\n")


def load_monthly_data_from_netcdf(
    output_base_dir="./Data/AP3_processed_monthly",
    variables=None,
    models=None,
    standardize=True,
):
    """
    Reconstruct the monthly AeroCom data dictionary directly from processed NetCDF files.

    Reads files structured as:
        <output_base_dir>/<variable>/<model>_<variable>_processed.nc

    and returns the same dictionary structure that the legacy master pickle used:
        data[model][variable] = xr.Dataset

    Missing variables are returned as None. This function lets downstream notebooks
    and scripts work without relying on the master pickle.

    Parameters
    ----------
    output_base_dir : str
        Root directory containing the variable sub-folders (default: Data/AP3_processed_monthly).
    variables : list of str, optional
        Variables to load. If None, all sub-folders of output_base_dir are scanned.
    models : list of str, optional
        Models to load. If None, all models with at least one file are loaded.
    standardize : bool, default True
        Apply standardize_dataset() to align coordinates and variable names.

    Returns
    -------
    dict
        {model: {variable: xr.Dataset or None}}
    """
    if not os.path.isdir(output_base_dir):
        raise FileNotFoundError(f"Processed monthly directory not found: {output_base_dir}")

    if variables is None:
        variables = sorted(
            d for d in os.listdir(output_base_dir)
            if os.path.isdir(os.path.join(output_base_dir, d)) and d != "derived"
        )
    else:
        variables = list(variables)

    data_dict = {}
    loaded_count = 0

    for var in variables:
        var_dir = os.path.join(output_base_dir, var)
        if not os.path.isdir(var_dir):
            continue

        for fname in os.listdir(var_dir):
            if not fname.endswith("_processed.nc"):
                continue

            # Filenames are: <model>_<variable>_processed.nc
            parts = fname.rsplit("_", 2)
            if len(parts) != 3 or parts[1] != var or parts[2] != "processed.nc":
                continue
            model = parts[0]

            if models is not None and model not in models:
                continue

            fpath = os.path.join(var_dir, fname)
            try:
                ds = xr.open_dataset(fpath)
                if standardize:
                    ds = standardize_dataset(ds, var)
                data_dict.setdefault(model, {})[var] = ds
                loaded_count += 1
            except Exception as e:
                logging.warning(f"Could not load {fpath}: {e}")
                data_dict.setdefault(model, {})[var] = None

    # Ensure every model has an entry for every requested variable
    for model in data_dict:
        for var in variables:
            if var not in data_dict[model]:
                data_dict[model][var] = None

    print(f"Loaded {loaded_count} NetCDF files for {len(data_dict)} models from {output_base_dir}")
    return data_dict
