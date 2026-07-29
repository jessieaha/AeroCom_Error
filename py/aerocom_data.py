"""
Modernized, automated AeroCom Phase III data extraction engine.
This module dynamically scans model directories, discovers files matching variable patterns,
standardizes coordinates to ('lon', 'lat', 'time'), and returns clean, analysis-ready
lazy xarray Datasets. Missing variables are gracefully marked as None.


"""

import os
import sys
import gc
import glob
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.times import decode_cf_datetime

# Configure standard logging to report missing files cleanly
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Default zlib encoding complevel for processed NetCDF writes
_ZLIB_COMPLEVEL = 4
# ==============================================================================
# GLOBAL CONFIGURATIONS
# ==============================================================================
# VARIABLES = (
#     "abs550aer", "depbc", "depdust", "depoa", "depso2", "depso4", "depss",
#     "emibc", "emidust", "emioa", "emiso2", "emiss", "loadbc", "loaddust",
#     "loadoa", "loadso2", "loadso4", "loadss", "od440aer", "od550aer",
#     "od870aer", "od865aer", "od550bc", "od550dust", "od550oa", "od550so4", "od550ss"
# )

# # Define regional models that use 870nm instead of 440nm for Angstrom Exponent
# AE_870_MODELS = {
#     'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010', 
#     'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010'
# }
# do auto search on 870 first otherwise 865 then 440 
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

def normalize_dataset_time(ds, var_hint=None, year=None):
    """Normalize time coordinate of a dataset to first-of-month."""
    # Lazy import: functions pulls optional viz deps (cartopy) at module level
    _script_dir = Path(__file__).resolve().parent
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))
    import functions

    if ds is None or not list(ds.data_vars):
        return None
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]
    if 'time' not in da.dims or len(da.time) == 0:
        return ds

    if var_hint and 'GEOS' in str(var_hint).upper():
        try:
            ds = xr.decode_cf(ds)
            da = ds[var_name]
        except Exception as e:
            print(f'  Warning: GEOS decode_cf failed for {var_hint}: {e}')

    def _to_month_start(values):
        try:
            return pd.to_datetime(values).to_period('M').to_timestamp().values
        except Exception:
            return np.array([
                np.datetime64(f'{t.year:04d}-{t.month:02d}-01', 'ns')
                for t in values
            ])

    def _decode_months_since(raw, units, year=None):
        """Decode 'months since YYYY-MM-DD' units for non-standard calendars without cftime.

        If a target year is supplied and the raw values are 1..12 while the parsed
        reference year is far from the target year (e.g. OsloCTM3 files that store
        1..12 but claim 'Monthssince 1850-01-01'), treat the values as 1-based
        months of the target year instead.
        """
        import re
        m = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', units)
        if not m:
            return None
        # Accept strings like 'Monthssince 1850-01-01', 'months since 1850-01-01', etc.
        if 'month' not in units.lower().replace(' ', ''):
            return None
        months = np.asarray(raw, dtype=np.int64)
        ref_year, ref_month, ref_day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        total_months = (ref_year * 12 + ref_month - 1) + months
        years = total_months // 12
        months_of_year = (total_months % 12) + 1
        dates = np.array([
            np.datetime64(f'{int(y):04d}-{int(m):02d}-01', 'ns')
            for y, m in zip(years, months_of_year)
        ])
        # Fallback for files with a bogus reference year but 1-based month-of-year values.
        if (year is not None and
                np.all(months >= 1) and np.all(months <= 12) and
                np.all(np.abs(years - year) > 100)):
            dates = np.array([
                np.datetime64(f'{year:04d}-{int(m):02d}-01', 'ns')
                for m in months
            ])
        return dates

    def _cftime_to_datetime64(values):
        """Convert cftime objects (e.g. DatetimeJulian, DatetimeNoLeap) to datetime64.

        Monthly data is normalised to first-of-month afterwards, so mapping to
        year-month-01 is sufficient and avoids invalid days in non-standard calendars.
        """
        if len(values) == 0:
            return values
        first = values[0]
        if hasattr(first, 'year') and hasattr(first, 'month'):
            return np.array([
                np.datetime64(f'{t.year:04d}-{t.month:02d}-01', 'ns')
                for t in values
            ])
        return values

    if not isinstance(da.time.values[0], (np.datetime64, pd.Timestamp)):
        try:
            raw = da.time.values
            # Drop any fill values (e.g. OsloCTM3 emiss has a trailing 9.96921e36).
            if raw.dtype.kind in 'iuf':
                valid_mask = raw < 1e30
                if not np.all(valid_mask):
                    da = da.isel(time=np.where(valid_mask)[0])
                    raw = da.time.values

            calendar = da.time.attrs.get('calendar', 'standard')
            is_nonstd_calendar = calendar in ('365_day', '360_day', 'noleap', 'all_leap', '366_day')
            values_are_months = (
                raw.dtype.kind in 'iuf' and
                np.all(raw >= 1) and np.all(raw <= 12)
            )

            if raw.dtype.kind in 'iuf' and np.all(raw > 100000) and np.all(raw < 999999):
                pass
            elif is_nonstd_calendar and values_are_months and year is not None:
                # OsloCTM3-style files store 1..12 (months of the target year) but
                # give a bogus reference date / units string. Ignore the reference and
                # map directly to first-of-month of the year from the filename.
                new_times = np.array([
                    np.datetime64(f'{year:04d}-{int(round(m)):02d}-01', 'ns')
                    for m in raw
                ])
                da = da.assign_coords(time=new_times)
            elif 'units' in da.time.attrs:
                units = da.time.attrs.get('units', '')
                try:
                    new_times = decode_cf_datetime(raw, units, calendar)
                except Exception:
                    # Fallback for non-standard calendars (e.g. 365_day) without cftime.
                    new_times = _decode_months_since(raw, units, year=year)
                    if new_times is None:
                        new_times = pd.to_datetime([str(t) for t in raw]).values
                # decode_cf_datetime may return cftime objects (Julian/NoLeap/365_day).
                # Convert them to datetime64 so pandas/xarray can handle them later.
                new_times = _cftime_to_datetime64(new_times)
                da = da.assign_coords(time=new_times)
            elif hasattr(da.indexes.get('time', None), 'to_datetimeindex'):
                try:
                    new_times = da.indexes['time'].to_datetimeindex().values
                except Exception:
                    # Fall back: manually build datetime64 from cftime year/month
                    new_times = np.array([
                        np.datetime64(f'{t.year:04d}-{t.month:02d}-01', 'ns')
                        for t in da.indexes['time']
                    ])
                da = da.assign_coords(time=new_times)
            else:
                da = da.assign_coords(
                    time=pd.to_datetime([str(t) for t in raw]).values
                )
        except Exception as e:
            print(f'  Warning: cftime conversion failed for {var_hint}: {e}')
            return ds

    da = functions.normalize_monthly_time(da)
    da = da.assign_coords(time=_to_month_start(da.time.values))
    return da.to_dataset(name=var_name)



def normalize_precipitation_units(precip_ds):
    """Convert model precipitation to mm day^-1.

    Handles kg/m2/s, g/m2/s, and the typo 'km m-2 s-1' seen in ECHAM6.3-HAM2.3.
    Also applies a sanity check: if the global mean is > 100 mm/day after the
    stated-unit conversion, the file is likely in g/m2/s despite the label.
    """
    if precip_ds is None:
        return None
    var_name = list(precip_ds.data_vars)[0]
    da = precip_ds[var_name]
    units = da.attrs.get('units', 'kg m-2 s-1').lower().replace(' ', '')
    # Common variants
    if units in ('kgm-2s-1', 'kgm^-2s^-1', 'kg/m2/s', 'kgm-2s-1'):
        factor = 86400.0  # kg m^-2 s^-1 -> mm day^-1
    elif units in ('gm-2s-1', 'gm^-2s^-1', 'g/m2/s', 'gm-2s-1'):
        factor = 86.4  # g m^-2 s^-1 -> mm day^-1
    elif units in ('mmday-1', 'mm/day', 'mmday^-1', 'mmd-1'):
        factor = 1.0
    elif units in ('ms-1', 'm/s', 'm s-1'):
        factor = 86400.0 * 1000.0  # m s^-1 -> mm day^-1 (water density ~1000 kg/m3)
    else:
        # 'km m-2 s-1' is treated as a typo for kg m^-2 s^-1
        factor = 86400.0
    da = da * factor
    # Sanity check: global mean should be ~2-3 mm/day. If > 100 mm/day, divide by 1000
    # because the file was probably in g m^-2 s^-1 despite the label.
    # Compute once per file (streaming path); use skipna for robustness.
    try:
        mean_da = da.mean(skipna=True)
        # Force compute for dask-backed arrays; .values also works for numpy.
        global_mean = float(np.asarray(mean_da.values))
    except Exception:
        global_mean = np.nan
    if not np.isnan(global_mean) and global_mean > 100.0:
        da = da / 1000.0
    da.attrs['units'] = 'mm day-1'
    return da.to_dataset(name=var_name)


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


def preprocess_for_save(ds, var_name, model_hint=None, temporal='monthly', year=None):
    """Full analysis-ready preprocess: coords, lon 0-360, monthly time, precip units.

    Pipeline (matches AAOD_error_attribution.ipynb cell-5 expectations):
      1. standardize_dataset (rename / squeeze / normalize_longitude)
      2. normalize_dataset_time (first-of-month) — monthly only
      3. normalize_precipitation_units when var_name is precip/pr
    """
    ds = standardize_dataset(ds, var_name)
    hint = model_hint if model_hint is not None else var_name
    if temporal == 'monthly':
        ds = normalize_dataset_time(ds, var_hint=hint, year=year)
        if ds is None:
            return None
    if var_name in ('precip', 'pr'):
        ds = normalize_precipitation_units(ds)
    return ds


def _netcdf_encoding(ds):
    """zlib compression encoding for all data variables."""
    return {
        name: {'zlib': True, 'complevel': _ZLIB_COMPLEVEL}
        for name in ds.data_vars
    }


# ==============================================================================
# SAFE FILE HANDLE HELPERS
# ==============================================================================
def _close_dataset_safely(ds):
    """Close an xarray Dataset, swallowing any errors from a corrupted backend."""
    try:
        if ds is not None:
            ds.close()
    except Exception:
        pass


def _log_failed_files(failed_files, output_base_dir):
    """Write a timestamped log of failed source NetCDF files and print a summary."""
    if not failed_files:
        return

    os.makedirs(output_base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_base_dir, f"failed_netcdf_files_{timestamp}.log")

    try:
        with open(log_file, 'w') as f:
            f.write(f"# Failed NetCDF source files - {timestamp}\n")
            f.write(f"# Total failures: {len(failed_files)}\n\n")
            for filepath, model_var, err in failed_files:
                f.write(f"{model_var}\n")
                f.write(f"  file: {filepath}\n")
                f.write(f"  error: {err}\n\n")
    except Exception as e:
        logging.warning(f"Could not write failed-files log {log_file}: {e}")
        return

    print("\n=============================================")
    print(f"FAILED NETCDF FILES: {len(failed_files)}")
    print("=============================================")
    for filepath, model_var, err in failed_files:
        print(f"  {model_var}")
        print(f"    {filepath}")
        print(f"    -> {err}")
    print(f"\nFull failure log written to: {log_file}")
    print("=============================================\n")


def _extract_year_from_path(filepath):
    """Extract a four-digit year from the filename (e.g. 2010)."""
    import re
    m = re.search(r'(20\d{2}|19\d{2})', os.path.basename(filepath))
    return int(m.group(1)) if m else None


# ==============================================================================
# FILE DISCOVERY
# ==============================================================================
def find_variable_file(base_dir, model, var, temporal):
    """Locate the first matching AeroCom NetCDF for model/variable/temporal.

    Returns
    -------
    str or None
        Absolute path to the first matching file, or None if not found.
    """
    model_dir = os.path.join(base_dir, model)
    var_names = [var]
    if var == 'precip':
        var_names.append('pr')
    elif var == 'pr':
        var_names.append('precip')

    patterns = []
    for vn in var_names:
        patterns.extend([
            os.path.join(model_dir, f"*{vn}_*Column*2010*{temporal}*.nc"),
            os.path.join(model_dir, f"*{vn}_*Surface*2010*{temporal}*.nc"),
            os.path.join(model_dir, f"*{vn}_*ModelLevel*2010*{temporal}*.nc"),
            os.path.join(model_dir, f"*{vn}_*2010*{temporal}*.nc"),
            os.path.join(base_dir, f"*{model}*{vn}_*Column*2010*{temporal}*.nc"),
            os.path.join(base_dir, f"*{model}*{vn}_*Surface*2010*{temporal}*.nc"),
            os.path.join(base_dir, f"*{model}*{vn}_*ModelLevel*2010*{temporal}*.nc"),
            os.path.join(base_dir, f"*{model}*{vn}_*2010*{temporal}*.nc"),
        ])

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


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
        logging.info(f"Scanning directory for model: {model} ...")

        for var in variables:
            filepath = find_variable_file(base_dir, model, var, temporal)

            if filepath and os.path.exists(filepath):
                try:
                    ds = xr.open_dataset(filepath, chunks='auto', engine='netcdf4')

                    if standardize:
                        ds = standardize_dataset(ds, var)

                    data_dict[model][var] = ds
                    logging.debug(f"  Found & opened: {var} -> {os.path.basename(filepath)}")
                except Exception as e:
                    logging.warning(f"  Error reading '{filepath}' for variable '{var}': {e}")
                    data_dict[model][var] = None
            else:
                data_dict[model][var] = None

    return data_dict


# ==============================================================================
# STREAMING PROCESS + SAVE
# ==============================================================================
def process_and_save_model_data(
    base_dirs,
    variables,
    temporal,
    output_base_dir,
    renew=False,
):
    """Open, normalize, and save one model x variable at a time.

    Never builds a master dictionary of Datasets. Each file is opened,
    preprocessed (lon / time / precip), written, and closed before the next.

    Parameters
    ----------
    base_dirs : list of (base_dir, models)
        Ordered source groups, e.g. [(dir_primary, primary_models), ...].
        A model present in an earlier group is not reprocessed from a later one.
    variables : iterable of str
        Variable names to process.
    temporal : str
        Frequency filter ('monthly', '3hourly', ...).
    output_base_dir : str
        Root for processed NetCDFs: {base}/{var}/{model}_{var}_processed.nc
    renew : bool, default False
        If False, skip when the output already exists and is at least as new
        as the source (mtime). If True, overwrite all outputs.

    Returns
    -------
    dict
        Summary counts: saved, skipped, missing, errors, plus a 'failed_files'
        list of (filepath, model/var, error_message) tuples.
    """
    print(f"\nStreaming process -> {output_base_dir} (RENEW={renew})")
    summary = {'saved': 0, 'skipped': 0, 'missing': 0, 'errors': 0}
    failed_files = []
    seen_models = set()

    for base_dir, models in base_dirs:
        for model in models:
            if model in seen_models:
                continue
            seen_models.add(model)
            logging.info(f"Processing model: {model} ...")

            for var in variables:
                out_dir = os.path.join(output_base_dir, var)
                out_file = os.path.join(out_dir, f"{model}_{var}_processed.nc")
                filepath = find_variable_file(base_dir, model, var, temporal)

                if filepath is None or not os.path.exists(filepath):
                    summary['missing'] += 1
                    continue

                year = None

                if not renew and os.path.exists(out_file):
                    try:
                        if os.path.getmtime(out_file) >= os.path.getmtime(filepath):
                            summary['skipped'] += 1
                            continue
                    except OSError:
                        pass  # fall through and rewrite

                os.makedirs(out_dir, exist_ok=True)

                ds_raw = None
                ds = None
                try:
                    # Decode time ourselves so malformed/missing-space units (e.g.
                    # OsloCTM3's 'Monthssince 1850-01-01' with calendar '365_day')
                    # do not make xarray throw before preprocess_for_save can fix them.
                    ds_raw = xr.open_dataset(
                        filepath, chunks='auto', engine='netcdf4', decode_times=False
                    )
                    # Only pass the target year for non-standard calendars that need
                    # the reference-year override (e.g. OsloCTM3 365_day files).
                    calendar = ds_raw.time.attrs.get('calendar', 'standard') if 'time' in ds_raw.coords else 'standard'
                    if calendar in ('365_day', '360_day', 'noleap', 'all_leap', '366_day'):
                        year = _extract_year_from_path(filepath)
                    ds = preprocess_for_save(
                        ds_raw, var, model_hint=f'{model}/{var}', temporal=temporal,
                        year=year,
                    )
                    if ds is None:
                        summary['missing'] += 1
                        continue

                    # Load into memory so the source handle can be closed immediately.
                    ds = ds.load()
                    ds.to_netcdf(
                        out_file,
                        engine='netcdf4',
                        encoding=_netcdf_encoding(ds),
                    )
                    ds.close()
                    summary['saved'] += 1
                    logging.info(f"  Saved {model}/{var}")

                except Exception as e:
                    summary['errors'] += 1
                    failed_files.append((filepath, f"{model}/{var}", str(e)))
                    logging.warning(f"  Error processing {model}/{var}: {e}")
                    # Force cleanup of any leaked HDF5 objects after a backend error.
                    gc.collect()

                finally:
                    _close_dataset_safely(ds)
                    _close_dataset_safely(ds_raw)
                    ds = None
                    ds_raw = None

    print("=============================================")
    print(
        f"Done: {summary['saved']} saved, {summary['skipped']} skipped, "
        f"{summary['missing']} missing, {summary['errors']} errors."
    )
    print("=============================================\n")

    _log_failed_files(failed_files, output_base_dir)
    summary['failed_files'] = failed_files
    return summary


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

            od870 = _get_dataarray(model_data.get('od870aer'), 'od870aer')
            od865 = _get_dataarray(model_data.get('od865aer'), 'od865aer')
            od_other = None 
            if od870 is not None:
                od_other = od870
                other_wavelength = 870
            elif od865 is not None:
                od_other = od865
                other_wavelength = 865
            else:
                od_other = _get_dataarray(model_data.get('od440aer'), 'od440aer')
                other_wavelength = 440

            if od550 is None:
                raise ValueError(f"Missing 550nm AOD for {model_name}.")
            elif od_other is None:
                raise ValueError(f"Missing other spectral bands to compute Angstrom Exponent for {model_name}.")

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

    Prefer process_and_save_model_data() for memory-efficient streaming writes.

    :param data_dict: The master dictionary {model: {variable: xr.Dataset}}
    :param output_base_dir: The root folder where processed data will be saved
    """
    print(f"\nSaving processed data to {output_base_dir} ...")

    saved_count = 0
    missing_count = 0

    for model, variables in data_dict.items():
        for var_name, ds in variables.items():
            if ds is not None:
                var_out_dir = os.path.join(output_base_dir, var_name)
                os.makedirs(var_out_dir, exist_ok=True)

                out_file = os.path.join(var_out_dir, f"{model}_{var_name}_processed.nc")

                try:
                    ds.to_netcdf(
                        out_file,
                        engine='netcdf4',
                        encoding=_netcdf_encoding(ds),
                    )
                    saved_count += 1
                except Exception as e:
                    print(f"  Error saving {out_file}: {e}")
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
                ds = xr.open_dataset(fpath, chunks='auto', engine='netcdf4')
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
