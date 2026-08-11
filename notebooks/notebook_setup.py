
import os
import sys
import gc
import glob

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
# ==============================================================================
# GLOBAL CONFIGURATIONS
# ==============================================================================


REGIONS = {
    'global': {
        'surface_type': 'all', 'lon_range': (0, 360), 'lat_range': (-90, 90),
        'time_slice': ('2010-01-01', '2010-12-31'), 'edge_weighted': False,
    },
    'africa': {
        'surface_type': 'land', 'lon_range': (15, 37), 'lat_range': (-15, 0),
        'time_slice': ('2010-06-01', '2010-09-30'), 'edge_weighted': False,
    },
    'amazon': {
        'surface_type': 'land', 'lon_range': (287, 317), 'lat_range': (-17, -3),
        'time_slice': ('2010-07-01', '2010-10-31'), 'edge_weighted': False,
    },
    'outflow_af': {
        'surface_type': 'ocean', 'lon_range': (350, 15), 'lat_range': (-15, 0),
        'time_slice': ('2010-06-01', '2010-09-30'), 'edge_weighted': True,
    },
    'se_asia': {
        'surface_type': 'land', 'lon_range': (95, 125), 'lat_range': (-5, 15),
        'time_slice': ('2010-03-01', '2010-03-31'), 'edge_weighted': False,
    },
    'boreal_na': {
        'surface_type': 'land', 'lon_range': (242, 258), 'lat_range': (52, 63),
        'time_slice': ('2010-06-01', '2010-08-31'), 'edge_weighted': False,
    },
    'eastern_siberia': {
        'surface_type': 'land', 'lon_range': (120, 170), 'lat_range': (50, 65),
        'time_slice': ('2010-07-01', '2010-07-31'), 'edge_weighted': False,
    },
    'west_russia': {
        'surface_type': 'land', 'lon_range': (33, 52), 'lat_range': (51, 60),
        'time_slice': ('2010-07-15', '2010-09-10'), 'edge_weighted': False,
    },
    'boreal_na_west': {
        'surface_type': 'land', 'lon_range': (242, 258), 'lat_range': (52, 63),
        'time_slice': ('2010-06-15', '2010-08-31'), 'edge_weighted': False,
    },
    'indonesia_peatland': {
        'surface_type': 'land',
        'lon_range': (98, 118), 'lat_range': (-5, 5),
        'time_slice': ('2010-08-01', '2010-11-15'), 'edge_weighted': False,
    }
}


def aggregate_region(model_dict, var_name, region_name, return_time_series=False, skipna=False):
    """Spatially aggregate `var_name` for every model."""
    cfg = REGIONS[region_name]
    result = {}
    for model, model_data in model_dict.items():
        if var_name not in model_data or model_data[var_name] is None:
            continue
        try:
            result[model] = ct.regional_aggregate(
                model_data[var_name], masks[region_name],
                spatial='mean', edge_weighted=cfg['edge_weighted'],
                time_slice=cfg['time_slice'], temporal='mean',
                return_time_series=return_time_series, skipna=skipna,
            )
        except Exception as e:
            print(f'  Aggregation failed for {var_name} {model} {region_name}: {e}')
    return result


def compute_derived_after_aggregation(monthly_dict, seasonal_dict):
    """Compute lifetime-like variables from aggregated load/emission."""
    lifetime_after_agg = 'lifetime' in DERIVED_VAR_AFTER_AGG
    lifetime_bcoa_after_agg = lifetime_after_agg or 'lifetime_BC_OA' in DERIVED_VAR_AFTER_AGG

    if lifetime_bcoa_after_agg:
        for agg in (monthly_dict, seasonal_dict):
            for region in REGIONS:
                load = agg[region].get('load_BC_OA', {})
                emi = agg[region].get('emi_BC_OA', {})
                out = {}
                for model in load:
                    if model in emi:
                        try:
                            out[model] = load[model] / (emi[model] * 3600 * 24)
                        except Exception as e:
                            print(f'  Failed post-aggregation lifetime_BC_OA for {model} {region}: {e}')
                agg[region]['lifetime_BC_OA'] = out

    if lifetime_after_agg:
        for agg in (monthly_dict, seasonal_dict):
            for region in REGIONS:
                load = agg[region].get('load_total', {})
                emi = agg[region].get('emi_total', {})
                out = {}
                for model in load:
                    if model in emi:
                        try:
                            out[model] = load[model] / (emi[model] * 3600 * 24)
                        except Exception as e:
                            print(f'  Failed post-aggregation lifetime for {model} {region}: {e}')
                agg[region]['lifetime'] = out


def short_model_name(model):
    """Compact tick label: strip common AeroCom CTRL suffixes."""
    name = str(model)
    for suffix in (
        '_AP3-CTRL2016-PD', '_AP3-CTRL2016', '_AP3-CTRL-2010',
        '_AP3-CTRL2019', '_AP3-CTRL', '-met2010',
    ):
        name = name.replace(suffix, '')
    return name

def filter_lifetime_after_agg(monthly_dict, seasonal_dict, max_days=None, min_days=None):
    """Mask unphysical after-aggregation lifetime values with NaN."""
    if max_days is None and min_days is None:
        return
    # Preserve unfiltered BC+OA lifetimes for diagnostics / decomp fallback.
    global lifetime_BC_OA_raw
    lifetime_BC_OA_raw = {region: {} for region in REGIONS}
    for region in REGIONS:
        for model, val in (seasonal_dict.get(region, {}).get('lifetime_BC_OA') or {}).items():
            try:
                v = float(val) if not hasattr(val, 'values') else float(val.values)
            except Exception:
                continue
            lifetime_BC_OA_raw[region][model] = v
    for var in ['lifetime', 'lifetime_BC_OA']:
        for agg in (monthly_dict, seasonal_dict):
            for region in REGIONS:
                if var not in agg[region]:
                    continue
                for model, val in list(agg[region][var].items()):
                    try:
                        v = float(val) if not hasattr(val, 'values') else float(val.values)
                    except Exception:
                        continue
                    if (not np.isfinite(v)
                            or (max_days is not None and v > max_days)
                            or (min_days is not None and v < min_days)):
                        
                        agg[region][var][model] = np.nan


# ==============================================================================
# DATA NORMALIZATION FUNCTIONS
# ==============================================================================

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
    try:
        global_mean = float(da.mean().values)
    except Exception:
        global_mean = np.nan
    if not np.isnan(global_mean) and global_mean > 100.0:
        da = da / 1000.0
    da.attrs['units'] = 'mm day-1'
    return da.to_dataset(name=var_name)


def normalize_dataset_time(ds, var_hint=None):
    """Normalise the time coordinate of a dataset to first-of-month."""
    if ds is None:
        return None
    if not list(ds.data_vars):
        return None
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]
    if 'time' not in da.dims or len(da.time) == 0:
        return ds
    if not isinstance(da.time.values[0], (np.datetime64, pd.Timestamp)):
        try:
            if hasattr(da.indexes['time'], 'to_datetimeindex'):
                try:
                    new_times = da.indexes['time'].to_datetimeindex().values
                except OverflowError:
                    new_times = np.array([
                        np.datetime64(f'{t.year:04d}-{t.month:02d}-01', 'ns')
                        for t in da.indexes['time']
                    ])
            else:
                new_times = pd.to_datetime([str(t) for t in da.time.values]).values
            da = da.assign_coords(time=new_times)
        except Exception as e:
            print(f'  Warning: cftime conversion failed for {var_hint}: {e}', flush=True)
            return ds
    da = functions.normalize_monthly_time(da)
    return da.to_dataset(name=var_name)


def sum_datasets(model_data, keys, out_name, require_all=True):
    """Sum variables into a Dataset.

    If require_all=True, return None unless every key in `keys` is present
    (avoids e.g. emi_BC_OA = emibc alone when emioa file is missing).
    Otherwise missing components are skipped (legacy behaviour for totals).
    """
    missing = [k for k in keys if model_data.get(k) is None]
    if require_all and missing:
        return None
    dsets = [model_data[k] for k in keys if model_data.get(k) is not None]
    if not dsets:
        return None
    arrays = [d[list(d.data_vars)[0]] for d in dsets]
    total = arrays[0].copy()
    for a in arrays[1:]:
        total = total + a
    return total.to_dataset(name=out_name)


def filter_lifetime(da, model_name, var_name, max_days=None, min_days=None):
    """Mask unrealistic lifetime values and return statistics."""
    if da is None:
        return None, {'n_total': 0, 'n_excluded': 0}
    if max_days is None and min_days is None:
        return da, {'n_total': int(da.size), 'n_excluded': 0}
    bad = np.isnan(da) | np.isinf(da)
    if max_days is not None:
        bad = bad | (da > max_days)
    if min_days is not None:
        bad = bad | (da < min_days)
    n_excluded = int(bad.sum())
    if n_excluded:
        print(f'  Lifetime filter: {model_name} {var_name} excluded {n_excluded:,} / {da.size:,} values',
              flush=True)
    return da.where(~bad), {'n_total': int(da.size), 'n_excluded': n_excluded}

###################################
## 