"""Test EXCLUDE_MODELS and lifetime filtering for Figure 3 analysis."""

import sys
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore', category=RuntimeWarning)

project_root = Path('/scistor/guest/gbb083/AeroCom')
py_dir = project_root / 'py'
if str(py_dir) not in sys.path:
    sys.path.insert(0, str(py_dir))

import functions
import cameo_toolbox as ct
import aerocom_data

# Test configuration
EXCLUDE_MODELS = ['GEOS-i33p2-met2010_AP3-CTRL', 'MIROC-SPRINTARS_AP3-CTRL']
LIFETIME_MAX_DAYS = 365
LIFETIME_MIN_DAYS = 1e-3

monthly_pickle = project_root / 'Data' / 'var_files' / 'original' / 'monthly' / 'monthly_aerocom_data.pickle'
VARIABLES = [
    'abs550aer', 'depbc', 'depdust', 'depoa', 'depso2', 'depso4', 'depss',
    'emibc', 'emidust', 'emioa', 'emiso2', 'emiss', 'loadbc', 'loaddust',
    'loadoa', 'loadso2', 'loadso4', 'loadss', 'od440aer', 'od550aer',
    'od870aer', 'od865aer', 'od550bc', 'od550dust', 'od550oa', 'od550so4', 'od550ss',
    'precip'
]

print(f'Test: EXCLUDE_MODELS = {EXCLUDE_MODELS}')
print(f'Test: LIFETIME_MAX_DAYS = {LIFETIME_MAX_DAYS}, LIFETIME_MIN_DAYS = {LIFETIME_MIN_DAYS}')
print(f'Loading {monthly_pickle}')
with open(monthly_pickle, 'rb') as f:
    raw_data = pickle.load(f)

models = sorted([m for m in raw_data.keys() if m not in EXCLUDE_MODELS])
print(f'Models after exclusion: {len(models)}')
assert not any(m in models for m in EXCLUDE_MODELS), 'Excluded models still present'
print('PASS: excluded models are absent from model list')


def normalize_dataset_time(ds, var_hint=None):
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
                new_times = da.indexes['time'].to_datetimeindex().values
            else:
                new_times = pd.to_datetime([str(t) for t in da.time.values]).values
            da = da.assign_coords(time=new_times)
        except Exception:
            return ds
    da = functions.normalize_monthly_time(da)
    return da.to_dataset(name=var_name)


def sum_datasets(model_data, keys, out_name):
    dsets = [model_data[k] for k in keys if model_data.get(k) is not None]
    if not dsets:
        return None
    arrays = [d[list(d.data_vars)[0]] for d in dsets]
    total = arrays[0].copy()
    for a in arrays[1:]:
        total = total + a
    return total.to_dataset(name=out_name)


LOAD_VARS = ['loadbc', 'loaddust', 'loadoa', 'loadso2', 'loadso4', 'loadss']
EMI_VARS = ['emibc', 'emidust', 'emioa', 'emiso2', 'emiss']
BC_OA_LOAD = ['loadbc', 'loadoa']
BC_OA_EMI = ['emibc', 'emioa']

data = {}
for m in models:
    normalized = {}
    for var in VARIABLES:
        if raw_data[m].get(var) is None:
            continue
        try:
            normalized[var] = normalize_dataset_time(raw_data[m][var], var_hint=f'{m}/{var}')
        except Exception:
            normalized[var] = None
    normalized['load_total'] = sum_datasets(normalized, LOAD_VARS, 'load_total')
    normalized['emi_total'] = sum_datasets(normalized, EMI_VARS, 'emi_total')
    normalized['load_BC_OA'] = sum_datasets(normalized, BC_OA_LOAD, 'load_BC_OA')
    normalized['emi_BC_OA'] = sum_datasets(normalized, BC_OA_EMI, 'emi_BC_OA')
    data[m] = normalized


def filter_lifetime(da, max_days=None, min_days=None):
    if da is None or (max_days is None and min_days is None):
        return da
    bad = np.isnan(da) | np.isinf(da)
    if max_days is not None:
        bad = bad | (da > max_days)
    if min_days is not None:
        bad = bad | (da < min_days)
    return da.where(~bad)


derived_vars = ['MEC', 'MAC', 'SSA', 'AE']
derived = {m: {} for m in models}
for m in models:
    for dv in derived_vars:
        ds = aerocom_data.calculate_derived_var(data[m], m, dv)
        derived[m][dv] = None if ds is None else ds[list(ds.data_vars)[0]]

    if data[m].get('load_BC_OA') is not None and data[m].get('emi_BC_OA') is not None:
        lt = data[m]['load_BC_OA']['load_BC_OA'] / (data[m]['emi_BC_OA']['emi_BC_OA'] * 3600 * 24)
        derived[m]['lifetime_BC_OA'] = filter_lifetime(lt, LIFETIME_MAX_DAYS, LIFETIME_MIN_DAYS)
    else:
        derived[m]['lifetime_BC_OA'] = None

# Check that no finite lifetime_BC_OA values exceed the bounds before aggregation
for m in models:
    da = derived[m]['lifetime_BC_OA']
    if da is None:
        continue
    finite = da.where(np.isfinite(da)).values
    finite = finite[~np.isnan(finite)]
    assert (finite <= LIFETIME_MAX_DAYS).all(), f'{m}: lifetime_BC_OA > {LIFETIME_MAX_DAYS}'
    assert (finite >= LIFETIME_MIN_DAYS).all(), f'{m}: lifetime_BC_OA < {LIFETIME_MIN_DAYS}'
print('PASS: no finite lifetime_BC_OA values outside filter bounds')

# Build regional means and regression data (only africa/amazon, seasonal)
REGIONS = {
    'africa': {
        'lon_range': (15, 37), 'lat_range': (-15, 0),
        'time_slice': ('2010-06-01', '2010-09-30'), 'edge_weighted': False,
    },
    'amazon': {
        'lon_range': (287, 317), 'lat_range': (-17, -3),
        'time_slice': ('2010-07-01', '2010-10-31'), 'edge_weighted': False,
    },
}

data_derived = {}
for m in models:
    data_derived[m] = {}
    for k, v in data[m].items():
        if v is not None:
            data_derived[m][k] = v[list(v.data_vars)[0]]
    for k, v in derived[m].items():
        if v is not None:
            data_derived[m][k] = v

sample_model = next((m for m in models if data[m].get('od550aer') is not None), None)
template = data[sample_model]['od550aer'].isel(time=0)
SURFACE_TYPE = 'all'  # 'all', 'land', or 'ocean'

masks = {}
for name, cfg in REGIONS.items():
    masks[name] = ct.create_region_mask(
        template, region=name,
        surface_type=SURFACE_TYPE,
        mask_registry=masks,
    )


def aggregate_region(model_dict, var_name, region_name, skipna=False):
    cfg = REGIONS[region_name]
    result = {}
    for model, model_data in model_dict.items():
        if var_name not in model_data or model_data[var_name] is None:
            continue
        try:
            val = ct.regional_aggregate(
                model_data[var_name], masks[region_name],
                spatial='mean', edge_weighted=cfg['edge_weighted'],
                time_slice=cfg['time_slice'], temporal='mean',
                skipna=skipna,
            )
            result[model] = val
        except Exception as e:
            print(f'  Aggregation failed for {var_name} {model} {region_name}: {e}')
    return result


model_seasonal = {}
for region in REGIONS:
    model_seasonal[region] = {}
    for var in ['SSA', 'MAC', 'AE', 'lifetime_BC_OA']:
        model_seasonal[region][var] = aggregate_region(data_derived, var, region, skipna=(var == 'lifetime_BC_OA'))

regression_rows = []
for region in REGIONS:
    for model in models:
        ssa = model_seasonal[region]['SSA'].get(model)
        mac = model_seasonal[region]['MAC'].get(model)
        ae = model_seasonal[region]['AE'].get(model)
        lt = model_seasonal[region]['lifetime_BC_OA'].get(model)
        if not all(v is not None for v in [ssa, mac, ae, lt]):
            continue
        if not np.isfinite(float(lt)):
            continue
        regression_rows.append({
            'region': region, 'model': model,
            'SSA': float(ssa), 'MAC': float(mac), 'AE': float(ae),
            'lifetime': float(lt), 'inv_lifetime': 1.0 / float(lt),
        })
reg_df = pd.DataFrame(regression_rows)
print(f'Regression rows: {len(reg_df)}')

# Verify no excluded models in regression data
for m in EXCLUDE_MODELS:
    assert m not in reg_df['model'].values, f'Excluded model {m} found in regression data'
print('PASS: excluded models are absent from regression data')

# Verify no lifetime values outside bounds in regression data
if LIFETIME_MAX_DAYS is not None:
    assert (reg_df['lifetime'] <= LIFETIME_MAX_DAYS).all(), 'lifetime > MAX found in regression data'
if LIFETIME_MIN_DAYS is not None:
    assert (reg_df['lifetime'] >= LIFETIME_MIN_DAYS).all(), 'lifetime < MIN found in regression data'
assert np.isfinite(reg_df['lifetime']).all(), 'non-finite lifetime found in regression data'
print('PASS: all regression lifetime values are within filter bounds and finite')

print('\nAll tests passed.')
