#!/usr/bin/env python
"""
Sensitivity tests for the regression-based AAOD error attribution
(Zhong et al. 2023, Sci. Adv.).

This script starts from the current `notebooks/error_attribution.ipynb`
methodology and tests a series of literature recommendations one at a time.
For each test it prints the resulting mean decomposition percentages, the
regression coefficients, and the constrained values.

Run with the project `.venv`:

    python notebooks/error_attribution_sensitivity_tests.py

"""

import sys
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
try:
    project_root = Path(__file__).parent.parent
except NameError:
    # Running inside a Jupyter notebook: use the working directory (project root)
    project_root = Path.cwd()
py_dir = project_root / 'py'
if str(py_dir) not in sys.path:
    sys.path.insert(0, str(py_dir))

import functions
import cameo_toolbox as ct
import aerocom_data

print(f'Project root: {project_root}')
print(f'Python dir:   {py_dir}')


# ---------------------------------------------------------------------------
# Configuration (matches the current notebook baseline)
# ---------------------------------------------------------------------------
EXCLUDE_MODELS = []
VARIABLES = [
    'abs550aer', 'depbc', 'depdust', 'depoa', 'depso2', 'depso4', 'depss',
    'emibc', 'emidust', 'emioa', 'emiso2', 'emiss', 'loadbc', 'loaddust',
    'loadoa', 'loadso2', 'loadso4', 'loadss', 'od440aer', 'od550aer',
    'od870aer', 'od865aer', 'od550bc', 'od550dust', 'od550oa', 'od550so4', 'od550ss',
    'precip'
]

DERIVED_VAR_AFTER_AGG = ['lifetime']
# The current baseline disables the per-cell lifetime filter because lifetime
# is derived from aggregated load / emission.
LIFETIME_MAX_DAYS = None
LIFETIME_MIN_DAYS = None
POST_AGG_LIFETIME_MAX_DAYS = None
POST_AGG_LIFETIME_MIN_DAYS = None

PAPER_DECOMP = {
    'amazon': {'pct_MAC': 47},
    'africa': {'pct_E': 40, 'pct_MAC': 31},
}

REGIONS = {
    'africa': {
        'surface_type': 'land',
        'lon_range': (15, 37), 'lat_range': (-15, 0),
        'time_slice': ('2010-06-01', '2010-09-30'), 'edge_weighted': False,
    },
    'amazon': {
        'surface_type': 'land',
        'lon_range': (287, 317), 'lat_range': (-17, -3),
        'time_slice': ('2010-07-01', '2010-10-31'), 'edge_weighted': False,
    },
    'outflow_af': {
        'surface_type': 'ocean',
        'lon_range': (350, 8), 'lat_range': (-15, 3),
        'time_slice': ('2010-06-01', '2010-09-30'), 'edge_weighted': True,
    },
}

from xarray.coding.times import decode_cf_datetime


# ---------------------------------------------------------------------------
# Time normalization
# ---------------------------------------------------------------------------
def normalize_dataset_time(ds: Optional[xr.Dataset], var_hint: Optional[str] = None) -> Optional[xr.Dataset]:
    """Normalize the time coordinate of a monthly dataset to first-of-month."""
    if ds is None or not list(ds.data_vars):
        return ds
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]
    if 'time' not in da.dims or len(da.time) == 0:
        return ds

    def _to_month_start(values):
        try:
            return pd.to_datetime(values).to_period('M').to_timestamp().values
        except Exception:
            return np.array([
                np.datetime64(f'{t.year:04d}-{t.month:02d}-01', 'ns')
                for t in values
            ])

    try:
        if not isinstance(da.time.values[0], (np.datetime64, pd.Timestamp)):
            raw = da.time.values
            # Numeric year-month convention (e.g. GEOS stores 201001 ... 201012)
            if raw.dtype.kind in 'iuf' and np.all(raw > 100000) and np.all(raw < 999999):
                pass
            else:
                try:
                    if 'units' in da.time.attrs:
                        try:
                            new_times = decode_cf_datetime(
                                da.time.values,
                                da.time.attrs.get('units'),
                                da.time.attrs.get('calendar', 'standard')
                            )
                        except Exception:
                            new_times = pd.to_datetime([str(t) for t in da.time.values]).values
                    elif hasattr(da.indexes.get('time', None), 'to_datetimeindex'):
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
                    print(f'  Warning: cftime conversion failed for {var_hint}: {e}')
                    return ds
        da = functions.normalize_monthly_time(da)
        da = da.assign_coords(time=_to_month_start(da.time.values))
    except Exception as e:
        print(f'  Warning: time normalization failed for {var_hint}: {e}')
        return ds
    return da.to_dataset(name=var_name)


# ---------------------------------------------------------------------------
# Load raw monthly data
# ---------------------------------------------------------------------------
print('\nLoading monthly model data from NetCDF...')
raw_data = aerocom_data.load_monthly_data_from_netcdf(
    output_base_dir=str(project_root / 'Data' / 'AP3_processed_monthly'),
    variables=VARIABLES,
)
missing_excluded = [m for m in EXCLUDE_MODELS if m not in raw_data]
if missing_excluded:
    print(f'Warning: excluded models not found in data: {missing_excluded}')
models = sorted([m for m in raw_data.keys() if m not in EXCLUDE_MODELS])
actually_excluded = [m for m in EXCLUDE_MODELS if m in raw_data]
print(f'Number of models: {len(models)} (excluded {len(actually_excluded)}: {actually_excluded})')

# Normalize time coordinates and build a clean data dict
data = {m: {} for m in models}
for m in models:
    for var, ds in raw_data[m].items():
        data[m][var] = normalize_dataset_time(ds, f'{m}/{var}')


# ---------------------------------------------------------------------------
# Helper to extract a DataArray from a variable dict entry
# ---------------------------------------------------------------------------
def get_da(model_dict: Dict[str, xr.Dataset], var: str) -> Optional[xr.DataArray]:
    if model_dict.get(var) is None:
        return None
    ds = model_dict[var]
    if var in ds.data_vars:
        return ds[var]
    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return ds[data_vars[0]]
    return None


# ---------------------------------------------------------------------------
# Add combined variables and normalize precipitation to mm/day
# ---------------------------------------------------------------------------
for m in models:
    md = data[m]
    bc = get_da(md, 'loadbc')
    oa = get_da(md, 'loadoa')
    if bc is not None and oa is not None:
        md['load_BC_OA'] = (bc + oa).to_dataset(name='load_BC_OA')
    emibc = get_da(md, 'emibc')
    emioa = get_da(md, 'emioa')
    if emibc is not None and emioa is not None:
        md['emi_BC_OA'] = (emibc + emioa).to_dataset(name='emi_BC_OA')
    loads = [get_da(md, k) for k in ['loadbc', 'loaddust', 'loadoa', 'loadso4', 'loadss'] if get_da(md, k) is not None]
    emis = [get_da(md, k) for k in ['emibc', 'emidust', 'emioa', 'emiso2', 'emiss'] if get_da(md, k) is not None]
    if loads:
        md['load_total'] = sum(loads).to_dataset(name='load_total')
    if emis:
        md['emi_total'] = sum(emis).to_dataset(name='emi_total')
    # Precipitation units: convert to mm day^-1
    if md.get('precip') is not None:
        da = get_da(md, 'precip')
        units = da.attrs.get('units', '')
        da = da * 86400.0  # kg m^-2 s^-1 -> mm day^-1
        # Some files are labelled g m^-2 s^-1 but actually contain kg-scale numbers
        if ('g' in str(units).lower() and 'kg' not in str(units).lower()) or float(da.mean()) < 0.1:
            da = da * 1000.0
        md['precip'] = da.to_dataset(name='precip')


# ---------------------------------------------------------------------------
# Compute derived grid-cell variables
# ---------------------------------------------------------------------------
derived = {m: {} for m in models}
for m in models:
    for dv in ['MEC', 'MAC', 'SSA', 'AE']:
        ds = aerocom_data.calculate_derived_var(data[m], m, dv)
        if ds is not None:
            derived[m][dv] = ds[list(ds.data_vars)[0]]
        else:
            derived[m][dv] = None

# Flatten base + derived variables into DataArrays for aggregation
data_derived = {m: {} for m in models}
for m in models:
    for k, v in data[m].items():
        if v is not None:
            data_derived[m][k] = get_da(data[m], k)
    for k, v in derived[m].items():
        if v is not None:
            data_derived[m][k] = v

print('\nDerived-variable availability (pre-aggregation):')
for dv in ['MEC', 'MAC', 'SSA', 'AE']:
    n = sum(1 for m in derived if derived[m].get(dv) is not None)
    print(f'  {dv:15s}: {n:2d} / {len(models)} models')


# ---------------------------------------------------------------------------
# Regional masks and aggregation helpers
# ---------------------------------------------------------------------------
sample_model = next((m for m in models if data[m].get('od550aer') is not None), None)
template = data[sample_model]['od550aer'].isel(time=0)
print(f'\nTemplate grid from {sample_model}: {template.dims}')

masks = {}
for name, cfg in REGIONS.items():
    masks[name] = ct.create_region_mask(
        template, region=name,
        surface_type=cfg['surface_type'],
        mask_registry=masks,
    )
print('Regions created:', list(masks.keys()))


def aggregate_region(model_dict: Dict[str, Dict[str, xr.DataArray]], var_name: str,
                     region_name: str, return_time_series: bool = False, skipna: bool = False) -> Dict[str, float]:
    """Spatially aggregate `var_name` for every model, returning time series or seasonal mean."""
    cfg = REGIONS[region_name]
    result = {}
    for model, md in model_dict.items():
        if var_name not in md or md[var_name] is None:
            continue
        try:
            result[model] = ct.regional_aggregate(
                md[var_name], masks[region_name],
                spatial='mean', edge_weighted=cfg['edge_weighted'],
                time_slice=cfg['time_slice'], temporal='mean',
                return_time_series=return_time_series, skipna=skipna,
            )
        except Exception as e:
            print(f'  Aggregation failed for {var_name} {model} {region_name}: {e}')
    return result


VARIABLES_TO_AGGREGATE = [
    'MEC', 'MAC', 'SSA', 'AE', 'load_BC_OA', 'emi_BC_OA', 'load_total', 'emi_total', 'precip', 'abs550aer'
]


def compute_seasonal_aggregates(model_dict: Dict[str, Dict[str, xr.DataArray]],
                                post_agg_max: Optional[float] = None,
                                post_agg_min: Optional[float] = None,
                                lifetime_overrides: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build model_seasonal and model_monthly, then add post-aggregation lifetimes."""
    model_monthly = {region: {var: aggregate_region(model_dict, var, region, return_time_series=True, skipna=False)
                             for var in VARIABLES_TO_AGGREGATE}
                     for region in REGIONS}
    model_seasonal = {region: {var: aggregate_region(model_dict, var, region, return_time_series=False, skipna=False)
                               for var in VARIABLES_TO_AGGREGATE}
                      for region in REGIONS}

    # BC+OA lifetime from aggregated load/emission
    for region in REGIONS:
        load = model_seasonal[region].get('load_BC_OA', {})
        emi = model_seasonal[region].get('emi_BC_OA', {})
        out = {}
        for model in load:
            if model in emi:
                try:
                    out[model] = load[model] / (emi[model] * 3600 * 24)
                except Exception:
                    pass
        model_seasonal[region]['lifetime_BC_OA'] = out
        model_monthly[region]['lifetime_BC_OA'] = {}
        for model in load:
            if model in emi and model in model_monthly[region]['load_BC_OA']:
                try:
                    model_monthly[region]['lifetime_BC_OA'][model] = model_monthly[region]['load_BC_OA'][model] / (model_monthly[region]['emi_BC_OA'][model] * 3600 * 24)
                except Exception:
                    pass
        # Total lifetime
        load = model_seasonal[region].get('load_total', {})
        emi = model_seasonal[region].get('emi_total', {})
        out = {}
        for model in load:
            if model in emi:
                try:
                    out[model] = load[model] / (emi[model] * 3600 * 24)
                except Exception:
                    pass
        model_seasonal[region]['lifetime'] = out

    # Apply overrides (e.g. deposition-based outflow lifetime)
    if lifetime_overrides:
        for region, overrides in lifetime_overrides.items():
            if region not in model_seasonal:
                continue
            model_seasonal[region]['lifetime_BC_OA'] = {model: val for model, val in overrides.items()}

    # Post-aggregation physicality filter
    if post_agg_max is not None or post_agg_min is not None:
        for var in ['lifetime', 'lifetime_BC_OA']:
            for region in REGIONS:
                if var not in model_seasonal[region]:
                    continue
                for model, val in list(model_seasonal[region][var].items()):
                    try:
                        v = float(val) if not hasattr(val, 'values') else float(val.values)
                    except Exception:
                        continue
                    if not np.isfinite(v) or (post_agg_max is not None and v > post_agg_max) or (post_agg_min is not None and v < post_agg_min):
                        model_seasonal[region][var][model] = np.nan
    return model_monthly, model_seasonal


# ---------------------------------------------------------------------------
# POLDER-GRASP observations
# ---------------------------------------------------------------------------
polder_path = project_root / 'Data' / 'AP3_POLDER_Collocated' / 'POLDER_GRASP_coloc_3h_AP3_2010_lon0_10.0_lat0_10.0.parquet'
print(f'\nLoading POLDER observations: {polder_path}')
polder_cols = ['time', 'longitude', 'latitude', 'AOD_550', 'AAOD_550', 'AOD_440', 'AOD_870']
polder_df = pd.read_parquet(polder_path, columns=polder_cols)
polder_df['time'] = pd.to_datetime(polder_df['time'])
polder_df['month'] = polder_df['time'].dt.to_period('M')
with np.errstate(divide='ignore', invalid='ignore'):
    polder_df['SSA'] = 1.0 - polder_df['AAOD_550'] / polder_df['AOD_550']
    polder_df['AE'] = -np.log(polder_df['AOD_550'] / polder_df['AOD_440']) / np.log(550.0 / 440.0)
polder_df = polder_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['SSA', 'AE', 'AAOD_550', 'AOD_550'])
print(f'POLDER rows after filtering: {len(polder_df)}')


def polder_region_mask(df: pd.DataFrame, lon_range: Tuple[float, float], lat_range: Tuple[float, float]) -> pd.Series:
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    if lon_min > lon_max:
        mask_lon = (df['longitude'] >= lon_min) | (df['longitude'] <= lon_max)
    else:
        mask_lon = (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)
    mask_lat = (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)
    return mask_lon & mask_lat


def polder_monthly_means(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    """Compute monthly regional means from POLDER 3-hourly data."""
    cfg = REGIONS[region_name]
    mask = polder_region_mask(df, cfg['lon_range'], cfg['lat_range'])
    sub = df[mask].copy()
    t0, t1 = cfg['time_slice']
    sub = sub[(sub['time'] >= t0) & (sub['time'] <= t1)]
    if sub.empty:
        return pd.DataFrame()
    sub['coslat'] = np.cos(np.deg2rad(sub['latitude']))
    rows = []
    for month, g in sub.groupby('month'):
        if len(g) < 3:
            continue
        w = g['coslat']
        rows.append({
            'month': month,
            'time': pd.Timestamp(month.start_time),
            'SSA': np.average(g['SSA'], weights=w),
            'AE': np.average(g['AE'], weights=w),
            'AAOD_550': np.average(g['AAOD_550'], weights=w),
            'AOD_550': np.average(g['AOD_550'], weights=w),
            'n': len(g),
        })
    return pd.DataFrame(rows)


polder_monthly = {region: polder_monthly_means(polder_df, region) for region in REGIONS}
print('POLDER monthly means per region:')
for region, df in polder_monthly.items():
    print(f'  {region}: {len(df)} months')


# ---------------------------------------------------------------------------
# GPCP observed precipitation
# ---------------------------------------------------------------------------
gpcp_path = project_root / 'Data' / 'Prec' / 'GPCP_2010_0-360.nc'
print(f'\nLoading GPCP precipitation: {gpcp_path}')
gpcp = xr.open_dataset(gpcp_path)
gpcp_precip = gpcp['sat_gauge_precip']
print(f'GPCP dims: {gpcp_precip.dims}, shape: {gpcp_precip.shape}')


def gpcp_region_mean(precip_da: xr.DataArray, region_name: str) -> xr.DataArray:
    cfg = REGIONS[region_name]
    lon_min, lon_max = cfg['lon_range']
    lat_min, lat_max = cfg['lat_range']
    if lon_min > lon_max:
        sub = xr.concat([precip_da.sel(lon=slice(lon_min, 360)), precip_da.sel(lon=slice(0, lon_max))], dim='lon')
    else:
        sub = precip_da.sel(lon=slice(lon_min, lon_max))
    sub = sub.sel(lat=slice(lat_max, lat_min))
    sub = sub.sel(time=slice(*cfg['time_slice']))
    coslat = np.cos(np.deg2rad(sub.lat))
    weights = coslat * xr.ones_like(sub.isel(time=0))
    weights = weights / weights.sum()
    spatial_mean = (sub * weights).sum(dim=['lat', 'lon'])
    return spatial_mean


gpcp_region = {region: gpcp_region_mean(gpcp_precip, region) for region in REGIONS}
print('\nGPCP regional mean precipitation (mm/day):')
for region, da in gpcp_region.items():
    mean_val = float(da.mean().values) if da.size else np.nan
    print(f'  {region}: {mean_val:.3f}')


# ---------------------------------------------------------------------------
# Model regressions
# ---------------------------------------------------------------------------
def build_regression_df(model_seasonal: Dict) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        for model in models:
            ssa = model_seasonal[region]['SSA'].get(model)
            mac = model_seasonal[region]['MAC'].get(model)
            ae = model_seasonal[region]['AE'].get(model)
            lt = model_seasonal[region]['lifetime_BC_OA'].get(model)
            precip = model_seasonal[region]['precip'].get(model)
            if not all(v is not None for v in [ssa, mac, ae, lt, precip]):
                continue
            if not np.isfinite(float(lt)):
                continue
            rows.append({
                'region': region, 'model': model,
                'SSA': float(ssa), 'MAC': float(mac), 'AE': float(ae),
                'lifetime': float(lt), 'inv_lifetime': 1.0 / float(lt),
                'precip': float(precip),
            })
    return pd.DataFrame(rows)


def fit_mac_ssa(reg_df: pd.DataFrame, intercept: bool = False) -> Dict[str, Dict]:
    results = {}
    for region in REGIONS:
        sub = reg_df[(reg_df['region'] == region)].dropna(subset=['MAC', 'SSA'])
        if len(sub) < 3:
            continue
        x = sub['SSA'].values
        y = sub['MAC'].values
        if intercept:
            X = np.column_stack([x, np.ones(len(x))])
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            F, b = coeffs[0], coeffs[1]
            y_pred = F * x + b
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
        else:
            F = np.nansum(x * y) / np.nansum(x ** 2)
            residuals = y - F * x
            r2 = 1 - np.nansum(residuals ** 2) / np.nansum(y ** 2)
            b = 0.0
        results[region] = {'F': F, 'b': b, 'r2': r2, 'n': len(sub)}
    return results


def fit_inv_lifetime(reg_df: pd.DataFrame) -> Dict[str, Dict]:
    results = {}
    for region in REGIONS:
        sub = reg_df[(reg_df['region'] == region)].dropna(subset=['inv_lifetime', 'precip', 'AE'])
        if len(sub) < 4:
            continue
        X = np.column_stack([sub['precip'].values, sub['AE'].values])
        y = sub['inv_lifetime'].values
        X_ols = np.column_stack([np.ones(len(X)), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_ols, y, rcond=None)
        y_pred = X_ols @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        results[region] = {
            'intercept': coeffs[0], 'alpha': coeffs[1], 'beta': coeffs[2],
            'r2': r2, 'n': len(sub)
        }
    return results


# ---------------------------------------------------------------------------
# Constrained estimates from observations
# ---------------------------------------------------------------------------
def constrain_estimates(mac_ssa_results: Dict, inv_lt_results: Dict,
                        aaod_obs: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Compute constrained E, tau, MAC from observations."""
    if aaod_obs is None:
        aaod_obs = {}
    rows = []
    for region in REGIONS:
        polder_df_region = polder_monthly[region]
        if polder_df_region.empty:
            continue
        if region not in mac_ssa_results or region not in inv_lt_results:
            continue
        gpcp_ts = gpcp_region[region]
        mac_params = mac_ssa_results[region]
        lt_params = inv_lt_results[region]
        ssa_obs_mean = polder_df_region['SSA'].mean()
        ae_obs_mean = polder_df_region['AE'].mean()
        aaod_obs_mean = aaod_obs.get(region, polder_df_region['AAOD_550'].mean())
        precip_obs_mean = float(gpcp_ts.mean().values) if gpcp_ts.size else np.nan
        mac_c = mac_params['F'] * ssa_obs_mean + mac_params.get('b', 0.0)
        inv_lt_c = lt_params['intercept'] + lt_params['alpha'] * precip_obs_mean + lt_params['beta'] * ae_obs_mean
        tau_c = 1.0 / inv_lt_c if inv_lt_c > 0 else np.nan
        e_c_g_m2_day = aaod_obs_mean / (tau_c * mac_c) if (tau_c > 0 and mac_c > 0) else np.nan
        e_c = e_c_g_m2_day / (1000.0 * 86400.0)
        rows.append({
            'region': region, 'SSA_obs': ssa_obs_mean, 'AE_obs': ae_obs_mean,
            'AAOD_obs': aaod_obs_mean, 'precip_obs': precip_obs_mean,
            'MAC_c': mac_c, 'tau_c': tau_c, 'E_c_g_m2_day': e_c_g_m2_day, 'E_c': e_c,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Error decomposition
# ---------------------------------------------------------------------------
def decompose(model_seasonal: Dict, constrained_df: pd.DataFrame,
              emission_adjustments: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    """Standard first-order error decomposition."""
    if emission_adjustments is None:
        emission_adjustments = {}
    rows = []
    for region in REGIONS:
        csub = constrained_df[constrained_df['region'] == region]
        if csub.empty:
            continue
        mac_c = csub['MAC_c'].mean()
        tau_c = csub['tau_c'].mean()
        aaod_c = csub['AAOD_obs'].mean()
        e_c = csub['E_c'].mean()
        for model in models:
            mac_model = model_seasonal[region]['MAC'].get(model)
            lt_model = model_seasonal[region]['lifetime_BC_OA'].get(model)
            aaod_model = None
            if model_seasonal[region].get('load_BC_OA') and model_seasonal[region]['load_BC_OA'].get(model) and \
               model_seasonal[region].get('MAC') and model_seasonal[region]['MAC'].get(model):
                aaod_model = float(model_seasonal[region]['MAC'][model]) * float(model_seasonal[region]['load_BC_OA'][model]) * 1000.0
            emi_model = model_seasonal[region].get('emi_BC_OA', {}).get(model)
            if any(v is None or np.isnan(v) for v in [mac_model, lt_model, emi_model, mac_c, tau_c, aaod_c, e_c]):
                continue
            mac_model = float(mac_model); lt_model = float(lt_model); emi_model = float(emi_model); e_c = float(e_c)
            # Apply optional emission adjustments (e.g. SOA)
            emi_model = emi_model + emission_adjustments.get(region, {}).get(model, 0.0)
            emi_model_g = emi_model * 1000.0 * 86400.0
            e_c_g = e_c * 1000.0 * 86400.0
            aaod_model = float(aaod_model) if aaod_model is not None else emi_model_g * lt_model * mac_model
            dMAC = mac_model - mac_c
            dtau = lt_model - tau_c
            dE_g = emi_model_g - e_c_g
            dAAOD_E = dE_g * tau_c * mac_c
            dAAOD_tau = e_c_g * dtau * mac_c
            dAAOD_MAC = e_c_g * tau_c * dMAC
            dAAOD_total = aaod_model - aaod_c
            cross = dAAOD_total - (dAAOD_E + dAAOD_tau + dAAOD_MAC)
            denom = abs(dAAOD_E) + abs(dAAOD_tau) + abs(dAAOD_MAC) + abs(cross)
            if denom > 0:
                pct_E = abs(dAAOD_E) / denom * 100
                pct_tau = abs(dAAOD_tau) / denom * 100
                pct_MAC = abs(dAAOD_MAC) / denom * 100
                pct_cross = abs(cross) / denom * 100
            else:
                pct_E = pct_tau = pct_MAC = pct_cross = np.nan
            rows.append({
                'region': region, 'model': model,
                'MAC_model': mac_model, 'MAC_c': mac_c,
                'tau_model': lt_model, 'tau_c': tau_c,
                'E_model': emi_model, 'E_c': e_c,
                'AAOD_model': aaod_model, 'AAOD_c': aaod_c,
                'pct_E': pct_E, 'pct_tau': pct_tau,
                'pct_MAC': pct_MAC, 'pct_cross': pct_cross,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# POLDER-GRASP homogenization helpers
# ---------------------------------------------------------------------------
def sample_model_aaod_at_polder(model_da: xr.DataArray, polder_sub: pd.DataFrame) -> Optional[float]:
    """Sample a model monthly AAOD field at the POLDER locations/times and return a seasonal weighted mean."""
    if model_da is None or polder_sub.empty:
        return np.nan
    lat_vals = model_da.lat.values
    lon_vals = model_da.lon.values
    lat_idx = np.abs(lat_vals[:, None] - polder_sub['latitude'].values[None, :]).argmin(axis=0)
    lon_idx = np.abs(lon_vals[:, None] - polder_sub['longitude'].values[None, :]).argmin(axis=0)
    times = pd.to_datetime(model_da.time.values)
    polder_times = pd.to_datetime(polder_sub['time'].values)
    month_idx = np.array([np.argmin(np.abs(times - pd.Timestamp(t).to_period('M').to_timestamp())) for t in polder_times])
    vals = model_da.values[month_idx, lat_idx, lon_idx]
    weights = np.cos(np.deg2rad(polder_sub['latitude'].values))
    finite = np.isfinite(vals)
    if finite.sum() < 3:
        return np.nan
    return float(np.average(vals[finite], weights=weights[finite]))


def homogenize_polder_aaod(model_seasonal: Dict, polder_df: pd.DataFrame) -> Dict[str, float]:
    """Build a regional-vs-sampled model regression and homogenize the observed regional AAOD."""
    homogenized = {}
    for region in REGIONS:
        if region == 'outflow_af':
            # Keep POLDER mean for outflow; homogenization is defined for source regions
            sub = polder_df[polder_region_mask(polder_df, REGIONS[region]['lon_range'], REGIONS[region]['lat_range'])]
            t0, t1 = REGIONS[region]['time_slice']
            sub = sub[(sub['time'] >= t0) & (sub['time'] <= t1)]
            if not sub.empty:
                homogenized[region] = float(np.average(sub['AAOD_550'], weights=np.cos(np.deg2rad(sub['latitude']))))
            continue
        regional = model_seasonal[region].get('abs550aer', {})
        sampled = {}
        for model, val in regional.items():
            da = get_da(data[model], 'abs550aer')
            sub = polder_df[polder_region_mask(polder_df, REGIONS[region]['lon_range'], REGIONS[region]['lat_range'])].copy()
            t0, t1 = REGIONS[region]['time_slice']
            sub = sub[(sub['time'] >= t0) & (sub['time'] <= t1)]
            s = sample_model_aaod_at_polder(da, sub)
            if np.isfinite(s):
                sampled[model] = s
        common = [m for m in regional if m in sampled and np.isfinite(float(regional[m]))]
        if len(common) < 3:
            print(f'  Homogenization skipped for {region}: only {len(common)} common models')
            continue
        x = np.array([float(regional[m]) for m in common])
        y = np.array([sampled[m] for m in common])
        # Regression: sampled = a * regional + b
        X = np.column_stack([x, np.ones(len(x))])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b = coeffs[0], coeffs[1]
        # Observed raw sampled mean
        sub = polder_df[polder_region_mask(polder_df, REGIONS[region]['lon_range'], REGIONS[region]['lat_range'])].copy()
        t0, t1 = REGIONS[region]['time_slice']
        sub = sub[(sub['time'] >= t0) & (sub['time'] <= t1)]
        if sub.empty:
            continue
        obs_sampled = float(np.average(sub['AAOD_550'], weights=np.cos(np.deg2rad(sub['latitude']))))
        homogenized_regional = (obs_sampled - b) / a if a != 0 else np.nan
        homogenized[region] = homogenized_regional
        print(f'  {region}: sampled = {a:.4f}*regional + {b:.4f}, R2={1 - np.sum((y - a*x - b)**2)/np.sum((y - y.mean())**2):.3f}, n={len(common)}')
        print(f'    raw sampled AAOD = {obs_sampled:.5f}, homogenized regional AAOD = {homogenized_regional:.5f}')
    return homogenized


# ---------------------------------------------------------------------------
# Outflow meta-model helpers
# ---------------------------------------------------------------------------
def fit_outflow_meta_model(model_seasonal: Dict, reg_df: pd.DataFrame) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """Fit AAOD_outflow = a*E*tau*MAC + b*E*tau + c*MAC + d using African source values."""
    region = 'outflow_af'
    src = 'africa'
    rows = []
    for model in models:
        aaod_out = model_seasonal[region].get('abs550aer', {}).get(model)
        e_src = model_seasonal[src].get('emi_BC_OA', {}).get(model)
        tau_src = model_seasonal[src].get('lifetime_BC_OA', {}).get(model)
        mac_src = model_seasonal[src].get('MAC', {}).get(model)
        if not all(v is not None and np.isfinite(float(v)) for v in [aaod_out, e_src, tau_src, mac_src]):
            continue
        rows.append({
            'model': model, 'AAOD_out': float(aaod_out),
            'E': float(e_src), 'tau': float(tau_src), 'MAC': float(mac_src),
            'X1': float(e_src) * float(tau_src) * float(mac_src),
            'X2': float(e_src) * float(tau_src),
            'X3': float(mac_src),
        })
    df = pd.DataFrame(rows)
    if len(df) < 4:
        return None, df
    X = df[['X1', 'X2', 'X3']].values
    y = df['AAOD_out'].values
    X_ols = np.column_stack([X, np.ones(len(X))])
    coeffs, _, _, _ = np.linalg.lstsq(X_ols, y, rcond=None)
    a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    y_pred = X_ols @ coeffs
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    slope, intercept, _, _, _ = stats.linregress(y, y_pred)
    print(f'  outflow meta-model: AAOD = {a:.4e}*X1 + {b:.4e}*X2 + {c:.4e}*X3 + {d:.4e}')
    print(f'    R2={r2:.3f}, RMSE={rmse:.5f}, predicted-vs-model slope={slope:.3f}, n={len(df)}')
    return {'a': a, 'b': b, 'c': c, 'd': d, 'r2': r2, 'rmse': rmse, 'slope': slope, 'n': len(df)}, df


def decompose_outflow_meta(model_seasonal: Dict, constrained_df: pd.DataFrame,
                           meta_params: Dict, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Decompose outflow AAOD error using the meta-model sensitivities to source E, tau, MAC."""
    region = 'outflow_af'
    src = 'africa'
    csrc = constrained_df[constrained_df['region'] == src]
    if csrc.empty or meta_params is None:
        return pd.DataFrame()
    e_c = float(csrc['E_c'].mean())
    tau_c = float(csrc['tau_c'].mean())
    mac_c = float(csrc['MAC_c'].mean())
    a, b, c, d = meta_params['a'], meta_params['b'], meta_params['c'], meta_params['d']
    y_c = a * (e_c * tau_c * mac_c) + b * (e_c * tau_c) + c * mac_c + d
    rows = []
    for _, row in meta_df.iterrows():
        model = row['model']
        e_m = row['E']
        tau_m = row['tau']
        mac_m = row['MAC']
        aaod_model = row['AAOD_out']
        dE = e_m - e_c
        dTau = tau_m - tau_c
        dMAC = mac_m - mac_c
        dAAOD_E = a * (tau_c * mac_c * dE) + b * (tau_c * dE)
        dAAOD_tau = a * (e_c * mac_c * dTau) + b * (e_c * dTau)
        dAAOD_MAC = a * (e_c * tau_c * dMAC) + c * dMAC
        dAAOD_total = aaod_model - y_c
        cross = dAAOD_total - (dAAOD_E + dAAOD_tau + dAAOD_MAC)
        denom = abs(dAAOD_E) + abs(dAAOD_tau) + abs(dAAOD_MAC) + abs(cross)
        if denom > 0:
            pct_E = abs(dAAOD_E) / denom * 100
            pct_tau = abs(dAAOD_tau) / denom * 100
            pct_MAC = abs(dAAOD_MAC) / denom * 100
            pct_cross = abs(cross) / denom * 100
        else:
            pct_E = pct_tau = pct_MAC = pct_cross = np.nan
        rows.append({
            'region': region, 'model': model,
            'AAOD_model': aaod_model, 'AAOD_c': y_c,
            'pct_E': pct_E, 'pct_tau': pct_tau, 'pct_MAC': pct_MAC, 'pct_cross': pct_cross,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Deposition-based outflow lifetime
# ---------------------------------------------------------------------------
def compute_deposition_lifetime_outflow() -> Dict[str, float]:
    """Compute outflow BC+OA lifetime as load / (abs(depbc)+abs(depoa)) for each model."""
    region = 'outflow_af'
    cfg = REGIONS[region]
    lifetimes = {}
    for model in models:
        load_da = data_derived[model].get('load_BC_OA')
        depbc = data_derived[model].get('depbc')
        depoa = data_derived[model].get('depoa')
        if load_da is None or depbc is None or depoa is None:
            continue
        try:
            dep_total = depbc.where(depbc > 0, -depbc) + depoa.where(depoa > 0, -depoa)
            # Aggregate to outflow region/season
            load_agg = ct.regional_aggregate(load_da, masks[region], spatial='mean', edge_weighted=cfg['edge_weighted'],
                                             time_slice=cfg['time_slice'], temporal='mean', return_time_series=False, skipna=False)
            dep_agg = ct.regional_aggregate(dep_total, masks[region], spatial='mean', edge_weighted=cfg['edge_weighted'],
                                            time_slice=cfg['time_slice'], temporal='mean', return_time_series=False, skipna=False)
            lt = float(load_agg) / (float(dep_agg) * 3600 * 24)
            if np.isfinite(lt) and lt > 0:
                lifetimes[model] = lt
        except Exception as e:
            print(f'  Deposition lifetime failed for {model}: {e}')
    return lifetimes


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
def print_test_header(name: str):
    print('\n' + '=' * 70)
    print(f'TEST: {name}')
    print('=' * 70)


def print_regressions(mac_ssa_results: Dict, inv_lt_results: Dict):
    print('\nModel regression coefficients:')
    for region, res in mac_ssa_results.items():
        if abs(res.get('b', 0.0)) < 1e-12:
            print(f"  {region}: MAC = {res['F']:.4f} * SSA  (uncentered R^2={res['r2']:.3f}, n={res['n']})")
        else:
            print(f"  {region}: MAC = {res['F']:.4f} * SSA + {res['b']:.4f}  (R^2={res['r2']:.3f}, n={res['n']})")
    for region, res in inv_lt_results.items():
        print(f"  {region}: 1/tau = {res['intercept']:.4f} + {res['alpha']:.4f}*Pr + {res['beta']:.4f}*AE  (R^2={res['r2']:.3f}, n={res['n']})")


def print_constrained(constrained_df: pd.DataFrame):
    print('\nConstrained values:')
    for _, row in constrained_df.iterrows():
        print(f"  {row['region']}: MAC_c={row['MAC_c']:.4f}, tau_c={row['tau_c']:.2f} days, E_c={row['E_c']:.3e} kg m^-2 s^-1")


def print_decomposition(decomp_df: pd.DataFrame, test_name: str, extra_regions: Optional[List[str]] = None):
    if decomp_df.empty:
        print('No decomposition rows available.')
        return
    regions = ['africa', 'amazon']
    if extra_regions:
        regions += [r for r in extra_regions if r not in regions]
    mean_decomp = decomp_df.groupby('region')[['pct_E', 'pct_tau', 'pct_MAC', 'pct_cross']].mean()
    counts = decomp_df.groupby('region').size()
    print(f'\n{test_name} mean decomposition:')
    for region in regions:
        if region not in mean_decomp.index:
            continue
        row = mean_decomp.loc[region]
        n = int(counts.loc[region])
        print(f"  {region} (n={n}): pct_E={row['pct_E']:.1f}%, pct_tau={row['pct_tau']:.1f}%, "
              f"pct_MAC={row['pct_MAC']:.1f}%, pct_cross={row['pct_cross']:.1f}%")
        if region in PAPER_DECOMP:
            for var, pap in PAPER_DECOMP[region].items():
                cur = row[var]
                print(f"    vs paper {var}: current={cur:.1f}%, paper≈{pap:.1f}%, diff={cur-pap:+.1f}%")


# ---------------------------------------------------------------------------
# Baseline test
# ---------------------------------------------------------------------------
# %%
print_test_header('1. Baseline')
model_monthly, model_seasonal = compute_seasonal_aggregates(data_derived)
reg_df = build_regression_df(model_seasonal)
mac_ssa_results = fit_mac_ssa(reg_df, intercept=False)
inv_lt_results = fit_inv_lifetime(reg_df)
constrained_df = constrain_estimates(mac_ssa_results, inv_lt_results)
decomp_df = decompose(model_seasonal, constrained_df)
print_regressions(mac_ssa_results, inv_lt_results)
print_constrained(constrained_df)
print_decomposition(decomp_df, 'Baseline')

baseline_decomp = decomp_df.groupby('region')[['pct_E', 'pct_tau', 'pct_MAC', 'pct_cross']].mean()


# ---------------------------------------------------------------------------
# Test 2: POLDER-GRASP homogenization
# ---------------------------------------------------------------------------
# %%
print_test_header('2. POLDER-GRASP homogenization')
homogenized_aaod = homogenize_polder_aaod(model_seasonal, polder_df)
constrained_df_hom = constrain_estimates(mac_ssa_results, inv_lt_results, aaod_obs=homogenized_aaod)
decomp_df_hom = decompose(model_seasonal, constrained_df_hom)
print_constrained(constrained_df_hom)
print_decomposition(decomp_df_hom, 'POLDER homogenization')


# ---------------------------------------------------------------------------
# Test 3: Include SOA in Amazon emissions
# ---------------------------------------------------------------------------
# %%
print_test_header('3. Include SOA in Amazon emissions')
# Infer a regional SOA flux from the mean model Amazon emi_BC_OA gap to the constrained E_c
amazon_c = constrained_df[constrained_df['region'] == 'amazon']
if amazon_c.empty:
    soa_flux = 0.0
else:
    amazon_emissions = [v for v in model_seasonal['amazon']['emi_BC_OA'].values() if np.isfinite(float(v))]
    mean_amazon_emi = float(np.mean(amazon_emissions)) if amazon_emissions else np.nan
    e_c_amazon = float(amazon_c['E_c'].mean())
    soa_flux = max(0.0, e_c_amazon - mean_amazon_emi)
    print(f'  Amazon constrained E_c = {e_c_amazon:.3e} kg m^-2 s^-1')
    print(f'  Mean model emi_BC_OA   = {mean_amazon_emi:.3e} kg m^-2 s^-1')
    print(f'  Inferred SOA flux added to each model = {soa_flux:.3e} kg m^-2 s^-1')

soa_adjustment = {'amazon': {model: soa_flux for model in model_seasonal['amazon']['emi_BC_OA'].keys()}}
decomp_df_soa = decompose(model_seasonal, constrained_df, emission_adjustments=soa_adjustment)
print_decomposition(decomp_df_soa, 'Amazon + inferred SOA')


# ---------------------------------------------------------------------------
# Test 4: MAC/SSA regression with intercept
# ---------------------------------------------------------------------------
# %%
print_test_header('4. MAC/SSA regression with intercept')
mac_ssa_int = fit_mac_ssa(reg_df, intercept=True)
inv_lt_int = fit_inv_lifetime(reg_df)  # unchanged
constrained_df_int = constrain_estimates(mac_ssa_int, inv_lt_int)
decomp_df_int = decompose(model_seasonal, constrained_df_int)
print_regressions(mac_ssa_int, inv_lt_int)
print_constrained(constrained_df_int)
print_decomposition(decomp_df_int, 'MAC/SSA with intercept')


# ---------------------------------------------------------------------------
# Test 5: Post-aggregation lifetime filter
# ---------------------------------------------------------------------------
# %%
decomp_df_filt_100 = None
for max_days in [365, 100]:
    print_test_header(f'5. Post-aggregation lifetime filter (max {max_days} days)')
    _, model_seasonal_filt = compute_seasonal_aggregates(data_derived, post_agg_max=max_days, post_agg_min=1e-3)
    reg_df_filt = build_regression_df(model_seasonal_filt)
    dropped = len(reg_df) - len(reg_df_filt)
    print(f'  Regression rows: {len(reg_df_filt)} (dropped {dropped} due to lifetime filter)')
    mac_ssa_filt = fit_mac_ssa(reg_df_filt, intercept=False)
    inv_lt_filt = fit_inv_lifetime(reg_df_filt)
    constrained_df_filt = constrain_estimates(mac_ssa_filt, inv_lt_filt)
    decomp_df_filt = decompose(model_seasonal_filt, constrained_df_filt)
    print_regressions(mac_ssa_filt, inv_lt_filt)
    print_constrained(constrained_df_filt)
    print_decomposition(decomp_df_filt, f'Lifetime filter max={max_days}d')
    if max_days == 100:
        decomp_df_filt_100 = decomp_df_filt.copy()


# ---------------------------------------------------------------------------
# Test 6: Outflow meta-model prediction
# ---------------------------------------------------------------------------
# %%
print_test_header('6. Outflow as meta-model prediction')
meta_params, meta_df = fit_outflow_meta_model(model_seasonal, reg_df)
if meta_params is not None:
    decomp_meta = decompose_outflow_meta(model_seasonal, constrained_df, meta_params, meta_df)
    print_decomposition(decomp_meta, 'Meta-model outflow', extra_regions=['outflow_af'])
    # Report whether extreme lifetimes are avoided
    baseline_outflow_lifetimes = [v for v in model_seasonal['outflow_af']['lifetime_BC_OA'].values() if np.isfinite(float(v))]
    print(f'  Baseline outflow lifetimes (load/emission): n={len(baseline_outflow_lifetimes)}, '
          f'mean={np.mean(baseline_outflow_lifetimes):.1f}, max={np.max(baseline_outflow_lifetimes):.1f} days')
    # Meta-model uses source-region lifetimes, which are much more stable
    source_lifetimes = [v for v in model_seasonal['africa']['lifetime_BC_OA'].values() if np.isfinite(float(v))]
    print(f'  Source-region lifetimes used by meta-model: n={len(source_lifetimes)}, '
          f'mean={np.mean(source_lifetimes):.1f}, max={np.max(source_lifetimes):.1f} days')


# ---------------------------------------------------------------------------
# Test 7: Use deposition for outflow lifetime
# ---------------------------------------------------------------------------
# %%
print_test_header('7. Use deposition for outflow lifetime')
dep_lifetime = compute_deposition_lifetime_outflow()
print(f'  Computed deposition-based outflow lifetimes for {len(dep_lifetime)} models:')
for model, lt in sorted(dep_lifetime.items())[:10]:
    print(f'    {model}: {lt:.2f} days')
if dep_lifetime:
    _, model_seasonal_dep = compute_seasonal_aggregates(data_derived, lifetime_overrides={'outflow_af': dep_lifetime})
    # Only recompute outflow regression; keep africa/amazon from baseline for source-region decomposition
    # We rebuild the full regression df and decomp for outflow
    reg_df_dep = build_regression_df(model_seasonal_dep)
    mac_ssa_dep = fit_mac_ssa(reg_df_dep, intercept=False)
    inv_lt_dep = fit_inv_lifetime(reg_df_dep)
    constrained_df_dep = constrain_estimates(mac_ssa_dep, inv_lt_dep)
    decomp_df_dep = decompose(model_seasonal_dep, constrained_df_dep)
    print('\nOutflow regression after deposition lifetime:')
    for region, res in inv_lt_dep.items():
        print(f"  {region}: 1/tau = {res['intercept']:.4f} + {res['alpha']:.4f}*Pr + {res['beta']:.4f}*AE  (R^2={res['r2']:.3f}, n={res['n']})")
    print_constrained(constrained_df_dep)
    outflow_c_dep = constrained_df_dep[constrained_df_dep['region'] == 'outflow_af']
    if outflow_c_dep.empty or not np.isfinite(float(outflow_c_dep['tau_c'].mean())):
        print('  Note: the outflow 1/tau regression from deposition lifetimes gives an unphysical constrained tau_c')
        print('        (negative inverse-lifetime), so the standard outflow decomposition is not produced.')
    print_decomposition(decomp_df_dep, 'Deposition outflow lifetime', extra_regions=['outflow_af'])


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
# %%
print_test_header('Summary comparison')
summary_data = {
    'Baseline': baseline_decomp,
    'POLDER homogenization': decomp_df_hom.groupby('region')[['pct_E', 'pct_tau', 'pct_MAC', 'pct_cross']].mean() if not decomp_df_hom.empty else pd.DataFrame(),
    'Amazon + inferred SOA': decomp_df_soa.groupby('region')[['pct_E', 'pct_tau', 'pct_MAC', 'pct_cross']].mean() if not decomp_df_soa.empty else pd.DataFrame(),
    'MAC/SSA intercept': decomp_df_int.groupby('region')[['pct_E', 'pct_tau', 'pct_MAC', 'pct_cross']].mean() if not decomp_df_int.empty else pd.DataFrame(),
    'Lifetime filter max=100d': decomp_df_filt_100.groupby('region')[['pct_E', 'pct_tau', 'pct_MAC', 'pct_cross']].mean() if (decomp_df_filt_100 is not None and not decomp_df_filt_100.empty) else pd.DataFrame(),
}
print('\nMean decomposition percentages by test and region:')
for region in ['africa', 'amazon']:
    print(f'\n{region.upper()}:')
    print(f"  {'Test':<30} {'n':>4}   pct_E  pct_tau  pct_MAC  pct_cross")
    for test_name, df in summary_data.items():
        if region in df.index:
            row = df.loc[region]
            # n is not stored in summary; show mean only
            print(f"  {test_name:<30}      {row['pct_E']:6.1f}  {row['pct_tau']:6.1f}  {row['pct_MAC']:6.1f}  {row['pct_cross']:6.1f}")
    print('  Paper targets:')
    for var, val in PAPER_DECOMP.get(region, {}).items():
        print(f"    {var} ≈ {val}%")

print('\nDone.')
