"""AOD error attribution for AeroCom biomass-burning aerosols (Zhong et al. 2022).

This script reproduces the framework of the paper
``s41467-022-33680-4`` (Nature Communications):

    AOD = E × τ × MEC                                    (1)

where E is the total aerosol emission, τ is the lifetime (burden / emission),
and MEC is the mass extinction coefficient (AOD / burden).  The notebook
produces:

* **Figure 2 style** – multi-region coloured scatter of MEC vs AE and
  precipitation vs 1/τ, plus validation panels (predicted vs modelled).
* **Figure 4 style** – stacked-bar decomposition of model AOD errors into
  emission, lifetime, MEC, and cross-term contributions (models alphabetical).
* **Figure 5/6 style** – African outflow Default / EC / MFC meta-model
  prediction and AOD error maps (model − POLDER).

## Configuration

* `EXCLUDE_MODELS` – list of model names to drop before processing.
* `INTERCEPT_0` – if True, force MEC–AE through origin; default False (OLS with intercept).
* `POLDER_HOMOGENIZE` – if True (default), apply Zhong-style regional AOD homogenization.
* `INCLUDE_AMAZON_SOA` – if True, inflate Amazon OA emissions (Sci. Adv. sensitivity);
  default False for Nat. Commun. 2022 baseline.
* `DERIVED_VAR_AFTER_AGG` – compute lifetime/MEC from regional means after aggregation (paper Methods).
* `LIFETIME_MIN_DAYS` / `LIFETIME_MAX_DAYS` – grid-cell lifetime filter (disabled when lifetime is post-agg).
* `SAVE_FIGURE` / `SAVE_CSV` – output control.

## Notes and caveats

* Loads directly from `Data/AP3_processed_monthly/` (no pickle).
* African outflow is treated as a meta-model prediction target (not load/emission lifetime).
* POLDER-GRASP and GPCP provide observational constraints.
"""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import linregress

warnings.filterwarnings('ignore', category=RuntimeWarning)

# -------------------------------------------------------------------------------
# 1. Configuration and data loading
# -------------------------------------------------------------------------------
project_root = Path('/scistor/guest/gbb083/AeroCom')
py_dir = project_root / 'py'
if str(py_dir) not in sys.path:
    sys.path.insert(0, str(py_dir))

import functions
import cameo_toolbox as ct
import aerocom_data

print(f'Project root: {project_root}')
print(f'Python dir:   {py_dir}')

# Model exclusion list (must match NetCDF model names exactly).
EXCLUDE_MODELS = []

# Lifetime outlier filtering.  Grid-cell lifetime values outside this range,
# plus NaN/inf, are masked before aggregation and regressions.
LIFETIME_MIN_DAYS = 1e-3
LIFETIME_MAX_DAYS = 365

# Paper Methods: τ and MEC from regional/seasonal means of burden, emission, AOD.
# Post-agg lifetime (and MEC) matches Zhong et al. 2022; grid-cell mean τ is too long.
DERIVED_VAR_AFTER_AGG = ['lifetime', 'MEC']
if 'lifetime' in DERIVED_VAR_AFTER_AGG:
    LIFETIME_MIN_DAYS = None
    LIFETIME_MAX_DAYS = None

# MEC vs AE regression: False -> OLS with intercept (default); True -> through origin.
INTERCEPT_0 = False

# Zhong-style regional AOD/AE homogenization for constrained E (default True).
POLDER_HOMOGENIZE = True
# Exclude monthly-only output models from the homogenization regression (paper SM Method 1).
MONTHLY_OUTPUT_MODELS = []

# Amazon SOA inflation is a 2023 Sci. Adv. sensitivity — off for Nat. Commun. 2022 baseline.
INCLUDE_AMAZON_SOA = False
AMAZON_SOA_FRACTION = 0.52

# Save figures to disk instead of rendering them inline.
SAVE_FIGURE = False

# Save CSV output tables.
SAVE_CSV = True

print(f'  INTERCEPT_0          = {INTERCEPT_0}  '
      f"({'through origin' if INTERCEPT_0 else 'OLS with intercept'})")
print(f'  POLDER_HOMOGENIZE    = {POLDER_HOMOGENIZE}')
print(f'  INCLUDE_AMAZON_SOA   = {INCLUDE_AMAZON_SOA}  (fraction={AMAZON_SOA_FRACTION})')
print(f'  DERIVED_VAR_AFTER_AGG= {DERIVED_VAR_AFTER_AGG}')
print(f'  EXCLUDE_MODELS      = {EXCLUDE_MODELS}')
print(f'  MONTHLY_OUTPUT_MODELS= {MONTHLY_OUTPUT_MODELS}')

VARIABLES = [
    'abs550aer', 'depbc', 'depdust', 'depoa', 'depso2', 'depso4', 'depss',
    'emibc', 'emidust', 'emioa', 'emiso2', 'emiss',
    'loadbc', 'loaddust', 'loadoa', 'loadso2', 'loadso4', 'loadss',
    'od440aer', 'od550aer', 'od870aer', 'od865aer',
    'od550bc', 'od550dust', 'od550oa', 'od550so4', 'od550ss',
    'precip'
]

print('Loading monthly data from processed NetCDF files...')
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

# -------------------------------------------------------------------------------
# 2. Normalise time, build summed variables, and normalise precipitation units
# -------------------------------------------------------------------------------
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
            print(f'  Warning: cftime conversion failed for {var_hint}: {e}')
            return ds
    da = functions.normalize_monthly_time(da)
    return da.to_dataset(name=var_name)


def sum_datasets(model_data, keys, out_name):
    """Sum a list of variables (Datasets) into a new Dataset, skipping missing ones."""
    dsets = [model_data[k] for k in keys if model_data.get(k) is not None]
    if not dsets:
        return None
    arrays = [d[list(d.data_vars)[0]] for d in dsets]
    total = arrays[0].copy()
    for a in arrays[1:]:
        total = total + a
    return total.to_dataset(name=out_name)


def normalize_precipitation_units(precip_ds):
    """Convert model precipitation to mm day⁻¹.

    Handles kg/m²/s, g/m²/s, and the mis-typed 'km m-2 s-1' seen in some files.
    """
    if precip_ds is None:
        return None
    var_name = list(precip_ds.data_vars)[0]
    da = precip_ds[var_name]
    units = da.attrs.get('units', 'kg m-2 s-1').lower().replace(' ', '')
    if units in ('kgm-2s-1', 'kgm^-2s^-1', 'kg/m2/s', 'kgm-2s-1'):
        factor = 86400.0
    elif units in ('gm-2s-1', 'gm^-2s^-1', 'g/m2/s', 'gm-2s-1'):
        factor = 86.4
    elif units in ('mmday-1', 'mm/day', 'mmday^-1', 'mmd-1'):
        factor = 1.0
    elif units in ('ms-1', 'm/s', 'm s-1'):
        factor = 86400.0 * 1000.0
    else:
        # 'km m-2 s-1' is treated as a typo for kg m⁻² s⁻¹
        factor = 86400.0
    da = da * factor
    try:
        global_mean = float(da.mean().values)
    except Exception:
        global_mean = np.nan
    if not np.isnan(global_mean) and global_mean > 100.0:
        da = da / 1000.0
    da.attrs['units'] = 'mm day-1'
    return da.to_dataset(name=var_name)


# Align with MEC species set in aerocom_data (no gaseous loadso2).
LOAD_VARS = ['loadbc', 'loaddust', 'loadoa', 'loadso4', 'loadss']
EMI_VARS = ['emibc', 'emidust', 'emioa', 'emiso2', 'emiss']

# Load variables relevant for the BC+OA optical properties (computed for completeness).
BCOA_LOAD_VARS = ['loadbc', 'loadoa']
BCOA_EMI_VARS = ['emibc', 'emioa']

data = {}
for m in models:
    normalized = {}
    for var in VARIABLES:
        if raw_data[m].get(var) is None:
            continue
        try:
            normalized[var] = normalize_dataset_time(raw_data[m][var], var_hint=f'{m}/{var}')
        except Exception as e:
            print(f'  Failed to normalise {m}/{var}: {e}')
            normalized[var] = None
    normalized['load_total'] = sum_datasets(normalized, LOAD_VARS, 'load_total')
    normalized['emi_total'] = sum_datasets(normalized, EMI_VARS, 'emi_total')
    normalized['load_BC_OA'] = sum_datasets(normalized, BCOA_LOAD_VARS, 'load_BC_OA')
    normalized['emi_BC_OA'] = sum_datasets(normalized, BCOA_EMI_VARS, 'emi_BC_OA')
    if normalized.get('precip') is not None:
        normalized['precip'] = normalize_precipitation_units(normalized['precip'])
    data[m] = normalized

# -------------------------------------------------------------------------------
# 3. Calculate derived variables (MEC, lifetime, AE, MAC, SSA)
# -------------------------------------------------------------------------------
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
        print(f'  Lifetime filter: {model_name} {var_name} excluded {n_excluded:,} / {da.size:,} values '
              f'({n_excluded / da.size * 100:.2f}%) outside [{min_days}, {max_days}] days')
    return da.where(~bad), {'n_total': int(da.size), 'n_excluded': n_excluded}


derived = {m: {} for m in models}
lifetime_filter_stats = []

for m in models:
    # MEC from the built-in calculator is in m² g⁻¹; convert to m² kg⁻¹ for AOD budgets.
    mec_ds = aerocom_data.calculate_derived_var(data[m], m, 'MEC')
    if mec_ds is not None:
        var_name = list(mec_ds.data_vars)[0]
        derived[m]['MEC'] = mec_ds[var_name] * 1000.0
    else:
        derived[m]['MEC'] = None

    # MAC and SSA are retained for completeness / comparison with AAOD work.
    for dv in ['MAC', 'SSA']:
        ds = aerocom_data.calculate_derived_var(data[m], m, dv)
        derived[m][dv] = None if ds is None else ds[list(ds.data_vars)[0]]

    # AE uses the 440/870 nm logic already encoded in aerocom_data.
    ae_ds = aerocom_data.calculate_derived_var(data[m], m, 'AE')
    derived[m]['AE'] = None if ae_ds is None else ae_ds[list(ae_ds.data_vars)[0]]

    # Total-aerosol lifetime at the grid-cell level.
    if data[m].get('load_total') is not None and data[m].get('emi_total') is not None:
        load_da = data[m]['load_total']['load_total']
        emi_da = data[m]['emi_total']['emi_total']
        lt = load_da / (emi_da * 3600 * 24)  # days
        lt, stats = filter_lifetime(lt, m, 'lifetime', LIFETIME_MAX_DAYS, LIFETIME_MIN_DAYS)
        derived[m]['lifetime'] = lt
        lifetime_filter_stats.append({'model': m, 'var': 'lifetime', **stats})
    else:
        derived[m]['lifetime'] = None

    # BC+OA lifetime (for comparison with the AAOD notebook).
    if data[m].get('load_BC_OA') is not None and data[m].get('emi_BC_OA') is not None:
        load_da = data[m]['load_BC_OA']['load_BC_OA']
        emi_da = data[m]['emi_BC_OA']['emi_BC_OA']
        lt = load_da / (emi_da * 3600 * 24)
        lt, stats = filter_lifetime(lt, m, 'lifetime_BC_OA', LIFETIME_MAX_DAYS, LIFETIME_MIN_DAYS)
        derived[m]['lifetime_BC_OA'] = lt
        lifetime_filter_stats.append({'model': m, 'var': 'lifetime_BC_OA', **stats})
    else:
        derived[m]['lifetime_BC_OA'] = None

if lifetime_filter_stats:
    total_excluded = sum(s['n_excluded'] for s in lifetime_filter_stats)
    total_cells = sum(s['n_total'] for s in lifetime_filter_stats)
    print(f'\nLifetime filter summary: excluded {total_excluded:,} / {total_cells:,} '
          f'grid-cell values ({total_excluded / total_cells * 100:.2f}%)')

# Flatten base variables and derived variables to DataArrays for aggregation.
data_derived = {}
for m in models:
    data_derived[m] = {}
    for k, v in data[m].items():
        if v is not None:
            data_derived[m][k] = v[list(v.data_vars)[0]]
    for k, v in derived[m].items():
        if v is not None:
            data_derived[m][k] = v

print('\nDerived-variable availability (pre-aggregation):')
for dv in ['MEC', 'AE', 'lifetime', 'lifetime_BC_OA', 'MAC', 'SSA']:
    n = sum(1 for m in derived if derived[m].get(dv) is not None)
    print(f'  {dv:15s}: {n:2d} / {len(models)} models')

# Variables available for aggregation.
variables_to_aggregate = [
    'MEC', 'AE', 'lifetime', 'lifetime_BC_OA', 'MAC', 'SSA',
    'load_total', 'emi_total', 'load_BC_OA', 'emi_BC_OA',
    'emibc', 'emioa', 'precip', 'od550aer',
]
LIFETIME_VARS = {'lifetime', 'lifetime_BC_OA'}

# Source regions for main Fig. 4 decomposition; outflow is a meta-model target.
SOURCE_REGIONS = ['africa', 'amazon', 'se_asia', 'boreal_na', 'eastern_siberia']
OUTFLOW_REGION = 'outflow_af'
OUTFLOW_SOURCE = 'africa'

# -------------------------------------------------------------------------------
# 4. Regional masks and monthly/seasonal aggregation
# -------------------------------------------------------------------------------
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
        'surface_type': 'ocean', 'lon_range': (350, 8), 'lat_range': (-15, 3),
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
}

sample_model = next((m for m in models if data[m].get('od550aer') is not None), None)
template = data[sample_model]['od550aer'].isel(time=0)
print(f'Template grid from {sample_model}: {template.dims}')

SURFACE_TYPE = None
masks = {}
for name, cfg in REGIONS.items():
    masks[name] = ct.create_region_mask(
        template, name=name,
        lon_range=cfg['lon_range'], lat_range=cfg['lat_range'],
        surface_type=SURFACE_TYPE if SURFACE_TYPE is not None else cfg.get('surface_type', 'all'),
        mask_registry=masks,
    )
print('Regions created:', list(masks.keys()))


def aggregate_region(model_dict, var_name, region_name, return_time_series=False, skipna=False):
    """Spatially aggregate `var_name` for every model."""
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
                return_time_series=return_time_series,
                skipna=skipna,
            )
            result[model] = val
        except Exception as e:
            print(f'  Aggregation failed for {var_name} {model} {region_name}: {e}')
    return result


model_monthly = {region: {var: aggregate_region(data_derived, var, region, return_time_series=True,
                                                  skipna=(var in LIFETIME_VARS))
                          for var in variables_to_aggregate}
                 for region in REGIONS}

model_seasonal = {region: {var: aggregate_region(data_derived, var, region, return_time_series=False,
                                                   skipna=(var in LIFETIME_VARS))
                           for var in variables_to_aggregate}
                  for region in REGIONS}


def compute_derived_after_aggregation(monthly_dict, seasonal_dict):
    """Paper Methods: τ = ⟨load⟩/⟨E⟩ and MEC = ⟨AOD⟩/⟨load⟩ after regional aggregation."""
    def _safe_div(num, den, scale=1.0):
        try:
            return num / (den * scale)
        except Exception:
            return None

    for agg in (monthly_dict, seasonal_dict):
        for region in REGIONS:
            if 'lifetime' in DERIVED_VAR_AFTER_AGG:
                load = agg[region].get('load_total', {})
                emi = agg[region].get('emi_total', {})
                out = {}
                for model in set(load) & set(emi):
                    val = _safe_div(load[model], emi[model], 86400.0)
                    if val is not None:
                        out[model] = val
                if out:
                    agg[region]['lifetime'] = out
            if 'lifetime_BC_OA' in DERIVED_VAR_AFTER_AGG or 'lifetime' in DERIVED_VAR_AFTER_AGG:
                load = agg[region].get('load_BC_OA', {})
                emi = agg[region].get('emi_BC_OA', {})
                out = {}
                for model in set(load) & set(emi):
                    val = _safe_div(load[model], emi[model], 86400.0)
                    if val is not None:
                        out[model] = val
                if out:
                    agg[region]['lifetime_BC_OA'] = out
            if 'MEC' in DERIVED_VAR_AFTER_AGG:
                aod = agg[region].get('od550aer', {})
                load = agg[region].get('load_total', {})
                out = {}
                for model in set(aod) & set(load):
                    val = _safe_div(aod[model], load[model], 1.0)
                    if val is not None:
                        out[model] = val
                if out:
                    agg[region]['MEC'] = out
    print(f'\nPost-aggregation derived vars applied: {DERIVED_VAR_AFTER_AGG}')


compute_derived_after_aggregation(model_monthly, model_seasonal)


def apply_amazon_soa_aod(monthly_dict, seasonal_dict):
    """Inflate Amazon OA emissions so SOA is AMAZON_SOA_FRACTION of total OA.

    Approximates Zhong et al. (2022/2023): model ``emioa`` is treated as POA.
    Updates ``emi_total`` and ``emi_BC_OA`` and recomputes Amazon lifetimes.
    """
    if not INCLUDE_AMAZON_SOA:
        print('\nAmazon SOA: SKIPPED')
        return
    f = AMAZON_SOA_FRACTION
    print('\n--- Amazon SOA adjustment ---')
    print(f'  emi_OA_total = emioa / (1 - {f:.2f}); added to emi_total / emi_BC_OA')
    for agg in (monthly_dict, seasonal_dict):
        region = 'amazon'
        emibc = agg[region].get('emibc', {})
        emioa = agg[region].get('emioa', {})
        emi_total = dict(agg[region].get('emi_total', {}))
        emi_bcoa = dict(agg[region].get('emi_BC_OA', {}))
        n = 0
        for model in set(emibc) & set(emioa):
            try:
                bc = emibc[model]
                oa = emioa[model]
                oa_total = oa / (1.0 - f)
                # Replace primary OA contribution with total OA (POA+SOA)
                if model in emi_total:
                    emi_total[model] = emi_total[model] - oa + oa_total
                if model in emi_bcoa:
                    emi_bcoa[model] = bc + oa_total
                n += 1
            except Exception as e:
                print(f'  SOA adjust failed for {model}: {e}')
        agg[region]['emi_total'] = emi_total
        agg[region]['emi_BC_OA'] = emi_bcoa
        # Recompute lifetimes with adjusted emissions
        for lt_name, load_name, emi_name in [
            ('lifetime', 'load_total', 'emi_total'),
            ('lifetime_BC_OA', 'load_BC_OA', 'emi_BC_OA'),
        ]:
            load = agg[region].get(load_name, {})
            emi = agg[region].get(emi_name, {})
            lt = {}
            for model in set(load) & set(emi):
                try:
                    lt[model] = load[model] / (emi[model] * 3600 * 24)
                except Exception:
                    continue
            if lt:
                agg[region][lt_name] = {**agg[region].get(lt_name, {}), **lt}
        print(f'  Adjusted Amazon emissions for {n} models')


apply_amazon_soa_aod(model_monthly, model_seasonal)

print('\nModel seasonal availability:')
for region in REGIONS:
    for var in ['MEC', 'AE', 'lifetime', 'precip', 'od550aer']:
        n = len(model_seasonal[region].get(var, {}))
        print(f'  {region:15s} {var}: {n} models')

# -------------------------------------------------------------------------------
# 5. Load POLDER observations and compute monthly regional means
# -------------------------------------------------------------------------------
polder_path = project_root / 'Data' / 'AP3_POLDER_Collocated' / 'POLDER_GRASP_coloc_3h_AP3_2010_lon0_10.0_lat0_10.0.parquet'
print(f'\nLoading POLDER observations: {polder_path}')

polder_cols = ['time', 'longitude', 'latitude', 'AOD_550', 'AAOD_550', 'AOD_440', 'AOD_870']
polder_df = pd.read_parquet(polder_path, columns=polder_cols)
print(f'POLDER rows: {len(polder_df)}')

polder_df['time'] = pd.to_datetime(polder_df['time'])
polder_df['month'] = polder_df['time'].dt.to_period('M')

with np.errstate(divide='ignore', invalid='ignore'):
    polder_df['AE'] = -np.log(polder_df['AOD_550'] / polder_df['AOD_440']) / np.log(550.0 / 440.0)
    polder_df['SSA'] = 1.0 - polder_df['AAOD_550'] / polder_df['AOD_550']

polder_df = polder_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=['AOD_550', 'AE', 'SSA', 'AAOD_550']
)
print(f'POLDER rows after filtering: {len(polder_df)}')


def polder_region_mask(df, lon_range, lat_range):
    """Boolean mask for POLDER lat/lon region (0-360 longitude)."""
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    if lon_min > lon_max:
        mask_lon = (df['longitude'] >= lon_min) | (df['longitude'] <= lon_max)
    else:
        mask_lon = (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)
    mask_lat = (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)
    return mask_lon & mask_lat


def polder_monthly_means(df, region_name):
    """Area-weighted monthly regional means from POLDER 3-hourly data."""
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
            'AOD_550': np.average(g['AOD_550'], weights=w),
            'AAOD_550': np.average(g['AAOD_550'], weights=w),
            'AE': np.average(g['AE'], weights=w),
            'SSA': np.average(g['SSA'], weights=w),
            'n': len(g),
        })
    return pd.DataFrame(rows)


polder_monthly = {region: polder_monthly_means(polder_df, region) for region in REGIONS}
print('\nPOLDER monthly means per region:')
for region, df in polder_monthly.items():
    print(f'  {region:15s}: {len(df)} months')


def sample_model_aod_at_polder(model_name, region_name):
    """Seasonal-mean model AOD sampled at POLDER observation locations."""
    cfg = REGIONS[region_name]
    if data[model_name].get('od550aer') is None:
        return np.nan
    da = data[model_name]['od550aer']['od550aer']
    da = da.sel(time=slice(*cfg['time_slice']))
    mask = polder_region_mask(polder_df, cfg['lon_range'], cfg['lat_range'])
    sub = polder_df[mask].copy()
    t0, t1 = cfg['time_slice']
    sub = sub[(sub['time'] >= t0) & (sub['time'] <= t1)]
    if sub.empty or 'time' not in da.dims:
        return np.nan
    try:
        # Nearest neighbour in space for each observation, then mean over time-matched samples
        pts = []
        for _, row in sub.sample(n=min(len(sub), 5000), random_state=0).iterrows():
            try:
                val = float(da.sel(lon=row['longitude'], lat=row['latitude'],
                                   time=row['time'], method='nearest').values)
                if np.isfinite(val):
                    pts.append(val)
            except Exception:
                continue
        return float(np.mean(pts)) if pts else np.nan
    except Exception as e:
        print(f'  POLDER sample failed {model_name}/{region_name}: {e}')
        return np.nan


def homogenize_polder_aod(model_seasonal_dict, regions, var='AOD_550'):
    """Zhong-style homogenization: regional_obs = (sampled_obs - b) / a.

    Applies to AOD (and optionally AE via var='AE'). Excludes MONTHLY_OUTPUT_MODELS
    from the sampled↔regional regression. Only intended for SOURCE_REGIONS.
    """
    print(f'\n--- POLDER {var} homogenization ---')
    homogenized = {}
    for region in regions:
        if polder_monthly[region].empty:
            homogenized[region] = np.nan
            continue
        obs_col = 'AOD_550' if var == 'AOD_550' else var
        if obs_col not in polder_monthly[region].columns:
            homogenized[region] = np.nan
            continue
        obs_sampled = float(polder_monthly[region][obs_col].mean())
        regional_key = 'od550aer' if var == 'AOD_550' else 'AE'
        regional = model_seasonal_dict[region].get(regional_key, {})
        sampled = {}
        for model in regional:
            if model in MONTHLY_OUTPUT_MODELS:
                continue
            if var == 'AOD_550':
                s = sample_model_aod_at_polder(model, region)
            else:
                # Use regional AE (homogenization of AE uses model regional AE vs obs AE)
                s = float(regional[model]) if regional[model] is not None else np.nan
            if np.isfinite(s):
                sampled[model] = s
        if var == 'AE':
            # For AE: x = model regional AE, y = same (identity); use obs directly when
            # we lack collocated model AE sampling. Fall back to raw sampled AE.
            homogenized[region] = obs_sampled
            print(f'  {region}: AE homogenization uses sampled obs AE={obs_sampled:.3f} '
                  f'(collocated AE sampling not available; raw AE retained)')
            continue
        common = [m for m in regional if m in sampled and np.isfinite(float(regional[m]))]
        if len(common) < 3:
            print(f'  {region}: homogenization skipped (n={len(common)}); using raw sampled mean')
            homogenized[region] = obs_sampled
            continue
        x = np.array([float(regional[m]) for m in common])
        y = np.array([sampled[m] for m in common])
        X = np.column_stack([x, np.ones(len(x))])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
        y_pred = a * x + b
        denom = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum((y - y_pred) ** 2) / denom if denom > 0 else np.nan
        homogenized_regional = (obs_sampled - b) / a if a != 0 else np.nan
        homogenized[region] = homogenized_regional
        print(f'  {region}: sampled = {a:.4f}*regional + {b:.4f} (R²={r2:.3f}, n={len(common)})')
        print(f'    raw sampled {var}={obs_sampled:.4f} -> homogenized={homogenized_regional:.4f}')
    return homogenized


# Homogenize source regions only (paper SM Method 1); outflow keeps sampled mean.
_homo_regions = [r for r in SOURCE_REGIONS if r in REGIONS]
if POLDER_HOMOGENIZE:
    aod_obs_by_region = homogenize_polder_aod(model_seasonal, _homo_regions, var='AOD_550')
    ae_obs_by_region = homogenize_polder_aod(model_seasonal, _homo_regions, var='AE')
else:
    print('\n--- POLDER homogenization DISABLED ---')
    aod_obs_by_region = {
        region: float(polder_monthly[region]['AOD_550'].mean())
        if not polder_monthly[region].empty else np.nan
        for region in _homo_regions
    }
    ae_obs_by_region = {
        region: float(polder_monthly[region]['AE'].mean())
        if not polder_monthly[region].empty else np.nan
        for region in _homo_regions
    }
    for region, val in aod_obs_by_region.items():
        print(f'  {region}: AOD_obs={val:.4f}')

# Outflow: always use raw sampled POLDER mean (not homogenized).
if OUTFLOW_REGION in REGIONS and not polder_monthly[OUTFLOW_REGION].empty:
    aod_obs_by_region[OUTFLOW_REGION] = float(polder_monthly[OUTFLOW_REGION]['AOD_550'].mean())
    ae_obs_by_region[OUTFLOW_REGION] = float(polder_monthly[OUTFLOW_REGION]['AE'].mean())

# -------------------------------------------------------------------------------
# 6. Load GPCP precipitation and compute monthly regional means
# -------------------------------------------------------------------------------
gpcp_path = project_root / 'Data' / 'Prec' / 'GPCP_2010_0-360.nc'
print(f'\nLoading GPCP precipitation: {gpcp_path}')
gpcp = xr.open_dataset(gpcp_path)
gpcp_precip = gpcp['sat_gauge_precip']
print(f'GPCP dims: {gpcp_precip.dims}, shape: {gpcp_precip.shape}')


def gpcp_region_mean(precip_da, region_name, apply_surface_mask=True):
    """Area-weighted regional mean of GPCP precipitation (mm day⁻¹).

    For land/ocean BB regions, apply the same surface mask as the models so
    maritime grid cells do not inflate SE Asia / Siberia precip.
    """
    cfg = REGIONS[region_name]
    lon_min, lon_max = cfg['lon_range']
    lat_min, lat_max = cfg['lat_range']
    if lon_min > lon_max:
        sub = xr.concat([
            precip_da.sel(lon=slice(lon_min, 360)),
            precip_da.sel(lon=slice(0, lon_max)),
        ], dim='lon')
    else:
        sub = precip_da.sel(lon=slice(lon_min, lon_max))
    if sub.lat.values[0] > sub.lat.values[-1]:
        sub = sub.sel(lat=slice(lat_max, lat_min))
    else:
        sub = sub.sel(lat=slice(lat_min, lat_max))
    sub = sub.sel(time=slice(*cfg['time_slice']))
    if apply_surface_mask and cfg.get('surface_type') in ('land', 'ocean') and region_name in masks:
        try:
            mask_on = masks[region_name].interp(lat=sub.lat, lon=sub.lon, method='nearest')
            sub = sub.where(mask_on > 0)
        except Exception as e:
            print(f'  GPCP surface mask skip {region_name}: {e}')
    coslat = np.cos(np.deg2rad(sub.lat))
    weights = coslat * xr.ones_like(sub.isel(time=0))
    weights = weights.where(np.isfinite(sub.isel(time=0)))
    wsum = weights.sum()
    if float(wsum) == 0:
        return sub.mean(dim=['lat', 'lon'])
    weights = weights / wsum
    return (sub * weights).sum(dim=['lat', 'lon'])


gpcp_region = {region: gpcp_region_mean(gpcp_precip, region) for region in REGIONS}
print('\nGPCP regional mean precipitation (mm/day):')
for region, da in gpcp_region.items():
    mean_val = float(da.mean().values) if da.size else np.nan
    print(f'  {region:15s}: {mean_val:.3f}')

# -------------------------------------------------------------------------------
# 7. Model regressions (MEC vs AE; 1/τ vs precipitation + AE)
# -------------------------------------------------------------------------------
print('\n--- Model regressions ---')

regression_rows = []
regression_excluded = []
for region in REGIONS:
    if region == 'global':
        continue
    for model in models:
        mec = model_seasonal[region]['MEC'].get(model)
        ae = model_seasonal[region]['AE'].get(model)
        lt = model_seasonal[region]['lifetime'].get(model)
        precip = model_seasonal[region]['precip'].get(model)
        aod = model_seasonal[region]['od550aer'].get(model)
        if not all(v is not None for v in [mec, ae, lt, precip, aod]):
            continue
        if not np.isfinite(float(lt)):
            regression_excluded.append({'region': region, 'model': model, 'reason': 'non-finite lifetime'})
            continue
        regression_rows.append({
            'region': region, 'model': model,
            'MEC': float(mec), 'AE': float(ae),
            'lifetime': float(lt), 'inv_lifetime': 1.0 / float(lt),
            'precip': float(precip), 'AOD': float(aod),
        })
if regression_excluded:
    print(f'  Excluded {len(regression_excluded)} points due to non-finite lifetime')
reg_df = pd.DataFrame(regression_rows)
print(f'Regression data rows (seasonal means): {len(reg_df)}')

# MEC vs AE: linear regression (intercept controlled by INTERCEPT_0).
mec_ae_results = {}
for region in REGIONS:
    if region == 'global':
        continue
    sub = reg_df[(reg_df['region'] == region)].dropna(subset=['MEC', 'AE'])
    if len(sub) < 3:
        print(f'{region}: skipped MEC vs AE regression (only {len(sub)} points)')
        continue
    x = sub['AE'].values
    y = sub['MEC'].values
    if INTERCEPT_0:
        slope = np.nansum(x * y) / np.nansum(x ** 2)
        inter = 0.0
        residuals = y - slope * x
        r2 = 1 - np.nansum(residuals ** 2) / np.nansum(y ** 2)
        x_sampled = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        # Approximate CI using residual std
        se = np.nanstd(residuals)
        ci_lower = slope * x_sampled - 1.96 * se
        ci_upper = slope * x_sampled + 1.96 * se
        fit = {'slope': slope, 'inter': inter, 'r2': r2, 'x_sampled': x_sampled,
               'ci_lower': ci_lower, 'ci_upper': ci_upper}
        print(f"{region}: MEC = {slope:.4f}*AE  (through origin; R²={r2:.3f}, n={len(sub)})")
    else:
        fit = functions.fit_data(x, y)
        print(f"{region}: MEC = {fit['slope']:.4f}*AE + {fit['inter']:.4f} "
              f"(R²={fit['r2']:.3f}, n={len(sub)})")
    mec_ae_results[region] = fit

# 1/lifetime vs precipitation + AE: multiple linear regression.
inv_lt_results = {}
for region in REGIONS:
    if region == 'global':
        continue
    sub = reg_df[(reg_df['region'] == region)].dropna(subset=['inv_lifetime', 'precip', 'AE'])
    if len(sub) < 4:
        print(f'{region}: skipped 1/tau regression (only {len(sub)} points)')
        continue
    X = np.column_stack([sub['precip'].values, sub['AE'].values])
    y = sub['inv_lifetime'].values
    X_ols = np.column_stack([np.ones(len(X)), X])
    coeffs, _, _, _ = np.linalg.lstsq(X_ols, y, rcond=None)
    y_pred = X_ols @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    inv_lt_results[region] = {
        'intercept': coeffs[0], 'alpha_pr': coeffs[1], 'beta_ae': coeffs[2],
        'r2': r2, 'n': len(sub)
    }
    print(f"{region}: 1/τ = {coeffs[0]:.4f} + {coeffs[1]:.4f}*Pr + {coeffs[2]:.4f}*AE "
          f"(R²={r2:.3f}, n={len(sub)})")

# Simple 1/tau vs precipitation regression for plotting (Figure 2b).
inv_lt_simple = {}
for region in REGIONS:
    if region == 'global':
        continue
    sub = reg_df[(reg_df['region'] == region)].dropna(subset=['inv_lifetime', 'precip'])
    if len(sub) < 3:
        continue
    fit = functions.fit_data(sub['precip'].values, sub['inv_lifetime'].values)
    inv_lt_simple[region] = fit

# -------------------------------------------------------------------------------
# 8. Constrained estimates from observations
# -------------------------------------------------------------------------------
print('\n--- Constrained estimates from observations ---')

constrained = []
for region in REGIONS:
    if region == 'global' or region == OUTFLOW_REGION:
        continue
    polder_df_region = polder_monthly[region]
    if polder_df_region.empty:
        print(f'  {region}: skipped (no POLDER data)')
        continue
    if region not in mec_ae_results or region not in inv_lt_results:
        print(f'  {region}: skipped (no model regression)')
        continue

    ae_obs_mean = float(ae_obs_by_region.get(
        region,
        polder_df_region['AE'].mean() if not polder_df_region.empty else np.nan
    ))
    aod_obs_mean = float(aod_obs_by_region.get(
        region, polder_df_region['AOD_550'].mean()
    ))
    precip_obs_mean = float(gpcp_region[region].mean().values) if gpcp_region[region].size else np.nan

    mec_params = mec_ae_results[region]
    mec_c = mec_params['slope'] * ae_obs_mean + mec_params['inter']

    lt_params = inv_lt_results[region]
    inv_lt_c = lt_params['intercept'] + lt_params['alpha_pr'] * precip_obs_mean + lt_params['beta_ae'] * ae_obs_mean
    tau_c_days = 1.0 / inv_lt_c if inv_lt_c > 0 else np.nan

    # E_c in kg m⁻² s⁻¹
    e_c = aod_obs_mean / (tau_c_days * 86400.0 * mec_c) if (tau_c_days > 0 and mec_c > 0) else np.nan

    constrained.append({
        'region': region,
        'AE_obs': ae_obs_mean,
        'AOD_obs': aod_obs_mean, 'precip_obs': precip_obs_mean,
        'MEC_c': mec_c, 'tau_c_days': tau_c_days, 'E_c': e_c,
        'AOD_homogenized': bool(POLDER_HOMOGENIZE),
        'MEC_intercept_used': not INTERCEPT_0,
    })
    print(f'  {region}: MEC_c={mec_c:.3f}, tau_c={tau_c_days:.2f} d, '
          f'E_c={e_c:.3e}, AOD_obs={aod_obs_mean:.4f} (homogenized={POLDER_HOMOGENIZE})')

constrained_df = pd.DataFrame(constrained)
print(f'Constrained rows: {len(constrained_df)}')
print(constrained_df.set_index('region')[['MEC_c', 'tau_c_days', 'E_c']])

# -------------------------------------------------------------------------------
# 9. Error decomposition
# -------------------------------------------------------------------------------
print('\n--- AOD error decomposition (source regions; outflow handled separately) ---')

decomp_rows = []
for region in SOURCE_REGIONS:
    if region not in REGIONS:
        continue
    csub = constrained_df[constrained_df['region'] == region]
    if csub.empty:
        print(f'  {region}: skipped (no constrained estimate)')
        continue
    mec_c = csub['MEC_c'].mean()
    tau_c_days = csub['tau_c_days'].mean()
    aod_c = csub['AOD_obs'].mean()
    e_c = csub['E_c'].mean()

    for model in models:
        mec_model = model_seasonal[region]['MEC'].get(model)
        lt_model = model_seasonal[region]['lifetime'].get(model)
        emi_model = model_seasonal[region]['emi_total'].get(model)
        aod_model = model_seasonal[region]['od550aer'].get(model)

        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [mec_model, lt_model, emi_model, aod_model]):
            continue
        if any(np.isnan(v) for v in [mec_c, tau_c_days, aod_c, e_c]):
            continue

        mec_model = float(mec_model)
        lt_model = float(lt_model)
        emi_model = float(emi_model)
        aod_model = float(aod_model)

        dAOD_total = aod_model - aod_c
        dAOD_E = (emi_model - e_c) * (tau_c_days * 86400.0) * mec_c
        dAOD_tau = e_c * ((lt_model - tau_c_days) * 86400.0) * mec_c
        dAOD_MEC = e_c * (tau_c_days * 86400.0) * (mec_model - mec_c)
        cross = dAOD_total - (dAOD_E + dAOD_tau + dAOD_MEC)

        denom = abs(dAOD_E) + abs(dAOD_tau) + abs(dAOD_MEC) + abs(cross)
        if denom > 0:
            pct_E = abs(dAOD_E) / denom * 100
            pct_tau = abs(dAOD_tau) / denom * 100
            pct_MEC = abs(dAOD_MEC) / denom * 100
            pct_cross = abs(cross) / denom * 100
        else:
            pct_E = pct_tau = pct_MEC = pct_cross = np.nan

        decomp_rows.append({
            'region': region, 'model': model,
            'MEC_model': mec_model, 'MEC_c': mec_c,
            'tau_model': lt_model, 'tau_c_days': tau_c_days,
            'E_model': emi_model, 'E_c': e_c,
            'AOD_model': aod_model, 'AOD_c': aod_c,
            'dAOD_E': dAOD_E, 'dAOD_tau': dAOD_tau,
            'dAOD_MEC': dAOD_MEC, 'dAOD_cross': cross,
            'dAOD_total': dAOD_total,
            'pct_E': pct_E, 'pct_tau': pct_tau,
            'pct_MEC': pct_MEC, 'pct_cross': pct_cross,
        })

decomp_df = pd.DataFrame(decomp_rows)
print(f'Decomposition rows: {len(decomp_df)}')
if not decomp_df.empty:
    print(decomp_df.groupby('region')[['pct_E', 'pct_tau', 'pct_MEC', 'pct_cross']].mean())


# -------------------------------------------------------------------------------
# 9b. African outflow meta-model (Fig. 5 / Fig. 6 style)
# -------------------------------------------------------------------------------
print('\n--- African outflow meta-model (prediction target) ---')
print('  Form: AOD_out = a*(E*τ*MEC) + b*(E*τ) + c*MEC + d  (source = africa)')

meta_rows = []
for model in models:
    aod_out = model_seasonal[OUTFLOW_REGION].get('od550aer', {}).get(model)
    e_src = model_seasonal[OUTFLOW_SOURCE].get('emi_total', {}).get(model)
    tau_src = model_seasonal[OUTFLOW_SOURCE].get('lifetime', {}).get(model)
    mec_src = model_seasonal[OUTFLOW_SOURCE].get('MEC', {}).get(model)
    if not all(v is not None and np.isfinite(float(v)) for v in [aod_out, e_src, tau_src, mec_src]):
        continue
    e_src, tau_src, mec_src = float(e_src), float(tau_src), float(mec_src)
    # Convert E*τ to burden units consistent with MEC (kg m^-2 * m^2 kg^-1)
    burden_term = e_src * (tau_src * 86400.0)  # kg m^-2
    etm = burden_term * mec_src
    meta_rows.append({
        'model': model,
        'AOD_out': float(aod_out),
        'E_tau_MEC': etm,
        'E_tau': burden_term,
        'MEC': mec_src,
        'E': e_src,
        'tau': tau_src,
    })
meta_df = pd.DataFrame(meta_rows)
outflow_meta_params = None
outflow_pred_df = pd.DataFrame()

if len(meta_df) < 5:
    print(f'  Insufficient models for outflow meta-model (n={len(meta_df)})')
else:
    X = np.column_stack([
        meta_df['E_tau_MEC'].values,
        meta_df['E_tau'].values,
        meta_df['MEC'].values,
        np.ones(len(meta_df)),
    ])
    y = meta_df['AOD_out'].values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    nmb = 100.0 * np.mean(y_pred - y) / np.mean(y) if np.mean(y) != 0 else np.nan
    rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    outflow_meta_params = {
        'a': float(coeffs[0]), 'b': float(coeffs[1]),
        'c': float(coeffs[2]), 'd': float(coeffs[3]),
        'r2': r2, 'nmb': nmb, 'rmse': rmse, 'n': len(meta_df),
    }
    print(f"  Fit: a={coeffs[0]:.4g}, b={coeffs[1]:.4g}, c={coeffs[2]:.4g}, d={coeffs[3]:.4g}")
    print(f"  R²={r2:.3f}, NMB={nmb:.1f}%, RMSE={rmse:.4f}, n={len(meta_df)}")

    c_src = constrained_df[constrained_df['region'] == OUTFLOW_SOURCE]
    aod_out_obs = float(aod_obs_by_region.get(
        OUTFLOW_REGION,
        polder_monthly[OUTFLOW_REGION]['AOD_550'].mean()
        if not polder_monthly[OUTFLOW_REGION].empty else np.nan
    ))
    pred_rows = []
    if not c_src.empty:
        e_c = float(c_src['E_c'].mean())
        tau_c = float(c_src['tau_c_days'].mean())
        mec_c = float(c_src['MEC_c'].mean())
        burden_c = e_c * tau_c * 86400.0
        aod_mfc = (coeffs[0] * burden_c * mec_c
                   + coeffs[1] * burden_c
                   + coeffs[2] * mec_c
                   + coeffs[3])
        for _, row in meta_df.iterrows():
            # EC: scale emission so source AOD matches observation
            aod_src_model = model_seasonal[OUTFLOW_SOURCE]['od550aer'].get(row['model'])
            aod_src_obs = float(aod_obs_by_region.get(OUTFLOW_SOURCE, np.nan))
            if aod_src_model is None or not np.isfinite(aod_src_obs) or float(aod_src_model) == 0:
                e_ec = row['E']
            else:
                e_ec = row['E'] * (aod_src_obs / float(aod_src_model))
            burden_ec = e_ec * row['tau'] * 86400.0
            aod_ec = (coeffs[0] * burden_ec * row['MEC']
                      + coeffs[1] * burden_ec
                      + coeffs[2] * row['MEC']
                      + coeffs[3])
            pred_rows.append({
                'model': row['model'],
                'AOD_default': row['AOD_out'],
                'AOD_meta_fit': float(
                    coeffs[0] * row['E_tau_MEC'] + coeffs[1] * row['E_tau']
                    + coeffs[2] * row['MEC'] + coeffs[3]
                ),
                'AOD_EC': float(aod_ec),
                'AOD_MFC': float(aod_mfc),
                'AOD_obs': aod_out_obs,
            })
        outflow_pred_df = pd.DataFrame(pred_rows).sort_values('model').reset_index(drop=True)
        print('\n  Outflow prediction summary (Default / EC / MFC / POLDER):')
        print(outflow_pred_df[['AOD_default', 'AOD_EC', 'AOD_MFC', 'AOD_obs']].describe())

# -------------------------------------------------------------------------------
# 10. Figure 2: linear regressions (SI Fig. 1 / main Fig. 2 style)
# -------------------------------------------------------------------------------
print('\n--- Figure 2: regressions ---')

plot_regions = [r for r in REGIONS if r != 'global']
colors = plt.cm.tab10(np.linspace(0, 1, len(plot_regions)))
region_colors = {r: colors[i] for i, r in enumerate(plot_regions)}

# Dedicated SI Fig. 1 / main Fig. 2 style panel: all regions, one colour each.
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for region in plot_regions:
    if region not in mec_ae_results:
        continue
    sub = reg_df[reg_df['region'] == region].dropna(subset=['MEC', 'AE'])
    if sub.empty:
        continue
    fit = mec_ae_results[region]
    x_line = fit['x_sampled']
    ax.scatter(sub['AE'], sub['MEC'], s=40, alpha=0.7, color=region_colors[region],
               label=region, edgecolors='none')
    ax.plot(x_line, fit['slope'] * x_line + fit['inter'], '-', color=region_colors[region], lw=2)
    ax.fill_between(x_line, fit['ci_lower'], fit['ci_upper'], color=region_colors[region], alpha=0.15)
ax.set_xlabel('Ångström Exponent', fontweight='bold')
ax.set_ylabel('MEC (m² kg⁻¹)', fontweight='bold')
ax.set_title('(a) MEC vs AE', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='best')

ax = axes[1]
for region in plot_regions:
    if region not in inv_lt_simple:
        continue
    sub = reg_df[reg_df['region'] == region].dropna(subset=['inv_lifetime', 'precip'])
    if sub.empty:
        continue
    fit = inv_lt_simple[region]
    x_line = fit['x_sampled']
    ax.scatter(sub['precip'], sub['inv_lifetime'], s=40, alpha=0.7,
               color=region_colors[region], label=region, edgecolors='none')
    ax.plot(x_line, fit['slope'] * x_line + fit['inter'], '-', color=region_colors[region], lw=2)
    ax.fill_between(x_line, fit['ci_lower'], fit['ci_upper'], color=region_colors[region], alpha=0.15)
ax.set_xlabel('Precipitation (mm day⁻¹)', fontweight='bold')
ax.set_ylabel('1 / lifetime (day⁻¹)', fontweight='bold')
ax.set_title('(b) 1/τ vs precipitation', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='best')

plt.suptitle(
    'Figure 2 / SI Fig. 1 style: multi-region MEC–AE and precip–1/τ (Zhong et al. 2022)',
    fontsize=14, fontweight='bold',
)
plt.tight_layout()
fig_path = project_root / 'notebooks' / 'AOD_error_attribution_figure2_si_style.png'
if SAVE_FIGURE:
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {fig_path}')
plt.show()

# Extended validation panels (same colour scheme).
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# (a) MEC vs AE
ax = axes[0, 0]
for region in plot_regions:
    if region not in mec_ae_results:
        continue
    sub = reg_df[reg_df['region'] == region].dropna(subset=['MEC', 'AE'])
    if sub.empty:
        continue
    fit = mec_ae_results[region]
    x_line = fit['x_sampled']
    ax.scatter(sub['AE'], sub['MEC'], s=30, alpha=0.6, color=region_colors[region], label=region)
    ax.plot(x_line, fit['slope'] * x_line + fit['inter'], '--', color=region_colors[region], lw=2)
    ax.fill_between(x_line, fit['ci_lower'], fit['ci_upper'], color=region_colors[region], alpha=0.15)
    csub = constrained_df[constrained_df['region'] == region]
    if not csub.empty:
        ax.axvline(csub['AE_obs'].mean(), color=region_colors[region], ls=':', alpha=0.7)
        ax.axhline(csub['MEC_c'].mean(), color=region_colors[region], ls='-.', alpha=0.7)
ax.set_xlabel('Ångström Exponent', fontweight='bold')
ax.set_ylabel('MEC (m² kg⁻¹)', fontweight='bold')
ax.set_title('(a) MEC vs AE', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# (b) 1/lifetime vs precipitation — all regions in different colours (paper Fig. 2 / SI style)
ax = axes[0, 1]
for region in plot_regions:
    if region not in inv_lt_simple:
        continue
    sub = reg_df[reg_df['region'] == region].dropna(subset=['inv_lifetime', 'precip'])
    if sub.empty:
        continue
    fit = inv_lt_simple[region]
    x_line = fit['x_sampled']
    ax.scatter(sub['precip'], sub['inv_lifetime'], s=40, alpha=0.7,
               color=region_colors[region], label=region, edgecolors='none')
    ax.plot(x_line, fit['slope'] * x_line + fit['inter'], '--', color=region_colors[region], lw=2)
    ax.fill_between(x_line, fit['ci_lower'], fit['ci_upper'], color=region_colors[region], alpha=0.12)
    csub = constrained_df[constrained_df['region'] == region]
    if not csub.empty and np.isfinite(csub['tau_c_days'].mean()) and csub['tau_c_days'].mean() > 0:
        ax.axvline(csub['precip_obs'].mean(), color=region_colors[region], ls=':', alpha=0.6)
        ax.axhline(1.0 / csub['tau_c_days'].mean(), color=region_colors[region], ls='-.', alpha=0.6)
ax.set_xlabel('Precipitation (mm day⁻¹)', fontweight='bold')
ax.set_ylabel('1 / lifetime (day⁻¹)', fontweight='bold')
ax.set_title('(b) 1/τ vs precipitation (all regions)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, loc='best')

# (c) Predicted vs observed MEC
ax = axes[1, 0]
for region in plot_regions:
    if region not in mec_ae_results:
        continue
    sub = reg_df[reg_df['region'] == region].dropna(subset=['AE', 'MEC'])
    if sub.empty:
        continue
    fit = mec_ae_results[region]
    pred = fit['slope'] * sub['AE'].values + fit['inter']
    ax.scatter(pred, sub['MEC'].values, color=region_colors[region], label=region, alpha=0.6)
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.5)
ax.set_xlabel('Predicted MEC (m² kg⁻¹)', fontweight='bold')
ax.set_ylabel('Model MEC (m² kg⁻¹)', fontweight='bold')
ax.set_title('(c) MEC regression validation', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# (d) Predicted vs observed 1/lifetime
ax = axes[1, 1]
for region in plot_regions:
    if region not in inv_lt_results:
        continue
    sub = reg_df[reg_df['region'] == region].dropna(subset=['inv_lifetime', 'precip', 'AE'])
    if sub.empty:
        continue
    fit = inv_lt_results[region]
    pred = fit['intercept'] + fit['alpha_pr'] * sub['precip'].values + fit['beta_ae'] * sub['AE'].values
    ax.scatter(pred, sub['inv_lifetime'].values, color=region_colors[region], label=region, alpha=0.6)
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.5)
ax.set_xlabel('Predicted 1/τ (day⁻¹)', fontweight='bold')
ax.set_ylabel('Model 1/τ (day⁻¹)', fontweight='bold')
ax.set_title('(d) Lifetime regression validation', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

plt.suptitle('Figure 2: Linear regressions for MEC and lifetime (Zhong et al. 2022)', fontsize=15, fontweight='bold')
plt.tight_layout()
fig_path = project_root / 'notebooks' / 'AOD_error_attribution_figure2.png'
if SAVE_FIGURE:
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {fig_path}')
plt.show()

# -------------------------------------------------------------------------------
# 11. Figure 4: stacked-bar AOD error attribution
# -------------------------------------------------------------------------------
print('\n--- Figure 4: AOD error attribution ---')

fig_regions = [r for r in plot_regions if not decomp_df[decomp_df['region'] == r].empty]
n_regions = len(fig_regions)
fig, axes = plt.subplots(2, int(np.ceil(n_regions / 2)), figsize=(7 * int(np.ceil(n_regions / 2)), 10))
axes_flat = axes.flatten() if n_regions > 1 else [axes]

for idx, region in enumerate(fig_regions):
    ax = axes_flat[idx]
    sub = decomp_df[decomp_df['region'] == region].copy()
    # Alphabetical model order (shared numbering style with paper Fig. 4)
    sub = sub.sort_values('model', ascending=True).reset_index(drop=True)
    if sub.empty:
        ax.set_visible(False)
        continue

    x = np.arange(len(sub))
    pos_bottom = np.zeros(len(sub))
    neg_bottom = np.zeros(len(sub))

    components = [
        ('dAOD_E', 'Emission', '#1f77b4'),
        ('dAOD_tau', 'Lifetime', '#ff7f0e'),
        ('dAOD_MEC', 'MEC', '#2ca02c'),
        ('dAOD_cross', 'Cross', '#d62728'),
    ]

    for col, label, color in components:
        vals = sub[col].to_numpy(dtype=float)
        bottoms = np.where(vals >= 0, pos_bottom, neg_bottom)
        ax.bar(x, vals, bottom=bottoms, color=color, width=0.8, label=label)
        pos_bottom = np.where(vals >= 0, pos_bottom + vals, pos_bottom)
        neg_bottom = np.where(vals < 0, neg_bottom + vals, neg_bottom)

    ax.scatter(x, sub['dAOD_total'], color='black', s=45, zorder=5, label='Total AOD error')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(np.arange(1, len(sub) + 1), fontsize=10)
    ax.set_ylabel('AOD error contribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model number (alphabetical)', fontsize=12, fontweight='bold')
    ax.set_title(region.upper(), fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

# Hide unused panels.
for j in range(idx + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle('Figure 4: AOD error attribution by region (Zhong et al. 2022)', fontsize=15, fontweight='bold')
plt.tight_layout()
fig_path = project_root / 'notebooks' / 'AOD_error_attribution_figure4.png'
if SAVE_FIGURE:
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {fig_path}')
plt.show()

# Shared alphabetical model legend for Fig. 4 numbering
common_model_list = sorted(decomp_df['model'].dropna().unique().tolist())
fig_leg, ax_leg = plt.subplots(figsize=(6, max(3, 0.28 * max(len(common_model_list), 1))))
ax_leg.axis('off')
legend_lines = ['SHARED MODEL LIST (alphabetical):']
for i, model in enumerate(common_model_list, start=1):
    legend_lines.append(f'  {i:>2d}. {model}')
ax_leg.text(0.02, 0.98, '\n'.join(legend_lines), transform=ax_leg.transAxes,
            va='top', ha='left', fontsize=11, family='monospace')
plt.tight_layout()
fig_path_leg = project_root / 'notebooks' / 'AOD_error_attribution_figure4_model_legend.png'
if SAVE_FIGURE:
    fig_leg.savefig(fig_path_leg, dpi=300, bbox_inches='tight')
    print(f'Saved: {fig_path_leg}')
plt.show()

# Mean percentage contribution per region.
mean_decomp = decomp_df.groupby('region')[['pct_E', 'pct_tau', 'pct_MEC', 'pct_cross']].mean().loc[fig_regions]
fig, ax = plt.subplots(figsize=(10, 5))
mean_decomp.plot(kind='bar', stacked=True, ax=ax,
                 color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Mean contribution to |AOD error| (%)', fontweight='bold')
ax.set_title('Figure 4 (summary): Mean AOD error attribution by region', fontweight='bold')
ax.legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
fig_path2 = project_root / 'notebooks' / 'AOD_error_attribution_figure4_mean_pct.png'
if SAVE_FIGURE:
    plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
    print(f'Saved: {fig_path2}')
plt.show()

# -------------------------------------------------------------------------------
# 12. Figure 5 / Figure 6: outflow Default–EC–MFC and paper Fig. 6
# -------------------------------------------------------------------------------
print('\n--- Figure 5: African outflow Default / EC / MFC ---')

# Regional map of study areas.
print('\n--- Regional map ---')
sample_field = data[sample_model]['od550aer']['od550aer'].isel(time=0)
region_boxes = {
    name: (cfg['lon_range'][0], cfg['lon_range'][1], cfg['lat_range'][0], cfg['lat_range'][1])
    for name, cfg in REGIONS.items() if name != 'global'
}
fig_map = ct.fake_uba_map(
    lon=sample_field.lon.values,
    lat=sample_field.lat.values,
    c_array=sample_field.values,
    labels='AOD 550 (background)',
    title='Study regions (AOD error attribution)',
    region_boxes=region_boxes,
    show_region_labels=True,
    region_edgecolor='magenta',
    region_linewidth=2.0,
    cbar_orientation='vertical',
    cbar_extend='neither',
    show=False,
)
if SAVE_FIGURE:
    save_path = project_root / 'notebooks' / 'AOD_error_attribution_regions.png'
    fig_map.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'Saved regional map: {save_path}')
fig_map

# Fig. 5 style: Default / EC / MFC outflow AOD (models alphabetical)
if outflow_pred_df.empty:
    print('  No outflow Default/EC/MFC predictions to plot.')
else:
    sub = outflow_pred_df.sort_values('model', ascending=True).reset_index(drop=True)
    x = np.arange(len(sub))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(sub)), 6))
    ax.bar(x - width, sub['AOD_default'], width=width, color='#2ca02c', label='Default', alpha=0.9)
    ax.bar(x, sub['AOD_EC'], width=width, color='#ff7f0e', label='EC', alpha=0.9)
    ax.bar(x + width, [sub['AOD_MFC'].iloc[0]] * len(sub), width=width, color='#1f77b4',
           label='MFC', alpha=0.9)
    ax.axhline(sub['AOD_obs'].iloc[0], color='red', linestyle=':', linewidth=2, label='POLDER')
    ax.axhline(sub['AOD_default'].median(), color='#2ca02c', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Default median')
    ax.axhline(sub['AOD_EC'].median(), color='#ff7f0e', linestyle='--', linewidth=1.5,
               alpha=0.7, label='EC median')
    ax.set_xticks(x)
    ax.set_xticklabels(np.arange(1, len(sub) + 1))
    ax.set_xlabel('Model number (alphabetical)', fontweight='bold')
    ax.set_ylabel('Africa outflow AOD', fontweight='bold')
    ax.set_title('Figure 5 style: Outflow AOD — Default / EC / MFC', fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig_path = project_root / 'notebooks' / 'AOD_error_attribution_figure5_outflow.png'
    if SAVE_FIGURE:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {fig_path}')
    plt.show()

    # Companion: meta-fit vs model and EC/MFC vs POLDER
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(sub['AOD_default'], sub['AOD_meta_fit'], s=40, alpha=0.8, label='meta-fit vs Default')
    lims = [
        min(sub['AOD_default'].min(), sub['AOD_meta_fit'].min()) * 0.9,
        max(sub['AOD_default'].max(), sub['AOD_meta_fit'].max()) * 1.1,
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('Default outflow AOD', fontweight='bold')
    ax.set_ylabel('Meta-model fitted AOD', fontweight='bold')
    title_r2 = outflow_meta_params['r2'] if outflow_meta_params else np.nan
    ax.set_title(f'(a) Meta-model fit (R²={title_r2:.3f})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.scatter(x, sub['AOD_default'], label='Default', s=40, color='#2ca02c')
    ax.scatter(x, sub['AOD_EC'], label='EC', s=40, marker='s', color='#ff7f0e')
    ax.axhline(sub['AOD_MFC'].iloc[0], color='#1f77b4', linestyle='--', linewidth=2, label='MFC')
    ax.axhline(sub['AOD_obs'].iloc[0], color='red', linestyle=':', linewidth=2, label='POLDER')
    ax.set_xticks(x)
    ax.set_xticklabels(np.arange(1, len(sub) + 1))
    ax.set_xlabel('Model number (alphabetical)', fontweight='bold')
    ax.set_ylabel('Outflow AOD', fontweight='bold')
    ax.set_title('(b) Default / EC / MFC / POLDER', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.suptitle('African outflow meta-model predictions', fontweight='bold')
    plt.tight_layout()
    fig_path = project_root / 'notebooks' / 'AOD_error_attribution_figure5_meta.png'
    if SAVE_FIGURE:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {fig_path}')
    plt.show()

print('\n--- Figure 6 (Zhong et al. 2022 main paper) ---')
# Display the published Fig. 6 (ECHAM-HAM Default / EC / MFC longitude transect).
paper_fig6 = project_root / 'notebooks' / 'AOD_paper_figure6.png'
if paper_fig6.exists():
    img = plt.imread(paper_fig6)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(
        'Figure 6 (Zhong et al. 2022): ECHAM-HAM AOD errors — Default / EC / MFC\n'
        'Source & outflow of Southern Hemisphere Africa (from s41467-022-33680-4.pdf)',
        fontweight='bold', fontsize=12,
    )
    plt.tight_layout()
    if SAVE_FIGURE:
        out_path = project_root / 'notebooks' / 'AOD_error_attribution_figure6_paper.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {out_path}')
    plt.show()
else:
    print(f'  Paper Fig. 6 image not found at {paper_fig6}')

# Model vs POLDER AOD scatter by region (diagnostic companion).
fig, ax = plt.subplots(figsize=(9, 7))
all_vals = []
for region in plot_regions:
    sub = decomp_df[decomp_df['region'] == region]
    if sub.empty:
        continue
    aod_obs = polder_monthly[region]['AOD_550'].mean() if not polder_monthly[region].empty else np.nan
    if np.isnan(aod_obs):
        continue
    ax.scatter([aod_obs] * len(sub), sub['AOD_model'], color=region_colors[region],
               label=region, alpha=0.6, s=60)
    all_vals.extend([aod_obs] + sub['AOD_model'].tolist())
max_val = max(all_vals) if all_vals else 1.0
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1')
ax.set_xlabel('POLDER AOD (observation)', fontweight='bold')
ax.set_ylabel('Model AOD', fontweight='bold')
ax.set_title('Diagnostic: Model vs POLDER AOD by region', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
fig_path = project_root / 'notebooks' / 'AOD_error_attribution_figure6_scatter.png'
if SAVE_FIGURE:
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {fig_path}')
plt.show()

# Helper: build a POLDER seasonal-mean AOD field on the model grid.
def polder_grid_seasonal(region_name, var='AOD_550'):
    """Return POLDER seasonal-mean AOD as an xarray DataArray on its native 1° grid."""
    cfg = REGIONS[region_name]
    mask = polder_region_mask(polder_df, cfg['lon_range'], cfg['lat_range'])
    sub = polder_df[mask].copy()
    t0, t1 = cfg['time_slice']
    sub = sub[(sub['time'] >= t0) & (sub['time'] <= t1)]
    if sub.empty:
        return None
    grouped = sub.groupby(['latitude', 'longitude'])[var].mean()
    da = grouped.to_xarray().rename({'latitude': 'lat', 'longitude': 'lon'})
    da = da.assign_coords(lon=(da.lon % 360)).sortby('lon')
    return da


def model_grid_seasonal(model_name, region_name):
    """Return model seasonal-mean AOD on the model's native grid."""
    cfg = REGIONS[region_name]
    da = data[model_name]['od550aer']['od550aer']
    da = da.sel(time=slice(*cfg['time_slice'])).mean('time')
    da = da.assign_coords(lon=(da.lon % 360)).sortby('lon')
    return da


def plot_region_aod_diff(region_name, model_name, ax=None):
    """Plot model - POLDER AOD difference for one region/model."""
    model_da = model_grid_seasonal(model_name, region_name)
    polder_da = polder_grid_seasonal(region_name)
    if polder_da is None:
        return None
    polder_on_model = polder_da.interp(lat=model_da.lat, lon=model_da.lon, method='nearest')
    diff = model_da - polder_on_model

    # Mask outside the region.
    region_mask = ct.create_region_mask(
        model_da, name=region_name,
        lon_range=REGIONS[region_name]['lon_range'],
        lat_range=REGIONS[region_name]['lat_range'],
        surface_type=REGIONS[region_name]['surface_type'],
    )
    diff = diff.where(region_mask > 0)

    if ax is None:
        fig = ct.fake_uba_map(
            lon=model_da.lon.values, lat=model_da.lat.values, c_array=diff.values,
            labels='AOD error (model - POLDER)', title=f'{region_name.upper()} AOD error ({model_name})',
            mycolor='diff', zmin=-0.3, zmax=0.3, show=False,
        )
        return fig
    else:
        lon_grid, lat_grid = np.meshgrid(model_da.lon.values, model_da.lat.values)
        im = ax.pcolormesh(lon_grid, lat_grid, diff.values, cmap='RdBu_r', vmin=-0.3, vmax=0.3, shading='auto')
        ax.coastlines()
        ax.set_title(f'{region_name.upper()} AOD error ({model_name})', fontweight='bold')
        return im


# Map AOD error for one selected model in a few key regions (Fig. 6 companion maps).
map_regions = [r for r in ['africa', 'amazon', 'outflow_af'] if r in REGIONS]
if map_regions:
    selected_model = next((m for m in sorted(models) if m in model_seasonal.get('africa', {}).get('od550aer', {})),
                          sorted(models)[0] if models else None)
    if selected_model is not None:
        print(f'Selected model for AOD-error maps: {selected_model}')
        fig, axes = plt.subplots(1, len(map_regions), figsize=(6 * len(map_regions), 5),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        if len(map_regions) == 1:
            axes = [axes]
        for ax, region in zip(axes, map_regions):
            im = plot_region_aod_diff(region, selected_model, ax=ax)
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.05, fraction=0.046)
        cbar.set_label('AOD error (model - POLDER)', fontweight='bold')
        plt.suptitle('Figure 6 companion: AOD error maps (model - POLDER)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_path = project_root / 'notebooks' / 'AOD_error_attribution_figure6_maps.png'
        if SAVE_FIGURE:
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f'Saved: {fig_path}')
        plt.show()

# -------------------------------------------------------------------------------
# 13. Optional CSV outputs and final summary
# -------------------------------------------------------------------------------
if SAVE_CSV:
    out_csv = project_root / 'notebooks' / 'AOD_error_attribution_regression_data.csv'
    reg_df.to_csv(out_csv, index=False)
    print(f'Saved CSV: {out_csv}')

    out_csv = project_root / 'notebooks' / 'AOD_error_attribution_constrained_estimates.csv'
    constrained_df.to_csv(out_csv, index=False)
    print(f'Saved CSV: {out_csv}')

    out_csv = project_root / 'notebooks' / 'AOD_error_attribution_decomposition.csv'
    decomp_df.to_csv(out_csv, index=False)
    print(f'Saved CSV: {out_csv}')

    if not outflow_pred_df.empty:
        out_csv = project_root / 'notebooks' / 'AOD_error_attribution_outflow_predictions.csv'
        outflow_pred_df.to_csv(out_csv, index=False)
        print(f'Saved CSV: {out_csv}')

print('\nDone.')
