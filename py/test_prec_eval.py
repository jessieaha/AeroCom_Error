"""Multi-model mean/median evaluation for Zhong et al. 2022 BB regions.

Configuration → Load → Aggregate → Global map → Ensemble eval → Paper figures
→ Diagnostics.  ``AGG_STAT`` switches ensemble aggregation between mean and median.

Regions and fire seasons match ``notebooks/AOD_error_attribution.py`` (Nat. Commun.
2022) with AAOD extras (west_russia, boreal_na_west) for the ROI map.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import cartopy.crs as ccrs
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import linregress

warnings.filterwarnings('ignore', category=RuntimeWarning)

# -------------------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------------------
project_root = Path('/scistor/guest/gbb083/AeroCom')
py_dir = project_root / 'py'
if str(py_dir) not in sys.path:
    sys.path.insert(0, str(py_dir))

import functions
import cameo_toolbox as ct
import aerocom_data

# Ensemble statistic for multi-model figures: 'median' (paper-like) or 'mean'.
AGG_STAT = 'median'
assert AGG_STAT in ('mean', 'median'), f'AGG_STAT must be mean|median, got {AGG_STAT}'

VARIABLE_SET = 'both'  # 'AOD' | 'AAOD' | 'both'
EXCLUDE_MODELS = []
COMPARE_TAU_DEFS = True
SHOW_AAOD_EXTRA_REGIONS = True
SAVE_FIGURE = True
SAVE_CSV = True
# Baseline Nat. Commun.: no Amazon SOA inflation in this diagnostic notebook.
INCLUDE_AMAZON_SOA = False
AMAZON_SOA_FRACTION = 0.52
# Post-aggregation lifetime / MEC (paper Methods).
DERIVED_VAR_AFTER_AGG = ['lifetime', 'MEC']
# Models with monthly-only output to exclude from homogenization regression.
MONTHLY_OUTPUT_MODELS = []

fig_dir = project_root / 'notebooks'
print(f'AGG_STAT={AGG_STAT}, VARIABLE_SET={VARIABLE_SET}, '
      f'DERIVED_VAR_AFTER_AGG={DERIVED_VAR_AFTER_AGG}')

SOURCE_REGIONS = ['africa', 'amazon', 'se_asia', 'boreal_na', 'eastern_siberia']
OUTFLOW_REGION = 'outflow_af'
OUTFLOW_SOURCE = 'africa'
AAOD_EXTRA_REGIONS = ['west_russia', 'boreal_na_west']

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

# Zhong et al. 2022 Table 1 targets (approx.; constrained / reported).
PAPER_TABLE1 = {
    'africa': {'precip': 1.1, 'AE': 1.40, 'AOD': 0.48, 'tau_c': 4.0, 'E_c': 27.9e-11, 'MEC_c': 5.5},
    'amazon': {'precip': 1.9, 'AE': 1.55, 'AOD': 0.35, 'tau_c': 4.3, 'E_c': 18.2e-11, 'MEC_c': 5.0},
    'se_asia': {'precip': 0.8, 'AE': 1.45, 'AOD': 0.88, 'tau_c': 3.9, 'E_c': 47.6e-11, 'MEC_c': 5.5},
    'boreal_na': {'precip': 2.0, 'AE': 1.50, 'AOD': 0.16, 'tau_c': 3.0, 'E_c': 10.3e-11, 'MEC_c': 5.0},
    'eastern_siberia': {'precip': 0.6, 'AE': 1.40, 'AOD': 0.12, 'tau_c': 4.4, 'E_c': 8.3e-11, 'MEC_c': 5.0},
}
PAPER_PCT = {'pct_E': 38.0, 'pct_tau': 22.0, 'pct_MEC': 27.0}

# Align load/emi with MEC species (no gaseous loadso2).
LOAD_VARS = ['loadbc', 'loaddust', 'loadoa', 'loadso4', 'loadss']
EMI_VARS = ['emibc', 'emidust', 'emioa', 'emiso2', 'emiss']
BCOA_LOAD_VARS = ['loadbc', 'loadoa']
BCOA_EMI_VARS = ['emibc', 'emioa']

VARIABLES = [
    'abs550aer', 'emibc', 'emidust', 'emioa', 'emiso2', 'emiss',
    'loadbc', 'loaddust', 'loadoa', 'loadso4', 'loadss',
    'od440aer', 'od550aer', 'od870aer', 'od865aer', 'precip',
]


def agg_stat(values, stat=None):
    """Aggregate a sequence with AGG_STAT (mean or median)."""
    stat = AGG_STAT if stat is None else stat
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.nanmedian(arr) if stat == 'median' else np.nanmean(arr))


def savefig(name):
    path = fig_dir / name
    if SAVE_FIGURE:
        plt.savefig(path, dpi=200, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.show()

# -------------------------------------------------------------------------------
# 2. Load models, normalise, derived vars
# -------------------------------------------------------------------------------
print('Loading monthly NetCDF...')
raw_data = aerocom_data.load_monthly_data_from_netcdf(
    output_base_dir=str(project_root / 'Data' / 'AP3_processed_monthly'),
    variables=VARIABLES,
)
models = sorted([m for m in raw_data.keys() if m not in EXCLUDE_MODELS])
print(f'Models ({len(models)}): {models}')


def normalize_dataset_time(ds, var_hint=None):
    if ds is None or not list(ds.data_vars):
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
            print(f'  Warning: time convert failed {var_hint}: {e}')
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


def normalize_precipitation_units(precip_ds):
    if precip_ds is None:
        return None
    var_name = list(precip_ds.data_vars)[0]
    da = precip_ds[var_name]
    units = da.attrs.get('units', 'kg m-2 s-1').lower().replace(' ', '')
    if units in ('mmday-1', 'mm/day', 'mmday^-1', 'mmd-1'):
        factor = 1.0
    elif units in ('gm-2s-1', 'gm^-2s^-1', 'g/m2/s'):
        factor = 86.4
    elif units in ('ms-1', 'm/s'):
        factor = 86400.0 * 1000.0
    else:
        factor = 86400.0
    da = da * factor
    try:
        if float(da.mean().values) > 100.0:
            da = da / 1000.0
    except Exception:
        pass
    da.attrs['units'] = 'mm day-1'
    return da.to_dataset(name=var_name)


data = {}
for m in models:
    normalized = {}
    for var in VARIABLES:
        if raw_data[m].get(var) is None:
            continue
        try:
            normalized[var] = normalize_dataset_time(raw_data[m][var], f'{m}/{var}')
        except Exception as e:
            print(f'  Failed {m}/{var}: {e}')
            normalized[var] = None
    normalized['load_total'] = sum_datasets(normalized, LOAD_VARS, 'load_total')
    normalized['emi_total'] = sum_datasets(normalized, EMI_VARS, 'emi_total')
    normalized['load_BC_OA'] = sum_datasets(normalized, BCOA_LOAD_VARS, 'load_BC_OA')
    normalized['emi_BC_OA'] = sum_datasets(normalized, BCOA_EMI_VARS, 'emi_BC_OA')
    if normalized.get('precip') is not None:
        normalized['precip'] = normalize_precipitation_units(normalized['precip'])
    data[m] = normalized

derived = {m: {} for m in models}
for m in models:
    mec_ds = aerocom_data.calculate_derived_var(data[m], m, 'MEC')
    derived[m]['MEC'] = None if mec_ds is None else mec_ds[list(mec_ds.data_vars)[0]] * 1000.0
    for dv in ['MAC', 'SSA', 'AE']:
        ds = aerocom_data.calculate_derived_var(data[m], m, dv)
        derived[m][dv] = None if ds is None else ds[list(ds.data_vars)[0]]
    # Grid-cell lifetime kept for A/B comparison even when post-agg is used.
    if data[m].get('load_total') is not None and data[m].get('emi_total') is not None:
        lt = data[m]['load_total']['load_total'] / (data[m]['emi_total']['emi_total'] * 86400.0)
        derived[m]['lifetime_grid'] = lt.where((lt > 1e-3) & (lt < 365))
    else:
        derived[m]['lifetime_grid'] = None
    if 'lifetime' not in DERIVED_VAR_AFTER_AGG:
        derived[m]['lifetime'] = derived[m]['lifetime_grid']
    else:
        derived[m]['lifetime'] = None

data_derived = {}
for m in models:
    data_derived[m] = {}
    for k, v in data[m].items():
        if v is not None:
            data_derived[m][k] = v[list(v.data_vars)[0]]
    for k, v in derived[m].items():
        if v is not None:
            data_derived[m][k] = v

# -------------------------------------------------------------------------------
# 3. Regional masks and aggregation
# -------------------------------------------------------------------------------
sample_model = next(m for m in models if data[m].get('od550aer') is not None)
template = data[sample_model]['od550aer'].isel(time=0)
masks = {}
for name, cfg in REGIONS.items():
    masks[name] = ct.create_region_mask(
        template, name=name,
        lon_range=cfg['lon_range'], lat_range=cfg['lat_range'],
        surface_type=cfg.get('surface_type', 'all'),
        mask_registry=masks,
    )
print('Regions:', list(masks.keys()))

variables_to_aggregate = [
    'MEC', 'AE', 'lifetime', 'lifetime_grid', 'MAC', 'SSA',
    'load_total', 'emi_total', 'load_BC_OA', 'emi_BC_OA',
    'precip', 'od550aer', 'abs550aer',
]


def aggregate_region(model_dict, var_name, region_name, return_time_series=False, skipna=False):
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
            print(f'  Agg failed {var_name} {model} {region_name}: {e}')
    return result


print('Aggregating (this can take a while)...')
model_monthly = {
    region: {
        var: aggregate_region(data_derived, var, region, return_time_series=True,
                              skipna=(var in ('lifetime', 'lifetime_grid')))
        for var in variables_to_aggregate
    }
    for region in REGIONS
}
model_seasonal = {
    region: {
        var: aggregate_region(data_derived, var, region, return_time_series=False,
                              skipna=(var in ('lifetime', 'lifetime_grid')))
        for var in variables_to_aggregate
    }
    for region in REGIONS
}


def compute_derived_after_aggregation(monthly_dict, seasonal_dict):
    """Paper-style τ = ⟨load⟩/⟨E⟩ and MEC = ⟨AOD⟩/⟨load⟩ after regional aggregation."""
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
                for model in load:
                    if model not in emi:
                        continue
                    val = _safe_div(load[model], emi[model], 86400.0)
                    if val is not None:
                        out[model] = val
                if out:
                    agg[region]['lifetime'] = out
            if 'MEC' in DERIVED_VAR_AFTER_AGG:
                aod = agg[region].get('od550aer', {})
                load = agg[region].get('load_total', {})
                out = {}
                for model in aod:
                    if model not in load:
                        continue
                    val = _safe_div(aod[model], load[model], 1.0)
                    if val is not None:
                        out[model] = val
                if out:
                    agg[region]['MEC'] = out


compute_derived_after_aggregation(model_monthly, model_seasonal)
print('Post-aggregation lifetime/MEC applied:', DERIVED_VAR_AFTER_AGG)

# -------------------------------------------------------------------------------
# 4. Observations: POLDER + GPCP (land-masked for land regions)
# -------------------------------------------------------------------------------
polder_path = project_root / 'Data' / 'AP3_POLDER_Collocated' / \
    'POLDER_GRASP_coloc_3h_AP3_2010_lon0_10.0_lat0_10.0.parquet'
polder_cols = ['time', 'longitude', 'latitude', 'AOD_550', 'AAOD_550', 'AOD_440', 'AOD_870']
polder_df = pd.read_parquet(polder_path, columns=polder_cols)
polder_df['time'] = pd.to_datetime(polder_df['time'])
polder_df['month'] = polder_df['time'].dt.to_period('M')
with np.errstate(divide='ignore', invalid='ignore'):
    polder_df['AE'] = -np.log(polder_df['AOD_550'] / polder_df['AOD_440']) / np.log(550.0 / 440.0)
    polder_df['SSA'] = 1.0 - polder_df['AAOD_550'] / polder_df['AOD_550']
polder_df = polder_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=['AOD_550', 'AE', 'SSA', 'AAOD_550']
)
print(f'POLDER rows: {len(polder_df)}')


def polder_region_mask(df, lon_range, lat_range):
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    if lon_min > lon_max:
        mask_lon = (df['longitude'] >= lon_min) | (df['longitude'] <= lon_max)
    else:
        mask_lon = (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)
    mask_lat = (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)
    return mask_lon & mask_lat


def polder_monthly_means(df, region_name):
    cfg = REGIONS[region_name]
    mask = polder_region_mask(df, cfg['lon_range'], cfg['lat_range'])
    sub = df[mask].copy()
    t0, t1 = cfg['time_slice']
    sub = sub[(sub['time'] >= t0) & (sub['time'] <= t1)]
    if sub.empty:
        return pd.DataFrame()
    sub['w'] = np.cos(np.deg2rad(sub['latitude']))
    rows = []
    for month, g in sub.groupby('month'):
        w = g['w']
        rows.append({
            'month': month,
            'AOD_550': np.average(g['AOD_550'], weights=w),
            'AAOD_550': np.average(g['AAOD_550'], weights=w),
            'AE': np.average(g['AE'], weights=w),
            'SSA': np.average(g['SSA'], weights=w),
        })
    return pd.DataFrame(rows)


polder_monthly = {r: polder_monthly_means(polder_df, r) for r in REGIONS}

gpcp_path = project_root / 'Data' / 'Prec' / 'GPCP_2010_0-360.nc'
gpcp = xr.open_dataset(gpcp_path)
gpcp_precip = gpcp['sat_gauge_precip']
print(f'GPCP: {gpcp_path}')


def gpcp_region_mean(precip_da, region_name, apply_land_mask=True):
    """Area-weighted GPCP mean; land BB regions use the same land mask as models."""
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
    # GPCP lat may be ascending or descending.
    if sub.lat.values[0] > sub.lat.values[-1]:
        sub = sub.sel(lat=slice(lat_max, lat_min))
    else:
        sub = sub.sel(lat=slice(lat_min, lat_max))
    sub = sub.sel(time=slice(*cfg['time_slice']))
    if apply_land_mask and cfg.get('surface_type') in ('land', 'ocean'):
        try:
            mask = masks[region_name]
            mask_on = mask.interp(lat=sub.lat, lon=sub.lon, method='nearest')
            sub = sub.where(mask_on > 0)
        except Exception as e:
            print(f'  GPCP land mask skip {region_name}: {e}')
    coslat = np.cos(np.deg2rad(sub.lat))
    weights = coslat * xr.ones_like(sub.isel(time=0))
    weights = weights.where(np.isfinite(sub.isel(time=0)))
    wsum = weights.sum()
    if float(wsum) == 0:
        return sub.mean(dim=['lat', 'lon'])
    weights = weights / wsum
    return (sub * weights).sum(dim=['lat', 'lon'])


gpcp_region = {r: gpcp_region_mean(gpcp_precip, r, apply_land_mask=True) for r in REGIONS}
gpcp_region_all = {r: gpcp_region_mean(gpcp_precip, r, apply_land_mask=False) for r in REGIONS}
print('GPCP land-masked seasonal means:')
for r in SOURCE_REGIONS:
    v = float(gpcp_region[r].mean().values) if gpcp_region[r].size else np.nan
    v2 = float(gpcp_region_all[r].mean().values) if gpcp_region_all[r].size else np.nan
    paper = PAPER_TABLE1.get(r, {}).get('precip', np.nan)
    print(f'  {r:16s} land={v:.3f}  all={v2:.3f}  paper={paper}')

def to_scalar(val):
    """Convert seasonal aggregate (float, 0-d array, or DataArray) to float."""
    if val is None:
        return np.nan
    try:
        if hasattr(val, 'values'):
            v = np.asarray(val.values).ravel()
            return float(v[0]) if v.size else np.nan
        return float(np.asarray(val).ravel()[0])
    except Exception:
        return np.nan


# Build long model×region table
rows = []
for region in REGIONS:
    if region == 'global':
        continue
    for model in models:
        aod = to_scalar(model_seasonal[region].get('od550aer', {}).get(model))
        aaod = to_scalar(model_seasonal[region].get('abs550aer', {}).get(model))
        load = to_scalar(model_seasonal[region].get('load_total', {}).get(model))
        emi = to_scalar(model_seasonal[region].get('emi_total', {}).get(model))
        tau = to_scalar(model_seasonal[region].get('lifetime', {}).get(model))
        tau_g = to_scalar(model_seasonal[region].get('lifetime_grid', {}).get(model))
        mec = to_scalar(model_seasonal[region].get('MEC', {}).get(model))
        ae = to_scalar(model_seasonal[region].get('AE', {}).get(model))
        precip = to_scalar(model_seasonal[region].get('precip', {}).get(model))
        etm = np.nan
        if all(np.isfinite(v) for v in [emi, tau, mec]):
            etm = emi * tau * 86400.0 * mec
        rows.append({
            'region': region, 'model': model,
            'AOD': aod, 'AAOD': aaod, 'E': emi, 'load': load,
            'tau': tau, 'tau_grid': tau_g, 'MEC': mec, 'AE': ae, 'precip': precip,
            'E_tau_MEC': etm,
            'inv_lifetime': (1.0 / tau) if np.isfinite(tau) and tau > 0 else np.nan,
        })
reg_df = pd.DataFrame(rows)
print(f'reg_df rows: {len(reg_df)}')
if SAVE_CSV:
    reg_df.to_csv(fig_dir / 'test_prec_model_region_table.csv', index=False)

# -------------------------------------------------------------------------------
# 5. Global ROI map (Fig. 1 style) via fake_uba_map
# -------------------------------------------------------------------------------
print('\n--- A0 / Fig.1: global ROI map ---')
roi_names = SOURCE_REGIONS + [OUTFLOW_REGION]
if SHOW_AAOD_EXTRA_REGIONS:
    roi_names = roi_names + [r for r in AAOD_EXTRA_REGIONS if r in REGIONS]
region_boxes = {
    name: (REGIONS[name]['lon_range'][0], REGIONS[name]['lon_range'][1],
           REGIONS[name]['lat_range'][0], REGIONS[name]['lat_range'][1])
    for name in roi_names
}

# Multi-model AGG_STAT of annual-mean AOD on sample grid
aod_stack = []
for m in models:
    da = data_derived[m].get('od550aer')
    if da is None:
        continue
    try:
        aod_stack.append(da.sel(time=slice('2010-01-01', '2010-12-31')).mean('time'))
    except Exception:
        continue
if aod_stack:
    # Align to a common lon/lat grid (first model) before concat.
    ref = aod_stack[0]
    aligned = []
    for da in aod_stack:
        try:
            aligned.append(da.interp(lat=ref.lat, lon=ref.lon, method='nearest'))
        except Exception:
            continue
    stacked = xr.concat(aligned, dim='model')
    ens_aod = stacked.median('model') if AGG_STAT == 'median' else stacked.mean('model')
    # Cartopy-friendly lon in [-180, 180]
    lon180 = ((ens_aod.lon.values + 180) % 360) - 180
    order = np.argsort(lon180)
    lon_plot = lon180[order]
    field_plot = np.asarray(ens_aod.values)[:, order]
    # Convert region boxes that use 0–360 to display coords for labels
    region_boxes_plot = {}
    for name, (lo0, lo1, la0, la1) in region_boxes.items():
        def _to180(lo):
            return ((lo + 180) % 360) - 180
        region_boxes_plot[name] = (_to180(lo0), _to180(lo1), la0, la1)

    try:
        fig_map = ct.fake_uba_map(
            lon=lon_plot, lat=ens_aod.lat.values, c_array=field_plot,
            labels='AOD 550 (multi-model ' + AGG_STAT + ')',
            title='Fig. 1 style: ROI boxes on multi-model AOD',
            region_boxes=region_boxes,  # fake_uba_map handles 0–360 wrap boxes
            show_region_labels=True,
            region_edgecolor='magenta',
            region_linewidth=2.0,
            zmin=0, zmax=0.6,
            cbar_orientation='vertical',
            show=False,
        )
        if SAVE_FIGURE:
            fig_map.savefig(fig_dir / 'test_prec_figure1_roi_map.png', dpi=200, bbox_inches='tight')
            print(f'Saved: {fig_dir / "test_prec_figure1_roi_map.png"}')
        plt.show()
    except Exception as e:
        print(f'  fake_uba_map failed ({e}); falling back to simple PlateCarree map')
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.coastlines()
        ax.set_global()
        mesh = ax.pcolormesh(lon_plot, ens_aod.lat.values, field_plot,
                             transform=ccrs.PlateCarree(), cmap='YlOrRd', vmin=0, vmax=0.6,
                             shading='auto')
        for name, (lo0, lo1, la0, la1) in region_boxes.items():
            # Draw possibly wrapped boxes
            if lo0 > lo1:
                segments = [(lo0, 360), (0, lo1)]
            else:
                segments = [(lo0, lo1)]
            for a, b in segments:
                ax.plot([a, b, b, a, a], [la0, la0, la1, la1, la0],
                        color='magenta', lw=2, transform=ccrs.PlateCarree())
            mid_lon = (lo0 + lo1) / 2 if lo0 <= lo1 else ((lo0 + lo1 + 360) / 2) % 360
            ax.text(mid_lon, (la0 + la1) / 2, name, color='magenta', fontsize=8,
                    ha='center', transform=ccrs.PlateCarree(),
                    path_effects=[PathEffects.withStroke(linewidth=2, foreground='w')])
        plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, label='AOD 550')
        ax.set_title('Fig. 1 style: ROI boxes on multi-model AOD', fontweight='bold')
        plt.tight_layout()
        savefig('test_prec_figure1_roi_map.png')

# -------------------------------------------------------------------------------
# 6. Ensemble regional bars and monthly time series (A2–A3)
# -------------------------------------------------------------------------------
print('\n--- A2: regional bars vs obs ---')
plot_regs = SOURCE_REGIONS + [OUTFLOW_REGION]
metrics = ['AOD', 'AAOD', 'precip', 'AE'] if VARIABLE_SET in ('both', 'AAOD') else ['AOD', 'precip', 'AE']
fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
if len(metrics) == 1:
    axes = [axes]
x = np.arange(len(plot_regs))
width = 0.25
for ax, metric in zip(axes, metrics):
    model_mean, model_med, obs_vals = [], [], []
    for region in plot_regs:
        sub = reg_df[reg_df['region'] == region]
        if metric == 'precip':
            model_mean.append(sub['precip'].mean())
            model_med.append(sub['precip'].median())
            obs_vals.append(float(gpcp_region[region].mean().values)
                            if gpcp_region[region].size else np.nan)
        elif metric == 'AOD':
            model_mean.append(sub['AOD'].mean())
            model_med.append(sub['AOD'].median())
            pm = polder_monthly[region]
            obs_vals.append(pm['AOD_550'].mean() if not pm.empty else np.nan)
        elif metric == 'AAOD':
            model_mean.append(sub['AAOD'].mean())
            model_med.append(sub['AAOD'].median())
            pm = polder_monthly[region]
            obs_vals.append(pm['AAOD_550'].mean() if not pm.empty else np.nan)
        elif metric == 'AE':
            model_mean.append(sub['AE'].mean())
            model_med.append(sub['AE'].median())
            pm = polder_monthly[region]
            obs_vals.append(pm['AE'].mean() if not pm.empty else np.nan)
    ax.bar(x - width, model_mean, width, label='model mean', color='#1f77b4')
    ax.bar(x, model_med, width, label='model median', color='#ff7f0e')
    ax.bar(x + width, obs_vals, width, label='obs', color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('_', '\n') for r in plot_regs], fontsize=8)
    ax.set_title(metric, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    if metric == metrics[0]:
        ax.legend(fontsize=8)
plt.suptitle(f'Regional ensemble vs observations (AGG_STAT default={AGG_STAT})', fontweight='bold')
plt.tight_layout()
savefig('test_prec_A2_regional_bars.png')

print('\n--- A3: monthly precip vs AOD ---')
n = len(SOURCE_REGIONS)
fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n), sharex=False)
if n == 1:
    axes = [axes]
for ax, region in zip(axes, SOURCE_REGIONS):
    cfg = REGIONS[region]
    # GPCP monthly
    gser = gpcp_region[region]
    gdf = gser.to_dataframe(name='precip').reset_index()
    gdf['month'] = pd.to_datetime(gdf['time']).dt.to_period('M')
    # POLDER monthly
    pm = polder_monthly[region]
    # Model monthly AGG_STAT AOD
    aod_m = model_monthly[region].get('od550aer', {})
    months = sorted(set().union(*[
        set(pd.to_datetime(v.time.values).to_period('M'))
        for v in aod_m.values() if hasattr(v, 'time')
    ])) if aod_m else []
    ens_aod_ts = []
    for mo in months:
        vals = []
        for m, da in aod_m.items():
            try:
                sel = da.sel(time=str(mo))
                vals.append(float(sel.values) if np.ndim(sel.values) == 0 else float(np.nanmean(sel.values)))
            except Exception:
                continue
        ens_aod_ts.append({'month': mo, 'AOD_ens': agg_stat(vals)})
    ens_df = pd.DataFrame(ens_aod_ts)
    ax2 = ax.twinx()
    if not gdf.empty:
        ax.plot(gdf['month'].astype(str), gdf['precip'], 'b-o', label='GPCP precip', ms=4)
    if not pm.empty:
        ax2.plot(pm['month'].astype(str), pm['AOD_550'], 'g-s', label='POLDER AOD', ms=4)
    if not ens_df.empty:
        ax2.plot(ens_df['month'].astype(str), ens_df['AOD_ens'], 'r-^',
                 label=f'Model {AGG_STAT} AOD', ms=4)
    ax.set_ylabel('Precip (mm d⁻¹)', color='b')
    ax2.set_ylabel('AOD', color='r')
    ax.set_title(region.upper(), fontweight='bold')
    ax.tick_params(axis='x', labelrotation=45)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc='upper right')
plt.suptitle('A3: Monthly GPCP precip vs POLDER / ensemble AOD', fontweight='bold')
plt.tight_layout()
savefig('test_prec_A3_monthly_precip_aod.png')

# -------------------------------------------------------------------------------
# 7. Diagnostics B1–B5
# -------------------------------------------------------------------------------
print('\n--- B1: AOD vs E×τ×MEC consistency ---')
fig, axes = plt.subplots(1, len(SOURCE_REGIONS), figsize=(3.2 * len(SOURCE_REGIONS), 4))
for ax, region in zip(axes, SOURCE_REGIONS):
    sub = reg_df[reg_df['region'] == region].dropna(subset=['AOD', 'E_tau_MEC'])
    ax.scatter(sub['E_tau_MEC'], sub['AOD'], s=30, alpha=0.7)
    if not sub.empty:
        lims = [0, max(sub['AOD'].max(), sub['E_tau_MEC'].max()) * 1.1]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.axvline(agg_stat(sub['E_tau_MEC']), color='C1', ls=':', label=f'ens {AGG_STAT} EτMEC')
        ax.axhline(agg_stat(sub['AOD']), color='C2', ls=':', label=f'ens {AGG_STAT} AOD')
        resid = np.abs(sub['AOD'] - sub['E_tau_MEC']) / sub['AOD'].replace(0, np.nan)
        ax.set_title(f'{region}\nmed |res|={resid.median():.0%}', fontsize=9)
    ax.set_xlabel('E×τ×MEC')
    ax.set_ylabel('AOD')
    ax.grid(True, alpha=0.3)
plt.suptitle('B1: Consistency AOD ≈ E×τ×MEC', fontweight='bold')
plt.tight_layout()
savefig('test_prec_B1_consistency.png')

if COMPARE_TAU_DEFS:
    print('\n--- B2: lifetime grid vs post-agg ---')
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SOURCE_REGIONS))
    tau_post = [agg_stat(reg_df[reg_df['region'] == r]['tau']) for r in SOURCE_REGIONS]
    tau_grid = [agg_stat(reg_df[reg_df['region'] == r]['tau_grid']) for r in SOURCE_REGIONS]
    paper_tau = [PAPER_TABLE1[r]['tau_c'] for r in SOURCE_REGIONS]
    ax.bar(x - 0.25, tau_grid, 0.25, label='grid-mean τ')
    ax.bar(x, tau_post, 0.25, label='post-agg ⟨load⟩/⟨E⟩')
    ax.bar(x + 0.25, paper_tau, 0.25, label='paper τ_c')
    ax.set_xticks(x)
    ax.set_xticklabels(SOURCE_REGIONS, rotation=30, ha='right')
    ax.set_ylabel('Lifetime (days)')
    ax.set_title(f'B2: Lifetime definitions (ensemble {AGG_STAT})', fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    savefig('test_prec_B2_lifetime_defs.png')

print('\n--- B3: Table 1 replica ---')
table_rows = []
for region in SOURCE_REGIONS:
    sub = reg_df[reg_df['region'] == region]
    pm = polder_monthly[region]
    paper = PAPER_TABLE1[region]
    table_rows.append({
        'region': region,
        'precip_ours': float(gpcp_region[region].mean().values) if gpcp_region[region].size else np.nan,
        'precip_paper': paper['precip'],
        'AE_ours': pm['AE'].mean() if not pm.empty else np.nan,
        'AE_paper': paper['AE'],
        'AOD_ours': pm['AOD_550'].mean() if not pm.empty else np.nan,
        'AOD_paper': paper['AOD'],
        'tau_ens': agg_stat(sub['tau']),
        'tau_paper': paper['tau_c'],
        'MEC_ens': agg_stat(sub['MEC']) / 1000.0,  # to m² g⁻¹ for paper units
        'MEC_paper': paper['MEC_c'],
        'E_ens': agg_stat(sub['E']),
        'E_paper': paper['E_c'],
    })
table1_df = pd.DataFrame(table_rows)
print(table1_df.to_string(index=False))
if SAVE_CSV:
    table1_df.to_csv(fig_dir / 'test_prec_B3_table1_replica.csv', index=False)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
pairs = [
    ('precip_ours', 'precip_paper', 'Precip'),
    ('AE_ours', 'AE_paper', 'AE'),
    ('AOD_ours', 'AOD_paper', 'AOD'),
    ('tau_ens', 'tau_paper', 'τ (d)'),
    ('MEC_ens', 'MEC_paper', 'MEC (m² g⁻¹)'),
    ('E_ens', 'E_paper', 'E'),
]
for ax, (o, p, title) in zip(axes.flat, pairs):
    ax.scatter(table1_df[p], table1_df[o], s=60)
    for _, row in table1_df.iterrows():
        ax.annotate(row['region'][:6], (row[p], row[o]), fontsize=7)
    lims = [
        min(table1_df[p].min(), table1_df[o].min()) * 0.8,
        max(table1_df[p].max(), table1_df[o].max()) * 1.2,
    ]
    ax.plot(lims, lims, 'k--', alpha=0.4)
    ax.set_xlabel('Paper')
    ax.set_ylabel('Ours')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
plt.suptitle('B3: Table 1 replica (ours vs Zhong 2022)', fontweight='bold')
plt.tight_layout()
savefig('test_prec_B3_table1.png')

print('\n--- B5: GPCP land vs all-surface ---')
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(SOURCE_REGIONS))
land_v = [float(gpcp_region[r].mean().values) for r in SOURCE_REGIONS]
all_v = [float(gpcp_region_all[r].mean().values) for r in SOURCE_REGIONS]
pap_v = [PAPER_TABLE1[r]['precip'] for r in SOURCE_REGIONS]
ax.bar(x - 0.25, land_v, 0.25, label='land-masked')
ax.bar(x, all_v, 0.25, label='all-surface')
ax.bar(x + 0.25, pap_v, 0.25, label='paper')
ax.set_xticks(x)
ax.set_xticklabels(SOURCE_REGIONS, rotation=30, ha='right')
ax.set_ylabel('Precip (mm d⁻¹)')
ax.set_title('B5: GPCP land-masked vs all-surface vs paper', fontweight='bold')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
savefig('test_prec_B5_gpcp_landmask.png')

print('\n--- B4 / SI Fig.10: homogenization QC (sampled vs regional AOD) ---')
homo_rows = []
for region in SOURCE_REGIONS:
    regional = model_seasonal[region].get('od550aer', {})
    pm = polder_monthly[region]
    obs = float(pm['AOD_550'].mean()) if not pm.empty else np.nan
    for model, reg_aod in regional.items():
        if reg_aod is None or not np.isfinite(float(reg_aod)):
            continue
        homo_rows.append({
            'region': region, 'model': model,
            'AOD_regional': float(reg_aod),
            'AOD_obs_sampled': obs,
        })
homo_df = pd.DataFrame(homo_rows)
if not homo_df.empty:
    fig, axes = plt.subplots(1, len(SOURCE_REGIONS), figsize=(3 * len(SOURCE_REGIONS), 4))
    for ax, region in zip(axes, SOURCE_REGIONS):
        sub = homo_df[homo_df['region'] == region]
        ax.scatter(sub['AOD_regional'], sub['AOD_obs_sampled'], s=25, alpha=0.6)
        if len(sub) >= 2:
            lr = linregress(sub['AOD_regional'], sub['AOD_obs_sampled'])
            x_line = np.linspace(sub['AOD_regional'].min(), sub['AOD_regional'].max(), 40)
            ax.plot(x_line, lr.slope * x_line + lr.intercept, 'r-',
                    label=f'R²={lr.rvalue**2:.2f}')
            ax.legend(fontsize=7)
        ax.set_xlabel('Model regional AOD')
        ax.set_ylabel('POLDER sampled AOD')
        ax.set_title(region, fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle('B4 / SI Fig. 10 style: regional vs sampled AOD', fontweight='bold')
    plt.tight_layout()
    savefig('test_prec_B4_homogenization_qc.png')

# -------------------------------------------------------------------------------
# 8. Fig. 2 / SI Fig. 1 regressions (C1)
# -------------------------------------------------------------------------------
print('\n--- C1 / Fig.2: MEC–AE and precip–1/τ ---')
plot_regions = SOURCE_REGIONS
colors = plt.cm.tab10(np.linspace(0, 1, len(plot_regions)))
region_colors = {r: colors[i] for i, r in enumerate(plot_regions)}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
for region in plot_regions:
    sub = reg_df[reg_df['region'] == region].dropna(subset=['MEC', 'AE'])
    if len(sub) < 3:
        continue
    ax.scatter(sub['AE'], sub['MEC'] / 1000.0, s=40, alpha=0.7,
               color=region_colors[region], label=region, edgecolors='none')
    lr = linregress(sub['AE'], sub['MEC'] / 1000.0)
    x_line = np.linspace(sub['AE'].min(), sub['AE'].max(), 50)
    ax.plot(x_line, lr.slope * x_line + lr.intercept, '-', color=region_colors[region], lw=2)
ax.set_xlabel('Ångström Exponent')
ax.set_ylabel('MEC (m² g⁻¹)')
ax.set_title('(a) MEC vs AE')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
for region in plot_regions:
    sub = reg_df[reg_df['region'] == region].dropna(subset=['inv_lifetime', 'precip'])
    if len(sub) < 3:
        continue
    ax.scatter(sub['precip'], sub['inv_lifetime'], s=40, alpha=0.7,
               color=region_colors[region], label=region, edgecolors='none')
    lr = linregress(sub['precip'], sub['inv_lifetime'])
    x_line = np.linspace(max(0, sub['precip'].min()), sub['precip'].max(), 50)
    ax.plot(x_line, lr.slope * x_line + lr.intercept, '-', color=region_colors[region], lw=2)
ax.set_xlabel('Precipitation (mm day⁻¹)')
ax.set_ylabel('1 / lifetime (day⁻¹)')
ax.set_title('(b) 1/τ vs precipitation')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.suptitle('Fig. 2 / SI Fig. 1 style: multi-region regressions', fontweight='bold')
plt.tight_layout()
savefig('test_prec_figure2_si_style.png')

# SI Fig. 2: predicted vs modelled 1/τ
print('\n--- SI Fig.2: predicted vs modelled 1/τ ---')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, use_ae, title in zip(
    axes,
    [True, False],
    ['(a) Pr + AE', '(b) Pr only'],
):
    for region in plot_regions:
        sub = reg_df[reg_df['region'] == region].dropna(subset=['inv_lifetime', 'precip', 'AE'])
        if len(sub) < 4:
            continue
        if use_ae:
            X = np.column_stack([np.ones(len(sub)), sub['precip'], sub['AE']])
        else:
            X = np.column_stack([np.ones(len(sub)), sub['precip']])
        y = sub['inv_lifetime'].values
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        ax.scatter(pred, y, s=25, alpha=0.6, color=region_colors[region], label=region)
    lims = [0, ax.get_xlim()[1]]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('Predicted 1/τ')
    ax.set_ylabel('Model 1/τ')
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.suptitle('SI Fig. 2 style: lifetime regression validation', fontweight='bold')
plt.tight_layout()
savefig('test_prec_si_fig2_lifetime_validation.png')

# -------------------------------------------------------------------------------
# 9. Constrained estimates, decomp, Fig. 4 / 5 style
# -------------------------------------------------------------------------------
print('\n--- Constrained estimates + error attribution ---')
INTERCEPT_0 = False


def fit_mec_ae(sub):
    sub = sub.dropna(subset=['MEC', 'AE'])
    if len(sub) < 3:
        return None
    y = sub['MEC'].values
    x = sub['AE'].values
    if INTERCEPT_0:
        slope = np.sum(x * y) / np.sum(x * x)
        inter = 0.0
    else:
        lr = linregress(x, y)
        slope, inter = lr.slope, lr.intercept
    return {'slope': slope, 'inter': inter}


def fit_inv_lt(sub):
    sub = sub.dropna(subset=['inv_lifetime', 'precip', 'AE'])
    if len(sub) < 4:
        return None
    X = np.column_stack([np.ones(len(sub)), sub['precip'], sub['AE']])
    y = sub['inv_lifetime'].values
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return {'intercept': coef[0], 'alpha_pr': coef[1], 'beta_ae': coef[2]}


mec_ae_results = {}
inv_lt_results = {}
for region in SOURCE_REGIONS:
    sub = reg_df[reg_df['region'] == region]
    mec_ae_results[region] = fit_mec_ae(sub)
    inv_lt_results[region] = fit_inv_lt(sub)

constrained_rows = []
for region in SOURCE_REGIONS:
    fit_m = mec_ae_results.get(region)
    fit_t = inv_lt_results.get(region)
    pm = polder_monthly[region]
    ae_obs = float(pm['AE'].mean()) if not pm.empty else np.nan
    aod_obs = float(pm['AOD_550'].mean()) if not pm.empty else np.nan
    precip_obs = float(gpcp_region[region].mean().values) if gpcp_region[region].size else np.nan
    if fit_m is None or fit_t is None or not np.isfinite(aod_obs):
        continue
    mec_c = fit_m['slope'] * ae_obs + fit_m['inter']
    inv_tau_c = fit_t['intercept'] + fit_t['alpha_pr'] * precip_obs + fit_t['beta_ae'] * ae_obs
    tau_c = 1.0 / inv_tau_c if inv_tau_c > 0 else np.nan
    e_c = aod_obs / (tau_c * 86400.0 * mec_c) if np.isfinite(tau_c) and mec_c else np.nan
    constrained_rows.append({
        'region': region, 'AE_obs': ae_obs, 'AOD_obs': aod_obs, 'precip_obs': precip_obs,
        'MEC_c': mec_c, 'tau_c_days': tau_c, 'E_c': e_c,
    })
    print(f'  {region}: τ_c={tau_c:.2f}d  MEC_c={mec_c/1000:.2f} m²/g  E_c={e_c:.3e}  AOD_obs={aod_obs:.3f}')
constrained_df = pd.DataFrame(constrained_rows)

decomp_rows = []
for region in SOURCE_REGIONS:
    c = constrained_df[constrained_df['region'] == region]
    if c.empty:
        continue
    mec_c = float(c['MEC_c'].iloc[0])
    tau_c = float(c['tau_c_days'].iloc[0])
    e_c = float(c['E_c'].iloc[0])
    aod_c = float(c['AOD_obs'].iloc[0])
    for _, row in reg_df[reg_df['region'] == region].iterrows():
        if not all(np.isfinite([row['AOD'], row['E'], row['tau'], row['MEC']])):
            continue
        dE = row['E'] - e_c
        dtau = row['tau'] - tau_c
        dMEC = row['MEC'] - mec_c
        # AOD = E * τ * 86400 * MEC
        dAOD_E = dE * tau_c * 86400.0 * mec_c
        dAOD_tau = e_c * dtau * 86400.0 * mec_c
        dAOD_MEC = e_c * tau_c * 86400.0 * dMEC
        dAOD_total = row['AOD'] - aod_c
        cross = dAOD_total - (dAOD_E + dAOD_tau + dAOD_MEC)
        abs_sum = abs(dAOD_E) + abs(dAOD_tau) + abs(dAOD_MEC) + abs(cross)
        decomp_rows.append({
            'region': region, 'model': row['model'],
            'AOD_model': row['AOD'], 'dAOD_E': dAOD_E, 'dAOD_tau': dAOD_tau,
            'dAOD_MEC': dAOD_MEC, 'dAOD_cross': cross, 'dAOD_total': dAOD_total,
            'pct_E': 100 * abs(dAOD_E) / abs_sum if abs_sum else np.nan,
            'pct_tau': 100 * abs(dAOD_tau) / abs_sum if abs_sum else np.nan,
            'pct_MEC': 100 * abs(dAOD_MEC) / abs_sum if abs_sum else np.nan,
            'pct_cross': 100 * abs(cross) / abs_sum if abs_sum else np.nan,
            'E_model': row['E'], 'E_c': e_c,
        })
decomp_df = pd.DataFrame(decomp_rows)
if not decomp_df.empty:
    print(decomp_df.groupby('region')[['pct_E', 'pct_tau', 'pct_MEC', 'pct_cross']].agg(AGG_STAT))
if SAVE_CSV:
    constrained_df.to_csv(fig_dir / 'test_prec_constrained.csv', index=False)
    decomp_df.to_csv(fig_dir / 'test_prec_decomposition.csv', index=False)

# Fig. 4 style alphabetical bars
print('\n--- Fig.4 style attribution ---')
if not decomp_df.empty:
    n_r = len(SOURCE_REGIONS)
    fig, axes = plt.subplots(2, int(np.ceil(n_r / 2)), figsize=(7 * int(np.ceil(n_r / 2)), 10))
    axes_flat = axes.flatten()
    for idx, region in enumerate(SOURCE_REGIONS):
        ax = axes_flat[idx]
        sub = decomp_df[decomp_df['region'] == region].sort_values('model').reset_index(drop=True)
        if sub.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(sub))
        pos_b = np.zeros(len(sub))
        neg_b = np.zeros(len(sub))
        for col, color in [('dAOD_E', '#1f77b4'), ('dAOD_tau', '#ff7f0e'),
                           ('dAOD_MEC', '#2ca02c'), ('dAOD_cross', '#d62728')]:
            vals = sub[col].to_numpy(float)
            bottoms = np.where(vals >= 0, pos_b, neg_b)
            ax.bar(x, vals, bottom=bottoms, color=color, width=0.8)
            pos_b = np.where(vals >= 0, pos_b + vals, pos_b)
            neg_b = np.where(vals < 0, neg_b + vals, neg_b)
        ax.scatter(x, sub['dAOD_total'], color='k', s=30, zorder=5)
        ax.axhline(0, color='k', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(np.arange(1, len(sub) + 1), fontsize=8)
        ax.set_title(region.upper())
        ax.set_xlabel('Model # (alphabetical)')
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle('Fig. 4 style: AOD error attribution', fontweight='bold')
    plt.tight_layout()
    savefig('test_prec_figure4_attribution.png')

    # Ensemble mean/median pct vs paper (SI Fig. 5 style)
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_pct = decomp_df.groupby('region')[['pct_E', 'pct_tau', 'pct_MEC', 'pct_cross']].mean()
    med_pct = decomp_df.groupby('region')[['pct_E', 'pct_tau', 'pct_MEC', 'pct_cross']].median()
    mean_pct = mean_pct.reindex(SOURCE_REGIONS)
    med_pct = med_pct.reindex(SOURCE_REGIONS)
    x = np.arange(len(SOURCE_REGIONS))
    # stacked median
    bottom = np.zeros(len(SOURCE_REGIONS))
    for col, color, lab in [
        ('pct_E', '#1f77b4', 'E'), ('pct_tau', '#ff7f0e', 'τ'),
        ('pct_MEC', '#2ca02c', 'MEC'), ('pct_cross', '#d62728', 'cross'),
    ]:
        ax.bar(x, med_pct[col], bottom=bottom, color=color, label=f'{lab} (median)')
        bottom = bottom + med_pct[col].fillna(0).values
    ax.axhline(PAPER_PCT['pct_E'], color='#1f77b4', ls='--', alpha=0.5)
    ax.axhline(PAPER_PCT['pct_E'] + PAPER_PCT['pct_tau'], color='#ff7f0e', ls='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SOURCE_REGIONS, rotation=30, ha='right')
    ax.set_ylabel('% of |AOD error|')
    ax.set_title('SI Fig. 5 style: ensemble median attribution vs paper ~38/22/27', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    savefig('test_prec_si_fig5_pct.png')

# Fig. 3 / SI 11: emissions
print('\n--- Fig.3 / SI11: emissions ---')
if not constrained_df.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SOURCE_REGIONS))
    e_model = [agg_stat(reg_df[reg_df['region'] == r]['E']) for r in SOURCE_REGIONS]
    e_c = []
    e_paper = []
    for r in SOURCE_REGIONS:
        c = constrained_df[constrained_df['region'] == r]
        e_c.append(float(c['E_c'].iloc[0]) if not c.empty else np.nan)
        e_paper.append(PAPER_TABLE1[r]['E_c'])
    ax.bar(x - 0.25, e_model, 0.25, label=f'Model {AGG_STAT}')
    ax.bar(x, e_c, 0.25, label='Constrained E_c')
    ax.bar(x + 0.25, e_paper, 0.25, label='Paper E_c')
    ax.set_xticks(x)
    ax.set_xticklabels(SOURCE_REGIONS, rotation=30, ha='right')
    ax.set_ylabel('Emission (kg m⁻² s⁻¹)')
    ax.set_title('Fig. 3 / SI Fig. 11 style: emissions', fontweight='bold')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    savefig('test_prec_figure3_emissions.png')

# Outflow meta-model Fig. 5 / SI 13
print('\n--- Fig.5 / SI13: outflow Default/EC/MFC ---')
meta_rows = []
for model in models:
    aod_out = model_seasonal[OUTFLOW_REGION].get('od550aer', {}).get(model)
    e_src = model_seasonal[OUTFLOW_SOURCE].get('emi_total', {}).get(model)
    tau_src = model_seasonal[OUTFLOW_SOURCE].get('lifetime', {}).get(model)
    mec_src = model_seasonal[OUTFLOW_SOURCE].get('MEC', {}).get(model)
    if not all(v is not None and np.isfinite(float(v)) for v in [aod_out, e_src, tau_src, mec_src]):
        continue
    burden = float(e_src) * float(tau_src) * 86400.0
    meta_rows.append({
        'model': model, 'AOD_out': float(aod_out),
        'E_tau_MEC': burden * float(mec_src), 'E_tau': burden, 'MEC': float(mec_src),
        'E': float(e_src), 'tau': float(tau_src),
    })
meta_df = pd.DataFrame(meta_rows)
outflow_pred_df = pd.DataFrame()
if len(meta_df) >= 5:
    X = np.column_stack([meta_df['E_tau_MEC'], meta_df['E_tau'], meta_df['MEC'], np.ones(len(meta_df))])
    y = meta_df['AOD_out'].values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f'  Outflow meta R²={r2:.3f}')
    c_src = constrained_df[constrained_df['region'] == OUTFLOW_SOURCE]
    aod_out_obs = float(polder_monthly[OUTFLOW_REGION]['AOD_550'].mean()) \
        if not polder_monthly[OUTFLOW_REGION].empty else np.nan
    pred_rows = []
    if not c_src.empty:
        e_c = float(c_src['E_c'].iloc[0])
        tau_c = float(c_src['tau_c_days'].iloc[0])
        mec_c = float(c_src['MEC_c'].iloc[0])
        burden_c = e_c * tau_c * 86400.0
        aod_mfc = coeffs[0] * burden_c * mec_c + coeffs[1] * burden_c + coeffs[2] * mec_c + coeffs[3]
        aod_src_obs = float(c_src['AOD_obs'].iloc[0])
        for _, row in meta_df.iterrows():
            aod_src = model_seasonal[OUTFLOW_SOURCE]['od550aer'].get(row['model'])
            if aod_src and float(aod_src) != 0:
                e_ec = row['E'] * (aod_src_obs / float(aod_src))
            else:
                e_ec = row['E']
            burden_ec = e_ec * row['tau'] * 86400.0
            aod_ec = (coeffs[0] * burden_ec * row['MEC'] + coeffs[1] * burden_ec
                      + coeffs[2] * row['MEC'] + coeffs[3])
            pred_rows.append({
                'model': row['model'],
                'AOD_default': row['AOD_out'],
                'AOD_meta_fit': float(coeffs[0] * row['E_tau_MEC'] + coeffs[1] * row['E_tau']
                                     + coeffs[2] * row['MEC'] + coeffs[3]),
                'AOD_EC': float(aod_ec), 'AOD_MFC': float(aod_mfc), 'AOD_obs': aod_out_obs,
            })
        outflow_pred_df = pd.DataFrame(pred_rows).sort_values('model').reset_index(drop=True)

if not outflow_pred_df.empty:
    sub = outflow_pred_df
    x = np.arange(len(sub))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(sub)), 5))
    ax.bar(x - w, sub['AOD_default'], w, color='#2ca02c', label='Default')
    ax.bar(x, sub['AOD_EC'], w, color='#ff7f0e', label='EC')
    ax.bar(x + w, [sub['AOD_MFC'].iloc[0]] * len(sub), w, color='#1f77b4', label='MFC')
    ax.axhline(sub['AOD_obs'].iloc[0], color='red', ls=':', lw=2, label='POLDER')
    ax.set_xticks(x)
    ax.set_xticklabels(np.arange(1, len(sub) + 1))
    ax.set_xlabel('Model # (alphabetical)')
    ax.set_ylabel('Outflow AOD')
    ax.set_title('Fig. 5 style: Default / EC / MFC', fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    savefig('test_prec_figure5_outflow.png')

    # SI Fig. 13 validation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sub['AOD_default'], sub['AOD_meta_fit'], s=40, alpha=0.8)
    lims = [min(sub['AOD_default'].min(), sub['AOD_meta_fit'].min()) * 0.9,
            max(sub['AOD_default'].max(), sub['AOD_meta_fit'].max()) * 1.1]
    ax.plot(lims, lims, 'k--')
    ax.set_xlabel('Default outflow AOD')
    ax.set_ylabel('Meta-model fit')
    ax.set_title(f'SI Fig. 13 style: outflow meta validation (R²={r2:.3f})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig('test_prec_si_fig13_outflow_meta.png')

# Fig. 6 embed
print('\n--- Fig.6: paper embed ---')
paper_fig6 = fig_dir / 'AOD_paper_figure6.png'
if paper_fig6.exists():
    img = plt.imread(paper_fig6)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Fig. 6 (Zhong 2022): ECHAM-HAM Default/EC/MFC (published)', fontweight='bold')
    plt.tight_layout()
    savefig('test_prec_figure6_paper.png')
else:
    print(f'  Missing {paper_fig6}')

# SI Fig. 12 leave-one-out sensitivity (τ_c / MEC_c)
print('\n--- SI Fig.12: leave-one-out constrained τ/MEC ---')
loo_rows = []
for region in SOURCE_REGIONS:
    sub_all = reg_df[reg_df['region'] == region]
    models_r = sorted(sub_all['model'].unique())
    pm = polder_monthly[region]
    ae_obs = float(pm['AE'].mean()) if not pm.empty else np.nan
    precip_obs = float(gpcp_region[region].mean().values) if gpcp_region[region].size else np.nan
    for drop in [None] + models_r:
        sub = sub_all if drop is None else sub_all[sub_all['model'] != drop]
        fm = fit_mec_ae(sub)
        ft = fit_inv_lt(sub)
        if fm is None or ft is None:
            continue
        mec_c = fm['slope'] * ae_obs + fm['inter']
        inv_t = ft['intercept'] + ft['alpha_pr'] * precip_obs + ft['beta_ae'] * ae_obs
        tau_c = 1.0 / inv_t if inv_t > 0 else np.nan
        loo_rows.append({
            'region': region,
            'dropped': 'all' if drop is None else drop,
            'tau_c': tau_c,
            'MEC_c_m2g': mec_c / 1000.0,
        })
loo_df = pd.DataFrame(loo_rows)
if not loo_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, ylab in zip(axes, ['tau_c', 'MEC_c_m2g'], ['τ_c (d)', 'MEC_c (m² g⁻¹)']):
        for region in SOURCE_REGIONS:
            sub = loo_df[loo_df['region'] == region]
            base = sub[sub['dropped'] == 'all']
            leave = sub[sub['dropped'] != 'all']
            if base.empty:
                continue
            ax.scatter([region] * len(leave), leave[col], s=20, alpha=0.5, color=region_colors[region])
            ax.scatter([region], base[col], s=80, marker='*', color=region_colors[region],
                       edgecolors='k', zorder=5)
        ax.set_ylabel(ylab)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y', alpha=0.3)
    axes[0].set_title('Constrained τ')
    axes[1].set_title('Constrained MEC')
    plt.suptitle('SI Fig. 12 style: leave-one-out constrained estimates', fontweight='bold')
    plt.tight_layout()
    savefig('test_prec_si_fig12_loo.png')

print('\nDone. Key outputs in', fig_dir)
print(f'AGG_STAT={AGG_STAT}. Switch to mean/median and re-run for the other ensemble.')
