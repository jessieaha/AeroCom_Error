"""AOD error attribution for AeroCom biomass-burning aerosols (Zhong et al. 2022).

Dual-ensemble gap analysis comparing the full AeroCom ensemble and the 17-model
paper subset (Fig. 4/5 legend order). Reproduces the framework of
``s41467-022-33680-4`` (Nature Communications):

    AOD = E × τ × MEC                                    (1)

Outputs (per ensemble: ``full`` and ``paper``):
* Figure 2 / SI – MEC vs AE and precipitation vs 1/τ regressions + LOO validation
* Table 1 comparison vs published constrained estimates
* Gap diagnostics when |gap_pct| > GAP_THRESHOLD_PCT
* Figure 4 – stacked-bar AOD error decomposition + mean pct comparison
* Figure 5 – African outflow Default / EC / MFC meta-model
* Regional map, robustness CSV, gap summary markdown

Out of scope (handled elsewhere):
* Fig. 3 emission inventories
* Fig. 6 ECHAM-HAM longitude transect rerun
* AERONET cross-check
"""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Agg backend only when not running inside IPython/Jupyter.
try:
    get_ipython()  # noqa: F821
    _IN_IPYTHON = True
except NameError:
    _IN_IPYTHON = False

import matplotlib.pyplot as plt
if not _IN_IPYTHON:
    import matplotlib
    matplotlib.use('Agg')

import cartopy.crs as ccrs
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import linregress

warnings.filterwarnings('ignore', category=RuntimeWarning)

# -------------------------------------------------------------------------------
# 1. Configuration and data loading
# -------------------------------------------------------------------------------
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path('/scistor/guest/gbb083/AeroCom')
py_dir = project_root / 'py'
if str(py_dir) not in sys.path:
    sys.path.insert(0, str(py_dir))

import functions
import cameo_toolbox as ct
import aerocom_data
import notebook_setup as setup

print(f'Project root: {project_root}', flush=True)
print(f'Python dir:   {py_dir}', flush=True)

EXCLUDE_MODELS = [
    'GISS-ModelE2p1p1-OMA_AP3-CTRL',
    'GISS-ModelE2p1p1-MATRIX_AP3-CTRL',
    'NorESM2-met2010_AP3-CTRL-v3',
    'GEOS-i33p2-met2010_AP3-CTRL-2010',
]

PAPER_MODELS = [
    'CAM5.3-Oslo_AP3-CTRL2016-PD', 'ECHAM6-HAM2_AP3-CTRL2016-PD',
    'ECHAM6-SALSA_CTRL2016-PD', 'ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD',
    'TM5_AP3-CTRL2016', 'CAM5-ATRAS_AP3-CTRL', 'EC-Earth3-AerChem-met2010_AP3-CTRL2019',
    'ECHAM6.3-HAM2.3-met2010_AP3-CTRL', 'ECHAM6.3-SALSA2.0-met2010_AP3-CTRL',
    'GEOS-i33p2-met2010_AP3-CTRL', 'GFDL-AM4-met2010_AP3-CTRL',
    'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010', 'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010',
    'INCA_AP3-CTRL', 'NorESM2-met2010_AP3-CTRL',
    'SPRINTARS-T213_AP3-CTRL2016-PD',  # paper #16 (prefer T213 over MIROC-SPRINTARS)
    'TM5-met2010_AP3-CTRL2019',
]

LIFETIME_MIN_DAYS = 1e-3
LIFETIME_MAX_DAYS = 365
DERIVED_VAR_AFTER_AGG = ['lifetime', 'MEC']
if 'lifetime' in DERIVED_VAR_AFTER_AGG:
    LIFETIME_MIN_DAYS = None
    LIFETIME_MAX_DAYS = None

# Post-aggregation lifetime filtering (box-budget outliers).
POST_AGG_LIFETIME_MAX_DAYS = 30.0
POST_AGG_LIFETIME_MIN_DAYS = 0.3
# True: τ = load / deposition; False: τ = load / emission
USE_DEPOSITION_FOR_LIFETIME = False

INTERCEPT_0 = False
POLDER_HOMOGENIZE = True
MONTHLY_OUTPUT_MODELS = []
INCLUDE_AMAZON_SOA = False
AMAZON_SOA_FRACTION = 0.52
SAVE_FIGURE = True
SAVE_CSV = True
GAP_THRESHOLD_PCT = 20.0
FIGURE_DIR = project_root / 'figure'
TABLE_DIR = project_root / 'tables'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_REGIONS = ['africa', 'amazon', 'se_asia', 'boreal_na', 'eastern_siberia']
OUTFLOW_REGION = 'outflow_af'
OUTFLOW_SOURCE = 'africa'
# Only these regions are masked/aggregated/used in the notebook.
ANALYSIS_REGIONS = SOURCE_REGIONS + [OUTFLOW_REGION]

PAPER_TABLE1 = {
    'amazon': {'precip': 1.9, 'AE': 1.4, 'AOD': 0.4, 'E': 18.2, 'tau': 4.3, 'MEC': 5.9},
    'africa': {'precip': 1.1, 'AE': 1.4, 'AOD': 0.56, 'E': 27.9, 'tau': 4.0, 'MEC': 5.9},
    'se_asia': {'precip': 0.8, 'AE': 1.2, 'AOD': 0.88, 'E': 47.6, 'tau': 3.9, 'MEC': 5.6},
    'boreal_na': {'precip': 1.8, 'AE': 1.3, 'AOD': 0.16, 'E': 10.3, 'tau': 3.0, 'MEC': 6.0},
    'eastern_siberia': {'precip': 0.6, 'AE': 1.1, 'AOD': 0.21, 'E': 8.3, 'tau': 4.4, 'MEC': 6.8},
}

PAPER_DECOMP_PCT = {'pct_E': 38, 'pct_tau': 22, 'pct_MEC': 27, 'pct_cross': 13}

REGION_COLORS = {
    'africa': '#ff7f0e',
    'amazon': '#2ca02c',
    'se_asia': '#d62728',
    'boreal_na': '#1f77b4',
    'eastern_siberia': '#8c564b',
    'outflow_af': '#9467bd',
}

# Zhong Fig. 2c/d reports Pearson correlation R (not R²).
PAPER_LOO_R = {'MEC': 0.72, 'inv_lifetime': 0.78}


# Shared helpers (regions, aggregation, lifetime, labels) live in notebook_setup.
short_model_name = setup.short_model_name


print(f'  INTERCEPT_0          = {INTERCEPT_0}', flush=True)
print(f'  POLDER_HOMOGENIZE    = {POLDER_HOMOGENIZE}', flush=True)
print(f'  INCLUDE_AMAZON_SOA   = {INCLUDE_AMAZON_SOA}', flush=True)
print(f'  ANALYSIS_REGIONS     = {ANALYSIS_REGIONS}', flush=True)
print(f'  DERIVED_VAR_AFTER_AGG= {DERIVED_VAR_AFTER_AGG}', flush=True)
print(f'  POST_AGG_LIFETIME    = [{POST_AGG_LIFETIME_MIN_DAYS}, {POST_AGG_LIFETIME_MAX_DAYS}] days',
      flush=True)
print(f'  USE_DEPOSITION_FOR_LIFETIME = {USE_DEPOSITION_FOR_LIFETIME}  '
      f"({'load/deposition' if USE_DEPOSITION_FOR_LIFETIME else 'load/emission'})",
      flush=True)
print(f'  GAP_THRESHOLD_PCT    = {GAP_THRESHOLD_PCT}', flush=True)

VARIABLES = [
    'abs550aer', 'depbc', 'depdust', 'depoa', 'depso2', 'depso4', 'depss',
    'emibc', 'emidust', 'emioa', 'emiso2', 'emiss',
    'loadbc', 'loaddust', 'loadoa', 'loadso2', 'loadso4', 'loadss',
    'od440aer', 'od550aer', 'od870aer', 'od865aer',
    'od550bc', 'od550dust', 'od550oa', 'od550so4', 'od550ss',
    'precip',
]

print('Loading monthly data from processed NetCDF files...', flush=True)
raw_data = aerocom_data.load_monthly_data_from_netcdf(
    output_base_dir=str(project_root / 'Data' / 'AP3_processed_monthly'),
    variables=VARIABLES,
)

missing_excluded = [m for m in EXCLUDE_MODELS if m not in raw_data]
if missing_excluded:
    print(f'Warning: excluded models not found in data: {missing_excluded}', flush=True)
models = sorted([m for m in raw_data if m not in EXCLUDE_MODELS])
actually_excluded = [m for m in EXCLUDE_MODELS if m in raw_data]
print(f'Number of models: {len(models)} (excluded {len(actually_excluded)})', flush=True)

print('\n--- PAPER_MODELS mapping ---', flush=True)
for i, m in enumerate(PAPER_MODELS, 1):
    status = 'OK' if m in models else 'MISSING'
    print(f'  {i:2d}. [{status}] {m}', flush=True)
paper_models = [m for m in PAPER_MODELS if m in models]
ENSEMBLES = {'full': models, 'paper': paper_models}
print(f'Ensembles: full={len(models)}, paper={len(paper_models)}', flush=True)

LOAD_VARS = ['loadbc', 'loaddust', 'loadoa', 'loadso4', 'loadss']
EMI_VARS = ['emibc', 'emidust', 'emioa', 'emiso2', 'emiss']
DEP_TOTAL_VARS = ['depbc', 'depdust', 'depoa', 'depso4', 'depss']
BCOA_LOAD_VARS = ['loadbc', 'loadoa']
BCOA_EMI_VARS = ['emibc', 'emioa']
BCOA_DEP_VARS = ['depbc', 'depoa']
LIFETIME_VARS = {'lifetime', 'lifetime_BC_OA'}

variables_to_aggregate = [
    'MEC', 'AE', 'lifetime', 'lifetime_BC_OA', 'MAC', 'SSA',
    'load_total', 'emi_total', 'dep_total', 'load_BC_OA', 'emi_BC_OA', 'dep_BC_OA',
    'emibc', 'emioa', 'precip', 'od550aer',
]

# Region definitions shared with AAOD (notebook_setup.REGIONS).
# Prefer setup.REGIONS everywhere; REGIONS alias kept for local cfg lookups.
REGIONS = setup.REGIONS


# Time / precip / lifetime filters: setup.normalize_* / setup.filter_lifetime

def recompute_ae_440(model_data):
    """Recompute AE from od440/od550 when od440aer is present."""
    od550_raw = model_data.get('od550aer')
    od440_raw = model_data.get('od440aer')
    if od550_raw is None or od440_raw is None:
        return None
    od550 = od550_raw[list(od550_raw.data_vars)[0]]
    od440 = od440_raw[list(od440_raw.data_vars)[0]]
    with np.errstate(divide='ignore', invalid='ignore'):
        ae = -np.log(od550 / od440) / np.log(550.0 / 440.0)
    ae = ae.where((od550 > 0) & (od440 > 0))
    return ae


# -------------------------------------------------------------------------------
# 2. Normalise time, build summed variables, and normalise precipitation units
# -------------------------------------------------------------------------------
print('Normalising model data...', flush=True)
data = {}
for m in models:
    normalized = {}
    for var in VARIABLES:
        if raw_data[m].get(var) is None:
            continue
        try:
            normalized[var] = setup.normalize_dataset_time(raw_data[m][var], var_hint=f'{m}/{var}')
        except Exception as e:
            print(f'  Failed to normalise {m}/{var}: {e}', flush=True)
            normalized[var] = None
    normalized['load_total'] = setup.sum_datasets(normalized, LOAD_VARS, 'load_total')
    normalized['emi_total'] = setup.sum_datasets(normalized, EMI_VARS, 'emi_total')
    normalized['load_BC_OA'] = setup.sum_datasets(
        normalized, BCOA_LOAD_VARS, 'load_BC_OA', require_all=True
    )
    normalized['emi_BC_OA'] = setup.sum_datasets(
        normalized, BCOA_EMI_VARS, 'emi_BC_OA', require_all=True
    )
    normalized['dep_total'] = setup.sum_deposition_datasets(
        normalized, DEP_TOTAL_VARS, 'dep_total', require_all=True
    )
    normalized['dep_BC_OA'] = setup.sum_deposition_datasets(
        normalized, BCOA_DEP_VARS, 'dep_BC_OA', require_all=True
    )
    if normalized['dep_BC_OA'] is None:
        miss = [k for k in BCOA_DEP_VARS if normalized.get(k) is None]
        print(f'  {m}: incomplete BC+OA deposition (missing {miss}); '
              f'dep_BC_OA=None', flush=True)
    if normalized.get('precip') is not None:
        normalized['precip'] = setup.normalize_precipitation_units(normalized['precip'])
    data[m] = normalized
n_dep_bcoa = sum(1 for m in models if data[m].get('dep_BC_OA') is not None)
n_dep_tot = sum(1 for m in models if data[m].get('dep_total') is not None)
print(f'  dep_BC_OA available: {n_dep_bcoa}/{len(models)}', flush=True)
print(f'  dep_total available: {n_dep_tot}/{len(models)}', flush=True)

# -------------------------------------------------------------------------------
# 3. Calculate derived variables (MEC, lifetime, AE, MAC, SSA)
# -------------------------------------------------------------------------------
print('Computing derived variables...', flush=True)
derived = {m: {} for m in models}
lifetime_filter_stats = []

for m in models:
    aerocom_data.align_model_grids(data[m], ref_var='od550aer', model_hint=m)

    mec_ds = aerocom_data.calculate_derived_var(data[m], m, 'MEC')
    if mec_ds is not None:
        var_name = list(mec_ds.data_vars)[0]
        derived[m]['MEC'] = mec_ds[var_name] * 1000.0
    else:
        derived[m]['MEC'] = None

    for dv in ['MAC', 'SSA']:
        ds = aerocom_data.calculate_derived_var(data[m], m, dv)
        derived[m][dv] = None if ds is None else ds[list(ds.data_vars)[0]]

    ae_440 = recompute_ae_440(data[m])
    if ae_440 is not None:
        derived[m]['AE'] = ae_440
    else:
        ae_ds = aerocom_data.calculate_derived_var(data[m], m, 'AE')
        derived[m]['AE'] = None if ae_ds is None else ae_ds[list(ae_ds.data_vars)[0]]

    # Pre-aggregation lifetimes only when not deferred to post-agg (DERIVED_VAR_AFTER_AGG).
    if 'lifetime' not in DERIVED_VAR_AFTER_AGG:
        if data[m].get('load_total') is not None and data[m].get('emi_total') is not None:
            load_da = data[m]['load_total']['load_total']
            emi_da = data[m]['emi_total']['emi_total']
            lt = load_da / (emi_da * 3600 * 24)
            lt, lt_stats = setup.filter_lifetime(lt, m, 'lifetime', LIFETIME_MAX_DAYS, LIFETIME_MIN_DAYS)
            derived[m]['lifetime'] = lt
            lifetime_filter_stats.append({'model': m, 'var': 'lifetime', **lt_stats})
        else:
            derived[m]['lifetime'] = None
    else:
        derived[m]['lifetime'] = None

    if 'lifetime_BC_OA' not in DERIVED_VAR_AFTER_AGG and 'lifetime' not in DERIVED_VAR_AFTER_AGG:
        if data[m].get('load_BC_OA') is not None and data[m].get('emi_BC_OA') is not None:
            load_da = data[m]['load_BC_OA']['load_BC_OA']
            emi_da = data[m]['emi_BC_OA']['emi_BC_OA']
            lt = load_da / (emi_da * 3600 * 24)
            lt, lt_stats = setup.filter_lifetime(lt, m, 'lifetime_BC_OA', LIFETIME_MAX_DAYS, LIFETIME_MIN_DAYS)
            derived[m]['lifetime_BC_OA'] = lt
            lifetime_filter_stats.append({'model': m, 'var': 'lifetime_BC_OA', **lt_stats})
        else:
            derived[m]['lifetime_BC_OA'] = None
    else:
        derived[m]['lifetime_BC_OA'] = None

data_derived = {}
for m in models:
    data_derived[m] = {}
    for k, v in data[m].items():
        if v is not None:
            data_derived[m][k] = v[list(v.data_vars)[0]]
    for k, v in derived[m].items():
        if v is not None:
            data_derived[m][k] = v

print('\nDerived-variable availability (pre-aggregation):', flush=True)
for dv in ['MEC', 'AE', 'lifetime', 'lifetime_BC_OA', 'MAC', 'SSA']:
    n = sum(1 for m in derived if derived[m].get(dv) is not None)
    print(f'  {dv:15s}: {n:2d} / {len(models)} models', flush=True)



# -------------------------------------------------------------------------------
# 4. Regional masks and monthly/seasonal aggregation
# -------------------------------------------------------------------------------
sample_model = next((m for m in models if data[m].get('od550aer') is not None), None)
template = data[sample_model]['od550aer'].isel(time=0)
print(f'Template grid from {sample_model}: {template.dims}', flush=True)

SURFACE_TYPE = None
masks = setup.create_analysis_masks(template, ANALYSIS_REGIONS, surface_type=SURFACE_TYPE)
print('Regions created (ANALYSIS_REGIONS):', list(masks.keys()), flush=True)

# Exclude post-aggregation-only derived vars from the spatial aggregate loop.
_vars_to_agg = [v for v in variables_to_aggregate if v not in DERIVED_VAR_AFTER_AGG]

model_monthly = {
    region: {
        var: setup.aggregate_region(
            data_derived, var, region, masks, return_time_series=True,
            skipna=(var in LIFETIME_VARS),
        )
        for var in _vars_to_agg
    }
    for region in ANALYSIS_REGIONS
}

model_seasonal = {
    region: {
        var: setup.aggregate_region(
            data_derived, var, region, masks, return_time_series=False,
            skipna=(var in LIFETIME_VARS),
        )
        for var in _vars_to_agg
    }
    for region in ANALYSIS_REGIONS
}

setup.compute_derived_after_aggregation(
    model_monthly, model_seasonal, DERIVED_VAR_AFTER_AGG,
    use_deposition_for_lifetime=USE_DEPOSITION_FOR_LIFETIME,
)
print(f'\nPost-aggregation derived vars applied: {DERIVED_VAR_AFTER_AGG}', flush=True)


def apply_amazon_soa_aod(monthly_dict, seasonal_dict):
    """Inflate Amazon OA emissions so SOA is AMAZON_SOA_FRACTION of total OA.

    When deposition-based lifetime is active, only emissions are adjusted;
    lifetimes are not overwritten with an emission-based recompute.
    """
    if not INCLUDE_AMAZON_SOA:
        print('\nAmazon SOA: SKIPPED', flush=True)
        return
    f = AMAZON_SOA_FRACTION
    print('\n--- Amazon SOA adjustment ---', flush=True)
    if USE_DEPOSITION_FOR_LIFETIME:
        print('  USE_DEPOSITION_FOR_LIFETIME=True: emissions adjusted, '
              'deposition-based lifetime NOT overwritten.', flush=True)
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
                if model in emi_total:
                    emi_total[model] = emi_total[model] - oa + oa_total
                if model in emi_bcoa:
                    emi_bcoa[model] = bc + oa_total
                n += 1
            except Exception as e:
                print(f'  SOA adjust failed for {model}: {e}', flush=True)
        agg[region]['emi_total'] = emi_total
        agg[region]['emi_BC_OA'] = emi_bcoa
        if not USE_DEPOSITION_FOR_LIFETIME:
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
        print(f'  Adjusted Amazon emissions for {n} models', flush=True)


apply_amazon_soa_aod(model_monthly, model_seasonal)
setup.filter_lifetime_after_agg(
    model_monthly, model_seasonal,
    POST_AGG_LIFETIME_MAX_DAYS, POST_AGG_LIFETIME_MIN_DAYS,
)
_method_label = 'load/deposition' if USE_DEPOSITION_FOR_LIFETIME else 'load/emission'
if 'lifetime' in DERIVED_VAR_AFTER_AGG:
    setup.print_global_lifetimes(
        model_seasonal, var='lifetime',
        method_label=f'lifetime total ({_method_label}, post-filter)',
    )
if 'lifetime_BC_OA' in DERIVED_VAR_AFTER_AGG or 'lifetime' in DERIVED_VAR_AFTER_AGG:
    setup.print_global_lifetimes(
        model_seasonal, var='lifetime_BC_OA',
        method_label=f'lifetime_BC_OA ({_method_label}, post-filter)',
    )

# Diagnostic: emission vs deposition lifetimes (always both methods).
_compare_regions = [r for r in ANALYSIS_REGIONS if r in model_seasonal]
lifetime_method_df = setup.compare_emission_vs_deposition_lifetimes(
    model_seasonal,
    regions=_compare_regions,
    load_key='load_BC_OA',
    emi_key='emi_BC_OA',
    dep_key='dep_BC_OA',
    out_csv=TABLE_DIR / 'AOD_lifetime_emission_vs_deposition_BC_OA.csv',
)

plot_regions = list(ANALYSIS_REGIONS)

# -------------------------------------------------------------------------------
# 5. Load POLDER observations and compute monthly regional means
# -------------------------------------------------------------------------------
polder_path = project_root / 'Data' / 'AP3_POLDER_Collocated' / 'POLDER_GRASP_coloc_3h_AP3_2010_lon0_10.0_lat0_10.0.parquet'
print(f'\nLoading POLDER observations: {polder_path}', flush=True)

polder_cols = ['time', 'longitude', 'latitude', 'AOD_550', 'AAOD_550', 'AOD_440', 'AOD_870']
polder_df = pd.read_parquet(polder_path, columns=polder_cols)
print(f'POLDER rows: {len(polder_df)}', flush=True)

polder_df['time'] = pd.to_datetime(polder_df['time'])
polder_df['month'] = polder_df['time'].dt.to_period('M')

with np.errstate(divide='ignore', invalid='ignore'):
    polder_df['AE'] = -np.log(polder_df['AOD_550'] / polder_df['AOD_440']) / np.log(550.0 / 440.0)
    polder_df['SSA'] = 1.0 - polder_df['AAOD_550'] / polder_df['AOD_550']

polder_df = polder_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=['AOD_550', 'AE', 'SSA', 'AAOD_550']
)
print(f'POLDER rows after filtering: {len(polder_df)}', flush=True)


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


polder_monthly = {region: polder_monthly_means(polder_df, region) for region in ANALYSIS_REGIONS}


def _polder_subsample(region_name, max_pts=8000):
    """Return POLDER subsample for a region/time window."""
    cfg = REGIONS[region_name]
    mask = polder_region_mask(polder_df, cfg['lon_range'], cfg['lat_range'])
    sub = polder_df[mask].copy()
    t0, t1 = cfg['time_slice']
    sub = sub[(sub['time'] >= t0) & (sub['time'] <= t1)]
    if sub.empty:
        return sub
    if len(sub) > max_pts:
        sub = sub.sample(n=max_pts, random_state=0)
    return sub


def sample_model_field_at_polder(model_da, polder_sub, min_pts=3):
    """Vectorized sampling of model field at POLDER locations/times; weighted mean."""
    if model_da is None or polder_sub.empty or 'time' not in model_da.dims:
        return np.nan
    lat_vals = model_da.lat.values
    lon_vals = model_da.lon.values
    lat_idx = np.abs(lat_vals[:, None] - polder_sub['latitude'].values[None, :]).argmin(axis=0)
    lon_idx = np.abs(lon_vals[:, None] - polder_sub['longitude'].values[None, :]).argmin(axis=0)
    times = pd.to_datetime(model_da.time.values)
    polder_times = pd.to_datetime(polder_sub['time'].values)
    month_idx = np.array([
        np.argmin(np.abs(times - pd.Timestamp(t).to_period('M').to_timestamp()))
        for t in polder_times
    ])
    vals = model_da.values[month_idx, lat_idx, lon_idx]
    weights = np.cos(np.deg2rad(polder_sub['latitude'].values))
    finite = np.isfinite(vals)
    if finite.sum() < min_pts:
        return np.nan
    return float(np.average(vals[finite], weights=weights[finite]))


_SAMPLE_CACHE = {}


def sample_model_aod_at_polder(model_name, region_name):
    """Seasonal-mean model AOD sampled at POLDER observation locations (vectorized)."""
    key = ('AOD', model_name, region_name)
    if key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[key]
    if data[model_name].get('od550aer') is None:
        _SAMPLE_CACHE[key] = np.nan
        return np.nan
    cfg = REGIONS[region_name]
    da = data[model_name]['od550aer']['od550aer'].sel(time=slice(*cfg['time_slice']))
    sub = _polder_subsample(region_name)
    val = sample_model_field_at_polder(da, sub)
    _SAMPLE_CACHE[key] = val
    return val


def sample_model_ae_at_polder(model_name, region_name):
    """Seasonal-mean model AE sampled at POLDER observation locations (vectorized)."""
    key = ('AE', model_name, region_name)
    if key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[key]
    if model_name not in data_derived or data_derived[model_name].get('AE') is None:
        _SAMPLE_CACHE[key] = np.nan
        return np.nan
    cfg = REGIONS[region_name]
    da = data_derived[model_name]['AE'].sel(time=slice(*cfg['time_slice']))
    sub = _polder_subsample(region_name)
    val = sample_model_field_at_polder(da, sub)
    _SAMPLE_CACHE[key] = val
    return val


# -------------------------------------------------------------------------------
# 6. Load GPCP precipitation and compute monthly regional means
# -------------------------------------------------------------------------------
gpcp_path = project_root / 'Data' / 'Prec' / 'GPCP_2010_0-360.nc'
print(f'\nLoading GPCP precipitation: {gpcp_path}', flush=True)
gpcp = xr.open_dataset(gpcp_path)
gpcp_precip = gpcp['sat_gauge_precip']


def gpcp_region_mean(precip_da, region_name, apply_surface_mask=True):
    """Area-weighted regional mean of GPCP precipitation (mm day-1)."""
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
            print(f'  GPCP surface mask skip {region_name}: {e}', flush=True)
    coslat = np.cos(np.deg2rad(sub.lat))
    weights = coslat * xr.ones_like(sub.isel(time=0))
    weights = weights.where(np.isfinite(sub.isel(time=0)))
    wsum = weights.sum()
    if float(wsum) == 0:
        return sub.mean(dim=['lat', 'lon'])
    weights = weights / wsum
    return (sub * weights).sum(dim=['lat', 'lon'])


gpcp_region = {region: gpcp_region_mean(gpcp_precip, region) for region in ANALYSIS_REGIONS}



# -------------------------------------------------------------------------------
# 7. Dual-ensemble API functions
# -------------------------------------------------------------------------------
def homogenize_polder_aod(model_seasonal_dict, ens_models, regions, var='AOD_550'):
    """Zhong-style homogenization: regional_obs = (sampled_obs - b) / a."""
    print(f'\n--- POLDER {var} homogenization (n_models={len(ens_models)}) ---', flush=True)
    homogenized = {}
    for region in regions:
        if polder_monthly[region].empty:
            homogenized[region] = np.nan
            continue
        obs_col = 'AOD_550' if var == 'AOD_550' else 'AE'
        obs_sampled = float(polder_monthly[region][obs_col].mean())
        regional_key = 'od550aer' if var == 'AOD_550' else 'AE'
        regional = model_seasonal_dict[region].get(regional_key, {})
        sampled = {}
        for model in ens_models:
            if model not in regional:
                continue
            if model in MONTHLY_OUTPUT_MODELS:
                continue
            if var == 'AOD_550':
                s = sample_model_aod_at_polder(model, region)
            else:
                s = sample_model_ae_at_polder(model, region)
            if np.isfinite(s):
                sampled[model] = s
        common = [m for m in ens_models if m in regional and m in sampled and np.isfinite(float(regional[m]))]
        if len(common) < 3:
            print(f'  {region}: homogenization skipped (n={len(common)}); raw={obs_sampled:.4f}', flush=True)
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
        print(f'  {region}: sampled={a:.4f}*reg+{b:.4f} R2={r2:.3f} raw={obs_sampled:.4f} -> {homogenized_regional:.4f}',
              flush=True)
    return homogenized


def get_obs_constraints(model_seasonal_dict, ens_models):
    """Return aod_obs_by_region and ae_obs_by_region for an ensemble."""
    _homo_regions = [r for r in SOURCE_REGIONS if r in REGIONS]
    if POLDER_HOMOGENIZE:
        aod_obs = homogenize_polder_aod(model_seasonal, ens_models, _homo_regions, var='AOD_550')
        ae_obs = homogenize_polder_aod(model_seasonal, ens_models, _homo_regions, var='AE')
    else:
        print('\n--- POLDER homogenization DISABLED ---', flush=True)
        aod_obs = {
            region: float(polder_monthly[region]['AOD_550'].mean())
            if not polder_monthly[region].empty else np.nan
            for region in _homo_regions
        }
        ae_obs = {
            region: float(polder_monthly[region]['AE'].mean())
            if not polder_monthly[region].empty else np.nan
            for region in _homo_regions
        }
    if OUTFLOW_REGION in REGIONS and not polder_monthly[OUTFLOW_REGION].empty:
        aod_obs[OUTFLOW_REGION] = float(polder_monthly[OUTFLOW_REGION]['AOD_550'].mean())
        ae_obs[OUTFLOW_REGION] = float(polder_monthly[OUTFLOW_REGION]['AE'].mean())
    return aod_obs, ae_obs


def build_reg_df(model_list):
    """Build regression dataframe for a model subset.

    Rows are kept whenever lifetime + precip are finite so precip–1/τ plots
    are not limited by missing MEC/AE/AOD. Optics fields may be NaN; MEC and
    precip+AE fits use dropna on their own predictors.
    """
    rows = []
    for region in ANALYSIS_REGIONS:
        for model in model_list:
            lt = model_seasonal[region]['lifetime'].get(model)
            precip = model_seasonal[region]['precip'].get(model)
            if lt is None or precip is None:
                continue
            if not np.isfinite(float(lt)):
                continue
            try:
                precip_f = float(precip)
            except Exception:
                continue
            if not np.isfinite(precip_f):
                continue

            def _opt_float(v):
                if v is None:
                    return np.nan
                try:
                    x = float(v)
                    return x if np.isfinite(x) else np.nan
                except Exception:
                    return np.nan

            mec = _opt_float(model_seasonal[region]['MEC'].get(model))
            ae = _opt_float(model_seasonal[region]['AE'].get(model))
            aod = _opt_float(model_seasonal[region]['od550aer'].get(model))
            rows.append({
                'region': region, 'model': model,
                'MEC': mec, 'AE': ae,
                'lifetime': float(lt), 'inv_lifetime': 1.0 / float(lt),
                'precip': precip_f, 'AOD': aod,
            })
    return pd.DataFrame(rows)


def fit_mec_ae(reg_df):
    """Fit MEC vs AE per region; returns dict per region."""
    results = {}
    for region in ANALYSIS_REGIONS:
        sub = reg_df[reg_df['region'] == region].dropna(subset=['MEC', 'AE'])
        if len(sub) < 3:
            continue
        x = sub['AE'].values
        y = sub['MEC'].values
        if INTERCEPT_0:
            slope = np.nansum(x * y) / np.nansum(x ** 2)
            inter = 0.0
            residuals = y - slope * x
            r2 = 1 - np.nansum(residuals ** 2) / np.nansum(y ** 2)
            x_sampled = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            se = np.nanstd(residuals)
            fit = {
                'slope': slope, 'inter': inter, 'r2': r2, 'x_sampled': x_sampled,
                'ci_lower': slope * x_sampled - 1.96 * se,
                'ci_upper': slope * x_sampled + 1.96 * se, 'n': len(sub),
            }
        else:
            fit = functions.fit_data(x, y)
            fit['n'] = len(sub)
        results[region] = fit
    return results


def fit_inv_lt(reg_df):
    """Fit 1/tau vs precipitation + AE per region."""
    results = {}
    for region in ANALYSIS_REGIONS:
        sub = reg_df[reg_df['region'] == region].dropna(subset=['inv_lifetime', 'precip', 'AE'])
        if len(sub) < 4:
            continue
        X = np.column_stack([sub['precip'].values, sub['AE'].values])
        y = sub['inv_lifetime'].values
        X_ols = np.column_stack([np.ones(len(X)), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_ols, y, rcond=None)
        y_pred = X_ols @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        results[region] = {
            'intercept': coeffs[0], 'alpha_pr': coeffs[1], 'beta_ae': coeffs[2],
            'r2': r2, 'n': len(sub),
        }
    return results


def fit_inv_lt_simple(reg_df):
    """Simple 1/tau vs precipitation regression for plotting."""
    results = {}
    for region in ANALYSIS_REGIONS:
        sub = reg_df[reg_df['region'] == region].dropna(subset=['inv_lifetime', 'precip'])
        if len(sub) < 3:
            continue
        fit = functions.fit_data(sub['precip'].values, sub['inv_lifetime'].values)
        fit['n'] = len(sub)
        results[region] = fit
    return results


def compute_constrained(fits, model_list, aod_obs, ae_obs):
    """Compute constrained estimates from observations."""
    mec_ae = fits['mec_ae']
    inv_lt = fits['inv_lt']
    rows = []
    for region in SOURCE_REGIONS:
        if region not in mec_ae or region not in inv_lt:
            continue
        ae_obs_mean = float(ae_obs.get(region, np.nan))
        aod_obs_mean = float(aod_obs.get(region, np.nan))
        precip_obs_mean = float(gpcp_region[region].mean().values) if gpcp_region[region].size else np.nan
        mec_params = mec_ae[region]
        mec_c = mec_params['slope'] * ae_obs_mean + mec_params['inter']
        lt_params = inv_lt[region]
        inv_lt_c = lt_params['intercept'] + lt_params['alpha_pr'] * precip_obs_mean + lt_params['beta_ae'] * ae_obs_mean
        tau_c_days = 1.0 / inv_lt_c if inv_lt_c > 0 else np.nan
        e_c = aod_obs_mean / (tau_c_days * 86400.0 * mec_c) if (tau_c_days > 0 and mec_c > 0) else np.nan
        rows.append({
            'region': region,
            'AE_obs': ae_obs_mean,
            'AOD_obs': aod_obs_mean,
            'precip_obs': precip_obs_mean,
            'MEC_c': mec_c,
            'tau_c_days': tau_c_days,
            'E_c': e_c,
            'AOD_homogenized': bool(POLDER_HOMOGENIZE),
            'MEC_intercept_used': not INTERCEPT_0,
        })
    return pd.DataFrame(rows)


def compute_decomp(constrained_df, reg_df, model_list):
    """AOD error decomposition per model and source region."""
    rows = []
    for region in SOURCE_REGIONS:
        csub = constrained_df[constrained_df['region'] == region]
        if csub.empty:
            continue
        mec_c = csub['MEC_c'].mean()
        tau_c_days = csub['tau_c_days'].mean()
        aod_c = csub['AOD_obs'].mean()
        e_c = csub['E_c'].mean()
        for model in model_list:
            mec_model = model_seasonal[region]['MEC'].get(model)
            lt_model = model_seasonal[region]['lifetime'].get(model)
            emi_model = model_seasonal[region]['emi_total'].get(model)
            aod_model = model_seasonal[region]['od550aer'].get(model)
            if any(v is None for v in [mec_model, lt_model, emi_model, aod_model]):
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
            rows.append({
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
    return pd.DataFrame(rows)


def compute_outflow(constrained_df, model_list, aod_obs):
    """African outflow meta-model (Eq. 6): Default / EC / MFC."""
    meta_rows = []
    for model in model_list:
        aod_out = model_seasonal[OUTFLOW_REGION].get('od550aer', {}).get(model)
        e_src = model_seasonal[OUTFLOW_SOURCE].get('emi_total', {}).get(model)
        tau_src = model_seasonal[OUTFLOW_SOURCE].get('lifetime', {}).get(model)
        mec_src = model_seasonal[OUTFLOW_SOURCE].get('MEC', {}).get(model)
        if not all(v is not None and np.isfinite(float(v)) for v in [aod_out, e_src, tau_src, mec_src]):
            continue
        e_src, tau_src, mec_src = float(e_src), float(tau_src), float(mec_src)
        burden_term = e_src * (tau_src * 86400.0)
        etm = burden_term * mec_src
        meta_rows.append({
            'model': model, 'AOD_out': float(aod_out),
            'E_tau_MEC': etm, 'E_tau': burden_term, 'MEC': mec_src,
            'E': e_src, 'tau': tau_src,
        })
    meta_df = pd.DataFrame(meta_rows)
    params = None
    pred_df = pd.DataFrame()
    if len(meta_df) < 5:
        return params, pred_df
    X = np.column_stack([
        meta_df['E_tau_MEC'].values, meta_df['E_tau'].values,
        meta_df['MEC'].values, np.ones(len(meta_df)),
    ])
    y = meta_df['AOD_out'].values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    nmb = 100.0 * np.mean(y_pred - y) / np.mean(y) if np.mean(y) != 0 else np.nan
    rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    params = {
        'a': float(coeffs[0]), 'b': float(coeffs[1]),
        'c': float(coeffs[2]), 'd': float(coeffs[3]),
        'r2': r2, 'nmb': nmb, 'rmse': rmse, 'n': len(meta_df),
    }
    c_src = constrained_df[constrained_df['region'] == OUTFLOW_SOURCE]
    aod_out_obs = float(aod_obs.get(OUTFLOW_REGION, np.nan))
    pred_rows = []
    if not c_src.empty:
        e_c = float(c_src['E_c'].mean())
        tau_c = float(c_src['tau_c_days'].mean())
        mec_c = float(c_src['MEC_c'].mean())
        burden_c = e_c * tau_c * 86400.0
        aod_mfc = coeffs[0] * burden_c * mec_c + coeffs[1] * burden_c + coeffs[2] * mec_c + coeffs[3]
        for _, row in meta_df.iterrows():
            aod_src_model = model_seasonal[OUTFLOW_SOURCE]['od550aer'].get(row['model'])
            aod_src_obs = float(aod_obs.get(OUTFLOW_SOURCE, np.nan))
            if aod_src_model is None or not np.isfinite(aod_src_obs) or float(aod_src_model) == 0:
                e_ec = row['E']
            else:
                e_ec = row['E'] * (aod_src_obs / float(aod_src_model))
            burden_ec = e_ec * row['tau'] * 86400.0
            aod_ec = coeffs[0] * burden_ec * row['MEC'] + coeffs[1] * burden_ec + coeffs[2] * row['MEC'] + coeffs[3]
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
        pred_df = pd.DataFrame(pred_rows).sort_values('model').reset_index(drop=True)
    return params, pred_df


def loo_validation(reg_df):
    """Per-region leave-one-out validation; pool pairs across SOURCE_REGIONS.

    Matches Zhong et al. 2022 Fig. 2c/d: regressions are fit within each BB
    region, then predicted vs modelled points from all regions are pooled for
    R / NMB / RMSE.
    """
    rows = []
    src = reg_df[reg_df['region'].isin(SOURCE_REGIONS)].copy()
    for region, sub_mec in src.dropna(subset=['MEC', 'AE']).groupby('region'):
        if len(sub_mec) < 4:
            continue
        for i, left_out in sub_mec.iterrows():
            train = sub_mec.drop(index=i)
            if INTERCEPT_0:
                x = train['AE'].values
                y = train['MEC'].values
                slope = np.nansum(x * y) / np.nansum(x ** 2)
                inter = 0.0
            else:
                fit = functions.fit_data(train['AE'].values, train['MEC'].values)
                slope, inter = fit['slope'], fit['inter']
            pred = slope * left_out['AE'] + inter
            rows.append({
                'variable': 'MEC', 'region': region, 'model': left_out['model'],
                'modelled': float(left_out['MEC']), 'predicted': float(pred),
                'left_out': f"{region}/{left_out['model']}",
            })
    for region, sub_lt in src.dropna(subset=['inv_lifetime', 'precip', 'AE']).groupby('region'):
        if len(sub_lt) < 5:
            continue
        for i, left_out in sub_lt.iterrows():
            train = sub_lt.drop(index=i)
            X = np.column_stack([train['precip'].values, train['AE'].values])
            y = train['inv_lifetime'].values
            X_ols = np.column_stack([np.ones(len(X)), X])
            coeffs, _, _, _ = np.linalg.lstsq(X_ols, y, rcond=None)
            pred = coeffs[0] + coeffs[1] * left_out['precip'] + coeffs[2] * left_out['AE']
            rows.append({
                'variable': 'inv_lifetime', 'region': region, 'model': left_out['model'],
                'modelled': float(left_out['inv_lifetime']), 'predicted': float(pred),
                'left_out': f"{region}/{left_out['model']}",
            })
    return pd.DataFrame(rows)


def compute_loo_metrics(loo_df, variable):
    """Pearson R, NMB, RMSE for LOO validation (Zhong Fig. 2c/d uses R not R²)."""
    sub = loo_df[loo_df['variable'] == variable].dropna(subset=['modelled', 'predicted'])
    if sub.empty:
        return {'r': np.nan, 'r2': np.nan, 'nmb': np.nan, 'rmse': np.nan, 'n': 0}
    y = sub['modelled'].values.astype(float)
    yp = sub['predicted'].values.astype(float)
    if len(y) < 2 or np.nanstd(y) == 0 or np.nanstd(yp) == 0:
        r = np.nan
    else:
        r = float(np.corrcoef(y, yp)[0, 1])
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    nmb = 100.0 * np.mean(yp - y) / np.mean(y) if np.mean(y) != 0 else np.nan
    rmse = float(np.sqrt(np.mean((yp - y) ** 2)))
    return {'r': r, 'r2': r2, 'nmb': nmb, 'rmse': rmse, 'n': len(sub)}


def build_table1_comparison(constrained_df, ensemble):
    """Compare constrained estimates with PAPER_TABLE1."""
    rows = []
    for region in SOURCE_REGIONS:
        paper = PAPER_TABLE1.get(region, {})
        csub = constrained_df[constrained_df['region'] == region]
        if csub.empty:
            continue
        row = csub.iloc[0]
        for var, paper_key, scale, display_key in [
            ('precip_obs', 'precip', 1.0, 'precip'),
            ('AE_obs', 'AE', 1.0, 'AE'),
            ('AOD_obs', 'AOD', 1.0, 'AOD'),
            ('E_c', 'E', 1e11, 'E (10^-11)'),
            ('tau_c_days', 'tau', 1.0, 'tau'),
            ('MEC_c', 'MEC', 1.0 / 1000.0, 'MEC (m2/g)'),
        ]:
            computed = float(row[var]) * scale if var == 'MEC_c' else (
                float(row[var]) * scale if var == 'E_c' else float(row[var])
            )
            paper_val = paper.get(paper_key if paper_key != 'E' else 'E', np.nan)
            if var == 'E_c':
                paper_val = paper.get('E', np.nan)
            gap = computed - paper_val
            gap_pct = 100.0 * gap / paper_val if paper_val not in (0, np.nan) and np.isfinite(paper_val) else np.nan
            rows.append({
                'ensemble': ensemble, 'region': region, 'variable': display_key,
                'computed': computed, 'paper': paper_val,
                'gap': gap, 'gap_pct': gap_pct,
            })
    return pd.DataFrame(rows)


def robustness_drop_extremes(reg_df, constrained_df, model_list, aod_obs, ae_obs, n_drop=2):
    """Drop n_drop extreme precip and AE models; recompute constrained."""
    rows = []
    base = compute_constrained(
        {'mec_ae': fit_mec_ae(reg_df), 'inv_lt': fit_inv_lt(reg_df)},
        model_list, aod_obs, ae_obs,
    )
    for region in SOURCE_REGIONS:
        sub = reg_df[reg_df['region'] == region]
        if sub.empty:
            continue
        for label, col in [('drop_precip_high', 'precip'), ('drop_precip_low', 'precip'),
                           ('drop_AE_high', 'AE'), ('drop_AE_low', 'AE')]:
            ascending = 'low' in label
            drop_models = sub.sort_values(col, ascending=ascending)['model'].head(n_drop).tolist()
            sub_reg = reg_df[~((reg_df['region'] == region) & (reg_df['model'].isin(drop_models)))]
            fits = {'mec_ae': fit_mec_ae(sub_reg), 'inv_lt': fit_inv_lt(sub_reg)}
            cnew = compute_constrained(fits, model_list, aod_obs, ae_obs)
            creg = cnew[cnew['region'] == region]
            if creg.empty:
                continue
            brow = creg.iloc[0]
            brow_base = base[base['region'] == region]
            if brow_base.empty:
                continue
            brow_base = brow_base.iloc[0]
            rows.append({
                'region': region, 'test': label, 'n_drop': n_drop,
                'dMEC_c': brow['MEC_c'] - brow_base['MEC_c'],
                'dTau_c': brow['tau_c_days'] - brow_base['tau_c_days'],
                'dE_c': brow['E_c'] - brow_base['E_c'],
            })
    return pd.DataFrame(rows)



# -------------------------------------------------------------------------------
# 8. Dual-ensemble main loop
# -------------------------------------------------------------------------------
results = {}
all_table1 = []
all_loo = []
all_robustness = []
all_reg = []
all_constrained = []
all_decomp = []
all_outflow = []

for ens_name, ens_models in ENSEMBLES.items():
    print(f'\n{"=" * 72}\nENSEMBLE: {ens_name} ({len(ens_models)} models)\n{"=" * 72}', flush=True)
    aod_obs, ae_obs = get_obs_constraints(model_seasonal, ens_models)
    reg_df_ens = build_reg_df(ens_models)
    mec_ae = fit_mec_ae(reg_df_ens)
    inv_lt = fit_inv_lt(reg_df_ens)
    inv_lt_simple = fit_inv_lt_simple(reg_df_ens)
    fits = {'mec_ae': mec_ae, 'inv_lt': inv_lt, 'inv_lt_simple': inv_lt_simple}
    constrained_df_ens = compute_constrained(fits, ens_models, aod_obs, ae_obs)
    decomp_df_ens = compute_decomp(constrained_df_ens, reg_df_ens, ens_models)
    outflow_params, outflow_pred_df = compute_outflow(constrained_df_ens, ens_models, aod_obs)
    loo_df_ens = loo_validation(reg_df_ens)
    table1_df = build_table1_comparison(constrained_df_ens, ens_name)
    robust_df = robustness_drop_extremes(reg_df_ens, constrained_df_ens, ens_models, aod_obs, ae_obs)

    reg_df_ens = reg_df_ens.copy()
    reg_df_ens['ensemble'] = ens_name
    constrained_df_ens = constrained_df_ens.copy()
    constrained_df_ens['ensemble'] = ens_name
    decomp_df_ens = decomp_df_ens.copy()
    decomp_df_ens['ensemble'] = ens_name
    if not outflow_pred_df.empty:
        outflow_pred_df = outflow_pred_df.copy()
        outflow_pred_df['ensemble'] = ens_name
    loo_df_ens = loo_df_ens.copy()
    loo_df_ens['ensemble'] = ens_name
    if not robust_df.empty:
        robust_df = robust_df.copy()
        robust_df['ensemble'] = ens_name

    results[ens_name] = {
        'models': ens_models,
        'aod_obs': aod_obs,
        'ae_obs': ae_obs,
        'reg_df': reg_df_ens,
        'mec_ae': mec_ae,
        'inv_lt': inv_lt,
        'inv_lt_simple': inv_lt_simple,
        'constrained_df': constrained_df_ens,
        'decomp_df': decomp_df_ens,
        'outflow_params': outflow_params,
        'outflow_pred_df': outflow_pred_df,
        'loo_df': loo_df_ens,
        'table1_df': table1_df,
        'robust_df': robust_df,
    }

    all_reg.append(reg_df_ens)
    all_constrained.append(constrained_df_ens)
    all_decomp.append(decomp_df_ens)
    if not outflow_pred_df.empty:
        all_outflow.append(outflow_pred_df)
    all_loo.append(loo_df_ens)
    all_table1.append(table1_df)
    if not robust_df.empty:
        all_robustness.append(robust_df)

    print(constrained_df_ens.set_index('region')[['MEC_c', 'tau_c_days', 'E_c']], flush=True)

# Back-compat aliases from 'full' ensemble
_full = results['full']
reg_df = _full['reg_df']
mec_ae_results = _full['mec_ae']
inv_lt_results = _full['inv_lt']
inv_lt_simple = _full['inv_lt_simple']
constrained_df = _full['constrained_df']
decomp_df = _full['decomp_df']
outflow_meta_params = _full['outflow_params']
outflow_pred_df = _full['outflow_pred_df']
aod_obs_by_region = _full['aod_obs']
ae_obs_by_region = _full['ae_obs']

combined_reg_df = pd.concat(all_reg, ignore_index=True) if all_reg else pd.DataFrame()
combined_constrained_df = pd.concat(all_constrained, ignore_index=True) if all_constrained else pd.DataFrame()
combined_decomp_df = pd.concat(all_decomp, ignore_index=True) if all_decomp else pd.DataFrame()
combined_loo_df = pd.concat(all_loo, ignore_index=True) if all_loo else pd.DataFrame()
combined_table1_df = pd.concat(all_table1, ignore_index=True) if all_table1 else pd.DataFrame()
combined_robust_df = pd.concat(all_robustness, ignore_index=True) if all_robustness else pd.DataFrame()
combined_outflow_df = pd.concat(all_outflow, ignore_index=True) if all_outflow else pd.DataFrame()

region_colors = {r: REGION_COLORS.get(r, plt.cm.tab10(i / max(len(plot_regions), 1)))
                 for i, r in enumerate(plot_regions)}

# -------------------------------------------------------------------------------
# 9. Figure 2: linear regressions (SI + extended, both ensembles)
# -------------------------------------------------------------------------------
print('\n--- Figure 2: regressions (both ensembles) ---', flush=True)


def plot_figure2_si(ens_name, res, suffix):
    """SI-style two-panel regression figure."""
    reg_df_e = res['reg_df']
    mec_ae = res['mec_ae']
    inv_lt_simple_e = res['inv_lt_simple']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    for region in plot_regions:
        if region not in mec_ae:
            continue
        sub = reg_df_e[reg_df_e['region'] == region].dropna(subset=['MEC', 'AE'])
        if sub.empty:
            continue
        fit = mec_ae[region]
        x_line = fit['x_sampled']
        ax.scatter(sub['AE'], sub['MEC'] / 1000.0, s=40, alpha=0.7,
                   color=region_colors[region], label=region, edgecolors='none')
        ax.plot(x_line, (fit['slope'] * x_line + fit['inter']) / 1000.0,
                '-', color=region_colors[region], lw=2)
        ax.fill_between(x_line, fit['ci_lower'] / 1000.0, fit['ci_upper'] / 1000.0,
                        color=region_colors[region], alpha=0.15)
    ax.set_xlabel('Angstrom Exponent', fontweight='bold')
    ax.set_ylabel('MEC (m2 g-1)', fontweight='bold')
    ax.set_title('(a) MEC vs AE', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    ax = axes[1]
    for region in plot_regions:
        if region not in inv_lt_simple_e:
            continue
        sub = reg_df_e[reg_df_e['region'] == region].dropna(subset=['inv_lifetime', 'precip'])
        if sub.empty:
            continue
        fit = inv_lt_simple_e[region]
        x_line = fit['x_sampled']
        ax.scatter(sub['precip'], sub['inv_lifetime'], s=40, alpha=0.7,
                   color=region_colors[region], label=region, edgecolors='none')
        ax.plot(x_line, fit['slope'] * x_line + fit['inter'], '-', color=region_colors[region], lw=2)
        ax.fill_between(x_line, fit['ci_lower'], fit['ci_upper'], color=region_colors[region], alpha=0.15)
    ax.set_xlabel('Precipitation (mm day-1)', fontweight='bold')
    ax.set_ylabel('1 / lifetime (day-1)', fontweight='bold')
    ax.set_title('(b) 1/tau vs precipitation', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    plt.suptitle(f'Figure 2 SI style ({ens_name} ensemble)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = project_root / 'notebooks' / f'AOD_error_attribution_figure2_si_style_{suffix}.png'
    if SAVE_FIGURE:
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {fig_path}', flush=True)
        plt.close(fig)
    if _IN_IPYTHON:
        plt.show()
    else:
        plt.close(fig)


def plot_figure2_extended(ens_name, res, suffix):
    """Extended 2x2 validation panels."""
    reg_df_e = res['reg_df']
    mec_ae = res['mec_ae']
    inv_lt_e = res['inv_lt']
    inv_lt_simple_e = res['inv_lt_simple']
    constrained_df_e = res['constrained_df']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    for region in plot_regions:
        if region not in mec_ae:
            continue
        sub = reg_df_e[reg_df_e['region'] == region].dropna(subset=['MEC', 'AE'])
        if sub.empty:
            continue
        fit = mec_ae[region]
        x_line = fit['x_sampled']
        ax.scatter(sub['AE'], sub['MEC'] / 1000.0, s=30, alpha=0.6, color=region_colors[region], label=region)
        ax.plot(x_line, (fit['slope'] * x_line + fit['inter']) / 1000.0,
                '--', color=region_colors[region], lw=2)
        csub = constrained_df_e[constrained_df_e['region'] == region]
        if not csub.empty:
            ax.axvline(csub['AE_obs'].mean(), color=region_colors[region], ls=':', alpha=0.7)
            ax.axhline(csub['MEC_c'].mean() / 1000.0, color=region_colors[region], ls='-.', alpha=0.7)
    ax.set_xlabel('Angstrom Exponent', fontweight='bold')
    ax.set_ylabel('MEC (m2 g-1)', fontweight='bold')
    ax.set_title('(a) MEC vs AE', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for region in plot_regions:
        if region not in inv_lt_simple_e:
            continue
        sub = reg_df_e[reg_df_e['region'] == region].dropna(subset=['inv_lifetime', 'precip'])
        if sub.empty:
            continue
        fit = inv_lt_simple_e[region]
        x_line = fit['x_sampled']
        ax.scatter(sub['precip'], sub['inv_lifetime'], s=40, alpha=0.7,
                   color=region_colors[region], label=region, edgecolors='none')
        ax.plot(x_line, fit['slope'] * x_line + fit['inter'], '--', color=region_colors[region], lw=2)
        csub = constrained_df_e[constrained_df_e['region'] == region]
        if not csub.empty and np.isfinite(csub['tau_c_days'].mean()) and csub['tau_c_days'].mean() > 0:
            ax.axvline(csub['precip_obs'].mean(), color=region_colors[region], ls=':', alpha=0.6)
            ax.axhline(1.0 / csub['tau_c_days'].mean(), color=region_colors[region], ls='-.', alpha=0.6)
    ax.set_xlabel('Precipitation (mm day-1)', fontweight='bold')
    ax.set_ylabel('1 / lifetime (day-1)', fontweight='bold')
    ax.set_title('(b) 1/tau vs precipitation', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best')

    ax = axes[1, 0]
    for region in plot_regions:
        if region not in mec_ae:
            continue
        sub = reg_df_e[reg_df_e['region'] == region].dropna(subset=['AE', 'MEC'])
        if sub.empty:
            continue
        fit = mec_ae[region]
        pred = fit['slope'] * sub['AE'].values + fit['inter']
        ax.scatter(pred / 1000.0, sub['MEC'].values / 1000.0, color=region_colors[region], label=region, alpha=0.6)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('Predicted MEC (m2 g-1)', fontweight='bold')
    ax.set_ylabel('Model MEC (m2 g-1)', fontweight='bold')
    ax.set_title('(c) MEC regression validation', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for region in plot_regions:
        if region not in inv_lt_e:
            continue
        sub = reg_df_e[reg_df_e['region'] == region].dropna(subset=['inv_lifetime', 'precip', 'AE'])
        if sub.empty:
            continue
        fit = inv_lt_e[region]
        pred = fit['intercept'] + fit['alpha_pr'] * sub['precip'].values + fit['beta_ae'] * sub['AE'].values
        ax.scatter(pred, sub['inv_lifetime'].values, color=region_colors[region], label=region, alpha=0.6)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('Predicted 1/tau (day-1)', fontweight='bold')
    ax.set_ylabel('Model 1/tau (day-1)', fontweight='bold')
    ax.set_title('(d) Lifetime regression validation', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.suptitle(f'Figure 2 extended ({ens_name} ensemble)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_path = project_root / 'notebooks' / f'AOD_error_attribution_figure2_{suffix}.png'
    if SAVE_FIGURE:
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {fig_path}', flush=True)
        plt.close(fig)
    if _IN_IPYTHON:
        plt.show()
    else:
        plt.close(fig)


for ens_name, res in results.items():
    plot_figure2_si(ens_name, res, suffix=ens_name)
    plot_figure2_extended(ens_name, res, suffix=ens_name)

# -------------------------------------------------------------------------------
# 10. Figure 2c/d LOO validation (both ensembles)
# -------------------------------------------------------------------------------
print('\n--- Figure 2c/d LOO validation ---', flush=True)


def plot_loo_validation(ens_name, res, suffix):
    loo_df_e = res['loo_df']
    if loo_df_e.empty:
        print(f'  {ens_name}: no LOO data', flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, variable, ylabel, paper_r in zip(
        axes,
        ['MEC', 'inv_lifetime'],
        ['MEC (m2 kg-1)', '1/tau (day-1)'],
        [PAPER_LOO_R['MEC'], PAPER_LOO_R['inv_lifetime']],
    ):
        sub = loo_df_e[loo_df_e['variable'] == variable].dropna(subset=['modelled', 'predicted'])
        if sub.empty:
            ax.set_visible(False)
            continue
        y = sub['modelled'].values
        yp = sub['predicted'].values
        if variable == 'MEC':
            y, yp = y / 1000.0, yp / 1000.0
            ylabel = 'MEC (m2 g-1)'
            # Metrics on paper units (m2 g-1) so RMSE matches Fig. 2c scale
            metrics_df = loo_df_e.copy()
            msub = metrics_df['variable'] == 'MEC'
            metrics_df.loc[msub, 'modelled'] = metrics_df.loc[msub, 'modelled'] / 1000.0
            metrics_df.loc[msub, 'predicted'] = metrics_df.loc[msub, 'predicted'] / 1000.0
            metrics = compute_loo_metrics(metrics_df, variable)
        else:
            metrics = compute_loo_metrics(loo_df_e, variable)
        for region in sub['region'].unique():
            rsub = sub[sub['region'] == region]
            yy = rsub['modelled'].values
            yyp = rsub['predicted'].values
            if variable == 'MEC':
                yy, yyp = yy / 1000.0, yyp / 1000.0
            ax.scatter(yyp, yy, color=region_colors.get(region, 'gray'), label=region, alpha=0.7, s=35)
        lims = [min(y.min(), yp.min()), max(y.max(), yp.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel(f'LOO predicted {ylabel}', fontweight='bold')
        ax.set_ylabel(f'Modelled {ylabel}', fontweight='bold')
        ax.set_title(
            f'{variable}: R={metrics["r"]:.3f} (paper {paper_r:.2f}), '
            f'NMB={metrics["nmb"]:.1f}%, RMSE={metrics["rmse"]:.4g}, n={metrics["n"]}',
            fontweight='bold', fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    plt.suptitle(f'LOO validation ({ens_name} ensemble)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = project_root / 'notebooks' / f'AOD_error_attribution_figure2_loo_{suffix}.png'
    if SAVE_FIGURE:
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {fig_path}', flush=True)
        plt.close(fig)
    if _IN_IPYTHON:
        plt.show()
    else:
        plt.close(fig)


for ens_name, res in results.items():
    plot_loo_validation(ens_name, res, suffix=ens_name)

# -------------------------------------------------------------------------------
# 11. Table 1 comparison and gap figures
# -------------------------------------------------------------------------------
print('\n--- Table 1 comparison ---', flush=True)
if not combined_table1_df.empty:
    print(combined_table1_df.pivot_table(index=['region', 'variable'], columns='ensemble', values='computed'),
          flush=True)


def plot_gap_figures(table1_df, ens_name, suffix):
    """Gap bar charts per region when |gap_pct| > GAP_THRESHOLD_PCT."""
    for region in SOURCE_REGIONS:
        sub = table1_df[table1_df['region'] == region]
        flagged = sub[sub['gap_pct'].abs() > GAP_THRESHOLD_PCT]
        if flagged.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(flagged))
        ax.bar(x, flagged['gap_pct'], color=[region_colors.get(region, 'gray')] * len(flagged))
        ax.axhline(GAP_THRESHOLD_PCT, color='red', ls='--', label=f'threshold +/-{GAP_THRESHOLD_PCT}%')
        ax.axhline(-GAP_THRESHOLD_PCT, color='red', ls='--')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(flagged['variable'], rotation=45, ha='right')
        ax.set_ylabel('Gap vs paper (%)', fontweight='bold')
        ax.set_title(f'{region}: constrained estimate gaps ({ens_name})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig_path = project_root / 'notebooks' / f'AOD_error_attribution_gap_{region}_{suffix}.png'
        if SAVE_FIGURE:
            fig.savefig(fig_path, dpi=200, bbox_inches='tight')
            print(f'Saved: {fig_path}', flush=True)
            plt.close(fig)
        if _IN_IPYTHON:
            plt.show()
        else:
            plt.close(fig)


for ens_name, res in results.items():
    plot_gap_figures(res['table1_df'], ens_name, suffix=ens_name)



# -------------------------------------------------------------------------------
# 12. Figure 4: stacked-bar AOD error attribution (both ensembles)
# -------------------------------------------------------------------------------
print('\n--- Figure 4: AOD error attribution ---', flush=True)


def plot_figure4(ens_name, res, suffix):
    decomp_df_e = res['decomp_df']
    fig_regions = [r for r in SOURCE_REGIONS if not decomp_df_e[decomp_df_e['region'] == r].empty]
    if not fig_regions:
        print(f'  {ens_name}: no decomposition regions; skipping Figure 4', flush=True)
        return
    n_regions = len(fig_regions)
    ncols = int(np.ceil(n_regions / 2))
    fig, axes = plt.subplots(2, ncols, figsize=(7 * ncols, 10))
    axes_flat = np.atleast_1d(axes).flatten()
    idx = -1
    components = [
        ('dAOD_E', 'Emission', '#1f77b4'),
        ('dAOD_tau', 'Lifetime', '#ff7f0e'),
        ('dAOD_MEC', 'MEC', '#2ca02c'),
        ('dAOD_cross', 'Cross', '#d62728'),
    ]
    for idx, region in enumerate(fig_regions):
        ax = axes_flat[idx]
        sub = decomp_df_e[decomp_df_e['region'] == region].sort_values('model').reset_index(drop=True)
        if sub.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(sub))
        pos_bottom = np.zeros(len(sub))
        neg_bottom = np.zeros(len(sub))
        for col, label, color in components:
            vals = sub[col].to_numpy(dtype=float)
            bottoms = np.where(vals >= 0, pos_bottom, neg_bottom)
            ax.bar(x, vals, bottom=bottoms, color=color, width=0.8, label=label)
            pos_bottom = np.where(vals >= 0, pos_bottom + vals, pos_bottom)
            neg_bottom = np.where(vals < 0, neg_bottom + vals, neg_bottom)
        ax.scatter(x, sub['dAOD_total'], color='black', s=45, zorder=5, label='Total')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(np.arange(1, len(sub) + 1), fontsize=10)
        ax.set_ylabel('AOD error contribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model number (alphabetical)', fontsize=12, fontweight='bold')
        ax.set_title(region.upper(), fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f'Figure 4: AOD error attribution ({ens_name})', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_path = project_root / 'notebooks' / f'AOD_error_attribution_figure4_{suffix}.png'
    if SAVE_FIGURE:
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {fig_path}', flush=True)
        plt.close(fig)
    if _IN_IPYTHON:
        plt.show()
    else:
        plt.close(fig)


for ens_name, res in results.items():
    plot_figure4(ens_name, res, suffix=ens_name)

# Mean pct comparison: paper vs full vs paper ensemble
mean_rows = []
for ens_name, res in results.items():
    decomp_df_e = res['decomp_df']
    if decomp_df_e.empty:
        continue
    m = decomp_df_e.groupby('region')[['pct_E', 'pct_tau', 'pct_MEC', 'pct_cross']].mean()
    m['source'] = ens_name
    mean_rows.append(m.reset_index())
paper_mean = pd.DataFrame([
    {'region': r, 'pct_E': PAPER_DECOMP_PCT['pct_E'], 'pct_tau': PAPER_DECOMP_PCT['pct_tau'],
     'pct_MEC': PAPER_DECOMP_PCT['pct_MEC'], 'pct_cross': PAPER_DECOMP_PCT['pct_cross'], 'source': 'paper_target'}
    for r in SOURCE_REGIONS
])
if mean_rows:
    mean_decomp_all = pd.concat(mean_rows + [paper_mean], ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_regions_src = [r for r in SOURCE_REGIONS if r in mean_decomp_all['region'].unique()]
    width = 0.25
    x = np.arange(len(plot_regions_src))
    sources = ['paper_target', 'full', 'paper']
    colors_src = {'paper_target': '#cccccc', 'full': '#1f77b4', 'paper': '#ff7f0e'}
    for i, src in enumerate(sources):
        sub = mean_decomp_all[mean_decomp_all['source'] == src].set_index('region').reindex(plot_regions_src)
        bottom = np.zeros(len(plot_regions_src))
        for col, color in zip(['pct_E', 'pct_tau', 'pct_MEC', 'pct_cross'],
                              ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
            vals = sub[col].fillna(0).values
            ax.bar(x + (i - 1) * width, vals, width=width, bottom=bottom, color=color,
                   alpha=0.85 if src != 'paper_target' else 0.4, label=f'{col} ({src})' if i == 0 else None)
            bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(plot_regions_src, rotation=45, ha='right')
    ax.set_ylabel('Mean contribution to |AOD error| (%)', fontweight='bold')
    ax.set_title('Figure 4 summary: mean pct (paper target vs full vs paper ensemble)', fontweight='bold')
    plt.tight_layout()
    fig_path2 = project_root / 'notebooks' / 'AOD_error_attribution_figure4_mean_pct_compare.png'
    if SAVE_FIGURE:
        fig.savefig(fig_path2, dpi=300, bbox_inches='tight')
        print(f'Saved: {fig_path2}', flush=True)
        plt.close(fig)
    if _IN_IPYTHON:
        plt.show()
    else:
        plt.close(fig)

# -------------------------------------------------------------------------------
# 13. Figure 5: African outflow Default / EC / MFC (both ensembles)
# -------------------------------------------------------------------------------
print('\n--- Figure 5: African outflow ---', flush=True)


def plot_figure5(ens_name, res, suffix):
    outflow_pred = res['outflow_pred_df']
    outflow_params = res['outflow_params']
    if outflow_pred.empty:
        print(f'  {ens_name}: no outflow predictions', flush=True)
        return
    sub = outflow_pred.sort_values('model').reset_index(drop=True)
    x = np.arange(len(sub))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(sub)), 6))
    ax.bar(x - width, sub['AOD_default'], width=width, color='#2ca02c', label='Default', alpha=0.9)
    ax.bar(x, sub['AOD_EC'], width=width, color='#ff7f0e', label='EC', alpha=0.9)
    ax.bar(x + width, [sub['AOD_MFC'].iloc[0]] * len(sub), width=width, color='#1f77b4', label='MFC', alpha=0.9)
    ax.axhline(sub['AOD_obs'].iloc[0], color='red', linestyle=':', linewidth=2, label='POLDER')
    ax.axhline(sub['AOD_default'].median(), color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(sub['AOD_EC'].median(), color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(np.arange(1, len(sub) + 1))
    ax.set_xlabel('Model number (alphabetical)', fontweight='bold')
    ax.set_ylabel('Africa outflow AOD', fontweight='bold')
    r2txt = outflow_params['r2'] if outflow_params else np.nan
    ax.set_title(f'Figure 5: Outflow AOD ({ens_name}, meta R2={r2txt:.3f})', fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig_path = project_root / 'notebooks' / f'AOD_error_attribution_figure5_outflow_{suffix}.png'
    if SAVE_FIGURE:
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {fig_path}', flush=True)
        plt.close(fig)
    if _IN_IPYTHON:
        plt.show()
    else:
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(sub['AOD_default'], sub['AOD_meta_fit'], s=40, alpha=0.8)
    lims = [
        min(sub['AOD_default'].min(), sub['AOD_meta_fit'].min()) * 0.9,
        max(sub['AOD_default'].max(), sub['AOD_meta_fit'].max()) * 1.1,
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('Default outflow AOD', fontweight='bold')
    ax.set_ylabel('Meta-model fitted AOD', fontweight='bold')
    ax.set_title(f'(a) Meta-model fit (R2={r2txt:.3f})', fontweight='bold')
    ax.grid(True, alpha=0.3)

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
    plt.suptitle(f'African outflow meta-model ({ens_name})', fontweight='bold')
    plt.tight_layout()
    fig_path2 = project_root / 'notebooks' / f'AOD_error_attribution_figure5_meta_{suffix}.png'
    if SAVE_FIGURE:
        fig.savefig(fig_path2, dpi=300, bbox_inches='tight')
        print(f'Saved: {fig_path2}', flush=True)
        plt.close(fig)
    if _IN_IPYTHON:
        plt.show()
    else:
        plt.close(fig)


for ens_name, res in results.items():
    plot_figure5(ens_name, res, suffix=ens_name)

# -------------------------------------------------------------------------------
# 14. Regional map (once)
# -------------------------------------------------------------------------------
print('\n--- Regional map ---', flush=True)
region_boxes = {
    name: (
        REGIONS[name]['lon_range'][0], REGIONS[name]['lon_range'][1],
        REGIONS[name]['lat_range'][0], REGIONS[name]['lat_range'][1],
    )
    for name in ANALYSIS_REGIONS
}
sample_field = data[sample_model]['od550aer']['od550aer'].isel(time=0)
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
    print(f'Saved regional map: {save_path}', flush=True)
    plt.close(fig_map)
if _IN_IPYTHON:
    pass  # display fig_map in notebook

# -------------------------------------------------------------------------------
# 15. AOD error maps companion (guard colorbar when im is None)
# -------------------------------------------------------------------------------
print('\n--- AOD error maps ---', flush=True)


def polder_grid_seasonal(region_name, var='AOD_550'):
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
    cfg = REGIONS[region_name]
    da = data[model_name]['od550aer']['od550aer']
    da = da.sel(time=slice(*cfg['time_slice'])).mean('time')
    da = da.assign_coords(lon=(da.lon % 360)).sortby('lon')
    return da


def plot_region_aod_diff(region_name, model_name, ax=None):
    model_da = model_grid_seasonal(model_name, region_name)
    polder_da = polder_grid_seasonal(region_name)
    if polder_da is None:
        return None
    polder_on_model = polder_da.interp(lat=model_da.lat, lon=model_da.lon, method='nearest')
    diff = model_da - polder_on_model
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
    lon_grid, lat_grid = np.meshgrid(model_da.lon.values, model_da.lat.values)
    im = ax.pcolormesh(lon_grid, lat_grid, diff.values, cmap='RdBu_r', vmin=-0.3, vmax=0.3,
                       shading='auto', transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(f'{region_name.upper()} AOD error ({short_model_name(model_name)})', fontweight='bold')
    return im


map_regions = [r for r in ['africa', 'amazon', 'outflow_af'] if r in REGIONS]
if map_regions:
    selected_model = next(
        (m for m in sorted(models) if m in model_seasonal.get('africa', {}).get('od550aer', {})),
        sorted(models)[0] if models else None,
    )
    if selected_model is not None:
        print(f'Selected model for AOD-error maps: {selected_model}', flush=True)
        fig, axes = plt.subplots(1, len(map_regions), figsize=(6 * len(map_regions), 5),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        if len(map_regions) == 1:
            axes = [axes]
        im = None
        for ax, region in zip(axes, map_regions):
            im = plot_region_aod_diff(region, selected_model, ax=ax)
        if im is not None:
            cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.05, fraction=0.046)
            cbar.set_label('AOD error (model - POLDER)', fontweight='bold')
        plt.suptitle('Figure 6 companion: AOD error maps (model - POLDER)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_path = project_root / 'notebooks' / 'AOD_error_attribution_figure6_maps.png'
        if SAVE_FIGURE:
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f'Saved: {fig_path}', flush=True)
            plt.close(fig)
        if _IN_IPYTHON:
            plt.show()
        else:
            plt.close(fig)

# Diagnostic scatter: model vs POLDER AOD by region (full ensemble)
fig, ax = plt.subplots(figsize=(9, 7))
all_vals = []
for region in plot_regions:
    sub = decomp_df[decomp_df['region'] == region]
    if sub.empty:
        continue
    aod_obs = float(aod_obs_by_region.get(region, np.nan))
    if not np.isfinite(aod_obs):
        continue
    ax.scatter([aod_obs] * len(sub), sub['AOD_model'], color=region_colors[region],
               label=region, alpha=0.6, s=60)
    all_vals.extend([aod_obs] + sub['AOD_model'].tolist())
if all_vals:
    max_val = max(all_vals)
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1')
ax.set_xlabel('POLDER AOD (observation)', fontweight='bold')
ax.set_ylabel('Model AOD', fontweight='bold')
ax.set_title('Diagnostic: Model vs POLDER AOD by region (full ensemble)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
fig_path = project_root / 'notebooks' / 'AOD_error_attribution_figure6_scatter.png'
if SAVE_FIGURE:
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {fig_path}', flush=True)
    plt.close(fig)
if _IN_IPYTHON:
    plt.show()
else:
    plt.close(fig)

# -------------------------------------------------------------------------------
# 16. CSV outputs, robustness, and gap summary markdown
# -------------------------------------------------------------------------------
print('\n--- Saving outputs ---', flush=True)

if SAVE_CSV:
    nb = project_root / 'notebooks'
    combined_reg_df.to_csv(nb / 'AOD_error_attribution_regression_data.csv', index=False)
    combined_constrained_df.to_csv(nb / 'AOD_error_attribution_constrained_estimates.csv', index=False)
    combined_decomp_df.to_csv(nb / 'AOD_error_attribution_decomposition.csv', index=False)
    if not combined_outflow_df.empty:
        combined_outflow_df.to_csv(nb / 'AOD_error_attribution_outflow_predictions.csv', index=False)
    combined_table1_df.to_csv(nb / 'AOD_error_attribution_table1_comparison.csv', index=False)
    combined_loo_df.to_csv(nb / 'AOD_error_attribution_loo_validation.csv', index=False)
    if not combined_robust_df.empty:
        combined_robust_df.to_csv(nb / 'AOD_error_attribution_robustness.csv', index=False)
    for ens_name, res in results.items():
        res['reg_df'].to_csv(nb / f'AOD_error_attribution_regression_data_{ens_name}.csv', index=False)
        res['constrained_df'].to_csv(nb / f'AOD_error_attribution_constrained_estimates_{ens_name}.csv', index=False)
        res['decomp_df'].to_csv(nb / f'AOD_error_attribution_decomposition_{ens_name}.csv', index=False)
        if not res['outflow_pred_df'].empty:
            res['outflow_pred_df'].to_csv(nb / f'AOD_error_attribution_outflow_predictions_{ens_name}.csv', index=False)
    print('Saved CSV outputs.', flush=True)

# Gap summary markdown
gap_md_lines = [
    '# AOD error attribution — gap summary vs Zhong et al. 2022 Table 1',
    '',
    f'Ensembles: full={len(models)} models, paper={len(paper_models)} models.',
    f'Gap threshold: |gap_pct| > {GAP_THRESHOLD_PCT}%.',
    '',
    '## Out of scope (no data in this repo)',
    '',
    '- Fig. 3 emission-inventory comparison (GFED/QFED/FEER/GFAS)',
    '- Fig. 6 ECHAM-HAM EC/MFC climate-model rerun',
    '- AERONET cross-check of regional AOD/AE',
    '',
    '## Known methodological caveats',
    '',
    '- GPCP SE Asia / Eastern Siberia precip in our boxes is much wetter than',
    '  Zhong Table 1 (our land-masked GPCP vs their regional means). This',
    '  propagates into constrained τ and E.',
    '- Homogenized AOD for SE Asia / Boreal NA can diverge strongly from',
    '  Table 1 when sampling/coverage differs from the paper.',
    '- LOO validation uses per-region leave-one-model-out, pooled across the',
    '  five SOURCE_REGIONS; metrics report Pearson R (as in Fig. 2c/d).',
    '',
]
for ens_name, res in results.items():
    gap_md_lines.append(f'## Ensemble: {ens_name}')
    gap_md_lines.append('')
    t1 = res['table1_df']
    flagged = t1[t1['gap_pct'].abs() > GAP_THRESHOLD_PCT].sort_values(['region', 'variable'])
    if flagged.empty:
        gap_md_lines.append('No variables exceed the gap threshold.')
    else:
        gap_md_lines.append('| region | variable | computed | paper | gap | gap_pct |')
        gap_md_lines.append('| --- | --- | --- | --- | --- | --- |')
        for _, row in flagged.iterrows():
            gap_md_lines.append(
                f"| {row['region']} | {row['variable']} | {row['computed']:.4g} | "
                f"{row['paper']:.4g} | {row['gap']:.4g} | {row['gap_pct']:.1f}% |"
            )
    gap_md_lines.append('')
    loo = res['loo_df']
    for var in ['MEC', 'inv_lifetime']:
        loo_m = loo.copy()
        if var == 'MEC' and not loo_m.empty:
            msub = loo_m['variable'] == 'MEC'
            loo_m.loc[msub, 'modelled'] = loo_m.loc[msub, 'modelled'] / 1000.0
            loo_m.loc[msub, 'predicted'] = loo_m.loc[msub, 'predicted'] / 1000.0
        m = compute_loo_metrics(loo_m, var)
        gap_md_lines.append(
            f'- LOO {var}: R={m["r"]:.3f} (paper {PAPER_LOO_R[var]:.2f}), '
            f'NMB={m["nmb"]:.1f}%, RMSE={m["rmse"]:.4g}, n={m["n"]}'
        )
    dmean = res['decomp_df'][['pct_E', 'pct_tau', 'pct_MEC', 'pct_cross']].mean()
    if len(dmean):
        gap_md_lines.append(
            f'- Fig.4 mean |error| %: E={dmean["pct_E"]:.1f} / tau={dmean["pct_tau"]:.1f} / '
            f'MEC={dmean["pct_MEC"]:.1f} / cross={dmean["pct_cross"]:.1f} '
            f'(paper {PAPER_DECOMP_PCT["pct_E"]}/{PAPER_DECOMP_PCT["pct_tau"]}/'
            f'{PAPER_DECOMP_PCT["pct_MEC"]}/{PAPER_DECOMP_PCT["pct_cross"]})'
        )
    op = res['outflow_params']
    if op:
        gap_md_lines.append(
            f'- Outflow meta-model: R2={op["r2"]:.3f}, NMB={op["nmb"]:.1f}%, '
            f'RMSE={op["rmse"]:.4f}, n={op["n"]}'
        )
    gap_md_lines.append('')

gap_md_path = project_root / 'notebooks' / 'AOD_error_attribution_gap_summary.md'
gap_md_path.write_text('\n'.join(gap_md_lines))
print(f'Saved gap summary: {gap_md_path}', flush=True)

# Final summary
print('\n--- Final summary ---', flush=True)
print(f'Models (full): {len(models)}; paper subset: {len(paper_models)}', flush=True)
for ens_name, res in results.items():
    print(f'  [{ens_name}] reg rows={len(res["reg_df"])}, constrained={len(res["constrained_df"])}, '
          f'decomp={len(res["decomp_df"])}, outflow={len(res["outflow_pred_df"])}', flush=True)
print('\nDone.', flush=True)
