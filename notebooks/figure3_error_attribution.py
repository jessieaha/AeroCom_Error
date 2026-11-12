"""
Regression-based error attribution for AeroCom biomass-burning aerosols.

This script follows the framework of Zhong et al. (2023, Sci. Adv.) and reproduces
Figure 3: decomposition of modeled AAOD errors into contributions from emission,
lifetime, MAC, and cross terms.

Key equations (from the paper):
    AAOD = E * tau * MAC                                    (1)
    1/tau = alpha * Pr + beta * AE + A                      (3)
    MAC = F * SSA                                            (user-requested no-intercept)
    E_C = AAOD_obs / (tau_C * MAC_C)                         (constrained emission)

Error decomposition (Eq. 4):
    dAAOD = dE*tau_C*MAC_C + E_C*d*tau*MAC_C + E_C*tau_C*dMAC + C
"""

import sys
import os
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Project paths
project_root = Path('/scistor/guest/gbb083/AeroCom')
py_dir = project_root / 'py'
if str(py_dir) not in sys.path:
    sys.path.insert(0, str(py_dir))

import functions
import cameo_toolbox as ct
import aerocom_data

print(f'Project root: {project_root}')
print(f'Python dir:   {py_dir}')


# -----------------------------------------------------------------------------
# Configuration: model exclusion and lifetime filtering
# -----------------------------------------------------------------------------
# List of model names to exclude from all loops, regressions, and plots.
# Names must match the keys in the monthly pickle exactly.
EXCLUDE_MODELS = []  # e.g., ['GEOS-i33p2-met2010_AP3-CTRL', 'MIROC-SPRINTARS_AP3-CTRL']

# Lifetime outlier filtering.  Values outside [LIFETIME_MIN_DAYS, LIFETIME_MAX_DAYS]
# (including NaN/inf) are masked before aggregation and regressions.  Set to None
# to disable either bound.
LIFETIME_MAX_DAYS = 365   # mask lifetime values > 365 days
LIFETIME_MIN_DAYS = 1e-3  # mask lifetime values < 0.001 days

# Save figure PNG files to disk?  Default False so figures are rendered in the
# notebook first.  CSV output is still saved by default.
SAVE_FIGURE = False


# -----------------------------------------------------------------------------
# 1. Load monthly model data
# -----------------------------------------------------------------------------
# Two loading options:
#   USE_PICKLE = False  -> load directly from Data/AP3_processed_monthly/ NetCDF files
#   USE_PICKLE = True   -> load from the legacy master pickle (backward compatible)
USE_PICKLE = True

VARIABLES = [
    'abs550aer', 'depbc', 'depdust', 'depoa', 'depso2', 'depso4', 'depss',
    'emibc', 'emidust', 'emioa', 'emiso2', 'emiss', 'loadbc', 'loaddust',
    'loadoa', 'loadso2', 'loadso4', 'loadss', 'od440aer', 'od550aer',
    'od870aer', 'od865aer', 'od550bc', 'od550dust', 'od550oa', 'od550so4', 'od550ss',
    'precip'
]

if USE_PICKLE:
    monthly_pickle = project_root / 'Data' / 'var_files' / 'original' / 'monthly' / 'monthly_aerocom_data.pickle'
    print(f'Loading monthly pickle: {monthly_pickle}')
    with open(monthly_pickle, 'rb') as f:
        raw_data = pickle.load(f)
else:
    print('Loading monthly data from processed NetCDF files...')
    raw_data = aerocom_data.load_monthly_data_from_netcdf(
        output_base_dir=str(project_root / 'Data' / 'AP3_processed_monthly'),
        variables=VARIABLES,
    )

# Apply model exclusion list before any processing
missing_excluded = [m for m in EXCLUDE_MODELS if m not in raw_data]
if missing_excluded:
    print(f'Warning: excluded models not found in data: {missing_excluded}')
models = sorted([m for m in raw_data.keys() if m not in EXCLUDE_MODELS])
actually_excluded = [m for m in EXCLUDE_MODELS if m in raw_data]
print(f'Number of models: {len(models)} (excluded {len(actually_excluded)}: {actually_excluded})')


# -----------------------------------------------------------------------------
# 2. Normalize time and build derived variables
# -----------------------------------------------------------------------------
def normalize_dataset_time(ds, var_hint=None):
    import aerocom_data
    year = locals()["year"] if "year" in locals() else None
    return aerocom_data.normalize_dataset_time(ds, var_hint=var_hint, year=year)


def sum_datasets(model_data, keys, out_name):
    import aerocom_data
    req = require_all if "require_all" in locals() else False
    return aerocom_data.sum_datasets(model_data, keys, out_name, require_all=req)


def normalize_precipitation_units(precip_ds):
    import aerocom_data
    return aerocom_data.normalize_precipitation_units(precip_ds)


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
        except Exception as e:
            print(f'  Failed to normalize {m}/{var}: {e}')
            normalized[var] = None
    normalized['load_total'] = sum_datasets(normalized, LOAD_VARS, 'load_total')
    normalized['emi_total'] = sum_datasets(normalized, EMI_VARS, 'emi_total')
    normalized['load_BC_OA'] = sum_datasets(normalized, BC_OA_LOAD, 'load_BC_OA')
    normalized['emi_BC_OA'] = sum_datasets(normalized, BC_OA_EMI, 'emi_BC_OA')
    # Normalize precipitation units to mm day^-1 for consistent analysis
    if normalized.get('precip') is not None:
        normalized['precip'] = normalize_precipitation_units(normalized['precip'])
    data[m] = normalized


# -----------------------------------------------------------------------------
# 3. Calculate derived variables
# -----------------------------------------------------------------------------
def filter_lifetime(da, model_name, var_name, max_days=None, min_days=None):
    import notebook_setup as _setup
    return _setup.filter_lifetime(da, model_name, var_name, max_days=max_days, min_days=min_days)


derived_vars = ['MEC', 'MAC', 'SSA', 'AE']
derived = {m: {} for m in models}
lifetime_filter_stats = []

for m in models:
    for dv in derived_vars:
        ds = aerocom_data.calculate_derived_var(data[m], m, dv)
        if ds is None:
            derived[m][dv] = None
        else:
            var_name = list(ds.data_vars)[0]
            derived[m][dv] = ds[var_name]

    # Lifetime for BC+OA only (paper focus)
    if data[m].get('load_BC_OA') is not None and data[m].get('emi_BC_OA') is not None:
        load_da = data[m]['load_BC_OA']['load_BC_OA']
        emi_da = data[m]['emi_BC_OA']['emi_BC_OA']
        lt = load_da / (emi_da * 3600 * 24)
        lt, stats = filter_lifetime(lt, m, 'lifetime_BC_OA', LIFETIME_MAX_DAYS, LIFETIME_MIN_DAYS)
        derived[m]['lifetime_BC_OA'] = lt
        lifetime_filter_stats.append({'model': m, 'var': 'lifetime_BC_OA', **stats})
    else:
        derived[m]['lifetime_BC_OA'] = None

    # Total aerosol lifetime (for completeness)
    if data[m].get('load_total') is not None and data[m].get('emi_total') is not None:
        load_da = data[m]['load_total']['load_total']
        emi_da = data[m]['emi_total']['emi_total']
        lt = load_da / (emi_da * 3600 * 24)
        lt, stats = filter_lifetime(lt, m, 'lifetime', LIFETIME_MAX_DAYS, LIFETIME_MIN_DAYS)
        derived[m]['lifetime'] = lt
        lifetime_filter_stats.append({'model': m, 'var': 'lifetime', **stats})
    else:
        derived[m]['lifetime'] = None

# Summary of lifetime filtering
if lifetime_filter_stats:
    total_excluded = sum(s['n_excluded'] for s in lifetime_filter_stats)
    total_cells = sum(s['n_total'] for s in lifetime_filter_stats)
    print(f'\nLifetime filter summary: excluded {total_excluded:,} / {total_cells:,} '
          f'grid-cell values ({total_excluded / total_cells * 100:.2f}%)')

# Flatten to DataArrays for aggregation
data_derived = {}
for m in models:
    data_derived[m] = {}
    for k, v in data[m].items():
        if v is not None:
            data_derived[m][k] = v[list(v.data_vars)[0]]
    for k, v in derived[m].items():
        if v is not None:
            data_derived[m][k] = v

print('\nDerived-variable availability:')
for dv in derived_vars + ['lifetime', 'lifetime_BC_OA']:
    n = sum(1 for m in derived if derived[m][dv] is not None)
    print(f'  {dv:15s}: {n:2d} / {len(models)} models')


# -----------------------------------------------------------------------------
# 4. Regional masks and monthly aggregation
# -----------------------------------------------------------------------------
import regions as _region_registry
REGIONS = _region_registry.aerocom_view([n for n in ['global', 'africa', 'amazon', 'outflow_af'] if n in _region_registry.REGIONS], year=2010)

sample_model = next((m for m in models if data[m].get('od550aer') is not None), None)
template = data[sample_model]['od550aer'].isel(time=0)
print(f'Template grid from {sample_model}: {template.dims}')

# Global override (legacy).  If None, each region's own surface_type is used.
SURFACE_TYPE = None  # 'all', 'land', 'ocean', or None to use per-region defaults

masks = ct.masks_for_regions(template, REGIONS)
print('Regions created:', list(masks.keys()))
for name, mask in masks.items():
    print(f'  {name}: surface_type={mask.attrs["surface"]}')


def aggregate_region(model_dict, var_name, region_name, return_time_series=False, skipna=False):
    import notebook_setup as _setup
    rts = return_time_series if "return_time_series" in locals() else False
    return _setup.aggregate_region(
        model_dict, var_name, region_name, masks, region=REGIONS,
        return_time_series=rts, skipna=skipna,
    )


# Monthly time series for model variables
variables_to_aggregate = ['MEC', 'MAC', 'SSA', 'AE', 'lifetime_BC_OA', 'load_BC_OA', 'emi_BC_OA', 'precip']
# Lifetime variables may contain masked NaNs from the outlier filter; skip them during aggregation.
LIFETIME_VARS = {'lifetime_BC_OA', 'lifetime'}
model_monthly = {region: {var: aggregate_region(data_derived, var, region, return_time_series=True,
                                                  skipna=(var in LIFETIME_VARS))
                          for var in variables_to_aggregate}
                 for region in REGIONS}

# Seasonal means for model variables (used for the paper-style regressions)
model_seasonal = {region: {var: aggregate_region(data_derived, var, region, return_time_series=False,
                                                   skipna=(var in LIFETIME_VARS))
                           for var in variables_to_aggregate}
                  for region in REGIONS}

print('\nModel seasonal availability:')
for region in REGIONS:
    for var in ['MAC', 'SSA', 'AE', 'lifetime_BC_OA', 'precip']:
        n = len(model_seasonal[region].get(var, {}))
        print(f'  {region} {var}: {n} models')


# -----------------------------------------------------------------------------
# 5. Load POLDER observations and compute monthly regional means
# -----------------------------------------------------------------------------
polder_path = project_root / 'Data' / 'AP3_POLDER_Collocated' / 'POLDER_GRASP_coloc_3h_AP3_2010_lon0_10.0_lat0_10.0.parquet'
print(f'\nLoading POLDER observations: {polder_path}')

polder_cols = ['time', 'longitude', 'latitude', 'AOD_550', 'AAOD_550', 'AOD_440', 'AOD_870']
polder_df = pd.read_parquet(polder_path, columns=polder_cols)
print(f'POLDER rows: {len(polder_df)}')

# Parse time and add month
polder_df['time'] = pd.to_datetime(polder_df['time'])
polder_df['month'] = polder_df['time'].dt.to_period('M')

# Compute SSA and AE (550 vs 440 nm)
with np.errstate(divide='ignore', invalid='ignore'):
    polder_df['SSA'] = 1.0 - polder_df['AAOD_550'] / polder_df['AOD_550']
    polder_df['AE'] = -np.log(polder_df['AOD_550'] / polder_df['AOD_440']) / np.log(550.0 / 440.0)

# Filter valid quality
polder_df = polder_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['SSA', 'AE', 'AAOD_550', 'AOD_550'])
print(f'POLDER rows after filtering: {len(polder_df)}')


def polder_region_mask(df, lon_range, lat_range):
    import ppe_io
    return ppe_io.polder_region_mask(df, lon_range, lat_range)


def polder_monthly_means(df, region_name):
    import ppe_io
    return ppe_io.polder_monthly_means(df, region_name, REGIONS)


polder_monthly = {region: polder_monthly_means(polder_df, region) for region in REGIONS}
print('\nPOLDER monthly means per region:')
for region, df in polder_monthly.items():
    print(f'  {region}: {len(df)} months')


# -----------------------------------------------------------------------------
# 6. Load GPCP precipitation and compute monthly regional means
# -----------------------------------------------------------------------------
gpcp_path = project_root / 'Data' / 'Prec' / 'GPCP_2010_0-360.nc'
print(f'\nLoading GPCP precipitation: {gpcp_path}')
gpcp = xr.open_dataset(gpcp_path)
gpcp_precip = gpcp['sat_gauge_precip']
print(f'GPCP dims: {gpcp_precip.dims}, shape: {gpcp_precip.shape}')


def gpcp_region_mean(precip_da, region_name):
    import ppe_io
    apply = False
    if "apply_surface_mask" in locals():
        apply = bool(apply_surface_mask)
    elif "apply_land_mask" in locals():
        apply = bool(apply_land_mask)
    mask = masks.get(region_name) if apply and "masks" in globals() else None
    return ppe_io.gpcp_region_mean(
        precip_da, region_name, REGIONS, surface_mask=mask, apply_surface_mask=apply,
    )


gpcp_region = {region: gpcp_region_mean(gpcp_precip, region) for region in REGIONS}
print('\nGPCP regional mean precipitation (mm/day):')
for region, da in gpcp_region.items():
    mean_val = float(da.mean().values) if da.size else np.nan
    print(f'  {region}: {mean_val:.3f}')


# -----------------------------------------------------------------------------
# 7. Derive model regressions
# -----------------------------------------------------------------------------
print('\n--- Model regressions ---')

# Build regression DataFrame: one row per model per region (seasonal mean)
# This follows the paper (Fig. 2 and Fig. S3), which use seasonally averaged data per model.
regression_rows = []
regression_excluded = []
for region in REGIONS:
    if region == 'global':
        continue  # paper focuses on source/outflow regions
    for model in models:
        ssa = model_seasonal[region]['SSA'].get(model)
        mac = model_seasonal[region]['MAC'].get(model)
        ae = model_seasonal[region]['AE'].get(model)
        lt = model_seasonal[region]['lifetime_BC_OA'].get(model)
        precip = model_seasonal[region]['precip'].get(model)
        if not all([v is not None for v in [ssa, mac, ae, lt, precip]]):
            continue
        if not np.isfinite(float(lt)):
            regression_excluded.append({'region': region, 'model': model, 'reason': 'non-finite lifetime'})
            continue
        regression_rows.append({
            'region': region, 'model': model,
            'SSA': float(ssa), 'MAC': float(mac), 'AE': float(ae),
            'lifetime': float(lt), 'inv_lifetime': 1.0 / float(lt),
            'precip': float(precip),
        })
if regression_excluded:
    print(f'  Excluded {len(regression_excluded)} model/region regression points due to lifetime filtering:')
    for item in regression_excluded:
        print(f'    {item["model"]} {item["region"]}')
reg_df = pd.DataFrame(regression_rows)
print(f'Regression data rows (seasonal means): {len(reg_df)}')

# MAC vs SSA: regression through origin (user request)
mac_ssa_results = {}
for region in REGIONS:
    if region == 'global':
        continue
    sub = reg_df[reg_df['region'] == region].dropna(subset=['MAC', 'SSA'])
    if len(sub) < 3:
        print(f'{region}: skipped MAC vs SSA regression (only {len(sub)} valid points)')
        continue
    x = sub['SSA'].values
    y = sub['MAC'].values
    # No-intercept regression: y = F*x
    F = np.nansum(x * y) / np.nansum(x ** 2)
    residuals = y - F * x
    r2 = 1 - np.nansum(residuals ** 2) / np.nansum(y ** 2)
    mac_ssa_results[region] = {'F': F, 'r2': r2, 'n': len(sub)}
    print(f'{region}: MAC = {F:.4f} * SSA (uncentered R^2 = {r2:.3f}, n={len(sub)})')

# 1/lifetime vs precipitation + AE: multiple linear regression
inv_lt_results = {}
for region in REGIONS:
    if region == 'global':
        continue
    sub = reg_df[reg_df['region'] == region].dropna(subset=['inv_lifetime', 'precip', 'AE'])
    if len(sub) < 4:
        print(f'{region}: skipped 1/tau regression (only {len(sub)} valid points)')
        continue
    X = np.column_stack([sub['precip'].values, sub['AE'].values])
    y = sub['inv_lifetime'].values
    # Add intercept column
    X_ols = np.column_stack([np.ones(len(X)), X])
    coeffs, _, _, _ = np.linalg.lstsq(X_ols, y, rcond=None)
    y_pred = X_ols @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    inv_lt_results[region] = {
        'intercept': coeffs[0], 'alpha': coeffs[1], 'beta': coeffs[2],
        'r2': r2, 'n': len(sub)
    }
    print(f'{region}: 1/tau = {coeffs[0]:.4f} + {coeffs[1]:.4f}*Pr + {coeffs[2]:.4f}*AE (R^2={r2:.3f}, n={len(sub)})')


# -----------------------------------------------------------------------------
# 8. Apply regressions to observations and constrain E
# -----------------------------------------------------------------------------
print('\n--- Constrained estimates from observations ---')

# Constrained estimates from seasonal-mean observations (following the paper)
constrained = []
for region in REGIONS:
    if region == 'global':
        continue
    polder_df_region = polder_monthly[region]
    if polder_df_region.empty:
        print(f'  {region}: skipped constrained estimate (no POLDER data)')
        continue
    if region not in mac_ssa_results or region not in inv_lt_results:
        print(f'  {region}: skipped constrained estimate (no model regression)')
        continue
    gpcp_ts = gpcp_region[region]
    mac_params = mac_ssa_results[region]
    lt_params = inv_lt_results[region]

    # Seasonal means of observations
    ssa_obs_mean = polder_df_region['SSA'].mean()
    ae_obs_mean = polder_df_region['AE'].mean()
    aaod_obs_mean = polder_df_region['AAOD_550'].mean()
    precip_obs_mean = float(gpcp_ts.mean().values) if gpcp_ts.size else np.nan

    mac_c = mac_params['F'] * ssa_obs_mean
    inv_lt_c = lt_params['intercept'] + lt_params['alpha'] * precip_obs_mean + lt_params['beta'] * ae_obs_mean
    tau_c = 1.0 / inv_lt_c if inv_lt_c > 0 else np.nan
    # E from AAOD/(tau*MAC) is in g m^-2 day^-1; convert to kg m^-2 s^-1 for comparison with model emissions
    e_c_g_m2_day = aaod_obs_mean / (tau_c * mac_c) if (tau_c > 0 and mac_c > 0) else np.nan
    e_c = e_c_g_m2_day / (1000.0 * 86400.0)

    constrained.append({
        'region': region,
        'SSA_obs': ssa_obs_mean, 'AE_obs': ae_obs_mean, 'AAOD_obs': aaod_obs_mean,
        'precip_obs': precip_obs_mean,
        'MAC_c': mac_c, 'tau_c': tau_c,
        'E_c_g_m2_day': e_c_g_m2_day, 'E_c': e_c,
    })

constrained_df = pd.DataFrame(constrained)
print(f'Constrained rows: {len(constrained_df)}')
print(constrained_df.set_index('region')[['MAC_c', 'tau_c', 'E_c']])


# -----------------------------------------------------------------------------
# 9. Model vs constrained comparison and error decomposition
# -----------------------------------------------------------------------------
print('\n--- Error decomposition ---')

# Model seasonal means for variables used in error decomposition
decomp_rows = []
for region in REGIONS:
    if region == 'global':
        continue
    csub = constrained_df[constrained_df['region'] == region]
    if csub.empty:
        print(f'  {region}: skipped error decomposition (no constrained estimate)')
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
            # AAOD = MAC * load_BC_OA (model internal consistency)
            aaod_model = float(model_seasonal[region]['MAC'][model]) * float(model_seasonal[region]['load_BC_OA'][model])
        # Model emission
        emi_model = model_seasonal[region].get('emi_BC_OA', {}).get(model)

        if any(v is None or np.isnan(v) for v in [mac_model, lt_model, emi_model, mac_c, tau_c, aaod_c, e_c]):
            continue

        mac_model = float(mac_model)
        lt_model = float(lt_model)
        emi_model = float(emi_model)
        e_c = float(e_c)
        # Convert emissions from kg m^-2 s^-1 to g m^-2 day^-1 for the AAOD product
        emi_model_g = emi_model * 1000.0 * 86400.0
        e_c_g = e_c * 1000.0 * 86400.0
        aaod_model = float(aaod_model) if aaod_model is not None else emi_model_g * lt_model * mac_model

        dMAC = mac_model - mac_c
        dtau = lt_model - tau_c
        dE_g = emi_model_g - e_c_g

        # AAOD error decomposition (Eq. 4) using g m^-2 day^-1 emissions
        dAAOD_E = dE_g * tau_c * mac_c
        dAAOD_tau = e_c_g * dtau * mac_c
        dAAOD_MAC = e_c_g * tau_c * dMAC
        dAAOD_total = aaod_model - aaod_c
        cross = dAAOD_total - (dAAOD_E + dAAOD_tau + dAAOD_MAC)

        # Relative percentage contributions (absolute)
        denom = abs(dAAOD_E) + abs(dAAOD_tau) + abs(dAAOD_MAC) + abs(cross)
        if denom > 0:
            pct_E = abs(dAAOD_E) / denom * 100
            pct_tau = abs(dAAOD_tau) / denom * 100
            pct_MAC = abs(dAAOD_MAC) / denom * 100
            pct_cross = abs(cross) / denom * 100
        else:
            pct_E = pct_tau = pct_MAC = pct_cross = np.nan

        decomp_rows.append({
            'region': region, 'model': model,
            'MAC_model': mac_model, 'MAC_c': mac_c,
            'tau_model': lt_model, 'tau_c': tau_c,
            'E_model': emi_model, 'E_c': e_c,
            'AAOD_model': aaod_model, 'AAOD_c': aaod_c,
            'dAAOD_E': dAAOD_E, 'dAAOD_tau': dAAOD_tau,
            'dAAOD_MAC': dAAOD_MAC, 'dAAOD_cross': cross,
            'dAAOD_total': dAAOD_total,
            'pct_E': pct_E, 'pct_tau': pct_tau,
            'pct_MAC': pct_MAC, 'pct_cross': pct_cross,
        })

decomp_df = pd.DataFrame(decomp_rows)
print(f'Decomposition rows: {len(decomp_df)}')
print(decomp_df.groupby('region')[['pct_E', 'pct_tau', 'pct_MAC', 'pct_cross']].mean())


# -----------------------------------------------------------------------------
# 10. Figure 3: error attribution plots
# -----------------------------------------------------------------------------
print('\n--- Saving Figure 3 plots ---')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (a) MAC vs SSA regression with model monthly points and observation mean
ax = axes[0, 0]
colors = {'africa': '#ff7f0e', 'amazon': '#2ca02c', 'outflow_af': '#9467bd'}
for region in ['africa', 'amazon', 'outflow_af']:
    if region not in mac_ssa_results:
        continue
    sub = reg_df[reg_df['region'] == region].dropna(subset=['MAC', 'SSA'])
    if sub.empty:
        continue
    ax.scatter(sub['SSA'], sub['MAC'], s=25, alpha=0.4, color=colors[region], label=f'{region} model')
    F = mac_ssa_results[region]['F']
    x_line = np.linspace(sub['SSA'].min(), sub['SSA'].max(), 100)
    ax.plot(x_line, F * x_line, '--', color=colors[region], linewidth=2,
            label=f'{region}: MAC={F:.2f}*SSA')
    # Observation mean
    csub = constrained_df[constrained_df['region'] == region]
    if not csub.empty:
        ax.scatter(csub['SSA_obs'].mean(), csub['MAC_c'].mean(), s=150, marker='*',
                   color=colors[region], edgecolors='black', zorder=5, label=f'{region} obs')
ax.set_xlabel('SSA', fontweight='bold')
ax.set_ylabel('MAC (m² g⁻¹)', fontweight='bold')
ax.set_title('(a) MAC vs SSA', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# (b) 1/lifetime vs precipitation + AE (model points colored by region)
ax = axes[0, 1]
for region in ['africa', 'amazon', 'outflow_af']:
    if region not in inv_lt_results:
        continue
    sub = reg_df[reg_df['region'] == region].dropna(subset=['inv_lifetime', 'precip', 'AE'])
    if sub.empty:
        continue
    sc = ax.scatter(sub['precip'], sub['inv_lifetime'], c=sub['AE'], cmap='viridis',
                    s=25, alpha=0.5, label=region)
if 'sc' in locals():
    plt.colorbar(sc, ax=ax, label='AE')
ax.set_xlabel('Precipitation (mm/day)', fontweight='bold')
ax.set_ylabel('1 / lifetime (day⁻¹)', fontweight='bold')
ax.set_title('(b) 1/lifetime vs precipitation colored by AE', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# (c) Constrained vs model E, tau, MAC
ax = axes[1, 0]
metrics = ['MAC', 'tau', 'E']
metric_labels = {'MAC': 'MAC (m² g⁻¹)', 'tau': 'Lifetime (days)', 'E': 'Emission (kg m⁻² s⁻¹)'}
width = 0.35
x = np.arange(len(metrics))
for idx, region in enumerate(['africa', 'amazon', 'outflow_af']):
    if region not in colors:
        continue
    sub = decomp_df[decomp_df['region'] == region]
    if sub.empty:
        continue
    means = [sub[f'{m}_c'].mean() for m in metrics]
    stds = [sub[f'{m}_c'].std() for m in metrics]
    ax.bar(x + idx*width - width/2, means, width, yerr=stds, label=f'{region} constrained',
           color=colors[region], alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([metric_labels[m] for m in metrics])
ax.set_ylabel('Mean value', fontweight='bold')
ax.set_title('(c) Constrained mean values by region', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

# (d) AAOD error decomposition (stacked bars per model, averaged by region)
ax = axes[1, 1]
for idx, region in enumerate(['africa', 'amazon', 'outflow_af']):
    sub = decomp_df[decomp_df['region'] == region].sort_values('dAAOD_total', key=abs, ascending=False)
    if sub.empty:
        continue
    n = len(sub)
    xpos = np.arange(n) + idx * (n + 1)
    ax.bar(xpos, sub['pct_E'], label='E' if idx == 0 else '', color='#1f77b4')
    ax.bar(xpos, sub['pct_tau'], bottom=sub['pct_E'], label='lifetime' if idx == 0 else '', color='#ff7f0e')
    ax.bar(xpos, sub['pct_MAC'], bottom=sub['pct_E'] + sub['pct_tau'], label='MAC' if idx == 0 else '', color='#2ca02c')
    ax.bar(xpos, sub['pct_cross'], bottom=sub['pct_E'] + sub['pct_tau'] + sub['pct_MAC'],
           label='cross' if idx == 0 else '', color='#d62728')
    ax.axvline(x=n - 0.5 + idx*(n+1), color='gray', linestyle='--', alpha=0.5)

ax.set_ylabel('Contribution to |AAOD error| (%)', fontweight='bold')
ax.set_xlabel('Model', fontweight='bold')
ax.set_title('(d) AAOD error decomposition', fontweight='bold')
ax.set_xticks([])
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Figure 3: Regression-based error attribution (Zhong et al. 2023)', fontsize=15, fontweight='bold')
plt.tight_layout()
fig_path = project_root / 'notebooks' / 'figure3_error_attribution.png'
if SAVE_FIGURE:
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {fig_path}')
plt.close()


# Per-region bar chart of mean percentage contributions
plot_regions = [r for r in ['africa', 'amazon', 'outflow_af'] if not decomp_df[decomp_df['region'] == r].empty]
fig, axes = plt.subplots(1, len(plot_regions), figsize=(6 * len(plot_regions), 5))
if len(plot_regions) == 1:
    axes = [axes]
for idx, region in enumerate(plot_regions):
    ax = axes[idx]
    sub = decomp_df[decomp_df['region'] == region]
    if sub.empty:
        continue
    mean_pcts = [sub['pct_E'].mean(), sub['pct_tau'].mean(), sub['pct_MAC'].mean(), sub['pct_cross'].mean()]
    labels = ['Emission', 'Lifetime', 'MAC', 'Cross']
    bars = ax.bar(labels, mean_pcts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Mean contribution to |AAOD error| (%)', fontweight='bold')
    ax.set_title(f'{region.upper()} — mean error attribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mean_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold')
plt.suptitle('Figure 3 (detail): Mean AAOD error attribution by region', fontweight='bold')
plt.tight_layout()
fig_path2 = project_root / 'notebooks' / 'figure3_error_attribution_mean_pct.png'
if SAVE_FIGURE:
    plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
    print(f'Saved: {fig_path2}')
plt.close()


# -----------------------------------------------------------------------------
# 11. Save CSV outputs
# -----------------------------------------------------------------------------
out_csv = project_root / 'notebooks' / 'figure3_regression_data.csv'
reg_df.to_csv(out_csv, index=False)
print(f'Saved: {out_csv}')

out_csv = project_root / 'notebooks' / 'figure3_constrained_estimates.csv'
constrained_df.to_csv(out_csv, index=False)
print(f'Saved: {out_csv}')

out_csv = project_root / 'notebooks' / 'figure3_error_decomposition.csv'
decomp_df.to_csv(out_csv, index=False)
print(f'Saved: {out_csv}')

out_csv = project_root / 'notebooks' / 'figure3_model_region_data.csv'
model_region_rows = []
for region in REGIONS:
    for model in models:
        row = {'region': region, 'model': model}
        for var in ['MEC', 'MAC', 'SSA', 'AE', 'lifetime_BC_OA', 'precip']:
            val = model_seasonal[region].get(var, {}).get(model)
            row[var] = float(val) if val is not None else np.nan
        model_region_rows.append(row)
pd.DataFrame(model_region_rows).to_csv(out_csv, index=False)
print(f'Saved: {out_csv}')

print('\nDone.')
