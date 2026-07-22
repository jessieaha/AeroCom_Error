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

# Notebook / project paths
notebook_path = Path('notebooks/error_attribution_paper.ipynb').resolve()
project_root = notebook_path.parent.parent
py_dir = project_root / 'py'

for p in (str(project_root), str(py_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import functions
import cameo_toolbox as ct
import aerocom_data

print(f'Project root: {project_root}')
print(f'Python dir:   {py_dir}')
print(f'Functions:    {functions.__file__}')
print(f'Aerocom data: {aerocom_data.__file__}')


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

monthly_pickle = project_root / 'Data' / 'var_files' / 'original' / 'monthly' / 'monthly_aerocom_data.pickle'

# Variable list matching the keys used in monthly_aerocom_data.pickle
VARIABLES = [
    'abs550aer', 'depbc', 'depdust', 'depoa', 'depso2', 'depso4', 'depss',
    'emibc', 'emidust', 'emioa', 'emiso2', 'emiss', 'loadbc', 'loaddust',
    'loadoa', 'loadso2', 'loadso4', 'loadss', 'od440aer', 'od550aer',
    'od870aer', 'od865aer', 'od550bc', 'od550dust', 'od550oa', 'od550so4', 'od550ss'
]

print(f'Loading monthly pickle: {monthly_pickle}')
with open(monthly_pickle, 'rb') as f:
    raw_data = pickle.load(f)

# Apply model exclusion list before any processing
missing_excluded = [m for m in EXCLUDE_MODELS if m not in raw_data]
if missing_excluded:
    print(f'Warning: excluded models not found in data: {missing_excluded}')
models = sorted([m for m in raw_data.keys() if m not in EXCLUDE_MODELS])
actually_excluded = [m for m in EXCLUDE_MODELS if m in raw_data]
print(f'Number of models: {len(models)} (excluded {len(actually_excluded)}: {actually_excluded})')
print('Models:', models)

# Show variable availability matrix (after exclusion)
avail = {var: sum(1 for m in models if raw_data[m].get(var) is not None)
         for var in VARIABLES}
print('\nVariable availability (out of {} models):'.format(len(models)))
for var, n in sorted(avail.items()):
    print(f'  {var:15s}: {n:2d}')


def normalize_dataset_time(ds, var_hint=None):
    """Normalize time coordinate of a dataset to first-of-month."""
    if ds is None:
        return None
    if not list(ds.data_vars):
        print(f'  Warning: empty dataset for {var_hint}, skipping')
        return None
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]
    if 'time' not in da.dims or len(da.time) == 0:
        return ds
    # Convert cftime to datetime64 when needed; normalize then truncates to first-of-month
    if not isinstance(da.time.values[0], (np.datetime64, pd.Timestamp)):
        try:
            # Use the CFTimeIndex conversion if available, otherwise fall back to string parsing
            if hasattr(da.indexes['time'], 'to_datetimeindex'):
                new_times = da.indexes['time'].to_datetimeindex().values
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
    # Extract DataArrays so variables with different names can be summed
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
    print(f'Normalizing {m} ...')
    normalized = {}
    for var in VARIABLES:
        if raw_data[m].get(var) is None:
            continue
        try:
            normalized[var] = normalize_dataset_time(raw_data[m][var], var_hint=f'{m}/{var}')
        except Exception as e:
            print(f'  Failed to normalize {m}/{var}: {e}')
            normalized[var] = None
    # Add total / BC+OA derived variables needed for lifetime and MEC convenience
    normalized['load_total'] = sum_datasets(normalized, LOAD_VARS, 'load_total')
    normalized['emi_total'] = sum_datasets(normalized, EMI_VARS, 'emi_total')
    normalized['load_BC_OA'] = sum_datasets(normalized, BC_OA_LOAD, 'load_BC_OA')
    normalized['emi_BC_OA'] = sum_datasets(normalized, BC_OA_EMI, 'emi_BC_OA')
    data[m] = normalized

print('Time coordinates normalized and total/BC+OA variables added for all models.')

# Quick sanity check
for m in models[:1]:
    ds = data[m]['od550aer']
    print(f'{m} od550aer time: {ds.time.values[:3]} ...')
    if data[m]['load_total'] is not None:
        print(f'{m} load_total time: {data[m]["load_total"].time.values[:3]} ...')


def filter_lifetime(da, model_name, var_name, max_days=None, min_days=None):
    """Mask unrealistic lifetime values (NaN, inf, and outside thresholds).

    Returns the filtered DataArray (invalid cells set to NaN) and a dict with
    the total number of cells and how many were excluded.
    """
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

    # Lifetime: total aerosol (load / emission, converted to days)
    if data[m].get('load_total') is not None and data[m].get('emi_total') is not None:
        load_da = data[m]['load_total']['load_total']
        emi_da = data[m]['emi_total']['emi_total']
        lt = load_da / (emi_da * 3600 * 24)
        lt, stats = filter_lifetime(lt, m, 'lifetime', LIFETIME_MAX_DAYS, LIFETIME_MIN_DAYS)
        derived[m]['lifetime'] = lt
        lifetime_filter_stats.append({'model': m, 'var': 'lifetime', **stats})
    else:
        derived[m]['lifetime'] = None

    # Lifetime for BC+OA only
    if data[m].get('load_BC_OA') is not None and data[m].get('emi_BC_OA') is not None:
        load_da = data[m]['load_BC_OA']['load_BC_OA']
        emi_da = data[m]['emi_BC_OA']['emi_BC_OA']
        lt = load_da / (emi_da * 3600 * 24)
        lt, stats = filter_lifetime(lt, m, 'lifetime_BC_OA', LIFETIME_MAX_DAYS, LIFETIME_MIN_DAYS)
        derived[m]['lifetime_BC_OA'] = lt
        lifetime_filter_stats.append({'model': m, 'var': 'lifetime_BC_OA', **stats})
    else:
        derived[m]['lifetime_BC_OA'] = None

# Add derived variables back into the model dictionaries for aggregation (as DataArrays)
data_derived = {}
for m in models:
    data_derived[m] = {}
    for k, v in data[m].items():
        if v is not None:
            data_derived[m][k] = v[list(v.data_vars)[0]]
    for k, v in derived[m].items():
        if v is not None:
            data_derived[m][k] = v

# Summary of lifetime filtering
if lifetime_filter_stats:
    total_excluded = sum(s['n_excluded'] for s in lifetime_filter_stats)
    total_cells = sum(s['n_total'] for s in lifetime_filter_stats)
    print(f'\nLifetime filter summary: excluded {total_excluded:,} / {total_cells:,} '
          f'grid-cell values ({total_excluded / total_cells * 100:.2f}%)')

# Availability report
print('Derived-variable availability:')
for dv in derived_vars + ['lifetime', 'lifetime_BC_OA']:
    n = sum(1 for m in derived if derived[m][dv] is not None)
    print(f'  {dv:15s}: {n:2d} / {len(models)} models')

# Sanity check one model
for m in models:
    if derived[m]['MEC'] is not None:
        print(f'\nExample {m}: MEC dims = {derived[m]["MEC"].dims}, mean = {float(derived[m]["MEC"].mean()):.4f}')
        break


cache_dir = project_root / 'Data' / 'AP3_processed_monthly' / 'derived'
cache_dir.mkdir(parents=True, exist_ok=True)

saved, failed = 0, 0
for m in models:
    for dv in derived_vars + ['lifetime', 'lifetime_BC_OA']:
        da = derived[m].get(dv)
        if da is None:
            continue
        out = cache_dir / f'{m}_{dv}.nc'
        try:
            da.to_dataset(name=dv).to_netcdf(out, engine='netcdf4')
            saved += 1
        except Exception as e:
            failed += 1
            print(f'Failed to save {out}: {e}')
print(f'Derived cache: {saved} files saved, {failed} failed -> {cache_dir}')

REGIONS = {
    'global': {
        'lon_range': (0, 360), 'lat_range': (-90, 90),
        'time_slice': ('2010-01-01', '2010-12-31'), 'edge_weighted': False,
    },
    'africa': {
        'lon_range': (15, 37), 'lat_range': (-15, 0),
        'time_slice': ('2010-06-01', '2010-09-30'), 'edge_weighted': False,
    },
    'amazon': {
        'lon_range': (287, 317), 'lat_range': (-17, -3),
        'time_slice': ('2010-07-01', '2010-10-31'), 'edge_weighted': False,
    },
    'outflow_af': {
        'lon_range': (350, 8), 'lat_range': (-15, 3),
        'time_slice': ('2010-06-01', '2010-09-30'), 'edge_weighted': True,
    },
}

# Find a template grid from a model with valid od550aer
sample_model = next((m for m in models if data[m].get('od550aer') is not None), None)
if sample_model is None:
    raise RuntimeError('No model has od550aer data.')

template = data[sample_model]['od550aer'].isel(time=0)
print(f'Template grid from {sample_model}: {template.dims}')

# SURFACE_TYPE can be changed to 'land' or 'ocean' for Africa/Amazon/outflow_af.
# Defaults: global=all, africa=land, amazon=land, outflow_af=ocean.
SURFACE_TYPE = 'all'

masks = {}
for name, cfg in REGIONS.items():
    masks[name] = ct.create_region_mask(
        template,
        region=name,
        surface_type=SURFACE_TYPE,
        mask_registry=masks,
    )
print('Regions created:', list(masks.keys()))
for name, mask in masks.items():
    print(f'  {name}: surface_type={mask.attrs["surface"]}')

# Plot the sample model's AOD550 using cameo_toolbox.uba_map_flex.
# The global map uses a curved Robinson projection; regional maps use a
# local PlateCarree projection to keep the regional boxes undistorted.
sample_aod = data[sample_model]['od550aer'].mean(dim='time')

ct.uba_map_flex(
    data=sample_aod,
    title=f'{sample_model} AOD550 2010 mean (Robinson)',
    projection_type='curved',
    extent='global',
    mycolor='YlOrRd',
    vmin=0, vmax=1.0,
    cbar_title='AOD550',
)


def aggregate_region(model_dict, var_name, region_name, return_time_series=False, skipna=False):
    """Spatially (+ temporally) aggregate `var_name` for every model, skipping missing variables."""
    cfg = REGIONS[region_name]
    result = {}
    for model, model_data in model_dict.items():
        if var_name not in model_data or model_data[var_name] is None:
            continue
        try:
            val = ct.regional_aggregate(
                model_data[var_name],
                masks[region_name],
                spatial='mean',
                edge_weighted=cfg['edge_weighted'],
                time_slice=cfg['time_slice'],
                temporal='mean',
                return_time_series=return_time_series,
                skipna=skipna,
            )
            result[model] = val
        except Exception as e:
            print(f'  Aggregation failed for {var_name} {model} {region_name}: {e}')
    return result


# Build regional-mean dictionaries for derived variables
LIFETIME_VARS = {'lifetime', 'lifetime_BC_OA'}
region_means = {var: {} for var in derived_vars + ['lifetime', 'lifetime_BC_OA']}
for region in REGIONS:
    for var in region_means:
        region_means[var][region] = aggregate_region(
            data_derived, var, region, skipna=(var in LIFETIME_VARS)
        )

# Print sample numbers
for region in REGIONS:
    print(f'\n{region.upper()} region mean:')
    for var in ['MEC', 'MAC', 'SSA', 'AE', 'lifetime']:
        vals = region_means[var].get(region, {})
        print(f'  {var}: {len(vals)} models, mean={np.mean(list(vals.values())) if vals else np.nan:.3f}')


def build_region_df(region, vars_dict):
    """Build a DataFrame of regional-mean values for common models."""
    available = {name: d for name, d in vars_dict.items() if region in d and d[region]}
    if not available:
        return None
    common_models = sorted(set.intersection(*[set(d[region].keys()) for d in available.values()]))
    if not common_models:
        return None
    df = pd.DataFrame({'model': common_models})
    for name, d in available.items():
        df[name] = [d[region][m] for m in common_models]
    return df


# Regional-mean data frame
region_data = {}
for region in REGIONS:
    df = build_region_df(region, {
        'SSA': region_means['SSA'],
        'MEC': region_means['MEC'],
        'MAC': region_means['MAC'],
        'AE': region_means['AE'],
        'lifetime': region_means['lifetime'],
        'lifetime_BC_OA': region_means['lifetime_BC_OA'],
    })
    if df is not None:
        region_data[region] = df
        print(f'{region}: {len(df)} models')
    else:
        print(f'{region}: no common models')

if region_data:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (region, df) in enumerate(region_data.items()):
        ax = axes[idx]
        ax.scatter(df['SSA'], df['MAC'], s=100, alpha=0.6, color=colors[idx],
                   edgecolors='black', linewidth=1.5)

        for _, row in df.iterrows():
            ax.annotate(row['model'].replace('_', '\n')[:15],
                        (row['SSA'], row['MAC']), fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords='offset points')

        ax.set_xlabel('Single Scattering Albedo (SSA)', fontsize=11, fontweight='bold')
        ax.set_ylabel('MAC (m² g⁻¹)', fontsize=11, fontweight='bold')
        ax.set_title(f'{region.upper()} — regional mean', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        stats_text = f"n={len(df)}\nMAC={df['MAC'].mean():.2f}±{df['MAC'].std():.2f}\nSSA={df['SSA'].mean():.3f}±{df['SSA'].std():.3f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('SSA vs MAC — regional mean (paper error-attribution baseline)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = project_root / 'notebooks' / 'error_attribution_SSA_MAC_regional.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {out.name}')
else:
    print('No region data available.')

def plot_combined_regions(region_data, x_col, y_col, x_label, y_label,
                          filename, force_intercept_zero=False):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (region, df) in enumerate(region_data.items()):
        color = colors[idx % len(colors)]
        x = df[x_col].values
        y = df[y_col].values
        ax.scatter(x, y, s=90, alpha=0.6, color=color, label=region.upper())

        if force_intercept_zero:
            slope = np.nansum(x * y) / np.nansum(x ** 2)
            residuals = y - slope * x
            r2 = 1 - np.nansum(residuals ** 2) / np.nansum(y ** 2)
            line = slope * x
            label = f"{region.upper()}: y={slope:.2f}x, unc. R²={r2:.2f}"
        else:
            slope, intercept, r, _, _ = linregress(x, y)
            line = slope * x + intercept
            r2 = r ** 2
            sign = '+' if intercept >= 0 else '-'
            label = f"{region.upper()}: y={slope:.2f}x {sign} {abs(intercept):.2f}, R²={r2:.2f}"

        ax.plot(x, line, '--', color=color, alpha=0.8)
        ax.text(0.02, 0.94 - idx * 0.12, label, transform=ax.transAxes,
                color=color, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor=color))

    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title('SSA-MAC regression by region', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(title='Region', loc='upper right')
    out = project_root / 'notebooks' / filename
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {out.name}')


for region, df in region_data.items():
    df['1-SSA'] = 1 - df['SSA']

plot_combined_regions(region_data, 'SSA', 'MAC', 'SSA', 'MAC (m² g⁻¹)',
                      'error_attribution_SSA_MAC_combined.png')
plot_combined_regions(region_data, '1-SSA', 'MAC', '1-SSA', 'MAC (m² g⁻¹)',
                      'error_attribution_MAC_vs_1minusSSA.png', force_intercept_zero=True)

def build_monthly_region_data(regions, variables):
    """Return {region: DataFrame} with one row per model per month."""
    monthly = {}
    for region in regions:
        ts = {}
        for var in variables:
            try:
                ts[var] = aggregate_region(
                    data_derived, var, region, return_time_series=True,
                    skipna=(var in LIFETIME_VARS)
                )
            except Exception as e:
                print(f'  Monthly aggregation failed for {var} in {region}: {e}')
                ts[var] = {}
        frames = []
        valid_sets = [set(ts[var]) for var in ts if ts[var]]
        if not valid_sets:
            monthly[region] = pd.DataFrame()
            continue
        common_models = sorted(set.intersection(*valid_sets))
        for model in common_models:
            df = pd.DataFrame({
                'model': model,
                'time': pd.to_datetime(ts['MEC'][model].time.values),
                'MEC': ts['MEC'][model].values,
            })
            for var in ['SSA', 'MAC', 'AE', 'lifetime']:
                if model in ts.get(var, {}):
                    df[var] = ts[var][model].values
            frames.append(df)
        monthly[region] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return monthly


monthly_region = build_monthly_region_data(REGIONS, ['MEC', 'SSA', 'MAC', 'AE', 'lifetime'])
for region, df in monthly_region.items():
    print(f'{region}: {len(df)} monthly rows ({df.model.nunique() if not df.empty else 0} models)')


if not all(df.empty for df in monthly_region.values()):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (region, df) in enumerate(monthly_region.items()):
        if df.empty or 'SSA' not in df.columns or 'MAC' not in df.columns:
            continue
        ax = axes[idx]
        ax.scatter(df['SSA'], df['MAC'], s=25, alpha=0.3, color=colors[idx], edgecolors='none')
        ax.set_xlabel('SSA', fontweight='bold')
        ax.set_ylabel('MAC (m² g⁻¹)', fontweight='bold')
        ax.set_title(f'{region.upper()} — monthly points', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.text(0.03, 0.97, f"{df.model.nunique()} models × {df.time.nunique()} months",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('SSA vs MAC — monthly regional means', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = project_root / 'notebooks' / 'error_attribution_SSA_MAC_monthly.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {out.name}')
else:
    print('No monthly data available.')

if region_data:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (region, df) in enumerate(region_data.items()):
        ax = axes[idx]
        ax.scatter(df['AE'], df['lifetime'], s=100, alpha=0.6, color=colors[idx],
                   edgecolors='black', linewidth=1.5)
        for _, row in df.iterrows():
            ax.annotate(row['model'].replace('_', '\n')[:12],
                        (row['AE'], row['lifetime']), fontsize=7, alpha=0.7,
                        xytext=(2, 2), textcoords='offset points')
        ax.set_xlabel('Angström Exponent (AE)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Lifetime (days)', fontsize=11, fontweight='bold')
        ax.set_title(f'{region.upper()} — lifetime vs AE', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('Lifetime vs AE by region — proxy for removal and size distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = project_root / 'notebooks' / 'error_attribution_lifetime_AE.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {out.name}')
else:
    print('No region data available.')

if region_data:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (region, df) in enumerate(region_data.items()):
        ax = axes[idx]
        ax.scatter(df['MAC'], df['lifetime'], s=100, alpha=0.6, color=colors[idx],
                   edgecolors='black', linewidth=1.5)
        for _, row in df.iterrows():
            ax.annotate(row['model'].replace('_', '\n')[:12],
                        (row['MAC'], row['lifetime']), fontsize=7, alpha=0.7,
                        xytext=(2, 2), textcoords='offset points')
        ax.set_xlabel('MAC (m² g⁻¹)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Lifetime (days)', fontsize=11, fontweight='bold')
        ax.set_title(f'{region.upper()} — lifetime vs MAC', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('Lifetime vs MAC by region — removal vs absorption efficiency', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = project_root / 'notebooks' / 'error_attribution_lifetime_MAC.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {out.name}')
else:
    print('No region data available.')

precip_dir = project_root / 'Data' / 'Prec'
gpcp_files = sorted(precip_dir.glob('GPCPMON_L3_2010*.nc4'))
print(f'Found {len(gpcp_files)} GPCP monthly files.')

gpcp_list = []
for f in gpcp_files:
    ds = xr.open_dataset(f)
    # Convert longitude from -180..180 to 0..360
    ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
    gpcp_list.append(ds[['sat_gauge_precip']])

precip = xr.concat(gpcp_list, dim='time')
# Normalize time to first-of-month so it matches the model data
precip = precip.assign_coords(time=pd.DatetimeIndex(precip.time.values).normalize().values)
print('Precipitation dataset:')
print(precip)
print(f'Time range: {precip.time.values[0]} to {precip.time.values[-1]}')

def precip_for_model_grid(precip_da, target_da):
    """Regrid GPCP precipitation onto the target model's lat/lon grid."""
    return precip_da.interp(lat=target_da.lat, lon=target_da.lon, method='linear')


# Aggregate precipitation for each region and each model's own grid
precip_region = {region: {} for region in REGIONS}
for region in REGIONS:
    mask = masks[region]
    for m in models:
        if data[m].get('od550aer') is None:
            continue
        target_da = data[m]['od550aer']
        precip_model = precip_for_model_grid(precip['sat_gauge_precip'], target_da)
        # Select fire-season months
        tslice = REGIONS[region]['time_slice']
        precip_season = precip_model.sel(time=slice(tslice[0], tslice[1]))
        # Build a tiny dataset-like dict for aggregate_models
        try:
            val = ct.regional_aggregate(
                precip_season,
                mask,
                spatial='mean',
                edge_weighted=REGIONS[region]['edge_weighted'],
                time_slice=None,  # already sliced
                temporal='mean',
                return_time_series=False,
            )
            precip_region[region][m] = float(val)
        except Exception as e:
            print(f'Precip aggregation failed for {m} {region}: {e}')

for region in REGIONS:
    vals = list(precip_region[region].values())
    print(f'{region}: {len(vals)} models, mean precip = {np.mean(vals):.2f} mm/day' if vals else f'{region}: no data')


for region, df in region_data.items():
    df['precip'] = df['model'].map(precip_region[region])
    print(f'{region}: precipitation available for {df["precip"].notna().sum()} / {len(df)} models')

# Drop rows without precipitation for the precipitation-sensitive plots
region_data_precip = {r: df.dropna(subset=['precip']).copy() for r, df in region_data.items()}
for region, df in region_data_precip.items():
    print(f'{region}: {len(df)} models with precipitation')

if any(not df.empty for df in region_data_precip.values()):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (region, df) in enumerate(region_data_precip.items()):
        if df.empty or 'SSA' not in df.columns or 'precip' not in df.columns:
            continue
        df['SSA_error'] = df['SSA'] - df['SSA'].mean()
        df['MAC_error'] = df['MAC'] - df['MAC'].mean()

        ax = axes[idx]
        sc = ax.scatter(df['precip'], df['SSA_error'], s=100, c=colors[idx], alpha=0.6,
                        edgecolors='black', linewidth=1.5)

        # Linear fit
        x = df['precip'].values
        y = df['SSA_error'].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() > 2:
            slope, intercept, r, _, _ = linregress(x[valid], y[valid])
            x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, '--', color='black', alpha=0.7)
            ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        for _, row in df.iterrows():
            ax.annotate(row['model'].replace('_', '\n')[:12],
                        (row['precip'], row['SSA_error']), fontsize=7, alpha=0.7,
                        xytext=(2, 2), textcoords='offset points')

        ax.set_xlabel('GPCP precipitation (mm/day)', fontsize=11, fontweight='bold')
        ax.set_ylabel('SSA error (model - mean)', fontsize=11, fontweight='bold')
        ax.set_title(f'{region.upper()} — SSA error vs precipitation', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='gray', linewidth=0.8)

    plt.suptitle('Model SSA error vs regional precipitation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = project_root / 'notebooks' / 'error_attribution_SSA_error_vs_precip.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {out.name}')
else:
    print('No precipitation data available for plotting.')

if any(not df.empty for df in region_data_precip.values()):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (region, df) in enumerate(region_data_precip.items()):
        ax = axes[idx]
        sc = ax.scatter(df['AE'], df['lifetime'], s=120, c=df['precip'],
                        cmap='YlGnBu', alpha=0.8, edgecolors='black', linewidth=0.5,
                        vmin=df['precip'].min(), vmax=df['precip'].max())
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('GPCP precip (mm/day)', fontsize=10, fontweight='bold')

        for _, row in df.iterrows():
            ax.annotate(row['model'].replace('_', '\n')[:12],
                        (row['AE'], row['lifetime']), fontsize=7, alpha=0.7,
                        xytext=(2, 2), textcoords='offset points')

        ax.set_xlabel('Angström Exponent (AE)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Lifetime (days)', fontsize=11, fontweight='bold')
        ax.set_title(f'{region.upper()} — lifetime vs AE colored by precipitation', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('Lifetime vs AE colored by precipitation — wet vs dry regimes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = project_root / 'notebooks' / 'error_attribution_lifetime_AE_precip.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {out.name}')
else:
    print('No precipitation data available for plotting.')

regime_stats = []
for region, df in monthly_region.items():
    if df.empty or 'precip' not in df.columns or 'SSA' not in df.columns or 'MAC' not in df.columns:
        continue
    median_precip = df['precip'].median()
    wet = df[df['precip'] > median_precip]
    dry = df[df['precip'] <= median_precip]
    for regime, sub in [('wet', wet), ('dry', dry)]:
        if sub.empty:
            continue
        regime_stats.append({
            'region': region,
            'regime': regime,
            'n': len(sub),
            'SSA_mean': sub['SSA'].mean(),
            'SSA_std': sub['SSA'].std(),
            'MAC_mean': sub['MAC'].mean(),
            'MAC_std': sub['MAC'].std(),
            'lifetime_mean': sub['lifetime'].mean() if 'lifetime' in sub.columns else np.nan,
            'AE_mean': sub['AE'].mean() if 'AE' in sub.columns else np.nan,
        })

regime_df = pd.DataFrame(regime_stats)
if not regime_df.empty:
    print(regime_df.round(3).to_string(index=False))
    regime_df.to_csv(project_root / 'notebooks' / 'error_attribution_regime_stats.csv', index=False)
else:
    print('No regime statistics available (precipitation not merged into monthly data yet).')

def build_monthly_precipitation(monthly_dict, precip_da):
    """Add a regional monthly precipitation column to each monthly DataFrame."""
    for region, df in monthly_region.items():
        if df.empty:
            df['precip'] = []
            continue

        # Precompute precipitation time series for every model in this region
        precip_ts = {}
        cfg = REGIONS[region]
        for model in df['model'].unique():
            if data[model].get('od550aer') is None:
                precip_ts[model] = None
                continue
            target_da = data[model]['od550aer']
            pm = precip_for_model_grid(precip_da, target_da)
            try:
                ts = ct.regional_aggregate(
                    pm,
                    masks[region],
                    spatial='mean',
                    edge_weighted=cfg['edge_weighted'],
                    time_slice=cfg['time_slice'],
                    temporal='mean',
                    return_time_series=True,
                )
                precip_ts[model] = ts
            except Exception as e:
                print(f'  Monthly precip failed for {model} {region}: {e}')
                precip_ts[model] = None

        precip_vals = []
        for _, row in df.iterrows():
            model = row['model']
            t = pd.to_datetime(row['time'])
            ts = precip_ts.get(model)
            if ts is None or 'time' not in ts.dims:
                precip_vals.append(np.nan)
                continue
            try:
                val = float(ts.sel(time=t, method='nearest').values)
            except Exception:
                val = np.nan
            precip_vals.append(val)
        df['precip'] = precip_vals
    return monthly_dict


monthly_region = build_monthly_precipitation(monthly_region, precip['sat_gauge_precip'])
for region, df in monthly_region.items():
    if df.empty or 'precip' not in df.columns:
        print(f'{region}: no monthly data')
        continue
    print(f'{region}: {df["precip"].notna().sum()}/{len(df)} monthly rows with precipitation')


# Recompute regime statistics with monthly precipitation
regime_stats = []
for region, df in monthly_region.items():
    if df.empty or 'precip' not in df.columns or 'SSA' not in df.columns or 'MAC' not in df.columns:
        continue
    median_precip = df['precip'].median()
    wet = df[df['precip'] > median_precip]
    dry = df[df['precip'] <= median_precip]
    for regime, sub in [('wet', wet), ('dry', dry)]:
        if sub.empty:
            continue
        regime_stats.append({
            'region': region,
            'regime': regime,
            'n': len(sub),
            'SSA_mean': sub['SSA'].mean(),
            'SSA_std': sub['SSA'].std(),
            'MAC_mean': sub['MAC'].mean(),
            'MAC_std': sub['MAC'].std(),
            'lifetime_mean': sub['lifetime'].mean() if 'lifetime' in sub.columns else np.nan,
            'AE_mean': sub['AE'].mean() if 'AE' in sub.columns else np.nan,
        })

regime_df = pd.DataFrame(regime_stats)
if not regime_df.empty:
    print(regime_df.round(3).to_string(index=False))
    out_csv = project_root / 'notebooks' / 'error_attribution_regime_stats.csv'
    regime_df.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv.name}')
else:
    print('No regime statistics available.')

if not all(df.empty for df in monthly_region.values()):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e']

    for idx, (region, df) in enumerate(monthly_region.items()):
        if df.empty or 'precip' not in df.columns or 'SSA' not in df.columns or 'MAC' not in df.columns:
            continue
        median_precip = df['precip'].median()
        wet = df[df['precip'] > median_precip]
        dry = df[df['precip'] <= median_precip]

        ax = axes[idx]
        ax.scatter(wet['SSA'], wet['MAC'], s=25, alpha=0.4, color=colors[0], label='wet', edgecolors='none')
        ax.scatter(dry['SSA'], dry['MAC'], s=25, alpha=0.4, color=colors[1], label='dry', edgecolors='none')
        ax.set_xlabel('SSA', fontsize=11, fontweight='bold')
        ax.set_ylabel('MAC (m² g⁻¹)', fontsize=11, fontweight='bold')
        ax.set_title(f'{region.upper()} — wet vs dry months', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(title='Regime', loc='best')

    plt.suptitle('SSA vs MAC split by precipitation regime (monthly means)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = project_root / 'notebooks' / 'error_attribution_wet_dry_SSA_MAC.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {out.name}')
else:
    print('No monthly data with precipitation available.')

fig = plt.figure(figsize=(16, 14))

# (a) SSA-MAC regional means
ax1 = plt.subplot(2, 2, 1)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for idx, (region, df) in enumerate(region_data.items()):
    ax1.scatter(df['SSA'], df['MAC'], s=100, alpha=0.6, color=colors[idx], label=region.upper())
    ax1.plot(df['SSA'], df['MAC'], '--', color=colors[idx], alpha=0.3)
ax1.set_xlabel('SSA', fontsize=12, fontweight='bold')
ax1.set_ylabel('MAC (m² g⁻¹)', fontsize=12, fontweight='bold')
ax1.set_title('(a) SSA-MAC by region', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(title='Region', loc='best')

# (b) SSA error vs precipitation
ax2 = plt.subplot(2, 2, 2)
for idx, (region, df) in enumerate(region_data_precip.items()):
    if df.empty or 'precip' not in df.columns:
        continue
    df['SSA_error'] = df['SSA'] - df['SSA'].mean()
    ax2.scatter(df['precip'], df['SSA_error'], s=100, alpha=0.6, color=colors[idx], label=region.upper())
ax2.set_xlabel('GPCP precipitation (mm/day)', fontsize=12, fontweight='bold')
ax2.set_ylabel('SSA error (model - mean)', fontsize=12, fontweight='bold')
ax2.set_title('(b) SSA error vs precipitation', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(0, color='gray', linewidth=0.8)
ax2.legend(title='Region', loc='best')

# (c) Lifetime vs AE colored by precipitation
ax3 = plt.subplot(2, 2, 3)
sc = None
for idx, (region, df) in enumerate(region_data_precip.items()):
    if df.empty or 'precip' not in df.columns:
        continue
    sc = ax3.scatter(df['AE'], df['lifetime'], s=100, c=df['precip'], cmap='YlGnBu',
                     alpha=0.8, edgecolors='black', linewidth=0.5)
if sc is not None:
    plt.colorbar(sc, ax=ax3, label='Precip (mm/day)', fraction=0.046, pad=0.04)
ax3.set_xlabel('AE', fontsize=12, fontweight='bold')
ax3.set_ylabel('Lifetime (days)', fontsize=12, fontweight='bold')
ax3.set_title('(c) Lifetime vs AE colored by precipitation', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')

# (d) wet vs dry boxplots of SSA
ax4 = plt.subplot(2, 2, 4)
box_data = []
box_labels = []
for region, df in monthly_region.items():
    if df.empty or 'precip' not in df.columns or 'SSA' not in df.columns:
        continue
    median_precip = df['precip'].median()
    wet = df[df['precip'] > median_precip]['SSA'].dropna()
    dry = df[df['precip'] <= median_precip]['SSA'].dropna()
    if not wet.empty and not dry.empty:
        box_data.extend([wet, dry])
        box_labels.extend([f'{region}\nwet', f'{region}\ndry'])

if box_data:
    bp = ax4.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e'] * (len(box_data) // 2)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_ylabel('SSA', fontsize=12, fontweight='bold')
    ax4.set_title('(d) SSA distribution by precipitation regime', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Aerosol error attribution — summary (Zhong et al. 2023 framework)', fontsize=15, fontweight='bold')
plt.tight_layout()
out = project_root / 'notebooks' / 'error_attribution_summary.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved: {out.name}')


summary_rows = []
for region, df in region_data.items():
    row = {
        'region': region.upper(),
        'n_models': len(df),
        'SSA_mean': df['SSA'].mean(), 'SSA_std': df['SSA'].std(),
        'MAC_mean': df['MAC'].mean(), 'MAC_std': df['MAC'].std(),
        'MEC_mean': df['MEC'].mean(), 'MEC_std': df['MEC'].std(),
        'AE_mean': df['AE'].mean(), 'AE_std': df['AE'].std(),
        'lifetime_mean': df['lifetime'].mean(), 'lifetime_std': df['lifetime'].std(),
    }
    if 'precip' in df.columns and df['precip'].notna().any():
        row['precip_mean'] = df['precip'].mean()
    summary_rows.append(row)

summary_table = pd.DataFrame(summary_rows)
print(summary_table.round(3).to_string(index=False))
out_csv = project_root / 'notebooks' / 'error_attribution_summary_table.csv'
summary_table.to_csv(out_csv, index=False)
print(f'Saved: {out_csv.name}')

# Export per-model per-region data for further analysis
all_data = []
for region, df in region_data.items():
    df = df.copy()
    df['region'] = region
    all_data.append(df)

if all_data:
    full_df = pd.concat(all_data, ignore_index=True)
    out_csv = project_root / 'notebooks' / 'error_attribution_model_region_data.csv'
    full_df.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv.name} ({len(full_df)} rows)')
    print(full_df.head().to_string(index=False))
else:
    print('No data to export.')

print("\n" + "="*80)
print("NOTEBOOK COMPLETE")
print("="*80)
print("\nGenerated files in notebooks/:")
out_dir = project_root / 'notebooks'
for f in sorted(out_dir.glob('error_attribution*')):
    print(f'  {f.name}')
print("="*80)
