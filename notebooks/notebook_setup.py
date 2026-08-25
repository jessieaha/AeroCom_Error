
import os
import sys
import gc
import glob

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
try:
    project_root = Path.cwd().parent
except NameError:
    project_root = Path('/scistor/guest/gbb083/AeroCom')
py_dir = project_root / 'py'
if str(py_dir) not in sys.path:
    sys.path.insert(0, str(py_dir))
sys.path.append(py_dir)
import cameo_toolbox as ct
import functions
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
    # Nature / AOD paper box (Zhong et al.); same as cameo_toolbox._REGION_DEFINITIONS.
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


def create_analysis_masks(template, region_names, surface_type=None):
    """Build region masks for ``region_names`` using ``REGIONS`` lon/lat boxes.

    Passes explicit ``lon_range``/``lat_range`` so regions absent from
    ``cameo_toolbox._REGION_DEFINITIONS`` (e.g. se_asia, boreal_na) still work.
    """
    masks = {}
    for name in region_names:
        if name not in REGIONS:
            raise KeyError(f'Region {name!r} not in notebook_setup.REGIONS')
        cfg = REGIONS[name]
        masks[name] = ct.create_region_mask(
            template,
            name=name,
            lon_range=cfg['lon_range'],
            lat_range=cfg['lat_range'],
            surface_type=(
                surface_type if surface_type is not None
                else cfg.get('surface_type', 'all')
            ),
            mask_registry=masks,
        )
    return masks


def aggregate_region(model_dict, var_name, region_name, masks, return_time_series=False, skipna=False):
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


def _lifetime_from_load_flux(load_dict, flux_dict, label, region):
    """τ = load / (flux * 86400) in days for models present in both dicts."""
    out = {}
    for model in load_dict:
        if model not in flux_dict:
            continue
        try:
            out[model] = load_dict[model] / (flux_dict[model] * 3600 * 24)
        except Exception as e:
            print(f'  Failed post-aggregation {label} for {model} {region}: {e}')
    return out


def compute_derived_after_aggregation(
    monthly_dict,
    seasonal_dict,
    DERIVED_VAR_AFTER_AGG,
    use_deposition_for_lifetime=False,
):
    """Compute lifetime-like variables from aggregated load/(emission|deposition).

    When ``use_deposition_for_lifetime`` is True, flux is ``dep_BC_OA`` /
    ``dep_total`` (require_all sums built upstream). Otherwise emission fluxes
    ``emi_BC_OA`` / ``emi_total`` are used. Lifetime is load / (flux * 86400) days.

    Regions are taken from the aggregation dict keys so callers with a local
    REGIONS map (e.g. AOD notebook) are not forced onto ``setup.REGIONS``.
    """
    lifetime_after_agg = 'lifetime' in DERIVED_VAR_AFTER_AGG
    lifetime_bcoa_after_agg = lifetime_after_agg or 'lifetime_BC_OA' in DERIVED_VAR_AFTER_AGG
    method = 'deposition' if use_deposition_for_lifetime else 'emission'
    print(f'\nPost-aggregation lifetimes using load/{method} '
          f'(USE_DEPOSITION_FOR_LIFETIME={use_deposition_for_lifetime})')

    if lifetime_bcoa_after_agg:
        flux_key = 'dep_BC_OA' if use_deposition_for_lifetime else 'emi_BC_OA'
        for agg in (monthly_dict, seasonal_dict):
            for region in agg:
                load = agg[region].get('load_BC_OA', {})
                flux = agg[region].get(flux_key, {})
                agg[region]['lifetime_BC_OA'] = _lifetime_from_load_flux(
                    load, flux, 'lifetime_BC_OA', region
                )

    if lifetime_after_agg:
        flux_key = 'dep_total' if use_deposition_for_lifetime else 'emi_total'
        for agg in (monthly_dict, seasonal_dict):
            for region in agg:
                load = agg[region].get('load_total', {})
                flux = agg[region].get(flux_key, {})
                agg[region]['lifetime'] = _lifetime_from_load_flux(
                    load, flux, 'lifetime', region
                )

    # AOD notebook: post-aggregation mass extinction = AOD / load_total.
    if 'MEC' in DERIVED_VAR_AFTER_AGG:
        for agg in (monthly_dict, seasonal_dict):
            for region in agg:
                aod = agg[region].get('od550aer', {})
                load = agg[region].get('load_total', {})
                out = {}
                for model in set(aod) & set(load):
                    try:
                        out[model] = aod[model] / load[model]
                    except Exception as e:
                        print(f'  Failed post-aggregation MEC for {model} {region}: {e}')
                agg[region]['MEC'] = out


def _scalarize_lifetime(val):
    """Convert aggregated lifetime entry to a finite float, or None."""
    try:
        v = float(val) if not hasattr(val, 'values') else float(np.asarray(val).mean())
    except Exception:
        return None
    return v if np.isfinite(v) else None


def print_global_lifetimes(seasonal_dict, var='lifetime_BC_OA', method_label=None):
    """Print seasonal/regional lifetime for the global box (or spatial-mean fallback)."""
    label = method_label or var
    global_map = (seasonal_dict.get('global') or {}).get(var) or {}
    print(f'\n--- Global lifetime ({label}) ---')
    if global_map:
        rows = []
        for model, val in sorted(global_map.items()):
            v = _scalarize_lifetime(val)
            if v is None:
                continue
            rows.append((model, v))
            print(f'  {model}: {v:.3f} days')
        if rows:
            vals = [v for _, v in rows]
            print(f'  n={len(vals)}, mean={np.mean(vals):.3f} d, '
                  f'median={np.median(vals):.3f} d, '
                  f'min={np.min(vals):.3f} d, max={np.max(vals):.3f} d')
        else:
            print('  No finite global lifetimes available.')
        return rows

    # Fallback: unweighted mean of available regional seasonal lifetimes.
    print('  No global region lifetimes; spatial-mean over available regions:')
    model_vals = {}
    for region, region_data in seasonal_dict.items():
        for model, val in (region_data.get(var) or {}).items():
            v = _scalarize_lifetime(val)
            if v is None:
                continue
            model_vals.setdefault(model, []).append(v)
    rows = []
    for model in sorted(model_vals):
        v = float(np.mean(model_vals[model]))
        rows.append((model, v))
        print(f'  {model}: {v:.3f} days (mean over {len(model_vals[model])} regions)')
    if rows:
        vals = [v for _, v in rows]
        print(f'  n={len(vals)}, mean={np.mean(vals):.3f} d, '
              f'median={np.median(vals):.3f} d')
    else:
        print('  No finite lifetimes available in any region.')
    return rows


def compare_emission_vs_deposition_lifetimes(
    seasonal_dict,
    regions=None,
    load_key='load_BC_OA',
    emi_key='emi_BC_OA',
    dep_key='dep_BC_OA',
    out_csv=None,
):
    """Side-by-side emission vs deposition lifetimes from aggregated load/flux.

    Returns a DataFrame with per-model/region τ_emi, τ_dep, difference, and ratio.
    Models missing any required flux for a method get NaN for that method (same
    require_all semantics as the upstream summed products).
    """
    if regions is None:
        regions = [r for r in REGIONS if r in seasonal_dict]
    rows = []
    for region in regions:
        load = (seasonal_dict.get(region) or {}).get(load_key) or {}
        emi = (seasonal_dict.get(region) or {}).get(emi_key) or {}
        dep = (seasonal_dict.get(region) or {}).get(dep_key) or {}
        models = sorted(set(load) | set(emi) | set(dep))
        for model in models:
            lt_emi = lt_dep = np.nan
            if model in load and model in emi:
                try:
                    lt_emi = float(load[model] / (emi[model] * 3600 * 24))
                except Exception:
                    lt_emi = np.nan
            if model in load and model in dep:
                try:
                    lt_dep = float(load[model] / (dep[model] * 3600 * 24))
                except Exception:
                    lt_dep = np.nan
            if not (np.isfinite(lt_emi) or np.isfinite(lt_dep)):
                continue
            diff = lt_dep - lt_emi if (np.isfinite(lt_emi) and np.isfinite(lt_dep)) else np.nan
            ratio = (
                lt_dep / lt_emi
                if (np.isfinite(lt_emi) and np.isfinite(lt_dep) and lt_emi != 0)
                else np.nan
            )
            rows.append({
                'region': region,
                'model': model,
                'lifetime_emission_days': lt_emi,
                'lifetime_deposition_days': lt_dep,
                'dep_minus_emi_days': diff,
                'dep_over_emi': ratio,
            })
    df = pd.DataFrame(rows)
    print('\n--- Emission vs deposition lifetime comparison ---')
    if df.empty:
        print('  No overlapping load/flux data to compare.')
        return df
    n_emi = int(df['lifetime_emission_days'].notna().sum())
    n_dep = int(df['lifetime_deposition_days'].notna().sum())
    n_both = int(
        (df['lifetime_emission_days'].notna() & df['lifetime_deposition_days'].notna()).sum()
    )
    print(f'  Rows: {len(df)} (model×region)')
    print(f'  Available emission-based:    {n_emi}')
    print(f'  Available deposition-based:  {n_dep}')
    print(f'  Available both methods:      {n_both}')
    if n_both:
        both = df.dropna(subset=['lifetime_emission_days', 'lifetime_deposition_days'])
        print(f'  Mean τ_emi={both["lifetime_emission_days"].mean():.3f} d, '
              f'τ_dep={both["lifetime_deposition_days"].mean():.3f} d, '
              f'mean(dep−emi)={both["dep_minus_emi_days"].mean():.3f} d')
    # Per-region model counts
    for region in regions:
        sub = df[df['region'] == region]
        if sub.empty:
            continue
        print(f'  {region}: emi={sub["lifetime_emission_days"].notna().sum()}, '
              f'dep={sub["lifetime_deposition_days"].notna().sum()}, '
              f'both={(sub["lifetime_emission_days"].notna() & sub["lifetime_deposition_days"].notna()).sum()}')
    missing_dep_models = sorted(
        df.loc[df['lifetime_deposition_days'].isna() & df['lifetime_emission_days'].notna(), 'model']
        .unique()
        .tolist()
    )
    if missing_dep_models:
        print(f'  Models with emission but no deposition lifetime (any region): '
              f'{missing_dep_models}')
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f'  Saved: {out_csv}')
    return df


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
    # Preserve unfiltered BC+OA lifetimes for diagnostics / sensitivity.
    global lifetime_BC_OA_raw
    regions = list(seasonal_dict.keys()) if seasonal_dict else list(REGIONS)
    lifetime_BC_OA_raw = {region: {} for region in regions}
    for region in regions:
        for model, val in (seasonal_dict.get(region, {}).get('lifetime_BC_OA') or {}).items():
            try:
                v = float(val) if not hasattr(val, 'values') else float(val.values)
            except Exception:
                continue
            lifetime_BC_OA_raw[region][model] = v
    for var in ['lifetime', 'lifetime_BC_OA']:
        for agg in (monthly_dict, seasonal_dict):
            for region in agg:
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


def positive_deposition(da):
    """Return deposition as a positive mass sink (handles signed fluxes)."""
    return da.where(da > 0, -da)


def sum_deposition_datasets(model_data, keys, out_name, require_all=True):
    """Sum deposition species into a positive-sink Dataset.

    Same require_all semantics as ``sum_datasets``: if any required species is
    missing, return None so lifetime is skipped for that model (no partial sums).
    Each component is converted to a positive sink before summing.
    """
    missing = [k for k in keys if model_data.get(k) is None]
    if require_all and missing:
        return None
    dsets = [model_data[k] for k in keys if model_data.get(k) is not None]
    if not dsets:
        return None
    arrays = [positive_deposition(d[list(d.data_vars)[0]]) for d in dsets]
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