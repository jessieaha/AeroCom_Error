
import os
import sys
import gc
import glob

import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import xarray as xr
try:
    project_root = Path.cwd().parent
except NameError:
    project_root = Path('/scistor/guest/gbb083/AeroCom')
py_dir = project_root / 'py'
_py_dir_here = Path(__file__).resolve().parent.parent / 'py'
for _p in (py_dir, _py_dir_here):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import cameo_toolbox as ct
import functions
import regions as region_registry
# ==============================================================================
# GLOBAL CONFIGURATIONS
# ==============================================================================
# Boxes and seasons live in py/regions.py. REGIONS keeps the original AeroCom
# 2010 biomass-burning set (notebooks loop over all of it); ALL_REGIONS adds
# the sea-salt and dust boxes.
BB_REGION_NAMES = tuple(
    n for n in (
        'global', 'africa', 'amazon', 'outflow_af', 'se_asia', 'boreal_na',
        'eastern_siberia', 'west_russia', 'boreal_na_west', 'indonesia_peatland',
    )
    if n in region_registry.REGIONS
)
REGIONS = region_registry.aerocom_view(BB_REGION_NAMES, year=2010)
ALL_REGIONS = region_registry.aerocom_view(year=2010)


def _is_agg_frame(obj):
    return isinstance(obj, pd.DataFrame) and {'region', 'model'}.issubset(set(obj.columns))


def _as_scalar(value):
    if value is None:
        return np.nan
    try:
        if hasattr(value, 'values') and not isinstance(value, (pd.Series, pd.DataFrame)):
            arr = np.asarray(value.values)
            if arr.size == 0:
                return np.nan
            return float(np.nanmean(arr))
        return float(value)
    except Exception:
        return np.nan


def _series_from_monthly_value(value):
    """Normalize monthly aggregated values to a pandas Series indexed by month."""
    if value is None:
        return pd.Series(dtype=float)

    if isinstance(value, xr.Dataset):
        if not value.data_vars:
            return pd.Series(dtype=float)
        value = value[list(value.data_vars)[0]]

    if isinstance(value, xr.DataArray):
        if 'time' not in value.dims:
            v = _as_scalar(value)
            return pd.Series(dtype=float) if not np.isfinite(v) else pd.Series([v])
        da = value
        if da.ndim > 1:
            non_time_dims = [d for d in da.dims if d != 'time']
            if non_time_dims:
                da = da.mean(dim=non_time_dims, skipna=True)
        vals = np.asarray(da.values).reshape(-1)
        times = pd.to_datetime(np.asarray(da['time'].values), errors='coerce')
        idx = pd.PeriodIndex(times, freq='M')
        return pd.Series(vals, index=idx, dtype=float)

    if isinstance(value, pd.Series):
        s = pd.to_numeric(value, errors='coerce')
        if isinstance(s.index, pd.PeriodIndex):
            if s.index.freq is None:
                s.index = pd.PeriodIndex(s.index.astype(str), freq='M')
            return s.astype(float)
        dt = pd.to_datetime(s.index, errors='coerce')
        return pd.Series(s.values, index=pd.PeriodIndex(dt, freq='M'), dtype=float)

    if isinstance(value, Mapping):
        if not value:
            return pd.Series(dtype=float)
        idx = pd.PeriodIndex(pd.to_datetime(list(value.keys()), errors='coerce'), freq='M')
        vals = pd.to_numeric(list(value.values()), errors='coerce')
        return pd.Series(vals, index=idx, dtype=float)

    v = _as_scalar(value)
    return pd.Series(dtype=float) if not np.isfinite(v) else pd.Series([v])


def seasonal_dict_to_dataframe(model_seasonal):
    """
    Convert dict[region][variable][model] -> DataFrame with columns:
    region, model, <all variables>.
    """
    rows = {}
    for region, var_map in (model_seasonal or {}).items():
        for var, model_map in (var_map or {}).items():
            for model, value in (model_map or {}).items():
                key = (str(region), str(model))
                if key not in rows:
                    rows[key] = {'region': str(region), 'model': str(model)}
                rows[key][str(var)] = _as_scalar(value)
    if not rows:
        return pd.DataFrame(columns=['region', 'model'])
    df = pd.DataFrame(rows.values())
    return df.sort_values(['region', 'model']).reset_index(drop=True)


def seasonal_dataframe_to_dict(df):
    """Convert seasonal DataFrame back to dict[region][variable][model]."""
    if df is None or df.empty:
        return {}
    required = {'region', 'model'}
    if not required.issubset(df.columns):
        raise ValueError("Seasonal DataFrame must include 'region' and 'model' columns.")
    value_cols = [c for c in df.columns if c not in ('region', 'model')]
    out = {}
    for _, row in df.iterrows():
        region = str(row['region'])
        model = str(row['model'])
        out.setdefault(region, {})
        for var in value_cols:
            out[region].setdefault(var, {})
            out[region][var][model] = row[var]
    return out


def monthly_dict_to_dataframe(model_monthly, month_col='month'):
    """
    Convert dict[region][variable][model->timeseries] -> DataFrame with columns:
    region, model, month, <all variables>.
    """
    rows = []
    for region, var_map in (model_monthly or {}).items():
        reg_model_month = {}
        for var, model_map in (var_map or {}).items():
            for model, value in (model_map or {}).items():
                s = _series_from_monthly_value(value)
                if s.empty:
                    continue
                for period, v in s.items():
                    key = (str(region), str(model), str(period))
                    if key not in reg_model_month:
                        reg_model_month[key] = {
                            'region': str(region),
                            'model': str(model),
                            month_col: str(period),
                        }
                    reg_model_month[key][str(var)] = _as_scalar(v)
        rows.extend(reg_model_month.values())
    if not rows:
        return pd.DataFrame(columns=['region', 'model', month_col])
    df = pd.DataFrame(rows)
    if month_col in df.columns:
        df[month_col] = pd.PeriodIndex(df[month_col].astype(str), freq='M')
    return df.sort_values(['region', 'model', month_col]).reset_index(drop=True)


def monthly_dataframe_to_dict(df, month_col='month'):
    """Convert monthly DataFrame back to dict[region][variable][model->pd.Series]."""
    if df is None or df.empty:
        return {}
    required = {'region', 'model', month_col}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Monthly DataFrame must include 'region', 'model', and '{month_col}' columns."
        )
    value_cols = [c for c in df.columns if c not in ('region', 'model', month_col)]
    out = {}
    work = df.copy()
    if not isinstance(work[month_col].dtype, pd.PeriodDtype):
        work[month_col] = pd.PeriodIndex(pd.to_datetime(work[month_col], errors='coerce'), freq='M')
    for (region, model), g in work.groupby(['region', 'model'], sort=False):
        g = g.sort_values(month_col)
        idx = pd.PeriodIndex(g[month_col].astype(str), freq='M')
        for var in value_cols:
            vals = pd.to_numeric(g[var], errors='coerce').to_numpy(dtype=float)
            out.setdefault(str(region), {}).setdefault(var, {})[str(model)] = pd.Series(vals, index=idx)
    return out


def convert_aggregation_to_dataframes(model_monthly, model_seasonal, month_col='month'):
    """Return (monthly_df, seasonal_df) from standard nested dict aggregations."""
    return (
        monthly_dict_to_dataframe(model_monthly, month_col=month_col),
        seasonal_dict_to_dataframe(model_seasonal),
    )


def _coerce_agg_pair(monthly_obj, seasonal_obj, month_col='month'):
    """
    Coerce monthly/seasonal objects to dict form for internal calculations.
    Returns (monthly_dict, seasonal_dict, monthly_was_df, seasonal_was_df).
    """
    monthly_was_df = _is_agg_frame(monthly_obj)
    seasonal_was_df = _is_agg_frame(seasonal_obj)
    monthly_dict = monthly_dataframe_to_dict(monthly_obj, month_col=month_col) if monthly_was_df else monthly_obj
    seasonal_dict = seasonal_dataframe_to_dict(seasonal_obj) if seasonal_was_df else seasonal_obj
    return monthly_dict, seasonal_dict, monthly_was_df, seasonal_was_df


def _restore_agg_object(obj_dict, was_df, is_monthly, month_col='month'):
    if not was_df:
        return obj_dict
    if is_monthly:
        return monthly_dict_to_dataframe(obj_dict, month_col=month_col)
    return seasonal_dict_to_dataframe(obj_dict)


def create_analysis_masks(template, region_names, region = REGIONS , surface_type=None):
    """Build region masks for ``region_names`` using ``REGIONS`` lon/lat boxes.

    Passes explicit ``lon_range``/``lat_range`` so regions absent from
    ``cameo_toolbox._REGION_DEFINITIONS`` (e.g. se_asia, boreal_na) still work.
    """
    masks = {}
    for name in region_names:
        if name not in region:
            raise KeyError(f'Region {name!r} not in notebook_setup.REGIONS')
        cfg = region[name]
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



def aggregate_region(model_dict, var_name, region_name, masks, region = REGIONS ,return_time_series=False, skipna=False):
    """Spatially aggregate `var_name` for every model."""
    cfg = region[region_name]
    months = cfg.get('season_months')
    result = {}
    for model, model_data in model_dict.items():
        if var_name not in model_data or model_data[var_name] is None:
            continue
        try:
            field = model_data[var_name]
            # Monthly series keep every month; only the seasonal mean is month-filtered.
            if months is not None and not return_time_series and 'time' in field.dims:
                field = field.where(field['time'].dt.month.isin(list(months)), drop=True)
            result[model] = ct.regional_aggregate(
                field, masks[region_name],
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
    month_col='month',
):
    """Compute lifetime-like variables from aggregated load/(emission|deposition).

    When ``use_deposition_for_lifetime`` is True, flux is ``dep_BC_OA`` /
    ``dep_total`` (require_all sums built upstream). Otherwise emission fluxes
    ``emi_BC_OA`` / ``emi_total`` are used. Lifetime is load / (flux * 86400) days.

    Regions are taken from the aggregation dict keys so callers with a local
    REGIONS map (e.g. AOD notebook) are not forced onto ``setup.REGIONS``.
    """
    monthly_dict, seasonal_dict, monthly_was_df, seasonal_was_df = _coerce_agg_pair(
        monthly_dict, seasonal_dict, month_col=month_col
    )

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

    monthly_out = _restore_agg_object(
        monthly_dict, monthly_was_df, is_monthly=True, month_col=month_col
    )
    seasonal_out = _restore_agg_object(
        seasonal_dict, seasonal_was_df, is_monthly=False, month_col=month_col
    )

    if monthly_was_df or seasonal_was_df:
        return monthly_out, seasonal_out


def _scalarize_lifetime(val):
    """Convert aggregated lifetime entry to a finite float, or None."""
    try:
        v = float(val) if not hasattr(val, 'values') else float(np.asarray(val).mean())
    except Exception:
        return None
    return v if np.isfinite(v) else None


def print_global_lifetimes(seasonal_dict, var='lifetime_BC_OA', method_label=None):
    """Print seasonal/regional lifetime for the global box (or spatial-mean fallback)."""
    if _is_agg_frame(seasonal_dict):
        seasonal_dict = seasonal_dataframe_to_dict(seasonal_dict)

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
    if _is_agg_frame(seasonal_dict):
        seasonal_dict = seasonal_dataframe_to_dict(seasonal_dict)

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
        '_AP3-CTRL2016-PD', 
        #  '_AP3-CTRL-2010','_AP3-CTRL2016','_AP3-CTRL2019'
         '_AP3-CTRL', '-met2010',
    ):
        name = name.replace(suffix, '')
    return name

def filter_lifetime_after_agg(monthly_dict, seasonal_dict, max_days=None, min_days=None, month_col='month'):
    """Mask unphysical after-aggregation lifetime values with NaN."""
    monthly_dict, seasonal_dict, monthly_was_df, seasonal_was_df = _coerce_agg_pair(
        monthly_dict, seasonal_dict, month_col=month_col
    )

    if max_days is None and min_days is None:
        if monthly_was_df or seasonal_was_df:
            return (
                _restore_agg_object(monthly_dict, monthly_was_df, is_monthly=True, month_col=month_col),
                _restore_agg_object(seasonal_dict, seasonal_was_df, is_monthly=False, month_col=month_col),
            )
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

    if monthly_was_df or seasonal_was_df:
        return (
            _restore_agg_object(monthly_dict, monthly_was_df, is_monthly=True, month_col=month_col),
            _restore_agg_object(seasonal_dict, seasonal_was_df, is_monthly=False, month_col=month_col),
        )


# ==============================================================================
# DATA NORMALIZATION FUNCTIONS
# ==============================================================================

def normalize_precipitation_units(precip_ds):
    import aerocom_data
    return aerocom_data.normalize_precipitation_units(precip_ds)


def normalize_dataset_time(ds, var_hint=None):
    import aerocom_data
    year = locals()["year"] if "year" in locals() else None
    return aerocom_data.normalize_dataset_time(ds, var_hint=var_hint, year=year)


def sum_datasets(model_data, keys, out_name, require_all=True):
    import aerocom_data
    req = require_all if "require_all" in locals() else False
    return aerocom_data.sum_datasets(model_data, keys, out_name, require_all=req)


def positive_deposition(da):
    import aerocom_data
    return aerocom_data.positive_deposition(da)


def sum_deposition_datasets(model_data, keys, out_name, require_all=True):
    import aerocom_data
    return aerocom_data.sum_deposition_datasets(model_data, keys, out_name, require_all=require_all)


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


# ==============================================================================
# DRY vs WET DEPOSITION FRACTION
# ==============================================================================
def drywet_fraction_table(model_seasonal, regions, dry_key, wet_key, species_label=''):
    """Per-model, per-region dry deposition fraction = dry / (dry + wet).

    ``model_seasonal[region][key]`` is a {model: scalar} dict of regional-mean
    seasonal fluxes. Fluxes are used as magnitudes (sign conventions differ
    between models). Returns a long DataFrame with dry, wet, dry_frac, wet_frac.
    """
    rows = []
    for region in regions:
        reg = model_seasonal.get(region, {})
        dry_d, wet_d = reg.get(dry_key, {}), reg.get(wet_key, {})
        for model in sorted(set(dry_d) & set(wet_d)):
            try:
                dry, wet = abs(_as_scalar(dry_d[model])), abs(_as_scalar(wet_d[model]))
            except Exception:
                continue
            tot = dry + wet
            if not np.isfinite(tot) or tot <= 0:
                continue
            rows.append({
                'region': region, 'model': model, 'species': species_label,
                'dry': dry, 'wet': wet,
                'dry_frac': dry / tot, 'wet_frac': wet / tot,
            })
    return pd.DataFrame(rows, columns=['region', 'model', 'species', 'dry', 'wet', 'dry_frac', 'wet_frac'])


def plot_drywet_fraction_bars(df, regions, title, save_path=None, region_colors=None, show=True):
    """Bar plot of dry/(dry+wet) per model, one panel per region."""
    import matplotlib.pyplot as plt

    if df is None or df.empty:
        print('No dry/wet fraction data to plot.')
        return None
    regions = [r for r in regions if r in set(df['region'])]
    models = sorted(df['model'].unique())
    labels = [short_model_name(m) for m in models]
    x = np.arange(len(models))
    fig, axes = plt.subplots(
        len(regions), 1, figsize=(max(9, 0.55 * len(models) + 3), 3.6 * len(regions)),
        squeeze=False, sharex=True,
    )
    for ax, region in zip(axes[:, 0], regions):
        sub = df[df['region'] == region].set_index('model')['dry_frac'].reindex(models)
        color = (region_colors or {}).get(region, 'tab:blue')
        ax.bar(x, sub.values, color=color, edgecolor='k', linewidth=0.6)
        med = np.nanmedian(sub.values)
        if np.isfinite(med):
            ax.axhline(med, color='k', ls='--', lw=1, label=f'multi-model median = {med:.2f}')
            ax.legend(fontsize=8, frameon=False, loc='upper right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('dry / (dry + wet)')
        ax.set_title(f'{region} (n={int(sub.notna().sum())} models)', fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_path}')
    if show:
        plt.show()
    return fig

###################################
## 