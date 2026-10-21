#!/usr/bin/env python3
"""Build a long-form regional seasonal parquet shared by AAOD / test_models.

Processes **one model at a time** so peak memory stays near a single model's
grids. Writes ``tables/regional_seasonal_aggregates.parquet`` with columns:

    region, dataset, var, source, value, unit, area,
    time_start, time_end, lon_min, lon_max, lat_min, lat_max, surface_type

``source`` values:
  - ``aggregated`` — regional mean of a raw monthly field
  - ``derived`` — MAC/SSA/AE/MEC, BC+OA / total sums, lifetime*
  - ``sampled`` — POLDER point mean in the box, or model field sampled at
    POLDER locations (dataset = model id)
  - ``homogenized`` — reconstructed regional POLDER AAOD/SSA (Zhong method)

Usage
-----
    python py/build_regional_aggregates.py
        # shared wide table via regional_aggregates.build_regional
        # (tables/aerocom_regional_seasonal.parquet)
    python py/build_regional_aggregates.py --legacy
        # previous long homogenized table
    python py/build_regional_aggregates.py --legacy --species du

Import helpers
--------------
    from build_regional_aggregates import (
        load_agg_cache, rows_to_model_seasonal, rows_to_obs_dicts,
        DEFAULT_OUT, ALL_VARIABLES,
    )
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'py'))
sys.path.insert(0, str(PROJECT_ROOT / 'notebooks'))

import aerocom_data  # noqa: E402
import cameo_toolbox as ct  # noqa: E402
import notebook_setup as setup  # noqa: E402
import regions as region_registry  # noqa: E402

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

DEFAULT_OUT = PROJECT_ROOT / 'tables' / 'regional_seasonal_aggregates.parquet'
DEFAULT_META = PROJECT_ROOT / 'tables' / 'regional_seasonal_aggregates_meta.json'
MONTHLY_DIR = PROJECT_ROOT / 'Data' / 'AP3_processed_monthly'
POLDER_PATH = (
    PROJECT_ROOT / 'Data' / 'AP3_POLDER_Collocated'
    / 'POLDER_GRASP_coloc_3h_AP3_2010_lon0_10.0_lat0_10.0.parquet'
)
GPCP_PATH = PROJECT_ROOT / 'Data' / 'Prec' / 'GPCP_2010_0-360.nc'

# Full variable set (notebook active + commented-out memory-heavy species).
ALL_VARIABLES = [
    'abs550aer', 'od440aer', 'od550aer', 'od870aer', 'od865aer',
    'od550bc', 'od550dust', 'od550oa', 'od550so4', 'od550ss',
    'depbc', 'depoa', 'depdust', 'depso2', 'depso4', 'depss',
    'emibc', 'emidust', 'emioa', 'emiso2', 'emiss',
    'loadbc', 'loadoa', 'loaddust', 'loadso2', 'loadso4', 'loadss',
    'precip',
]
# Dust build: dust budget + the column optics needed for SSA / AE / MEC.
DUST_VARIABLES = [
    'abs550aer', 'od440aer', 'od550aer', 'od870aer', 'od865aer', 'od550dust',
    'loaddust', 'emidust', 'depdust', 'wetdu', 'precip',
]
SPECIES_OUT = {
    'bb': PROJECT_ROOT / 'tables' / 'regional_seasonal_aggregates.parquet',
    'du': PROJECT_ROOT / 'tables' / 'regional_seasonal_aggregates_du.parquet',
}

EXCLUDE_MODELS_DEFAULT = [
    # 'GEOS-i33p2-met2010_AP3-CTRL-2010',
    'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010',
    'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010',
    # 'GISS-ModelE2p1p1-MATRIX_AP3-CTRL',
    # 'NorESM2-met2010_AP3-CTRL-v3',
]

LOAD_VARS = ['loadbc', 'loaddust', 'loadoa', 'loadso4', 'loadss']
EMI_VARS = ['emibc', 'emidust', 'emioa', 'emiso2', 'emiss']
DEP_TOTAL_VARS = ['depbc', 'depdust', 'depoa', 'depso4', 'depss']
BC_OA_LOAD = ['loadbc', 'loadoa']
BC_OA_EMI = ['emibc', 'emioa']
BC_OA_DEP = ['depbc', 'depoa']

DERIVED_PRE = ['MEC', 'MAC', 'SSA', 'AE']

# Per-species sums carried as 'derived', the lifetime budget keys
# (load, emission, deposition, output name) and homogenized obs pairs.
SUM_VARS_BY_SPECIES = {
    'bb': ('load_total', 'emi_total', 'dep_total', 'load_BC_OA', 'emi_BC_OA', 'dep_BC_OA'),
    'du': ('load_total', 'emi_total', 'dep_total', 'load_du', 'emi_du', 'dep_du', 'wet_du'),
}
LIFETIME_KEYS = {
    'bb': ('load_BC_OA', 'emi_BC_OA', 'dep_BC_OA', 'lifetime_BC_OA'),
    'du': ('load_du', 'emi_du', 'dep_du', 'lifetime_du'),
}
HOMOG_KEYS = {
    'bb': (('aaod', 'abs550aer'), ('ssa', 'SSA')),
    'du': (('aaod', 'abs550aer'), ('ssa', 'SSA'), ('aod', 'od550aer')),
}
HOMOG_OBS = {
    'aaod': ('AAOD_550', 'AAOD_550'),
    'ssa': ('SSA', 'SSA'),
    'aod': ('AOD_550', 'AOD_550'),
}
DERIVED_AFTER = ['lifetime_BC_OA']
AGGREGATED_RAW = list(ALL_VARIABLES)  # raw fields after normalize
DERIVED_COMBINED = [
    'load_total', 'emi_total', 'dep_total',
    'load_BC_OA', 'emi_BC_OA', 'dep_BC_OA',
] + DERIVED_PRE + DERIVED_AFTER + ['lifetime_BC_OA_emission', 'lifetime_BC_OA_raw']

POST_AGG_LIFETIME_MAX_DAYS = 50.0
POST_AGG_LIFETIME_MIN_DAYS = 0.3
USE_DEPOSITION_FOR_LIFETIME = True

EARTH_RADIUS_M = 6_371_000.0

UNIT_BY_VAR = {
    'precip': 'mm day-1',
    'abs550aer': '1', 'od440aer': '1', 'od550aer': '1', 'od870aer': '1',
    'od865aer': '1', 'od550bc': '1', 'od550dust': '1', 'od550oa': '1',
    'od550so4': '1', 'od550ss': '1',
    'SSA': '1', 'AE': '1', 'AAOD_550': '1', 'AOD_550': '1', 'AOD_440': '1',
    'AOD_870': '1',
    'MAC': 'm2 g-1', 'MEC': 'm2 g-1',
    'lifetime_BC_OA': 'days', 'lifetime_BC_OA_emission': 'days',
    'lifetime_BC_OA_raw': 'days', 'lifetime': 'days',
}
for _k in (
    'loadbc', 'loadoa', 'loaddust', 'loadso2', 'loadso4', 'loadss',
    'load_total', 'load_BC_OA',
):
    UNIT_BY_VAR[_k] = 'kg m-2'
for _k in (
    'emibc', 'emioa', 'emidust', 'emiso2', 'emiss',
    'depbc', 'depoa', 'depdust', 'depso2', 'depso4', 'depss',
    'emi_total', 'dep_total', 'emi_BC_OA', 'dep_BC_OA',
):
    UNIT_BY_VAR[_k] = 'kg m-2 s-1'


for _k in ('load_du',):
    UNIT_BY_VAR[_k] = 'kg m-2'
for _k in ('emi_du', 'dep_du', 'wet_du', 'wetdu'):
    UNIT_BY_VAR[_k] = 'kg m-2 s-1'
for _k in ('lifetime_du', 'lifetime_du_emission', 'lifetime_du_raw'):
    UNIT_BY_VAR[_k] = 'days'
UNIT_BY_VAR['MEC_du'] = 'm2 g-1'


def _unit(var: str) -> str:
    return UNIT_BY_VAR.get(var, '1')


def _region_cfg(region: str) -> dict:
    """AeroCom 2010 view of any registered region (BB, SS or dust)."""
    return setup.ALL_REGIONS[region]


def _in_season(region: str, times: pd.Series) -> pd.Series:
    """Time-slice + optional month-of-year filter for observation rows."""
    cfg = _region_cfg(region)
    t0, t1 = cfg['time_slice']
    ok = (times >= t0) & (times <= pd.Timestamp(t1) + pd.Timedelta(days=1))
    months = cfg.get('season_months')
    if months is not None:
        ok &= times.dt.month.isin(list(months))
    return ok


def _season_da(da: xr.DataArray, region: str) -> xr.DataArray:
    cfg = _region_cfg(region)
    da = da.sel(time=slice(*cfg['time_slice']))
    months = cfg.get('season_months')
    if months is not None:
        da = da.where(da['time'].dt.month.isin(list(months)), drop=True)
    return da


def _first_da(ds_or_da):
    if ds_or_da is None:
        return None
    if isinstance(ds_or_da, xr.Dataset):
        return ds_or_da[list(ds_or_da.data_vars)[0]]
    return ds_or_da


def _region_meta(region: str) -> dict:
    cfg = _region_cfg(region)
    lon_min, lon_max = cfg['lon_range']
    lat_min, lat_max = cfg['lat_range']
    t0, t1 = cfg['time_slice']
    return {
        'region': region,
        'time_start': t0,
        'time_end': t1,
        'lon_min': float(lon_min),
        'lon_max': float(lon_max),
        'lat_min': float(lat_min),
        'lat_max': float(lat_max),
        'surface_type': cfg.get('surface_type', 'all'),
    }


def _region_area_m2(da: xr.DataArray, mask, edge_weighted: bool) -> float:
    """Approximate region area (m²) on the model's grid via cos-lat weights."""
    if da is None:
        return np.nan
    try:
        weights, _ = ct._spatial_weights(da, mask, edge_weighted=edge_weighted)
        # weights ≈ cos(lat) * dlat_deg * dlon_deg → convert degree² to m²
        area = float(weights.sum().values) * (np.pi / 180.0) ** 2 * EARTH_RADIUS_M ** 2
        return area if np.isfinite(area) and area > 0 else np.nan
    except Exception:
        return np.nan


def _scalar(val) -> float:
    try:
        if hasattr(val, 'values'):
            return float(np.asarray(val).mean())
        return float(val)
    except Exception:
        return np.nan


def _row(region: str, dataset: str, var: str, source: str, value, area: float) -> dict:
    meta = _region_meta(region)
    return {
        **meta,
        'dataset': dataset,
        'var': var,
        'source': source,
        'value': _scalar(value),
        'unit': _unit(var),
        'area': float(area) if area is not None and np.isfinite(area) else np.nan,
    }


# ---------------------------------------------------------------------------
# Model processing (one model)
# ---------------------------------------------------------------------------
def _normalize_model(raw_model: dict, model: str) -> dict:
    normalized = {}
    for var, ds in raw_model.items():
        if ds is None:
            continue
        try:
            normalized[var] = aerocom_data.normalize_dataset_time(
                ds, var_hint=f'{model}/{var}', year=2010
            )
        except Exception as e:
            print(f'  Failed to normalize {model}/{var}: {e}')
            normalized[var] = None
    normalized['load_total'] = setup.sum_datasets(
        normalized, LOAD_VARS, 'load_total', require_all=False
    )
    normalized['emi_total'] = setup.sum_datasets(
        normalized, EMI_VARS, 'emi_total', require_all=False
    )
    normalized['dep_total'] = setup.sum_deposition_datasets(
        normalized, DEP_TOTAL_VARS, 'dep_total', require_all=False
    )
    normalized['load_BC_OA'] = setup.sum_datasets(
        normalized, BC_OA_LOAD, 'load_BC_OA', require_all=True
    )
    normalized['emi_BC_OA'] = setup.sum_datasets(
        normalized, BC_OA_EMI, 'emi_BC_OA', require_all=True
    )
    normalized['dep_BC_OA'] = setup.sum_deposition_datasets(
        normalized, BC_OA_DEP, 'dep_BC_OA', require_all=True
    )
    normalized['load_du'] = setup.sum_datasets(normalized, ['loaddust'], 'load_du')
    normalized['emi_du'] = setup.sum_datasets(normalized, ['emidust'], 'emi_du')
    normalized['dep_du'] = setup.sum_deposition_datasets(normalized, ['depdust'], 'dep_du')
    normalized['wet_du'] = setup.sum_deposition_datasets(normalized, ['wetdu'], 'wet_du')
    if normalized.get('precip') is not None:
        normalized['precip'] = setup.normalize_precipitation_units(normalized['precip'])
    return normalized


def _derive_pre_agg(normalized: dict, model: str) -> dict:
    aerocom_data.align_model_grids(normalized, ref_var='od550aer', model_hint=model)
    out = {}
    for dv in DERIVED_PRE:
        ds = aerocom_data.calculate_derived_var(normalized, model, dv)
        out[dv] = None if ds is None else ds[list(ds.data_vars)[0]]
    # Dust mass extinction per cell (m2 g-1), as MEC_du in the PPE tables.
    od_du = _first_da(normalized.get('od550dust'))
    load_du = _first_da(normalized.get('loaddust'))
    if od_du is not None and load_du is not None:
        try:
            load_du = aerocom_data._align_da_to_ref(load_du, od_du, model_hint=f'{model}/loaddust')
            out['MEC_du'] = (od_du / (load_du * 1e3)).where(load_du > 0)
        except Exception as e:
            print(f'  MEC_du failed for {model}: {e}')
    return out


def _build_masks(template, region_names: Iterable[str]):
    return setup.create_analysis_masks(template, list(region_names), region=setup.ALL_REGIONS)


def _aggregate_one(
    da: xr.DataArray,
    mask,
    region: str,
    return_time_series: bool = False,
) -> Optional[float]:
    cfg = _region_cfg(region)
    try:
        if cfg.get('season_months') is not None and 'time' in da.dims:
            da = da.where(da['time'].dt.month.isin(list(cfg['season_months'])), drop=True)
        return ct.regional_aggregate(
            da, mask,
            spatial='mean',
            edge_weighted=cfg['edge_weighted'],
            time_slice=cfg['time_slice'],
            temporal='mean',
            return_time_series=return_time_series,
            skipna=False,
        )
    except Exception as e:
        print(f'  Aggregation failed {region}: {e}')
        return None


def _load_one_model_netcdf(model: str, variables: List[str]) -> dict:
    """Load one model's monthly fields without dask chunks (faster on NFS)."""
    out = {v: None for v in variables}
    n = 0
    for var in variables:
        fpath = MONTHLY_DIR / var / f'{model}_{var}_processed.nc'
        if not fpath.exists():
            continue
        try:
            ds = xr.open_dataset(fpath, engine='netcdf4')
            ds = aerocom_data.standardize_dataset(ds, var)
            # Load into memory and close file handle
            ds = ds.load()
            ds.close()
            out[var] = ds
            n += 1
        except Exception as e:
            print(f'  Could not load {fpath.name}: {e}')
            out[var] = None
    print(f'  Loaded {n}/{len(variables)} NetCDF files for {model}')
    return out


# ---------------------------------------------------------------------------
# POLDER / GPCP
# ---------------------------------------------------------------------------
def _polder_region_mask(df: pd.DataFrame, lon_range, lat_range) -> pd.Series:
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    if lon_min > lon_max:
        mask_lon = (df['longitude'] >= lon_min) | (df['longitude'] <= lon_max)
    else:
        mask_lon = (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)
    mask_lat = (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)
    return mask_lon & mask_lat


def load_polder_df() -> pd.DataFrame:
    cols = ['time', 'longitude', 'latitude', 'AOD_550', 'AAOD_550', 'AOD_440', 'AOD_870']
    df = pd.read_parquet(POLDER_PATH, columns=cols)
    df['time'] = pd.to_datetime(df['time'])
    with np.errstate(divide='ignore', invalid='ignore'):
        df['SSA'] = 1.0 - df['AAOD_550'] / df['AOD_550']
        df['AE'] = -np.log(df['AOD_550'] / df['AOD_870']) / np.log(550.0 / 870.0)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['SSA', 'AE', 'AAOD_550', 'AOD_550']
    )
    # Model lon is 0–360
    lon = df['longitude'].astype(float)
    if lon.min() < 0:
        df['longitude'] = lon % 360
    return df


def polder_seasonal_rows(polder_df: pd.DataFrame, region_names: List[str]) -> List[dict]:
    rows = []
    for region in region_names:
        cfg = _region_cfg(region)
        mask = _polder_region_mask(polder_df, cfg['lon_range'], cfg['lat_range'])
        sub = polder_df[mask].copy()
        sub = sub[_in_season(region, sub['time'])]
        if sub.empty:
            continue
        w = np.cos(np.deg2rad(sub['latitude'].values))
        for var in ('AAOD_550', 'AOD_550', 'SSA', 'AE'):
            vals = sub[var].values
            finite = np.isfinite(vals) & np.isfinite(w)
            if finite.sum() < 3:
                continue
            mean = float(np.average(vals[finite], weights=w[finite]))
            rows.append(_row(region, 'POLDER', var, 'sampled', mean, np.nan))
    return rows


def _sample_at_polder(
    model_da: Optional[xr.DataArray],
    polder_sub: pd.DataFrame,
) -> float:
    """Sample model monthly field at pre-filtered POLDER points; weighted mean."""
    if model_da is None or polder_sub is None or polder_sub.empty:
        return np.nan
    try:
        lat_vals = np.asarray(model_da.lat.values)
        lon_vals = np.asarray(model_da.lon.values)
        # Cap samples for speed (cos-lat weighted subsample)
        sub = polder_sub
        if len(sub) > 20000:
            sub = sub.sample(n=20000, random_state=0)
        lat_idx = np.abs(lat_vals[:, None] - sub['latitude'].values[None, :]).argmin(axis=0)
        lon_idx = np.abs(lon_vals[:, None] - sub['longitude'].values[None, :]).argmin(axis=0)
        times = pd.to_datetime(model_da.time.values)
        # Map each POLDER time to nearest model month index
        month_starts = pd.to_datetime(
            pd.Series(times).dt.to_period('M').dt.to_timestamp()
        )
        polder_months = pd.to_datetime(sub['time']).dt.to_period('M').dt.to_timestamp()
        # Build month → index lookup
        uniq = {m: i for i, m in enumerate(month_starts)}
        month_idx = np.array([uniq.get(m, -1) for m in polder_months])
        ok = month_idx >= 0
        if ok.sum() < 3:
            return np.nan
        vals = np.asarray(model_da.values)[month_idx[ok], lat_idx[ok], lon_idx[ok]]
        weights = np.cos(np.deg2rad(sub['latitude'].values[ok]))
        finite = np.isfinite(vals) & np.isfinite(weights)
        if finite.sum() < 3:
            return np.nan
        return float(np.average(vals[finite], weights=weights[finite]))
    except Exception as e:
        print(f'  POLDER sample failed: {e}')
        return np.nan


def _polder_subs_by_region(
    polder_df: pd.DataFrame, region_names: List[str]
) -> Dict[str, pd.DataFrame]:
    """Pre-filter POLDER to each region's box + fire season once."""
    out = {}
    for region in region_names:
        cfg = _region_cfg(region)
        mask = _polder_region_mask(polder_df, cfg['lon_range'], cfg['lat_range'])
        sub = polder_df.loc[mask].copy()
        sub = sub[_in_season(region, sub['time'])]
        out[region] = sub
        print(f'  POLDER {region}: {len(sub)} points in season', flush=True)
    return out


def process_one_model(
    model: str,
    region_names: List[str],
    variables: List[str],
    polder_subs: Optional[Dict[str, pd.DataFrame]] = None,
    species: str = 'bb',
) -> Tuple[List[dict], dict]:
    """Load → normalize → derive → aggregate one model. Returns (rows, sample_pairs).

    ``sample_pairs[region]['aaod'|'ssa'] = (regional, sampled_at_polder)`` for
    homogenization (only when ``polder_subs`` is provided).
    """
    raw_model = _load_one_model_netcdf(model, variables)
    if not any(v is not None for v in raw_model.values()):
        print(f'  Skip {model}: no files')
        return [], {}

    normalized = _normalize_model(raw_model, model)
    del raw_model
    gc.collect()

    # Template for masks
    template = None
    for cand in ('od550aer', 'abs550aer', 'loadbc', 'emibc', 'precip'):
        ds = normalized.get(cand)
        if ds is not None:
            template = _first_da(ds)
            if template is not None:
                template = template.isel(time=0) if 'time' in template.dims else template
                break
    if template is None:
        print(f'  Skip {model}: no template grid')
        return [], {}

    masks = _build_masks(template, region_names)
    derived = _derive_pre_agg(normalized, model)

    # Flatten to DataArrays for aggregation
    fields: Dict[str, Optional[xr.DataArray]] = {}
    for var in variables:
        fields[var] = _first_da(normalized.get(var))
    sum_vars = SUM_VARS_BY_SPECIES[species]
    for var in sum_vars:
        fields[var] = _first_da(normalized.get(var))
    for var in DERIVED_PRE + (['MEC_du'] if species == 'du' else []):
        fields[var] = derived.get(var)
    load_key, emi_key, dep_key, lt_key = LIFETIME_KEYS[species]

    rows: List[dict] = []
    sample_pairs: Dict[str, dict] = {}

    for region in region_names:
        cfg = _region_cfg(region)
        mask = masks[region]
        # Area from any available field
        area = np.nan
        for ref_var in ('od550aer', 'abs550aer', 'load_BC_OA', 'loadbc', 'precip'):
            if fields.get(ref_var) is not None:
                area = _region_area_m2(fields[ref_var], mask, cfg['edge_weighted'])
                if np.isfinite(area):
                    break

        seasonal_vals: Dict[str, float] = {}
        for var, da in fields.items():
            if da is None:
                continue
            val = _aggregate_one(da, mask, region)
            if val is None or not np.isfinite(_scalar(val)):
                continue
            source = 'aggregated'
            if var in sum_vars or var in DERIVED_PRE or var == 'MEC_du':
                source = 'derived'
            rows.append(_row(region, model, var, source, val, area))
            seasonal_vals[var] = _scalar(val)

        # Post-agg lifetimes
        load = seasonal_vals.get(load_key)
        emi = seasonal_vals.get(emi_key)
        dep = seasonal_vals.get(dep_key)
        lt_emi = lt_dep = np.nan
        if load is not None and emi is not None and emi != 0:
            lt_emi = load / (emi * 86400.0)
        if load is not None and dep is not None and dep != 0:
            lt_dep = load / (dep * 86400.0)

        lt_active = lt_dep if USE_DEPOSITION_FOR_LIFETIME else lt_emi
        if np.isfinite(lt_active):
            rows.append(_row(
                region, model, f'{lt_key}_raw', 'derived', lt_active, area
            ))
            lt_filt = lt_active
            if (
                (POST_AGG_LIFETIME_MAX_DAYS is not None and lt_filt > POST_AGG_LIFETIME_MAX_DAYS)
                or (POST_AGG_LIFETIME_MIN_DAYS is not None and lt_filt < POST_AGG_LIFETIME_MIN_DAYS)
            ):
                lt_filt = np.nan
            rows.append(_row(
                region, model, lt_key, 'derived', lt_filt, area
            ))
        if np.isfinite(lt_emi):
            rows.append(_row(
                region, model, f'{lt_key}_emission', 'derived', lt_emi, area
            ))

        # Model sampled at POLDER (for homogenization only)
        if polder_subs is not None:
            sample_pairs[region] = {}
            psub = polder_subs.get(region)
            for obs_key, model_key in HOMOG_KEYS[species]:
                regional = seasonal_vals.get(model_key)
                da = fields.get(model_key)
                if da is not None:
                    try:
                        da_seas = _season_da(da, region)
                    except Exception:
                        da_seas = da
                else:
                    da_seas = None
                sampled = _sample_at_polder(da_seas, psub)
                if regional is not None and np.isfinite(regional):
                    sample_pairs[region][obs_key] = (regional, sampled)
                if np.isfinite(sampled):
                    rows.append(_row(
                        region, model, model_key, 'sampled', sampled, area
                    ))

    # Drop heavy objects
    del normalized, derived, fields, masks, template
    gc.collect()
    return rows, sample_pairs


def homogenize_from_pairs(
    all_pairs: Dict[str, Dict[str, List[Tuple[float, float]]]],
    polder_sampled: Dict[str, Dict[str, float]],
    source_regions: List[str],
    pair_keys: Iterable[str] = ('aaod', 'ssa'),
) -> List[dict]:
    """Zhong-style homogenization: sampled = a * regional + b → obs_h = (obs_s - b) / a."""
    rows = []
    for region in source_regions:
        for pair_key in pair_keys:
            obs_var, polder_col = HOMOG_OBS[pair_key]
            pairs = all_pairs.get(region, {}).get(pair_key, [])
            pairs = [(r, s) for r, s in pairs if np.isfinite(r) and np.isfinite(s)]
            obs_s = polder_sampled.get(region, {}).get(polder_col, np.nan)
            if len(pairs) < 3 or not np.isfinite(obs_s):
                # Fall back to sampled
                if np.isfinite(obs_s):
                    rows.append(_row(region, 'POLDER', obs_var, 'homogenized', obs_s, np.nan))
                continue
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            X = np.column_stack([x, np.ones(len(x))])
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            a, b = float(coeffs[0]), float(coeffs[1])
            if abs(a) < 1e-12:
                obs_h = obs_s
            else:
                obs_h = (obs_s - b) / a
            rows.append(_row(region, 'POLDER', obs_var, 'homogenized', obs_h, np.nan))
            print(f'  Homogenize {region} {obs_var}: a={a:.4f}, b={b:.4f}, '
                  f'sampled={obs_s:.5f} → homog={obs_h:.5f} (n={len(pairs)})')
    return rows


def gpcp_seasonal_rows(region_names: List[str]) -> List[dict]:
    if not GPCP_PATH.exists():
        print(f'GPCP missing: {GPCP_PATH}')
        return []
    gpcp = xr.open_dataset(GPCP_PATH)
    precip = gpcp['sat_gauge_precip']
    rows = []
    for region in region_names:
        cfg = _region_cfg(region)
        lon_min, lon_max = cfg['lon_range']
        lat_min, lat_max = cfg['lat_range']
        if lon_min > lon_max:
            sub = xr.concat([
                precip.sel(lon=slice(lon_min, 360)),
                precip.sel(lon=slice(0, lon_max)),
            ], dim='lon')
        else:
            sub = precip.sel(lon=slice(lon_min, lon_max))
        sub = sub.sel(lat=slice(lat_max, lat_min))
        sub = _season_da(sub, region)
        if sub.size == 0:
            continue
        coslat = np.cos(np.deg2rad(sub.lat))
        weights = coslat * xr.ones_like(sub.isel(time=0))
        weights = weights / weights.sum()
        spatial = (sub * weights).sum(dim=['lat', 'lon'])
        mean_val = float(spatial.mean().values)
        # Approx area on GPCP grid
        area = float((coslat * xr.ones_like(sub.isel(time=0))).sum().values)
        # crude: leave NaN for GPCP area (obs not on AeroCom grid)
        rows.append(_row(region, 'GPCP', 'precip', 'aggregated', mean_val, np.nan))
    gpcp.close()
    return rows


# ---------------------------------------------------------------------------
# Cache I/O helpers (used by notebooks)
# ---------------------------------------------------------------------------
def load_agg_cache(path=None) -> pd.DataFrame:
    path = Path(path or DEFAULT_OUT)
    if not path.exists():
        raise FileNotFoundError(f'Aggregate cache not found: {path}')
    return pd.read_parquet(path)


def rows_to_model_seasonal(
    df: pd.DataFrame,
    regions: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Rebuild ``model_seasonal[region][var][model] = float`` from long-form rows.

    Excludes observation datasets (POLDER, GPCP). Prefer filtered lifetime
    (``lifetime_BC_OA``) over raw.
    """
    sources = sources or ['aggregated', 'derived']
    sub = df[df['source'].isin(sources)].copy()
    sub = sub[~sub['dataset'].isin(['POLDER', 'GPCP'])]
    if regions is not None:
        sub = sub[sub['region'].isin(regions)]
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for _, row in sub.iterrows():
        val = row['value']
        if not np.isfinite(val):
            # Keep NaN lifetimes so notebooks see filtered models as present-but-NaN
            if row['var'] not in ('lifetime_BC_OA', 'lifetime', 'lifetime_BC_OA_raw'):
                continue
        out.setdefault(row['region'], {}).setdefault(row['var'], {})[row['dataset']] = float(val)
    return out


def rows_to_obs_dicts(df: pd.DataFrame, regions: Optional[List[str]] = None):
    """Return observation dicts used by AAOD / test_models.

    Returns
    -------
    aaod_obs_by_region : dict[str, float]
        Prefer homogenized; fall back to sampled.
    ssa_obs_by_region : dict[str, float]
        Prefer homogenized; fall back to sampled.
    gpcp_seasonal : dict[str, float]
        Seasonal-mean GPCP precip (mm day⁻¹).
    polder_sampled : dict[str, dict[str, float]]
        ``polder_sampled[region][var]`` for AAOD_550, AOD_550, SSA, AE.
    """
    sub = df[df['dataset'].isin(['POLDER', 'GPCP'])].copy()
    if regions is not None:
        sub = sub[sub['region'].isin(regions)]

    polder_sampled: Dict[str, Dict[str, float]] = {}
    aaod_obs: Dict[str, float] = {}
    ssa_obs: Dict[str, float] = {}
    gpcp_seasonal: Dict[str, float] = {}

    for _, row in sub.iterrows():
        region, var, source, val = row['region'], row['var'], row['source'], row['value']
        if not np.isfinite(val):
            continue
        if row['dataset'] == 'GPCP' and var == 'precip':
            gpcp_seasonal[region] = float(val)
        elif row['dataset'] == 'POLDER':
            if source == 'sampled':
                polder_sampled.setdefault(region, {})[var] = float(val)
            if var == 'AAOD_550':
                if source == 'homogenized' or region not in aaod_obs:
                    if source == 'homogenized':
                        aaod_obs[region] = float(val)
                    elif source == 'sampled' and region not in aaod_obs:
                        aaod_obs[region] = float(val)
            if var == 'SSA':
                if source == 'homogenized':
                    ssa_obs[region] = float(val)
                elif source == 'sampled' and region not in ssa_obs:
                    ssa_obs[region] = float(val)

    # Prefer homogenized when present (overwrite sampled defaults)
    for _, row in sub[(sub['dataset'] == 'POLDER') & (sub['source'] == 'homogenized')].iterrows():
        if row['var'] == 'AAOD_550' and np.isfinite(row['value']):
            aaod_obs[row['region']] = float(row['value'])
        if row['var'] == 'SSA' and np.isfinite(row['value']):
            ssa_obs[row['region']] = float(row['value'])

    return aaod_obs, ssa_obs, gpcp_seasonal, polder_sampled


def gpcp_seasonal_to_region_series(gpcp_seasonal: Dict[str, float]):
    """Wrap scalar seasonal means as 0-d DataArrays for notebook compatibility.

    Notebooks often call ``gpcp_region[region].mean().values``. A 0-d array
    satisfies that without storing monthly time series.
    """
    out = {}
    for region, val in gpcp_seasonal.items():
        out[region] = xr.DataArray(float(val), name='precip')
    return out


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def build(
    out_path: Path = DEFAULT_OUT,
    meta_path: Path = DEFAULT_META,
    variables: Optional[List[str]] = None,
    exclude_models: Optional[List[str]] = None,
    region_names: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    do_homogenize: bool = True,
    species: str = 'bb',
) -> pd.DataFrame:
    if species not in LIFETIME_KEYS:
        raise ValueError(f'species must be one of {sorted(LIFETIME_KEYS)}')
    variables = list(variables or (DUST_VARIABLES if species == 'du' else ALL_VARIABLES))
    # Only keep variables that exist on disk
    available = {
        p.name for p in MONTHLY_DIR.iterdir()
        if p.is_dir() and p.name != 'derived'
    } if MONTHLY_DIR.exists() else set()
    variables = [v for v in variables if v in available]
    print(f'Variables to load ({len(variables)}): {variables}')

    exclude_models = list(exclude_models or EXCLUDE_MODELS_DEFAULT)
    if region_names is None:
        region_names = (['global'] + region_registry.regions_for_species('du')
                        if species == 'du' else list(setup.REGIONS.keys()))
    region_names = list(region_names)
    print(f'Species {species}; regions ({len(region_names)}): {region_names}')

    # Discover models from abs550aer or od550aer dir (dust: models with dust optics)
    model_set = set()
    for var in (('od550dust', 'loaddust') if species == 'du' else ('od550aer', 'abs550aer', 'loadbc')):
        vdir = MONTHLY_DIR / var
        if not vdir.exists():
            continue
        for f in vdir.glob('*_processed.nc'):
            # <model>_<var>_processed.nc
            name = f.name
            suffix = f'_{var}_processed.nc'
            if name.endswith(suffix):
                model_set.add(name[: -len(suffix)])
    if model_names:
        models = sorted(m for m in model_names if m in model_set and m not in exclude_models)
        missing = [m for m in model_names if m not in model_set]
        if missing:
            print(f'Warning: requested models not found: {missing}')
    else:
        models = sorted(m for m in model_set if m not in exclude_models)
    print(f'Models ({len(models)}); excluded {exclude_models}')

    print(f'\nLoading POLDER: {POLDER_PATH}')
    polder_df = load_polder_df()
    print(f'  POLDER rows: {len(polder_df)}')
    polder_subs = None
    if do_homogenize:
        print('Pre-filtering POLDER by region/season for homogenization...')
        polder_subs = _polder_subs_by_region(polder_df, region_names)

    all_rows: List[dict] = []
    # Collect homogenization pairs: region -> key -> list[(regional, sampled)]
    pair_keys = [k for k, _ in HOMOG_KEYS[species]]
    homog_pairs: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
        r: {k: [] for k in pair_keys} for r in region_names
    }

    for i, model in enumerate(models, 1):
        print(f'\n[{i}/{len(models)}] {model}', flush=True)
        rows, pairs = process_one_model(
            model, region_names, variables, polder_subs=polder_subs, species=species
        )
        all_rows.extend(rows)
        for region, d in pairs.items():
            for key in pair_keys:
                if key in d:
                    homog_pairs[region][key].append(d[key])
        print(f'  rows so far: {len(all_rows)}')

    # Observations
    print('\n--- POLDER seasonal sampled ---')
    polder_rows = polder_seasonal_rows(polder_df, region_names)
    all_rows.extend(polder_rows)

    polder_sampled = {}
    for r in polder_rows:
        polder_sampled.setdefault(r['region'], {})[r['var']] = r['value']

    if do_homogenize:
        print('\n--- Homogenization (source regions) ---')
        default_sources = ('africa', 'amazon') if species == 'bb' else \
            tuple(region_registry.source_regions(species))
        source_regions = [r for r in default_sources if r in region_names]
        # Outflow: copy sampled into homogenized for AAOD/SSA consistency
        homog_rows = homogenize_from_pairs(homog_pairs, polder_sampled, source_regions,
                                           pair_keys=pair_keys)
        all_rows.extend(homog_rows)
        for region in region_names:
            if region in source_regions:
                continue
            for var in [HOMOG_OBS[k][0] for k in pair_keys]:
                if var in polder_sampled.get(region, {}):
                    all_rows.append(_row(
                        region, 'POLDER', var, 'homogenized',
                        polder_sampled[region][var], np.nan,
                    ))

    print('\n--- GPCP ---')
    all_rows.extend(gpcp_seasonal_rows(region_names))

    df = pd.DataFrame(all_rows)
    # Drop non-finite values except lifetime NaNs (filtered)
    lt_key = LIFETIME_KEYS[species][3]
    keep_nan_vars = {lt_key, 'lifetime', f'{lt_key}_raw'}
    mask_ok = df['value'].notna() | df['var'].isin(keep_nan_vars)
    df = df.loc[mask_ok].reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f'\nWrote {len(df)} rows → {out_path}')

    meta = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'n_rows': int(len(df)),
        'n_models': len(models),
        'models': models,
        'variables': variables,
        'species': species,
        'regions': {name: {
            'lon_range': list(_region_cfg(name)['lon_range']),
            'lat_range': list(_region_cfg(name)['lat_range']),
            'time_slice': list(_region_cfg(name)['time_slice']),
            'season_months': list(_region_cfg(name).get('season_months') or []),
            'surface_type': _region_cfg(name).get('surface_type', 'all'),
            'edge_weighted': _region_cfg(name).get('edge_weighted', False),
        } for name in region_names},
        'use_deposition_for_lifetime': USE_DEPOSITION_FOR_LIFETIME,
        'post_agg_lifetime_days': [POST_AGG_LIFETIME_MIN_DAYS, POST_AGG_LIFETIME_MAX_DAYS],
        'exclude_models': exclude_models,
        'parquet': str(out_path),
    }
    meta_path = Path(meta_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'Wrote meta → {meta_path}')
    return df


def main(argv=None):
    """Shared wide table by default. ``--legacy`` keeps the long homogenized file."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--species', default='bb', choices=sorted(LIFETIME_KEYS),
                   help='bb: BC+OA budget on the AeroCom BB regions (default); '
                        'du: dust budget on the dust regions of py/regions.py')
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--meta', type=Path, default=None)
    p.add_argument('--no-homogenize', action='store_true')
    p.add_argument('--regions', nargs='*', default=None,
                   help='Subset of setup.ALL_REGIONS (default: species regions)')
    p.add_argument('--models', nargs='*', default=None,
                   help='Subset of AeroCom models (default: all non-excluded)')
    p.add_argument('--legacy', action='store_true',
                   help='Write the long homogenized seasonal table (previous builder)')
    p.add_argument('--quality', default=None, choices=('ppe', 'aerocom'))
    args = p.parse_args(argv)
    if args.legacy:
        out = args.out or SPECIES_OUT[args.species]
        meta = args.meta or out.with_name(out.stem + '_meta.json')
        build(
            out_path=out,
            meta_path=meta,
            region_names=args.regions,
            model_names=args.models,
            do_homogenize=not args.no_homogenize,
            species=args.species,
        )
        return
    from regional_aggregates import build_regional
    build_regional(
        source='aerocom',
        regions=args.regions,
        species=args.species,
        models=args.models,
        quality=args.quality,
        out_path=args.out,
    )


if __name__ == '__main__':
    main()
