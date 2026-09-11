#!/usr/bin/env python3
"""Build a wide-form MONTHLY regional aggregate parquet for the sea-salt notebook.

Unlike ``py/build_regional_aggregates.py`` (BC+OA/AAOD focused, seasonal-only,
bakes each region's ``time_slice`` into the aggregation), this script:

  - Aggregates every model x variable x region for **all 12 months of 2010**,
    completely ignoring any notion of "season" during aggregation.
  - Regions are purely spatial (``lon_range``, ``lat_range``, ``surface_type``,
    ``edge_weighted``). Season/``time_slice`` filtering happens downstream, in
    the notebook, as a pandas filter on this cached monthly table
    (see ``notebook_setup.seasonal_from_monthly_df``).
  - Writes ONE wide parquet: columns ``region, model, month, <variables...>`` --
    the same shape as ``setup.monthly_dict_to_dataframe(...)`` already used by
    the notebook, so downstream code barely changes.
  - Parallelizes per-model processing with ``dask.delayed`` (a local
    multi-core ``LocalCluster`` by default; optional
    ``dask_jobqueue.SLURMCluster`` for multi-node scaling via ``--slurm``).

Usage
-----
    python py/build_monthly_regional_aggregates.py
    python py/build_monthly_regional_aggregates.py --models CAM5-ATRAS_AP3-CTRL --regions NAO ANTAO
    python py/build_monthly_regional_aggregates.py --n-workers 16

Import helpers (used by notebooks)
-----------------------------------
    from build_monthly_regional_aggregates import (
        load_monthly_cache, SPATIAL_REGIONS, AGG_VARS, DEFAULT_OUT,
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
from typing import Dict, List, Optional

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

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

MONTHLY_DIR = PROJECT_ROOT / 'Data' / 'AP3_processed_monthly'
DEFAULT_OUT = PROJECT_ROOT / 'tables' / 'regional_monthly_aggregates_ss.parquet'
DEFAULT_META = PROJECT_ROOT / 'tables' / 'regional_monthly_aggregates_ss_meta.json'

# ---------------------------------------------------------------------------
# Spatial region definitions (sea-salt notebook regions), decoupled from any
# time_slice/season. Season filtering happens downstream on the cached
# monthly table -- see notebook_setup.seasonal_from_monthly_df().
# ---------------------------------------------------------------------------
SPATIAL_REGIONS: Dict[str, dict] = {
    'global': {'surface_type': 'ocean', 'lon_range': (0, 360), 'lat_range': (-90, 90), 'edge_weighted': False},
    'ARCO':   {'surface_type': 'ocean', 'lon_range': (0, 360), 'lat_range': (60, 90), 'edge_weighted': True},
    'NPO':    {'surface_type': 'ocean', 'lon_range': (120, 260), 'lat_range': (20, 60), 'edge_weighted': True},
    'TPO':    {'surface_type': 'ocean', 'lon_range': (120, 280), 'lat_range': (-20, 20), 'edge_weighted': False},
    'SPO':    {'surface_type': 'ocean', 'lon_range': (120, 290), 'lat_range': (-60, -20), 'edge_weighted': False},
    'NAO':    {'surface_type': 'ocean', 'lon_range': (280, 360), 'lat_range': (20, 60), 'edge_weighted': False},
    'TAO':    {'surface_type': 'ocean', 'lon_range': (290, 360), 'lat_range': (-20, 20), 'edge_weighted': False},
    'SAO':    {'surface_type': 'ocean', 'lon_range': (290, 360), 'lat_range': (-60, -20), 'edge_weighted': False},
    'TIO':    {'surface_type': 'ocean', 'lon_range': (20, 120), 'lat_range': (-20, 20), 'edge_weighted': False},
    'SIO':    {'surface_type': 'ocean', 'lon_range': (20, 120), 'lat_range': (-60, -20), 'edge_weighted': False},
    'ANTAO':  {'surface_type': 'ocean', 'lon_range': (0, 360), 'lat_range': (-90, -60), 'edge_weighted': False},
}

EXCLUDE_MODELS_DEFAULT = [
    'GEOS-i33p2-met2010_AP3-CTRL-2010',
    'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010',
    'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010',
    'GISS-ModelE2p1p1-MATRIX_AP3-CTRL',
    'NorESM2-met2010_AP3-CTRL-v3',
]

# Raw NetCDF variables to load per model (mirrors the notebook's VARIABLES list).
VARIABLES = [
    'od440aer', 'od550aer', 'od870aer', 'od865aer',
    'abs550aer', 'od550ss', 'od550so4',
    'precip',
    'loadbc', 'loaddust', 'loadoa', 'loadso4', 'loadss',
    'emibc', 'emidust', 'emioa', 'emiso2', 'emiss',
    'depbc', 'depdust', 'depoa', 'depso4', 'depss',
]
LOAD_VARS = ['loadbc', 'loaddust', 'loadoa', 'loadso4', 'loadss']
EMI_VARS = ['emibc', 'emidust', 'emioa', 'emiso2', 'emiss']
DEP_VARS = ['depbc', 'depdust', 'depoa', 'depso4', 'depss']

# Aggregated output columns (raw fields + pre-aggregation derived vars).
# Mirrors the notebook's `variables_pre_aggregated` list minus the
# after-aggregation lifetime variables (those are derived downstream, see
# notebook_setup.seasonal_from_monthly_df / LIFETIME_SPECS in the notebook).
AGG_VARS = [
    'MEC', 'MAC', 'SSA', 'AE', 'MEC_ss', 'od550aer',
    'od550so4', 'depso4', 'emiso2',
    'od550ss', 'loadss', 'depss', 'emiss',
    'precip', 'abs550aer', 'loadso4',
    'load_total', 'emi_total', 'dep_total',
]


def _first_da(ds_or_da):
    if ds_or_da is None:
        return None
    if isinstance(ds_or_da, xr.Dataset):
        return ds_or_da[list(ds_or_da.data_vars)[0]]
    return ds_or_da


def discover_models(variables: Optional[List[str]] = None, exclude_models: Optional[List[str]] = None) -> List[str]:
    """Discover available models from the od550aer/abs550aer/loadbc file listings."""
    exclude_models = set(exclude_models or EXCLUDE_MODELS_DEFAULT)
    model_set = set()
    for var in ('od550aer', 'abs550aer', 'loadbc'):
        vdir = MONTHLY_DIR / var
        if not vdir.exists():
            continue
        suffix = f'_{var}_processed.nc'
        for f in vdir.glob('*_processed.nc'):
            if f.name.endswith(suffix):
                model_set.add(f.name[: -len(suffix)])
    return sorted(m for m in model_set if m not in exclude_models)


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
            ds = ds.load()
            ds.close()
            out[var] = ds
            n += 1
        except Exception as e:
            print(f'  [{model}] Could not load {fpath.name}: {e}')
    print(f'  [{model}] Loaded {n}/{len(variables)} NetCDF files', flush=True)
    return out


def _derive_pre_agg(normalized: dict, model: str) -> Dict[str, Optional[xr.DataArray]]:
    """Compute MEC, MEC_ss, SSA, MAC, AE.

    Inlined to match exactly the notebook's pre-aggregation derived-variable
    cell (od550aer / total_load / od550ss / loadss / abs550aer based ratios),
    since ``aerocom_data.calculate_derived_var`` does not support MEC_ss.
    """
    aerocom_data.align_model_grids(normalized, ref_var='od550aer', model_hint=model)

    aod550 = aerocom_data._get_dataarray(normalized.get('od550aer'), 'od550aer')
    if aod550 is None:
        return {}

    load_keys = ['loadbc', 'loaddust', 'loadoa', 'loadso4', 'loadss']
    loads = []
    for key in load_keys:
        load_da = aerocom_data._get_dataarray(normalized.get(key), key)
        if load_da is not None:
            loads.append(aerocom_data._align_da_to_ref(load_da, aod550, model_hint=f'{model}/{key}'))
    total_load = sum(loads) if loads else None

    aod_ss = aerocom_data._get_dataarray(normalized.get('od550ss'), 'od550ss')
    load_ss = aerocom_data._get_dataarray(normalized.get('loadss'), 'loadss')
    aaod = aerocom_data._get_dataarray(normalized.get('abs550aer'), 'abs550aer')

    out: Dict[str, Optional[xr.DataArray]] = {}

    try:
        if total_load is not None:
            out['MEC'] = aod550 / (total_load * 1e3)
    except Exception as e:
        print(f'  [{model}] Could not calculate MEC: {e}')

    try:
        if aod_ss is not None and load_ss is not None:
            aod_ss_a = aerocom_data._align_da_to_ref(aod_ss, aod550, model_hint=f'{model}/od550ss')
            load_ss_a = aerocom_data._align_da_to_ref(load_ss, aod550, model_hint=f'{model}/loadss')
            out['MEC_ss'] = aod_ss_a / (load_ss_a * 1e3)
    except Exception as e:
        print(f'  [{model}] Could not calculate MEC_ss: {e}')

    try:
        if aaod is not None:
            aaod_a = aerocom_data._align_da_to_ref(aaod, aod550, model_hint=f'{model}/abs550aer')
            out['SSA'] = 1.0 - aaod_a / aod550
    except Exception as e:
        print(f'  [{model}] Could not calculate SSA: {e}')

    try:
        if aaod is not None and total_load is not None:
            aaod_a = aerocom_data._align_da_to_ref(aaod, aod550, model_hint=f'{model}/abs550aer')
            out['MAC'] = aaod_a / (total_load * 1e3)
    except Exception as e:
        print(f'  [{model}] Could not calculate MAC: {e}')

    aod_other, other_wavelength = None, None
    for var_name, wavelength in (('od870aer', 870), ('od865aer', 865), ('od440aer', 440)):
        candidate = aerocom_data._get_dataarray(normalized.get(var_name), var_name)
        if candidate is not None:
            aod_other, other_wavelength = candidate, wavelength
            break
    try:
        if aod_other is not None:
            aod_other_a = aerocom_data._align_da_to_ref(aod_other, aod550, model_hint=f'{model}/spectral_aod')
            out['AE'] = -np.log(aod550 / aod_other_a) / np.log(550.0 / other_wavelength)
    except Exception as e:
        print(f'  [{model}] Could not calculate AE: {e}')

    return out


def process_one_model(
    model: str,
    variables: List[str],
    spatial_regions: Dict[str, dict],
) -> pd.DataFrame:
    """Load -> normalize -> derive -> aggregate ALL 12 months for one model.

    Returns a wide DataFrame: ``region, model, month, <AGG_VARS...>``. Every
    region gets every calendar month present in the model's NetCDF time axis --
    ``time_slice`` from any season config is intentionally NOT applied here.
    """
    empty = pd.DataFrame(columns=['region', 'model', 'month'] + AGG_VARS)

    raw = _load_one_model_netcdf(model, variables)
    if not any(v is not None for v in raw.values()):
        print(f'  Skip {model}: no files found')
        return empty

    normalized: Dict[str, Optional[xr.Dataset]] = {}
    for var, ds in raw.items():
        if ds is None:
            normalized[var] = None
            continue
        try:
            normalized[var] = aerocom_data.normalize_dataset_time(ds, var_hint=f'{model}/{var}', year=2010)
        except Exception as e:
            print(f'  [{model}] Failed to normalize {var}: {e}')
            normalized[var] = None
    del raw
    gc.collect()

    normalized['load_total'] = setup.sum_datasets(normalized, LOAD_VARS, 'load_total', require_all=True)
    normalized['emi_total'] = setup.sum_datasets(normalized, EMI_VARS, 'emi_total', require_all=True)
    normalized['dep_total'] = setup.sum_deposition_datasets(normalized, DEP_VARS, 'dep_total', require_all=True)

    derived = _derive_pre_agg(normalized, model)

    fields: Dict[str, Optional[xr.DataArray]] = {}
    for var in ('od550aer', 'od550so4', 'depso4', 'emiso2', 'od550ss', 'loadss',
                'depss', 'emiss', 'precip', 'abs550aer', 'loadso4',
                'load_total', 'emi_total', 'dep_total'):
        fields[var] = _first_da(normalized.get(var))
    for var in ('MEC', 'MAC', 'SSA', 'AE', 'MEC_ss'):
        fields[var] = derived.get(var)

    template = fields.get('od550aer')
    if template is None:
        print(f'  Skip {model}: no od550aer template grid')
        return empty
    if 'time' in template.dims:
        template = template.isel(time=0)

    masks = {}
    for name, cfg in spatial_regions.items():
        masks[name] = ct.create_region_mask(
            template, name=name,
            lon_range=cfg['lon_range'], lat_range=cfg['lat_range'],
            surface_type=cfg.get('surface_type', 'all'),
            mask_registry=masks,
        )

    # acc[(region, model, month)] -> {var: value, ...}
    acc: Dict[tuple, dict] = {}
    for region, cfg in spatial_regions.items():
        mask = masks[region]
        for var, da in fields.items():
            if da is None:
                continue
            try:
                series = ct.regional_aggregate(
                    da, mask, spatial='mean', edge_weighted=cfg['edge_weighted'],
                    time_slice=None, temporal='mean',
                    return_time_series=True, skipna=True,
                )
            except Exception as e:
                print(f'  [{model}] Aggregation failed {var} {region}: {e}')
                continue
            if series is None or 'time' not in getattr(series, 'dims', ()):
                continue
            months = pd.to_datetime(series['time'].values).to_period('M').astype(str)
            vals = np.asarray(series.values, dtype=float)
            for month, val in zip(months, vals):
                key = (region, model, month)
                acc.setdefault(key, {'region': region, 'model': model, 'month': month})[var] = float(val)

    del normalized, derived, fields, masks, template
    gc.collect()

    if not acc:
        return empty

    wide = pd.DataFrame(list(acc.values()))
    for col in AGG_VARS:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide[['region', 'model', 'month'] + AGG_VARS]
    return wide.sort_values(['region', 'month']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cache I/O helper (used by the notebook)
# ---------------------------------------------------------------------------
def load_monthly_cache(path=None) -> pd.DataFrame:
    path = Path(path or DEFAULT_OUT)
    if not path.exists():
        raise FileNotFoundError(f'Monthly aggregate cache not found: {path}')
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Main build (dask-parallelized across models)
# ---------------------------------------------------------------------------
def build_monthly(
    out_path: Path = DEFAULT_OUT,
    meta_path: Path = DEFAULT_META,
    variables: Optional[List[str]] = None,
    exclude_models: Optional[List[str]] = None,
    region_names: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    n_workers: int = 8,
    threads_per_worker: int = 1,
    use_slurm: bool = False,
    slurm_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    variables = list(variables or VARIABLES)
    spatial_regions = {
        name: cfg for name, cfg in SPATIAL_REGIONS.items()
        if region_names is None or name in region_names
    }
    if not spatial_regions:
        raise ValueError(f'No matching regions in SPATIAL_REGIONS for {region_names!r}')

    all_models = discover_models(variables, exclude_models)
    if model_names:
        models = [m for m in model_names if m in all_models]
        missing = [m for m in model_names if m not in all_models]
        if missing:
            print(f'Warning: requested models not found: {missing}')
    else:
        models = all_models
    print(f'Models ({len(models)}): {models}')
    print(f'Regions ({len(spatial_regions)}): {list(spatial_regions)}')

    import dask
    from dask.distributed import Client, LocalCluster

    cluster = None
    client = None
    if use_slurm:
        from dask_jobqueue import SLURMCluster  # optional dependency
        kw = dict(slurm_kwargs or {})
        n_jobs = kw.pop('n_jobs', 4)
        cluster = SLURMCluster(**kw)
        cluster.scale(jobs=n_jobs)
    else:
        cluster = LocalCluster(
            n_workers=n_workers, threads_per_worker=threads_per_worker, processes=True,
        )
    client = Client(cluster)
    print(f'Dask dashboard: {client.dashboard_link}', flush=True)

    try:
        tasks = [dask.delayed(process_one_model)(m, variables, spatial_regions) for m in models]
        results = list(dask.compute(*tasks))
    finally:
        client.close()
        cluster.close()

    frames = [df for df in results if df is not None and not df.empty]
    if not frames:
        raise RuntimeError('No data aggregated for any model/region.')
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(['region', 'model', 'month']).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    print(f'\nWrote {len(full)} rows -> {out_path}')

    meta = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'n_rows': int(len(full)),
        'n_models': len(models),
        'models': models,
        'variables': variables,
        'agg_vars': AGG_VARS,
        'regions': {
            name: {
                'lon_range': list(cfg['lon_range']),
                'lat_range': list(cfg['lat_range']),
                'surface_type': cfg.get('surface_type', 'all'),
                'edge_weighted': cfg.get('edge_weighted', False),
            }
            for name, cfg in spatial_regions.items()
        },
        'note': (
            'Monthly aggregates ignore any time_slice/season; season filtering '
            'happens downstream (see notebook_setup.seasonal_from_monthly_df).'
        ),
        'exclude_models': list(exclude_models or EXCLUDE_MODELS_DEFAULT),
        'parquet': str(out_path),
    }
    meta_path = Path(meta_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'Wrote meta -> {meta_path}')
    return full


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--meta', type=Path, default=DEFAULT_META)
    p.add_argument('--regions', nargs='*', default=None,
                   help='Subset of SPATIAL_REGIONS (default: all)')
    p.add_argument('--models', nargs='*', default=None,
                   help='Subset of AeroCom models (default: all non-excluded)')
    p.add_argument('--n-workers', type=int, default=8)
    p.add_argument('--threads-per-worker', type=int, default=1)
    p.add_argument('--slurm', action='store_true',
                   help='Use dask_jobqueue.SLURMCluster instead of a local cluster '
                        '(advanced; requires the dask-jobqueue package).')
    p.add_argument('--slurm-queue', type=str, default=None)
    p.add_argument('--slurm-cores', type=int, default=4)
    p.add_argument('--slurm-memory', type=str, default='16GB')
    p.add_argument('--slurm-jobs', type=int, default=4)
    p.add_argument('--slurm-walltime', type=str, default='02:00:00')
    args = p.parse_args(argv)

    slurm_kwargs = None
    if args.slurm:
        slurm_kwargs = dict(
            queue=args.slurm_queue,
            cores=args.slurm_cores,
            memory=args.slurm_memory,
            walltime=args.slurm_walltime,
            n_jobs=args.slurm_jobs,
        )

    build_monthly(
        out_path=args.out,
        meta_path=args.meta,
        region_names=args.regions,
        model_names=args.models,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        use_slurm=args.slurm,
        slurm_kwargs=slurm_kwargs,
    )


if __name__ == '__main__':
    main()
