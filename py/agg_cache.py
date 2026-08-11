"""Regional-mean and POLDER intermediate caches for AAOD/AOD notebooks.

Caches live under ``project_root/tables/cache/`` as parquet + a JSON meta
fingerprint so a notebook can skip NetCDF load / aggregation / POLDER
homogenization when configuration is unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


def cache_dir(table_dir: Path) -> Path:
    d = Path(table_dir) / 'cache'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(_jsonable(dict(config)), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def meta_path(table_dir: Path, prefix: str) -> Path:
    return cache_dir(table_dir) / f'{prefix}_meta.json'


def write_meta(table_dir: Path, prefix: str, meta: Mapping[str, Any]) -> Path:
    path = meta_path(table_dir, prefix)
    path.write_text(json.dumps(_jsonable(dict(meta)), indent=2) + '\n')
    return path


def read_meta(table_dir: Path, prefix: str) -> Optional[dict]:
    path = meta_path(table_dir, prefix)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def cache_is_valid(table_dir: Path, prefix: str, expected_fp: str,
                   required_files: Iterable[str]) -> bool:
    meta = read_meta(table_dir, prefix)
    if not meta or meta.get('fingerprint') != expected_fp:
        return False
    base = cache_dir(table_dir)
    return all((base / name).exists() for name in required_files)


# ---------------------------------------------------------------------------
# model_seasonal: region -> var -> model -> scalar
# ---------------------------------------------------------------------------

def _to_float(val: Any) -> float:
    if val is None:
        return np.nan
    if hasattr(val, 'values'):
        try:
            return float(np.asarray(val).mean())
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def seasonal_to_frame(model_seasonal: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> pd.DataFrame:
    rows = []
    for region, var_dict in model_seasonal.items():
        for var, model_dict in (var_dict or {}).items():
            for model, val in (model_dict or {}).items():
                rows.append({
                    'region': region,
                    'variable': var,
                    'model': model,
                    'value': _to_float(val),
                })
    return pd.DataFrame(rows, columns=['region', 'variable', 'model', 'value'])


def frame_to_seasonal(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    if df is None or df.empty:
        return out
    for (region, var), g in df.groupby(['region', 'variable'], sort=False):
        out.setdefault(region, {})[var] = {
            str(m): float(v) if pd.notna(v) else np.nan
            for m, v in zip(g['model'], g['value'])
        }
    return out


def save_model_seasonal(table_dir: Path, prefix: str,
                        model_seasonal: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Path:
    path = cache_dir(table_dir) / f'{prefix}_model_seasonal.parquet'
    seasonal_to_frame(model_seasonal).to_parquet(path, index=False)
    return path


def load_model_seasonal(table_dir: Path, prefix: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    path = cache_dir(table_dir) / f'{prefix}_model_seasonal.parquet'
    return frame_to_seasonal(pd.read_parquet(path))


def save_lifetime_raw(table_dir: Path, prefix: str,
                      lifetime_raw: Mapping[str, Mapping[str, Any]]) -> Path:
    """Save lifetime_BC_OA_raw as region/model/value parquet."""
    rows = []
    for region, md in (lifetime_raw or {}).items():
        for model, val in (md or {}).items():
            rows.append({'region': region, 'model': model, 'value': _to_float(val)})
    path = cache_dir(table_dir) / f'{prefix}_lifetime_raw.parquet'
    pd.DataFrame(rows, columns=['region', 'model', 'value']).to_parquet(path, index=False)
    return path


def load_lifetime_raw(table_dir: Path, prefix: str) -> Dict[str, Dict[str, float]]:
    path = cache_dir(table_dir) / f'{prefix}_lifetime_raw.parquet'
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: Dict[str, Dict[str, float]] = {}
    for region, g in df.groupby('region', sort=False):
        out[str(region)] = {
            str(m): float(v) if pd.notna(v) else np.nan
            for m, v in zip(g['model'], g['value'])
        }
    return out


# ---------------------------------------------------------------------------
# POLDER monthly regional means: dict[region] -> DataFrame
# ---------------------------------------------------------------------------

def save_polder_monthly(table_dir: Path, prefix: str,
                        polder_monthly: Mapping[str, pd.DataFrame]) -> Path:
    frames = []
    for region, df in polder_monthly.items():
        if df is None or df.empty:
            continue
        part = df.copy()
        part['region'] = region
        if 'month' in part.columns:
            part['month'] = part['month'].astype(str)
        frames.append(part)
    path = cache_dir(table_dir) / f'{prefix}_polder_monthly.parquet'
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    else:
        pd.DataFrame({'region': pd.Series(dtype=str)}).to_parquet(path, index=False)
    return path


def load_polder_monthly(table_dir: Path, prefix: str,
                        regions: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    path = cache_dir(table_dir) / f'{prefix}_polder_monthly.parquet'
    df = pd.read_parquet(path)
    out: Dict[str, pd.DataFrame] = {}
    region_list = list(regions) if regions is not None else (
        sorted(df['region'].unique()) if 'region' in df.columns else []
    )
    for region in region_list:
        if df.empty or 'region' not in df.columns:
            out[region] = pd.DataFrame()
            continue
        part = df[df['region'] == region].drop(columns=['region']).copy()
        if 'month' in part.columns:
            part['month'] = part['month'].astype(str).map(lambda s: pd.Period(s, freq='M'))
        if 'time' in part.columns:
            part['time'] = pd.to_datetime(part['time'])
        out[region] = part.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Homogenized / regional obs scalars
# ---------------------------------------------------------------------------

def save_obs_by_region(table_dir: Path, prefix: str, name: str,
                       obs: Mapping[str, Any]) -> Path:
    rows = [{'region': r, 'value': _to_float(v)} for r, v in (obs or {}).items()]
    path = cache_dir(table_dir) / f'{prefix}_{name}.parquet'
    pd.DataFrame(rows, columns=['region', 'value']).to_parquet(path, index=False)
    return path


def load_obs_by_region(table_dir: Path, prefix: str, name: str) -> Dict[str, float]:
    path = cache_dir(table_dir) / f'{prefix}_{name}.parquet'
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {
        str(r): float(v) if pd.notna(v) else np.nan
        for r, v in zip(df['region'], df['value'])
    }


def save_json(table_dir: Path, prefix: str, name: str, obj: Any) -> Path:
    path = cache_dir(table_dir) / f'{prefix}_{name}.json'

    def _default(o):
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=_default) + '\n')
    return path


def load_json(table_dir: Path, prefix: str, name: str) -> Any:
    path = cache_dir(table_dir) / f'{prefix}_{name}.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_ensemble_obs(table_dir: Path, prefix: str,
                      ensemble_obs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Path:
    """Save {ensemble: {aod_obs: {region: val}, ae_obs: {...}}} as long parquet."""
    rows = []
    for ens, bundle in (ensemble_obs or {}).items():
        for kind, region_map in bundle.items():
            for region, val in (region_map or {}).items():
                rows.append({
                    'ensemble': ens,
                    'kind': kind,
                    'region': region,
                    'value': _to_float(val),
                })
    path = cache_dir(table_dir) / f'{prefix}_ensemble_obs.parquet'
    pd.DataFrame(rows, columns=['ensemble', 'kind', 'region', 'value']).to_parquet(path, index=False)
    return path


def load_ensemble_obs(table_dir: Path, prefix: str
                      ) -> Dict[str, Dict[str, Dict[str, float]]]:
    path = cache_dir(table_dir) / f'{prefix}_ensemble_obs.parquet'
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (ens, kind), g in df.groupby(['ensemble', 'kind'], sort=False):
        out.setdefault(str(ens), {})[str(kind)] = {
            str(r): float(v) if pd.notna(v) else np.nan
            for r, v in zip(g['region'], g['value'])
        }
    return out


def save_polder_samples(table_dir: Path, prefix: str,
                        samples: Mapping[Tuple[str, str, str], float]) -> Path:
    """samples keyed by (region, model, var) -> float."""
    rows = [
        {'region': r, 'model': m, 'variable': v, 'value': _to_float(val)}
        for (r, m, v), val in (samples or {}).items()
    ]
    path = cache_dir(table_dir) / f'{prefix}_polder_samples.parquet'
    pd.DataFrame(rows, columns=['region', 'model', 'variable', 'value']).to_parquet(path, index=False)
    return path


def load_polder_samples(table_dir: Path, prefix: str) -> Dict[Tuple[str, str, str], float]:
    path = cache_dir(table_dir) / f'{prefix}_polder_samples.parquet'
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {
        (str(r), str(m), str(v)): float(val) if pd.notna(val) else np.nan
        for r, m, v, val in zip(df['region'], df['model'], df['variable'], df['value'])
    }
