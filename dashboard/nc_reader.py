"""netCDF4 readers for ensemble fields used by the dashboard and box job.

Opens one group at a time and reads in the file's chunk size (optical is
typically 32 members × 12 months × 96 × 192) instead of loading the whole
ensemble through xarray.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

import PPE_set as cfg
from regional_aggregates import apply_burden_scale

# Match on-disk chunking of optical_monthly.nc.
ENS_CHUNK = 32

GROUP_VARS = {
    "optical": (
        "od550aer", "abs550aer", "od440aer", "od865aer",
        "od550dust", "abs550dust", "od550ss", "od550bc", "od550oa", "od550so4",
        "ANG_550nm_865nm",
    ),
    "burden": ("loaddust", "loadss", "loadbc", "loadoa", "loadso4"),
    "emi": ("emidust", "emiss", "emibc", "emioa", "emiso2", "emi_SO4", "emi_DMS"),
    "precip": ("precip",),
    "meteo": ("stress_water",),
}


def ensemble_path(ppe_id: str, group: str) -> Path:
    return cfg.ensemble_file(ppe_id, group)


def open_group(ppe_id: str, group: str) -> nc.Dataset:
    path = ensemble_path(ppe_id, group)
    if not path.is_file():
        raise FileNotFoundError(path)
    return nc.Dataset(str(path), "r")


def _coords(ds: nc.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    lon = np.asarray(ds.variables["lon"][:], dtype=float)
    lat = np.asarray(ds.variables["lat"][:], dtype=float)
    ens = np.asarray(ds.variables["ens"][:], dtype=int)
    raw_time = ds.variables["time"]
    units = getattr(raw_time, "units", "days since 2000-01-01")
    calendar = getattr(raw_time, "calendar", "standard")
    stamped = nc.num2date(raw_time[:], units, calendar=calendar)
    times = pd.to_datetime([f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in stamped])
    return lon, lat, ens, times


def iter_ens_chunks(n_ens: int, chunk: int = ENS_CHUNK) -> Iterator[slice]:
    for start in range(0, n_ens, chunk):
        yield slice(start, min(start + chunk, n_ens))


def read_var(
    ds: nc.Dataset,
    name: str,
    ens_slice: Optional[slice] = None,
    time_slice: Optional[slice] = None,
) -> np.ndarray:
    """Read ``name`` as float32 with optional ens/time slices. Shape (ens, time, lat, lon)."""
    if name not in ds.variables:
        raise KeyError(name)
    var = ds.variables[name]
    e = ens_slice or slice(None)
    t = time_slice or slice(None)
    data = np.asarray(var[e, t, :, :], dtype=np.float32)
    return data


def load_group_dataset(ppe_id: str, group: str, var_names: Optional[Sequence[str]] = None) -> xr.Dataset:
    """Load one ensemble group into an xarray Dataset via netCDF4 chunked reads."""
    with open_group(ppe_id, group) as ds:
        lon, lat, ens, times = _coords(ds)
        names = [v for v in (var_names or GROUP_VARS.get(group, ())) if v in ds.variables]
        if not names and var_names:
            names = [v for v in var_names if v in ds.variables]
        if not names:
            # Fall back to every 4-D float field.
            names = [
                n for n, v in ds.variables.items()
                if getattr(v, "ndim", 0) == 4 and n not in ("lon", "lat", "ens", "time")
            ]
        data_vars = {}
        n_ens = len(ens)
        for name in names:
            blocks = []
            for sl in iter_ens_chunks(n_ens):
                blocks.append(read_var(ds, name, ens_slice=sl))
            data_vars[name] = (["ens", "time", "lat", "lon"], np.concatenate(blocks, axis=0))
        out = xr.Dataset(
            data_vars,
            coords={
                "ens": ens,
                "time": times,
                "lat": lat,
                "lon": lon,
            },
        )
    if group == "burden":
        out = apply_burden_scale(out, ppe_id)
    return out


def open_available(ppe_id: str, groups: Sequence[str] = ("optical", "burden", "emi", "precip", "meteo")) -> xr.Dataset:
    """Merge available ensemble groups for one PPE id (netCDF4 underneath)."""
    parts: List[xr.Dataset] = []
    for group in groups:
        path = ensemble_path(ppe_id, group)
        if path.is_file():
            parts.append(load_group_dataset(ppe_id, group))
    if not parts:
        raise FileNotFoundError(f"no ensemble files for {ppe_id}")
    return xr.merge(parts, compat="override", join="outer")


def map_mean_field(
    ppe_id: str,
    var: str,
    months: Optional[Sequence[int]] = None,
    ens: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lon, lat, 2-D map) for one PPE variable via netCDF4.

    Derived vars (SSA, AE) are computed from the optical group.
    """
    from dashboard import data as D

    group = D.GROUP_BY_VAR.get(var)
    if var in D.DERIVED_VARS:
        group = "optical"
    if group is None:
        raise KeyError(var)

    with open_group(ppe_id, group) as ds:
        lon, lat, ens_ids, times = _coords(ds)
        month_idx = np.arange(len(times))
        if months:
            month_of = times.month.to_numpy()
            keep = np.isin(month_of, list(months))
            if not keep.any():
                raise ValueError(f"no months {months} in {ppe_id}")
            month_idx = month_idx[keep]

        def _reduce(block: np.ndarray) -> np.ndarray:
            # block: (ens, time, lat, lon) or (time, lat, lon)
            if block.ndim == 4:
                finite = np.isfinite(block)
                acc = np.where(finite, block, 0.0).sum(axis=(0, 1))
                cnt = finite.sum(axis=(0, 1))
            else:
                finite = np.isfinite(block)
                acc = np.where(finite, block, 0.0).sum(axis=0)
                cnt = finite.sum(axis=0)
            out = np.full(acc.shape, np.nan, dtype=np.float32)
            ok = cnt > 0
            out[ok] = (acc[ok] / cnt[ok]).astype(np.float32)
            return out

        def _load_raw(name: str) -> np.ndarray:
            if ens is not None:
                if int(ens) not in ens_ids:
                    raise KeyError(f"ens {ens} not in {ppe_id}")
                e_i = int(np.where(ens_ids == int(ens))[0][0])
                return np.asarray(ds.variables[name][e_i, month_idx, :, :], dtype=np.float32)
            blocks = []
            for sl in iter_ens_chunks(len(ens_ids)):
                # Read full time then index months to keep chunk-aligned ens reads.
                chunk = np.asarray(ds.variables[name][sl, :, :, :], dtype=np.float32)
                blocks.append(chunk[:, month_idx, :, :])
            return np.concatenate(blocks, axis=0)

        if var in D.DERIVED_VARS:
            fields = {}
            for need in ("od550aer", "abs550aer", "od440aer", "od865aer", "ANG_550nm_865nm"):
                if need in ds.variables:
                    fields[need] = _load_raw(need)
            # Build xarray-like dict for _derive on numpy via DataArrays
            da_fields: Dict[str, xr.DataArray] = {
                k: xr.DataArray(v) for k, v in fields.items()
            }
            derived = D._derive(da_fields, var)
            data = _reduce(np.asarray(derived.values, dtype=np.float32))
        else:
            if var not in ds.variables:
                raise KeyError(var)
            data = _reduce(_load_raw(var))
        return lon, lat, data


def polygon_bounds(ring: Sequence[Sequence[float]]) -> Dict[str, float]:
    lons = [float(p[0]) for p in ring]
    lats = [float(p[1]) for p in ring]
    return {
        "lon_min": float(min(lons)),
        "lon_max": float(max(lons)),
        "lat_min": float(min(lats)),
        "lat_max": float(max(lats)),
    }
