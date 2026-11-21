"""Field loading and polygon aggregation for the map dashboard.

Uses the same ensemble NetCDFs and land/sea weights as
``py/regional_aggregates.py``. Polygons replace the rectangular box mask.
"""

from __future__ import annotations

import base64
import functools
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath

import PPE_set as cfg
import cameo_toolbox as ct
import regions as reg
from regional_aggregates import apply_burden_scale, regional_means

log = logging.getLogger("dashboard")

PROJECT_ROOT = cfg.PROJECT_ROOT
AEROCOM_DIR = PROJECT_ROOT / "Data" / "AP3_processed_monthly"

# Variables the UI can request. Derived AE keys are computed on the fly.
RAW_VARS = (
    "od550aer", "abs550aer",
    "od550dust", "abs550dust", "od550ss", "od550bc", "od550oa", "od550so4",
    "od440aer", "od865aer",
    "emidust", "emiss",
    "loaddust", "loadss", "loadbc", "loadoa", "loadso4",
    "ANG_550nm_865nm",
)
DERIVED_VARS = ("AE_440_550", "AE_550_865", "SSA")
ALL_VARS = RAW_VARS + DERIVED_VARS

VAR_LABELS = {
    "od550aer": "AOD$_{550}$",
    "abs550aer": "AAOD$_{550}$",
    "od550dust": "Dust AOD$_{550}$",
    "abs550dust": "Dust AAOD$_{550}$",
    "od550ss": "Sea-salt AOD$_{550}$",
    "od550bc": "BC AOD$_{550}$",
    "od550oa": "OA AOD$_{550}$",
    "od550so4": "SO$_4$ AOD$_{550}$",
    "od440aer": "AOD$_{440}$",
    "od865aer": "AOD$_{865}$",
    "emidust": "Dust emission",
    "emiss": "Sea-salt emission",
    "loaddust": "Dust load",
    "loadss": "Sea-salt load",
    "loadbc": "BC load",
    "loadoa": "OA load",
    "loadso4": "SO$_4$ load",
    "ANG_550nm_865nm": "AE 550–865 (archived)",
    "AE_440_550": "AE 440–550",
    "AE_550_865": "AE 550–865",
    "AE_440_670": "AE 440–670 (ICON)",
    "SSA": "Column SSA",
}

# Group preference for opening only what exists on disk.
GROUP_BY_VAR = {
    "od550aer": "optical", "abs550aer": "optical",
    "od550dust": "optical", "abs550dust": "optical",
    "od550ss": "optical", "od550bc": "optical", "od550oa": "optical", "od550so4": "optical",
    "od440aer": "optical", "od865aer": "optical", "ANG_550nm_865nm": "optical",
    "emidust": "emi", "emiss": "emi",
    "loaddust": "burden", "loadss": "burden", "loadbc": "burden",
    "loadoa": "burden", "loadso4": "burden",
}


def _lon360(lon: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(lon, dtype=float), 360.0)


def aerocom_models() -> List[str]:
    vdir = AEROCOM_DIR / "od550aer"
    if not vdir.is_dir():
        return []
    suffix = "_od550aer_processed.nc"
    names = sorted(p.name[: -len(suffix)] for p in vdir.glob(f"*{suffix}"))
    exclude = {
        "GEOS-i33p2-met2010_AP3-CTRL-2010",
        "GISS-ModelE2p1p1-OMA_AP3-CTRL-2010",
        "GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010",
        "GISS-ModelE2p1p1-MATRIX_AP3-CTRL",
        "NorESM2-met2010_AP3-CTRL-v3",
    }
    return [n for n in names if n not in exclude]


def available_ppe_ids() -> List[str]:
    out = []
    for ppe_id in ("pd2024", "pd2010"):
        if cfg.ensemble_file(ppe_id, "optical").is_file():
            out.append(ppe_id)
    return out


@functools.lru_cache(maxsize=16)
def _open_group(ppe_id: str, group: str) -> xr.Dataset:
    """Open one ensemble group. Burden loads are scaled; other groups are not."""
    path = cfg.ensemble_file(ppe_id, group)
    if not path.is_file():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path, engine=cfg.XR_WRITE_ENGINE)
    if group == "burden":
        ds = apply_burden_scale(ds, ppe_id)
    return ds


@functools.lru_cache(maxsize=8)
def _open_aerocom_var(model: str, var: str) -> Optional[xr.DataArray]:
    path = AEROCOM_DIR / var / f"{model}_{var}_processed.nc"
    if not path.is_file():
        return None
    ds = xr.open_dataset(path)
    name = var if var in ds.data_vars else next(iter(ds.data_vars))
    da = ds[name].load()
    ds.close()
    if "time" in da.dims:
        # Keep 2010 months; drop leap-day if present by month index later.
        da = da.assign_coords(time=pd.to_datetime(da.time.values))
    return da


def ppe_var_available(ppe_id: str, var: str) -> bool:
    if var in DERIVED_VARS:
        if var == "AE_440_550":
            return ppe_var_available(ppe_id, "od550aer") and ppe_var_available(ppe_id, "od440aer")
        if var == "AE_550_865":
            return (ppe_var_available(ppe_id, "ANG_550nm_865nm")
                    or (ppe_var_available(ppe_id, "od550aer") and ppe_var_available(ppe_id, "od865aer")))
        if var == "SSA":
            return ppe_var_available(ppe_id, "od550aer") and ppe_var_available(ppe_id, "abs550aer")
        return False
    group = GROUP_BY_VAR.get(var)
    if group is None:
        return False
    path = cfg.ensemble_file(ppe_id, group)
    if not path.is_file():
        return False
    try:
        ds = _open_group(ppe_id, group)
    except FileNotFoundError:
        return False
    return var in ds.data_vars


def aerocom_var_available(model: str, var: str) -> bool:
    if var in DERIVED_VARS:
        if var == "AE_440_550":
            return aerocom_var_available(model, "od550aer") and aerocom_var_available(model, "od440aer")
        if var == "AE_550_865":
            return aerocom_var_available(model, "od550aer") and (
                aerocom_var_available(model, "od865aer") or aerocom_var_available(model, "od870aer")
            )
        if var == "SSA":
            return aerocom_var_available(model, "od550aer") and aerocom_var_available(model, "abs550aer")
        return False
    if var == "od870aer":
        return (AEROCOM_DIR / "od870aer" / f"{model}_od870aer_processed.nc").is_file()
    return (AEROCOM_DIR / var / f"{model}_{var}_processed.nc").is_file()


def _derive(ds_or_fields: Dict[str, xr.DataArray], var: str) -> xr.DataArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        if var == "AE_440_550":
            aod, od440 = ds_or_fields["od550aer"], ds_or_fields["od440aer"]
            return -np.log(aod / od440) / np.log(550.0 / 440.0)
        if var == "AE_550_865":
            if "ANG_550nm_865nm" in ds_or_fields and ds_or_fields["ANG_550nm_865nm"] is not None:
                return ds_or_fields["ANG_550nm_865nm"]
            aod = ds_or_fields["od550aer"]
            if "od865aer" in ds_or_fields and ds_or_fields["od865aer"] is not None:
                return -np.log(aod / ds_or_fields["od865aer"]) / np.log(550.0 / 865.0)
            if "od870aer" in ds_or_fields and ds_or_fields["od870aer"] is not None:
                return -np.log(aod / ds_or_fields["od870aer"]) / np.log(550.0 / 870.0)
            raise KeyError("no 865/870 nm field for AE_550_865")
        if var == "SSA":
            return (1.0 - ds_or_fields["abs550aer"] / ds_or_fields["od550aer"]).clip(0.0, 1.0)
    raise KeyError(var)


def _month_mask(time: pd.DatetimeIndex, months: Sequence[int]) -> np.ndarray:
    return np.isin(time.month, list(months))


def _var_array(ppe_id: str, var: str) -> Optional[xr.DataArray]:
    group = GROUP_BY_VAR.get(var)
    if group is None:
        return None
    try:
        ds = _open_group(ppe_id, group)
    except FileNotFoundError:
        return None
    return ds[var] if var in ds else None


def _get_ppe_field(ppe_id: str, var: str) -> xr.DataArray:
    if var in DERIVED_VARS:
        need = {
            "AE_440_550": ("od550aer", "od440aer"),
            "AE_550_865": ("od550aer", "od865aer", "ANG_550nm_865nm"),
            "SSA": ("od550aer", "abs550aer"),
        }[var]
        return _derive({k: _var_array(ppe_id, k) for k in need}, var)
    da = _var_array(ppe_id, var)
    if da is None:
        raise KeyError(f"{var} not in {ppe_id} ensemble")
    return da


def _get_aerocom_field(model: str, var: str) -> xr.DataArray:
    if var in DERIVED_VARS:
        if var == "AE_440_550":
            fields = {"od550aer": _open_aerocom_var(model, "od550aer"),
                      "od440aer": _open_aerocom_var(model, "od440aer")}
        elif var == "AE_550_865":
            fields = {
                "od550aer": _open_aerocom_var(model, "od550aer"),
                "od865aer": _open_aerocom_var(model, "od865aer"),
                "od870aer": _open_aerocom_var(model, "od870aer"),
            }
        else:
            fields = {"od550aer": _open_aerocom_var(model, "od550aer"),
                      "abs550aer": _open_aerocom_var(model, "abs550aer")}
        if any(v is None for k, v in fields.items() if k in ("od550aer", "od440aer", "abs550aer")):
            raise KeyError(f"missing inputs for {var} in {model}")
        return _derive(fields, var)
    da = _open_aerocom_var(model, var)
    if da is None:
        raise KeyError(f"{var} not found for {model}")
    return da


def _select_time(da: xr.DataArray, months: Optional[Sequence[int]]) -> xr.DataArray:
    if months is None or "time" not in da.dims:
        return da
    times = pd.to_datetime(da.time.values)
    keep = _month_mask(times, months)
    if not keep.any():
        raise ValueError(f"no months {list(months)} in the time axis")
    return da.isel(time=np.where(keep)[0])


def _reduce_map(da: xr.DataArray, ens: Optional[int] = None) -> xr.DataArray:
    """Collapse ens/time to one (lat, lon) field.

    The ensemble mean is taken in member batches so a full year of every
    member is not held in memory at once.
    """
    work = da
    if "ens" in work.dims and ens is not None:
        if int(ens) not in set(np.asarray(work.ens.values).astype(int)):
            raise KeyError(f"ens={ens} not in ensemble")
        work = work.sel(ens=int(ens))
    elif "ens" in work.dims:
        nens = int(work.sizes["ens"])
        acc = count = None
        lat = lon = None
        for start in range(0, nens, 32):
            sub = work.isel(ens=slice(start, start + 32))
            if "time" in sub.dims:
                sub = sub.mean(dim="time", skipna=True)
            block = np.asarray(sub.transpose("ens", "lat", "lon").values, dtype=np.float32)
            finite = np.isfinite(block)
            if acc is None:
                acc = np.zeros(block.shape[1:], dtype=np.float64)
                count = np.zeros(block.shape[1:], dtype=np.float64)
                lat, lon = sub.lat, sub.lon
            acc += np.where(finite, block, 0.0).sum(axis=0)
            count += finite.sum(axis=0)
        out = np.full(acc.shape, np.nan, dtype=np.float32)
        ok = count > 0
        out[ok] = (acc[ok] / count[ok]).astype(np.float32)
        return xr.DataArray(out, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    if "time" in work.dims:
        work = work.mean(dim="time", skipna=True)
    return work.transpose("lat", "lon")


def _orient_for_map(lon, lat, data: np.ndarray):
    """North-up rows and longitudes running −180..180. One sample per cell."""
    data = np.asarray(data, dtype=np.float32)
    lon = _lon360(lon)
    signed = np.where(lon > 180.0, lon - 360.0, lon)
    order = np.argsort(signed)
    data = np.ascontiguousarray(data[:, order])
    lon = signed[order]
    lat = np.asarray(lat, dtype=float)
    if lat.size > 1 and lat[0] < lat[-1]:
        data = np.ascontiguousarray(data[::-1])
        lat = lat[::-1]
    diffs = np.diff(lon)
    diffs = diffs[diffs > 0]
    grid_deg = float(np.median(diffs)) if diffs.size else float("nan")
    return lon, lat, data, grid_deg


def pack_field(
    dataset: str,
    var: str,
    lon,
    lat,
    data: np.ndarray,
    months: Optional[Sequence[int]] = None,
    ens: Optional[int] = None,
    model: Optional[str] = None,
    members: Optional[Sequence[int]] = None,
    label: Optional[str] = None,
) -> dict:
    """Map payload. ``values_b64`` is little-endian float32, row-major, NaN if empty.

    Rows run north to south and columns west to east. Bytes stay much smaller
    than a nested JSON matrix.
    """
    del members
    lon, lat, data, grid_deg = _orient_for_map(lon, lat, data)
    finite = data[np.isfinite(data)]
    zmin = float(np.nanpercentile(finite, 2)) if finite.size else 0.0
    zmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
    if not np.isfinite(zmin):
        zmin = 0.0
    if not np.isfinite(zmax) or zmax <= zmin:
        zmax = zmin + 1.0
    return {
        "dataset": dataset,
        "model": model,
        "var": var,
        "label": label or VAR_LABELS.get(var, var),
        "months": list(months) if months else None,
        "ens": ens,
        "lon": lon.tolist(),
        "lat": lat.tolist(),
        "values_b64": base64.b64encode(data.tobytes()).decode("ascii"),
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "grid_deg": grid_deg,
        "zmin": zmin,
        "zmax": zmax,
        "n_finite": int(finite.size),
    }


_FIELD_CACHE: dict = {}
_FIELD_LOCK = threading.Lock()


def field_payload(
    dataset: str,
    var: str,
    months: Optional[Sequence[int]] = None,
    ens: Optional[int] = None,
    model: Optional[str] = None,
    spex_mode: str = "regridded",
    spex_res: float = 1.0,
    spex_year: Optional[int] = None,
) -> dict:
    key = (
        dataset, var, tuple(int(m) for m in months) if months else (),
        ens, model or "", spex_mode, float(spex_res), spex_year or 0,
    )
    with _FIELD_LOCK:
        cached = _FIELD_CACHE.get(key)
    if cached is not None:
        return cached
    payload = _field_payload(
        dataset, var, months=months, ens=ens, model=model,
        spex_mode=spex_mode, spex_res=spex_res, spex_year=spex_year,
    )
    with _FIELD_LOCK:
        if len(_FIELD_CACHE) >= 24:
            _FIELD_CACHE.pop(next(iter(_FIELD_CACHE)))
        _FIELD_CACHE[key] = payload
    return payload


def _field_payload(
    dataset: str,
    var: str,
    months: Optional[Sequence[int]] = None,
    ens: Optional[int] = None,
    model: Optional[str] = None,
    spex_mode: str = "regridded",
    spex_res: float = 1.0,
    spex_year: Optional[int] = None,
) -> dict:
    if dataset == "icon2025":
        from dashboard import iconmap
        return iconmap.field_payload(var, months=months, ens=ens)
    if dataset in ("spexone", "polder"):
        from dashboard import obsmap
        return obsmap.field_payload(
            dataset, var, months=months,
            res=float(spex_res), mode=spex_mode, year=spex_year,
        )
    if dataset in ("pd2024", "pd2010"):
        from dashboard import nc_reader
        lon, lat, data = nc_reader.map_mean_field(dataset, var, months=months, ens=ens)
        return pack_field(
            dataset=dataset, var=var, lon=lon, lat=lat, data=data,
            months=months, ens=ens, model=model,
        )
    if dataset == "aerocom":
        if not model:
            raise ValueError("model is required for AeroCom")
        da = _select_time(_get_aerocom_field(model, var), months)
        field = _reduce_map(da)
        return pack_field(
            dataset=dataset, var=var, lon=field.lon.values, lat=field.lat.values,
            data=np.asarray(field.values), months=months, ens=ens, model=model,
        )
    raise ValueError(f"unknown dataset {dataset!r}")


def _unwrap_ring_lons(verts: np.ndarray) -> np.ndarray:
    """Make consecutive ring longitudes continuous (|Δlon| ≤ 180)."""
    out = np.asarray(verts, dtype=float).copy()
    for i in range(1, out.shape[0]):
        while out[i, 0] - out[i - 1, 0] > 180.0:
            out[i, 0] -= 360.0
        while out[i, 0] - out[i - 1, 0] < -180.0:
            out[i, 0] += 360.0
    return out


def _ring_vertices(ring: Sequence[Sequence[float]]) -> np.ndarray:
    verts = np.asarray(ring, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError("ring must be a sequence of [lon, lat] pairs")
    if verts.shape[0] < 3:
        raise ValueError("polygon needs at least 3 vertices")
    if not np.allclose(verts[0], verts[-1]):
        verts = np.vstack([verts, verts[0]])
    return _unwrap_ring_lons(verts)


def points_in_ring(lon: np.ndarray, lat: np.ndarray, ring: Sequence[Sequence[float]]) -> np.ndarray:
    """True where each point falls inside the ring (longitudes −180..180 or 0..360)."""
    verts = _ring_vertices(ring)
    lon0 = _lon360(np.asarray(lon, dtype=float).ravel())
    lat0 = np.asarray(lat, dtype=float).ravel()
    lo_min = float(verts[:, 0].min()) - 1.0
    lo_max = float(verts[:, 0].max()) + 1.0
    candidates = np.stack([lon0, lon0 + 360.0, lon0 - 360.0], axis=1)
    in_window = (candidates >= lo_min) & (candidates <= lo_max)
    pick = np.where(in_window.any(axis=1), in_window.argmax(axis=1), 0)
    lon_pts = candidates[np.arange(candidates.shape[0]), pick]
    points = np.column_stack([lon_pts, lat0])
    return MplPath(verts, closed=True).contains_points(points, radius=1e-9)


def polygon_mask(lon: np.ndarray, lat: np.ndarray, ring: Sequence[Sequence[float]]) -> np.ndarray:
    """True where cell centres fall inside the lon/lat ring.

    Vertices are unwrapped so a box that crosses 0° (Sahara 18°W–30°E) stays a
    short rectangle, matching ``cameo_toolbox.create_mask``.
    """
    lon2d, lat2d = np.meshgrid(_lon360(lon), np.asarray(lat, dtype=float))
    return points_in_ring(lon2d, lat2d, ring).reshape(lon2d.shape)


def region_ring(cfg_r: dict) -> List[List[float]]:
    """Closed ring in degrees, unwrapped across 0° so Leaflet can draw it.

    Extra regions loaded from ``extra_regions.csv`` may store the drawn polygon
    in ``ring``; that outline is used instead of the bounding box.
    """
    stored = cfg_r.get("ring")
    if stored:
        ring = [[float(p[0]), float(p[1])] for p in stored]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        lons = [p[0] for p in ring]
        # A Pacific box stored as 212–277° sits only on the right-hand world
        # copy. Shift a compact box that lies past 180° into −180..180.
        if max(lons) - min(lons) <= 180.0 and max(lons) > 180.0:
            ring = [[lon - 360.0 if lon > 180.0 else lon, lat] for lon, lat in ring]
        return ring
    lo0, lo1 = float(cfg_r["lon_range"][0]), float(cfg_r["lon_range"][1])
    la0, la1 = float(cfg_r["lat_range"][0]), float(cfg_r["lat_range"][1])
    south, north = min(la0, la1), max(la0, la1)
    if lo0 > lo1:
        west, east = lo0 - 360.0, lo1
    else:
        west = lo0 - 360.0 if lo0 > 180.0 else lo0
        east = lo1 - 360.0 if lo1 > 180.0 else lo1
        if west > east:
            west, east = (lo0 - 360.0 if lo0 > 180.0 else lo0), (lo1 - 360.0 if lo1 > 180.0 else lo1)
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


def region_weights_polygon(
    template: xr.DataArray,
    ring: Sequence[Sequence[float]],
    surface_type: str = "all",
) -> xr.DataArray:
    """Area weights inside ``ring``, using the same land/sea and cosine weights
    as ``regional_aggregates.weights_from_defs`` / ``_spatial_weights``."""
    tpl = ct._spatial_template(template)
    lon = np.asarray(tpl.lon.values, dtype=float)
    lat = np.asarray(tpl.lat.values, dtype=float)
    inside = polygon_mask(lon, lat, ring)
    box = xr.DataArray(
        inside.astype(float),
        coords={"lat": tpl.lat, "lon": tpl.lon},
        dims=("lat", "lon"),
        name="drawn",
    )
    surface = ct._land_sea_mask(box, surface_type)
    mask = (box * surface).astype(float)
    mask.attrs.update({"name": "drawn", "surface": surface_type})
    weights, _ = ct._spatial_weights(tpl, mask, edge_weighted=False)
    return weights.fillna(0.0).transpose("lat", "lon")


def _weighted_mean_2d(field: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(field)
    w = weights * finite
    den = w.sum()
    if den <= 0:
        return float("nan")
    return float(np.nansum(np.where(finite, field, 0.0) * weights) / den)


def region_mean_payload(
    dataset: str,
    var: str,
    ring: Sequence[Sequence[float]],
    months: Optional[Sequence[int]] = None,
    surface_type: str = "all",
    model: Optional[str] = None,
    ens: Optional[int] = None,
) -> dict:
    if dataset == "icon2025":
        from dashboard import iconmap
        return iconmap.region_mean_payload(
            var, ring, months=months, surface_type=surface_type, ens=ens,
        )
    if dataset in ("spexone", "polder"):
        from dashboard import obsmap
        return obsmap.region_mean_payload(
            dataset, var, ring, months=months, surface_type=surface_type,
        )
    if dataset in ("pd2024", "pd2010"):
        da = _select_time(_get_ppe_field(dataset, var), months)
        if ens is not None and "ens" in da.dims:
            da = da.sel(ens=ens)
        tpl = da
        for dim in ("ens", "time"):
            if dim in tpl.dims:
                tpl = tpl.isel({dim: 0}, drop=True)
        weights = region_weights_polygon(tpl, ring, surface_type=surface_type)
        w = np.asarray(weights.values, dtype=np.float64)
        # regional_means expects weights with a leading region dim.
        w_reg = weights.expand_dims(region=["drawn"])

        if "ens" in da.dims and "time" in da.dims:
            # Same path as the seasonal tables: (ens, time) then month-average.
            etr = regional_means(da, w_reg)  # ens, time, region=1
            with np.errstate(all="ignore"):
                member_means = np.nanmean(etr[:, :, 0], axis=1)
            member_means = member_means[np.isfinite(member_means)]
            map_mean = float(np.nanmean(member_means)) if member_means.size else float("nan")
        elif "time" in da.dims:
            # Promote a singleton ens so regional_means applies.
            etr = regional_means(da.expand_dims(ens=[0]), w_reg)
            with np.errstate(all="ignore"):
                map_mean = float(np.nanmean(etr[0, :, 0]))
            member_means = np.array([])
        else:
            member_means = np.array([])
            map_mean = _weighted_mean_2d(np.asarray(da.transpose("lat", "lon").values, float), w)

        summary = {"mean": map_mean, "n_members": int(member_means.size)}
        if member_means.size:
            summary.update({
                "median": float(np.median(member_means)),
                "p05": float(np.percentile(member_means, 5)),
                "p95": float(np.percentile(member_means, 95)),
                "min": float(np.min(member_means)),
                "max": float(np.max(member_means)),
            })
    elif dataset == "aerocom":
        if not model:
            raise ValueError("model is required for AeroCom")
        da = _select_time(_get_aerocom_field(model, var), months)
        tpl = da.isel(time=0, drop=True) if "time" in da.dims else da
        weights = region_weights_polygon(tpl, ring, surface_type=surface_type)
        w = np.asarray(weights.values, dtype=np.float64)
        if "time" in da.dims:
            xt = np.nanmean(np.asarray(da.transpose("time", "lat", "lon").values, float), axis=0)
        else:
            xt = np.asarray(da.transpose("lat", "lon").values, float)
        map_mean = _weighted_mean_2d(xt, w)
        summary = {"mean": map_mean, "n_members": 0}
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    return {
        "dataset": dataset,
        "model": model,
        "var": var,
        "label": VAR_LABELS.get(var, var),
        "months": list(months) if months else None,
        "surface_type": surface_type,
        "n_cells": int((w > 0).sum()),
        **summary,
    }


@functools.lru_cache(maxsize=1)
def meta_payload() -> dict:
    ppe_ids = available_ppe_ids()
    models = aerocom_models()
    datasets = []
    for ppe_id in ppe_ids:
        run = cfg.get_run(ppe_id)
        vars_ok = [v for v in ALL_VARS if ppe_var_available(ppe_id, v)]
        ds = _open_group(ppe_id, "optical")
        times = pd.to_datetime(ds.time.values) if "time" in ds.dims else pd.DatetimeIndex([])
        months = sorted({int(m) for m in times.month}) if len(times) else list(range(1, 13))
        members = [int(e) for e in sorted(set(ds.ens.values.astype(int)))] if "ens" in ds.dims else []
        datasets.append({
            "id": ppe_id,
            "kind": "ppe",
            "label": run["label"],
            "period": list(run["period"]),
            "variables": vars_ok,
            "months": months,
            "members": members,
        })
    # AeroCom: union of variables that at least one model has; UI filters per model.
    aero_vars = []
    for v in ALL_VARS:
        if any(aerocom_var_available(m, v) for m in models[:5]):  # sample a few
            aero_vars.append(v)
        elif v in ("od550aer", "abs550aer", "od550dust", "emidust", "emiss",
                   "loaddust", "AE_440_550", "SSA"):
            aero_vars.append(v)
    datasets.append({
        "id": "aerocom",
        "kind": "aerocom",
        "label": "AeroCom AP3",
        "period": ["2010-01-01", "2010-12-31"],
        "variables": sorted(set(aero_vars), key=lambda x: ALL_VARS.index(x) if x in ALL_VARS else 99),
        "months": list(range(1, 13)),
        "members": [],
        "models": models,
    })
    from dashboard import iconmap, obsmap
    datasets.extend(iconmap.dataset_meta())
    datasets.extend(obsmap.dataset_meta())

    boxes = []
    for name, cfg_r in reg.REGIONS.items():
        if name == "global":
            continue
        ring = region_ring(cfg_r)
        boxes.append({
            "name": name,
            "lon_range": [float(cfg_r["lon_range"][0]), float(cfg_r["lon_range"][1])],
            "lat_range": [float(cfg_r["lat_range"][0]), float(cfg_r["lat_range"][1])],
            "surface_type": cfg_r.get("surface_type", "all"),
            "season_months": list(cfg_r.get("season_months") or []),
            "species": list(cfg_r.get("species") or []),
            "role": cfg_r.get("role"),
            "color": cfg_r.get("color", "#d62728"),
            "ring": ring,
        })

    seasons = {
        "JJA": [6, 7, 8],
        "DJF": [12, 1, 2],
        "SOND": [9, 10, 11, 12],
        "ASON": [8, 9, 10, 11],
        "Sep-Feb": [9, 10, 11, 12, 1, 2],
        "annual": list(range(1, 13)),
    }
    return {
        "datasets": datasets,
        "variables": [{"id": v, "label": VAR_LABELS.get(v, v)} for v in list(ALL_VARS) + ["AE_440_670"]],
        "regions": boxes,
        "seasons": seasons,
    }
