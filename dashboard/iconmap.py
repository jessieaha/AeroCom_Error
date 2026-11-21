"""ICON-HAM PPE (hjia) on the native R2B4 grid.

Maps are area-binned to 1° so the browser can draw them. A drawn polygon is
averaged on the native cells with the same spherical-triangle areas and
Natural Earth land/sea test as ``py/icon_regional.py``.
"""

from __future__ import annotations

import functools
from typing import List, Optional, Sequence

import numpy as np

import icon_regional as ir
from dashboard import data as D
from dashboard.grid import LAT, LON, bin_indices

N_LAT = int(LAT.size)
N_LON = int(LON.size)


def _root():
    return ir._icon_root()


def member_ids() -> List[int]:
    return [int(m) for m in ir.member_ids(_root())]


def _folder_ready(name: str) -> bool:
    folder = ir.ICON_VARS.get(name)
    return bool(folder) and (_root() / folder).is_dir()


def available_vars() -> List[str]:
    hidden = {"ANG_440nm_670nm", "ANG_550nm_865nm"}
    out = [name for name in ir.ICON_VARS if name not in hidden and _folder_ready(name)]
    if _folder_ready("od550aer") and _folder_ready("abs550aer"):
        out.append("SSA")
    if _folder_ready("ANG_440nm_670nm"):
        out.append("AE_440_670")
    if _folder_ready("ANG_550nm_865nm"):
        out.append("AE_550_865")
    return out


@functools.lru_cache(maxsize=1)
def geometry():
    """Cell centres, area (steradians), land mask and the month axis."""
    sample = _root() / "TAU_2D_550nm" / "ens_000_PD_TAU_2D_550nm.nc"
    import xarray as xr
    with xr.open_dataset(sample) as ds:
        lon = np.degrees(np.asarray(ds["clon"].values, dtype=float)) % 360.0
        lat = np.degrees(np.asarray(ds["clat"].values, dtype=float))
        blo = np.asarray(ds["clon_bnds"].values, dtype=float)
        bla = np.asarray(ds["clat_bnds"].values, dtype=float)
        time = np.asarray(ds["time"].values)
    area = ir._spherical_excess(blo, bla)
    area = np.where(np.isfinite(area) & (area > 0), area, 0.0)
    import cameo_toolbox as ct
    land = np.asarray(ct.point_is_land(lon, lat), dtype=bool)
    months = np.array([int(t) // 100 % 100 for t in time], dtype=int)
    return {"lon": lon, "lat": lat, "area": area, "land": land, "months": months}


def dataset_meta() -> list:
    if not (_root() / "TAU_2D_550nm").is_dir():
        return []
    geom = geometry()
    months = sorted({int(m) for m in geom["months"]})
    run = __import__("PPE_set", fromlist=["get_run"]).get_run("icon2025")
    return [{
        "id": "icon2025",
        "kind": "ppe",
        "label": run["label"],
        "period": list(run["period"]),
        "variables": available_vars(),
        "months": months,
        "members": member_ids(),
        "default_member": 0,
        "note": (
            "ICON-HAM (hjia), native R2B4. The map is an area-weighted 1° bin. "
            "A drawn polygon uses the native cell areas. "
            "Ensemble mean reads every member the first time."
        ),
    }]


def _raw_cells(member: int, var: str, months: Sequence[int]) -> np.ndarray:
    folder = ir.ICON_VARS[var]
    opened = ir._open_field(_root(), f"{member:03d}", folder)
    n = geometry()["lon"].size
    if opened is None:
        return np.full(n, np.nan)
    data, time = opened
    mon = np.array([int(t) // 100 % 100 for t in time], dtype=int)
    keep = np.isin(mon, list(months))
    if not np.any(keep):
        raise ValueError(f"no ICON months {list(months)} in member {member}")
    with np.errstate(all="ignore"):
        return np.nanmean(data[keep], axis=0)


def cell_mean(member: int, var: str, months: Sequence[int]) -> np.ndarray:
    if var == "SSA":
        od = _raw_cells(member, "od550aer", months)
        ab = _raw_cells(member, "abs550aer", months)
        with np.errstate(divide="ignore", invalid="ignore"):
            ssa = 1.0 - ab / od
        ssa = np.where(np.isfinite(od) & (od > 0), ssa, np.nan)
        return np.clip(ssa, 0.0, 1.0)
    if var == "AE_440_670":
        return _raw_cells(member, "ANG_440nm_670nm", months)
    if var == "AE_550_865":
        return _raw_cells(member, "ANG_550nm_865nm", months)
    if var not in ir.ICON_VARS:
        raise KeyError(var)
    return _raw_cells(member, var, months)


def bin_cells(values: np.ndarray) -> np.ndarray:
    geom = geometry()
    area = geom["area"]
    finite = np.isfinite(values) & (area > 0)
    out = np.full((N_LAT, N_LON), np.nan)
    if not np.any(finite):
        return out
    iy, ix = bin_indices(geom["lon"][finite], geom["lat"][finite])
    acc = np.zeros((N_LAT, N_LON))
    w = np.zeros((N_LAT, N_LON))
    np.add.at(acc, (iy, ix), values[finite] * area[finite])
    np.add.at(w, (iy, ix), area[finite])
    ok = w > 0
    out[ok] = acc[ok] / w[ok]
    return out


def _months(months: Optional[Sequence[int]]) -> List[int]:
    if months:
        return [int(m) for m in months]
    return sorted({int(m) for m in geometry()["months"]})


def _member_grid(member: int, var: str, months: Sequence[int]) -> np.ndarray:
    return bin_cells(cell_mean(member, var, months))


def _grid_cache(var: str, months: Sequence[int], ens: Optional[int]):
    from pathlib import Path
    cache = Path(__file__).resolve().parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tag = "mean" if ens is None else f"e{int(ens):03d}"
    month_tag = "-".join(f"{int(m):02d}" for m in months)
    return cache / f"icon_{var}_{month_tag}_{tag}.npy"


def gridded(var: str, months: Sequence[int], ens: Optional[int]) -> np.ndarray:
    """1° map. The binned result is saved so the next request does not reread ICON."""
    path = _grid_cache(var, months, ens)
    if path.is_file():
        return np.load(path)
    if ens is not None:
        out = _member_grid(int(ens), var, months)
    else:
        acc = np.zeros((N_LAT, N_LON))
        count = np.zeros((N_LAT, N_LON))
        # One file at a time. NetCDF/HDF5 aborts if several members are read at once.
        for member in member_ids():
            grid = _member_grid(member, var, months)
            finite = np.isfinite(grid)
            acc[finite] += grid[finite]
            count[finite] += 1.0
        out = np.full_like(acc, np.nan)
        ok = count > 0
        out[ok] = acc[ok] / count[ok]
    np.save(path, np.asarray(out, dtype=np.float32))
    return out


def field_payload(var: str, months: Optional[Sequence[int]] = None, ens: Optional[int] = None) -> dict:
    months = _months(months)
    grid = gridded(var, months, ens)
    return D.pack_field(
        dataset="icon2025",
        var=var,
        lon=LON,
        lat=LAT,
        data=grid,
        months=months,
        ens=ens,
        members=member_ids(),
        label=D.VAR_LABELS.get(var, var),
    )


def _cell_weights(ring, surface_type: str) -> np.ndarray:
    geom = geometry()
    inside = D.points_in_ring(geom["lon"], geom["lat"], ring)
    if surface_type == "land":
        inside = inside & geom["land"]
    elif surface_type in ("ocean", "sea"):
        inside = inside & ~geom["land"]
    elif surface_type != "all":
        raise ValueError("surface_type must be all, land, or ocean")
    return np.where(inside, geom["area"], 0.0)


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & (weights > 0)
    den = weights[finite].sum()
    if den <= 0:
        return float("nan")
    return float(np.sum(values[finite] * weights[finite]) / den)


def region_mean_payload(var, ring, months=None, surface_type="all", ens: Optional[int] = None) -> dict:
    months = _months(months)
    weights = _cell_weights(ring, surface_type)
    if weights.sum() <= 0:
        raise ValueError("polygon contains no ICON cells for that surface")

    def one(member: int) -> float:
        return _weighted(cell_mean(member, var, months), weights)

    targets = [int(ens)] if ens is not None else member_ids()
    means = []
    for member in targets:
        val = one(member)
        if np.isfinite(val):
            means.append(val)
    arr = np.asarray(means, dtype=float)
    summary = {
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "n_members": int(arr.size),
    }
    if arr.size:
        summary.update({
            "median": float(np.median(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })
    return {
        "dataset": "icon2025",
        "model": None,
        "var": var,
        "label": D.VAR_LABELS.get(var, var),
        "months": months,
        "surface_type": surface_type,
        "n_cells": int((weights > 0).sum()),
        **summary,
    }
