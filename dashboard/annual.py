"""Annual cycle of a drawn polygon: spatial mean and spatial standard deviation.

The standard deviation is the area-weighted spread of cell values inside the
polygon for that month's map (ensemble-mean map, or the selected member).
It is not the ensemble spread.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dashboard import data as D


def spatial_mean_std(field: np.ndarray, weights: np.ndarray):
    """Area-weighted mean and standard deviation. Returns (mean, std, n_cells)."""
    field = np.asarray(field, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if field.size != weights.size:
        raise ValueError("field and weights have different shapes")
    finite = np.isfinite(field) & np.isfinite(weights) & (weights > 0)
    n = int(finite.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    w = weights[finite]
    x = field[finite]
    den = float(w.sum())
    mean = float(np.sum(w * x) / den)
    var = float(np.sum(w * (x - mean) ** 2) / den)
    return mean, float(np.sqrt(max(var, 0.0))), n


def _by_calendar_month(months: Sequence[int], means, stds, counts):
    """Average repeated calendar months (ICON has June in two years)."""
    bucket = {}
    for month, mean, std, n in zip(months, means, stds, counts):
        bucket.setdefault(int(month), []).append((mean, std, n))
    rows = []
    for month in range(1, 13):
        parts = bucket.get(month)
        if not parts:
            rows.append({"month": month, "mean": None, "std": None, "n_cells": 0})
            continue
        finite = [p for p in parts if np.isfinite(p[0])]
        if not finite:
            rows.append({"month": month, "mean": None, "std": None, "n_cells": 0})
            continue
        rows.append({
            "month": month,
            "mean": float(np.mean([p[0] for p in finite])),
            "std": float(np.mean([p[1] for p in finite])),
            "n_cells": int(np.mean([p[2] for p in finite])),
        })
    return rows


def _ppe_annual(dataset: str, var: str, ring, surface: str, ens: Optional[int]):
    da = D._select_time(D._get_ppe_field(dataset, var), list(range(1, 13)))
    tpl = da.isel({d: 0 for d in da.dims if d in ("ens", "time")}, drop=True)
    weights = np.asarray(D.region_weights_polygon(tpl, ring, surface).values, dtype=np.float64)
    months = pd.to_datetime(da.time.values).month.to_numpy()
    if "ens" not in da.dims:
        field = np.asarray(da.transpose("time", "lat", "lon").values, dtype=np.float32)
    elif ens is not None:
        field = np.asarray(
            da.sel(ens=int(ens)).transpose("time", "lat", "lon").values, dtype=np.float32,
        )
    else:
        nens = int(da.sizes["ens"])
        acc = np.zeros((da.sizes["time"],) + weights.shape, dtype=np.float64)
        count = np.zeros_like(acc)
        for start in range(0, nens, 32):
            block = np.asarray(
                da.isel(ens=slice(start, start + 32)).transpose("ens", "time", "lat", "lon").values,
                dtype=np.float32,
            )
            finite = np.isfinite(block)
            acc += np.where(finite, block, 0.0).sum(axis=0)
            count += finite.sum(axis=0)
        field = np.full(acc.shape, np.nan, dtype=np.float32)
        ok = count > 0
        field[ok] = (acc[ok] / count[ok]).astype(np.float32)
    means, stds, counts = [], [], []
    for i in range(field.shape[0]):
        mean, std, n = spatial_mean_std(field[i], weights)
        means.append(mean)
        stds.append(std)
        counts.append(n)
    return _by_calendar_month(months, means, stds, counts)


def _obs_annual(dataset: str, var: str, ring, surface: str):
    from dashboard import obsmap
    import xarray as xr
    from dashboard.grid import LAT, LON
    data = obsmap.cube(dataset)
    key = obsmap.VAR_KEY[var]
    rows_m, rows_mean, rows_std, rows_n = [], [], [], []
    # One representative grid for the weights.
    sample = np.asarray(data[key][0], dtype=float)
    da = xr.DataArray(sample, coords={"lat": LAT, "lon": LON}, dims=("lat", "lon"))
    weights = np.asarray(D.region_weights_polygon(da, ring, surface).values, dtype=np.float64)
    for i, month in enumerate(data["month"]):
        grid = np.asarray(data[key][i], dtype=float)
        mean, std, n = spatial_mean_std(grid, weights)
        rows_m.append(int(month))
        rows_mean.append(mean)
        rows_std.append(std)
        rows_n.append(n)
    return _by_calendar_month(rows_m, rows_mean, rows_std, rows_n)


def _icon_annual(var: str, ring, surface: str, ens: Optional[int]):
    import icon_regional as ir
    from dashboard import iconmap
    member = 0 if ens is None else int(ens)
    weights = iconmap._cell_weights(ring, surface)
    if var == "SSA":
        od, time = ir._open_field(iconmap._root(), f"{member:03d}", ir.ICON_VARS["od550aer"])
        ab, _ = ir._open_field(iconmap._root(), f"{member:03d}", ir.ICON_VARS["abs550aer"])
        with np.errstate(divide="ignore", invalid="ignore"):
            data = 1.0 - ab / od
        data = np.where(np.isfinite(od) & (od > 0), data, np.nan)
    elif var == "AE_440_670":
        data, time = ir._open_field(iconmap._root(), f"{member:03d}", ir.ICON_VARS["ANG_440nm_670nm"])
    elif var == "AE_550_865":
        data, time = ir._open_field(iconmap._root(), f"{member:03d}", ir.ICON_VARS["ANG_550nm_865nm"])
    else:
        folder = ir.ICON_VARS.get(var)
        if folder is None:
            raise KeyError(var)
        opened = ir._open_field(iconmap._root(), f"{member:03d}", folder)
        if opened is None:
            raise KeyError(f"{var} missing for ICON member {member}")
        data, time = opened
    months = [int(t) // 100 % 100 for t in time]
    means, stds, counts = [], [], []
    for i in range(data.shape[0]):
        mean, std, n = spatial_mean_std(data[i], weights)
        means.append(mean)
        stds.append(std)
        counts.append(n)
    return _by_calendar_month(months, means, stds, counts)


def annual_cycle(dataset: str, var: str, ring, surface_type: str = "all", ens: Optional[int] = None) -> dict:
    surface = surface_type or "all"
    if dataset in ("pd2024", "pd2010", "aerocom"):
        if dataset == "aerocom":
            raise ValueError("pick one AeroCom model on the map; the annual cycle uses a PPE or an observation")
        rows = _ppe_annual(dataset, var, ring, surface, ens)
        note = "Spatial mean and spatial standard deviation of the ensemble-mean map" if ens is None else f"Member {ens}"
    elif dataset in ("spexone", "polder"):
        rows = _obs_annual(dataset, var, ring, surface)
        note = "Spatial mean and spatial standard deviation of the 1° monthly map"
    elif dataset == "icon2025":
        rows = _icon_annual(var, ring, surface, ens)
        note = f"ICON member {0 if ens is None else int(ens)}. Spatial spread on the native grid."
    else:
        raise ValueError(f"unknown dataset {dataset}")
    return {
        "dataset": dataset,
        "var": var,
        "label": D.VAR_LABELS.get(var, var),
        "surface_type": surface,
        "ens": ens,
        "note": note,
        "months": rows,
    }
