"""Full regional aggregate for one drawn polygon.

ECHAM uses netCDF4 chunked reads into the same ``regional_columns`` path as
the seasonal tables. ICON walks members one file at a time on the native grid.
Results land in ``dashboard/cache/polygons`` as Polars-written parquet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

import ppe_io
from regional_aggregates import ens_label, regional_columns

from dashboard import data as D
from dashboard import nc_reader
from dashboard.polygons import monthly_path, polygon_path, read_polygon, seasonal_path, status_path

ECHAM = ("pd2024", "pd2010")


def _write_status(slug: str, **fields) -> None:
    path = status_path(slug)
    current = {}
    if path.is_file():
        current = json.loads(path.read_text())
    current.update(fields)
    path.write_text(json.dumps(current, indent=2))


def _frame_from_columns(ppe_id: str, ens, times, columns: dict, region: str) -> pd.DataFrame:
    ens = np.asarray(ens).astype(int)
    times = pd.to_datetime(times)
    n_e, n_t = len(ens), len(times)
    e_i = np.repeat(np.arange(n_e), n_t)
    t_i = np.tile(np.arange(n_t), n_e)
    df = pd.DataFrame({
        "ppe_id": ppe_id,
        "ens": ens[e_i],
        "model": [ens_label(int(e)) for e in ens[e_i]],
        "experiment": ppe_id,
        "region": region,
        "time": times[t_i],
    })
    for name, arr in columns.items():
        df[name] = np.asarray(arr)[:, :, 0].reshape(-1).astype(np.float32)
    return df


def aggregate_echam(ppe_id: str, spec: dict) -> pd.DataFrame:
    ds = nc_reader.open_available(ppe_id)
    try:
        template = ds["od550aer"].isel(ens=0, time=0, drop=True)
        weights = D.region_weights_polygon(template, spec["ring"], spec["surface_type"])
        weights = weights.expand_dims(region=[spec["name"]])
        columns = regional_columns(ds, weights, quality="ppe")
        return _frame_from_columns(
            ppe_id, ds["ens"].values, ds["time"].values, columns, spec["name"],
        )
    finally:
        ds.close()


def aggregate_icon(spec: dict) -> pd.DataFrame:
    from dashboard import iconmap
    import icon_regional as ir

    root = iconmap._root()
    members = iconmap.member_ids()
    weights = iconmap._cell_weights(spec["ring"], spec["surface_type"])
    icon_vars = [name for name in ir.EXTENSIVE if name in ir.ICON_VARS]
    sample = ir._open_field(root, f"{members[0]:03d}", "TAU_2D_550nm")
    if sample is None:
        raise FileNotFoundError("ICON optical sample missing")
    _, time = sample
    months = [int(t) // 100 % 100 for t in time]
    when = pd.to_datetime([f"{int(t):08d}" for t in time])
    rows = []
    for member in members:
        row = {
            "ppe_id": "icon2025",
            "ens": member,
            "model": str(member),
            "experiment": "icon2025",
            "region": spec["name"],
        }
        series = {}
        for var in icon_vars:
            folder = ir.ICON_VARS[var]
            opened = ir._open_field(root, f"{member:03d}", folder)
            if opened is None:
                series[var] = [np.nan] * len(when)
                continue
            data, _ = opened
            values = []
            for i in range(data.shape[0]):
                mean, _std, _n = _wmean(data[i], weights)
                values.append(mean)
            series[var] = values
        for i, stamp in enumerate(when):
            item = dict(row)
            item["time"] = stamp
            item["month"] = months[i]
            for var, values in series.items():
                item[var] = values[i]
            rows.append(item)
    return pd.DataFrame(rows)


def _wmean(values, weights) -> tuple:
    from dashboard.annual import spatial_mean_std
    return spatial_mean_std(values, weights)


def _seasonal(monthly: pd.DataFrame) -> pd.DataFrame:
    work = monthly.copy()
    work["time"] = pd.to_datetime(work["time"])
    value_cols = [c for c in work.columns if c not in
                  ("ppe_id", "ens", "model", "experiment", "region", "time", "month")]
    grouped = work.groupby(["ppe_id", "ens", "model", "experiment", "region"], sort=False)
    out = grouped[value_cols].mean(numeric_only=True)
    out["n_months"] = grouped.size()
    return out.reset_index()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    work = df.copy()
    if "time" in work.columns:
        work["time"] = pd.to_datetime(work["time"]).astype("datetime64[ms]")
    pl.from_pandas(work).write_parquet(path)


def run(slug: str, ppe_ids: Sequence[str] = ("pd2024", "pd2010", "icon2025")) -> Path:
    spec = read_polygon(slug)
    bounds = nc_reader.polygon_bounds(spec["ring"])
    monthly_out = monthly_path(slug)
    seasonal_out = seasonal_path(slug)
    _write_status(
        slug, state="running", message="starting", ppe_ids=list(ppe_ids),
        lon_min=bounds["lon_min"], lon_max=bounds["lon_max"],
        lat_min=bounds["lat_min"], lat_max=bounds["lat_max"],
        monthly=str(monthly_out), seasonal=str(seasonal_out),
    )
    print(
        f"polygon lon [{bounds['lon_min']:.3f}, {bounds['lon_max']:.3f}]  "
        f"lat [{bounds['lat_min']:.3f}, {bounds['lat_max']:.3f}]"
    )
    print(f"monthly -> {monthly_out}")
    print(f"seasonal -> {seasonal_out}")
    frames = []
    for ppe_id in ppe_ids:
        _write_status(slug, state="running", message=f"aggregating {ppe_id}")
        try:
            if ppe_id == "icon2025":
                frames.append(aggregate_icon(spec))
            elif ppe_id in ECHAM:
                frames.append(aggregate_echam(ppe_id, spec))
            else:
                raise KeyError(ppe_id)
        except Exception as exc:
            _write_status(slug, state="running", message=f"{ppe_id} failed: {exc}")
            print(f"{ppe_id} failed: {exc}")
            continue
    if not frames:
        _write_status(slug, state="failed", message="no ensemble produced a table")
        raise RuntimeError("no ensemble produced a table")
    monthly = pd.concat(frames, ignore_index=True)
    seasonal = ppe_io.add_budget_columns(_seasonal(monthly))
    monthly = ppe_io.add_budget_columns(monthly)
    _write_parquet(monthly, monthly_out)
    _write_parquet(seasonal, seasonal_out)
    _write_status(
        slug, state="done", message="monthly and seasonal tables ready",
        monthly=str(monthly_out), seasonal=str(seasonal_out),
        lon_min=bounds["lon_min"], lon_max=bounds["lon_max"],
        lat_min=bounds["lat_min"], lat_max=bounds["lat_max"],
        n_rows_monthly=int(len(monthly)), n_rows=int(len(seasonal)),
    )
    print(f"wrote {monthly_out} ({len(monthly)} rows)")
    print(f"wrote {seasonal_out} ({len(seasonal)} rows)")
    return seasonal_out


if __name__ == "__main__":
    import sys
    slug = sys.argv[1]
    try:
        run(slug)
    except Exception as exc:
        _write_status(slug, state="failed", message=str(exc))
        raise
