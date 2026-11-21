"""Register a drawn polygon in the main region list and regional tables.

Geometry is appended to ``Data/PPE_regional/extra_regions.csv``, which
``regions.load_extra_regions`` reads into ``REGIONS``. Monthly and seasonal
rows already produced by the box job are appended to the pipeline parquet
files (PPE and ICON kept separate). Re-registering the same name replaces
that region's rows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import polars as pl

import PPE_set as cfg
import regions as reg
from dashboard.nc_reader import polygon_bounds
from dashboard.polygons import monthly_path, read_polygon, seasonal_path, slugify

CSV_COLUMNS = (
    "name", "surface_type", "lon0", "lon1", "lat0", "lat1",
    "season_months", "species", "role", "color", "note", "ring",
)


def _lon_pair(west: float, east: float) -> tuple:
    """0..360 pair. west>east in that frame means the box crosses 0°E."""
    return (west % 360.0, east % 360.0)


def _read_csv_rows() -> list:
    path = reg.EXTRA_REGIONS_CSV
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(rows: list) -> None:
    path = reg.EXTRA_REGIONS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def _merge_table(target: Path, incoming: pl.DataFrame, region: str) -> int:
    if incoming.is_empty():
        return 0
    if target.is_file():
        old = pl.read_parquet(target)
        if "region" in old.columns:
            old = old.filter(pl.col("region") != region)
        out = pl.concat([old, incoming], how="diagonal_relaxed")
    else:
        out = incoming
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    out.write_parquet(tmp)
    tmp.replace(target)
    return incoming.height


def register_polygon(
    slug: str,
    species: Sequence[str] = ("du", "ss", "bb", "total", "marine"),
    season_months: Sequence[int] | None = None,
) -> dict:
    spec = read_polygon(slug)
    name = spec["name"].strip()
    slug = slugify(name)
    monthly = monthly_path(slug)
    seasonal = seasonal_path(slug)
    if not monthly.is_file() or not seasonal.is_file():
        raise FileNotFoundError(
            f"aggregate tables for {name!r} are not ready ({monthly.name})"
        )
    bounds = polygon_bounds(spec["ring"])
    lon0, lon1 = _lon_pair(bounds["lon_min"], bounds["lon_max"])
    months = list(season_months) if season_months else list(range(1, 13))
    species = tuple(s for s in species if s) or ("du", "ss", "bb", "total", "marine")
    row = {
        "name": name,
        "surface_type": spec.get("surface_type") or "all",
        "lon0": f"{lon0:.6f}",
        "lon1": f"{lon1:.6f}",
        "lat0": f"{bounds['lat_min']:.6f}",
        "lat1": f"{bounds['lat_max']:.6f}",
        "season_months": ",".join(str(int(m)) for m in months),
        "species": ",".join(species),
        "role": "source",
        "color": "#ff00aa",
        "note": "drawn polygon; table values use the polygon mask, lon/lat is the bounding box",
        "ring": json.dumps(spec["ring"]),
    }
    rows = [r for r in _read_csv_rows() if r.get("name") != name]
    rows.append(row)
    _write_csv_rows(rows)
    reg.load_extra_regions()

    monthly_df = pl.read_parquet(monthly).with_columns(pl.lit(name).alias("region"))
    seasonal_df = pl.read_parquet(seasonal).with_columns(pl.lit(name).alias("region"))
    icon_m = cfg.REGIONAL_DIR / "icon2025_regional_monthly.parquet"
    icon_s = cfg.REGIONAL_DIR / "icon2025_regional_seasonal.parquet"
    written = {}
    for label, frame, ppe_path, icon_path in (
        ("monthly", monthly_df, cfg.REGIONAL_MONTHLY, icon_m),
        ("seasonal", seasonal_df, cfg.REGIONAL_SEASONAL, icon_s),
    ):
        if "ppe_id" in frame.columns:
            ppe = frame.filter(pl.col("ppe_id") != "icon2025")
            icon = frame.filter(pl.col("ppe_id") == "icon2025")
        else:
            ppe, icon = frame, frame.clear()
        written[label] = {
            "ppe_rows": _merge_table(ppe_path, ppe, name),
            "icon_rows": _merge_table(icon_path, icon, name),
            "ppe_path": str(ppe_path),
            "icon_path": str(icon_path),
        }
    return {
        "name": name,
        "slug": slug,
        "csv": str(reg.EXTRA_REGIONS_CSV),
        "lon_range": [lon0, lon1],
        "lat_range": [bounds["lat_min"], bounds["lat_max"]],
        "species": list(species),
        "tables": written,
    }
