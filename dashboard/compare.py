"""Month-by-month comparison of every model that has a regional value.

PPE and hjia values are the median of all members and all years in that
calendar month. AeroCom is one point per model from the seasonal table
(there is no monthly AeroCom regional file). Observations are POLDER for
the 2010 comparison and SPEXone for 2024 onwards.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import polars as pl

import PPE_set as cfg
import ppe_io
import regions as reg


def _frames() -> pl.DataFrame:
    parts = []
    if cfg.REGIONAL_MONTHLY.is_file():
        parts.append(pl.read_parquet(cfg.REGIONAL_MONTHLY))
    icon = cfg.REGIONAL_DIR / "icon2025_regional_monthly.parquet"
    if icon.is_file():
        parts.append(pl.read_parquet(icon))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def _column(df: pl.DataFrame, var: str) -> Optional[str]:
    if var in df.columns:
        return var
    aliases = {
        "AE": ("AE_550_865", "AE_440_550", "AE"),
        "AE_440_550": ("AE_440_550", "AE"),
        "AE_550_865": ("AE_550_865", "AE"),
        "SSA": ("SSA",),
    }
    for name in aliases.get(var, ()):
        if name in df.columns:
            return name
    return None


def _model_rows(region: str, var: str, month: int) -> List[dict]:
    df = _frames()
    if df.is_empty() or "region" not in df.columns or region not in df["region"].unique().to_list():
        return []
    col = _column(df, var)
    if col is None:
        return []
    work = df.filter(pl.col("region") == region)
    if "time" in work.columns:
        work = work.with_columns(pl.col("time").dt.month().alias("_month"))
        work = work.filter(pl.col("_month") == int(month))
    rows = []
    if work.is_empty():
        return rows
    for key, group in work.group_by("ppe_id"):
        pid = key[0] if isinstance(key, tuple) else key
        values = group[col].drop_nulls().to_numpy()
        values = values[np.isfinite(values.astype(float))]
        if values.size == 0:
            continue
        try:
            label = ppe_io.ppe_style(str(pid))["label"]
        except Exception:
            label = str(pid)
        rows.append({
            "id": str(pid),
            "label": label,
            "value": float(np.median(values)),
            "n": int(values.size),
            "group": "2024" if str(pid) in ("pd2024", "icon2025") else "2010",
            "note": "monthly median, all years",
        })
    return rows


def _aerocom_rows(region: str, var: str) -> List[dict]:
    from dashboard.aggplots import load_aerocom
    frame = None
    for species in ("ss", "du", "bb", "total"):
        candidate, _note = load_aerocom(species, [region])
        if candidate is None or candidate.empty:
            continue
        if var in candidate.columns or (var == "AE" and "AE" in candidate.columns):
            frame = candidate
            break
    if frame is None or frame.empty:
        return []
    col = var if var in frame.columns else None
    if col is None and var == "AE":
        col = "AE" if "AE" in frame.columns else ("AE_550_865" if "AE_550_865" in frame.columns else None)
    if col is None or "model" not in frame.columns:
        return []
    rows = []
    sub = frame.loc[frame["region"] == region]
    for _, item in sub.iterrows():
        value = item.get(col)
        if value is None or not np.isfinite(float(value)):
            continue
        rows.append({
            "id": str(item["model"]),
            "label": str(item["model"]),
            "value": float(value),
            "n": 1,
            "group": "2010",
            "note": "AeroCom seasonal",
        })
    return rows


def _obs(region: str, var: str, month: int, ring, surface: str) -> List[dict]:
    from dashboard import obsmap
    if ring is None and region in reg.REGIONS:
        from dashboard.data import region_ring
        ring = region_ring(reg.REGIONS[region])
    if not ring:
        return []
    obs_var = {"AE": "AE_440_550", "AE_550_865": "AE_550_865", "AE_440_550": "AE_440_550"}.get(var, var)
    if obs_var not in obsmap.VAR_KEY and obs_var not in ("od550aer", "SSA"):
        if obs_var not in obsmap.VAR_KEY:
            return []
    rows = []
    for dataset, label, group in (
        ("polder", "POLDER 2010", "2010"),
        ("spexone", "SPEXone 2024+", "2024"),
    ):
        try:
            value = obsmap.region_value(dataset, obs_var if obs_var in obsmap.VAR_KEY else var,
                                        ring, [int(month)], surface)
        except Exception:
            continue
        if value is None or not np.isfinite(value):
            continue
        rows.append({
            "id": dataset, "label": label, "value": float(value), "n": 1,
            "group": group, "note": "observation",
        })
    return rows


def compare_month(var: str, month: int, region: str = "", ring=None, surface_type: str = "all") -> dict:
    month = int(month)
    rows = []
    if region:
        rows.extend(_model_rows(region, var, month))
        rows.extend(_aerocom_rows(region, var))
    rows.extend(_obs(region, var, month, ring, surface_type))
    if not rows:
        raise ValueError("no model or observation value for this month and region")
    return {
        "var": var,
        "month": month,
        "where": region or "drawn polygon",
        "rows": rows,
    }


def validation(region: str, var: str, month: int) -> dict:
    payload = compare_month(var, month, region=region)
    return {
        "region": region,
        "var": var,
        "month": int(month),
        "y2010": [r for r in payload["rows"] if r["group"] == "2010"],
        "y2024": [r for r in payload["rows"] if r["group"] == "2024"],
    }
