#!/usr/bin/env python3
"""SPEXone_regrid_model.py

Regrid one month of SPEXone L2 parquet to a regular design grid.
Optionally map a model to the same bins (MAP_MODEL=True).

Usage (for SLURM array jobs — month from argv or SLURM_ARRAY_TASK_ID)::

    python py/SPEXone_regrid_model.py 3 2025
    python py/SPEXone_regrid_model.py $SLURM_ARRAY_TASK_ID 2025

Temporal averaging (TIME_FREQ, default ``'3H'``):
  df['date'] = df['date'].dt.round('3H')   # kept as output column ``date``
  Also supports 'H', '6H', 'D', 'MS'/'M', or None (no temporal binning).

Uncertainty:
  regrid_err_*      = within-bin spread (std or half-IQR; controlled by SPREAD).
  *_uncertainty_*   = mean of per-pixel source uncertainties (fully correlated).

USE_AAOD=True  → AAOD = (1-SSA)*AOT with propagated uncertainty.
USE_AAOD=False → keep SSA columns.
MAP_MODEL=True → nearest-neighbour model values at bin centres.
"""

import glob
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# SETTINGS
# =============================================================================
project_root = Path("/scistor/guest/gbb083/AeroCom")
SPEXONE_DIR = project_root / "Data" / "SPEXone"
OUT_DIR = project_root / "Data" / "SPEXone_regridded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_RES_DEG = 1.0          # degrees; None to skip spatial binning
# Temporal averaging of L2 times into `date` (kept in output):
#   '3H' (default) → df['date'].dt.round('3H')  — same as interpolate_model_to_spexone
#   'H','6H','D'   → dt.round(...) for fixed offsets; 'MS'/'M' → monthly period start/end
#   None           → no temporal aggregation (each L2 timestamp kept)
TIME_FREQ = "3H"
MIN_POINTS = 2
AGG = "mean"                # 'median' or 'mean'
SPREAD = "std"              # 'iqr' or 'std'

ssa_uncertainty_threshold = 0.7  # filter out pixels with ssa_uncertainty > threshold
aot_uncertainty_threshold = 1.0  # filter out pixels with aot_uncertainty > threshold
# Wavelengths as integers (columns use e.g. aot_550nm)
WAVELENGTHS = [340, 355, 380, 440, 500, 532, 550, 670, 865]

USE_AAOD = True             # True → AAOD; False → keep SSA
MAP_MODEL = False           # True → interpolate model to bins
MODEL_PATH = ""
MODEL_VAR_MAP = {"aod550": "aod550"}
MODEL_FILE_GLOB = "{var}_*_{date_str}.nc"
MODEL_TIME_ROUND = "3H"

# =============================================================================
# Month / year from argv (SLURM-friendly)
# =============================================================================
# Prefer argv; fall back to SLURM_ARRAY_TASK_ID for month.
if len(sys.argv) >= 3:
    month = int(sys.argv[1])
    year = str(sys.argv[2])
elif len(sys.argv) == 2:
    month = int(sys.argv[1])
    year = os.environ.get("SPEX_YEAR", "2025")
elif "SLURM_ARRAY_TASK_ID" in os.environ:
    month = int(os.environ["SLURM_ARRAY_TASK_ID"])
    year = os.environ.get("SPEX_YEAR", "2025")
else:
    print("Usage: python SPEXone_regrid_model.py <month> <year>")
    print("   or: SPEX_YEAR=2025 sbatch --array=1-12 ... (uses SLURM_ARRAY_TASK_ID)")
    sys.exit(1)

if not (1 <= month <= 12):
    print(f"Error: month must be 1–12, got {month}")
    sys.exit(1)

spex_files = sorted(
    glob.glob(str(SPEXONE_DIR / f"Extract*L2*{year}_{month:02d}*v4.0.parquet"))
)
output_parquet = str(OUT_DIR / f"SPEXone_L2_regridded_{year}_{month:02d}_v4_0.parquet")

print(f"Year={year} month={month:02d}")
print(f"Input files ({len(spex_files)}): {spex_files}")
print(f"Output: {output_parquet}")
if not spex_files:
    print("No SPEXone files for this month — exiting.")
    sys.exit(0)

# =============================================================================
# Helpers
# =============================================================================
wvl_tags = [f"{w}nm" for w in WAVELENGTHS]


def _bin_center(series, res):
    return (np.floor(series / res) * res + res / 2.0).astype("float32")


def _round_date(dt_series, freq):
    """Temporal average stamp kept as column ``date``.

    Default ``'3H'`` / ``'3h'`` matches interpolate_model_to_spexone:
        df_obs['date'].dt.round('3h')
    """
    if not freq:
        return dt_series
    TF = str(freq).strip()
    # Pandas ≥2 prefers lowercase hour unit ('3h'); accept '3H' from settings
    TF_u = TF.upper()
    tz = dt_series.dt.tz
    if TF_u in ("M", "MS"):
        out = dt_series.dt.to_period("M").dt.to_timestamp(
            how="start" if TF_u == "MS" else "end"
        )
        if tz is not None and out.dt.tz is None:
            out = out.dt.tz_localize(tz)
    else:
        # Fixed offsets: round to nearest bin centre (3h default)
        round_freq = TF_u.replace("H", "h") if "H" in TF_u else TF
        # Day etc. stay uppercase-compatible via to_offset
        if round_freq.endswith("D") or round_freq == "D":
            round_freq = round_freq  # 'D' still valid
        out = dt_series.dt.round(round_freq)
        if tz is not None and out.dt.tz is None:
            out = out.dt.tz_localize(tz)
        elif tz is not None and out.dt.tz != tz:
            out = out.dt.tz_convert(tz)
    return out


def _half_iqr(x):
    if len(x) == 0:
        return np.nan
    q25, q75 = np.nanpercentile(x.to_numpy(), [25, 75])
    return float(0.5 * (q75 - q25))


def aggregate_uncertainty_correlated(u):
    """Mean of per-pixel uncertainties (fully correlated → no √N reduction)."""
    v = u.to_numpy(dtype=float)
    finite = v[np.isfinite(v)]
    return float(np.mean(finite)) if len(finite) else np.nan


def propagate_aaod_uncertainty(aot, ssa, sig_aot, sig_ssa):
    """σ_AAOD = sqrt(((1-SSA)·σ_AOT)² + (AOT·σ_SSA)²)."""
    return np.sqrt(((1.0 - ssa) * sig_aot) ** 2 + (aot * sig_ssa) ** 2)


def regrid_spexone(df):
    """Regrid one SPEXone DataFrame to (lon_bin, lat_bin, date).

    ``date`` is the temporally rounded L2 time (default: round to 3H).
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["date"] = _round_date(df["date"], TIME_FREQ)
    
    aot_u = [f"aot_uncertainty_{w}" for w in wvl_tags if f"aot_uncertainty_{w}" in df.columns]
    ssa_u = [f"ssa_uncertainty_{w}" for w in wvl_tags if f"ssa_uncertainty_{w}" in df.columns]
    if aot_u:
        df = df[df[aot_u].le(aot_uncertainty_threshold).all(axis=1)]
    if ssa_u:
        df = df[df[ssa_u].le(ssa_uncertainty_threshold).all(axis=1)]

    if GRID_RES_DEG and GRID_RES_DEG > 0:
        df["lon_bin"] = _bin_center(df["lon"], GRID_RES_DEG)
        df["lat_bin"] = _bin_center(df["lat"], GRID_RES_DEG)
    else:
        df["lon_bin"] = df["lon"].astype("float32")
        df["lat_bin"] = df["lat"].astype("float32")

    center_fun = np.nanmedian if AGG.lower() == "median" else np.nanmean
    if SPREAD.lower() == "iqr":
        spread_fun = _half_iqr
    elif SPREAD.lower() == "std":
        spread_fun = lambda s: float(np.nanstd(s.to_numpy()))
    else:
        raise ValueError(f"Unknown SPREAD={SPREAD!r}")

    if USE_AAOD:
        for wvl in wvl_tags:
            need = [f"aot_{wvl}", f"ssa_{wvl}", f"aot_uncertainty_{wvl}", f"ssa_uncertainty_{wvl}"]
            if any(c not in df.columns for c in need):
                continue
            aot = df[f"aot_{wvl}"].to_numpy(dtype=float)
            ssa = df[f"ssa_{wvl}"].to_numpy(dtype=float)
            sig_aot = df[f"aot_uncertainty_{wvl}"].to_numpy(dtype=float)
            sig_ssa = df[f"ssa_uncertainty_{wvl}"].to_numpy(dtype=float)
            df[f"aaod_{wvl}"] = ((1.0 - ssa) * aot).astype("float32")
            df[f"_aaod_uncertainty_{wvl}"] = propagate_aaod_uncertainty(
                aot, ssa, sig_aot, sig_ssa
            ).astype("float32")

    # Spatial + temporal bins; keep rounded stamp as `date`
    gcols = ["lon_bin", "lat_bin", "date"]
    agg_spec = {"n": ("lon_bin", "size")}

    for wvl in wvl_tags:
        aot_col = f"aot_{wvl}"
        if aot_col not in df.columns:
            continue
        agg_spec[f"aod_{wvl}"] = (aot_col, center_fun)
        agg_spec[f"spread_aod_{wvl}"] = (aot_col, spread_fun)
        agg_spec[f"regrid_err_aod_{wvl}"] = (aot_col, spread_fun)
        sig_aot = f"aot_uncertainty_{wvl}"
        if sig_aot in df.columns:
            agg_spec[f"aod_uncertainty_{wvl}"] = (sig_aot, aggregate_uncertainty_correlated)

        if USE_AAOD:
            if f"aaod_{wvl}" in df.columns:
                agg_spec[f"aaod_{wvl}"] = (f"aaod_{wvl}", center_fun)
                agg_spec[f"spread_aaod_{wvl}"] = (f"aaod_{wvl}", spread_fun)
                agg_spec[f"regrid_err_aaod_{wvl}"] = (f"aaod_{wvl}", spread_fun)
            if f"_aaod_uncertainty_{wvl}" in df.columns:
                agg_spec[f"aaod_uncertainty_{wvl}"] = (
                    f"_aaod_uncertainty_{wvl}", aggregate_uncertainty_correlated
                )
        else:
            ssa_col = f"ssa_{wvl}"
            if ssa_col in df.columns:
                agg_spec[f"ssa_{wvl}"] = (ssa_col, center_fun)
                agg_spec[f"spread_ssa_{wvl}"] = (ssa_col, spread_fun)
                agg_spec[f"regrid_err_ssa_{wvl}"] = (ssa_col, spread_fun)
            sig_ssa = f"ssa_uncertainty_{wvl}"
            if sig_ssa in df.columns:
                agg_spec[f"ssa_uncertainty_{wvl}"] = (sig_ssa, aggregate_uncertainty_correlated)

    out = df.groupby(gcols, dropna=True, observed=True).agg(**agg_spec).reset_index()
    out = out[out["n"] >= MIN_POINTS].copy()
    return out.reset_index(drop=True)


def map_model_to_bins(binned):
    """Nearest-neighbour model values at bin centres (optional)."""
    import xarray as xr

    out = binned.copy()
    for suffix in MODEL_VAR_MAP.values():
        out[f"model_{suffix}"] = np.nan

    # `date` already rounded (default 3H); optional extra round for model clock
    unique_dates = pd.to_datetime(out["date"], utc=True)
    if MODEL_TIME_ROUND:
        unique_dates = unique_dates.dt.round(MODEL_TIME_ROUND)

    for udate in unique_dates.unique():
        date_str = pd.Timestamp(udate).strftime("%Y%m%d")
        mask = unique_dates == udate
        for var_name, suffix in MODEL_VAR_MAP.items():
            tmpl = MODEL_FILE_GLOB.format(var=var_name, date_str=date_str)
            matches = glob.glob(os.path.join(MODEL_PATH, "**", tmpl), recursive=True)
            if len(matches) != 1:
                print(f"  Missing/multiple model file: {tmpl}")
                continue
            try:
                ds = xr.open_dataset(matches[0], engine="netcdf4")
                lat_name = next((c for c in ("lat", "latitude") if c in ds.coords), None)
                lon_name = next((c for c in ("lon", "longitude") if c in ds.coords), None)
                dv = var_name if var_name in ds else list(ds.data_vars)[0]
                da = ds[dv]
                for tdim in ("time", "valid_time"):
                    if tdim in da.dims:
                        da = da.isel({tdim: 0})
                lat_grid = da[lat_name].values.astype(float)
                lon_grid = da[lon_name].values.astype(float)
                vals = da.values.astype(float)
                ds.close()
            except Exception as e:
                print(f"  Model load error {matches[0]}: {e}")
                continue

            lats = out.loc[mask, "lat_bin"].to_numpy(dtype=float)
            lons = out.loc[mask, "lon_bin"].to_numpy(dtype=float)
            lat_idx = np.argmin(np.abs(lat_grid[:, None] - lats[None, :]), axis=0)
            lon_idx = np.argmin(np.abs(lon_grid[:, None] - lons[None, :]), axis=0)
            if vals.ndim == 2 and vals.shape[0] == len(lat_grid):
                interped = vals[lat_idx, lon_idx]
            elif vals.ndim == 2 and vals.shape[1] == len(lat_grid):
                interped = vals[lon_idx, lat_idx]
            else:
                interped = np.full(len(lats), np.nan)
            out.loc[mask, f"model_{suffix}"] = interped.astype("float32")
    return out


# =============================================================================
# MAIN (one month)
# =============================================================================
if os.path.exists(output_parquet):
    os.remove(output_parquet)

pqwriter = None
total_in = 0
total_out = 0

for spex_file in spex_files:
    print(f"Reading {Path(spex_file).name} ...", flush=True)
    df_obs = pd.read_parquet(spex_file)
    total_in += len(df_obs)
    print(f"  {len(df_obs):,} rows")

    df_binned = regrid_spexone(df_obs)
    if df_binned.empty:
        print("  → no bins; skipping")
        continue

    if MAP_MODEL:
        if not MODEL_PATH:
            print("  MAP_MODEL=True but MODEL_PATH empty — skip model")
        else:
            df_binned = map_model_to_bins(df_binned)

    # Keep `date` column (rounded temporal stamp), ns-UTC for consistent parquet schema
    if "date" in df_binned.columns:
        df_binned["date"] = (
            pd.to_datetime(df_binned["date"], utc=True)
            .dt.tz_convert("UTC")
            .astype("datetime64[ns, UTC]")
        )

    total_out += len(df_binned)
    table = pa.Table.from_pandas(df_binned, preserve_index=False)
    if pqwriter is None:
        pqwriter = pq.ParquetWriter(
            output_parquet, table.schema, use_dictionary=True, compression="snappy"
        )
    pqwriter.write_table(table)
    print(
        f"  → {len(df_binned):,} bins "
        f"({'AAOD' if USE_AAOD else 'SSA'}, res={GRID_RES_DEG}°, "
        f"time={TIME_FREQ or 'none'})"
    )

if pqwriter is not None:
    pqwriter.close()
    print(f"Wrote {total_out:,} bins from {total_in:,} pixels → {output_parquet}")
else:
    print("No output written.")
