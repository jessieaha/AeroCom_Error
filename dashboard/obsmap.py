"""SPEXone and POLDER maps.

SPEXone supports regridded archives (resolution in the filename, or legacy
files without a token treated as 1°) and a native monthly overpass mode that
reads ``Data/SPEXone/Extract*L2*`` parquets for one calendar month.
POLDER-GRASP stays on the 2010 collocation file, reduced to 1°.
"""

from __future__ import annotations

import re
import threading
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import ppe_io
from dashboard import data as D
from dashboard.grid import LAT, LON, bin_indices

CACHE = Path(__file__).resolve().parent / "cache"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEX_REGRID_DIR = PROJECT_ROOT / "Data" / "SPEXone_regridded"
SPEX_NATIVE_DIR = PROJECT_ROOT / "Data" / "SPEXone"
VARS = ("od550aer", "abs550aer", "SSA", "AE_440_550", "AE_550_865")
VAR_KEY = {
    "od550aer": "aod",
    "abs550aer": "aaod",
    "SSA": "ssa",
    "AE_440_550": "ae440",
    "AE_550_865": "ae865",
}

# Filename patterns. Legacy files without a resolution token are 1°.
_REGRID_RES = re.compile(
    r"SPEXone_L2_regridded_(\d{4})_(\d{2})_(\d+(?:\.\d+)?)deg_v4_0\.parquet$"
)
_REGRID_LEGACY = re.compile(
    r"SPEXone_L2_regridded_(\d{4})_(\d{2})_v4_0\.parquet$"
)


def _empty():
    return {k: [] for k in ("aod", "aaod", "ssa", "ae440", "ae865", "year", "month")}


def _quality(aod, aaod):
    with np.errstate(divide="ignore", invalid="ignore"):
        ssa = 1.0 - aaod / aod
    ok = (
        np.isfinite(aod) & (aod > 0.01)
        & np.isfinite(aaod) & (aaod > 1e-5) & (aaod < 0.95 * aod)
        & np.isfinite(ssa) & (ssa >= 0.5) & (ssa <= 1.02)
    )
    aod = np.where(ok, aod, np.nan)
    aaod = np.where(ok, aaod, np.nan)
    ssa = np.where(ok, ssa, np.nan)
    return aod, aaod, ssa


def _ae(aod, other, wave):
    with np.errstate(divide="ignore", invalid="ignore"):
        ae = -np.log(aod / other) / np.log(550.0 / wave)
    return np.where(np.isfinite(aod) & np.isfinite(other) & (other > 0) & (aod > 0), ae, np.nan)


def _accumulate(store, year, month, lon, lat, aod, aaod, ae440, ae865):
    aod, aaod, ssa = _quality(np.asarray(aod, float), np.asarray(aaod, float))
    ae440 = np.where(np.isfinite(aod), np.asarray(ae440, float), np.nan)
    ae865 = np.where(np.isfinite(aod), np.asarray(ae865, float), np.nan)
    fields = {"aod": aod, "aaod": aaod, "ssa": ssa, "ae440": ae440, "ae865": ae865}
    finite = np.isfinite(lon) & np.isfinite(lat)
    iy, ix = bin_indices(lon[finite], lat[finite])
    packed = {}
    for key, values in fields.items():
        acc = np.zeros((LAT.size, LON.size))
        cnt = np.zeros((LAT.size, LON.size))
        v = values[finite]
        ok = np.isfinite(v)
        np.add.at(acc, (iy[ok], ix[ok]), v[ok])
        np.add.at(cnt, (iy[ok], ix[ok]), 1.0)
        grid = np.full(acc.shape, np.nan, dtype=np.float32)
        good = cnt > 0
        grid[good] = (acc[good] / cnt[good]).astype(np.float32)
        packed[key] = grid
    store["year"].append(int(year))
    store["month"].append(int(month))
    for key in ("aod", "aaod", "ssa", "ae440", "ae865"):
        store[key].append(packed[key])


def list_spex_resolutions() -> List[float]:
    """Resolutions available on disk (legacy unnamed files count as 1.0)."""
    found = set()
    if not SPEX_REGRID_DIR.is_dir():
        return []
    for path in SPEX_REGRID_DIR.glob("SPEXone_L2_regridded_*_v4_0.parquet"):
        hit = _REGRID_RES.search(path.name)
        if hit:
            found.add(float(hit.group(3)))
        elif _REGRID_LEGACY.search(path.name):
            found.add(1.0)
    return sorted(found)


def _spex_files_for_res(res: float) -> List[Path]:
    files = []
    if not SPEX_REGRID_DIR.is_dir():
        return files
    token = f"{res:g}"
    for path in sorted(SPEX_REGRID_DIR.glob("SPEXone_L2_regridded_*_v4_0.parquet")):
        hit = _REGRID_RES.search(path.name)
        if hit and abs(float(hit.group(3)) - res) < 1e-9:
            files.append(path)
        elif abs(res - 1.0) < 1e-9 and _REGRID_LEGACY.search(path.name):
            files.append(path)
    return files


def _build_spex(res: float = 1.0) -> dict:
    files = _spex_files_for_res(res)
    if not files:
        raise FileNotFoundError(f"no SPEXone regrid files at {res}° under {SPEX_REGRID_DIR}")
    store = _empty()
    for path in files:
        hit = _REGRID_RES.search(path.name) or _REGRID_LEGACY.search(path.name)
        year, month = int(hit.group(1)), int(hit.group(2))
        df = pd.read_parquet(
            path, columns=["lon_bin", "lat_bin", "aod_550nm", "aaod_550nm", "aod_440nm", "aod_865nm"],
        )
        aod = df["aod_550nm"].to_numpy()
        _accumulate(
            store, year, month,
            df["lon_bin"].to_numpy(), df["lat_bin"].to_numpy(),
            aod, df["aaod_550nm"].to_numpy(),
            _ae(aod, df["aod_440nm"].to_numpy(), 440.0),
            _ae(aod, df["aod_865nm"].to_numpy(), 865.0),
        )
    return _stack(store)


def _build_spex_native(year: int, month: int) -> dict:
    """One overpass month from native L2 Extract* parquets (not the 1° mean)."""
    if not SPEX_NATIVE_DIR.is_dir():
        raise FileNotFoundError(
            f"{SPEX_NATIVE_DIR} is empty or missing — place SPEXone L2 Extract* files there"
        )
    files = sorted(SPEX_NATIVE_DIR.glob(f"Extract*L2*{year}_{month:02d}*v4.0.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no native SPEXone L2 files for {year}-{month:02d} in {SPEX_NATIVE_DIR}"
        )
    store = _empty()
    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    lon = df["longitude"].to_numpy() if "longitude" in df.columns else df["lon"].to_numpy()
    lat = df["latitude"].to_numpy() if "latitude" in df.columns else df["lat"].to_numpy()
    aod_col = "aod_550nm" if "aod_550nm" in df.columns else "aot_550nm"
    aod = df[aod_col].to_numpy(dtype=float)
    if "aaod_550nm" in df.columns:
        aaod = df["aaod_550nm"].to_numpy(dtype=float)
    elif "ssa_550nm" in df.columns:
        aaod = aod * (1.0 - df["ssa_550nm"].to_numpy(dtype=float))
    else:
        aaod = np.full_like(aod, np.nan)
    a440 = df["aod_440nm"].to_numpy(dtype=float) if "aod_440nm" in df.columns else (
        df["aot_440nm"].to_numpy(dtype=float) if "aot_440nm" in df.columns else np.full_like(aod, np.nan)
    )
    a865 = df["aod_865nm"].to_numpy(dtype=float) if "aod_865nm" in df.columns else (
        df["aot_865nm"].to_numpy(dtype=float) if "aot_865nm" in df.columns else np.full_like(aod, np.nan)
    )
    _accumulate(store, year, month, lon, lat, aod, aaod, _ae(aod, a440, 440.0), _ae(aod, a865, 865.0))
    return _stack(store)


def _build_polder() -> dict:
    cols = ["time", "longitude", "latitude", "AOD_550", "AAOD_550", "AOD_440", "AOD_870"]
    df = pd.read_parquet(ppe_io.POLDER_PATH, columns=cols)
    when = pd.to_datetime(df["time"])
    store = _empty()
    for month, idx in df.groupby(when.dt.month).groups.items():
        sub = df.loc[idx]
        aod_m = sub["AOD_550"].to_numpy()
        _accumulate(
            store, 2010, int(month),
            sub["longitude"].to_numpy(), sub["latitude"].to_numpy(),
            aod_m, sub["AAOD_550"].to_numpy(),
            _ae(aod_m, sub["AOD_440"].to_numpy(), 440.0),
            _ae(aod_m, sub["AOD_870"].to_numpy(), 870.0),
        )
    return _stack(store)


def _stack(store: dict) -> dict:
    if not store["year"]:
        raise FileNotFoundError("no observation rows")
    order = np.lexsort((store["month"], store["year"]))
    out = {
        "year": np.asarray(store["year"], dtype=np.int16)[order],
        "month": np.asarray(store["month"], dtype=np.int16)[order],
    }
    for key in ("aod", "aaod", "ssa", "ae440", "ae865"):
        out[key] = np.stack(store[key])[order]
    return out


def _cache_path(dataset: str, res: float = 1.0, native: bool = False, year: int = 0, month: int = 0) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    if native:
        return CACHE / f"spexone_native_{year}_{month:02d}.npz"
    if dataset == "spexone":
        return CACHE / f"spexone_{res:g}deg.npz"
    return CACHE / f"{dataset}_1deg.npz"


_CUBE_LOCK = threading.Lock()
_CUBES: dict = {}


def cube(
    dataset: str,
    res: float = 1.0,
    mode: str = "regridded",
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> dict:
    key = (dataset, float(res), mode, year or 0, month or 0)
    with _CUBE_LOCK:
        if key in _CUBES:
            return _CUBES[key]
        if dataset == "spexone" and mode == "native":
            if year is None or month is None:
                raise ValueError("native SPEXone needs year and month")
            path = _cache_path("spexone", native=True, year=year, month=month)
            if not path.is_file():
                built = _build_spex_native(year, month)
                np.savez_compressed(path, **built)
            loaded = np.load(path)
            out = {k: np.array(loaded[k]) for k in loaded.files}
        elif dataset == "spexone":
            path = _cache_path("spexone", res=res)
            if not path.is_file():
                built = _build_spex(res)
                np.savez_compressed(path, **built)
            loaded = np.load(path)
            out = {k: np.array(loaded[k]) for k in loaded.files}
        else:
            path = _cache_path("polder")
            if not path.is_file():
                built = _build_polder()
                np.savez_compressed(path, **built)
            loaded = np.load(path)
            out = {k: np.array(loaded[k]) for k in loaded.files}
        _CUBES[key] = out
        return out


def _select(dataset: str, months: Sequence[int], year_months: Optional[Sequence[Tuple[int, int]]],
            res: float = 1.0, mode: str = "regridded", year: Optional[int] = None,
            month: Optional[int] = None):
    data = cube(dataset, res=res, mode=mode, year=year, month=month)
    year_a = data["year"]
    month_a = data["month"]
    keep = np.isin(month_a, list(months))
    if year_months:
        ym = {(int(y), int(m)) for y, m in year_months}
        keep = keep & np.array([(int(y), int(m)) in ym for y, m in zip(year_a, month_a)])
    if not np.any(keep):
        raise ValueError(f"no {dataset} data in months {list(months)}")
    return data, keep


def gridded(dataset: str, var: str, months: Sequence[int],
            year_months: Optional[Sequence[Tuple[int, int]]] = None,
            res: float = 1.0, mode: str = "regridded",
            year: Optional[int] = None, month: Optional[int] = None) -> np.ndarray:
    key = VAR_KEY[var]
    data, keep = _select(dataset, months, year_months, res=res, mode=mode, year=year, month=month)
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(data[key][keep], axis=0)


def dataset_meta() -> list:
    out = []
    resolutions = list_spex_resolutions()
    native_ok = SPEX_NATIVE_DIR.is_dir() and any(SPEX_NATIVE_DIR.glob("Extract*L2*.parquet"))
    if resolutions or native_ok:
        months = sorted({
            int(m.group(1))
            for p in SPEX_REGRID_DIR.glob("SPEXone_L2_regridded_*_v4_0.parquet")
            if (m := re.search(r"_(\d{2})_(?:\d+(?:\.\d+)?deg_)?v4", p.name))
        }) if SPEX_REGRID_DIR.is_dir() else list(range(1, 13))
        if not months and native_ok:
            months = sorted({
                int(m.group(1))
                for p in SPEX_NATIVE_DIR.glob("Extract*L2*.parquet")
                if (m := re.search(r"_(\d{2})_v4", p.name))
            })
        note = "SPEXone L2. Regridded files include the grid in the name (legacy files are 1°)."
        if not native_ok:
            note += f" Native folder {SPEX_NATIVE_DIR} is empty."
        out.append({
            "id": "spexone",
            "kind": "obs",
            "label": "SPEXone",
            "period": ["2024-07-01", "2025-12-31"],
            "variables": list(VARS),
            "months": months or list(range(1, 13)),
            "members": [],
            "resolutions": resolutions or [1.0],
            "native_available": native_ok,
            "note": note,
        })
    if ppe_io.POLDER_PATH.is_file():
        out.append({
            "id": "polder",
            "kind": "obs",
            "label": "POLDER-GRASP 2010",
            "period": ["2010-01-01", "2010-12-31"],
            "variables": list(VARS),
            "months": list(range(1, 13)),
            "members": [],
            "note": "POLDER-GRASP 2010 collocation, averaged to 1° after the same AOD/AAOD/SSA quality filter as the notebooks.",
        })
    return out


def field_payload(
    dataset: str,
    var: str,
    months: Optional[Sequence[int]] = None,
    res: float = 1.0,
    mode: str = "regridded",
    year: Optional[int] = None,
) -> dict:
    if var not in VAR_KEY:
        raise KeyError(f"{var} is not an observation field")
    months = list(months) if months else list(range(1, 13))
    if dataset == "spexone" and mode == "native":
        if len(months) != 1:
            raise ValueError("native SPEXone overpass uses one calendar month")
        month = int(months[0])
        year = int(year or (2024 if month >= 7 else 2025))
        grid = gridded(dataset, var, [month], res=res, mode="native", year=year, month=month)
        label = f"{D.VAR_LABELS.get(var, var)} (native {year}-{month:02d})"
    else:
        grid = gridded(dataset, var, months, res=res, mode="regridded")
        label = f"{D.VAR_LABELS.get(var, var)} ({res:g}°)" if dataset == "spexone" else D.VAR_LABELS.get(var, var)
    return D.pack_field(
        dataset=dataset, var=var, lon=LON, lat=LAT, data=grid, months=months,
        label=label,
    )


def region_value(
    dataset: str,
    var: str,
    ring,
    months: Sequence[int],
    surface_type: str = "all",
    year_months: Optional[Sequence[Tuple[int, int]]] = None,
    res: float = 1.0,
) -> float:
    grid = gridded(dataset, var, months, year_months=year_months, res=res)
    import xarray as xr
    da = xr.DataArray(grid, coords={"lat": LAT, "lon": LON}, dims=("lat", "lon"))
    weights = D.region_weights_polygon(da, ring, surface_type=surface_type)
    return D._weighted_mean_2d(grid, np.asarray(weights.values, dtype=float))


def region_mean_payload(dataset, var, ring, months=None, surface_type="all",
                        res: float = 1.0, mode: str = "regridded", year: Optional[int] = None) -> dict:
    if var not in VAR_KEY:
        raise KeyError(f"{var} is not an observation field")
    months = list(months) if months else list(range(1, 13))
    if mode == "native":
        month = int(months[0])
        year = int(year or (2024 if month >= 7 else 2025))
        grid = gridded(dataset, var, [month], res=res, mode="native", year=year, month=month)
    else:
        grid = gridded(dataset, var, months, res=res)
    import xarray as xr
    da = xr.DataArray(grid, coords={"lat": LAT, "lon": LON}, dims=("lat", "lon"))
    weights = D.region_weights_polygon(da, ring, surface_type=surface_type)
    w = np.asarray(weights.values, dtype=float)
    return {
        "dataset": dataset,
        "model": None,
        "var": var,
        "label": D.VAR_LABELS.get(var, var),
        "months": months,
        "surface_type": surface_type,
        "n_cells": int((w > 0).sum()),
        "mean": D._weighted_mean_2d(grid, w),
        "n_members": 0,
    }
