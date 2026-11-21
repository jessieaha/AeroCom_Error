#!/usr/bin/env python3
"""Regional monthly / seasonal means for the ICON-HAM PPE (hjia).

ICON files are monthly means on the native R2B4 grid (ncells=20480), one
variable directory and one member file:

    .../ICON-HAM-PPE_24_25/PD/<var>/ens_XXX_PD_<var>.nc

Cells inside each ``regions.py`` box are area-weighted (spherical triangle
area, land/ocean mask). No remapping onto the ECHAM T63 grid.

AE for BB/DU is the archived ANG_440nm_670nm (not 440–550). SS uses
ANG_550nm_865nm. There is no wet-deposition stream, so dry/total ratios
are not formed. Precipitation ``pr`` (kg m-2 s-1) is stored as mm day-1.

    python py/icon_regional.py
    python py/icon_regional.py --members 000,001 --workers 4
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

import PPE_set as cfg
import regions as reg
from regional_aggregates import SPECIES_MIN_OD

log = logging.getLogger("icon_regional")

PPE_ID = "icon2025"
MIN_AOD = 0.01
MIN_AAOD = 1e-5
MAX_AAOD_FRAC = 0.95
MIN_SSA, MAX_SSA = 0.5, 1.02

# AeroCom / parquet name -> ICON directory (and usually the NetCDF variable).
ICON_VARS = {
    "od550aer": "TAU_2D_550nm",
    "abs550aer": "ABS_2D_550nm",
    "od550ss": "TAU_COMP_SS_550nm",
    "od550so4": "TAU_COMP_SO4_550nm",
    "od550dust": "TAU_COMP_DU_550nm",
    "od550bc": "TAU_COMP_BC_550nm",
    "od550oa": "TAU_COMP_OC_550nm",
    "loadss": "burden_ss",
    "loadso4": "burden_so4",
    "loadbc": "burden_bc",
    "loadoa": "burden_oc",
    "loaddust": "burden_du",
    "burden_SO2": "burden_so2",
    "burden_DMS": "burden_dms",
    "emiss": "emi_ss",
    "emiso2": "emi_so2",
    "emibc": "emi_bc",
    "emioa": "emi_oc",
    "emidust": "emi_du",
    "emi_SO4": "emi_so4",
    "emi_DMS": "emi_dms",
    "dryss": "ddep_ss",
    "drybc": "ddep_bc",
    "dryoa": "ddep_oc",
    "drydu": "ddep_du",
    "dryso4": "ddep_so4",
    "precip": "pr",
    "ANG_440nm_670nm": "ANG_440nm_670nm",
    "ANG_550nm_865nm": "ANG_550nm_865nm",
}

# Extensive fields (area mean). Ratios are built per cell before averaging.
EXTENSIVE = [
    "od550aer", "abs550aer", "od550ss", "od550so4", "od550dust", "od550bc", "od550oa",
    "loadss", "loadso4", "loadbc", "loadoa", "loaddust", "burden_SO2", "burden_DMS",
    "emiss", "emiso2", "emibc", "emioa", "emidust", "emi_SO4", "emi_DMS",
    "dryss", "drybc", "dryoa", "drydu", "dryso4", "precip",
    "ANG_440nm_670nm", "ANG_550nm_865nm",
]


def _icon_root() -> Path:
    return Path(cfg.get_run(PPE_ID)["root"])


def member_ids(root: Path) -> List[str]:
    folder = root / "TAU_2D_550nm"
    ids = []
    for path in sorted(folder.glob("ens_*_PD_TAU_2D_550nm.nc")):
        ids.append(path.name.split("_")[1])
    return ids


def _open_field(root: Path, member: str, folder: str) -> Optional[tuple]:
    path = root / folder / f"ens_{member}_PD_{folder}.nc"
    if not path.is_file():
        return None
    with xr.open_dataset(path) as ds:
        name = next(v for v in ds.data_vars if ds[v].dims == ("time", "ncells"))
        data = np.asarray(ds[name].values, dtype=np.float64)
        time = np.asarray(ds["time"].values)
    return data, time


def _spherical_excess(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Steradians of each spherical triangle. lon/lat are (ncells, 3) radians."""
    xyz = np.stack((
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ), axis=-1)

    def angle(i, j, k):
        ui, uj, uk = xyz[:, i], xyz[:, j], xyz[:, k]
        n1 = np.cross(uj, ui)
        n2 = np.cross(uj, uk)
        n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
        n2 /= np.linalg.norm(n2, axis=1, keepdims=True)
        sin = np.einsum("ij,ij->i", np.cross(n1, n2), uj)
        cos = np.clip(np.einsum("ij,ij->i", n1, n2), -1.0, 1.0)
        return np.arctan2(sin, cos)

    excess = angle(1, 0, 2) + angle(2, 1, 0) + angle(0, 2, 1) - np.pi
    return np.abs(excess)


def _land_mask(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Same Natural Earth land test as gridded AeroCom and PPE masks."""
    import cameo_toolbox as ct
    return ct.point_is_land(lon_deg, lat_deg)


def build_region_weights(root: Path) -> Dict[str, dict]:
    """Per-region cell weights and the month axis from member 000."""
    sample = root / "TAU_2D_550nm" / "ens_000_PD_TAU_2D_550nm.nc"
    with xr.open_dataset(sample) as ds:
        lon = np.degrees(np.asarray(ds["clon"].values, dtype=float)) % 360.0
        lat = np.degrees(np.asarray(ds["clat"].values, dtype=float))
        blo = np.asarray(ds["clon_bnds"].values, dtype=float)
        bla = np.asarray(ds["clat_bnds"].values, dtype=float)
        time = np.asarray(ds["time"].values)
    area = _spherical_excess(blo, bla)
    area = np.where(np.isfinite(area) & (area > 0), area, 0.0)
    log.info("land mask on %d ICON cells", lon.size)
    land = _land_mask(lon, lat)
    names = []
    for species in ("bb", "ss", "du"):
        names.extend(reg.regions_for_species(species))
    names = list(dict.fromkeys(names))
    weights = {}
    for name in names:
        box = reg.REGIONS[name]
        lo0, lo1 = box["lon_range"]
        la0, la1 = box["lat_range"]
        lon_ok = (lon >= lo0) & (lon <= lo1) if lo0 <= lo1 else (lon >= lo0) | (lon <= lo1)
        lat_ok = (lat >= min(la0, la1)) & (lat <= max(la0, la1))
        mask = lon_ok & lat_ok
        surface = box.get("surface_type", "all")
        if surface == "ocean":
            mask = mask & ~land
        elif surface == "land":
            mask = mask & land
        w = np.where(mask, area, 0.0)
        if w.sum() <= 0:
            log.warning("region %s has no ICON cells", name)
        weights[name] = w
    months = np.array([int(t) // 100 % 100 for t in time], dtype=int)
    years = np.array([int(t) // 10000 for t in time], dtype=int)
    return {"weights": weights, "months": months, "years": years, "time": time}


def _wmean(data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Area mean over cells for each time. data is (time, ncells)."""
    finite = np.isfinite(data)
    w = weight[None, :] * finite
    num = np.nansum(np.where(finite, data, 0.0) * weight, axis=1)
    den = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 0] = np.nan
    return out


def _ratios(fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-cell ratios, same definitions as regional_aggregates.derived_fields."""
    od = fields.get("od550aer")
    ab = fields.get("abs550aer")
    out = {}
    if od is None or ab is None:
        return out
    ssa = 1.0 - ab / od
    aod_ok = (od > MIN_AOD) & np.isfinite(od)
    abs_ok = (
        aod_ok & (ab > MIN_AAOD) & (ab < od * MAX_AAOD_FRAC) & np.isfinite(ab)
        & (ssa >= MIN_SSA) & (ssa <= MAX_SSA)
    )
    aod_f = np.where(aod_ok, od, np.nan)
    aaod_f = np.where(abs_ok, ab, np.nan)
    out["SSA"] = np.where(abs_ok, np.clip(ssa, 0.0, 1.0), np.nan)

    def ratio(num, load, species_od=None):
        if num is None or load is None:
            return None
        r = num / (load * 1e3)
        r = np.where(load > 0, r, np.nan)
        if species_od is not None:
            r = np.where(species_od >= SPECIES_MIN_OD, r, np.nan)
        return r

    loads = [fields.get(k) for k in ("loadbc", "loadoa", "loaddust", "loadss", "loadso4")]
    if all(v is not None for v in loads):
        load_total = sum(loads)
        out["MEC"] = ratio(aod_f, load_total)
        out["MAC_total"] = ratio(aaod_f, load_total)
    bcoa = None
    if fields.get("loadbc") is not None and fields.get("loadoa") is not None:
        bcoa = fields["loadbc"] + fields["loadoa"]
        out["MAC_BC_OA"] = ratio(aaod_f, bcoa)
        out["rBC"] = np.where(bcoa > 0, fields["loadbc"] / bcoa, np.nan)
    out["MEC_ss"] = ratio(fields.get("od550ss"), fields.get("loadss"), fields.get("od550ss"))
    out["MEC_so4"] = ratio(fields.get("od550so4"), fields.get("loadso4"), fields.get("od550so4"))
    out["MEC_du"] = ratio(fields.get("od550dust"), fields.get("loaddust"), fields.get("od550dust"))
    if fields.get("ANG_440nm_670nm") is not None:
        # Closest archived Angstrom; BB/DU notebooks read this as AE_440_550.
        out["AE_440_550"] = np.where(aod_ok, fields["ANG_440nm_670nm"], np.nan)
    if fields.get("ANG_550nm_865nm") is not None:
        out["AE_550_865"] = np.where(aod_ok, fields["ANG_550nm_865nm"], np.nan)
    if od is not None and fields.get("od550dust") is not None:
        out["dust_frac"] = np.where(aod_ok, fields["od550dust"] / od, np.nan)
    return {k: v for k, v in out.items() if v is not None}


def _one_member(args) -> List[dict]:
    member, root_s, bundle_path = args
    root = Path(root_s)
    with open(bundle_path, "rb") as handle:
        bundle = pickle.load(handle)
    weights = bundle["weights"]
    months = bundle["months"]
    years = bundle["years"]
    fields = {}
    for name, folder in ICON_VARS.items():
        opened = _open_field(root, member, folder)
        if opened is None:
            continue
        data, _time = opened
        if name == "precip":
            data = data * 86400.0  # kg m-2 s-1 -> mm day-1
        fields[name] = data
    if "od550aer" not in fields:
        return []
    derived = _ratios(fields)
    rows = []
    ntime = fields["od550aer"].shape[0]
    for region, weight in weights.items():
        if weight.sum() <= 0:
            continue
        series = {name: _wmean(arr, weight) for name, arr in {**fields, **derived}.items()
                  if arr is not None and arr.shape[0] == ntime}
        for i in range(ntime):
            row = {
                "ppe_id": PPE_ID,
                "ens": int(member),
                "model": member,
                "experiment": "PD",
                "region": region,
                "time": pd.Timestamp(year=int(years[i]), month=int(months[i]), day=1),
            }
            for name, values in series.items():
                row[name] = float(values[i]) if np.isfinite(values[i]) else np.nan
            rows.append(row)
    return rows


def _seasonal(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    value_cols = [c for c in monthly.columns if c not in
                  ("ppe_id", "ens", "model", "experiment", "region", "time")]
    for (ens, model, region), sub in monthly.groupby(["ens", "model", "region"], sort=False):
        months = set(reg.REGIONS[region]["season_months"])
        keep = sub.loc[sub["time"].dt.month.isin(months)]
        if keep.empty:
            continue
        rec = {
            "ppe_id": PPE_ID, "ens": int(ens), "model": model, "experiment": "PD",
            "region": region, "n_months": int(len(keep)),
        }
        for col in value_cols:
            rec[col] = float(keep[col].mean(skipna=True))
        rows.append(rec)
    return pd.DataFrame(rows)


def extract(members: Optional[Sequence[str]] = None, workers: int = 8,
            regions: Optional[Sequence[str]] = None, merge: bool = False) -> None:
    root = _icon_root()
    ids = member_ids(root)
    if members:
        want = set(members)
        ids = [m for m in ids if m in want]
    if not ids:
        raise SystemExit(f"No ICON members under {root}")
    log.info("ICON members: %d  root=%s", len(ids), root)
    bundle = build_region_weights(root)
    if regions:
        missing = [name for name in regions if name not in bundle["weights"]]
        if missing:
            raise KeyError(f"ICON weights have no {missing}")
        bundle["weights"] = {name: bundle["weights"][name] for name in regions}
    cache = cfg.REGIONAL_DIR / "_icon_weights.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as handle:
        pickle.dump(
            {"weights": bundle["weights"], "months": bundle["months"], "years": bundle["years"]},
            handle, protocol=4,
        )

    from concurrent.futures import ProcessPoolExecutor
    tasks = [(m, str(root), str(cache)) for m in ids]
    rows: List[dict] = []
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            rows.extend(_one_member(task))
            if i % 20 == 0:
                log.info("  %d / %d members", i, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, part in enumerate(pool.map(_one_member, tasks, chunksize=2), 1):
                rows.extend(part)
                if i % 20 == 0:
                    log.info("  %d / %d members", i, len(tasks))
    monthly = pd.DataFrame(rows)
    if monthly.empty:
        raise SystemExit("ICON extract produced no rows")
    seasonal = _seasonal(monthly)
    mon_path = cfg.REGIONAL_DIR / "icon2025_regional_monthly.parquet"
    sea_path = cfg.REGIONAL_DIR / "icon2025_regional_seasonal.parquet"
    if merge:
        keep = list(regions or [])
        for path, new in ((mon_path, monthly), (sea_path, seasonal)):
            old = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
            if not old.empty and keep:
                old = old.loc[~old["region"].isin(keep)]
            new = pd.concat([old, new], ignore_index=True, sort=False)
            new.to_parquet(path, index=False)
            log.info("Merged %s (%d rows)", path.name, len(new))
    else:
        monthly.to_parquet(mon_path, index=False)
        seasonal.to_parquet(sea_path, index=False)
    log.info("Wrote %s (%d rows) and %s (%d rows)",
             mon_path.name, len(monthly), sea_path.name, len(seasonal))
    cache.unlink(missing_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", default="", help="Comma-separated ids, e.g. 000,001")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    members = [m.strip() for m in args.members.split(",") if m.strip()] or None
    extract(members, workers=args.workers)


if __name__ == "__main__":
    main()
