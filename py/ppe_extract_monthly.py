#!/usr/bin/env python3
"""Stream monthly PPE fields into ensemble NetCDFs (AOD_SS_ERROR inputs).

Read-only over the Pace PPE experiment tree. Per-member temps land on
scratch; durable products are written under Data/PPE_processed_monthly/.
Never writes into the source directory. No pickle.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import PPE_set as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ppe_extract")

ENS_DIR_RE = re.compile(r"^PPE_ENS_(.+)$")


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------
def ens_id_from_name(name: str) -> int:
    """Map PPE_ENS_12 -> 12, PPE_ENS_Control -> CONTROL_ENS_ID."""
    m = ENS_DIR_RE.match(name)
    if not m:
        raise ValueError(f"Not a PPE_ENS_* directory: {name}")
    tag = m.group(1)
    if tag.lower() == "control":
        return cfg.CONTROL_ENS_ID
    return int(tag)


def experiment_label(name: str) -> str:
    m = ENS_DIR_RE.match(name)
    return m.group(1) if m else name


def discover_ensembles(root: Path, only: Optional[Sequence[str]] = None) -> List[Path]:
    dirs = [p for p in root.glob(cfg.PPE_PATTERN) if p.is_dir()]

    def sort_key(p: Path):
        try:
            eid = ens_id_from_name(p.name)
        except ValueError:
            return (1, p.name)
        # Control last among negatives; numeric members ascending.
        return (0, eid if eid >= 0 else 10**9, p.name)

    dirs = sorted(dirs, key=sort_key)
    if only:
        wanted = set(only)
        dirs = [p for p in dirs if p.name in wanted]
        missing = wanted - {p.name for p in dirs}
        if missing:
            raise FileNotFoundError(f"Requested ensembles not found: {sorted(missing)}")
    return dirs


def _is_excluded(path: Path) -> bool:
    name = path.name
    return any(s in name for s in cfg.EXCLUDE_NAME_SUBSTR)


def find_source_file(ens_dir: Path, group: str) -> Optional[Path]:
    """Return the first matching NetCDF for a source group, or None."""
    pattern = cfg.FILE_GLOBS[group]
    matches = [
        p for p in ens_dir.glob(pattern)
        if p.is_file() and not _is_excluded(p) and p.stat().st_size > 0
    ]
    # Prefer non-symlink outputs when both exist (unlikely).
    matches.sort(key=lambda p: (p.is_symlink(), p.name))
    return matches[0] if matches else None


def source_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# Open / extract
# ---------------------------------------------------------------------------
def open_dataset(path: Path, variables: Optional[Sequence[str]] = None) -> xr.Dataset:
    """Open with configured engine; optionally subset to named variables."""
    engines = [cfg.XR_ENGINE]
    if "h5netcdf" not in engines:
        engines.append("h5netcdf")
    if "netcdf4" not in engines:
        engines.append("netcdf4")

    last_err: Optional[Exception] = None
    ds = None
    for eng in engines:
        try:
            ds = xr.open_dataset(path, engine=eng)
            break
        except Exception as e:
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(f"Could not open {path}: {last_err}")

    if variables:
        keep = [v for v in variables if v in ds.data_vars]
        extra = [c for c in ("time_bnds",) if c in ds.variables]
        ds = ds[keep + extra]
    return ds


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename["longitude"] = "lon"
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename["latitude"] = "lat"
    if rename:
        ds = ds.rename(rename)
    return ds


def normalize_time_month_start(obj: xr.DataArray | xr.Dataset):
    """Label each timestep by calendar month start so streams align."""
    if "time" not in obj.coords:
        return obj
    times = pd.to_datetime(np.asarray(obj["time"].values))
    month_starts = times.to_period("M").to_timestamp().to_numpy()
    return obj.assign_coords(time=month_starts)


def to_monthly_mean(ds: xr.Dataset) -> xr.Dataset:
    """Resample high-frequency data to calendar-month means."""
    if "time" not in ds.coords:
        return ds
    n = int(ds.sizes.get("time", 0))
    if n <= 12:
        return ds
    log.info("  Resampling %d timesteps -> monthly mean", n)
    # Drop non-lat/lon/time extras that break resample (e.g. spectral).
    keep_dims = set()
    for v in ds.data_vars:
        keep_dims.update(ds[v].dims)
    drop_coords = [c for c in ds.coords if c not in ("time", "lat", "lon") and c not in keep_dims]
    if drop_coords:
        ds = ds.drop_vars([c for c in drop_coords if c in ds.variables], errors="ignore")
    monthly = ds.resample(time="MS").mean(keep_attrs=True)
    return monthly


def extract_var_from_file(
    path: Path,
    ham_name: str,
    *,
    allow_hifreq_resample: bool,
) -> Optional[xr.DataArray]:
    """Load one variable; optionally monthly-mean if high-frequency."""
    try:
        ds = open_dataset(path, variables=[ham_name])
    except Exception as e:
        log.warning("  Failed to open %s for %s: %s", path.name, ham_name, e)
        return None
    try:
        if ham_name not in ds.data_vars:
            return None
        ds = standardize_coords(ds)
        da = ds[ham_name]
        if allow_hifreq_resample and "time" in da.dims and da.sizes.get("time", 0) > 12:
            ds2 = to_monthly_mean(ds[[ham_name]])
            da = ds2[ham_name]
        drop_dims = [d for d in da.dims if d not in ("time", "lat", "lon")]
        if drop_dims:
            log.warning("  %s has unexpected dims %s — skipping", ham_name, drop_dims)
            return None
        da = da.load()
        da = normalize_time_month_start(da)
        return da
    finally:
        ds.close()


def resolve_variable(
    ens_dir: Path,
    aerocom_name: str,
) -> Tuple[Optional[xr.DataArray], Optional[Path], str, str]:
    """Return (dataarray, source_path, status, ham_name).

    status is monthly | hifreq_resampled | missing.
    """
    ham_name, preferred, _units = cfg.VAR_MAP[aerocom_name]
    # Preferred first, then other groups.
    groups: List[str] = [preferred] + [g for g in cfg.SOURCE_FALLBACK if g != preferred]
    for group in groups:
        path = find_source_file(ens_dir, group)
        if path is None:
            continue
        allow_resample = group == "hifreq"
        da = extract_var_from_file(path, ham_name, allow_hifreq_resample=allow_resample)
        if da is None:
            continue
        status = "hifreq_resampled" if group == "hifreq" else "monthly"
        return da, path, status, ham_name
    return None, None, "missing", ham_name


def build_member_dataset(ens_dir: Path) -> Tuple[xr.Dataset, List[dict]]:
    """Extract all mapped variables for one ensemble member."""
    rows: List[dict] = []
    data_vars: Dict[str, xr.DataArray] = {}
    exp = experiment_label(ens_dir.name)
    eid = ens_id_from_name(ens_dir.name)

    for aerocom_name in cfg.VAR_MAP:
        da, path, status, ham_name = resolve_variable(ens_dir, aerocom_name)
        units = cfg.VAR_MAP[aerocom_name][2]
        row = {
            "experiment": exp,
            "ens_id": eid,
            "aerocom_name": aerocom_name,
            "ham_name": ham_name,
            "source_file": str(path) if path else "",
            "status": status,
            "units": units,
        }
        rows.append(row)
        if da is None:
            log.warning("  MISSING %s (%s) in %s", aerocom_name, ham_name, ens_dir.name)
            continue
        da = da.astype(np.float32)
        da.name = aerocom_name
        da.attrs = dict(da.attrs)
        da.attrs["ham_name"] = ham_name
        da.attrs["aerocom_name"] = aerocom_name
        da.attrs["units"] = da.attrs.get("units", units)
        da.attrs["source_file"] = path.name if path else ""
        data_vars[aerocom_name] = da

    # Record explicitly-missing deposition / wet / dry names.
    for miss in cfg.MISSING_VARS:
        rows.append(
            {
                "experiment": exp,
                "ens_id": eid,
                "aerocom_name": miss,
                "ham_name": "",
                "source_file": "",
                "status": "missing",
                "units": "",
            }
        )

    if not data_vars:
        raise RuntimeError(f"No variables extracted for {ens_dir.name}")

    # Align all fields onto the shared month-start time axis (inner join).
    ref_name = next(
        (n for n in ("od550aer", "emiss", "loadss", "precip_na") if n in data_vars),
        next(iter(data_vars)),
    )
    ref_time = data_vars[ref_name]["time"]
    aligned = {}
    for name, da in data_vars.items():
        if "time" in da.dims:
            da = da.reindex(time=ref_time)
        aligned[name] = da
    data_vars = aligned

    ds = xr.Dataset(data_vars)
    ds = standardize_coords(ds)

    # Derived precip mm day-1 from precip_na.
    if "precip_na" in ds:
        precip = (ds["precip_na"] * 86400.0).astype(np.float32)
        precip.name = "precip"
        precip.attrs = {
            "long_name": "precipitation rate",
            "units": "mm day-1",
            "derived_from": "precip_na",
            "conversion": "precip_na * 86400",
        }
        ds["precip"] = precip
        rows.append(
            {
                "experiment": exp,
                "ens_id": eid,
                "aerocom_name": "precip",
                "ham_name": "precip_na*86400",
                "source_file": next(
                    (r["source_file"] for r in rows if r["aerocom_name"] == "precip_na"),
                    "",
                ),
                "status": "monthly",
                "units": "mm day-1",
            }
        )

    ds = ds.expand_dims(ens=[eid])
    ds = ds.assign_coords(experiment=("ens", np.array([exp], dtype="U64")))
    ds.attrs["title"] = f"PPE monthly extract {ens_dir.name}"
    ds.attrs["source_dir"] = str(ens_dir)
    return ds, rows


def member_out_path(ens_dir: Path) -> Path:
    return cfg.SCRATCH_MEMBERS / f"{ens_dir.name}.nc"


def sources_for_member(ens_dir: Path) -> List[Path]:
    paths = []
    for group in cfg.FILE_GLOBS:
        if group == "hifreq":
            continue  # do not gate renew on the 25 GB file
        p = find_source_file(ens_dir, group)
        if p is not None:
            paths.append(p)
    return paths


def member_is_fresh(ens_dir: Path, out: Path) -> bool:
    if cfg.RENEW or not out.exists():
        return False
    try:
        out_m = out.stat().st_mtime
    except OSError:
        return False
    srcs = sources_for_member(ens_dir)
    if not srcs:
        return False
    return out_m >= max(source_mtime(p) for p in srcs)


def netcdf_encoding(ds: xr.Dataset, with_ens_chunks: bool = True) -> dict:
    enc = {}
    for name, da in ds.data_vars.items():
        e = {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "_FillValue": np.float32(np.nan),
        }
        if with_ens_chunks and set(da.dims) >= {"ens", "time", "lat", "lon"}:
            e["chunksizes"] = tuple(
                min(cfg.ENS_CHUNK.get(d, da.sizes[d]), da.sizes[d]) for d in da.dims
            )
        enc[name] = e
    return enc


def provenance_from_existing_member(ens_dir: Path, out: Path) -> List[dict]:
    """Rebuild provenance rows from an existing scratch member file."""
    exp = experiment_label(ens_dir.name)
    eid = ens_id_from_name(ens_dir.name)
    rows: List[dict] = []
    try:
        ds = xr.open_dataset(out, engine=getattr(cfg, "XR_WRITE_ENGINE", cfg.XR_ENGINE))
    except Exception as e:
        log.warning("Could not read existing %s for provenance: %s", out, e)
        return rows
    try:
        for aerocom_name, (ham_name, _grp, units) in cfg.VAR_MAP.items():
            if aerocom_name in ds.data_vars:
                da = ds[aerocom_name]
                rows.append(
                    {
                        "experiment": exp,
                        "ens_id": eid,
                        "aerocom_name": aerocom_name,
                        "ham_name": da.attrs.get("ham_name", ham_name),
                        "source_file": da.attrs.get("source_file", ""),
                        "status": "monthly",
                        "units": da.attrs.get("units", units),
                    }
                )
            else:
                rows.append(
                    {
                        "experiment": exp,
                        "ens_id": eid,
                        "aerocom_name": aerocom_name,
                        "ham_name": ham_name,
                        "source_file": "",
                        "status": "missing",
                        "units": units,
                    }
                )
        if "precip" in ds.data_vars:
            rows.append(
                {
                    "experiment": exp,
                    "ens_id": eid,
                    "aerocom_name": "precip",
                    "ham_name": "precip_na*86400",
                    "source_file": ds["precip"].attrs.get("derived_from", "precip_na"),
                    "status": "monthly",
                    "units": "mm day-1",
                }
            )
        for miss in cfg.MISSING_VARS:
            rows.append(
                {
                    "experiment": exp,
                    "ens_id": eid,
                    "aerocom_name": miss,
                    "ham_name": "",
                    "source_file": "",
                    "status": "missing",
                    "units": "",
                }
            )
    finally:
        ds.close()
    return rows


def write_member(ens_dir: Path) -> Tuple[Path, List[dict]]:
    out = member_out_path(ens_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if member_is_fresh(ens_dir, out):
        log.info("Skip (fresh) %s", ens_dir.name)
        return out, provenance_from_existing_member(ens_dir, out)

    log.info("Extracting %s ...", ens_dir.name)
    ds, rows = build_member_dataset(ens_dir)
    tmp = out.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(
            tmp,
            engine=getattr(cfg, "XR_WRITE_ENGINE", cfg.XR_ENGINE),
            encoding=netcdf_encoding(ds, with_ens_chunks=False),
        )
        tmp.replace(out)
    finally:
        ds.close()
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    log.info("  Wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out, rows


# ---------------------------------------------------------------------------
# Ensemble concat
# ---------------------------------------------------------------------------
def write_group_ensemble(member_paths: Sequence[Path], group: str, var_names: Sequence[str]) -> Path:
    """Concat member files along ens for one variable group."""
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.OUTPUT_DIR / f"{group}_monthly.nc"
    log.info("Building %s from %d members ...", out.name, len(member_paths))

    datasets = []
    for path in member_paths:
        write_eng = getattr(cfg, "XR_WRITE_ENGINE", cfg.XR_ENGINE)
        ds = xr.open_dataset(path, engine=write_eng)
        present = [v for v in var_names if v in ds.data_vars]
        if not present:
            ds.close()
            continue
        sub = ds[present]
        if "experiment" in ds.coords:
            sub = sub.assign_coords(experiment=ds.coords["experiment"])
        datasets.append(sub.load())
        ds.close()

    if not datasets:
        raise RuntimeError(f"No member data for group {group}")

    combined = xr.concat(datasets, dim="ens", combine_attrs="override")
    combined = combined.sortby("ens")
    combined.attrs["title"] = f"PPE ensemble monthly {group}"
    combined.attrs["n_ensemble"] = int(combined.sizes.get("ens", 0))
    combined.attrs["source_root"] = str(cfg.PPE_ROOT)

    tmp = out.with_suffix(".nc.tmp")
    write_eng = getattr(cfg, "XR_WRITE_ENGINE", cfg.XR_ENGINE)
    combined.to_netcdf(
        tmp,
        engine=write_eng,
        encoding=netcdf_encoding(combined, with_ens_chunks=True),
    )
    combined.close()
    tmp.replace(out)
    log.info("  Wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


def append_provenance(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames = [
        "experiment",
        "ens_id",
        "aerocom_name",
        "ham_name",
        "source_file",
        "status",
        "units",
    ]
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def reset_provenance(path: Path) -> None:
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Restrict to these ensemble directory names (e.g. PPE_ENS_1)",
    )
    p.add_argument(
        "--renew",
        action="store_true",
        help="Overwrite existing scratch member files and ensemble NetCDFs",
    )
    p.add_argument(
        "--skip-ensemble",
        action="store_true",
        help="Only write per-member scratch files (skip group concat)",
    )
    p.add_argument(
        "--ppe-root",
        type=Path,
        default=cfg.PPE_ROOT,
        help="Override PPE experiments root (read-only)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.renew:
        cfg.RENEW = True

    cfg.SCRATCH_MEMBERS.mkdir(parents=True, exist_ok=True)
    cfg.SCRATCH_LOGS.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    root = args.ppe_root
    if not root.is_dir():
        log.error("PPE root not found: %s", root)
        return 1

    ensembles = discover_ensembles(root, only=args.only)
    log.info("Found %d ensembles under %s", len(ensembles), root)
    if not ensembles:
        return 1

    # Fresh provenance for this run (full or restricted).
    reset_provenance(cfg.PROVENANCE_CSV)

    member_paths: List[Path] = []
    for ens_dir in ensembles:
        try:
            path, rows = write_member(ens_dir)
            member_paths.append(path)
            append_provenance(rows, cfg.PROVENANCE_CSV)
        except Exception as e:
            log.exception("Failed %s: %s", ens_dir.name, e)

    if not member_paths:
        log.error("No member files produced")
        return 1

    if args.skip_ensemble:
        log.info("Skipping ensemble concat (--skip-ensemble)")
        return 0

    for group, var_names in cfg.GROUP_VARS.items():
        try:
            write_group_ensemble(member_paths, group, var_names)
        except Exception as e:
            log.exception("Failed group %s: %s", group, e)
            return 1

    log.info("Done. Provenance: %s", cfg.PROVENANCE_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
