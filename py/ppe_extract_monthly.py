#!/usr/bin/env python3
"""Stream monthly PPE fields into ensemble NetCDFs for one registered PPE run.

Read-only over the Pace PPE experiment tree. Per-member extracts and the
ensemble group NetCDFs land on scratch under
``/scratch-shared/jzhang1/PPE/<ppe_id>/`` (see ``PPE_set.run_dirs``).
Never writes into the source directory. No pickle.

    python py/ppe_extract_monthly.py --ppe pd2010 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

PROVENANCE_FIELDS = [
    "ppe_id",
    "experiment",
    "ens_id",
    "aerocom_name",
    "ham_name",
    "source_file",
    "status",
    "units",
]


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
    dirs = []
    for p in root.glob(cfg.PPE_PATTERN):
        if not p.is_dir():
            continue
        try:
            ens_id_from_name(p.name)
        except ValueError:
            continue
        dirs.append(p)

    def sort_key(p: Path):
        eid = ens_id_from_name(p.name)
        return (eid if eid >= 0 else 10**9, p.name)

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


def find_source_file(ens_dir: Path, group: str, globs: Dict[str, str],
                     group_roots: Optional[Dict[str, Path]] = None) -> Optional[Path]:
    """Return the first matching NetCDF for a source group, or None.

    ``group_roots`` maps a group to another experiments root holding the
    same PPE_ENS_* member directories (e.g. precip delivered separately).
    """
    pattern = globs.get(group)
    if pattern is None:
        return None
    search_dir = ens_dir
    if group_roots and group in group_roots:
        search_dir = Path(group_roots[group]) / ens_dir.name
    matches = [
        p for p in search_dir.glob(pattern)
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
    """Label each timestep by calendar month start so streams align.

    Handles mid-month (emi, 2024 rad) and end-of-month (2010 burden/rad,
    e.g. 2010-01-31T23:52) stamps alike.
    """
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
    keep_dims = set()
    for v in ds.data_vars:
        keep_dims.update(ds[v].dims)
    drop_coords = [c for c in ds.coords if c not in ("time", "lat", "lon") and c not in keep_dims]
    if drop_coords:
        ds = ds.drop_vars([c for c in drop_coords if c in ds.variables], errors="ignore")
    return ds.resample(time="MS").mean(keep_attrs=True)


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
    ppe_id: str,
) -> Tuple[Optional[xr.DataArray], Optional[Path], str, str]:
    """Return (dataarray, source_path, status, ham_name).

    status is monthly | hifreq_resampled | missing.
    """
    run = cfg.get_run(ppe_id)
    globs = run["file_globs"]
    ham_name = cfg.VAR_MAP[aerocom_name][0]
    preferred = cfg.preferred_group(ppe_id, aerocom_name)
    groups: List[str] = [preferred] + [g for g in cfg.SOURCE_FALLBACK if g != preferred]
    groups = [g for g in groups if g in globs]
    for group in groups:
        path = find_source_file(ens_dir, group, globs, run.get("group_roots"))
        if path is None:
            continue
        allow_resample = group == "hifreq"
        da = extract_var_from_file(path, ham_name, allow_hifreq_resample=allow_resample)
        if da is None:
            continue
        status = "hifreq_resampled" if group == "hifreq" else "monthly"
        return da, path, status, ham_name
    return None, None, "missing", ham_name


def _prov_row(ppe_id, exp, eid, aerocom_name, ham_name, source_file, status, units) -> dict:
    return {
        "ppe_id": ppe_id,
        "experiment": exp,
        "ens_id": eid,
        "aerocom_name": aerocom_name,
        "ham_name": ham_name,
        "source_file": source_file,
        "status": status,
        "units": units,
    }


def build_member_dataset(ens_dir: Path, ppe_id: str) -> Tuple[xr.Dataset, List[dict]]:
    """Extract all mapped variables for one ensemble member."""
    rows: List[dict] = []
    data_vars: Dict[str, xr.DataArray] = {}
    exp = experiment_label(ens_dir.name)
    eid = ens_id_from_name(ens_dir.name)

    for aerocom_name in cfg.VAR_MAP:
        da, path, status, ham_name = resolve_variable(ens_dir, aerocom_name, ppe_id)
        units = cfg.VAR_MAP[aerocom_name][2]
        rows.append(_prov_row(ppe_id, exp, eid, aerocom_name, ham_name,
                              str(path) if path else "", status, units))
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

    for miss in cfg.MISSING_VARS:
        rows.append(_prov_row(ppe_id, exp, eid, miss, "", "", "missing", ""))

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
    ds = standardize_coords(xr.Dataset(aligned))

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
        src = next((r["source_file"] for r in rows if r["aerocom_name"] == "precip_na"), "")
        rows.append(_prov_row(ppe_id, exp, eid, "precip", "precip_na*86400", src,
                              "monthly", "mm day-1"))
    else:
        rows.append(_prov_row(ppe_id, exp, eid, "precip", "", "", "missing", "mm day-1"))

    ds = ds.expand_dims(ens=[eid])
    ds = ds.assign_coords(experiment=("ens", np.array([exp], dtype="U64")))
    ds.attrs["title"] = f"PPE monthly extract {ppe_id} {ens_dir.name}"
    ds.attrs["ppe_id"] = ppe_id
    ds.attrs["source_dir"] = str(ens_dir)
    return ds, rows


def member_out_path(ens_dir: Path, ppe_id: str) -> Path:
    return cfg.run_dirs(ppe_id)["members"] / f"{ens_dir.name}.nc"


def sources_for_member(ens_dir: Path, globs: Dict[str, str]) -> List[Path]:
    paths = []
    for group in globs:
        if group == "hifreq":
            continue  # do not gate renew on the 25 GB file
        p = find_source_file(ens_dir, group, globs)
        if p is not None:
            paths.append(p)
    return paths


def member_is_fresh(ens_dir: Path, out: Path, globs: Dict[str, str], renew: bool) -> bool:
    if renew or not out.exists():
        return False
    try:
        out_m = out.stat().st_mtime
    except OSError:
        return False
    srcs = sources_for_member(ens_dir, globs)
    if not srcs:
        return False
    if out_m < max(source_mtime(p) for p in srcs):
        return False
    # Stale if the variable map grew since the member was written.
    try:
        with xr.open_dataset(out, engine=cfg.XR_WRITE_ENGINE) as ds:
            have = set(ds.data_vars)
    except Exception:
        return False
    missing_now = set(cfg.VAR_MAP) - have
    # Precip vars are legitimately absent when the run has no precip stream.
    if not any(g in globs for g in ("vphysc", "echam")):
        missing_now -= set(cfg.PRECIP_VARS)
    return not missing_now


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


def provenance_from_existing_member(ens_dir: Path, out: Path, ppe_id: str) -> List[dict]:
    """Rebuild provenance rows from an existing scratch member file."""
    exp = experiment_label(ens_dir.name)
    eid = ens_id_from_name(ens_dir.name)
    rows: List[dict] = []
    try:
        ds = xr.open_dataset(out, engine=cfg.XR_WRITE_ENGINE)
    except Exception as e:
        log.warning("Could not read existing %s for provenance: %s", out, e)
        return rows
    try:
        for aerocom_name, (ham_name, _grp, units) in cfg.VAR_MAP.items():
            if aerocom_name in ds.data_vars:
                da = ds[aerocom_name]
                rows.append(_prov_row(ppe_id, exp, eid, aerocom_name,
                                      da.attrs.get("ham_name", ham_name),
                                      da.attrs.get("source_file", ""), "monthly",
                                      da.attrs.get("units", units)))
            else:
                rows.append(_prov_row(ppe_id, exp, eid, aerocom_name, ham_name, "",
                                      "missing", units))
        if "precip" in ds.data_vars:
            rows.append(_prov_row(ppe_id, exp, eid, "precip", "precip_na*86400",
                                  ds["precip"].attrs.get("derived_from", "precip_na"),
                                  "monthly", "mm day-1"))
        else:
            rows.append(_prov_row(ppe_id, exp, eid, "precip", "", "", "missing", "mm day-1"))
        for miss in cfg.MISSING_VARS:
            rows.append(_prov_row(ppe_id, exp, eid, miss, "", "", "missing", ""))
    finally:
        ds.close()
    return rows


def write_member(ens_dir: Path, ppe_id: str, renew: bool = False) -> Tuple[Path, List[dict]]:
    globs = cfg.get_run(ppe_id)["file_globs"]
    out = member_out_path(ens_dir, ppe_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    if member_is_fresh(ens_dir, out, globs, renew):
        log.info("Skip (fresh) %s", ens_dir.name)
        return out, provenance_from_existing_member(ens_dir, out, ppe_id)

    log.info("Extracting %s/%s ...", ppe_id, ens_dir.name)
    ds, rows = build_member_dataset(ens_dir, ppe_id)
    tmp = out.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(
            tmp,
            engine=cfg.XR_WRITE_ENGINE,
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
def write_group_ensemble(
    member_paths: Sequence[Path],
    group: str,
    var_names: Sequence[str],
    ppe_id: str,
    source_root: Path,
) -> Optional[Path]:
    """Concat member files along ens for one variable group.

    Returns None (no file) when no member carries any variable of the group,
    e.g. precip for a PPE without a precip stream.
    """
    out_dir = cfg.run_dirs(ppe_id)["ensemble"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{group}_monthly.nc"
    log.info("Building %s/%s from %d members ...", ppe_id, out.name, len(member_paths))

    datasets = []
    for path in member_paths:
        ds = xr.open_dataset(path, engine=cfg.XR_WRITE_ENGINE)
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
        log.warning("  No member data for group %s in %s; not written", group, ppe_id)
        if out.exists():
            out.unlink()
        return None

    combined = xr.concat(datasets, dim="ens", combine_attrs="override")
    combined = combined.sortby("ens")
    combined.attrs["title"] = f"PPE ensemble monthly {group} ({ppe_id})"
    combined.attrs["ppe_id"] = ppe_id
    combined.attrs["n_ensemble"] = int(combined.sizes.get("ens", 0))
    combined.attrs["source_root"] = str(source_root)

    tmp = out.with_suffix(".nc.tmp")
    combined.to_netcdf(
        tmp,
        engine=cfg.XR_WRITE_ENGINE,
        encoding=netcdf_encoding(combined, with_ens_chunks=True),
    )
    combined.close()
    tmp.replace(out)
    log.info("  Wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


def write_provenance(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (int(r["ens_id"]) if int(r["ens_id"]) >= 0 else 10**9,
                                       r["aerocom_name"]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROVENANCE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PROVENANCE_FIELDS})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ppe",
        default=cfg.DEFAULT_PPE,
        choices=sorted(cfg.PPE_RUNS),
        help="Registered PPE run (PPE_set.PPE_RUNS)",
    )
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
        default=None,
        help="Override the run's experiments root (read-only)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel member extracts (processes)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ppe_id = args.ppe
    run = cfg.get_run(ppe_id)
    renew = bool(args.renew or cfg.RENEW)
    dirs = cfg.run_dirs(ppe_id)
    for key in ("members", "ensemble", "logs"):
        dirs[key].mkdir(parents=True, exist_ok=True)

    root = args.ppe_root or run["root"]
    if not root.is_dir():
        log.error("PPE root not found: %s", root)
        return 1

    ensembles = discover_ensembles(root, only=args.only)
    log.info("[%s] Found %d ensembles under %s", ppe_id, len(ensembles), root)
    if not ensembles:
        return 1

    member_paths: List[Path] = []
    all_rows: List[dict] = []
    failed: List[str] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(write_member, d, ppe_id, renew): d for d in ensembles}
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    path, rows = fut.result()
                    member_paths.append(path)
                    all_rows.extend(rows)
                except Exception as e:
                    log.exception("Failed %s: %s", d.name, e)
                    failed.append(d.name)
    else:
        for d in ensembles:
            try:
                path, rows = write_member(d, ppe_id, renew)
                member_paths.append(path)
                all_rows.extend(rows)
            except Exception as e:
                log.exception("Failed %s: %s", d.name, e)
                failed.append(d.name)

    write_provenance(all_rows, dirs["provenance"])
    if failed:
        log.warning("Failed members (%d): %s", len(failed), failed)
    if not member_paths:
        log.error("No member files produced")
        return 1

    if args.skip_ensemble:
        log.info("Skipping ensemble concat (--skip-ensemble)")
        return 0

    for group, var_names in cfg.GROUP_VARS.items():
        try:
            write_group_ensemble(sorted(member_paths), group, var_names, ppe_id, root)
        except Exception as e:
            log.exception("Failed group %s: %s", group, e)
            return 1

    log.info("Done. Provenance: %s", dirs["provenance"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
