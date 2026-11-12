#!/usr/bin/env python3
"""Regional monthly / seasonal tables for PPE ensembles and AeroCom models.

One caller, ``build_regional``, covers both sources. Region geometry comes
from ``regions.REGIONS`` unless ``region_defs`` is passed (a notebook can
recompute a new box without editing the registry). ``surface_type`` is
honoured by the land/sea mask.

    python py/regional_aggregates.py --source ppe --ppe pd2010
    python py/regional_aggregates.py --source aerocom --species ss --regions SAO_2_DJF

``py/ppe_regional_aggregates.py`` re-exports this module.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, PROJECT_ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import PPE_set as cfg  # noqa: E402
import regions as reg  # noqa: E402
import cameo_toolbox as ct  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("regional")

GROUPS = ("burden", "emi", "optical", "precip", "meteo")
ENS_BATCH = 8
# Species mass coefficients only where that species' 550 nm AOD >= this.
SPECIES_MIN_OD = 1e-3
ID_COLS = ["ppe_id", "ens", "model", "experiment", "region"]

# Aerosol quality filter shared with the notebooks (ppe_io.QUALITY_FILTER_PARAMS).
# "aerocom" is the stricter threshold used by the older monthly sea-salt builder.
QUALITY_FILTER_PARAMS = {
    "min_aod": 0.01,
    "min_aaod": 1e-5,
    "max_aaod_fraction": 0.95,
    "min_ssa": 0.5,
    "max_ssa": 1.02,
}
AEROCOM_QUALITY = {
    "min_aod": 0.05,
    "min_aaod": 0.001,
    "max_aaod_fraction": 0.95,
    "min_ssa": 0.5,
    "max_ssa": 1.0,
}
QUALITY_PRESETS = {"ppe": QUALITY_FILTER_PARAMS, "aerocom": AEROCOM_QUALITY}

AEROCOM_DIR = PROJECT_ROOT / "Data" / "AP3_processed_monthly"
AEROCOM_EXCLUDE = {
    "GEOS-i33p2-met2010_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-OMA_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-MATRIX_AP3-CTRL",
    "NorESM2-met2010_AP3-CTRL-v3",
}
AEROCOM_VARS = (
    "od550aer", "abs550aer", "od440aer", "od865aer", "od870aer",
    "od550ss", "od550bc", "od550oa", "od550dust", "od550so4",
    "loadbc", "loadoa", "loaddust", "loadso4", "loadss",
    "emibc", "emioa", "emidust", "emiso2", "emiss", "emi_SO4", "emi_DMS",
    "precip",
)
AEROCOM_SEASONAL = PROJECT_ROOT / "tables" / "aerocom_regional_seasonal.parquet"


def ens_label(ens_id: int) -> str:
    eid = int(ens_id)
    return "Control" if eid < 0 else str(eid)


# ---------------------------------------------------------------------------
# Gridded inputs
# ---------------------------------------------------------------------------
def open_ensemble(ppe_id: str) -> xr.Dataset:
    """Merge the run's group NetCDFs into one (ens, time, lat, lon) Dataset."""
    parts = []
    globs = cfg.get_run(ppe_id)["file_globs"]
    for group in GROUPS:
        path = cfg.ensemble_file(ppe_id, group)
        if not path.is_file():
            if group == "precip" and not cfg.get_run(ppe_id)["has_precip"]:
                log.info("  %s: no precip ensemble (has_precip=False)", ppe_id)
                continue
            if group == "meteo" and "echam" not in globs:
                log.info("  %s: no meteo ensemble (no echam stream)", ppe_id)
                continue
            raise FileNotFoundError(f"Missing {ppe_id} ensemble file: {path}")
        ds = xr.open_dataset(path, engine=cfg.XR_WRITE_ENGINE)
        parts.append(ds)
    merged = xr.merge(parts, compat="override", join="outer", combine_attrs="drop_conflicts")
    return merged


def apply_burden_scale(ds: xr.Dataset, ppe_id: str, override: Optional[float] = None) -> xr.Dataset:
    ds = ds.copy()
    for v in cfg.LOAD_SCALE_VARS:
        if v not in ds:
            continue
        scale = override if override is not None else cfg.burden_scale_for(ppe_id, v)
        if scale == 1.0:
            continue
        da = ds[v] * scale
        da.attrs = dict(ds[v].attrs, burden_mass_scale=float(scale), units="kg m-2")
        ds[v] = da
    return ds


def resolve_quality(quality=None) -> dict:
    """``None`` or ``"ppe"`` is the default filter; ``"aerocom"`` is the strict one."""
    if quality is None or quality == "ppe":
        return dict(QUALITY_FILTER_PARAMS)
    if isinstance(quality, str):
        if quality not in QUALITY_PRESETS:
            raise KeyError(f"Unknown quality preset {quality!r}")
        return dict(QUALITY_PRESETS[quality])
    return {**QUALITY_FILTER_PARAMS, **dict(quality)}


def quality_filter(aod, aaod, quality=None):
    """Return (aod_f, aaod_f, ssa_f) with the notebook quality filter."""
    p = resolve_quality(quality)
    ssa = 1.0 - aaod / aod
    aod_mask = (aod > p["min_aod"]) & np.isfinite(aod)
    abs_mask = (
        aod_mask
        & (aaod > p["min_aaod"])
        & (aaod < aod * p["max_aaod_fraction"])
        & np.isfinite(aaod)
        & (ssa >= p["min_ssa"]) & (ssa <= p["max_ssa"]) & np.isfinite(ssa)
    )
    return aod.where(aod_mask), aaod.where(abs_mask), ssa.where(abs_mask).clip(0.0, 1.0)


def _get(ds: xr.Dataset, name: str) -> Optional[xr.DataArray]:
    return ds[name] if name in ds.data_vars else None


def _sum(ds: xr.Dataset, names: Sequence[str]) -> Optional[xr.DataArray]:
    if any(n not in ds.data_vars for n in names):
        return None
    total = ds[names[0]]
    for n in names[1:]:
        total = total + ds[n]
    return total


def _ratio(num: Optional[xr.DataArray], load: Optional[xr.DataArray],
           species_od: Optional[xr.DataArray] = None) -> Optional[xr.DataArray]:
    """Optical depth / column load in m2 g-1 (load kg m-2 -> g m-2).

    With ``species_od`` the ratio is kept only where that species' AOD is at
    least SPECIES_MIN_OD; cells with a near-zero load otherwise give ratios
    of 1e10-1e25 that dominate regional means.
    """
    if num is None or load is None:
        return None
    out = (num / (load * 1e3)).where(load > 0)
    if species_od is not None:
        out = out.where(species_od >= SPECIES_MIN_OD)
    return out


def _species_ssa(od: Optional[xr.DataArray], ab: Optional[xr.DataArray],
                 quality=None) -> Optional[xr.DataArray]:
    if od is None or ab is None:
        return None
    p = resolve_quality(quality)
    return (1.0 - ab / od).where(od > p["min_aod"]).clip(0.0, 1.0)


def derived_fields(ds: xr.Dataset, quality=None) -> Dict[str, xr.DataArray]:
    """Per-cell ratio quantities (area-averaged afterwards).

    MEC / MAC_total / MAC_BC_OA use the quality-filtered column AOD / AAOD;
    species coefficients (MEC_ss, MEC_du, MAC_du, MEC_so4) use the raw
    component optical depths, as in the notebooks.
    """
    out: Dict[str, Optional[xr.DataArray]] = {}
    aod, aaod = _get(ds, "od550aer"), _get(ds, "abs550aer")
    load_total = _sum(ds, reg.TOTAL_LOAD)
    load_bcoa = _sum(ds, reg.SPECIES["bb"]["load"])
    aod_f = aod
    if aod is not None and aaod is not None:
        aod_f, aaod_f, out["SSA"] = quality_filter(aod, aaod, quality)
        out["MEC"] = _ratio(aod_f, load_total)
        out["MAC_total"] = _ratio(aaod_f, load_total)
        out["MAC_BC_OA"] = _ratio(aaod_f, load_bcoa)

    # Angstrom exponents: 440-550 (AAOD / SPEXone convention) and 550-865 (SS).
    od440, od865 = _get(ds, "od440aer"), _get(ds, "od865aer")
    if aod_f is not None and od440 is not None:
        out["AE_440_550"] = -np.log(aod_f / od440.where(np.isfinite(aod_f))) / np.log(550.0 / 440.0)
    if aod_f is not None and od865 is not None:
        out["AE_550_865"] = -np.log(aod_f / od865.where(np.isfinite(aod_f))) / np.log(550.0 / 865.0)
    # AeroCom archives 870 nm much more often than 865 nm. 550-870 is the
    # stand-in Angstrom pair when 865 nm is absent (see box_sensitivity.prepare).
    od870 = _get(ds, "od870aer")
    if aod_f is not None and od870 is not None:
        out["AE_550_870"] = -np.log(aod_f / od870.where(np.isfinite(aod_f))) / np.log(550.0 / 870.0)

    bc, oa = _get(ds, "loadbc"), _get(ds, "loadoa")
    if bc is not None and oa is not None:
        out["rBC"] = (bc / (bc + oa)).where((bc + oa) > 0)

    od_ss, od_so4, od_du = _get(ds, "od550ss"), _get(ds, "od550so4"), _get(ds, "od550dust")
    out["MEC_ss"] = _ratio(od_ss, _get(ds, "loadss"), od_ss)
    out["MEC_so4"] = _ratio(od_so4, _get(ds, "loadso4"), od_so4)
    out["MEC_du"] = _ratio(od_du, _get(ds, "loaddust"), od_du)
    out["MAC_du"] = _ratio(_get(ds, "abs550dust"), _get(ds, "loaddust"), od_du)
    out["SSA_du"] = _species_ssa(_get(ds, "od550dust"), _get(ds, "abs550dust"), quality)
    out["SSA_ss"] = _species_ssa(_get(ds, "od550ss"), _get(ds, "abs550ss"), quality)
    du_od = _get(ds, "od550dust")
    if du_od is not None and aod is not None:
        out["dust_frac"] = (du_od / aod).where(aod > resolve_quality(quality)["min_aod"])
    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------------------
# Regional weights and aggregation
# ---------------------------------------------------------------------------
def region_weights(template: xr.DataArray, names: Sequence[str]) -> xr.DataArray:
    """(region, lat, lon) area weights. Same masks as AeroCom."""
    defs = {name: reg.REGIONS[name] for name in names}
    return weights_from_defs(template, defs)


def weights_from_defs(template: xr.DataArray, region_defs: Dict[str, dict]) -> xr.DataArray:
    """Area weights from ``create_region_mask`` (Natural Earth land/sea, edge weights)."""
    tpl = template
    for dim in ("ens", "time"):
        if dim in tpl.dims:
            tpl = tpl.isel({dim: 0}, drop=True)
    masks = ct.masks_for_regions(tpl, region_defs)
    names = list(region_defs)
    stack = []
    for name in names:
        edge = bool(region_defs[name].get("edge_weighted", False))
        w, _ = ct._spatial_weights(tpl, masks[name], edge_weighted=edge)
        stack.append(np.asarray(w.fillna(0.0).transpose("lat", "lon").values, dtype=np.float64))
    return xr.DataArray(
        np.stack(stack), dims=("region", "lat", "lon"),
        coords={"region": names, "lat": tpl.lat, "lon": tpl.lon},
    )


def regional_means(field: xr.DataArray, weights: xr.DataArray) -> np.ndarray:
    """skipna area-weighted mean -> array (ens, time, region)."""
    field = field.transpose("ens", "time", "lat", "lon")
    x = np.asarray(field.values, dtype=np.float64)
    finite = np.isfinite(x)
    x0 = np.where(finite, x, 0.0)
    w = np.asarray(weights.values, dtype=np.float64)
    num = np.einsum("etyx,ryx->etr", x0, w, optimize=True)
    den = np.einsum("etyx,ryx->etr", finite.astype(np.float64), w, optimize=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def regional_columns(ds: xr.Dataset, weights: xr.DataArray,
                     quality=None) -> Dict[str, np.ndarray]:
    """Area-weighted regional means of raw and derived fields.

    Returns ``{variable: array(ens, time, region)}``. ``ds`` must carry
    dimensions ens, time, lat, lon.
    """
    raw_vars = [v for v in cfg.VAR_MAP if v in ds.data_vars]
    for derived in ("precip", "stress_water"):
        if derived in ds.data_vars and derived not in raw_vars:
            raw_vars.append(derived)
    chunks: Dict[str, List[np.ndarray]] = {}
    n_ens = ds.sizes["ens"]
    for start in range(0, n_ens, ENS_BATCH):
        sub = ds[raw_vars].isel(ens=slice(start, start + ENS_BATCH)).load()
        fields = {v: sub[v] for v in raw_vars}
        fields.update(derived_fields(sub, quality=quality))
        for name, da in fields.items():
            chunks.setdefault(name, []).append(regional_means(da, weights))
        del sub, fields
        log.info("  members %d-%d of %d", start, min(start + ENS_BATCH, n_ens) - 1, n_ens)
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def build_monthly(ppe_id: str, region_names: Sequence[str],
                  burden_scale: Optional[float] = None,
                  region_defs: Optional[Dict[str, dict]] = None,
                  quality=None) -> pd.DataFrame:
    if burden_scale is None and cfg.get_run(ppe_id)["burden_scale"] is None:
        raise RuntimeError(
            f"{ppe_id}: burden_scale not verified (PPE_set.PPE_RUNS); run "
            "py/ppe_validate.py and set it, or pass --burden-scale."
        )
    log.info("[%s] opening ensemble files", ppe_id)
    ds = open_ensemble(ppe_id)
    ds = apply_burden_scale(ds, ppe_id, burden_scale)

    if region_defs is not None:
        region_names = list(region_defs)
    template = ds["od550aer"].isel(ens=0, time=0, drop=True)
    if region_defs is not None:
        weights = weights_from_defs(template, region_defs)
    else:
        weights = region_weights(template, region_names)
    log.info("[%s] %d members, %d months, %d regions", ppe_id, ds.sizes["ens"],
             ds.sizes["time"], len(region_names))
    columns = regional_columns(ds, weights, quality=quality)
    log.info("[%s] aggregated %d variables", ppe_id, len(columns))

    ens = ds["ens"].values.astype(int)
    exps = (ds["experiment"].values.astype(str) if "experiment" in ds.coords
            else np.array([ens_label(e) for e in ens]))
    times = pd.to_datetime(ds["time"].values)
    n_e, n_t, n_r = len(ens), len(times), len(region_names)
    idx = pd.MultiIndex.from_product([range(n_e), range(n_t), range(n_r)])
    e_i, t_i, r_i = (np.asarray(idx.get_level_values(k)) for k in range(3))
    df = pd.DataFrame({
        "ppe_id": ppe_id,
        "ens": ens[e_i],
        "model": [ens_label(e) for e in ens[e_i]],
        "experiment": exps[e_i],
        "region": np.asarray(region_names)[r_i],
        "time": times[t_i],
    })
    for name, arr in columns.items():
        df[name] = arr.reshape(-1).astype(np.float32)
    if "precip" not in df.columns:
        df["precip"] = np.float32(np.nan)
    ds.close()
    return df


def seasonal_from_monthly(monthly: pd.DataFrame,
                          seasons: Optional[Dict[str, Sequence[int]]] = None) -> pd.DataFrame:
    """Mean over each region's season_months (month-of-year; wraps allowed).

    ``seasons`` overrides ``regions.REGIONS`` for boxes that are not registered
    (the ±5° sensitivity set).
    """
    work = monthly.copy()
    work["_mon"] = pd.to_datetime(work["time"]).dt.month
    keep = np.zeros(len(work), dtype=bool)
    seasons = seasons or {}
    for r in work["region"].unique():
        # dict.get evaluates its default immediately, so do not look up REGIONS
        # until we know the box is not in the override map.
        months = seasons.get(r)
        if months is None:
            months = reg.REGIONS[r]["season_months"]
        keep |= (work["region"].to_numpy() == r) & work["_mon"].isin(list(months)).to_numpy()
    work = work.loc[keep]
    value_cols = [c for c in work.columns if c not in ID_COLS + ["time", "_mon"]]
    grouped = work.groupby(ID_COLS, sort=False)
    out = grouped[value_cols].mean()
    out["n_months"] = grouped.size()
    return out.reset_index()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _merge_existing(new: pd.DataFrame, path: Path, ppe_ids: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        return new
    old = pd.read_parquet(path)
    old = old.loc[~old["ppe_id"].isin(ppe_ids)]
    return pd.concat([old, new], ignore_index=True, sort=False)


def write_manifest(ppe_ids: Sequence[str], monthly: pd.DataFrame, region_names: Sequence[str],
                   burden_scales: Dict[str, float]) -> None:
    manifest = {}
    if cfg.REGIONAL_MANIFEST.is_file():
        manifest = json.loads(cfg.REGIONAL_MANIFEST.read_text())
    manifest.setdefault("ppe", {})
    for ppe_id in ppe_ids:
        run = cfg.get_run(ppe_id)
        sub = monthly.loc[monthly["ppe_id"] == ppe_id]
        files = {}
        for g in GROUPS:
            p = cfg.ensemble_file(ppe_id, g)
            if p.is_file():
                files[g] = {"path": str(p), "mtime": p.stat().st_mtime}
        manifest["ppe"][ppe_id] = {
            "label": run["label"],
            "source_root": str(run["root"]),
            "period": list(run["period"]),
            "has_precip": bool(run["has_precip"]),
            "burden_scale": burden_scales[ppe_id],
            "n_ens": int(sub["ens"].nunique()),
            "months": sorted({str(t)[:7] for t in sub["time"].unique()}),
            "ensemble_files": files,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    manifest["regions"] = list(region_names)
    manifest["region_hash"] = reg.definition_hash(region_names)
    manifest["region_definitions"] = {
        n: {k: reg.REGIONS[n][k] for k in
            ("surface_type", "lon_range", "lat_range", "season_months", "edge_weighted",
             "species", "role")}
        for n in region_names
    }
    manifest["quality_filter"] = QUALITY_FILTER_PARAMS
    manifest["species_min_od"] = SPECIES_MIN_OD
    manifest["columns"] = [c for c in monthly.columns if c not in ID_COLS + ["time"]]
    cfg.REGIONAL_MANIFEST.write_text(json.dumps(manifest, indent=2, default=list))


def available_ppes() -> List[str]:
    return [p for p in cfg.PPE_RUNS if cfg.ensemble_file(p, "optical").is_file()]


def region_defs_from(regions: Optional[Sequence[str]] = None,
                     region_defs: Optional[Dict[str, dict]] = None,
                     species: Optional[str] = None) -> Dict[str, dict]:
    """Registry boxes, or an explicit dict so a notebook can recompute a new box.

    Each definition keeps ``surface_type`` so the land/sea mask still applies.
    """
    if region_defs is not None:
        defs = dict(region_defs)
        if regions is not None:
            missing = [n for n in regions if n not in defs]
            if missing:
                raise KeyError(f"region_defs missing {missing}")
            defs = {n: defs[n] for n in regions}
        return defs
    if regions is None and species is not None:
        names = ["global"] + reg.regions_for_species(species) if "global" in reg.REGIONS else reg.regions_for_species(species)
    else:
        names = list(regions) if regions is not None else list(reg.REGIONS)
    missing = [n for n in names if n not in reg.REGIONS]
    if missing:
        raise KeyError(f"Unknown regions {missing}; pass region_defs to use a custom box")
    return {n: reg.REGIONS[n] for n in names}


def _seasons_of(defs: Dict[str, dict]) -> Dict[str, Sequence[int]]:
    return {n: tuple(c["season_months"]) for n, c in defs.items()}


def aerocom_model_names(exclude: Optional[Sequence[str]] = None) -> List[str]:
    vdir = AEROCOM_DIR / "od550aer"
    if not vdir.is_dir():
        return []
    suffix = "_od550aer_processed.nc"
    skip = set(AEROCOM_EXCLUDE if exclude is None else exclude)
    names = []
    for path in sorted(vdir.glob("*_processed.nc")):
        if path.name.endswith(suffix):
            names.append(path.name[: -len(suffix)])
    return [n for n in names if n not in skip]


def _open_aerocom_var(model: str, var: str) -> Optional[xr.DataArray]:
    import aerocom_data

    path = AEROCOM_DIR / var / f"{model}_{var}_processed.nc"
    if not path.is_file():
        return None
    ds = xr.open_dataset(path)
    try:
        ds = aerocom_data.preprocess_for_save(ds, var, model_hint=model, year=2010)
    except Exception as exc:
        log.warning("  %s %s preprocess failed: %s", model, var, exc)
        ds.close()
        return None
    if ds is None or var not in ds:
        if ds is not None:
            ds.close()
        return None
    da = ds[var].load()
    ds.close()
    if "lat" not in getattr(da, "dims", ()) or "lon" not in da.dims:
        return None
    if da["lat"].ndim != 1 or da["lon"].ndim != 1:
        return None
    return da


def aerocom_dataset(model: str, variables: Sequence[str] = AEROCOM_VARS) -> Optional[xr.Dataset]:
    """One AeroCom model on a common grid, with a singleton ``ens`` dimension."""
    import aerocom_data

    ref = _open_aerocom_var(model, "od550aer")
    if ref is None:
        log.warning("  %s: no od550aer", model)
        return None
    data = {"od550aer": ref}
    for var in variables:
        if var == "od550aer":
            continue
        da = _open_aerocom_var(model, var)
        if da is None:
            continue
        da = aerocom_data._align_da_to_ref(da, ref, model_hint=f"{model}/{var}")
        if "time" in da.dims and "time" in ref.dims:
            da = da.reindex(time=ref.time)
        data[var] = da
    return xr.Dataset(data).expand_dims(ens=[0])


def _frame_from_columns(columns, region_names, ppe_id, ens, models, times) -> pd.DataFrame:
    n_e, n_t, n_r = len(ens), len(times), len(region_names)
    idx = pd.MultiIndex.from_product([range(n_e), range(n_t), range(n_r)])
    e_i, t_i, r_i = (np.asarray(idx.get_level_values(k)) for k in range(3))
    df = pd.DataFrame({
        "ppe_id": ppe_id,
        "ens": np.asarray(ens)[e_i],
        "model": np.asarray(models)[e_i],
        "experiment": np.asarray(models)[e_i],
        "region": np.asarray(region_names)[r_i],
        "time": np.asarray(times)[t_i],
    })
    for name, arr in columns.items():
        df[name] = arr.reshape(-1).astype(np.float32)
    if "precip" not in df.columns:
        df["precip"] = np.float32(np.nan)
    return df


def build_aerocom_monthly(defs: Dict[str, dict], models: Optional[Sequence[str]] = None,
                          quality=None) -> pd.DataFrame:
    models = list(models or aerocom_model_names())
    frames = []
    for i, model in enumerate(models, start=1):
        log.info("AeroCom %d/%d %s", i, len(models), model)
        ds = aerocom_dataset(model)
        if ds is None:
            continue
        weights = weights_from_defs(ds["od550aer"].isel(ens=0, time=0, drop=True), defs)
        columns = regional_columns(ds, weights, quality=quality)
        times = pd.to_datetime(ds["time"].values)
        frames.append(_frame_from_columns(
            columns, list(defs), "AeroCom", np.array([i]), [model], times,
        ))
        ds.close()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_regional(
    source: str,
    regions: Optional[Sequence[str]] = None,
    region_defs: Optional[Dict[str, dict]] = None,
    species: Optional[str] = None,
    ppe_ids: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    quality=None,
    out_path: Optional[Path] = None,
    seasonal: bool = True,
    burden_scale: Optional[float] = None,
) -> pd.DataFrame:
    """Area-weighted regional table for a PPE ensemble or the AeroCom models.

    ``region_defs`` overrides ``regions.REGIONS``. Each box still carries
    ``surface_type``, so land and ocean masking is unchanged. ``quality``
    is ``None`` (PPE filter), ``"aerocom"`` (stricter), or a dict of thresholds.
    Lifetimes are added on the returned frame; the PPE parquet on disk stays
    the raw regional means that ``ppe_io.load_regional`` turns into lifetimes.
    """
    import ppe_io

    source = source.lower()
    if source not in ("ppe", "aerocom"):
        raise ValueError("source must be 'ppe' or 'aerocom'")
    defs = region_defs_from(regions, region_defs, species)
    seasons = _seasons_of(defs)
    if source == "ppe":
        ids = list(ppe_ids or available_ppes())
        frames = [build_monthly(pid, list(defs), burden_scale=burden_scale,
                                region_defs=defs, quality=quality) for pid in ids]
        monthly = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    else:
        monthly = build_aerocom_monthly(defs, models, quality=quality)
    if monthly.empty:
        return monthly
    table = seasonal_from_monthly(monthly, seasons=seasons) if seasonal else monthly
    out = ppe_io.add_budget_columns(table, mask_outflow_lifetimes=False)
    if species is not None:
        out = ppe_io.apply_species_aliases(out, species)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        log.info("Wrote %s (%d rows)", out_path, len(out))
    elif source == "aerocom":
        path = AEROCOM_SEASONAL if seasonal else AEROCOM_SEASONAL.with_name(
            "aerocom_regional_monthly.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        log.info("Wrote %s (%d rows)", path, len(out))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="ppe", choices=("ppe", "aerocom"))
    p.add_argument("--ppe", nargs="+", default=None, choices=sorted(cfg.PPE_RUNS),
                   help="PPE runs to (re)build (default: all with ensemble files)")
    p.add_argument("--models", nargs="*", default=None,
                   help="AeroCom models (default: all non-excluded)")
    p.add_argument("--species", default=None, choices=["bb", "ss", "du"])
    p.add_argument("--regions", nargs="+", default=None,
                   help="Region subset (default: all in py/regions.py, or the species)")
    p.add_argument("--quality", default=None, choices=("ppe", "aerocom"),
                   help="AOD quality filter (default: ppe)")
    p.add_argument("--burden-scale", type=float, default=None,
                   help="Override PPE_RUNS burden_scale (validation runs)")
    p.add_argument("--out-dir", type=Path, default=cfg.REGIONAL_DIR)
    p.add_argument("--out", type=Path, default=None,
                   help="Seasonal parquet for a one-off build (AeroCom default is tables/)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.source == "aerocom":
        build_regional(
            "aerocom", regions=args.regions, species=args.species,
            models=args.models, quality=args.quality, out_path=args.out,
        )
        return 0
    ppe_ids = args.ppe or available_ppes()
    if not ppe_ids:
        log.error("No PPE ensemble files found under %s", cfg.SCRATCH_ROOT)
        return 1
    region_names = args.regions or list(reg.REGIONS)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = out_dir / cfg.REGIONAL_MONTHLY.name
    seasonal_path = out_dir / cfg.REGIONAL_SEASONAL.name

    frames, scales = [], {}
    for ppe_id in ppe_ids:
        frames.append(build_monthly(ppe_id, region_names, args.burden_scale))
        scales[ppe_id] = (args.burden_scale if args.burden_scale is not None
                          else cfg.get_run(ppe_id)["burden_scale"])
    monthly = pd.concat(frames, ignore_index=True, sort=False)
    seasonal = seasonal_from_monthly(monthly)

    monthly_all = _merge_existing(monthly, monthly_path, ppe_ids)
    seasonal_all = _merge_existing(seasonal, seasonal_path, ppe_ids)
    monthly_all.to_parquet(monthly_path, index=False)
    seasonal_all.to_parquet(seasonal_path, index=False)
    if out_dir == cfg.REGIONAL_DIR:
        write_manifest(ppe_ids, monthly_all, region_names, scales)
    log.info("Wrote %s (%d rows, %.1f MB)", monthly_path, len(monthly_all),
             monthly_path.stat().st_size / 1e6)
    log.info("Wrote %s (%d rows, %.1f MB)", seasonal_path, len(seasonal_all),
             seasonal_path.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
