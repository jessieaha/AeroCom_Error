#!/usr/bin/env python3
"""Box-size and box-location sensitivity for sea salt and biomass burning.

Baselines
---------
* Sea salt: ``SAO_SOND`` and the remote South Pacific box ``SPO_DJF``
  (formerly ``remote_SO_DJF``). North Pacific is not a baseline.
* Biomass burning: ``africa`` and ``amazon``.
  Geometry and ``surface_type`` are copied from ``regions.REGIONS``.

Each baseline is shifted by -5, 0, +5 degrees in longitude and in latitude,
and its width and height are changed by -5, 0, +5 degrees (centre fixed).
A perturbed box is kept only when it is at least 7 degrees in latitude and
15 degrees in longitude.

The nominal (0, 0, 0) box keeps the baseline name so it matches ``regions.REGIONS``.

    python py/box_sensitivity.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, PROJECT_ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import PPE_set as cfg  # noqa: E402
import aerocom_data  # noqa: E402
import cameo_toolbox as ct  # noqa: E402
import ppe_io  # noqa: E402
import ppe_regional_aggregates as pra  # noqa: E402
import regions as reg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("box_sensitivity")

MIN_DLAT = 7.0
MIN_DLON = 15.0
SHIFT_DEG = (-5, 0, 5)
SIZE_DEG = (-5, 0, 5)

# Species -> baseline region names (geometry comes from regions.REGIONS).
BASELINES = {
    "ss": ("SAO_SOND", "SPO_DJF"),
    "bb": ("africa", "amazon"),
}
AEROCOM_EXTRA = {
    "ss": (),
    "bb": (),
}

PPE_SEASONAL = cfg.REGIONAL_DIR / "box_sensitivity_seasonal.parquet"
AEROCOM_SEASONAL = PROJECT_ROOT / "tables" / "aerocom_box_sensitivity_seasonal.parquet"
DEFS_JSON = cfg.REGIONAL_DIR / "box_sensitivity_defs.json"

MONTHLY_DIR = PROJECT_ROOT / "Data" / "AP3_processed_monthly"
AEROCOM_EXCLUDE = {
    "GEOS-i33p2-met2010_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-OMA_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-MATRIX_AP3-CTRL",
    "NorESM2-met2010_AP3-CTRL-v3",
}
AEROCOM_VARS = (
    "od550aer", "abs550aer", "od440aer", "od865aer", "od870aer",
    "od550ss", "od550bc", "od550oa",
    "loadbc", "loadoa", "loaddust", "loadso4", "loadss",
    "emibc", "emioa", "emidust", "emiso2", "emiss",
    "precip",
)

# Relations summarised in the sensitivity notebook: (x, y, inverse_y, through_origin).
RELATIONS = {
    "ss": (
        ("SSA", "MEC_ss", False, False),
        ("1-SSA", "MAC", False, True),
        ("AE", "lifetime_ss", True, False),
        ("precip", "lifetime_ss", True, False),
    ),
    "bb": (
        ("1-SSA", "MAC", False, True),
        ("AE", "lifetime_BC_OA", True, False),
        ("precip", "lifetime_BC_OA", True, False),
    ),
}


def _span(lon0: float, lon1: float) -> Tuple[float, float, float]:
    """Return (lon0, lon1_unwrapped, width) with lon1 >= lon0."""
    lon0, lon1 = float(lon0) % 360.0, float(lon1) % 360.0
    if lon1 < lon0:
        lon1 += 360.0
    return lon0, lon1, lon1 - lon0


def _fold_lon(lon0: float, lon1: float) -> Tuple[float, float]:
    """Fold an unwrapped [lon0, lon1] back to 0..360 (lon0 > lon1 means wrap)."""
    width = lon1 - lon0
    a = lon0 % 360.0
    b = (lon0 + width) % 360.0
    if width >= 360.0 - 1e-6:
        return 0.0, 360.0
    return a, b


def perturb_box(baseline: str, dlon: int, dlat: int, dsize: int) -> Optional[dict]:
    """One perturbed copy of a registered region, or None if it is too small."""
    src = reg.REGIONS[baseline]
    lon0, lon1, width = _span(*src["lon_range"])
    lat0, lat1 = float(src["lat_range"][0]), float(src["lat_range"][1])
    lat0, lat1 = min(lat0, lat1), max(lat0, lat1)
    new_w = width + float(dsize)
    new_h = (lat1 - lat0) + float(dsize)
    if new_w < MIN_DLON - 1e-6 or new_h < MIN_DLAT - 1e-6 or new_w >= 360.0:
        return None
    clon = (lon0 + lon1) / 2.0 + float(dlon)
    clat = (lat0 + lat1) / 2.0 + float(dlat)
    nlat0, nlat1 = clat - new_h / 2.0, clat + new_h / 2.0
    if nlat0 < -90.0 or nlat1 > 90.0:
        return None
    nlon0, nlon1 = _fold_lon(clon - new_w / 2.0, clon + new_w / 2.0)
    if dlon == 0 and dlat == 0 and dsize == 0:
        name = baseline
    else:
        name = f"{baseline}__lon{dlon:+d}_lat{dlat:+d}_sz{dsize:+d}"
    return dict(
        surface_type=src["surface_type"],
        lon_range=(float(nlon0), float(nlon1)),
        lat_range=(float(nlat0), float(nlat1)),
        season_months=tuple(src["season_months"]),
        edge_weighted=bool(src["edge_weighted"]),
        species=tuple(src["species"]),
        role="source",
        color=src["color"],
        baseline=baseline,
        dlon=int(dlon),
        dlat=int(dlat),
        dsize=int(dsize),
        name=name,
    )


def sensitivity_defs(species: Optional[Iterable[str]] = None,
                     include_extra: bool = False) -> Dict[str, dict]:
    """All kept boxes, keyed by region name. Nominal box uses the baseline name."""
    species = list(species or BASELINES)
    out: Dict[str, dict] = {}
    for sp in species:
        for baseline in BASELINES[sp]:
            for dlon in SHIFT_DEG:
                for dlat in SHIFT_DEG:
                    for dsize in SIZE_DEG:
                        box = perturb_box(baseline, dlon, dlat, dsize)
                        if box is None:
                            log.info("drop %s dlon=%s dlat=%s dsize=%s (below %s x %s)",
                                     baseline, dlon, dlat, dsize, MIN_DLAT, MIN_DLON)
                            continue
                        out[box["name"]] = box
        if include_extra:
            for name in AEROCOM_EXTRA[sp]:
                src = reg.REGIONS[name]
                lon0, lon1 = src["lon_range"]
                lat0, lat1 = src["lat_range"]
                out[name] = dict(
                    surface_type=src["surface_type"],
                    lon_range=(float(lon0), float(lon1)),
                    lat_range=(float(min(lat0, lat1)), float(max(lat0, lat1))),
                    season_months=tuple(src["season_months"]),
                    edge_weighted=bool(src["edge_weighted"]),
                    species=tuple(src["species"]),
                    role=src["role"],
                    color=src["color"],
                    baseline=name,
                    dlon=0, dlat=0, dsize=0,
                    name=name,
                )
    return out


def seasons_of(defs: Dict[str, dict]) -> Dict[str, Tuple[int, ...]]:
    return {name: tuple(cfg["season_months"]) for name, cfg in defs.items()}


def _with_cell_counts(defs: Dict[str, dict], template: xr.DataArray) -> Dict[str, dict]:
    weights = pra.weights_from_defs(template, defs)
    out = {}
    for name in defs:
        n = int(np.count_nonzero(weights.sel(region=name).values > 0))
        out[name] = dict(defs[name], n_cells=n)
    return out


def _frame_from_columns(columns: Dict[str, np.ndarray], region_names: Sequence[str],
                        ppe_id: str, ens: np.ndarray, models: Sequence[str],
                        times: pd.DatetimeIndex) -> pd.DataFrame:
    n_e, n_t, n_r = len(ens), len(times), len(region_names)
    idx = pd.MultiIndex.from_product([range(n_e), range(n_t), range(n_r)])
    e_i, t_i, r_i = (np.asarray(idx.get_level_values(k)) for k in range(3))
    df = pd.DataFrame({
        "ppe_id": ppe_id,
        "ens": ens[e_i],
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


def _attach(df: pd.DataFrame, defs: Dict[str, dict]) -> pd.DataFrame:
    meta = pd.DataFrame([
        {"region": name, "baseline": c["baseline"], "dlon": c["dlon"], "dlat": c["dlat"],
         "dsize": c["dsize"], "lon0": c["lon_range"][0], "lon1": c["lon_range"][1],
         "lat0": c["lat_range"][0], "lat1": c["lat_range"][1],
         "n_cells": c.get("n_cells", np.nan),
         "season": ",".join(str(m) for m in c["season_months"])}
        for name, c in defs.items()
    ])
    return df.merge(meta, on="region", how="left")


def cell_counts(defs: Dict[str, dict], ppe_id: str = "pd2024") -> Dict[str, dict]:
    """T63 cell counts after the land/ocean mask (same grid as the PPE)."""
    path = cfg.ensemble_file(ppe_id, "optical")
    ds = xr.open_dataset(path)
    template = ds["od550aer"].isel(ens=0, time=0, drop=True).load()
    ds.close()
    return _with_cell_counts(defs, template)


def build_ppe(defs: Dict[str, dict], ppe_ids: Sequence[str],
              counted: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    seasons = seasons_of(defs)
    frames = []
    for ppe_id in ppe_ids:
        monthly = pra.build_monthly(ppe_id, list(defs), region_defs=defs)
        seasonal = pra.seasonal_from_monthly(monthly, seasons=seasons)
        frames.append(seasonal)
        log.info("[%s] seasonal rows %d", ppe_id, len(seasonal))
    return _attach(pd.concat(frames, ignore_index=True), counted or defs)


def aerocom_model_names() -> List[str]:
    vdir = MONTHLY_DIR / "od550aer"
    suffix = "_od550aer_processed.nc"
    names = []
    for path in sorted(vdir.glob("*_processed.nc")):
        if path.name.endswith(suffix):
            names.append(path.name[: -len(suffix)])
    return [n for n in names if n not in AEROCOM_EXCLUDE]


def _open_aerocom_var(model: str, var: str) -> Optional[xr.DataArray]:
    path = MONTHLY_DIR / var / f"{model}_{var}_processed.nc"
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
    if "lat" not in da.dims or "lon" not in da.dims:
        return None
    if da["lat"].ndim != 1 or da["lon"].ndim != 1:
        return None
    return da


def _aerocom_dataset(model: str) -> Optional[xr.Dataset]:
    ref = _open_aerocom_var(model, "od550aer")
    if ref is None:
        log.warning("  %s: no od550aer", model)
        return None
    data = {"od550aer": ref}
    for var in AEROCOM_VARS:
        if var == "od550aer":
            continue
        da = _open_aerocom_var(model, var)
        if da is None:
            continue
        da = aerocom_data._align_da_to_ref(da, ref, model_hint=f"{model}/{var}")
        if "time" in da.dims and "time" in ref.dims:
            da = da.reindex(time=ref.time)
        data[var] = da
    ds = xr.Dataset(data)
    return ds.expand_dims(ens=[0])


def build_aerocom(defs: Dict[str, dict], models: Optional[Sequence[str]] = None,
                  counted: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    models = list(models or aerocom_model_names())
    seasons = seasons_of(defs)
    frames = []
    for i, model in enumerate(models, start=1):
        log.info("AeroCom %d/%d %s", i, len(models), model)
        ds = _aerocom_dataset(model)
        if ds is None:
            continue
        weights = pra.weights_from_defs(ds["od550aer"].isel(ens=0, time=0, drop=True), defs)
        columns = pra.regional_columns(ds, weights)
        times = pd.to_datetime(ds["time"].values)
        monthly = _frame_from_columns(
            columns, list(defs), "AeroCom", np.array([i]), [model], times,
        )
        frames.append(pra.seasonal_from_monthly(monthly, seasons=seasons))
        ds.close()
    if not frames:
        return pd.DataFrame()
    return _attach(pd.concat(frames, ignore_index=True), counted or defs)


def build(ppe_ids: Sequence[str] = ("pd2024", "pd2010")) -> None:
    for names in BASELINES.values():
        for name in names:
            box = perturb_box(name, 0, 0, 0)
            src = reg.REGIONS[name]
            if not np.allclose(box["lon_range"], src["lon_range"], atol=1e-6):
                raise RuntimeError(f"nominal lon drifted for {name}: {box['lon_range']}")
            lat = (min(src["lat_range"]), max(src["lat_range"]))
            if not np.allclose(box["lat_range"], lat, atol=1e-6):
                raise RuntimeError(f"nominal lat drifted for {name}: {box['lat_range']}")
    ppe_defs = sensitivity_defs(include_extra=False)
    aero_defs = sensitivity_defs(include_extra=True)
    log.info("PPE boxes: %d  AeroCom boxes: %d", len(ppe_defs), len(aero_defs))
    counted = cell_counts(aero_defs, ppe_ids[0])
    small = [n for n, c in counted.items() if c["n_cells"] < 8]
    if small:
        log.info("Boxes with <8 T63 cells: %s", small)
    ppe = build_ppe(ppe_defs, ppe_ids, counted=counted)
    aero = build_aerocom(aero_defs, counted=counted)
    PPE_SEASONAL.parent.mkdir(parents=True, exist_ok=True)
    AEROCOM_SEASONAL.parent.mkdir(parents=True, exist_ok=True)
    ppe.to_parquet(PPE_SEASONAL, index=False)
    aero.to_parquet(AEROCOM_SEASONAL, index=False)
    payload = {
        name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items() if k != "name"}
        for name, cfg in counted.items()
    }
    DEFS_JSON.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s (%d rows)", PPE_SEASONAL, len(ppe))
    log.info("Wrote %s (%d rows)", AEROCOM_SEASONAL, len(aero))
    log.info("Wrote %s", DEFS_JSON)


def load_defs() -> Dict[str, dict]:
    if DEFS_JSON.is_file():
        raw = json.loads(DEFS_JSON.read_text())
        for cfg in raw.values():
            cfg["lon_range"] = tuple(cfg["lon_range"])
            cfg["lat_range"] = tuple(cfg["lat_range"])
            cfg["season_months"] = tuple(cfg["season_months"])
        return raw
    return sensitivity_defs(include_extra=True)


def prepare(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Budget lifetimes and the species' MAC / AE aliases."""
    if df is None or df.empty:
        return df
    df = ppe_io.add_budget_columns(df, mask_outflow_lifetimes=False)
    df = ppe_io.apply_species_aliases(df, species)
    # PPE sea-salt AE is 550-865. Most AeroCom models only archive 870 nm;
    # use that pair where 865 nm is missing. The two pairs differ by ~1%.
    if species == "ss" and "AE_550_870" in df.columns and "AE" in df.columns:
        ae = df["AE"].astype(float)
        df["AE"] = ae.where(np.isfinite(ae), df["AE_550_870"].astype(float))
    return df


def ppe_frame(species: str, nominal_only: bool = False) -> pd.DataFrame:
    if not PPE_SEASONAL.is_file():
        raise FileNotFoundError(
            f"{PPE_SEASONAL} not built; run `python py/box_sensitivity.py`."
        )
    df = pd.read_parquet(PPE_SEASONAL)
    df = df.loc[df["baseline"].isin(BASELINES[species])]
    if nominal_only:
        df = df.loc[(df["dlon"] == 0) & (df["dlat"] == 0) & (df["dsize"] == 0)]
    df = prepare(df, species)
    df = df.rename(columns={"model": "member"})
    df["model"] = [ppe_io.member_key(p, m) for p, m in zip(df["ppe_id"], df["member"])]
    return df.reset_index(drop=True)


def aerocom_frame(species: Optional[str] = None,
                  regions: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not AEROCOM_SEASONAL.is_file():
        print(f"AeroCom sensitivity table missing: {AEROCOM_SEASONAL}")
        return pd.DataFrame()
    df = pd.read_parquet(AEROCOM_SEASONAL)
    if species is not None and regions is None:
        keep = set(BASELINES[species]) | set(AEROCOM_EXTRA[species])
        df = df.loc[df["baseline"].isin(keep)]
    df = prepare(df, species or "ss")
    if regions is not None:
        df = df.loc[df["region"].isin(list(regions))]
    return df.reset_index(drop=True)


def screen_lifetime(df: pd.DataFrame, columns: Sequence[str],
                    lo: float = 0.2, hi: float = 30.0) -> pd.DataFrame:
    """Set lifetimes outside [lo, hi] days to NaN (AeroCom outliers)."""
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        lt = df[col].astype(float)
        df[col] = lt.where((lt >= lo) & (lt <= hi))
    return df


# ---------------------------------------------------------------------------
# Map (same layout as cameo_toolbox.fake_uba_map, one colour per baseline)
# ---------------------------------------------------------------------------
_MAP_COLORS = ["#DBD6D6", "#9D579A", "#009BD5", "#5EAD35", "#FABB00", "#CE1F5E"]


def ensemble_mean_aod(ppe_id: str, variables: Sequence[str],
                      months: Optional[Sequence[int]] = None) -> xr.DataArray:
    """Ensemble- and time-mean species AOD from the PPE optical group file."""
    path = cfg.ensemble_file(ppe_id, "optical")
    ds = xr.open_dataset(path)
    parts = []
    for var in variables:
        if var not in ds:
            raise KeyError(f"{var} not in {path}")
        parts.append(ds[var])
    field = parts[0]
    for extra in parts[1:]:
        field = field + extra
    if months is not None:
        field = field.where(field["time"].dt.month.isin(list(months)), drop=True)
    mean = field.mean(dim=("ens", "time"), skipna=True).load()
    ds.close()
    mean.attrs["variables"] = "+".join(variables)
    mean.attrs["ppe_id"] = ppe_id
    return mean


def mask_to_surface(field: xr.DataArray, names: Sequence[str],
                    blank_outside: bool = False) -> xr.DataArray:
    """Keep values only on each box's land or ocean.

    With ``blank_outside``, cells outside the region mask are NaN as well.
    That is the regional sanity map. Global maps leave the field outside the
    boxes unchanged and should not call this.
    """
    import cameo_toolbox as ct
    import regions as reg

    out = field.copy(deep=True)
    data = np.array(out.values, dtype=float, copy=True)
    lon = np.mod(np.asarray(out.lon.values, dtype=float), 360.0)
    lat = np.asarray(out.lat.values, dtype=float)
    order = np.argsort(lon)
    lon = lon[order]
    data = data[:, order]
    lon2d, lat2d = np.meshgrid(lon, lat)
    template = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    defs = {name: reg.REGIONS[name] for name in names}
    masks = ct.masks_for_regions(template, defs)
    for name in names:
        inside = reg.box_mask(lon2d, lat2d, reg.REGIONS[name]["lon_range"], reg.REGIONS[name]["lat_range"])
        keep = np.asarray(masks[name].values, dtype=float) > 0.5
        if blank_outside:
            data[~keep] = np.nan
        else:
            data[inside & ~keep] = np.nan
        n_in = int(inside.sum())
        n_keep = int((inside & keep).sum())
        land = ct.point_is_land(lon2d[inside], lat2d[inside]) if n_in else np.array([])
        print(f"  {name:16} {reg.REGIONS[name]['surface_type']:5}  "
              f"box cells {n_in:4d}  kept {n_keep:4d}  "
              f"land in box {100 * land.mean():5.1f}%" if n_in else f"  {name}: empty")
    # restore original longitude order
    undo = np.empty_like(order)
    undo[order] = np.arange(order.size)
    out = out.copy(data=data[:, undo])
    return out


def plot_world_field(field: xr.DataArray, boxes: Sequence[dict], projection, title: str,
                     cbar_label: str, filename: Path,
                     extent: Optional[Tuple[float, float, float, float]] = None,
                     zmin: Optional[float] = None, zmax: Optional[float] = None):
    """Global map with region boxes in true geographic coordinates.

    ``projection`` is a Cartopy CRS (Mercator, Robinson, Orthographic, …).
    The field and the boxes both use Plate Carrée longitude/latitude, so the
    outlines stay on the coastlines.
    """
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    lon = np.mod(np.asarray(field.lon.values, dtype=float), 360.0)
    lat = np.asarray(field.lat.values, dtype=float)
    order = np.argsort(lon)
    lon = lon[order]
    data = np.asarray(field.values, dtype=float)[:, order]
    finite = data[np.isfinite(data)]
    if zmin is None:
        zmin = float(np.nanpercentile(finite, 2)) if finite.size else 0.0
    if zmax is None:
        zmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "daod", ["#DBD6D6", "#9D579A", "#009BD5", "#5EAD35", "#FABB00", "#CE1F5E"], N=12)
    norm = mcolors.BoundaryNorm(np.linspace(zmin, zmax, 13), ncolors=12, clip=True)
    fig = plt.figure(figsize=(12.5, 6.4))
    ax = plt.axes(projection=projection)
    mesh = ax.pcolormesh(
        ct.cell_edges(lon), ct.cell_edges(lat), data,
        transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="flat",
    )
    ax.coastlines(resolution="110m", zorder=4)
    try:
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.3, zorder=4)
    except Exception:
        pass
    for box in boxes:
        blat0, blat1 = (float(v) for v in box["lat_range"])
        ct.draw_box_outline(
            ax, box["lon_range"], box["lat_range"],
            color=box.get("color", "magenta"), lw=box.get("lw", 1.8),
        )
        if box.get("label"):
            west, east = ct._box_span(box["lon_range"])
            ax.text(
                0.5 * (west + east), max(blat0, blat1) + 1.2, box["label"],
                transform=ccrs.PlateCarree(), ha="center", va="bottom", fontsize=8,
                color=box.get("color", "magenta"), zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            )
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(list(extent), crs=ccrs.PlateCarree())
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.72, pad=0.02, extend="both")
    cbar.set_label(cbar_label)
    ax.set_title(title)
    fig.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {filename}")
    return filename


def plot_study_map(field: xr.DataArray, boxes: Sequence[dict], title: str, cbar_label: str,
                   filename: Path, extent: Optional[Tuple[float, float, float, float]] = None,
                   central_longitude: Optional[float] = None,
                   land_facecolor: Optional[str] = "0.85"):
    """Boxed maps use ``cameo_toolbox.plot_boxed_field`` (native cell edges + land).

    With no boxes, the curved ``uba_map_flex`` projection is used instead.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if boxes:
        ct.plot_boxed_field(
            field, boxes=boxes, title=title, cbar_label=cbar_label,
            savefile=str(filename), show=False, extent=extent,
            central_longitude=central_longitude, land_facecolor=land_facecolor,
        )
    else:
        ct.plot_curved_field(
            field, title=title, cbar_label=cbar_label,
            savefile=str(filename), show=False,
            extent=list(extent) if extent is not None else "global",
        )
    print(f"  Saved {filename}")
    return filename


def map_boxes(defs: Dict[str, dict], baselines: Sequence[str]) -> List[dict]:
    """Drawing records: nominal solid and labelled, shifts dashed, size changes dotted."""
    records = []
    for name, cfg in defs.items():
        if cfg["baseline"] not in baselines:
            continue
        nominal = cfg["dlon"] == 0 and cfg["dlat"] == 0 and cfg["dsize"] == 0
        size_only = cfg["dlon"] == 0 and cfg["dlat"] == 0 and cfg["dsize"] != 0
        records.append(dict(
            lon_range=tuple(cfg["lon_range"]),
            lat_range=tuple(cfg["lat_range"]),
            color=cfg.get("color", reg.REGIONS[cfg["baseline"]]["color"]),
            lw=2.4 if nominal else (1.6 if size_only else 0.9),
            ls="-" if nominal else (":" if size_only else "--"),
            label=cfg["baseline"] if nominal else None,
            label_below="box" in cfg["baseline"],
            z=3 if nominal else 2,
        ))
    records.sort(key=lambda r: r["z"])
    return records


def region_boxes(region_names: Sequence[str], *, nominal_only: bool = True) -> List[dict]:
    """Solid labelled outlines for registered analysis regions (0..360 lon)."""
    records = []
    for name in region_names:
        if name not in reg.REGIONS:
            continue
        cfg = reg.REGIONS[name]
        records.append(dict(
            lon_range=tuple(cfg["lon_range"]),
            lat_range=tuple(cfg["lat_range"]),
            color=cfg.get("color", "magenta"),
            lw=2.2,
            ls="-",
            label=name,
            label_below="box" in name,
            z=3,
        ))
    return records


def extent_for_regions(region_names: Sequence[str], pad: float = 12.0
                       ) -> Tuple[Optional[float], Tuple[float, float, float, float]]:
    """Return (central_longitude, geographic extent) covering the named boxes.

    Extent is true longitude/latitude for ``set_extent(..., crs=PlateCarree)``.
    A box that crosses 0°E is unwrapped (358..30 → -2..30). Boxes west of
    180°E are expressed in -180..180. ``central_longitude`` only centres the
    map projection; it is not subtracted from the coordinates.
    """
    boxes = region_boxes(region_names)
    if not boxes:
        return None, (-180.0, 180.0, -90.0, 90.0)
    clon = None
    for box in boxes:
        lo0, lo1 = box["lon_range"]
        if lo0 > lo1:
            clon = ct.box_central_longitude((lo0, lo1))
            if clon > 180:
                clon -= 360.0
            break
    if clon is None:
        mids = []
        for box in boxes:
            lo0, lo1 = box["lon_range"]
            if lo0 == 0 and lo1 == 360:
                continue
            mid = ct.box_central_longitude((lo0, lo1))
            mids.append(mid - 360.0 if mid > 180 else mid)
        clon = float(np.mean(mids)) if mids else 0.0

    wests, easts, souths, norths = [], [], [], []
    for box in boxes:
        lo0, lo1 = (float(v) for v in box["lon_range"])
        la0, la1 = (float(v) for v in box["lat_range"])
        if lo0 > lo1:
            west, east = lo0 - 360.0, lo1
        elif lo0 > 180.0:
            west, east = lo0 - 360.0, lo1 - 360.0
        else:
            west, east = lo0, lo1
        wests.append(west)
        easts.append(east)
        souths.append(min(la0, la1))
        norths.append(max(la0, la1))
    extent = (min(wests) - pad, max(easts) + pad, min(souths) - pad, max(norths) + pad)
    return clon, extent


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
