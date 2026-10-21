"""Shared PPE loaders and observation helpers for the analysis notebooks.

* ``load_regional``      monthly / seasonal regional tables for one or more
                         PPEs (``Data/PPE_regional``), with budget sums,
                         lifetimes and per-species column aliases added.
* ``open_gridded`` /     lazy access to the scratch ensemble NetCDFs and a
  ``sample_at_obs``      vectorised sampler (all members at once) used by the
                         Zhong-style homogenisation.
* ``load_spexone`` /     observation tables (SPEXone 2024-25, POLDER-GRASP 2010)
  ``load_polder``        with a common column layout.
* ``homogenize``         sampled = a * regional + b  ->  (obs - b) / a.

Members are keyed by ``model = "<ppe_id>:<member>"`` so tables from several
PPEs can be concatenated without collisions; ``member`` keeps the bare
ensemble label ("Control", "1", ...).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

PY_DIR = Path(__file__).resolve().parent
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

import PPE_set as cfg  # noqa: E402
import regions as reg  # noqa: E402

PROJECT_ROOT = cfg.PROJECT_ROOT
SPEXONE_GLOB = str(
    PROJECT_ROOT / "Data" / "SPEXone_regridded" / "SPEXone_L2_regridded_{year}_{month:02d}_v4_0.parquet"
)
POLDER_PATH = (
    PROJECT_ROOT / "Data" / "AP3_POLDER_Collocated"
    / "POLDER_GRASP_coloc_3h_AP3_2010_lon0_10.0_lat0_10.0.parquet"
)

QUALITY_FILTER_PARAMS = {
    "min_aod": 0.01,
    "min_aaod": 1e-5,
    "max_aaod_fraction": 0.95,
    "min_ssa": 0.5,
    "max_ssa": 1.02,
}

# ---------------------------------------------------------------------------
# Budget sums and lifetimes (formed after aggregation; regional means are linear)
# ---------------------------------------------------------------------------
M_S, M_SO2, M_SO4, M_DMS = 32.06, 64.07, 96.06, 62.13
# Fraction of emitted SO2/DMS sulfur that ends up as sulfate (no SO4 production
# diagnostics in the PPE); see AAOD_error_attribution_PPE potential issues.
SO2_TO_SO4_FRAC = 0.5

PRIM_LOAD = ("loadbc", "loadoa", "loaddust", "loadss")
PRIM_EMI = ("emibc", "emioa", "emidust", "emiss")
AER_LOAD = PRIM_LOAD + ("loadso4",)
GAS_LOAD = ("burden_SO2", "burden_DMS")

# name -> (burden weights, emission weights, label)
LIFETIME_DEFS: Dict[str, Tuple[Dict[str, float], Dict[str, float], str]] = {
    "lifetime_BC_OA": ({"loadbc": 1.0, "loadoa": 1.0}, {"emibc": 1.0, "emioa": 1.0}, "BC+OA"),
    "lifetime_ss": ({"loadss": 1.0}, {"emiss": 1.0}, "sea salt"),
    "lifetime_du": ({"loaddust": 1.0}, {"emidust": 1.0}, "mineral dust"),
    "lifetime_prim": ({v: 1.0 for v in PRIM_LOAD}, {v: 1.0 for v in PRIM_EMI},
                      "primary aerosol (BC+OA+DU+SS)"),
    # Aerosol burden over primary emission + sulfate-equivalent SO2/DMS source.
    "lifetime": (
        {v: 1.0 for v in AER_LOAD},
        {**{v: 1.0 for v in PRIM_EMI}, "emi_SO4": 1.0,
         "emiso2": SO2_TO_SO4_FRAC * M_SO4 / M_SO2,
         "emi_DMS": SO2_TO_SO4_FRAC * M_SO4 / M_DMS},
        "total aerosol (SO4-eq. source)",
    ),
    # Plain mass sum used by the AeroCom notebooks (load_total / emi_total).
    "lifetime_mass": ({v: 1.0 for v in reg.TOTAL_LOAD}, {v: 1.0 for v in reg.TOTAL_EMI},
                      "total aerosol (plain mass sum, AeroCom style)"),
    "lifetime_SO4": (
        {"loadso4": 1.0},
        {"emi_SO4": 1.0, "emiso2": SO2_TO_SO4_FRAC * M_SO4 / M_SO2,
         "emi_DMS": SO2_TO_SO4_FRAC * M_SO4 / M_DMS},
        "SO4 (SO4-eq. source)",
    ),
    "lifetime_S": (
        {"loadso4": M_S / M_SO4, "burden_SO2": M_S / M_SO2, "burden_DMS": M_S / M_DMS},
        {"emi_SO4": M_S / M_SO4, "emiso2": M_S / M_SO2, "emi_DMS": M_S / M_DMS},
        "sulfur family (S mass)",
    ),
    "lifetime_naive": (
        {v: 1.0 for v in AER_LOAD + GAS_LOAD},
        {v: 1.0 for v in PRIM_EMI + ("emi_SO4", "emiso2", "emi_DMS")},
        "naive mass sum (diagnostic)",
    ),
}


def lifetime_load_key(name: str) -> str:
    """Column holding the burden sum of a lifetime definition."""
    special = {"lifetime_BC_OA": "load_BC_OA", "lifetime_mass": "load_total",
               "lifetime_ss": "load_ss", "lifetime_du": "load_du"}
    return special.get(name, name.replace("lifetime", "load_lt"))


def lifetime_emi_key(name: str) -> str:
    special = {"lifetime_BC_OA": "emi_BC_OA", "lifetime_mass": "emi_total",
               "lifetime_ss": "emi_ss", "lifetime_du": "emi_du"}
    return special.get(name, name.replace("lifetime", "emi_lt"))


def _weighted_sum(df: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    if any(k not in df.columns for k in weights):
        return pd.Series(np.nan, index=df.index)
    total = None
    for k, w in weights.items():
        term = df[k].astype(float) * float(w)
        total = term if total is None else total + term
    return total


def add_budget_columns(df: pd.DataFrame, mask_outflow_lifetimes: bool = True) -> pd.DataFrame:
    """Add load/emission sums and emission lifetimes (days) for LIFETIME_DEFS.

    Outflow-role regions get NaN lifetimes (no local source; as in the
    AeroCom notebooks the outflow AOD/AAOD is predicted from the source box).
    """
    df = df.copy()
    for name, (load_w, emi_w, _label) in LIFETIME_DEFS.items():
        lk, ek = lifetime_load_key(name), lifetime_emi_key(name)
        df[lk] = _weighted_sum(df, load_w)
        df[ek] = _weighted_sum(df, emi_w)
        with np.errstate(divide="ignore", invalid="ignore"):
            lt = df[lk] / (df[ek] * 86400.0)
        df[name] = lt.where(np.isfinite(lt) & (lt > 0))
    if mask_outflow_lifetimes and "region" in df.columns:
        outflow = df["region"].map(lambda r: reg.REGIONS.get(r, {}).get("role") == "outflow")
        df.loc[outflow.fillna(False).astype(bool), list(LIFETIME_DEFS)] = np.nan
    return df


def apply_species_aliases(df: pd.DataFrame, species: Optional[str]) -> pd.DataFrame:
    """Add notebook column names (e.g. "MAC", "AE") for one species."""
    if species is None:
        return df
    df = df.copy()
    for alias, col in reg.SPECIES[species]["aliases"].items():
        df[alias] = df[col] if col in df.columns else np.nan
    return df


# ---------------------------------------------------------------------------
# Regional tables
# ---------------------------------------------------------------------------
def member_key(ppe_id: str, member: str) -> str:
    return f"{ppe_id}:{member}"


def load_regional(
    ppe_ids: Optional[Sequence[str]] = None,
    species: Optional[str] = None,
    regions: Optional[Sequence[str]] = None,
    seasonal: bool = True,
    exclude_members: Sequence[str] = (),
) -> pd.DataFrame:
    """Regional PPE table with budget sums, lifetimes and species aliases.

    ``exclude_members`` accepts bare labels ("Control") or ppe-scoped keys
    ("pd2010:Control"). Monthly tables gain a ``month`` Period column.
    """
    path = cfg.REGIONAL_SEASONAL if seasonal else cfg.REGIONAL_MONTHLY
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not built; run `python py/ppe_regional_aggregates.py`."
        )
    df = pd.read_parquet(path)
    if ppe_ids is not None:
        missing = sorted(set(ppe_ids) - set(df["ppe_id"].unique()))
        if missing:
            print(f"  Warning: PPE(s) not in {path.name}: {missing}")
        df = df.loc[df["ppe_id"].isin(ppe_ids)]
    if regions is None and species is not None:
        regions = reg.regions_for_species(species)
    if regions is not None:
        df = df.loc[df["region"].isin(regions)]
    df = df.rename(columns={"model": "member"})
    df["model"] = [member_key(p, m) for p, m in zip(df["ppe_id"], df["member"])]
    if exclude_members:
        ex = set(exclude_members)
        df = df.loc[~(df["member"].isin(ex) | df["model"].isin(ex))]
    if not seasonal:
        df["month"] = pd.PeriodIndex(pd.to_datetime(df["time"]), freq="M")
    df = add_budget_columns(df)
    df = apply_species_aliases(df, species)
    return df.reset_index(drop=True)


def has_precip(df: pd.DataFrame, ppe_id: str) -> bool:
    sub = df.loc[df["ppe_id"] == ppe_id]
    return "precip" in sub.columns and bool(np.isfinite(sub["precip"].astype(float)).any())


def ppe_style(ppe_id: str) -> dict:
    run = cfg.get_run(ppe_id)
    return {"color": run["color"], "marker": run["marker"], "label": run["label"]}


def ppe_obs_name(ppe_id: str) -> str:
    return cfg.get_run(ppe_id)["obs"]


def ppe_year_months(ppe_id: str) -> List[Tuple[int, int]]:
    return list(cfg.get_run(ppe_id)["year_months"])


# ---------------------------------------------------------------------------
# Gridded ensemble access (scratch)
# ---------------------------------------------------------------------------
def open_gridded(ppe_id: str, variables: Iterable[str]) -> xr.Dataset:
    """Lazily open the scratch ensemble files holding ``variables``.

    Aerosol loads are multiplied by the run's ``burden_scale``.
    """
    variables = list(variables)
    parts = []
    for group, names in cfg.GROUP_VARS.items():
        want = [v for v in variables if v in names]
        if not want:
            continue
        path = cfg.ensemble_file(ppe_id, group)
        if not path.is_file():
            continue
        ds = xr.open_dataset(path, engine=cfg.XR_WRITE_ENGINE)
        parts.append(ds[[v for v in want if v in ds.data_vars]])
    if not parts:
        raise FileNotFoundError(f"{ppe_id}: none of {variables} found on scratch")
    ds = xr.merge(parts, compat="override", join="outer", combine_attrs="drop_conflicts")
    for v in cfg.LOAD_SCALE_VARS:
        scale = cfg.burden_scale_for(ppe_id, v)
        if v in ds and scale not in (None, 1.0):
            ds[v] = ds[v] * scale
    return ds


def apply_aerosol_quality_filter(aod, aaod=None, ssa=None, min_aod=0.01, min_aaod=1e-5,
                                 max_aaod_fraction=0.95, min_ssa=0.5, max_ssa=1.02):
    """Mask AOD / AAOD / SSA (xarray or pandas) with the notebook quality filter."""
    def _where(var, cond):
        if var is None:
            return None
        if hasattr(var, "where"):
            return var.where(cond)
        out = var.copy()
        out[~np.asarray(cond)] = np.nan
        return out

    aod_mask = (aod > min_aod) & np.isfinite(aod)
    abs_mask = aod_mask
    if aaod is not None:
        abs_mask = (abs_mask & (aaod > min_aaod) & (aaod < aod * max_aaod_fraction)
                    & np.isfinite(aaod))
    if ssa is not None:
        abs_mask = abs_mask & (ssa >= min_ssa) & (ssa <= max_ssa) & np.isfinite(ssa)
    aod_out = _where(aod, aod_mask)
    aaod_out = _where(aaod, abs_mask)
    ssa_out = _where(ssa, abs_mask)
    if ssa_out is not None:
        ssa_out = ssa_out.clip(0.0, 1.0)
    return aod_out, aaod_out, ssa_out


# obs-matching quantity -> raw variables needed and point formula
_SAMPLE_RECIPES = {
    "AOD": ("od550aer",),
    "AAOD": ("od550aer", "abs550aer"),
    "SSA": ("od550aer", "abs550aer"),
    "AE": ("od550aer", "od440aer"),
    "AOD_du": ("od550dust",),
}


def _point_values(ds: xr.Dataset, quantity: str) -> xr.DataArray:
    if quantity == "AOD":
        return ds["od550aer"]
    if quantity in ("AAOD", "SSA"):
        _, aaod_f, ssa = apply_aerosol_quality_filter(
            ds["od550aer"], ds["abs550aer"], 1.0 - ds["abs550aer"] / ds["od550aer"],
            **QUALITY_FILTER_PARAMS,
        )
        return aaod_f if quantity == "AAOD" else ssa
    if quantity == "AE":
        aod = ds["od550aer"].where(ds["od550aer"] > QUALITY_FILTER_PARAMS["min_aod"])
        return -np.log(aod / ds["od440aer"]) / np.log(550.0 / 440.0)
    if quantity == "AOD_du":
        return ds["od550dust"]
    raise KeyError(quantity)


# (ppe_id, var) -> loaded (ens, time, lat, lon) field; ~0.2-0.3 GB each.
_GRID_CACHE: Dict[Tuple[str, str], xr.DataArray] = {}


def cached_gridded(ppe_id: str, variables: Iterable[str]) -> xr.Dataset:
    """Like ``open_gridded`` but loads each variable once per session."""
    variables = list(variables)
    missing = [v for v in variables if (ppe_id, v) not in _GRID_CACHE]
    if missing:
        with open_gridded(ppe_id, missing) as ds:
            for v in missing:
                _GRID_CACHE[(ppe_id, v)] = ds[v].drop_vars("experiment", errors="ignore").load()
    return xr.Dataset({v: _GRID_CACHE[(ppe_id, v)] for v in variables})


def clear_gridded_cache() -> None:
    _GRID_CACHE.clear()


def sample_at_obs(ppe_id: str, obs: pd.DataFrame, quantity: str,
                  max_points: int = 20000) -> pd.Series:
    """Cos-lat weighted mean of the gridded PPE field at observation points.

    ``obs`` needs ``latitude``, ``lon360`` and ``date``. Returns a Series
    indexed by ppe-scoped member key. All members are sampled in one
    vectorised ``isel`` on fields cached by ``cached_gridded``.
    """
    if obs.empty:
        return pd.Series(dtype=float)
    sub = obs.dropna(subset=["latitude", "lon360", "date"])
    if len(sub) > max_points:
        sub = sub.sample(n=max_points, random_state=0)
    ds = cached_gridded(ppe_id, _SAMPLE_RECIPES[quantity])
    try:
        lat = ds["lat"].values
        lon = ds["lon"].values
        times = pd.to_datetime(ds["time"].values).to_period("M")
        month_lookup = {p: i for i, p in enumerate(times)}
        obs_months = pd.to_datetime(sub["date"].values).to_period("M")
        t_idx = np.array([month_lookup.get(p, -1) for p in obs_months])
        ok = t_idx >= 0
        if ok.sum() < 3:
            return pd.Series(dtype=float)
        sub = sub.loc[ok]
        t_idx = t_idx[ok]
        y_idx = np.abs(lat[:, None] - sub["latitude"].to_numpy()[None, :]).argmin(axis=0)
        x_idx = np.abs(lon[:, None] - sub["lon360"].to_numpy()[None, :]).argmin(axis=0)
        pts = xr.Dataset({
            "t": ("pt", t_idx), "y": ("pt", y_idx), "x": ("pt", x_idx),
        })
        sel = ds.isel(time=pts["t"], lat=pts["y"], lon=pts["x"]).load()
        vals = np.asarray(_point_values(sel, quantity).transpose("ens", "pt").values, float)
        w = np.cos(np.deg2rad(sub["latitude"].to_numpy(float)))[None, :]
        finite = np.isfinite(vals)
        num = np.where(finite, vals * w, 0.0).sum(axis=1)
        den = np.where(finite, w, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(finite.sum(axis=1) >= 3, num / den, np.nan)
        labels = ["Control" if int(e) < 0 else str(int(e)) for e in ds["ens"].values]
        return pd.Series(mean, index=[member_key(ppe_id, m) for m in labels])
    finally:
        del ds


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
def _finish_obs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    aod_f, aaod_f, ssa_f = apply_aerosol_quality_filter(
        df["AOD_550"], df["AAOD_550"], df["SSA"], **QUALITY_FILTER_PARAMS
    )
    df["AOD_550"], df["AAOD_550"], df["SSA"] = aod_f, aaod_f, ssa_f
    return df


def load_spexone(year_months: Sequence[Tuple[int, int]], ae_pair: str = "440_550",
                 wvl: str = "550nm") -> pd.DataFrame:
    """SPEXone L2 regridded (1 deg, daily) with AOD/AAOD/SSA/AE columns.

    ``ae_pair`` "440_550" (AAOD convention) or "550_865" (sea salt).
    """
    files = [Path(SPEXONE_GLOB.format(year=y, month=m)) for y, m in year_months]
    present = [f for f in files if f.is_file()]
    for f in files:
        if not f.is_file():
            print(f"  SPEXone missing: {f.name}")
    if not present:
        raise FileNotFoundError("No SPEXone monthly parquet files for the requested months.")
    df = pd.concat([pd.read_parquet(f) for f in present], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["month"] = df["date"].dt.to_period("M")
    df["lon360"] = (df["lon_bin"].astype(float) + 360.0) % 360.0
    df["latitude"] = df["lat_bin"].astype(float)
    df["AOD_550"] = df.get(f"aod_{wvl}", np.nan)
    df["AAOD_550"] = df.get(f"aaod_{wvl}", np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["SSA"] = df[f"ssa_{wvl}"] if f"ssa_{wvl}" in df.columns else 1.0 - df["AAOD_550"] / df["AOD_550"]
        if ae_pair == "550_865" and "aod_865nm" in df.columns:
            df["AE"] = -np.log(df["AOD_550"] / df["aod_865nm"]) / np.log(550.0 / 865.0)
        elif "aod_440nm" in df.columns:
            df["AE"] = -np.log(df["AOD_550"] / df["aod_440nm"]) / np.log(550.0 / 440.0)
        else:
            df["AE"] = np.nan
    df = _finish_obs(df)
    print(f"SPEXone rows: {len(df):,}  valid AOD+AAOD+SSA: "
          f"{df[['AOD_550', 'AAOD_550', 'SSA']].notna().all(axis=1).sum():,}")
    return df


def load_polder(ae_pair: str = "440_550") -> pd.DataFrame:
    """POLDER-GRASP 2010 (3-hourly collocation file) in the SPEXone layout."""
    cols = ["time", "longitude", "latitude", "AOD_550", "AAOD_550", "AOD_440", "AOD_870"]
    df = pd.read_parquet(POLDER_PATH, columns=cols)
    df["date"] = pd.to_datetime(df["time"])
    df["month"] = df["date"].dt.to_period("M")
    df["lon360"] = df["longitude"].astype(float) % 360.0
    df["latitude"] = df["latitude"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["SSA"] = 1.0 - df["AAOD_550"] / df["AOD_550"]
        if ae_pair == "550_865":
            df["AE"] = -np.log(df["AOD_550"] / df["AOD_870"]) / np.log(550.0 / 870.0)
        else:
            df["AE"] = -np.log(df["AOD_550"] / df["AOD_440"]) / np.log(550.0 / 440.0)
    df = _finish_obs(df)
    print(f"POLDER rows: {len(df):,}")
    return df


def load_obs_for_ppe(ppe_id: str, ae_pair: str = "440_550") -> pd.DataFrame:
    """Observation table matching a PPE's period (SPEXone for 2024-25, POLDER for 2010)."""
    obs = ppe_obs_name(ppe_id)
    if obs == "SPEXone":
        return load_spexone(ppe_year_months(ppe_id), ae_pair=ae_pair)
    if obs == "POLDER":
        return load_polder(ae_pair=ae_pair)
    raise KeyError(obs)


def obs_in_region(df: pd.DataFrame, region: str,
                  period: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
    """Rows inside a region box, its season months and (optionally) a period."""
    if df.empty:
        return df
    inside = reg.in_box(df["lon360"].to_numpy(), df["latitude"].to_numpy(), region)
    months = list(reg.REGIONS[region]["season_months"])
    sub = df.loc[inside & df["date"].dt.month.isin(months).to_numpy()]
    if period is not None:
        t0, t1 = period
        sub = sub.loc[(sub["date"] >= pd.Timestamp(t0))
                      & (sub["date"] <= pd.Timestamp(t1) + pd.Timedelta(days=1))]
    return sub


def weighted_mean(sub: pd.DataFrame, col: str, min_n: int = 3) -> float:
    sub = sub.dropna(subset=[col, "latitude"])
    if len(sub) < min_n:
        return np.nan
    w = np.cos(np.deg2rad(sub["latitude"].to_numpy(dtype=float)))
    return float(np.average(sub[col].to_numpy(dtype=float), weights=w))


def obs_region_means(obs: pd.DataFrame, region_names: Sequence[str],
                     period: Optional[Tuple[str, str]] = None) -> Dict[str, dict]:
    out = {}
    for region in region_names:
        sub = obs_in_region(obs, region, period)
        out[region] = {c: weighted_mean(sub, c) for c in ("AOD_550", "AAOD_550", "SSA", "AE")}
        out[region]["n"] = int(len(sub))
    return out


# ---------------------------------------------------------------------------
# Homogenisation
# ---------------------------------------------------------------------------
def homogenize(regional: pd.Series, sampled: pd.Series, obs_sampled: float,
               min_n: int = 3) -> dict:
    """Fit sampled = a * regional + b over members; return (obs - b) / a."""
    common = regional.index.intersection(sampled.index)
    x = regional.reindex(common).astype(float)
    y = sampled.reindex(common).astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    out = {"raw_mean": obs_sampled, "n_models": int(len(x)), "a": np.nan, "b": np.nan,
           "r2": np.nan, "homogenized": obs_sampled, "skipped": True, "x": x, "y": y}
    if len(x) < min_n or not np.isfinite(obs_sampled):
        return out
    coef, *_ = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - (a * x + b)) ** 2) / ss_tot if ss_tot > 0 else np.nan
    out.update(a=a, b=b, r2=r2, skipped=False,
               homogenized=(obs_sampled - b) / a if a != 0 else np.nan)
    return out


def add_lifetime_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Columns for marine (sea salt + sulfate) and BC+OA lifetime covariates.

    Marine source uses the same SO2/DMS yield as ``LIFETIME_DEFS['lifetime_SO4']``.
    Deposition lifetimes are filled only when those fluxes are present (AeroCom);
    the PPE extract does not include them.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    w_so2 = SO2_TO_SO4_FRAC * M_SO4 / M_SO2
    w_dms = SO2_TO_SO4_FRAC * M_SO4 / M_DMS

    def _col(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df[name].astype(float)

    so4_src = _col("emi_SO4") + w_so2 * _col("emiso2") + w_dms * _col("emi_DMS")
    df["load_marine"] = _col("loadss") + _col("loadso4")
    df["emi_marine"] = _col("emiss") + so4_src
    with np.errstate(divide="ignore", invalid="ignore"):
        df["lifetime_marine"] = df["load_marine"] / (df["emi_marine"] * 86400.0)
        df["ss_burden_frac"] = _col("loadss") / df["load_marine"]
        df["dms_source_frac"] = (w_dms * _col("emi_DMS")) / so4_src
        df["so2_source_frac"] = (w_so2 * _col("emiso2")) / so4_src
        dep = _col("depss") + _col("depso4")
        df["lifetime_marine_dep"] = df["load_marine"] / (dep * 86400.0)
        if {"depbc", "depoa"} <= set(df.columns) and "load_BC_OA" in df.columns:
            df["lifetime_BC_OA_dep"] = _col("load_BC_OA") / ((_col("depbc") + _col("depoa")) * 86400.0)
    for name in ("lifetime_marine", "lifetime_marine_dep", "lifetime_BC_OA_dep"):
        if name in df.columns:
            df[name] = df[name].where(np.isfinite(df[name]) & (df[name] > 0))
    return df


def lookup_obs_value(homog_df, obs_means, pid: str, region: str, var: str,
                     source_regions: Optional[Sequence[str]] = None) -> float:
    """Homogenized observation when the fit was kept, otherwise the sampled mean.

    ``source_regions``, when set, limits the homogenized value to those regions
    (the AAOD notebook does not homogenize the outflow box).
    """
    col = {"AOD": "AOD_550", "AAOD": "AAOD_550"}.get(var, var)
    use_homog = homog_df is not None and not getattr(homog_df, "empty", True)
    if source_regions is not None and region not in source_regions:
        use_homog = False
    if use_homog:
        hit = homog_df.loc[
            (homog_df["ppe_id"] == pid) & (homog_df["region"] == region)
            & (homog_df["variable"] == var) & ~homog_df["skipped"]
        ]
        if not hit.empty:
            return float(hit["homogenized"].iloc[0])
    return float(obs_means.get(pid, {}).get(region, {}).get(col, np.nan))


def obs_lines_for(homog_df, obs_means, var: str, regions: Sequence[str],
                  transform=lambda v: v,
                  source_regions: Optional[Sequence[str]] = None) -> Dict:
    """``(ppe_id, region) -> (value, observation name)`` over ``regions``.

    Values come from ``lookup_obs_value``. Pass ``source_regions`` when
    homogenization applies only to a subset (biomass-burning sources).
    """
    return {
        (pid, r): (
            transform(lookup_obs_value(homog_df, obs_means, pid, r, var, source_regions)),
            ppe_obs_name(pid),
        )
        for pid in obs_means for r in regions
    }


def polder_region_mask(df: pd.DataFrame, lon_range, lat_range) -> pd.Series:
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    if lon_min > lon_max:
        mask_lon = (df["longitude"] >= lon_min) | (df["longitude"] <= lon_max)
    else:
        mask_lon = (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
    mask_lat = (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max)
    return mask_lon & mask_lat


def polder_monthly_means(df: pd.DataFrame, region_name: str,
                         regions: Mapping[str, dict]) -> pd.DataFrame:
    """Area-weighted monthly regional means from a POLDER table."""
    cfg = regions[region_name]
    sub = df.loc[polder_region_mask(df, cfg["lon_range"], cfg["lat_range"])].copy()
    if "time_slice" in cfg:
        t0, t1 = cfg["time_slice"]
        sub = sub[(sub["time"] >= t0) & (sub["time"] <= t1)]
    if sub.empty:
        return pd.DataFrame()
    sub["coslat"] = np.cos(np.deg2rad(sub["latitude"]))
    rows = []
    for month, g in sub.groupby("month"):
        if len(g) < 3:
            continue
        w = g["coslat"]
        rows.append({
            "month": month,
            "time": pd.Timestamp(month.start_time),
            "SSA": np.average(g["SSA"], weights=w),
            "AE": np.average(g["AE"], weights=w),
            "AAOD_550": np.average(g["AAOD_550"], weights=w),
            "AOD_550": np.average(g["AOD_550"], weights=w),
            "n": len(g),
        })
    return pd.DataFrame(rows)


def gpcp_region_mean(precip_da, region_name: str, regions: Mapping[str, dict],
                     surface_mask=None, apply_surface_mask: bool = False):
    """Area-weighted GPCP mean. Optional land/ocean mask matches the model box."""
    cfg = regions[region_name]
    lon_min, lon_max = cfg["lon_range"]
    lat_min, lat_max = cfg["lat_range"]
    if lon_min > lon_max:
        sub = xr.concat([
            precip_da.sel(lon=slice(lon_min, 360)),
            precip_da.sel(lon=slice(0, lon_max)),
        ], dim="lon")
    else:
        sub = precip_da.sel(lon=slice(lon_min, lon_max))
    if sub.lat.values[0] > sub.lat.values[-1]:
        sub = sub.sel(lat=slice(lat_max, lat_min))
    else:
        sub = sub.sel(lat=slice(lat_min, lat_max))
    if "time_slice" in cfg:
        sub = sub.sel(time=slice(*cfg["time_slice"]))
    if (apply_surface_mask and surface_mask is not None
            and cfg.get("surface_type") in ("land", "ocean")):
        try:
            mask_on = surface_mask.interp(lat=sub.lat, lon=sub.lon, method="nearest")
            sub = sub.where(mask_on > 0)
        except Exception as exc:
            print(f"  GPCP surface mask skip {region_name}: {exc}")
    coslat = np.cos(np.deg2rad(sub.lat))
    weights = coslat * xr.ones_like(sub.isel(time=0))
    weights = weights.where(np.isfinite(sub.isel(time=0)))
    wsum = weights.sum()
    if float(wsum) == 0:
        return sub.mean(dim=["lat", "lon"])
    weights = weights / wsum
    return (sub * weights).sum(dim=["lat", "lon"])
