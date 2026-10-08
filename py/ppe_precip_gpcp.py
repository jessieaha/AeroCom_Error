"""PPE / AeroCom precipitation vs GPCP for the AAOD fire-season regions.

Shared by ``notebooks/AAOD_error_attribution_PPE.py`` and the light
precipitation-only runner ``notebooks/PPE_precip_vs_GPCP.py``.

Missing GPCP months are never filled. Every model–GPCP comparison is
restricted to the months where GPCP exists (``precip_gpcp_matched``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr

import cameo_toolbox as ct

GPCP_VAR = "sat_gauge_precip"  # mm day-1
GPCP_FILE = "GPCPMON_L3_{year}{month:02d}_V3.3.nc4"
PRECIP_LABEL = r"Precipitation (mm day$^{-1}$)"
MATCHED_COL = "precip_gpcp_matched"

AEROCOM_EXCLUDE_MODELS = [  # same list as AAOD_error_attribution.ipynb
    "GEOS-i33p2-met2010_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-OMA_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-MATRIX_AP3-CTRL",
    "NorESM2-met2010_AP3-CTRL-v3",
]

# outflow_af and africa boxes share an edge, so the outflow label goes below.
MAP_DISPLAY_NAMES = {"outflow_af": "outflow"}
MAP_LABEL_BELOW = {"outflow"}
SCATTER_PANELS = {
    "amazon": ("(a)", "Amazon"),
    "africa": ("(b)", "Africa"),
    "outflow_af": ("(c)", "African outflow"),
}

SaveFig = Callable[[plt.Figure, str], object]


def month_key(t) -> str:
    t = pd.Timestamp(t)
    return f"{t.year}-{t.month:02d}"


# ---------------------------------------------------------------------------
# GPCP loading / regional means
# ---------------------------------------------------------------------------
def load_gpcp_months(
    gpcp_dir: Path, year_months: Iterable[Tuple[int, int]]
) -> Tuple[Optional[xr.DataArray], List[str]]:
    """Stack available GPCP monthly files on a 0–360 grid; return (da, missing)."""
    das, missing = [], []
    for year, month in year_months:
        path = Path(gpcp_dir) / GPCP_FILE.format(year=year, month=month)
        if not path.is_file():
            missing.append(f"{year}-{month:02d}")
            continue
        with xr.open_dataset(path) as ds:
            da = ds[GPCP_VAR].isel(time=0, drop=True).load()
        da = da.assign_coords(lon=(da.lon + 360.0) % 360.0).sortby("lon").sortby("lat")
        das.append(da.expand_dims(time=[pd.Timestamp(year, month, 1)]))
    if not das:
        return None, missing
    return xr.concat(das, dim="time"), missing


def _box_lon_mask(lon: xr.DataArray, lon_range) -> xr.DataArray:
    lon0, lon1 = lon_range
    if lon0 <= lon1:
        return (lon >= lon0) & (lon <= lon1)
    return (lon >= lon0) | (lon <= lon1)


def gpcp_region_subset(da: xr.DataArray, cfg: dict) -> xr.DataArray:
    lat0, lat1 = cfg["lat_range"]
    sub = da.sel(lat=slice(lat0, lat1))
    return sub.where(_box_lon_mask(sub.lon, cfg["lon_range"]), drop=True)


def gpcp_region_monthly(da: xr.DataArray, cfg: dict) -> pd.Series:
    sub = gpcp_region_subset(da, cfg)
    return sub.weighted(np.cos(np.deg2rad(sub.lat))).mean(("lat", "lon")).to_series()


def gpcp_region_tables(
    da: Optional[xr.DataArray], regions: Dict[str, dict], year_months
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, Set[str]]]:
    """Fire-season GPCP means over available months only.

    Returns (gpcp_obs, monthly_rows, matched_months) where matched_months[region]
    is the set of 'YYYY-MM' fire-season months that have GPCP data.
    """
    gpcp_obs: Dict[str, float] = {}
    matched: Dict[str, Set[str]] = {}
    rows = []
    all_keys = [f"{y}-{m:02d}" for y, m in year_months]
    for region, cfg in regions.items():
        months = set(cfg["season_months"])
        ts = gpcp_region_monthly(da, cfg) if da is not None else pd.Series(dtype=float)
        have = {month_key(t): float(v) for t, v in ts.items()}
        for key in all_keys:
            rows.append({
                "region": region, "month": key, "in_season": int(key[-2:]) in months,
                "gpcp_available": key in have, "precip_obs": have.get(key, np.nan),
            })
        matched[region] = {k for k in have if int(k[-2:]) in months}
        vals = [have[k] for k in sorted(matched[region])]
        gpcp_obs[region] = float(np.mean(vals)) if vals else np.nan
    return gpcp_obs, pd.DataFrame(rows), matched


def matched_model_precip(
    monthly_df: pd.DataFrame, matched: Dict[str, Set[str]], precip_col: str = "precip"
) -> pd.DataFrame:
    """Per region/model mean of monthly model precip over GPCP-available season months."""
    work = monthly_df[["region", "model", "month", precip_col]].copy()
    work["month"] = work["month"].astype(str)
    keep = np.zeros(len(work), dtype=bool)
    for region, months in matched.items():
        keep |= (work["region"] == region).to_numpy() & work["month"].isin(months).to_numpy()
    out = (work.loc[keep].groupby(["region", "model"])[precip_col]
           .agg(["mean", "count"]).reset_index()
           .rename(columns={"mean": MATCHED_COL, "count": "n_months_matched"}))
    out["model"] = out["model"].astype(str)
    return out


def precip_check_table(
    matched_df: pd.DataFrame, gpcp_obs: Dict[str, float], matched: Dict[str, Set[str]]
) -> pd.DataFrame:
    rows = []
    for region, obs in gpcp_obs.items():
        s = matched_df.loc[matched_df["region"] == region, MATCHED_COL].dropna()
        if s.empty:
            continue
        q = s.quantile([0.05, 0.5, 0.95])
        rows.append({
            "region": region, "months": ",".join(sorted(matched.get(region, []))),
            "precip_obs": obs, "ppe_q05": q.loc[0.05], "ppe_median": q.loc[0.5],
            "ppe_q95": q.loc[0.95], "ppe_min": s.min(), "ppe_max": s.max(),
            "pct_members_drier": float((s < obs).mean() * 100) if np.isfinite(obs) else np.nan,
        })
    return pd.DataFrame(rows)


def load_aerocom_precip(
    parquet: Path, regions: Iterable[str], exclude: Iterable[str] = AEROCOM_EXCLUDE_MODELS
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """AeroCom (2010) regional fire-season precip per model and GPCP 2010 from the cache."""
    parquet = Path(parquet)
    if not parquet.is_file():
        return pd.DataFrame(columns=["region", "model", "precip"]), {}
    long = pd.read_parquet(parquet)
    long = long[long["region"].isin(list(regions)) & (long["var"] == "precip")]
    gpcp_2010 = {
        r: float(v) for r, v in
        long.loc[long["dataset"] == "GPCP", ["region", "value"]].itertuples(index=False)
    }
    mod = long[~long["dataset"].isin(["POLDER", "GPCP"] + list(exclude))
               & (long["source"] == "aggregated")]
    df = mod.rename(columns={"dataset": "model", "value": "precip"})[["region", "model", "precip"]]
    return df.reset_index(drop=True), gpcp_2010


# ---------------------------------------------------------------------------
# Map helpers (AOD_error_attribution style via ct.fake_uba_map)
# ---------------------------------------------------------------------------
def _lon180(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def region_boxes_180(regions: Dict[str, dict]) -> Dict[str, tuple]:
    return {
        MAP_DISPLAY_NAMES.get(name, name): (
            _lon180(cfg["lon_range"][0]), _lon180(cfg["lon_range"][1]),
            cfg["lat_range"][0], cfg["lat_range"][1],
        )
        for name, cfg in regions.items()
    }


def label_region_boxes(fig: plt.Figure, region_boxes: Dict[str, tuple], color="magenta"):
    import cartopy.crs as ccrs
    import matplotlib.patheffects as PathEffects
    ax = fig.axes[0]
    for name, (lon0, lon1, lat0, lat1) in region_boxes.items():
        below = name in MAP_LABEL_BELOW
        ax.text(
            (lon0 + lon1) / 2.0, lat0 - 1.0 if below else lat1 + 0.5, name,
            transform=ccrs.PlateCarree(), ha="center", va="top" if below else "bottom",
            fontsize=10, color=color, zorder=6,
            path_effects=[PathEffects.withStroke(linewidth=2.5, foreground="white")],
        )


def styled_map(lon, lat, field, regions, save_fig: SaveFig, filename: str, **kwargs):
    boxes = region_boxes_180(regions)
    fig = ct.fake_uba_map(
        lon=lon, lat=lat, c_array=field, region_boxes=boxes, show_region_labels=False,
        region_edgecolor="magenta", region_linewidth=2.0, cbar_orientation="vertical",
        show=False, **kwargs,
    )
    label_region_boxes(fig, boxes)
    save_fig(fig, filename)


def _to180(da: xr.DataArray) -> xr.DataArray:
    return da.assign_coords(lon=((da.lon + 180.0) % 360.0) - 180.0).sortby("lon").sortby("lat")


def _wrap_lon(da: xr.DataArray) -> xr.DataArray:
    """Periodic pad in longitude so interpolation works across 0°/360°."""
    return xr.concat(
        [da.assign_coords(lon=da.lon - 360.0), da, da.assign_coords(lon=da.lon + 360.0)],
        dim="lon",
    )


def _common_months(ppe: xr.DataArray, gpcp: xr.DataArray, months: Iterable[int]) -> List[str]:
    ppe_keys = {month_key(t) for t in ppe.time.values}
    return sorted(
        month_key(t) for t in gpcp.time.values
        if month_key(t) in ppe_keys and pd.Timestamp(t).month in set(months)
    )


def _select_months(da: xr.DataArray, keys: Iterable[str]) -> xr.DataArray:
    keys = set(keys)
    sel = [month_key(t) in keys for t in da.time.values]
    return da.isel(time=np.flatnonzero(sel))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_precip_seasonal_cycle(
    monthly_df: pd.DataFrame, gpcp_rows: pd.DataFrame, regions: Dict[str, dict],
    colors: Dict[str, str], year_months, save_fig: SaveFig,
    filename: str = "precip_ppe_vs_gpcp_monthly.png",
):
    order = [f"{y}-{m:02d}" for y, m in year_months]
    work = monthly_df[["region", "model", "month", "precip"]].copy()
    work["month"] = work["month"].astype(str)
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, len(regions), figsize=(5.2 * len(regions), 4.2), squeeze=False)
    for ax, (region, cfg) in zip(axes.ravel(), regions.items()):
        color = colors.get(region, "#333333")
        q = (work[work["region"] == region].groupby("month")["precip"]
             .quantile([0.05, 0.5, 0.95]).unstack().reindex(order))
        ax.fill_between(x, q[0.05], q[0.95], color=color, alpha=0.25, label="PPE 5–95%")
        ax.plot(x, q[0.5], color=color, lw=2, label="PPE median")
        g = gpcp_rows[gpcp_rows["region"] == region].set_index("month").reindex(order)
        ax.plot(x, g["precip_obs"], "k-o", ms=4, lw=1.5, label="GPCP")
        missing = ~g["gpcp_available"].fillna(False).astype(bool).to_numpy()
        for i in np.flatnonzero(missing):
            ax.axvline(i, color="gray", ls=":", lw=1.2,
                       label="GPCP missing" if i == np.flatnonzero(missing)[0] else None)
        for i, mon in enumerate(order):
            if int(mon[-2:]) in cfg["season_months"]:
                ax.axvspan(i - 0.5, i + 0.5, color="gold", alpha=0.12, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels([pd.Timestamp(m).strftime("%b\n%y") for m in order], fontsize=7)
        ax.set_ylabel(PRECIP_LABEL)
        ax.set_title(f"{region} (shaded: fire season)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Regional monthly precipitation: PPE vs GPCP (Aug 2024 – Jul 2025)")
    fig.tight_layout()
    save_fig(fig, filename)


def plot_precip_season_box(
    matched_df: pd.DataFrame, aerocom_df: pd.DataFrame, gpcp_obs: Dict[str, float],
    gpcp_2010: Dict[str, float], matched: Dict[str, Set[str]], regions: Dict[str, dict],
    colors: Dict[str, str], save_fig: SaveFig,
    filename: str = "precip_ppe_aerocom_vs_gpcp_season.png",
):
    fig, axes = plt.subplots(1, len(regions), figsize=(4.4 * len(regions), 4.6), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, region in zip(axes.ravel(), regions):
        color = colors.get(region, "#333333")
        ppe = matched_df.loc[matched_df["region"] == region, MATCHED_COL].dropna()
        aer = aerocom_df.loc[aerocom_df["region"] == region, "precip"].dropna()
        data = [(ppe, f"PPE\n2024-25\n(n={len(ppe)})", color),
                (aer, f"AeroCom\n2010\n(n={len(aer)})", "black")]
        data = [d for d in data if len(d[0])]
        if not data:
            ax.set_title(f"{region}: no data")
            continue
        bp = ax.boxplot([d[0].to_numpy() for d in data], widths=0.5, showfliers=False,
                        patch_artist=True)
        for patch, (_, _, c) in zip(bp["boxes"], data):
            patch.set_facecolor(c if c != "black" else "white")
            patch.set_alpha(0.5 if c != "black" else 1.0)
        for i, (vals, _, c) in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=8, alpha=0.4,
                       color=c, zorder=3)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels([d[1] for d in data], fontsize=8)
        obs_now, obs_2010 = gpcp_obs.get(region, np.nan), gpcp_2010.get(region, np.nan)
        if np.isfinite(obs_now):
            ax.axhline(obs_now, color="b", ls="--", lw=1.3, label=f"GPCP 2024-25 ({obs_now:.2f})")
        if np.isfinite(obs_2010):
            ax.axhline(obs_2010, color="gray", ls=":", lw=1.5, label=f"GPCP 2010 ({obs_2010:.2f})")
        months_txt = ", ".join(pd.Timestamp(m).strftime("%b %y") for m in sorted(matched.get(region, [])))
        ax.set_ylabel(r"Fire-season precipitation (mm day$^{-1}$)")
        ax.set_title(f"{region}\nPPE/GPCP months: {months_txt}", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Fire-season regional precipitation vs GPCP (only months with GPCP data)")
    fig.tight_layout()
    save_fig(fig, filename)


def plot_precip_maps(
    ppe_precip: xr.DataArray, gpcp_da: xr.DataArray, regions: Dict[str, dict],
    save_fig: SaveFig, months: Iterable[int],
):
    """GPCP mean and control run (first ensemble member) − GPCP over identical months."""
    keys = _common_months(ppe_precip, gpcp_da, months)
    if not keys:
        print("Skip precip maps: no common PPE/GPCP months")
        return
    label = ", ".join(pd.Timestamp(k).strftime("%b %y") for k in keys)
    gpcp_mean = _select_months(gpcp_da, keys).mean("time")
    ctrl = _select_months(ppe_precip, keys).mean("time")
    if "ens" in ctrl.dims:
        ctrl = ctrl.isel(ens=0, drop=True)
    gpcp_on_ppe = _wrap_lon(gpcp_mean).interp(lat=ctrl.lat, lon=ctrl.lon, method="linear")
    g = _to180(gpcp_mean)
    styled_map(g.lon.values, g.lat.values, g.values, regions, save_fig, "precip_gpcp_map.png",
               zmin=0.0, zmax=12.0, labels=r"GPCP precipitation (mm day$^{-1}$)",
               title=f"GPCP precipitation\n{label}", cbar_extend="max")
    d = _to180(ctrl - gpcp_on_ppe)
    styled_map(d.lon.values, d.lat.values, d.values, regions, save_fig,
               "precip_ppe_minus_gpcp_map.png", zmin=-6.0, zmax=6.0, mycolor="diff",
               labels=r"Control run − GPCP (mm day$^{-1}$)",
               title=f"Control run − GPCP\n{label}", cbar_extend="both")


def plot_precip_scatter(
    ppe_precip: xr.DataArray, gpcp_da: xr.DataArray, regions: Dict[str, dict],
    save_fig: SaveFig, filename: str = "precip_ppe_ensmean_vs_gpcp_scatter.png",
) -> pd.DataFrame:
    """GPCP_analysis.ipynb Fig. S4-style scatter with the PPE ensemble mean.

    Points are GPCP grid-box-months (fire season, GPCP-available months only);
    y is the PPE ensemble mean interpolated to the GPCP grid, error bars the
    ensemble interquartile range.
    """
    panels = [r for r in SCATTER_PANELS if r in regions]
    ppe_wrap = _wrap_lon(ppe_precip)
    stats_rows = []
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.2), squeeze=False)
    for ax, region in zip(axes.ravel(), panels):
        cfg = regions[region]
        panel, label = SCATTER_PANELS[region]
        keys = _common_months(ppe_precip, gpcp_da, cfg["season_months"])
        gsub = gpcp_region_subset(_select_months(gpcp_da, keys), cfg)
        psub = _select_months(ppe_wrap, keys).interp(lat=gsub.lat, lon=gsub.lon, method="linear")
        arr = psub.transpose("ens", "time", "lat", "lon").values if "ens" in psub.dims \
            else psub.values[None]
        mean = np.nanmean(arr, axis=0)
        q25, q75 = np.nanpercentile(arr, 25, axis=0), np.nanpercentile(arr, 75, axis=0)
        g = gsub.transpose("time", "lat", "lon").values.ravel()
        m = mean.ravel()
        e_lo = np.clip(m - q25.ravel(), 0, None)
        e_hi = np.clip(q75.ravel() - m, 0, None)
        valid = np.isfinite(g) & np.isfinite(m) & (g >= 0) & (m >= 0)
        r_val, p_val = pearsonr(g[valid], m[valid]) if valid.sum() >= 3 else (np.nan, np.nan)
        stats_rows.append({"region": region, "months": ",".join(keys), "n_points": int(valid.sum()),
                           "n_members": arr.shape[0], "r": r_val, "p": p_val,
                           "gpcp_mean": float(np.mean(g[valid])) if valid.any() else np.nan,
                           "ppe_mean": float(np.mean(m[valid])) if valid.any() else np.nan})
        ax.errorbar(g[valid], m[valid], yerr=[e_lo[valid], e_hi[valid]], fmt="o",
                    color={"amazon": "tab:blue", "africa": "tab:orange"}.get(region, "tab:green"),
                    ecolor="gray", alpha=0.35, markersize=3, linewidth=0.6, label="_nolegend_")
        max_val = max(np.nanmax(g[valid]), np.nanmax(m[valid])) * 1.05 if valid.any() else 1.0
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.2, label="1:1")
        p_str = f"{p_val:.2e}" if np.isfinite(p_val) and p_val >= 1e-300 else "< 10\u207b\u00b3\u2070\u2070"
        ax.text(0.05, 0.95, f"r = {r_val:.3f}\np = {p_str}\nn = {int(valid.sum())}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel("GPCP precipitation (mm/day)", fontsize=11)
        ax.set_ylabel("PPE ensemble-mean precip (mm/day)", fontsize=11)
        months_txt = ", ".join(pd.Timestamp(k).strftime("%b %y") for k in keys)
        ax.set_title(f"{panel} {label}\n{months_txt}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    n_members = stats_rows[0]["n_members"] if stats_rows else 0
    fig.suptitle(
        f"Modelled vs GPCP precipitation \u2014 PPE ensemble mean ({n_members} members)\n"
        "Error bars: ensemble interquartile range (25th\u201375th percentile); "
        "GPCP grid-box-months, fire season, months with GPCP only",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, filename)
    return pd.DataFrame(stats_rows)
