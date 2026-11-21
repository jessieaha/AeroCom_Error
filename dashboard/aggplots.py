"""Monthly aggregate plots for the dashboard window.

Reads monthly regional parquet with Polars. Plot kinds match the PPE notebooks
(MAC vs 1−SSA, MEC vs AE, lifetime vs AE, precip colour, time series) with
spread overlays and SPEXone / POLDER comparison marks.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

import ppe_io  # noqa: E402
import ppe_plots as pp  # noqa: E402
import regions as reg  # noqa: E402
from dashboard import obsmap  # noqa: E402
from dashboard.data import region_ring  # noqa: E402
from dashboard.polygons import monthly_path, read_polygon, slugify  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "figure" / "dashboard"
# Notebook figures already drawn with AeroCom diamonds.
SPECIES_FIGURE_DIR = {
    "du": PROJECT_ROOT / "figure" / "PPE_du",
    "ss": PROJECT_ROOT / "figure" / "PPE_ss",
    "bb": PROJECT_ROOT / "figure" / "PPE_bb",
    "total": PROJECT_ROOT / "figure" / "PPE_AAOD",
    "marine": PROJECT_ROOT / "figure" / "PPE_ss",
}
# (species, kind, style) -> filename inside that species folder.
NOTEBOOK_FIGURES = {
    ("du", "column", "scatter"): "combined_AOD_vs_SSA.png",
    ("du", "mec_ae", "scatter"): "combined_MEC_du_vs_AE.png",
    ("du", "mec_ae", "density"): "combined_MEC_du_vs_AE_density.png",
    ("du", "lifetime_ae", "scatter"): "combined_inv_lifetime_du_vs_AE.png",
    ("du", "lifetime_precip", "scatter"): "combined_inv_lifetime_du_vs_precip.png",
    ("bb", "mac_ssa", "scatter"): "combined_MAC_vs_1SSA.png",
    ("bb", "mac_ssa", "density"): "combined_MAC_vs_1SSA_density.png",
    ("bb", "mec_ae", "scatter"): "combined_MEC_vs_AE.png",
    ("bb", "mec_ae", "density"): "combined_MEC_vs_AE_density.png",
    ("bb", "lifetime_ae", "scatter"): "combined_inv_lifetime_bcoa_vs_AE.png",
    ("bb", "lifetime_ae", "density"): "combined_inv_lifetime_bcoa_vs_AE_density.png",
    ("bb", "lifetime_precip", "scatter"): "combined_inv_lifetime_bcoa_vs_precip.png",
    ("bb", "column", "scatter"): "combined_AAOD_vs_SSA.png",
}
AEROCOM_LONG = {
    "du": PROJECT_ROOT / "tables" / "regional_seasonal_aggregates_du.parquet",
    "bb": PROJECT_ROOT / "tables" / "regional_seasonal_aggregates.parquet",
    "total": PROJECT_ROOT / "tables" / "regional_seasonal_aggregates.parquet",
    "marine": PROJECT_ROOT / "tables" / "regional_seasonal_aggregates.parquet",
}
AEROCOM_SS = PROJECT_ROOT / "tables" / "aerocom_ss_seasonal.parquet"
AEROCOM_EXCLUDE = {
    "GEOS-i33p2-met2010_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-OMA_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-MATRIX_AP3-CTRL",
    "NorESM2-met2010_AP3-CTRL-v3",
}

# species -> plot-kind -> (x, y, through_origin, title, inverse_y)
# column = full-column AOD and SSA (all aerosols together), not a species optical depth.
_COLUMN = ("SSA", "od550aer", False, r"Column optics: AOD$_{550}$ vs SSA (all aerosols)", False)

PLOT_DEFS = {
    "du": {
        "column": _COLUMN,
        "mac_ssa": ("1-SSA", "MAC", True, r"Dust: MAC vs 1 − SSA", False),
        "mec_ae": ("AE", "MEC_du", False, r"Dust: MEC$_{DU}$ vs AE", False),
        "lifetime_ae": ("AE", "lifetime_du", False, r"Dust lifetime vs AE", False),
        "lifetime_precip": ("AE", "lifetime_du", False, r"Dust lifetime vs AE (precip colour)", False),
        "timeseries": ("month", "od550aer", False, r"Dust monthly time series", False),
    },
    "ss": {
        "column": _COLUMN,
        "mac_ssa": ("1-SSA", "MAC", True, r"Sea salt: MAC vs 1 − SSA", False),
        "mec_ae": ("AE", "MEC_ss", False, r"Sea salt: MEC$_{SS}$ vs AE", False),
        "lifetime_ae": ("AE", "lifetime_ss", False, r"Sea salt lifetime vs AE", False),
        "lifetime_precip": ("AE", "lifetime_ss", False, r"Sea salt lifetime vs AE (precip colour)", False),
        "timeseries": ("month", "od550aer", False, r"Sea salt monthly time series", False),
    },
    "bb": {
        "column": _COLUMN,
        "mac_ssa": ("1-SSA", "MAC", True, r"Biomass burning: MAC vs 1 − SSA", False),
        "mec_ae": ("AE", "MAC", False, r"Biomass burning: MAC vs AE", False),
        "lifetime_ae": ("AE", "lifetime_BC_OA", False, r"BC+OA lifetime vs AE", False),
        "lifetime_precip": ("AE", "lifetime_BC_OA", False, r"BC+OA lifetime vs AE (precip colour)", False),
        "timeseries": ("month", "od550aer", False, r"BB monthly time series", False),
    },
    "total": {
        "column": ("SSA", "od550aer", False, r"Total column: AOD$_{550}$ vs SSA (BC+OA+DU+SS+SO$_4$)", False),
        "mac_ssa": ("1-SSA", "MAC", True, r"Total column: MAC$_{tot}$ vs 1 − SSA", False),
        "mec_ae": ("AE", "abs550aer", False, r"Total column: AAOD$_{550}$ vs AE", False),
        "lifetime_ae": ("AE", "lifetime", False, r"Total aerosol lifetime (SO4-eq. source) vs AE", False),
        "lifetime_mass": ("AE", "lifetime_mass", False, r"Total aerosol lifetime (plain mass sum) vs AE", False),
        "lifetime_precip": ("AE", "lifetime", False, r"Total aerosol lifetime vs AE (precip colour)", False),
        "timeseries": ("month", "od550aer", False, r"Total column AOD$_{550}$ monthly time series", False),
    },
    "marine": {
        "column": _COLUMN,
        "mac_ssa": ("1-SSA", "MAC", True, r"Marine (SS+SO$_4$): MAC vs 1 − SSA", False),
        "mec_ae": ("AE", "MEC_ss", False, r"Marine: MEC$_{SS}$ vs AE", False),
        "lifetime_ae": ("AE", "lifetime_marine", False, r"Marine lifetime vs AE", False),
        "lifetime_precip": ("AE", "lifetime_marine", False, r"Marine lifetime vs AE (precip colour)", False),
        "timeseries": ("month", "od550aer", False, r"Marine monthly time series", False),
    },
}

LIFETIME_COLS = {
    "du": ("lifetime_du", "lifetime_du_dep"),
    "ss": ("lifetime_ss", "lifetime_ss_dep"),
    "bb": ("lifetime_BC_OA", "lifetime_BC_OA_dep"),
    "total": ("lifetime", "lifetime_dep"),
    "marine": ("lifetime_marine", "lifetime_marine_dep"),
}

AE_VAR = {"du": "AE_440_550", "ss": "AE_550_865", "bb": "AE_440_550",
          "total": "AE_440_550", "marine": "AE_550_865"}

LABELS = {
    "SSA": "SSA", "1-SSA": r"1 − SSA", "MAC": "MAC", "AE": "AE",
    "MEC_du": r"MEC$_{DU}$", "MEC_ss": r"MEC$_{SS}$",
    "lifetime_du": r"$\tau_{DU}$ (days)", "lifetime_ss": r"$\tau_{SS}$ (days)",
    "lifetime_BC_OA": r"$\tau_{BC+OA}$ (days)",     "lifetime": r"$\tau$ SO4-eq. (days)",
    "lifetime_mass": r"$\tau$ mass sum (days)",
    "lifetime_marine": r"$\tau_{marine}$ (days)",
    "abs550aer": r"AAOD$_{550}$",
    "lifetime_du_dep": r"$\tau_{DU}$ dep (days)", "lifetime_ss_dep": r"$\tau_{SS}$ dep (days)",
    "lifetime_BC_OA_dep": r"$\tau_{BC+OA}$ dep (days)",
    "lifetime_dep": r"$\tau$ dep (days)", "lifetime_marine_dep": r"$\tau_{marine}$ dep (days)",
    "od550aer": r"AOD$_{550}$", "month": "Month", "precip": "Precipitation",
}

MARINE_INFO = (
    "Marine = sea salt + sulfate. Sulfate source is primary SO₄ plus half of "
    "SO₂ and DMS converted by molar mass "
    "(SO2_TO_SO4_FRAC × M_SO4/M_SO2 and M_SO4/M_DMS)."
)

STYLES = (
    "scatter", "density",
    "ppe_std", "ppe_minmax",
    "aerocom_std", "aerocom_minmax", "aerocom_models",
)


def _box(name: str):
    if name in reg.REGIONS:
        cfg = reg.REGIONS[name]
        return region_ring(cfg), list(cfg["season_months"]), cfg.get("surface_type", "all")
    spec = read_polygon(slugify(name))
    return spec["ring"], list(range(1, 13)), spec.get("surface_type", "all")


def _read_parquet(path: Path) -> pd.DataFrame:
    return pl.read_parquet(path).to_pandas()


def _species_alias_frame(df: pd.DataFrame, species: str) -> pd.DataFrame:
    df = ppe_io.add_budget_columns(df)
    df = ppe_io.add_lifetime_drivers(df)
    if species in reg.SPECIES:
        df = df.copy()
        for alias, col in reg.SPECIES[species]["aliases"].items():
            # Keep a column the AeroCom table already named (MAC, AE).
            if col in df.columns:
                df[alias] = df[col]
            elif alias not in df.columns:
                df[alias] = np.nan
    else:
        # total / marine: still want MAC and AE column aliases from dust/ss defaults
        df = df.copy()
        if "MAC_total" in df.columns:
            df["MAC"] = df["MAC_total"]
        if "SSA" not in df.columns and {"od550aer", "abs550aer"} <= set(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                df["SSA"] = 1.0 - df["abs550aer"] / df["od550aer"]
        if species == "marine" and "AE_550_865" in df.columns:
            df["AE"] = df["AE_550_865"]
        elif "AE_440_550" in df.columns:
            df["AE"] = df["AE_440_550"]
        elif "AE_550_865" in df.columns:
            df["AE"] = df["AE_550_865"]
    with np.errstate(divide="ignore", invalid="ignore"):
        if {"loaddust", "depdust"} <= set(df.columns):
            df["lifetime_du_dep"] = df["loaddust"] / (df["depdust"] * 86400.0)
        if {"loadss", "depss"} <= set(df.columns):
            df["lifetime_ss_dep"] = df["loadss"] / (df["depss"] * 86400.0)
        if {"load_BC_OA", "depbc", "depoa"} <= set(df.columns):
            df["lifetime_BC_OA_dep"] = df["load_BC_OA"] / ((df["depbc"] + df["depoa"]) * 86400.0)
    return pp.with_one_minus_ssa(df)


def notebook_figure(species: str, kind: str, style: str) -> Optional[Path]:
    """Existing notebook PNG for this species and plot, if that file is on disk."""
    name = NOTEBOOK_FIGURES.get((species, kind, style))
    if not name:
        return None
    path = SPECIES_FIGURE_DIR.get(species, FIGURE_DIR) / name
    return path if path.is_file() else None


def _pivot_aerocom_long(path: Path, regions: Sequence[str]) -> Optional[pd.DataFrame]:
    """Wide AeroCom table (one row per model and region) from a long seasonal file."""
    if not path.is_file():
        return None
    long = _read_parquet(path)
    if "region" not in long.columns:
        return None
    long = long.loc[long["region"].isin(list(regions))]
    if long.empty:
        return None
    if "dataset" in long.columns:
        long = long.loc[~long["dataset"].isin(["POLDER", "GPCP", *AEROCOM_EXCLUDE])]
    if "source" in long.columns:
        long = long.loc[long["source"].isin(["aggregated", "derived"])]
    if long.empty or "var" not in long.columns:
        return None
    wide = (
        long.pivot_table(
            index=["region", "dataset"], columns="var", values="value", aggfunc="first",
        )
        .reset_index()
        .rename(columns={"dataset": "model"})
    )
    wide.columns.name = None
    wide["ppe_id"] = "AeroCom"
    if "lifetime_du" not in wide.columns and "lifetime_du_emission" in wide.columns:
        wide["lifetime_du"] = wide["lifetime_du_emission"]
    if "lifetime_BC_OA" not in wide.columns and "lifetime_BC_OA_emission" in wide.columns:
        wide["lifetime_BC_OA"] = wide["lifetime_BC_OA_emission"]
    if "lifetime" not in wide.columns and {"load_total", "emi_total"} <= set(wide.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            wide["lifetime"] = wide["load_total"] / (wide["emi_total"] * 86400.0)
            wide["lifetime_mass"] = wide["lifetime"]
    return wide


def load_aerocom(species: str, regions: Sequence[str]) -> Tuple[Optional[pd.DataFrame], str]:
    """AeroCom model points from the tables the notebooks already use."""
    frames = []
    notes = []
    if species == "ss":
        if AEROCOM_SS.is_file():
            ss = _read_parquet(AEROCOM_SS)
            ss = ss.loc[ss["region"].isin(list(regions))]
            if not ss.empty:
                if "ppe_id" not in ss.columns:
                    ss["ppe_id"] = "AeroCom"
                optical = PROJECT_ROOT / "tables" / "aerocom_box_sensitivity_seasonal.parquet"
                if optical.is_file():
                    opt = _read_parquet(optical)
                    opt = opt.loc[opt["region"].isin(list(regions)) & ~opt["region"].astype(str).str.contains("__")]
                    keep = [c for c in ("MEC_ss", "MEC", "MAC", "MAC_total", "SSA", "AE_440_550",
                                        "od550ss", "abs550aer", "loadss", "emiss") if c in opt.columns]
                    if keep and not opt.empty:
                        ss = ss.merge(opt[["model", "region", *keep]], on=["model", "region"],
                                      how="left", suffixes=("", "_opt"))
                        for col in keep:
                            alt = f"{col}_opt"
                            if alt in ss.columns:
                                ss[col] = ss[col].fillna(ss[alt]) if col in ss.columns else ss[alt]
                                ss = ss.drop(columns=[alt])
                frames.append(ss)
                notes.append(f"AeroCom sea salt from {AEROCOM_SS.name} (seasonal model points, all years pooled)")
    else:
        paths = [AEROCOM_LONG[species]] if species in AEROCOM_LONG else []
        if species in ("total", "marine"):
            paths = [
                PROJECT_ROOT / "tables" / "regional_seasonal_aggregates_du.parquet",
                PROJECT_ROOT / "tables" / "regional_seasonal_aggregates.parquet",
            ]
        for path in paths:
            wide = _pivot_aerocom_long(path, regions)
            if wide is not None and not wide.empty:
                frames.append(wide)
                notes.append(f"AeroCom from {path.name}")
    if not frames:
        return None, "no AeroCom regional table for these boxes"
    aero = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True, sort=False)
    aero = aero.drop_duplicates(subset=[c for c in ("model", "region") if c in aero.columns])
    n_models = int(aero["model"].nunique()) if "model" in aero.columns else 0
    notes.append(f"{n_models} AeroCom models")
    return _species_alias_frame(aero, species if species in reg.SPECIES else species), "; ".join(notes)


def has_deposition(df: pd.DataFrame, species: str) -> bool:
    emi, dep = LIFETIME_COLS[species]
    return dep in df.columns and df[dep].notna().any()


def load_monthly(
    species: str,
    regions: Sequence[str],
    months: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
    """Return (ppe_df, aerocom_df_or_None, caption_note)."""
    known = [r for r in regions if r in reg.REGIONS and r != "global"]
    custom = []
    for name in regions:
        if name in reg.REGIONS or name == "global":
            continue
        try:
            slug = slugify(name)
            read_polygon(slug)
            if monthly_path(slug).is_file():
                custom.append(name)
        except (FileNotFoundError, ValueError):
            continue
    names = known + custom
    if not names:
        raise ValueError("pick at least one region whose monthly table exists")

    def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        if "model" in out.columns and "member" not in out.columns:
            out = out.rename(columns={"model": "member"})
            out["model"] = [ppe_io.member_key(p, m) for p, m in zip(out["ppe_id"], out["member"])]
        return _species_alias_frame(out, species)

    frames = []
    if known:
        import PPE_set as cfg
        parts = [_read_parquet(cfg.REGIONAL_MONTHLY)]
        icon = cfg.REGIONAL_DIR / "icon2025_regional_monthly.parquet"
        if icon.is_file():
            parts.append(_read_parquet(icon))
        raw = pd.concat(parts, ignore_index=True, sort=False)
        frames.append(_normalize(raw.loc[raw["region"].isin(known)]))

    if custom:
        extra = pd.concat(
            [_read_parquet(monthly_path(slugify(name))) for name in custom],
            ignore_index=True,
        )
        frames.append(_normalize(extra))

    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True, sort=False)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df["month"] = df["time"].dt.month
    if months:
        df = df.loc[df["month"].isin(list(months))].copy()

    aero, aero_note = load_aerocom(species, names)
    note_parts = [aero_note]
    if not has_deposition(df, species):
        note_parts.append("no deposition fluxes in these tables - emission lifetime only")
    return df, aero, "; ".join(note_parts)


def _obs_marks(species: str, names: Sequence[str], months: Sequence[int], xcol: str, ycol: str):
    obs_x, obs_y = {}, {}
    for pid in ("pd2024", "pd2010", "icon2025"):
        dataset = "spexone" if ppe_io.ppe_obs_name(pid) == "SPEXone" else "polder"
        year_months = ppe_io.ppe_year_months(pid)
        label = ppe_io.ppe_obs_name(pid)
        for name in names:
            try:
                ring, _season, surface = _box(name)
            except (FileNotFoundError, ValueError):
                continue
            use_months = months or list(range(1, 13))
            try:
                if xcol in ("AE",) or "lifetime" in ycol:
                    val = obsmap.region_value(
                        dataset, AE_VAR[species], ring, use_months, surface, year_months,
                    )
                    obs_x[(pid, name)] = (val, label)
                if xcol in ("SSA", "1-SSA"):
                    ssa = obsmap.region_value(
                        dataset, "SSA", ring, use_months, surface, year_months,
                    )
                    shown = (1.0 - ssa) if xcol == "1-SSA" else ssa
                    obs_x[(pid, name)] = (shown, label)
                if ycol == "od550aer":
                    aod = obsmap.region_value(
                        dataset, "od550aer", ring, use_months, surface, year_months,
                    )
                    obs_y[(pid, name)] = (aod, label)
            except (FileNotFoundError, ValueError, KeyError):
                continue
    return obs_x, obs_y


def _xy(df: pd.DataFrame, region: str, xcol: str, ycol: str):
    sub = df.loc[df["region"] == region]
    if xcol not in sub.columns or ycol not in sub.columns:
        return np.array([]), np.array([])
    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _spread_point(x: np.ndarray, y: np.ndarray, mode: str):
    """Return (x_med, y_med, x_lo, x_hi, y_lo, y_hi) or None."""
    if len(x) < 2:
        return None
    xm, ym = np.median(x), np.median(y)
    if mode.endswith("std"):
        return xm, ym, xm - np.std(x), xm + np.std(x), ym - np.std(y), ym + np.std(y)
    # minmax after dropping >3σ
    zx = np.abs(x - np.mean(x)) <= 3 * (np.std(x) + 1e-12)
    zy = np.abs(y - np.mean(y)) <= 3 * (np.std(y) + 1e-12)
    keep = zx & zy
    if keep.sum() < 2:
        keep = np.ones(len(x), dtype=bool)
    xx, yy = x[keep], y[keep]
    return xm, ym, xx.min(), xx.max(), yy.min(), yy.max()


def _draw_spread(ax, df, region, xcol, ycol, ppe_ids, style: str, aerocom_df=None):
    if style in ("ppe_std", "ppe_minmax"):
        mode = "std" if style == "ppe_std" else "minmax"
        for ppe_id in ppe_ids:
            st = ppe_io.ppe_style(ppe_id)
            x, y = _xy(df.loc[df["ppe_id"] == ppe_id], region, xcol, ycol)
            sp = _spread_point(x, y, mode)
            if sp is None:
                continue
            xm, ym, x0, x1, y0, y1 = sp
            ax.errorbar(
                [xm], [ym], xerr=[[xm - x0], [x1 - xm]], yerr=[[ym - y0], [y1 - ym]],
                fmt="o", color=st["color"], ecolor=st["color"], capsize=3,
                label=f"{st['label']} median±",
            )
    if style in ("aerocom_std", "aerocom_minmax", "aerocom_models") and aerocom_df is not None:
        sub = aerocom_df.loc[aerocom_df["region"] == region]
        if sub.empty or xcol not in sub.columns or ycol not in sub.columns:
            return
        if style == "aerocom_models":
            for model, g in sub.groupby(sub.get("model", sub.get("ppe_id", pd.Series(["AeroCom"] * len(sub))))):
                x = g[xcol].to_numpy(float)
                y = g[ycol].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.any():
                    ax.scatter(x[ok], y[ok], s=28, marker="D", edgecolors="k",
                               facecolors="none", label=str(model))
        else:
            x = sub[xcol].to_numpy(float)
            y = sub[ycol].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            sp = _spread_point(x[ok], y[ok], "std" if style == "aerocom_std" else "minmax")
            if sp is not None:
                xm, ym, x0, x1, y0, y1 = sp
                ax.errorbar(
                    [xm], [ym], xerr=[[xm - x0], [x1 - xm]], yerr=[[ym - y0], [y1 - ym]],
                    fmt="D", color="k", ecolor="k", capsize=3, label="AeroCom median±",
                )


_ID_COLS = {
    "ppe_id", "ens", "model", "member", "experiment", "region", "time", "month",
}


def list_plot_columns(species: str) -> List[str]:
    """Numeric fields the x/y menus can offer, without reading the whole table."""
    import PPE_set as cfg
    names = list(pl.scan_parquet(cfg.REGIONAL_MONTHLY).collect_schema().names())
    names += [
        "1-SSA", "AE", "MAC", "MEC", "MEC_ss", "MEC_du",
        "lifetime", "lifetime_mass", "lifetime_marine",
        "lifetime_ss", "lifetime_du", "lifetime_BC_OA",
    ]
    out = []
    for name in names:
        if name in _ID_COLS or name in out:
            continue
        out.append(name)
    return out


def list_saved_figures(species: str) -> List[dict]:
    folder = SPECIES_FIGURE_DIR.get(species, FIGURE_DIR)
    if not folder.is_dir():
        return []
    return [
        {"name": path.name, "url": f"/figures/{folder.name}/{path.name}"}
        for path in sorted(folder.glob("*.png"))
        if path.name.startswith(("ppe_", "combined_"))
    ]


def render(
    kind: str,
    species: str,
    regions: Sequence[str],
    months: Optional[Sequence[int]] = None,
    style: str = "scatter",
    lifetime_source: str = "emission",
    timeseries_var: Optional[str] = None,
    save: bool = False,
    x: Optional[str] = None,
    y: Optional[str] = None,
) -> Tuple[bytes, Optional[Path], str]:
    if species not in PLOT_DEFS:
        raise ValueError("species must be du, ss, bb, total or marine")
    custom_xy = bool(x and y)
    if not custom_xy and kind not in PLOT_DEFS[species]:
        raise ValueError(f"kind must be one of {list(PLOT_DEFS[species])}")
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")

    df, aero, note = load_monthly(species, regions, months=months)
    names = [r for r in regions if r in set(df["region"])][:6]
    if not names:
        raise ValueError("no rows for the chosen regions / months")

    if custom_xy:
        if x not in df.columns or y not in df.columns:
            raise ValueError(f"{x} or {y} is not in this table")
        xcol, ycol, origin, title = x, y, False, f"{y} vs {x}"
        kind = "xy"
    else:
        xcol, ycol, origin, title, _inv = PLOT_DEFS[species][kind]
    if kind.startswith("lifetime") and lifetime_source == "deposition":
        emi, dep = LIFETIME_COLS[species]
        if not has_deposition(df, species):
            note = (note + "; " if note else "") + "deposition lifetime unavailable - using emission"
            lifetime_source = "emission"
        else:
            ycol = dep
            title = title.replace("lifetime", "deposition lifetime")

    if kind == "timeseries":
        ycol = timeseries_var or ycol

    month_list = list(months) if months else sorted(df["month"].dropna().unique().astype(int).tolist())
    obs_x, obs_y = ({}, {})
    if kind != "timeseries":
        obs_x, obs_y = _obs_marks(species, names, month_list, xcol, ycol)

    buf = io.BytesIO()
    ppe_ids = ("pd2024", "pd2010", "icon2025")

    if kind == "timeseries":
        fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8), squeeze=False)
        for ax, region in zip(axes.ravel(), names):
            for ppe_id in ppe_ids:
                st = ppe_io.ppe_style(ppe_id)
                sub = df.loc[(df["ppe_id"] == ppe_id) & (df["region"] == region)]
                if sub.empty or ycol not in sub.columns:
                    continue
                g = sub.groupby("month")[ycol].median()
                ax.plot(g.index, g.values, color=st["color"], marker=st["marker"],
                        lw=1.4, label=st["label"])
            ax.set_title(region)
            ax.set_xlabel("Month")
            ax.set_ylabel(LABELS.get(ycol, ycol))
            ax.set_xticks(range(1, 13))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(title + "  ·  monthly tables")
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    elif kind == "lifetime_precip" and "precip" in df.columns:
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.4), squeeze=False)
        sc = None
        for ax, region in zip(axes.ravel(), names):
            for ppe_id in ppe_ids:
                st = ppe_io.ppe_style(ppe_id)
                sub = df.loc[(df["ppe_id"] == ppe_id) & (df["region"] == region)]
                if sub.empty or xcol not in sub.columns or ycol not in sub.columns:
                    continue
                x = sub[xcol].to_numpy(float)
                y = sub[ycol].to_numpy(float)
                c = sub["precip"].to_numpy(float) if "precip" in sub.columns else np.zeros(len(sub))
                ok = np.isfinite(x) & np.isfinite(y)
                sc = ax.scatter(x[ok], y[ok], c=c[ok], s=18, cmap="viridis",
                                marker=st["marker"], alpha=0.7, label=st["label"])
            if style != "scatter":
                _draw_spread(ax, df, region, xcol, ycol, ppe_ids, style, aero)
            ax.set_title(region)
            ax.set_xlabel(LABELS.get(xcol, xcol))
            ax.set_ylabel(LABELS.get(ycol, ycol))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        if sc is not None:
            fig.colorbar(sc, ax=list(axes.ravel()), fraction=0.03, pad=0.04, label="precip")
        fig.suptitle(title + f"  ·  {note}")
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    elif style == "density":
        def saver(fig, _name):
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        pp.density_by_region(
            df, xcol, ycol, names, saver, "plot.png",
            title=title + "  ·  monthly density",
            ppe_ids=ppe_ids, labels=LABELS, aerocom_df=aero,
        )
    elif style in ("ppe_std", "ppe_minmax", "aerocom_std", "aerocom_minmax", "aerocom_models"):
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.4), squeeze=False)
        for ax, region in zip(axes.ravel(), names):
            # faint scatter under the spread
            for ppe_id in ppe_ids:
                st = ppe_io.ppe_style(ppe_id)
                x, y = _xy(df.loc[df["ppe_id"] == ppe_id], region, xcol, ycol)
                if len(x):
                    ax.scatter(x, y, s=8, alpha=0.15, color=st["color"], edgecolors="none")
            _draw_spread(ax, df, region, xcol, ycol, ppe_ids, style, aero)
            for lines, axline in ((obs_x, ax.axvline), (obs_y, ax.axhline)):
                for (pid, r), (val, lab) in (lines or {}).items():
                    if r != region or not np.isfinite(val):
                        continue
                    color = ppe_io.ppe_style(pid)["color"]
                    axline(val, color=color, ls="--", lw=1.2, label=lab)
            if origin:
                ax.plot([0, 1], [0, 1], transform=ax.get_xaxis_transform()
                        if False else None)
            ax.set_title(region)
            ax.set_xlabel(LABELS.get(xcol, xcol))
            ax.set_ylabel(LABELS.get(ycol, ycol))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(title + f"  ·  {style}  ·  {note}")
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        def saver(fig, _name):
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        def precip_extra(ax, region):
            return None

        pp.scatter_by_region(
            df, xcol, ycol, names, saver,
            filename="plot.png",
            title=title + "  ·  monthly tables · " + note,
            ppe_ids=ppe_ids,
            labels=LABELS,
            through_origin=origin,
            aerocom_df=aero,
            obs_x=obs_x or None,
            obs_y=obs_y or None,
            extra=precip_extra if kind == "lifetime_precip" else None,
        )

    png = buf.getvalue()
    saved = None
    if save:
        folder = SPECIES_FIGURE_DIR.get(species, FIGURE_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        region_tag = "-".join(names[:3])
        fname = f"dashboard_{species}_{kind}_{style}_{region_tag}.png"
        saved = folder / fname
        # Keep notebook combined_*.png files; a new draw gets its own name.
        saved.write_bytes(png)
    return png, saved, note


# Back-compat for the old endpoint signature
def render_legacy(kind: str, species: str, regions: Sequence[str]) -> bytes:
    mapped = {"column": "mac_ssa", "optics": "mec_ae", "lifetime": "lifetime_ae"}.get(kind, kind)
    if species == "bb" and kind == "column":
        mapped = "mac_ssa"
    png, _, _ = render(mapped, species, regions, style="scatter")
    return png
