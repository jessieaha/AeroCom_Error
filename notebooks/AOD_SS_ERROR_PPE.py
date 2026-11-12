# %% [markdown]
# # AOD sea-salt error analysis on the Pace PPEs
#
# Port of ``AOD_SS_ERROR.ipynb`` onto the regional PPE store
# (``Data/PPE_regional``, built by ``py/ppe_regional_aggregates.py``).
# Every registered PPE in ``PPE_IDS`` is drawn as its own cloud, with AeroCom
# AP3 2010 as open diamonds:
#   * pd2024 (Aug 2024 - Jul 2025) is compared with SPEXone;
#   * pd2010 (2010, the AeroCom / POLDER year) is compared with POLDER-GRASP.
# Sea-salt regions follow ``regions.REGIONS`` (surface_type unchanged):
# remote_SO and the nested Pacific box are not in the lifetime analysis.
# Sea-salt regions: SAO_SOND and SAO_2_DJF (highest 1/τ vs AE).
# Box-size and box-location sensitivity (±5°) is in
# ``notebooks/PPE_box_sensitivity.py``.
# Emission-based lifetimes only (no wet/dry/dep streams in the PPE extract);
# precip regressions are skipped for PPEs without a precip stream.
#
# Run: `.venv-1/bin/python notebooks/AOD_SS_ERROR_PPE.py`
# Or:  `sbatch run_notebook_or_py.sbatch notebooks/AOD_SS_ERROR_PPE.py`

# %%
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr

try:
    get_ipython()  # noqa: F821
    _IN_IPYTHON = True
except NameError:
    _IN_IPYTHON = False

import matplotlib
if not _IN_IPYTHON:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

py_dir = project_root / "py"
nb_dir = project_root / "notebooks"
for p in (str(project_root), str(py_dir), str(nb_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import PPE_set as ppe_cfg
import box_sensitivity as boxes
import ppe_io
import ppe_plots as pp
import regions as reg

print(f"Project root: {project_root}")

# %%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SPECIES = "ss"
PPE_IDS: List[str] = ["pd2024", "pd2010", "icon2025"]
EXCLUDE_MEMBERS: List[str] = []  # e.g. ["Control"] or ["pd2010:17"]

# Sea-salt regions with the highest emission-lifetime R² (1/τ vs AE):
# SAO_SOND ≈ 0.16, SAO_2_DJF ≈ 0.14. North Pacific and the low-R² basins are omitted.
_REGION_ORDER = ["SAO_SOND", "SAO_2_DJF"]
SOURCE_REGIONS = reg.source_regions(SPECIES)
ANALYSIS_REGIONS = [r for r in _REGION_ORDER if r in SOURCE_REGIONS]

SAVE_FIGURE = True
FIGURE_DIR = project_root / "figure" / "PPE_ss"
TABLE_DIR = ppe_cfg.ANALYSIS_DIR / SPECIES
TABLE_DIR.mkdir(parents=True, exist_ok=True)
save_fig = pp.FigureSaver(FIGURE_DIR, save=SAVE_FIGURE)

HOMOGENIZE_VARS = {"AOD": "AOD_550", "AAOD": "AAOD_550", "SSA": "SSA"}

AXIS_LABELS = {
    "SSA": r"SSA [-]",
    "1-SSA": r"1 − SSA [-]",
    "MEC": r"MEC (m$^{2}$ g$^{-1}$)",
    "MEC_ss": r"MEC$_{SS}$ (m$^{2}$ g$^{-1}$)",
    "MEC_so4": r"MEC$_{SO4}$ (m$^{2}$ g$^{-1}$)",
    "MAC": r"MAC (m$^{2}$ g$^{-1}$)",
    "AE": r"AE 550–865 [-]",
    "precip": r"Precipitation (mm day$^{-1}$)",
    "relhum_sfc": r"Near-surface RH [-]",
    "stress_water": r"Water wind stress $|\tau|$ (Pa)",
    "lifetime_ss": r"Sea-salt lifetime $\tau_{SS}$ (day)",
    "lifetime_SO4": r"Sulfate lifetime $\tau_{SO4}$ (day)",
    "lifetime_mass": r"Lifetime $\tau$ (day)",
    "inv_lifetime_ss": r"1/$\tau_{SS}$ (day$^{-1}$)",
    "inv_lifetime_SO4": r"1/$\tau_{SO4}$ (day$^{-1}$)",
    "inv_lifetime_mass": r"1/$\tau$ (day$^{-1}$)",
    "inv_lifetime_marine": r"1/$\tau_{SS+SO4}$ (day$^{-1}$)",
    "od550aer": r"AOD$_{550}$ [-]",
    "od550ss": r"AOD$_{SS,550}$ [-]",
    "ss_burden_frac": r"Sea-salt share of SS+SO4 burden [-]",
    "dms_source_frac": r"DMS share of SO4-equivalent source [-]",
    "so2_source_frac": r"SO2 share of SO4-equivalent source [-]",
    "lifetime_marine": r"Marine lifetime $\tau_{SS+SO4}$ (day)",
    "lifetime_marine_dep": r"Marine lifetime from deposition (day)",
}

print("--- Active configuration ---")
print(f"  PPE_IDS          = {PPE_IDS}")
print(f"  ANALYSIS_REGIONS = {ANALYSIS_REGIONS}")
for pid in PPE_IDS:
    run = ppe_cfg.get_run(pid)
    print(f"  {pid}: {run['period']}  burden_scale={run['burden_scale']}  "
          f"precip={run['has_precip']}  obs={run['obs']}")
print(f"  FIGURE_DIR       = {FIGURE_DIR}")
print(f"  TABLE_DIR        = {TABLE_DIR}")

# %%
# -----------------------------------------------------------------------------
# Regional PPE tables (seasonal = mean over each region's season months)
# -----------------------------------------------------------------------------
model_seasonal_df = ppe_io.load_regional(PPE_IDS, species=SPECIES, regions=ANALYSIS_REGIONS,
                                         seasonal=True, exclude_members=EXCLUDE_MEMBERS)
# The two remote boxes are aggregated by py/box_sensitivity.py (the main
# regional store may not have been rebuilt since they were added).
try:
    nominal = boxes.ppe_frame(SPECIES, nominal_only=True)
except FileNotFoundError as exc:
    print(exc)
    nominal = pd.DataFrame()
if not nominal.empty:
    nominal = nominal.loc[~nominal["region"].isin(set(model_seasonal_df["region"]))]
    if EXCLUDE_MEMBERS:
        ex = set(EXCLUDE_MEMBERS)
        nominal = nominal.loc[~(nominal["member"].isin(ex) | nominal["model"].isin(ex))]
    model_seasonal_df = pd.concat([model_seasonal_df, nominal], ignore_index=True, sort=False)
model_monthly_df = ppe_io.load_regional(PPE_IDS, species=SPECIES, regions=ANALYSIS_REGIONS,
                                        seasonal=False, exclude_members=EXCLUDE_MEMBERS)
PPE_IDS = [p for p in PPE_IDS if p in set(model_seasonal_df["ppe_id"])]
PRECIP_PPES = [p for p in PPE_IDS if ppe_io.has_precip(model_seasonal_df, p)]
print(f"Seasonal rows: {len(model_seasonal_df)}  PPEs with precip: {PRECIP_PPES}")

availability = [c for c in ["MEC_ss", "MEC", "MAC", "SSA", "AE", "lifetime_ss",
                            "lifetime_mass", "precip"] if c in model_seasonal_df.columns]
print(model_seasonal_df.groupby(["ppe_id", "region"])[availability].count())

aerocom_df = boxes.screen_lifetime(
    boxes.aerocom_frame(SPECIES, ANALYSIS_REGIONS),
    ["lifetime_ss", "lifetime_mass", "lifetime"],
)
if aerocom_df.empty:
    print("AeroCom sea-salt overlay skipped (run python py/box_sensitivity.py).")
else:
    print(f"AeroCom models: {aerocom_df['model'].nunique()}  "
          f"regions: {sorted(aerocom_df['region'].unique())}")
    aerocom_df.to_csv(TABLE_DIR / "aerocom_seasonal_wide.csv", index=False)

# %%
# -----------------------------------------------------------------------------
# Range check (after burden scaling)
# -----------------------------------------------------------------------------
print("\n--- Derived-variable range check (seasonal) ---")
range_df = pp.range_check(
    model_seasonal_df, ["MEC_ss", "MEC", "lifetime_ss", "lifetime_mass"],
    expected={"MEC_ss": (0.0, 15.0), "MEC": (0.0, 15.0)},
)
range_df.to_csv(TABLE_DIR / "ppe_range_check.csv", index=False)

# %%
# -----------------------------------------------------------------------------
# Observations (SPEXone for pd2024, POLDER for pd2010) and homogenization
# -----------------------------------------------------------------------------
obs_tables: Dict[str, pd.DataFrame] = {}
obs_means: Dict[str, Dict[str, dict]] = {}
homog_rows = []
for pid in PPE_IDS:
    if not ppe_cfg.get_run(pid).get("extractable", True):
        print(f"  Skip homogenization for {pid} (native ICON grid, no T63 ensemble)")
        continue
    obs_name = ppe_io.ppe_obs_name(pid)
    print(f"\n--- {pid}: {obs_name} ---")
    try:
        obs = ppe_io.load_obs_for_ppe(pid, ae_pair="550_865")
    except Exception as e:
        print(f"  {obs_name} load failed: {e}")
        continue
    period = ppe_cfg.get_run(pid)["period"]
    obs_tables[pid] = obs
    obs_means[pid] = ppe_io.obs_region_means(obs, ANALYSIS_REGIONS, period)
    seas = model_seasonal_df.loc[model_seasonal_df["ppe_id"] == pid].set_index("model")
    for region in ANALYSIS_REGIONS:
        m = obs_means[pid][region]
        print(f"  {region}: n={m['n']}  AOD={m['AOD_550']:.4f}  SSA={m['SSA']:.4f}  "
              f"AE={m['AE']:.3f}")
        sub = ppe_io.obs_in_region(obs, region, period)
        regional_all = seas.loc[seas["region"] == region]
        for var, obs_col in HOMOGENIZE_VARS.items():
            model_col = {"AOD": "od550aer", "AAOD": "abs550aer", "SSA": "SSA"}[var]
            sampled = ppe_io.sample_at_obs(pid, sub, var)
            res = ppe_io.homogenize(regional_all[model_col], sampled, m[obs_col])
            pp.homogenization_scatter(res, region, var, pid, obs_name, save_fig)
            homog_rows.append({
                "ppe_id": pid, "obs": obs_name, "region": region, "variable": var,
                **{k: res[k] for k in ("raw_mean", "a", "b", "r2", "homogenized",
                                       "n_models", "skipped")},
            })
            if not res["skipped"]:
                print(f"    {var}: sampled={res['a']:.3f}*regional{res['b']:+.4f} "
                      f"(R²={res['r2']:.2f})  raw={res['raw_mean']:.4f} -> "
                      f"homogenized={res['homogenized']:.4f}")

homog_df = pd.DataFrame(homog_rows)
if not homog_df.empty:
    homog_df.to_csv(TABLE_DIR / "obs_homogenized.csv", index=False)
    print(f"Wrote {TABLE_DIR / 'obs_homogenized.csv'}")


# %%
# -----------------------------------------------------------------------------
# Optical relations: MEC(_SS) and MAC vs SSA / 1 − SSA
# -----------------------------------------------------------------------------
df_plot = pp.with_one_minus_ssa(model_seasonal_df)
aer_plot = pp.with_one_minus_ssa(aerocom_df)
common = dict(regions=ANALYSIS_REGIONS, saver=save_fig, ppe_ids=PPE_IDS, labels=AXIS_LABELS,
              aerocom_df=aer_plot)

pp.scatter_by_region(df_plot, "SSA", "MEC", filename="ppe_SSA_vs_MEC.png",
                     title="PPE + AeroCom total aerosol: MEC vs SSA",
                     obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS), **common)
pp.scatter_by_region(df_plot, "1-SSA", "MEC", filename="ppe_1SSA_vs_MEC.png",
                     title="PPE + AeroCom total aerosol: MEC vs 1−SSA (through origin)",
                     through_origin=True, obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS, transform=lambda v: 1 - v), **common)
pp.scatter_by_region(df_plot, "1-SSA", "MAC", filename="ppe_1SSA_vs_MAC.png",
                     title="PPE + AeroCom total aerosol: MAC vs 1−SSA (through origin)",
                     through_origin=True, obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS, transform=lambda v: 1 - v), **common)
pp.scatter_by_region(df_plot, "SSA", "MEC_ss", filename="ppe_SSA_vs_MEC_ss.png",
                     title="PPE + AeroCom sea salt: MEC$_{SS}$ vs SSA",
                     obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS), **common)
pp.scatter_by_region(df_plot, "1-SSA", "MEC_ss", filename="ppe_1SSA_vs_MEC_ss.png",
                     title="PPE + AeroCom sea salt: MEC$_{SS}$ vs 1−SSA (through origin)",
                     through_origin=True, obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS, transform=lambda v: 1 - v), **common)
pp.scatter_by_region(df_plot, "AE", "MEC_ss", filename="ppe_MEC_ss_vs_AE.png",
                     title=r"PPE + AeroCom sea salt: MEC$_{SS}$ vs AE",
                     obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "AE", ANALYSIS_REGIONS), **common)
pp.scatter_by_region(df_plot, "AE", "MEC", filename="ppe_MEC_vs_AE.png",
                     title=r"PPE + AeroCom total aerosol: MEC vs AE",
                     obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "AE", ANALYSIS_REGIONS), **common)
pp.scatter_by_region(df_plot, "SSA", "od550aer", filename="ppe_AOD_vs_SSA.png",
                     title="PPE + AeroCom: AOD vs SSA", obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS),
                     obs_y=ppe_io.obs_lines_for(homog_df, obs_means, "AOD", ANALYSIS_REGIONS), **common)

# %%
# -----------------------------------------------------------------------------
# Lifetime relations: 1/τ vs precip (PPEs with precip) and vs AE
# -----------------------------------------------------------------------------
for lt, name in (("lifetime_ss", "sea salt"), ("lifetime_mass", "total aerosol")):
    if PRECIP_PPES:
        pp.scatter_by_region(df_plot, "precip", lt, inverse_y=True,
                             filename=f"ppe_inv_{lt}_vs_precip.png",
                             title=rf"PPE + AeroCom {name}: 1/$\tau$ vs precipitation",
                             **{**common, "ppe_ids": PRECIP_PPES})
    pp.scatter_by_region(df_plot, "AE", lt, inverse_y=True,
                         filename=f"ppe_inv_{lt}_vs_AE.png",
                         title=rf"PPE + AeroCom {name}: 1/$\tau$ vs AE",
                         obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "AE", ANALYSIS_REGIONS), **common)
    pp.lifetime_3d(model_seasonal_df, lt, ANALYSIS_REGIONS, save_fig,
                   filename=f"ppe_3d_inv_{lt}_vs_precip_AE.png",
                   title=rf"PPE {name}: 1/$\tau$ vs precip and AE",
                   labels=AXIS_LABELS, ppe_ids=PPE_IDS)

# %%
# -----------------------------------------------------------------------------
# Lifetime drivers: explore correlations BEFORE any multi-linear fit
# AeroCom CTRL has no u* / wind stress; ~11 models have ModelLevel rh.
# PPE 2024 has relhum_sfc + stress_water=hypot(ustrw,vstrw) (no friction velocity).
# Prefer u* when available; here RH + water stress are the PPE proxies.
# -----------------------------------------------------------------------------
METEO_PPE = "pd2024"
EXPLORE_LIFETIMES = [
    ("lifetime_ss", "sea salt"),
    ("lifetime_SO4", "sulfate"),
    ("lifetime_marine", "SS+SO4 marine"),
    ("lifetime_mass", "total aerosol (mass)"),
]
EXPLORE_PREDICTORS = ["AE", "precip", "relhum_sfc", "stress_water"]

# Merge sampled AeroCom surface RH (metno AP3 CTRL) onto the overlay table
_rh_csv = TABLE_DIR / "aerocom_relhum_sfc_regional_sample.csv"
if not aerocom_df.empty and _rh_csv.is_file():
    _rh = pd.read_csv(_rh_csv)
    # GISS filenames include "-2010"; strip trailing year tags for join
    _rh["model_key"] = _rh["model"].str.replace(r"-20\d{2}$", "", regex=True)
    aerocom_df = aerocom_df.copy()
    aerocom_df["model_key"] = aerocom_df["model"].astype(str)
    aerocom_df = aerocom_df.merge(
        _rh[["model_key", "region", "relhum_sfc"]].rename(columns={"relhum_sfc": "relhum_sfc"}),
        on=["model_key", "region"], how="left",
    )
    n_rh = int(aerocom_df["relhum_sfc"].notna().sum())
    print(f"AeroCom surface RH joined for {n_rh} seasonal rows "
          f"({aerocom_df.loc[aerocom_df['relhum_sfc'].notna(), 'model'].nunique()} models)")
    aerocom_df.to_csv(TABLE_DIR / "aerocom_seasonal_wide.csv", index=False)
elif not _rh_csv.is_file():
    print(f"No {_rh_csv.name}: AeroCom RH overlay skipped "
          "(see Data/AP3_meteo_sample + aerocom_meteo_inventory.csv)")

model_seasonal_df = ppe_io.add_lifetime_drivers(model_seasonal_df)
aerocom_df = ppe_io.add_lifetime_drivers(aerocom_df)
df_plot = pp.with_one_minus_ssa(model_seasonal_df)
aer_plot = pp.with_one_minus_ssa(aerocom_df)


def _season_month_mask(df: pd.DataFrame, regions: List[str]) -> pd.Series:
    keep = np.zeros(len(df), dtype=bool)
    months = pd.to_datetime(df["time"]).dt.month
    for region in regions:
        season = set(reg.REGIONS[region]["season_months"])
        keep |= (df["region"].to_numpy() == region) & months.isin(season).to_numpy()
    return pd.Series(keep, index=df.index)


def _corr_rows(df: pd.DataFrame, source: str, lifetimes, predictors, regions) -> List[dict]:
    rows = []
    for region in regions:
        sub = df.loc[df["region"] == region]
        for lt, _ in lifetimes:
            if lt not in sub.columns:
                continue
            y = 1.0 / sub[lt].astype(float)
            for xcol in predictors:
                if xcol not in sub.columns:
                    continue
                pair = pd.DataFrame({"x": sub[xcol].astype(float), "y": y}).replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                if len(pair) < 5:
                    continue
                r = float(pair["x"].corr(pair["y"]))
                rows.append(dict(
                    source=source, region=region, lifetime=lt, predictor=xcol,
                    r_inv_tau=r, n=len(pair),
                    x_min=float(pair["x"].min()), x_med=float(pair["x"].median()),
                    x_max=float(pair["x"].max()),
                ))
    return rows


print("\n--- Lifetime ↔ meteo exploration (Pearson r of 1/τ vs predictor) ---")
corr_rows = []
corr_rows += _corr_rows(df_plot, "PPE_seasonal", EXPLORE_LIFETIMES, EXPLORE_PREDICTORS, ANALYSIS_REGIONS)
corr_rows += _corr_rows(aer_plot, "AeroCom_seasonal", EXPLORE_LIFETIMES,
                        [c for c in EXPLORE_PREDICTORS if c != "stress_water"], ANALYSIS_REGIONS)

meteo_cols = [c for c in ("relhum_sfc", "stress_water") if c in model_monthly_df.columns]
mon_explore = pd.DataFrame()
if METEO_PPE in PPE_IDS and meteo_cols:
    mon_explore = model_monthly_df.loc[model_monthly_df["ppe_id"] == METEO_PPE].copy()
    mon_explore = mon_explore.loc[_season_month_mask(mon_explore, ANALYSIS_REGIONS)]
    mon_explore = ppe_io.add_lifetime_drivers(mon_explore)
    corr_rows += _corr_rows(mon_explore, "PPE_monthly_in_season", EXPLORE_LIFETIMES,
                            EXPLORE_PREDICTORS, ANALYSIS_REGIONS)

corr_df = pd.DataFrame(corr_rows)
corr_path = TABLE_DIR / "ppe_lifetime_meteo_correlations.csv"
if not corr_df.empty:
    corr_df.to_csv(corr_path, index=False)
    print(corr_df.pivot_table(index=["source", "lifetime", "predictor"], columns="region",
                              values="r_inv_tau").round(2).to_string())
    print(f"Saved: {corr_path}")

# Scatter panels: each lifetime × each available predictor (AeroCom diamonds)
common_explore = dict(regions=ANALYSIS_REGIONS, saver=save_fig, ppe_ids=PPE_IDS,
                      labels=AXIS_LABELS, aerocom_df=aer_plot)
for lt, lt_name in EXPLORE_LIFETIMES:
    if lt not in df_plot.columns:
        print(f"Skip scatters for {lt}: not in PPE table")
        continue
    for xcol in EXPLORE_PREDICTORS:
        if xcol not in df_plot.columns and xcol not in aer_plot.columns:
            continue
        # precip / PPE meteo: only PPEs that carry the column
        if xcol == "precip":
            ids = PRECIP_PPES
        elif xcol in ("relhum_sfc", "stress_water"):
            ids = [p for p in PPE_IDS if p == METEO_PPE]
        else:
            ids = PPE_IDS
        if not ids:
            continue
        if xcol not in df_plot.columns:
            continue
        aer_use = (
            aer_plot if (xcol in aer_plot.columns and aer_plot[xcol].notna().any())
            else None
        )
        pp.scatter_by_region(
            df_plot, xcol, lt, inverse_y=True,
            filename=f"ppe_inv_{lt}_vs_{xcol}.png",
            title=rf"PPE + AeroCom {lt_name}: 1/$\tau$ vs {AXIS_LABELS.get(xcol, xcol)}",
            **{**common_explore, "ppe_ids": ids, "aerocom_df": aer_use},
        )

# Optional nested OLS only after the exploration (monthly, pd2024, lifetime_ss)
NESTED_MODELS = {
    "AE": ["AE"],
    "AE+RH": ["AE", "relhum_sfc"],
    "AE+stress": ["AE", "stress_water"],
    "AE+RH+stress": ["AE", "relhum_sfc", "stress_water"],
    "AE+precip": ["AE", "precip"],
}


def _standardized_ols(y: np.ndarray, X: np.ndarray, names: List[str]) -> dict:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    A = np.column_stack([np.ones(len(y)), Z])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum((y - pred) ** 2) / ss_tot if ss_tot > 0 else np.nan
    raw = {names[i]: float(coef[i + 1] / sd[i]) for i in range(len(names))}
    std = {f"beta_{names[i]}": float(coef[i + 1]) for i in range(len(names))}
    intercept = float(coef[0] - np.sum(coef[1:] * mu / sd))
    return {"A": intercept, "r2": float(r2), "n": int(len(y)),
            "pred": pred, **raw, **std}


if METEO_PPE in PPE_IDS and meteo_cols and not mon_explore.empty:
    print(f"\n--- Nested monthly OLS for 1/τ_ss ({METEO_PPE}; after exploration) ---")
    mon = mon_explore.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["lifetime_ss", "AE"] + meteo_cols
    )
    mon = mon.loc[mon["lifetime_ss"] > 0].copy()
    mon["inv_lifetime_ss"] = 1.0 / mon["lifetime_ss"]
    coef_rows = []
    fig, axes = plt.subplots(
        len(ANALYSIS_REGIONS), 1,
        figsize=(6.5, 3.8 * len(ANALYSIS_REGIONS)), squeeze=False,
    )
    for ax, region in zip(axes[:, 0], ANALYSIS_REGIONS):
        sub = mon.loc[mon["region"] == region].copy()
        if len(sub) < 8:
            ax.set_title(f"{region}: too few months (n={len(sub)})")
            continue
        y = sub["inv_lifetime_ss"].to_numpy(float)
        full_names = [c for c in NESTED_MODELS["AE+RH+stress"] if c in sub.columns]
        full = _standardized_ols(y, sub[full_names].to_numpy(float), full_names)
        sub = sub.assign(inv_pred=full["pred"])
        for model_name, cols in NESTED_MODELS.items():
            cols = [c for c in cols if c in sub.columns]
            if not cols:
                continue
            fit = _standardized_ols(y, sub[cols].to_numpy(float), cols)
            row = {"ppe_id": METEO_PPE, "region": region, "model": model_name,
                   "A": fit["A"], "r2": fit["r2"], "n": fit["n"]}
            for c in EXPLORE_PREDICTORS:
                row[c] = fit.get(c, np.nan)
                row[f"beta_{c}"] = fit.get(f"beta_{c}", np.nan)
            coef_rows.append(row)
            print(f"  {region:18} {model_name:14} R²={fit['r2']:.3f}  n={fit['n']}")
        sc = ax.scatter(sub["inv_pred"], sub["inv_lifetime_ss"],
                        c=sub["stress_water"], s=12, cmap="viridis", alpha=0.7)
        lim = max(sub["inv_pred"].max(), sub["inv_lifetime_ss"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        fig.colorbar(sc, ax=ax, shrink=0.8).set_label(AXIS_LABELS["stress_water"])
        ax.set_xlabel(r"Predicted 1/$\tau_{SS}$ (day$^{-1}$)")
        ax.set_ylabel(r"Observed 1/$\tau_{SS}$ (day$^{-1}$)")
        ax.set_title(
            f"{region}: AE+RH+stress R²={full['r2']:.3f} "
            f"(β_AE={full.get('beta_AE', np.nan):+.2f}, "
            f"β_RH={full.get('beta_relhum_sfc', np.nan):+.2f}, "
            f"β_τ={full.get('beta_stress_water', np.nan):+.2f})",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        rf"{METEO_PPE}: monthly 1/$\tau_{{SS}}$ from AE, RH and water stress "
        r"(explore first; colour = $|\tau|_{\mathrm{water}}$)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    save_fig(fig, "ppe_inv_lifetime_ss_vs_AE_RH_stress.png")
    coef_df = pd.DataFrame(coef_rows)
    coef_path = TABLE_DIR / "ppe_lifetime_ss_meteo_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"Saved: {coef_path}")
else:
    print("Skip nested meteo OLS (need pd2024 monthly relhum_sfc / stress_water)")

# %%
# -----------------------------------------------------------------------------
# What changes sea-salt and sea-salt+sulfate lifetime (marine drivers)
# -----------------------------------------------------------------------------
# Emission lifetime only (PPE has no depss/wetss/dryss). Sulfate source is
# primary SO4 plus half of SO2 and DMS, converted to SO4 mass. DMS is the
# remote source; SO2 is the continental-pollution source.
# lifetime_drivers already added in the meteo-exploration block above.
marine = dict(regions=ANALYSIS_REGIONS, saver=save_fig, ppe_ids=PPE_IDS,
              labels=AXIS_LABELS, aerocom_df=aer_plot)
pp.scatter_by_region(df_plot, "ss_burden_frac", "lifetime_ss", inverse_y=True,
                     filename="ppe_inv_lifetime_ss_vs_ss_fraction.png",
                     title=r"Sea salt: 1/$\tau_{SS}$ vs sea-salt share of SS+SO4",
                     **marine)
for xcol, fname, title in (
    ("precip", "ppe_inv_lifetime_marine_vs_precip.png",
     r"SS+SO4: 1/$\tau$ vs precipitation"),
    ("AE", "ppe_inv_lifetime_marine_vs_AE.png",
     r"SS+SO4: 1/$\tau$ vs AE"),
    ("dms_source_frac", "ppe_inv_lifetime_marine_vs_DMS.png",
     r"SS+SO4: 1/$\tau$ vs DMS share of the sulfate source"),
    ("so2_source_frac", "ppe_inv_lifetime_marine_vs_SO2.png",
     r"SS+SO4: 1/$\tau$ vs SO2 share of the sulfate source"),
):
    ids = PRECIP_PPES if xcol == "precip" else PPE_IDS
    pp.scatter_by_region(df_plot, xcol, "lifetime_marine", inverse_y=True,
                         filename=fname, title=title,
                         **{**marine, "ppe_ids": ids})
if ("lifetime_marine_dep" in aer_plot.columns
        and np.isfinite(aer_plot["lifetime_marine_dep"].astype(float)).any()):
    pp.scatter_by_region(
        df_plot, "precip", "lifetime_marine", inverse_y=True,
        filename="ppe_inv_lifetime_marine_dep_vs_precip.png",
        title=r"SS+SO4 vs precipitation (AeroCom y is deposition lifetime)",
        aerocom_ycol="lifetime_marine_dep",
        **{**marine, "ppe_ids": PRECIP_PPES},
    )

# %%
# -----------------------------------------------------------------------------
# Observation maps (native lon/lat spacing, region boxes, centred at box middle)
# -----------------------------------------------------------------------------
import cameo_toolbox as ct

for pid, obs in obs_tables.items():
    if obs.empty or not SAVE_FIGURE:
        continue
    period = ppe_cfg.get_run(pid)["period"]
    obs_name = ppe_io.ppe_obs_name(pid)
    for col, cbar in (("AOD_550", r"AOD$_{550}$ [-]"), ("SSA", r"SSA [-]")):
        for region in ANALYSIS_REGIONS:
            sub = ppe_io.obs_in_region(obs, region, period)
            if sub.empty:
                continue
            lon = sub["lon360"].to_numpy(float)
            lat = sub["latitude"].to_numpy(float)
            lon_u = np.unique(np.round(lon, 6))
            lat_u = np.unique(np.round(lat, 6))
            if lon_u.size < 2 or lat_u.size < 2:
                continue
            dlon = float(np.median(np.diff(lon_u)))
            dlat = float(np.median(np.diff(lat_u)))
            lon_bin = np.round(lon / dlon) * dlon
            lat_bin = np.round(lat / dlat) * dlat
            grid = (
                pd.DataFrame({"lon": lon_bin, "lat": lat_bin, "z": sub[col].to_numpy(float)})
                .groupby(["lon", "lat"], as_index=False)["z"].mean()
            )
            lon_axis = np.sort(grid["lon"].unique())
            lat_axis = np.sort(grid["lat"].unique())
            z = (
                grid.pivot(index="lat", columns="lon", values="z")
                .reindex(index=lat_axis, columns=lon_axis)
                .to_numpy(float)
            )
            field = xr.DataArray(
                z, coords={"lat": lat_axis, "lon": lon_axis}, dims=("lat", "lon"),
            )
            clon, ext = boxes.extent_for_regions([region], pad=8.0)
            out = FIGURE_DIR / f"{pid}_{obs_name}_{col}_{region}.png"
            ct.plot_boxed_field(
                field, boxes=boxes.region_boxes([region]),
                title=f"{region} {obs_name} {col}",
                cbar_label=cbar, savefile=str(out), show=False,
                extent=ext, central_longitude=clon, land_facecolor="0.85",
            )
            print(f"  Saved {out.name}")

# %%
# -----------------------------------------------------------------------------
# Regional map: PPE 2024 ensemble-mean sea-salt AOD, DJF, with analysis boxes
# Native T63 lon/lat bins, land shaded, lon centred on each map's boxes.
# -----------------------------------------------------------------------------
print("\n--- Sea-salt regional map (PPE 2024, DJF od550ss) ---")
ss_field = boxes.ensemble_mean_aod("pd2024", ["od550ss"], months=(12, 1, 2))
atlantic = [r for r in ANALYSIS_REGIONS if r in reg.REGIONS]
clon_a, ext_a = boxes.extent_for_regions(atlantic, pad=12.0)
boxes.plot_study_map(
    ss_field, boxes.region_boxes(atlantic),
    title="PPE 2024 ensemble-mean sea-salt AOD$_{550}$ (DJF), South Atlantic",
    cbar_label=r"AOD$_{SS,550}$ [-]",
    filename=FIGURE_DIR / "ppe2024_seasalt_AOD_boxes_SAO.png",
    extent=ext_a, central_longitude=clon_a, land_facecolor="0.85",
)
boxes.plot_study_map(
    ss_field, [],
    title="PPE 2024 ensemble-mean sea-salt AOD$_{550}$ (DJF)",
    cbar_label=r"AOD$_{SS,550}$ [-]",
    filename=FIGURE_DIR / "ppe2024_seasalt_AOD_DJF_curved.png",
)

print("\nDone.")
print(f"Figures in: {FIGURE_DIR}")
print(f"Tables in:  {TABLE_DIR}")
