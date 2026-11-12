# %% [markdown]
# # AAOD error attribution on the Pace PPEs (SSA–MAC, then lifetime)
#
# Port of the first-stage relations in ``AAOD_error_attribution.ipynb`` onto the
# regional PPE store (``Data/PPE_regional``; ``py/ppe_regional_aggregates.py``).
# Each PPE in ``PPE_IDS`` is its own cloud:
#   * pd2024 (Aug 2024 - Jul 2025): SPEXone, GPCP 2024-25;
#   * pd2010 (2010, AeroCom / POLDER year): POLDER-GRASP, GPCP 2010.
# Emission-based lifetimes only (no deposition in the PPE extract); see
# ``ppe_io.LIFETIME_DEFS``. Precip-based regressions and the constrained
# lifetime are skipped for PPEs without a precip stream.
#
# Run: `sbatch run_notebook_or_py.sbatch notebooks/AAOD_error_attribution_PPE.py`

# %%
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

import cameo_toolbox as ct
import PPE_set as ppe_cfg
import ppe_io
import ppe_plots as pp
import ppe_precip_gpcp as pg
import regions as reg

print(f"Project root: {project_root}")

# %%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SPECIES = "bb"
PPE_IDS: List[str] = ["pd2024", "pd2010", "icon2025"]
EXCLUDE_MEMBERS: List[str] = []

MAC_X_COL = "1-SSA"  # theory: MAC = A*(1-SSA)
INTERCEPT_0 = True
BROWN_SSA_INTERCEPT = 0.969
BROWN_SSA_SLOPE = -0.779

SOURCE_REGIONS = ["africa", "amazon"]
# As in AAOD_error_attribution.ipynb: the outflow box has no local fire emission,
# so no load/emission lifetime there; its AAOD is predicted from African source
# E, tau, MAC with the Zhong et al. (2022) Eq. 6 meta-model.
OUTFLOW_REGION = "outflow_af"
OUTFLOW_SOURCE = reg.REGIONS[OUTFLOW_REGION]["source"]
ANALYSIS_REGIONS = SOURCE_REGIONS + [OUTFLOW_REGION]
LIFETIME_REGIONS = list(SOURCE_REGIONS)
REGIONS = {r: reg.REGIONS[r] for r in ANALYSIS_REGIONS}
REGION_COLORS = reg.region_colors(ANALYSIS_REGIONS)

LIFETIME_COLS = ["lifetime_BC_OA", "lifetime_prim", "lifetime", "lifetime_SO4",
                 "lifetime_S", "lifetime_naive"]

SAVE_FIG = True
FIGURE_DIR = project_root / "figure" / "PPE_bb"
TABLE_DIR = ppe_cfg.ANALYSIS_DIR / SPECIES
TABLE_DIR.mkdir(parents=True, exist_ok=True)
save_fig = pp.FigureSaver(FIGURE_DIR, save=SAVE_FIG)
GPCP_DIR = project_root / "Data" / "Prec"

AXIS_LABELS = {
    "SSA": r"SSA [-]",
    "1-SSA": r"1 − SSA [-]",
    "MAC": r"MAC$_{\mathrm{BC+OA}}$ (m$^{2}$ g$^{-1}$)",
    "MEC": r"MEC (m$^{2}$ g$^{-1}$)",
    "AE": r"AE 440–550 [-]",
    "precip": r"Model precipitation (mm day$^{-1}$)",
    pg.MATCHED_COL: r"Model precipitation, GPCP months (mm day$^{-1}$)",
    "rBC": r"rBC = BC/(BC+OA) [-]",
    "abs550aer": r"AAOD$_{550}$ [-]",
    "od550aer": r"AOD$_{550}$ [-]",
    "load_BC_OA": r"BC+OA burden (kg m$^{-2}$)",
    "emi_BC_OA": r"BC+OA emission (kg m$^{-2}$ s$^{-1}$)",
}
for _lt in LIFETIME_COLS:
    _lab = ppe_io.LIFETIME_DEFS[_lt][2]
    AXIS_LABELS[_lt] = rf"{_lab} lifetime $\tau$ (day)"
    AXIS_LABELS[f"inv_{_lt}"] = rf"1/$\tau$, {_lab} (day$^{{-1}}$)"

ISSUES: List[str] = []

print("--- Active configuration ---")
print(f"  PPE_IDS          = {PPE_IDS}")
print(f"  ANALYSIS_REGIONS = {ANALYSIS_REGIONS}")
for pid in PPE_IDS:
    run = ppe_cfg.get_run(pid)
    print(f"  {pid}: {run['period']}  burden_scale={run['burden_scale']}  "
          f"precip={run['has_precip']}  obs={run['obs']}")
print(f"  MAC_X_COL={MAC_X_COL!r}  INTERCEPT_0={INTERCEPT_0}")
print(f"  FIGURE_DIR = {FIGURE_DIR}")
print(f"  TABLE_DIR  = {TABLE_DIR}")

# %%
# -----------------------------------------------------------------------------
# Regional PPE tables
# -----------------------------------------------------------------------------
model_seasonal_df = ppe_io.load_regional(PPE_IDS, species=SPECIES, regions=ANALYSIS_REGIONS,
                                         seasonal=True, exclude_members=EXCLUDE_MEMBERS)
model_monthly_df = ppe_io.load_regional(PPE_IDS, species=SPECIES, regions=ANALYSIS_REGIONS,
                                        seasonal=False, exclude_members=EXCLUDE_MEMBERS)
PPE_IDS = [p for p in PPE_IDS if p in set(model_seasonal_df["ppe_id"])]
PRECIP_PPES = [p for p in PPE_IDS if ppe_io.has_precip(model_seasonal_df, p)]
for pid in PPE_IDS:
    if pid not in PRECIP_PPES:
        ISSUES.append(f"{pid}: no precipitation stream; precip regressions, GPCP matching "
                      "and the constrained lifetime are skipped for this PPE.")
print(f"Seasonal rows: {len(model_seasonal_df)}  PPEs with precip: {PRECIP_PPES}")
avail = [c for c in ["MAC", "SSA", "AE", "rBC", *LIFETIME_COLS, "precip", "abs550aer"]
         if c in model_seasonal_df.columns]
print(model_seasonal_df.groupby(["ppe_id", "region"])[avail].count())

# %%
# -----------------------------------------------------------------------------
# GPCP observed precipitation per PPE period; model precip over GPCP months
# -----------------------------------------------------------------------------
gpcp: Dict[str, dict] = {}
matched_parts = []
for pid in PPE_IDS:
    ym = ppe_io.ppe_year_months(pid)
    da, missing = pg.load_gpcp_months(GPCP_DIR, ym)
    for key in missing:
        ISSUES.append(f"{pid}: GPCP {key} missing in Data/Prec; excluded (not filled).")
    obs, rows, matched = pg.gpcp_region_tables(da, REGIONS, ym)
    rows.assign(ppe_id=pid).to_csv(TABLE_DIR / f"{pid}_gpcp_region_monthly.csv", index=False)
    gpcp[pid] = {"da": da, "obs": obs, "rows": rows, "matched": matched}
    for region in ANALYSIS_REGIONS:
        print(f"  {pid} {region}: GPCP season mean = {obs[region]:.3f} mm/day "
              f"over {sorted(matched[region])}")
    if pid in PRECIP_PPES:
        mon = model_monthly_df.loc[model_monthly_df["ppe_id"] == pid]
        matched_parts.append(pg.matched_model_precip(mon, matched))
if matched_parts:
    model_seasonal_df = model_seasonal_df.merge(pd.concat(matched_parts), on=["region", "model"],
                                                how="left")
else:
    model_seasonal_df[pg.MATCHED_COL] = np.nan

# %%
# -----------------------------------------------------------------------------
# Range checks
# -----------------------------------------------------------------------------
def lifetime_range_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in LIFETIME_COLS:
        for (pid, region), g in df.groupby(["ppe_id", "region"]):
            s = g[col].replace([np.inf, -np.inf], np.nan).dropna()
            s = s[s > 0]
            if s.empty:
                continue
            q = s.quantile([0.05, 0.5, 0.95])
            rows.append({
                "ppe_id": pid, "lifetime": col, "definition": ppe_io.LIFETIME_DEFS[col][2],
                "region": region, "tau_q05_d": q.loc[0.05], "tau_median_d": q.loc[0.5],
                "tau_q95_d": q.loc[0.95], "inv_tau_median_per_d": 1.0 / q.loc[0.5],
                "n": len(s),
            })
    return pd.DataFrame(rows)


print("\n--- Derived-variable range check (seasonal) ---")
rc = pp.range_check(model_seasonal_df, ["MAC", *LIFETIME_COLS], expected={"MAC": (0.0, 15.0)})
for r in rc.itertuples(index=False):
    if r.var == "MAC" and getattr(r, "frac_in_expected", 1.0) < 0.5:
        ISSUES.append(f"{r.ppe_id}: only {100 * r.frac_in_expected:.0f}% of MAC in [0, 15] "
                      "m2 g-1 (burden-scale issue?).")
    if r.var.startswith("lifetime") and np.isfinite(getattr(r, "inv_median_per_d", np.nan)):
        if not 0.01 <= r.inv_median_per_d <= 10:
            ISSUES.append(f"{r.ppe_id}: 1/{r.var} median={r.inv_median_per_d:.3g} d-1 outside "
                          "~0.01-10; check burden_scale / emission units.")
lt_table = lifetime_range_table(model_seasonal_df)
if not lt_table.empty:
    print("\nLifetime definitions by PPE / region (seasonal, days):")
    print(lt_table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    lt_table.to_csv(TABLE_DIR / "ppe_lifetime_range_check.csv", index=False)

precip_checks = []
for pid in PRECIP_PPES:
    sub = model_seasonal_df.loc[model_seasonal_df["ppe_id"] == pid,
                                ["region", "model", pg.MATCHED_COL]]
    tab = pg.precip_check_table(sub, gpcp[pid]["obs"], gpcp[pid]["matched"]).assign(ppe_id=pid)
    for r in tab.itertuples(index=False):
        print(f"  {pid} {r.region}: PPE median={r.ppe_median:.3f} [q05={r.ppe_q05:.3f}, "
              f"q95={r.ppe_q95:.3f}]  GPCP={r.precip_obs:.3f}")
        if np.isfinite(r.precip_obs) and not (r.ppe_min <= r.precip_obs <= r.ppe_max):
            ISSUES.append(f"{pid}: GPCP precip in {r.region} ({r.precip_obs:.3f} mm/day) outside "
                          f"the PPE range [{r.ppe_min:.3f}, {r.ppe_max:.3f}]; constrained "
                          "lifetime is an extrapolation.")
    precip_checks.append(tab)
if precip_checks:
    pd.concat(precip_checks).to_csv(TABLE_DIR / "ppe_precip_vs_gpcp.csv", index=False)

# %%
# -----------------------------------------------------------------------------
# Observations (SPEXone / POLDER), coverage and homogenization (AAOD, SSA, AE)
# -----------------------------------------------------------------------------
obs_tables: Dict[str, pd.DataFrame] = {}
obs_means: Dict[str, Dict[str, dict]] = {}
homog_rows, coverage_rows = [], []
for pid in PPE_IDS:
    if not ppe_cfg.get_run(pid).get("extractable", True):
        print(f"  Skip homogenization for {pid} (native ICON grid, no T63 ensemble)")
        continue
    obs_name = ppe_io.ppe_obs_name(pid)
    period = ppe_cfg.get_run(pid)["period"]
    print(f"\n--- {pid}: {obs_name} ---")
    try:
        obs = ppe_io.load_obs_for_ppe(pid, ae_pair="440_550")
    except Exception as e:
        print(f"  {obs_name} load failed: {e}")
        ISSUES.append(f"{pid}: {obs_name} load failed ({e}).")
        continue
    obs_tables[pid] = obs
    obs_means[pid] = ppe_io.obs_region_means(obs, ANALYSIS_REGIONS, period)
    seas = model_seasonal_df.loc[model_seasonal_df["ppe_id"] == pid].set_index("model")
    for region in ANALYSIS_REGIONS:
        m = obs_means[pid][region]
        print(f"  {region}: n={m['n']}  AAOD={m['AAOD_550']:.5f}  SSA={m['SSA']:.4f}  "
              f"AE={m['AE']:.3f}")
        sub = ppe_io.obs_in_region(obs, region, period)
        for mon in REGIONS[region]["season_months"]:
            n_mon = int((sub["date"].dt.month == mon).sum()) if not sub.empty else 0
            coverage_rows.append({"ppe_id": pid, "obs": obs_name, "region": region,
                                  "month": mon, "n": n_mon})
            if n_mon == 0:
                ISSUES.append(f"{pid}: {obs_name} has 0 rows in {region} month {mon}.")
        regional = seas.loc[seas["region"] == region]
        for var, obs_col, model_col in (("AAOD", "AAOD_550", "abs550aer"),
                                        ("SSA", "SSA", "SSA"), ("AE", "AE", "AE")):
            sampled = ppe_io.sample_at_obs(pid, sub, var)
            res = ppe_io.homogenize(regional[model_col], sampled, m[obs_col])
            pp.homogenization_scatter(res, region, var, pid, obs_name, save_fig)
            homog_rows.append({
                "ppe_id": pid, "obs": obs_name, "region": region, "variable": var,
                **{k: res[k] for k in ("raw_mean", "a", "b", "r2", "homogenized",
                                       "n_models", "skipped")},
            })
            if not res["skipped"]:
                print(f"    {var}: raw={res['raw_mean']:.5f} -> "
                      f"homogenized={res['homogenized']:.5f} (R²={res['r2']:.2f})")

homog_df = pd.DataFrame(homog_rows)
if not homog_df.empty:
    homog_df.to_csv(TABLE_DIR / "obs_homogenized.csv", index=False)
if coverage_rows:
    pd.DataFrame(coverage_rows).to_csv(TABLE_DIR / "obs_region_month_coverage.csv", index=False)


def gpcp_lines(regions=LIFETIME_REGIONS) -> Dict:
    out = {}
    for pid in PPE_IDS:
        yr = ppe_cfg.get_run(pid)["period"][0][:4]
        for r in regions:
            out[(pid, r)] = (gpcp[pid]["obs"].get(r, np.nan), f"GPCP ({pid}, {yr})")
    return out


# %%
# -----------------------------------------------------------------------------
# Study-region map on the pd2024 SPEXone (or first available obs) AOD
# -----------------------------------------------------------------------------
def plot_region_map(obs: pd.DataFrame, obs_name: str, filename: str):
    if obs.empty:
        return
    o = obs.assign(lat_bin=np.floor(obs["latitude"]) + 0.5,
                   lon_bin=np.floor(((obs["lon360"] + 180) % 360) - 180) + 0.5)
    grid = o.pivot_table(index="lat_bin", columns="lon_bin", values="AOD_550", aggfunc="mean")
    grid = grid.reindex(index=np.arange(-89.5, 90.0, 1.0), columns=np.arange(-179.5, 180.0, 1.0))
    boxes = pg.region_boxes_180(REGIONS)
    fig_map = ct.fake_uba_map(
        lon=grid.columns.to_numpy(), lat=grid.index.to_numpy(), c_array=grid.to_numpy(),
        zmin=0.0, zmax=1.0, labels=f"{obs_name} AOD 550 nm",
        title=f"{obs_name} AOD550 mean", region_boxes=boxes, show_region_labels=False,
        region_edgecolor="magenta", region_linewidth=2.0, cbar_orientation="vertical",
        cbar_extend="max", show=False,
    )
    pg.label_region_boxes(fig_map, boxes)
    save_fig(fig_map, filename)


for pid, obs in obs_tables.items():
    try:
        plot_region_map(obs, ppe_io.ppe_obs_name(pid), f"{pid}_region_map.png")
    except Exception as e:
        print(f"  Region map failed for {pid}: {e}")

# %%
# -----------------------------------------------------------------------------
# AeroCom AP3 (2010) regional seasonal table for overlay
# -----------------------------------------------------------------------------
AEROCOM_AGG_PARQUET = project_root / "tables" / "regional_seasonal_aggregates.parquet"
AEROCOM_EXCLUDE_MODELS = pg.AEROCOM_EXCLUDE_MODELS
AEROCOM_LIFETIME_VAR = "lifetime_BC_OA"  # load/deposition, 0.3-50 d filter


def load_aerocom_seasonal() -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not AEROCOM_AGG_PARQUET.is_file():
        ISSUES.append(f"AeroCom overlay skipped: {AEROCOM_AGG_PARQUET.name} not built.")
        return pd.DataFrame(), {}
    long = pd.read_parquet(AEROCOM_AGG_PARQUET)
    long = long[long["region"].isin(ANALYSIS_REGIONS)]
    gpcp_2010 = {
        r["region"]: float(r["value"])
        for _, r in long[(long["dataset"] == "GPCP") & (long["var"] == "precip")].iterrows()
    }
    mod = long[~long["dataset"].isin(["POLDER", "GPCP"] + AEROCOM_EXCLUDE_MODELS)
               & long["source"].isin(["aggregated", "derived"])]
    wide = (mod.pivot_table(index=["region", "dataset"], columns="var", values="value",
                            aggfunc="first").reset_index().rename(columns={"dataset": "model"}))
    wide.columns.name = None
    # Cache AE is per-cell 550/870; rebuild AE440-550 from regional-mean AODs.
    wide["AE_550_870"] = wide.get("AE")
    with np.errstate(divide="ignore", invalid="ignore"):
        wide["AE"] = -np.log(wide["od550aer"] / wide["od440aer"]) / np.log(550.0 / 440.0)
        wide["rBC"] = wide["loadbc"] / (wide["loadbc"] + wide["loadoa"])
    wide["lifetime_BC_OA"] = wide.get(AEROCOM_LIFETIME_VAR, np.nan)
    lt_emi = wide.get("lifetime_BC_OA_emission", pd.Series(np.nan, index=wide.index))
    wide["lifetime_BC_OA_emission"] = lt_emi.where((lt_emi >= 0.3) & (lt_emi <= 50.0))
    wide[pg.MATCHED_COL] = wide.get("precip", np.nan)
    no_lt = ~wide["region"].isin(LIFETIME_REGIONS)
    wide.loc[no_lt, ["lifetime_BC_OA", "lifetime_BC_OA_emission"]] = np.nan
    print(f"AeroCom models: {wide['model'].nunique()}")
    return wide, gpcp_2010


print("\n--- AeroCom overlay table ---")
aerocom_df, aerocom_gpcp_2010 = load_aerocom_seasonal()
if not aerocom_df.empty:
    aerocom_df.to_csv(TABLE_DIR / "aerocom_seasonal_wide.csv", index=False)
    ISSUES.append(
        "AeroCom overlay uses tables/regional_seasonal_aggregates.parquet (year 2010). Only "
        "pd2010 shares that year (same meteorology year as POLDER / GPCP 2010). AeroCom "
        "lifetime_BC_OA is load/deposition (0.3-50 d filter); PPE is load/emission, so "
        "lifetime_BC_OA_emission is also overlaid like-for-like."
    )

# %%
# -----------------------------------------------------------------------------
# Constrained lifetime: 1/tau = A + alpha*Pr + beta*AE at (GPCP, obs AE)
# -----------------------------------------------------------------------------
constrained_rows = []
for pid in PPE_IDS:
    for lt_col in [c for c in LIFETIME_COLS if c != "lifetime_naive"]:
        for region in LIFETIME_REGIONS:
            row = {"ppe_id": pid, "lifetime": lt_col, "region": region,
                   "precip_obs": gpcp[pid]["obs"].get(region, np.nan),
                   "AE_obs": ppe_io.lookup_obs_value(homog_df, obs_means, pid, region, "AE", source_regions=SOURCE_REGIONS)}
            if pid not in PRECIP_PPES:
                row["note"] = "no PPE precip"
                constrained_rows.append(row)
                continue
            sub = model_seasonal_df.loc[
                (model_seasonal_df["ppe_id"] == pid) & (model_seasonal_df["region"] == region),
                [pg.MATCHED_COL, "AE", lt_col],
            ].replace([np.inf, -np.inf], np.nan).dropna()
            sub = sub.loc[sub[lt_col] > 0]
            row["n"] = len(sub)
            if len(sub) >= 5:
                y = 1.0 / sub[lt_col].to_numpy(dtype=float)
                X = np.column_stack([np.ones(len(sub)), sub[pg.MATCHED_COL], sub["AE"]])
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - np.sum((y - X @ coef) ** 2) / ss_tot if ss_tot > 0 else np.nan
                inv_c = coef[0] + coef[1] * row["precip_obs"] + coef[2] * row["AE_obs"]
                row.update({
                    "A": coef[0], "alpha_pr": coef[1], "beta_ae": coef[2], "r2": r2,
                    "tau_ppe_median_d": float(np.median(sub[lt_col])),
                    "inv_tau_constrained": inv_c,
                    "tau_constrained_d": 1.0 / inv_c if inv_c > 0 else np.nan,
                })
            constrained_rows.append(row)
constrained_df = pd.DataFrame(constrained_rows)
constrained_df.to_csv(TABLE_DIR / "ppe_constrained_lifetime.csv", index=False)
print("\nConstrained lifetimes (GPCP precip + obs AE):")
print(constrained_df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# %%
# -----------------------------------------------------------------------------
# African outflow meta-model (Zhong et al. 2022 Eq. 6)
#   AAOD_out = A*(E*tau*MAC) + B*(E*tau) + C*MAC + D   [E, tau, MAC from africa]
# -----------------------------------------------------------------------------
def fit_outflow_meta(df: pd.DataFrame, label: str) -> Tuple[Optional[dict], pd.DataFrame]:
    need = {"emi_BC_OA", "lifetime_BC_OA", "MAC", "abs550aer"}
    if df.empty or not need <= set(df.columns):
        return None, pd.DataFrame()
    src = df.loc[df["region"] == OUTFLOW_SOURCE, ["model", "emi_BC_OA", "lifetime_BC_OA", "MAC"]]
    out = df.loc[df["region"] == OUTFLOW_REGION, ["model", "abs550aer"]]
    meta = src.merge(out.rename(columns={"abs550aer": "AAOD_out"}), on="model")
    meta = meta.replace([np.inf, -np.inf], np.nan).dropna()
    meta = meta.loc[meta["lifetime_BC_OA"] > 0]
    if len(meta) < 5:
        print(f"  {label}: insufficient members for outflow meta-model (n={len(meta)})")
        return None, meta
    e_tau = meta["emi_BC_OA"].to_numpy(float) * 1000.0 * 86400.0 * meta["lifetime_BC_OA"].to_numpy(float)
    mac = meta["MAC"].to_numpy(float)
    X = np.column_stack([e_tau * mac, e_tau, mac, np.ones(len(meta))])
    y = meta["AAOD_out"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    meta["AAOD_meta_fit"] = X @ coef
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - meta["AAOD_meta_fit"]) ** 2) / ss_tot if ss_tot > 0 else np.nan
    params = {"dataset": label, "A": coef[0], "B": coef[1], "C": coef[2], "D": coef[3],
              "r2": r2, "rmse": float(np.sqrt(np.mean((y - meta["AAOD_meta_fit"]) ** 2))),
              "n": len(meta)}
    print(f"  {label}: R^2={r2:.3f} n={len(meta)}")
    return params, meta


print("\n--- Outflow meta-model ---")
outflow_params, outflow_meta_rows = [], {}
_sources = [(pid, model_seasonal_df.loc[model_seasonal_df["ppe_id"] == pid]) for pid in PPE_IDS]
_sources.append(("AeroCom", aerocom_df))
for label, df_src in _sources:
    params, meta = fit_outflow_meta(df_src, label)
    outflow_meta_rows[label] = meta
    if params is not None:
        outflow_params.append(params)
if outflow_params:
    pd.DataFrame(outflow_params).to_csv(TABLE_DIR / "outflow_meta_model_params.csv", index=False)

# %%
# -----------------------------------------------------------------------------
# Model-side figures (PPE clouds only)
# -----------------------------------------------------------------------------
df_plot = pp.with_one_minus_ssa(model_seasonal_df)
aer_plot = pp.with_one_minus_ssa(aerocom_df)
common = dict(saver=save_fig, ppe_ids=PPE_IDS, labels=AXIS_LABELS)


def brown_line(ax, _region):
    xs = np.linspace(0, 0.6, 50)
    ax.plot(xs, BROWN_SSA_INTERCEPT + BROWN_SSA_SLOPE * xs, color="gray", ls=":", lw=1.4,
            label="Brown et al. 2021")


pp.scatter_by_region(df_plot, "SSA", "MAC", ANALYSIS_REGIONS, filename="ppe_MAC_vs_SSA.png",
                     title="PPE: MAC vs SSA", **common)
pp.scatter_by_region(df_plot, MAC_X_COL, "MAC", ANALYSIS_REGIONS,
                     filename="ppe_MAC_vs_1SSA.png", through_origin=INTERCEPT_0,
                     title="PPE: MAC vs 1−SSA" + (" (through origin)" if INTERCEPT_0 else ""),
                     **common)
pp.scatter_by_region(df_plot, "rBC", "SSA", ANALYSIS_REGIONS, filename="ppe_SSA_vs_rBC.png",
                     title="PPE: SSA vs rBC (Brown line)", extra=brown_line, **common)
if "MEC" in df_plot.columns:
    pp.scatter_by_region(df_plot, "AE", "MEC", ANALYSIS_REGIONS, filename="ppe_MEC_vs_AE.png",
                         title=r"PPE: MEC vs AE",
                         obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "AE", ANALYSIS_REGIONS,
                                                    source_regions=SOURCE_REGIONS),
                         **common)
for lt in ("lifetime_BC_OA", "lifetime"):
    if PRECIP_PPES:
        pp.scatter_by_region(df_plot, pg.MATCHED_COL, lt, LIFETIME_REGIONS, inverse_y=True,
                             filename=f"ppe_inv_{lt}_vs_precip.png",
                             title=rf"PPE: 1/$\tau$ ({ppe_io.LIFETIME_DEFS[lt][2]}) vs precipitation",
                             obs_x=gpcp_lines(), **{**common, "ppe_ids": PRECIP_PPES})
    pp.scatter_by_region(df_plot, "AE", lt, LIFETIME_REGIONS, inverse_y=True,
                         filename=f"ppe_inv_{lt}_vs_AE.png",
                         title=rf"PPE: 1/$\tau$ ({ppe_io.LIFETIME_DEFS[lt][2]}) vs AE",
                         obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "AE", LIFETIME_REGIONS, source_regions=SOURCE_REGIONS), **common)
    pp.lifetime_3d(df_plot, lt, LIFETIME_REGIONS, save_fig,
                   filename=f"ppe_3d_inv_{lt}_vs_precip_AE.png",
                   title=rf"PPE: 1/$\tau$ ({ppe_io.LIFETIME_DEFS[lt][2]}) vs precip and AE",
                   labels=AXIS_LABELS, ppe_ids=PPE_IDS)


def plot_lifetime_definitions(df, filename="ppe_lifetime_definitions.png"):
    cols = [c for c in LIFETIME_COLS if c in df.columns]
    fig, axes = plt.subplots(len(PPE_IDS), len(LIFETIME_REGIONS),
                             figsize=(5 * len(LIFETIME_REGIONS), 4.2 * len(PPE_IDS)),
                             squeeze=False, sharey=True)
    for i, pid in enumerate(PPE_IDS):
        for j, region in enumerate(LIFETIME_REGIONS):
            ax = axes[i, j]
            sub = df.loc[(df["ppe_id"] == pid) & (df["region"] == region)]
            vals = [sub[c].replace([np.inf, -np.inf], np.nan).dropna().loc[lambda s: s > 0]
                    .to_numpy() for c in cols]
            ax.boxplot(vals, showfliers=False)
            ax.set_xticks(range(1, len(cols) + 1))
            ax.set_xticklabels([c.replace("lifetime_", "").replace("lifetime", "total")
                                for c in cols], rotation=45, ha="right")
            ax.set_yscale("log")
            ax.set_title(f"{ppe_io.ppe_style(pid)['label']}: {region}")
            ax.grid(True, alpha=0.3, which="both")
        axes[i, 0].set_ylabel(r"Lifetime $\tau$ (day)")
    fig.suptitle("PPE emission-based lifetime definitions (season, regional)")
    fig.tight_layout()
    save_fig(fig, filename)


plot_lifetime_definitions(model_seasonal_df)

# %%
# -----------------------------------------------------------------------------
# Precipitation vs GPCP (PPEs with precip; same figures as PPE_precip_vs_GPCP.py)
# -----------------------------------------------------------------------------
_aer_precip = (aerocom_df[["region", "model", "precip"]]
               if "precip" in aerocom_df else pd.DataFrame(columns=["region", "model", "precip"]))
MAP_MONTHS = sorted({m for r in ANALYSIS_REGIONS for m in REGIONS[r]["season_months"]})
for pid in PRECIP_PPES:
    g = gpcp[pid]
    if g["da"] is None:
        continue
    pid_saver = lambda fig, name, _p=pid: save_fig(fig, f"{_p}_{name}")  # noqa: E731
    mon = model_monthly_df.loc[model_monthly_df["ppe_id"] == pid]
    seas = model_seasonal_df.loc[model_seasonal_df["ppe_id"] == pid]
    pg.plot_precip_seasonal_cycle(mon, g["rows"], REGIONS, REGION_COLORS,
                                  ppe_io.ppe_year_months(pid), pid_saver)
    pg.plot_precip_season_box(seas, _aer_precip, g["obs"], aerocom_gpcp_2010, g["matched"],
                              REGIONS, REGION_COLORS, pid_saver)
    try:
        with ppe_io.open_gridded(pid, ["precip"]) as ds:
            grid_precip = ds["precip"].drop_vars("experiment", errors="ignore").load()
        pg.plot_precip_maps(grid_precip, g["da"], REGIONS, pid_saver, MAP_MONTHS)
        pg.plot_precip_scatter(grid_precip, g["da"], REGIONS, pid_saver).to_csv(
            TABLE_DIR / f"{pid}_precip_ensmean_vs_gpcp_scatter_stats.csv", index=False)
        del grid_precip
    except FileNotFoundError as e:
        print(f"  Skip precip maps for {pid}: {e}")

# %%
# -----------------------------------------------------------------------------
# Combined PPE + AeroCom figures
# -----------------------------------------------------------------------------
combined = dict(common, aerocom_df=aer_plot)
if "MEC" in df_plot.columns:
    pp.scatter_by_region(df_plot, "AE", "MEC", ANALYSIS_REGIONS,
                         filename="combined_MEC_vs_AE.png",
                         title=r"ECHAM + ICON + AeroCom: MEC vs AE",
                         **combined)
pp.scatter_by_region(df_plot, "1-SSA", "MAC", ANALYSIS_REGIONS,
                     filename="combined_MAC_vs_1SSA.png", through_origin=INTERCEPT_0,
                     title="PPE + AeroCom: MAC vs 1−SSA" + (" (through origin)" if INTERCEPT_0 else ""),
                     obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS, transform=lambda v: 1 - v, source_regions=SOURCE_REGIONS), **combined)
pp.scatter_by_region(df_plot, "SSA", "MAC", ANALYSIS_REGIONS, filename="combined_MAC_vs_SSA.png",
                     title="PPE + AeroCom: MAC vs SSA", obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS, source_regions=SOURCE_REGIONS), **combined)
pp.scatter_by_region(df_plot, "rBC", "SSA", ANALYSIS_REGIONS, filename="combined_SSA_vs_rBC.png",
                     title="PPE + AeroCom: SSA vs rBC", obs_y=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS, source_regions=SOURCE_REGIONS),
                     extra=brown_line, **combined)
pp.scatter_by_region(df_plot, "SSA", "abs550aer", ANALYSIS_REGIONS,
                     filename="combined_AAOD_vs_SSA.png", title="PPE + AeroCom: AAOD vs SSA",
                     obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS, source_regions=SOURCE_REGIONS), obs_y=ppe_io.obs_lines_for(homog_df, obs_means, "AAOD", ANALYSIS_REGIONS, source_regions=SOURCE_REGIONS), **combined)
aer_gpcp_lines = {("AeroCom", r): (aerocom_gpcp_2010.get(r, np.nan), "GPCP 2010 (AeroCom yr)")
                  for r in LIFETIME_REGIONS}
for aer_lt, tag, desc in (("lifetime_BC_OA", "bcoa", "AeroCom load/deposition"),
                          ("lifetime_BC_OA_emission", "bcoa_aeroemi", "AeroCom load/emission")):
    pp.scatter_by_region(df_plot, pg.MATCHED_COL, "lifetime_BC_OA", LIFETIME_REGIONS,
                         inverse_y=True, aerocom_ycol=aer_lt,
                         filename=f"combined_inv_lifetime_{tag}_vs_precip.png",
                         title=rf"PPE (load/emission) + {desc}: 1/$\tau_{{BC+OA}}$ vs precipitation",
                         obs_x={**gpcp_lines(), **aer_gpcp_lines},
                         **{**combined, "ppe_ids": PRECIP_PPES})
    pp.scatter_by_region(df_plot, "AE", "lifetime_BC_OA", LIFETIME_REGIONS, inverse_y=True,
                         aerocom_ycol=aer_lt, filename=f"combined_inv_lifetime_{tag}_vs_AE.png",
                         title=rf"PPE (load/emission) + {desc}: 1/$\tau_{{BC+OA}}$ vs AE440-550",
                         obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "AE", LIFETIME_REGIONS, source_regions=SOURCE_REGIONS), **combined)


def plot_combined_outflow_meta(filename="combined_outflow_meta_model.png"):
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    vals = []
    for p in outflow_params:
        meta = outflow_meta_rows[p["dataset"]]
        if p["dataset"] == "AeroCom":
            style = pp.AEROCOM_STYLE
            lab = "AeroCom"
        else:
            st = ppe_io.ppe_style(p["dataset"])
            style = dict(s=16, alpha=0.5, color=st["color"], marker=st["marker"])
            lab = st["label"]
        ax.scatter(meta["AAOD_out"], meta["AAOD_meta_fit"], **style,
                   label=f"{lab} (R²={p['r2']:.2f}, n={p['n']})")
        vals += list(meta["AAOD_out"]) + list(meta["AAOD_meta_fit"])
    if not vals:
        plt.close(fig)
        return
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    ax.plot([lo, hi], [lo, hi], "k:", lw=1)
    for pid in obs_means:
        v = obs_means[pid].get(OUTFLOW_REGION, {}).get("AAOD_550", np.nan)
        if np.isfinite(v):
            ax.axvline(v, color=ppe_io.ppe_style(pid)["color"], ls="--", lw=1.1,
                       label=ppe_io.ppe_obs_name(pid))
    ax.set_xlabel(r"Model outflow AAOD$_{550}$ [-]")
    ax.set_ylabel(r"Meta-model outflow AAOD$_{550}$ [-]")
    ax.set_title("Outflow AAOD from African E·τ·MAC (Eq. 6)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, filename)


plot_combined_outflow_meta()

# %%
# -----------------------------------------------------------------------------
# What changes BC+OA lifetime (removal: precip and size; mixture: rBC)
# -----------------------------------------------------------------------------
model_seasonal_df = ppe_io.add_lifetime_drivers(model_seasonal_df)
if not aerocom_df.empty:
    aerocom_df = ppe_io.add_lifetime_drivers(aerocom_df)
df_plot = pp.with_one_minus_ssa(model_seasonal_df)
aer_plot = pp.with_one_minus_ssa(aerocom_df) if not aerocom_df.empty else aerocom_df
bb_drivers = dict(saver=save_fig, ppe_ids=PPE_IDS, labels=AXIS_LABELS, aerocom_df=aer_plot)
pp.scatter_by_region(df_plot, "rBC", "lifetime_BC_OA", LIFETIME_REGIONS, inverse_y=True,
                     filename="combined_inv_lifetime_BC_OA_vs_rBC.png",
                     title=r"BC+OA: 1/$\tau$ vs BC fraction of BC+OA",
                     **bb_drivers)
if ("lifetime_BC_OA_dep" in getattr(aer_plot, "columns", [])
        and np.isfinite(aer_plot["lifetime_BC_OA_dep"].astype(float)).any()):
    pp.scatter_by_region(
        df_plot, "precip", "lifetime_BC_OA", LIFETIME_REGIONS, inverse_y=True,
        filename="combined_inv_lifetime_BC_OA_dep_vs_precip.png",
        title=r"BC+OA vs precipitation (AeroCom y is deposition lifetime)",
        aerocom_ycol="lifetime_BC_OA_dep",
        **{**bb_drivers, "ppe_ids": PRECIP_PPES},
    )

# %%
# -----------------------------------------------------------------------------
# Potential issues / gap list
# -----------------------------------------------------------------------------
ISSUES.extend([
    "Missing PPE variables (PPE_set.MISSING_VARS): " + ", ".join(ppe_cfg.MISSING_VARS)
    + ". Deposition-based lifetimes are unavailable; emission lifetimes only.",
    "SO4 is mostly secondary: total/SO4 lifetimes use an SO4-equivalent source "
    f"(emi_SO4 + {ppe_io.SO2_TO_SO4_FRAC}*(96/64)*emiso2 + {ppe_io.SO2_TO_SO4_FRAC}*(96/62)*emi_DMS).",
    "Regional lifetimes are burden / local emission: transport in/out of the box biases them; "
    "no lifetime in outflow_af (Eq. 6 meta-model instead).",
    "PPE and SPEXone/POLDER AE use 440/550 nm; the AeroCom table AE (550/870) is rebuilt "
    "as 440/550 from regional-mean AODs.",
    "Burdens: pd2024 aerosol loads come from hifreq_monthly (monthly mean of 3-hourly "
    "output; no scaling). The *_burden.nc stream is ~230-260x lower; pd2010 only has that "
    "stream and is calibrated with the pd2024 stream factor per species "
    "(PPE_set.BURDEN_STREAM_FACTOR; see Data/PPE_regional/validation_<ppe>.csv).",
    "Seasons use month-of-year filtering; pd2024 fire seasons straddle Aug 2024 - Jul 2025, "
    "pd2010 seasons are within 2010.",
])
seen, unique_issues = set(), []
for item in ISSUES:
    if item not in seen:
        seen.add(item)
        unique_issues.append(item)
issues_path = TABLE_DIR / "potential_issues.md"
with open(issues_path, "w", encoding="utf-8") as f:
    f.write("# PPE AAOD attribution — potential issues\n\n")
    f.write("Generated by `notebooks/AAOD_error_attribution_PPE.py`.\n\n")
    for i, item in enumerate(unique_issues, 1):
        f.write(f"{i}. {item}\n")
print(f"Wrote {issues_path} ({len(unique_issues)} items)")

print("\nDone.")
print(f"Figures in: {FIGURE_DIR}")
print(f"Tables in:  {TABLE_DIR}")
