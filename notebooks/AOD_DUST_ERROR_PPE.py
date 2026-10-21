# %% [markdown]
# # Mineral dust error analysis on the Pace PPEs
#
# Same structure as ``AOD_SS_ERROR_PPE.py`` for the dust regions of
# ``py/regions.py``:
#   * sahara      (lon 343-30E, lat 15-30N, JJA)
#   * australia   (Lake Eyre basin, lon 130-145E, lat 32-22S, SON+DJF)
#   * atacama     (Atacama / Altiplano, lon 288-294E, lat 28-18S, ASON; ~3x5 T63 cells)
#   * outflow_sahara (tropical North Atlantic, lon 330-343E, lat 10-25N, JJA)
#
# Relations (each PPE in ``PPE_IDS`` is its own cloud, AeroCom AP3 diamonds):
#   * optics:   MEC_du / MAC_du vs SSA_du and 1 − SSA_du; column MEC vs SSA
#   * lifetime: 1/τ_du (load / emission) vs AE and precip (PPEs with precip)
#   * source:   dust emission vs ustar_acrit (time fraction above u* threshold)
#   * outflow:  AOD_dust(outflow) = A(EτMEC) + B(Eτ) + C·MEC + D from Saharan E, τ, MEC
# Observations: SPEXone for pd2024, POLDER-GRASP for pd2010 (column AOD / SSA / AE).
#
# Inputs:  Data/PPE_regional/*.parquet          (py/ppe_regional_aggregates.py)
#          tables/regional_seasonal_aggregates_du.parquet
#                                               (py/build_regional_aggregates.py --species du)
# Run: `sbatch run_notebook_or_py.sbatch notebooks/AOD_DUST_ERROR_PPE.py`

# %%
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
import ppe_io
import ppe_plots as pp
import regions as reg

print(f"Project root: {project_root}")

# %%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SPECIES = "du"
PPE_IDS: List[str] = ["pd2024", "pd2010"]
EXCLUDE_MEMBERS: List[str] = []

SOURCE_REGIONS = reg.source_regions(SPECIES)       # sahara, australia, atacama
OUTFLOW_REGION = "outflow_sahara"
OUTFLOW_SOURCE = reg.REGIONS[OUTFLOW_REGION]["source"]
ANALYSIS_REGIONS = SOURCE_REGIONS + [OUTFLOW_REGION]
LIFETIME_REGIONS = list(SOURCE_REGIONS)

AEROCOM_PARQUET = project_root / "tables" / "regional_seasonal_aggregates_du.parquet"
AEROCOM_EXCLUDE_MODELS = [
    "GISS-ModelE2p1p1-OMA_AP3-CTRL-2010",
    "GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010",
]
# Same window as the AeroCom BC+OA lifetime filter.
AEROCOM_LIFETIME_RANGE = (0.3, 50.0)

SAVE_FIGURE = True
FIGURE_DIR = project_root / "figure" / "PPE_du"
TABLE_DIR = ppe_cfg.ANALYSIS_DIR / SPECIES
TABLE_DIR.mkdir(parents=True, exist_ok=True)
save_fig = pp.FigureSaver(FIGURE_DIR, save=SAVE_FIGURE)

AXIS_LABELS = {
    "SSA": "Column SSA (1)",
    "1-SSA": "1 − column SSA (1)",
    "SSA_du": "Dust SSA (1)",
    "1-SSA_du": "1 − dust SSA (1)",
    "MEC": r"Column MEC (m$^{2}$ g$^{-1}$)",
    "MEC_du": r"MEC$_{DU}$ (m$^{2}$ g$^{-1}$)",
    "MAC_du": r"MAC$_{DU}$ (m$^{2}$ g$^{-1}$)",
    "AE": "AE 440-550 (1)",
    "precip": r"Precipitation (mm day$^{-1}$)",
    "lifetime_du": r"Dust lifetime $\tau_{DU}$ (day)",
    "inv_lifetime_du": r"1/$\tau_{DU}$ (day$^{-1}$)",
    "od550aer": "AOD$_{550}$ (1)",
    "od550dust": "AOD$_{DU,550}$ (1)",
    "dust_frac": "Dust share of AOD$_{550}$ (1)",
    "emi_du": r"Dust emission (kg m$^{-2}$ s$^{-1}$)",
    "ustar_acrit": r"Time fraction u$_*$ > threshold (%)",
}

ISSUES: List[str] = [
    "atacama covers only ~3x5 T63 cells incl. Andean terrain: regional means are noisy.",
    "australia season (Sep-Feb) wraps the calendar year: pd2010 uses Jan-Feb and Sep-Dec "
    "of 2010; pd2024 uses Sep 2024 - Feb 2025.",
    "Dust lifetime is load / local emission: in source boxes transport out of the box "
    "shortens it; no lifetime in outflow_sahara.",
    "Observations are column (total) AOD / SSA / AE; dust-only optics are model-side.",
]

print("--- Active configuration ---")
print(f"  PPE_IDS          = {PPE_IDS}")
print(f"  ANALYSIS_REGIONS = {ANALYSIS_REGIONS}")
for pid in PPE_IDS:
    run = ppe_cfg.get_run(pid)
    print(f"  {pid}: {run['period']}  burden_scale={run['burden_scale']}  "
          f"precip={run['has_precip']}  obs={run['obs']}")
print(f"  FIGURE_DIR = {FIGURE_DIR}")
print(f"  TABLE_DIR  = {TABLE_DIR}")

# %%
# -----------------------------------------------------------------------------
# Regional PPE tables
# -----------------------------------------------------------------------------
model_seasonal_df = ppe_io.load_regional(PPE_IDS, species=SPECIES, regions=ANALYSIS_REGIONS,
                                         seasonal=True, exclude_members=EXCLUDE_MEMBERS)
PPE_IDS = [p for p in PPE_IDS if p in set(model_seasonal_df["ppe_id"])]
PRECIP_PPES = [p for p in PPE_IDS if ppe_io.has_precip(model_seasonal_df, p)]
print(f"Seasonal rows: {len(model_seasonal_df)}  PPEs with precip: {PRECIP_PPES}")
avail = [c for c in ["od550dust", "MEC_du", "MAC_du", "SSA_du", "SSA", "AE", "lifetime_du",
                     "dust_frac", "ustar_acrit", "precip"] if c in model_seasonal_df.columns]
print(model_seasonal_df.groupby(["ppe_id", "region"])[avail].count())

print("\n--- Range check (seasonal) ---")
range_df = pp.range_check(
    model_seasonal_df, ["MEC_du", "MAC_du", "SSA_du", "dust_frac", "lifetime_du", "od550dust"],
    expected={"MEC_du": (0.1, 2.0), "SSA_du": (0.85, 1.0), "lifetime_du": (0.5, 15.0)},
)
range_df.to_csv(TABLE_DIR / "ppe_range_check.csv", index=False)
for r in range_df.itertuples(index=False):
    frac = getattr(r, "frac_in_expected", np.nan)
    if np.isfinite(frac) and frac < 0.5:
        ISSUES.append(f"{r.ppe_id}: only {100 * frac:.0f}% of {r.var} in the expected range.")

# %%
# -----------------------------------------------------------------------------
# AeroCom AP3 dust table (2010)
# -----------------------------------------------------------------------------
def load_aerocom_dust() -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, float]]]:
    if not AEROCOM_PARQUET.is_file():
        print(f"AeroCom dust table missing: {AEROCOM_PARQUET}")
        ISSUES.append("AeroCom dust overlay skipped: run "
                      "`python py/build_regional_aggregates.py --species du`.")
        return pd.DataFrame(), {}, {}
    long = pd.read_parquet(AEROCOM_PARQUET)
    long = long[long["region"].isin(ANALYSIS_REGIONS)]
    gpcp_2010 = {r["region"]: float(r["value"]) for _, r in
                 long[(long["dataset"] == "GPCP") & (long["var"] == "precip")].iterrows()}
    polder = {}
    for _, r in long[(long["dataset"] == "POLDER")].iterrows():
        polder.setdefault(r["region"], {})[f"{r['var']}_{r['source']}"] = float(r["value"])
    mod = long[~long["dataset"].isin(["POLDER", "GPCP"] + AEROCOM_EXCLUDE_MODELS)
               & long["source"].isin(["aggregated", "derived"])]
    wide = (mod.pivot_table(index=["region", "dataset"], columns="var", values="value",
                            aggfunc="first").reset_index().rename(columns={"dataset": "model"}))
    wide.columns.name = None
    with np.errstate(divide="ignore", invalid="ignore"):
        if {"od550aer", "od440aer"} <= set(wide.columns):
            wide["AE_550_870"] = wide.get("AE")
            wide["AE"] = -np.log(wide["od550aer"] / wide["od440aer"]) / np.log(550.0 / 440.0)
        if {"od550dust", "od550aer"} <= set(wide.columns):
            wide["dust_frac"] = wide["od550dust"] / wide["od550aer"]
        if {"loaddust", "emidust"} <= set(wide.columns):
            wide["lifetime_du_emission"] = wide["loaddust"] / (wide["emidust"] * 86400.0)
    if "lifetime_du_emission" in wide:
        lo, hi = AEROCOM_LIFETIME_RANGE
        lt = wide["lifetime_du_emission"]
        wide["lifetime_du_emission"] = lt.where((lt >= lo) & (lt <= hi))
    if {"emi_du"} <= set(wide.columns):
        wide["emi_du"] = wide["emi_du"]
    elif "emidust" in wide:
        wide["emi_du"] = wide["emidust"]
    no_lt = ~wide["region"].isin(LIFETIME_REGIONS)
    for c in ("lifetime_du", "lifetime_du_emission"):
        if c in wide:
            wide.loc[no_lt, c] = np.nan
    print(f"AeroCom dust models: {wide['model'].nunique()}")
    return wide, gpcp_2010, polder


print("\n--- AeroCom dust overlay ---")
aerocom_df, aerocom_gpcp_2010, aerocom_polder = load_aerocom_dust()
if not aerocom_df.empty:
    aerocom_df.to_csv(TABLE_DIR / "aerocom_seasonal_wide.csv", index=False)

# %%
# -----------------------------------------------------------------------------
# Observations and homogenization (AOD, SSA, AE column quantities)
# -----------------------------------------------------------------------------
obs_means: Dict[str, Dict[str, dict]] = {}
homog_rows = []
for pid in PPE_IDS:
    obs_name = ppe_io.ppe_obs_name(pid)
    period = ppe_cfg.get_run(pid)["period"]
    print(f"\n--- {pid}: {obs_name} ---")
    try:
        obs = ppe_io.load_obs_for_ppe(pid, ae_pair="440_550")
    except Exception as e:
        print(f"  {obs_name} load failed: {e}")
        ISSUES.append(f"{pid}: {obs_name} load failed ({e}).")
        continue
    obs_means[pid] = ppe_io.obs_region_means(obs, ANALYSIS_REGIONS, period)
    seas = model_seasonal_df.loc[model_seasonal_df["ppe_id"] == pid].set_index("model")
    for region in ANALYSIS_REGIONS:
        m = obs_means[pid][region]
        print(f"  {region}: n={m['n']}  AOD={m['AOD_550']:.4f}  SSA={m['SSA']:.4f}  "
              f"AE={m['AE']:.3f}")
        if m["n"] == 0:
            ISSUES.append(f"{pid}: no {obs_name} rows in {region} season.")
            continue
        sub = ppe_io.obs_in_region(obs, region, period)
        regional = seas.loc[seas["region"] == region]
        for var, obs_col, model_col in (("AOD", "AOD_550", "od550aer"), ("SSA", "SSA", "SSA"),
                                        ("AE", "AE", "AE")):
            res = ppe_io.homogenize(regional[model_col], ppe_io.sample_at_obs(pid, sub, var),
                                    m[obs_col])
            pp.homogenization_scatter(res, region, var, pid, obs_name, save_fig)
            homog_rows.append({
                "ppe_id": pid, "obs": obs_name, "region": region, "variable": var,
                **{k: res[k] for k in ("raw_mean", "a", "b", "r2", "homogenized",
                                       "n_models", "skipped")},
            })
    del obs

homog_df = pd.DataFrame(homog_rows)
if not homog_df.empty:
    homog_df.to_csv(TABLE_DIR / "obs_homogenized.csv", index=False)


# %%
# -----------------------------------------------------------------------------
# Optical relations
# -----------------------------------------------------------------------------
df_plot = pp.with_one_minus_ssa(pp.with_one_minus_ssa(model_seasonal_df), "SSA_du")
aer_plot = pp.with_one_minus_ssa(aerocom_df)
common = dict(saver=save_fig, ppe_ids=PPE_IDS, labels=AXIS_LABELS)

pp.scatter_by_region(df_plot, "SSA_du", "MEC_du", ANALYSIS_REGIONS,
                     filename="ppe_MEC_du_vs_SSA_du.png", title="PPE dust: MEC$_{DU}$ vs dust SSA",
                     **common)
pp.scatter_by_region(df_plot, "1-SSA_du", "MAC_du", ANALYSIS_REGIONS,
                     filename="ppe_MAC_du_vs_1SSA_du.png", through_origin=True,
                     title="PPE dust: MAC$_{DU}$ vs 1 − dust SSA (through origin)", **common)
pp.scatter_by_region(df_plot, "SSA", "MEC", ANALYSIS_REGIONS,
                     filename="combined_MEC_vs_SSA.png", title="PPE + AeroCom: column MEC vs SSA",
                     aerocom_df=aer_plot, obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS), **common)
pp.scatter_by_region(df_plot, "SSA", "MEC_du", ANALYSIS_REGIONS,
                     filename="combined_MEC_du_vs_SSA.png",
                     title="PPE + AeroCom: MEC$_{DU}$ vs column SSA",
                     aerocom_df=aer_plot, obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS), **common)
pp.scatter_by_region(df_plot, "dust_frac", "SSA", ANALYSIS_REGIONS,
                     filename="combined_SSA_vs_dust_frac.png",
                     title="PPE + AeroCom: column SSA vs dust share of AOD",
                     aerocom_df=aer_plot, obs_y=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS), **common)
pp.scatter_by_region(df_plot, "SSA", "od550aer", ANALYSIS_REGIONS,
                     filename="combined_AOD_vs_SSA.png", title="PPE + AeroCom: AOD vs SSA",
                     aerocom_df=aer_plot, obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS), obs_y=ppe_io.obs_lines_for(homog_df, obs_means, "AOD", ANALYSIS_REGIONS),
                     **common)

# %%
# -----------------------------------------------------------------------------
# Lifetime and source relations
# -----------------------------------------------------------------------------
pp.scatter_by_region(df_plot, "AE", "lifetime_du", LIFETIME_REGIONS, inverse_y=True,
                     filename="combined_inv_lifetime_du_vs_AE.png",
                     title=r"PPE (load/emission) + AeroCom (load/emission): 1/$\tau_{DU}$ vs AE",
                     aerocom_df=aer_plot, aerocom_ycol="lifetime_du_emission",
                     obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "AE", LIFETIME_REGIONS), **common)
if PRECIP_PPES:
    pp.scatter_by_region(df_plot, "precip", "lifetime_du", LIFETIME_REGIONS, inverse_y=True,
                         filename="combined_inv_lifetime_du_vs_precip.png",
                         title=r"PPE + AeroCom: 1/$\tau_{DU}$ vs precipitation",
                         aerocom_df=aer_plot, aerocom_ycol="lifetime_du_emission",
                         obs_x={("AeroCom", r): (aerocom_gpcp_2010.get(r, np.nan), "GPCP 2010")
                                for r in LIFETIME_REGIONS},
                         **{**common, "ppe_ids": PRECIP_PPES})
pp.lifetime_3d(df_plot, "lifetime_du", LIFETIME_REGIONS, save_fig,
               filename="ppe_3d_inv_lifetime_du_vs_precip_AE.png",
               title=r"PPE dust: 1/$\tau_{DU}$ vs precip and AE", labels=AXIS_LABELS,
               ppe_ids=PPE_IDS)
pp.scatter_by_region(df_plot, "ustar_acrit", "emi_du", LIFETIME_REGIONS,
                     filename="ppe_emi_du_vs_ustar_acrit.png",
                     title="PPE dust: emission vs u$_*$-threshold exceedance", **common)
pp.scatter_by_region(df_plot, "emi_du", "od550dust", LIFETIME_REGIONS,
                     filename="combined_AOD_du_vs_emi_du.png",
                     title="PPE + AeroCom: dust AOD vs dust emission",
                     aerocom_df=aer_plot, **common)

# %%
# -----------------------------------------------------------------------------
# Saharan outflow meta-model (Zhong et al. 2022 Eq. 6 analogue for dust)
#   AOD_du(outflow) = A*(E*tau*MEC) + B*(E*tau) + C*MEC + D   [sahara E, tau, MEC]
# -----------------------------------------------------------------------------
def fit_outflow_meta(df: pd.DataFrame, label: str, lt_col: str = "lifetime_du"
                     ) -> Tuple[Optional[dict], pd.DataFrame]:
    need = {"emi_du", lt_col, "MEC_du", "od550dust"}
    if df.empty or not need <= set(df.columns):
        return None, pd.DataFrame()
    src = df.loc[df["region"] == OUTFLOW_SOURCE, ["model", "emi_du", lt_col, "MEC_du"]]
    out = df.loc[df["region"] == OUTFLOW_REGION, ["model", "od550dust"]]
    meta = src.merge(out.rename(columns={"od550dust": "AOD_out"}), on="model")
    meta = meta.replace([np.inf, -np.inf], np.nan).dropna()
    meta = meta.loc[meta[lt_col] > 0]
    if len(meta) < 5:
        print(f"  {label}: insufficient members for outflow meta-model (n={len(meta)})")
        return None, meta
    e_tau = meta["emi_du"].to_numpy(float) * 1000.0 * 86400.0 * meta[lt_col].to_numpy(float)
    mec = meta["MEC_du"].to_numpy(float)
    X = np.column_stack([e_tau * mec, e_tau, mec, np.ones(len(meta))])
    y = meta["AOD_out"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    meta["AOD_meta_fit"] = X @ coef
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - meta["AOD_meta_fit"]) ** 2) / ss_tot if ss_tot > 0 else np.nan
    print(f"  {label}: R^2={r2:.3f} n={len(meta)}")
    return {"dataset": label, "A": coef[0], "B": coef[1], "C": coef[2], "D": coef[3],
            "r2": r2, "n": len(meta)}, meta


print("\n--- Saharan outflow meta-model ---")
outflow_params, outflow_meta = [], {}
for pid in PPE_IDS:
    params, meta = fit_outflow_meta(model_seasonal_df.loc[model_seasonal_df["ppe_id"] == pid], pid)
    outflow_meta[pid] = meta
    if params:
        outflow_params.append(params)
params, meta = fit_outflow_meta(aerocom_df, "AeroCom", lt_col="lifetime_du_emission")
outflow_meta["AeroCom"] = meta
if params:
    outflow_params.append(params)
if outflow_params:
    pd.DataFrame(outflow_params).to_csv(TABLE_DIR / "outflow_meta_model_params.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    vals = []
    for p in outflow_params:
        m = outflow_meta[p["dataset"]]
        if p["dataset"] == "AeroCom":
            style, lab = pp.AEROCOM_STYLE, "AeroCom"
        else:
            st = ppe_io.ppe_style(p["dataset"])
            style, lab = dict(s=16, alpha=0.5, color=st["color"], marker=st["marker"]), st["label"]
        ax.scatter(m["AOD_out"], m["AOD_meta_fit"], **style,
                   label=f"{lab} (R²={p['r2']:.2f}, n={p['n']})")
        vals += list(m["AOD_out"]) + list(m["AOD_meta_fit"])
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    ax.plot([lo, hi], [lo, hi], "k:", lw=1)
    ax.set_xlabel("Model outflow dust AOD$_{550}$ (1)")
    ax.set_ylabel("Meta-model outflow dust AOD$_{550}$ (1)")
    ax.set_title("Saharan outflow dust AOD from E·τ·MEC")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "combined_outflow_meta_model.png")

# %%
# -----------------------------------------------------------------------------
# Issues
# -----------------------------------------------------------------------------
for pid in PPE_IDS:
    if pid not in PRECIP_PPES:
        ISSUES.append(f"{pid}: no precipitation stream; precip relations skipped.")
issues_path = TABLE_DIR / "potential_issues.md"
with open(issues_path, "w", encoding="utf-8") as f:
    f.write("# PPE dust analysis — potential issues\n\n")
    for i, item in enumerate(dict.fromkeys(ISSUES), 1):
        f.write(f"{i}. {item}\n")
print(f"Wrote {issues_path}")
print("\nDone.")
print(f"Figures in: {FIGURE_DIR}")
print(f"Tables in:  {TABLE_DIR}")
