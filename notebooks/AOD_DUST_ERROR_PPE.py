# %% [markdown]
# # Mineral dust error analysis on the Pace PPEs
#
# Same structure as ``AOD_SS_ERROR_PPE.py`` for the dust regions of
# ``py/regions.py``:
#   * sahara      (lon 342-30E, lat 13-30N, JJA, land only)
#   * australia   (lon 115-145E, lat 32-18S, Sep-Feb, land only)
#   * atacama     (lon 288-300E, lat 32-20S, ASON, land only)
#   * outflow_sahara (Atlantic, lon 310-342E, lat 13-30N, JJA, ocean only)
#
# Relations (each PPE in ``PPE_IDS`` is its own cloud, AeroCom AP3 diamonds):
#   * optics:   MEC_du / MAC_du vs SSA_du and 1 − SSA_du; column MEC vs SSA
#   * lifetime: 1/τ_du (load / emission) vs AE and precip (PPEs with precip)
#   * source:   dust emission vs ustar_acrit (time fraction above u* threshold)
#   * outflow:  AOD_dust(outflow) = A(EτMEC) + B(Eτ) + C·MEC + D from Saharan E, τ, MEC
# Observations: SPEXone for pd2024, POLDER-GRASP for pd2010 (column AOD / SSA / AE).
#
# Pipeline (tables first, then this script):
#   1. py/ppe_regional_aggregates.py --ppe pd2024     ECHAM 2024-25 seasonal rows
#   2. pd2010 member burden/emi files + optical ensemble
#      (the 2010 burden ensemble NetCDF is not on scratch)
#   3. py/icon_regional.py                            ICON-HAM native-grid rows
#   4. py/build_regional_aggregates.py --legacy --species du
#                                                     AeroCom AP3 seasonal rows
#   5. notebooks/AOD_DUST_ERROR_PPE.py                all figure/PPE_du plots
# Regions and seasons come from py/regions.py. ICON-HAM is omitted on Atacama.
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
PPE_IDS: List[str] = ["pd2024", "pd2010", "icon2025"]
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
    "SSA": r"Column SSA [-]",
    "1-SSA": r"1 − column SSA [-]",
    "SSA_du": r"Dust SSA [-]",
    "1-SSA_du": r"1 − dust SSA [-]",
    "MEC": r"Column MEC (m$^{2}$ g$^{-1}$)",
    "MEC_du": r"MEC$_{DU}$ (m$^{2}$ g$^{-1}$)",
    "MAC": r"Column MAC (m$^{2}$ g$^{-1}$)",
    "MAC_du": r"MAC$_{DU}$ (m$^{2}$ g$^{-1}$)",
    "AE": r"AE [-]",
    "precip": r"Precipitation (mm day$^{-1}$)",
    "lifetime_du": r"Dust lifetime $\tau_{DU}$ (day)",
    "inv_lifetime_du": r"1/$\tau_{DU}$ (day$^{-1}$)",
    "od550aer": r"AOD$_{550}$ [-]",
    "od550dust": r"AOD$_{DU,550}$ [-]",
    "dust_frac": r"Dust share of AOD$_{550}$ [-]",
    "emi_du": r"Dust emission (kg m$^{-2}$ s$^{-1}$)",
    "ustar_acrit": r"Time fraction u$_*$ > threshold (%)",
}

def _season_label(months) -> str:
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months = list(months)
    if months == list(range(months[0], months[0] + len(months))) or months[0] > months[-1]:
        return f"{names[months[0] - 1]}–{names[months[-1] - 1]}"
    return ", ".join(names[m - 1] for m in months)


SEASON_TITLES = {
    name: f"{name} ({_season_label(reg.REGIONS[name]['season_months'])})"
    for name in ANALYSIS_REGIONS
}

ISSUES: List[str] = [
    "atacama covers only ~3x5 T63 cells incl. Andean terrain: regional means are noisy.",
    "australia season (Sep-Feb) wraps the calendar year: pd2010 uses Jan-Feb and Sep-Dec "
    "of 2010; pd2024 uses Sep 2024 - Feb 2025.",
    "Dust lifetime is load / local emission: in source boxes transport out of the box "
    "shortens it; no lifetime in outflow_sahara.",
    "ECHAM PPE streams have no dust deposition flux, so a load/deposition lifetime "
    "exists for AeroCom (depdust) only. ICON archives dry deposition alone, which is "
    "not the total flux a satellite deposition product would constrain.",
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
# ICON-HAM has no Atacama dust source (emission is identically zero). Drop those
# rows so the member never appears on an Atacama panel.
model_seasonal_df = model_seasonal_df.loc[
    ~((model_seasonal_df["ppe_id"] == "icon2025") & (model_seasonal_df["region"] == "atacama"))
]
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
    if {"abs550aer", "load_total"} <= set(wide.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            wide["MAC"] = wide["abs550aer"] / (wide["load_total"].astype(float) * 1e3)
        wide.loc[~(wide["load_total"] > 0), "MAC"] = np.nan
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
                     title="ECHAM + ICON + AeroCom: MAC$_{DU}$ vs 1 − dust SSA (through origin)",
                     aerocom_df=aer_plot, **common)
pp.density_by_region(
    df_plot, "1-SSA_du", "MAC_du", ANALYSIS_REGIONS, save_fig,
    filename="ppe_MAC_du_vs_1SSA_du_density.png",
    title=r"Dust: MAC$_{DU}$ vs 1−SSA — three PPE densities, AeroCom points",
    labels=AXIS_LABELS, aerocom_df=aer_plot,
)
pp.scatter_by_region(
    df_plot, "1-SSA", "MAC", ANALYSIS_REGIONS,
    filename="ppe_MAC_vs_1SSA.png", through_origin=True,
    title="PPE + AeroCom: column MAC vs 1 − column SSA (through origin)",
    aerocom_df=aer_plot, **common,
)
pp.density_by_region(
    df_plot, "1-SSA", "MAC", ANALYSIS_REGIONS, save_fig,
    filename="ppe_MAC_vs_1SSA_density.png",
    title=r"Column MAC vs 1−SSA — three PPE densities, AeroCom points",
    labels=AXIS_LABELS, aerocom_df=aer_plot,
)
pp.scatter_by_region(df_plot, "SSA", "MEC", ANALYSIS_REGIONS,
                     filename="combined_MEC_vs_SSA.png", title="PPE + AeroCom: column MEC vs SSA",
                     aerocom_df=aer_plot, obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS), **common)
pp.scatter_by_region(df_plot, "SSA", "MEC_du", ANALYSIS_REGIONS,
                     filename="combined_MEC_du_vs_SSA.png",
                     title="PPE + AeroCom: MEC$_{DU}$ vs column SSA",
                     aerocom_df=aer_plot, obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "SSA", ANALYSIS_REGIONS), **common)
pp.scatter_by_region(df_plot, "AE", "MEC_du", ANALYSIS_REGIONS,
                     filename="combined_MEC_du_vs_AE.png",
                     title=r"PPE + AeroCom: MEC$_{DU}$ vs AE",
                     aerocom_df=aer_plot,
                     obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "AE", ANALYSIS_REGIONS), **common)
pp.density_by_region(
    df_plot, "AE", "MEC_du", ANALYSIS_REGIONS, save_fig,
    filename="combined_MEC_du_vs_AE_density.png",
    title=r"Dust: MEC$_{DU}$ vs AE — three PPE densities, AeroCom points",
    labels=AXIS_LABELS, aerocom_df=aer_plot,
)
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
# AeroCom lifetime_du in the dust table is load / total deposition (depdust).
# The ECHAM PPEs do not archive that flux. ICON dry deposition is not used here:
# in the Sahara it is a few tenths of a percent of the emission, so a dry-only
# lifetime is hundreds of days and is not a total deposition constraint.
if ("lifetime_du" in getattr(aer_plot, "columns", [])
        and np.isfinite(aer_plot["lifetime_du"].astype(float)).any()):
    pp.scatter_by_region(
        df_plot.iloc[0:0], "AE", "lifetime_du", LIFETIME_REGIONS, inverse_y=True,
        filename="combined_inv_lifetime_du_dep_vs_AE.png",
        title=r"AeroCom: 1/$\tau_{DU}$ from deposition (load / depdust) vs AE",
        ylabel=r"1/$\tau_{DU}$ from deposition (day$^{-1}$)",
        aerocom_df=aer_plot, aerocom_ycol="lifetime_du",
        obs_x=ppe_io.obs_lines_for(homog_df, obs_means, "AE", LIFETIME_REGIONS),
        **common,
    )

def _dust_lifetime_boxes():
    """1/τ spread. Each panel keeps its own y-scale.

    Outflow sits beside Sahara and is narrower: it only has deposition
    lifetimes (2024 total deposition and AeroCom depdust). The y-limit there
    is fixed at 0.16 day^-1; source panels are not clipped.
    """
    short = {"pd2024": "2024", "pd2010": "2010", "icon2025": "ICON"}
    groups = [(pid, "lifetime_du", df_plot, short[pid], ppe_io.ppe_style(pid)["color"])
              for pid in PPE_IDS]
    groups.append(("AeroCom", "lifetime_du_emission", aer_plot, "AC emi", "#333333"))
    groups.append(("AeroCom", "lifetime_du", aer_plot, "AC dep", "#f4a261"))
    dep2024_path = TABLE_DIR / "pd2024_outflow_dep_lifetime.csv"
    dep2024 = pd.read_csv(dep2024_path) if dep2024_path.is_file() else pd.DataFrame()
    # Sahara, then the narrower outflow panel, then the other sources.
    panels = ["sahara", OUTFLOW_REGION] + [r for r in LIFETIME_REGIONS if r != "sahara"]
    # Width follows the number of boxes so the two-box outflow panel is not stretched.
    nboxes = []
    for region in panels:
        if region == OUTFLOW_REGION:
            nboxes.append(2 if not dep2024.empty else 1)
        else:
            nboxes.append(sum(1 for pid, *_ in groups if pid != "icon2025" or region != "atacama"))
    fig = plt.figure(figsize=(sum(nboxes) * 0.95 + 1.2, 4.8))
    gs = fig.add_gridspec(1, len(panels), width_ratios=nboxes, wspace=0.38)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(panels))]
    for ax, region in zip(axes, panels):
        data, colors, ticks = [], [], []
        if region == OUTFLOW_REGION:
            if not dep2024.empty and "lifetime_du_dep" in dep2024.columns:
                tau = dep2024["lifetime_du_dep"].astype(float)
                tau = tau[np.isfinite(tau) & (tau > 0)]
                if not tau.empty:
                    data.append((1.0 / tau).to_numpy())
                    colors.append(ppe_io.ppe_style("pd2024")["color"])
                    ticks.append(f"2024 dep\n(n={len(tau)})")
            if aer_plot is not None and not aer_plot.empty and "depdust" in aer_plot.columns:
                sub = aer_plot.loc[aer_plot["region"] == region]
                with np.errstate(divide="ignore", invalid="ignore"):
                    tau = sub["loaddust"].astype(float) / (sub["depdust"].astype(float) * 86400.0)
                tau = tau[np.isfinite(tau) & (tau > 0)]
                if not tau.empty:
                    data.append((1.0 / tau).to_numpy())
                    colors.append("#f4a261")
                    ticks.append(f"AC dep\n(n={len(tau)})")
        else:
            for pid, col, frame, lab, color in groups:
                if frame is None or frame.empty or col not in frame.columns:
                    continue
                sub = frame.loc[frame["ppe_id"] == pid] if "ppe_id" in frame.columns else frame
                tau = sub.loc[sub["region"] == region, col].astype(float)
                tau = tau[np.isfinite(tau) & (tau > 0)]
                if tau.empty:
                    continue
                data.append((1.0 / tau).to_numpy())
                colors.append(color)
                ticks.append(f"{lab}\n(n={len(tau)})")
        title = SEASON_TITLES.get(region, region)
        if region == OUTFLOW_REGION:
            title = title + "\nload / deposition"
        if not data:
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.3)
            continue
        bp = ax.boxplot(data, tick_labels=ticks, showfliers=True, patch_artist=True,
                        medianprops=dict(color="black", lw=1.4))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
        if region == OUTFLOW_REGION:
            ax.set_ylim(0, 0.16)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel(r"1/$\tau_{DU}$ (day$^{-1}$)")
    fig.suptitle(r"Dust 1/$\tau$: sources are load/emission; outflow is load/deposition", y=1.02)
    fig.subplots_adjust(top=0.82, wspace=0.45)
    save_fig(fig, "ppe_lifetime_du_boxplot.png")

_dust_lifetime_boxes()
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
                     title="PPE + AeroCom: dust AOD vs dust emission (no ICON-HAM on Atacama)",
                     aerocom_df=aer_plot, region_titles=SEASON_TITLES, **common)

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
    ax.set_xlabel(r"Model outflow dust AOD$_{550}$ [-]")
    ax.set_ylabel(r"Meta-model outflow dust AOD$_{550}$ [-]")
    ax.set_title("Saharan outflow dust AOD from E·τ·MEC")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "combined_outflow_meta_model.png")


def fit_outflow_aaod(df: pd.DataFrame, label: str, lt_col: str = "lifetime_du"
                     ) -> Tuple[Optional[dict], pd.DataFrame]:
    """Outflow column AAOD from Saharan E, τ and MAC inferred from column SSA.

    MAC = α(1 − SSA) is fit through the origin on the Sahara source. The same
    four-term transport model as the dust-AOD meta-model then predicts outflow
    column AAOD. Dust-only MAC is not required, so ICON and AeroCom are included.
    """
    need = {"emi_du", lt_col, "SSA", "MAC"}
    if df.empty or not need <= set(df.columns) or "abs550aer" not in df.columns:
        return None, pd.DataFrame()
    src = df.loc[df["region"] == OUTFLOW_SOURCE, ["model", "emi_du", lt_col, "SSA", "MAC"]]
    out = df.loc[df["region"] == OUTFLOW_REGION, ["model", "abs550aer"]]
    meta = src.merge(out.rename(columns={"abs550aer": "AAOD_out"}), on="model")
    meta = meta.replace([np.inf, -np.inf], np.nan).dropna()
    meta = meta.loc[meta[lt_col] > 0]
    one_minus = 1.0 - meta["SSA"].to_numpy(float)
    mac = meta["MAC"].to_numpy(float)
    if len(meta) < 5 or not np.any(one_minus):
        print(f"  {label}: insufficient members for AAOD meta-model (n={len(meta)})")
        return None, meta
    alpha = float(np.sum(one_minus * mac) / np.sum(one_minus ** 2))
    mac_hat = alpha * one_minus
    e_tau = meta["emi_du"].to_numpy(float) * 1000.0 * 86400.0 * meta[lt_col].to_numpy(float)
    X = np.column_stack([e_tau * mac_hat, e_tau, mac_hat, np.ones(len(meta))])
    y = meta["AAOD_out"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    meta = meta.copy()
    meta["MAC_from_SSA"] = mac_hat
    meta["AAOD_meta_fit"] = X @ coef
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - meta["AAOD_meta_fit"]) ** 2) / ss_tot if ss_tot > 0 else np.nan
    print(f"  {label}: AAOD R^2={r2:.3f} n={len(meta)}  MAC=α(1−SSA) α={alpha:.3g}")
    return {"dataset": label, "A": coef[0], "B": coef[1], "C": coef[2], "D": coef[3],
            "alpha_MAC_1SSA": alpha, "r2": r2, "n": len(meta)}, meta


print("\n--- Saharan outflow AAOD meta-model (MAC from column SSA) ---")
aaod_params, aaod_meta = [], {}
for pid in PPE_IDS:
    params, meta = fit_outflow_aaod(model_seasonal_df.loc[model_seasonal_df["ppe_id"] == pid], pid)
    aaod_meta[pid] = meta
    if params:
        aaod_params.append(params)
params, meta = fit_outflow_aaod(aerocom_df, "AeroCom", lt_col="lifetime_du_emission")
aaod_meta["AeroCom"] = meta
if params:
    aaod_params.append(params)
if aaod_params:
    pd.DataFrame(aaod_params).to_csv(TABLE_DIR / "outflow_aaod_meta_model_params.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    vals = []
    for p in aaod_params:
        m = aaod_meta[p["dataset"]]
        if p["dataset"] == "AeroCom":
            style, lab = pp.AEROCOM_STYLE, "AeroCom"
        else:
            st = ppe_io.ppe_style(p["dataset"])
            style, lab = dict(s=16, alpha=0.5, color=st["color"], marker=st["marker"]), st["label"]
        ax.scatter(m["AAOD_out"], m["AAOD_meta_fit"], **style,
                   label=f"{lab} (R²={p['r2']:.2f}, n={p['n']})")
        vals += list(m["AAOD_out"]) + list(m["AAOD_meta_fit"])
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    ax.plot([lo, hi], [lo, hi], "k:", lw=1)
    ax.set_xlabel(r"Model outflow column AAOD$_{550}$ [-]")
    ax.set_ylabel(r"Meta-model outflow AAOD$_{550}$ [-]")
    ax.set_title(r"Saharan outflow AAOD from E·τ·MAC, MAC = α(1−SSA)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "combined_outflow_aaod_meta_model.png")

# %%
# -----------------------------------------------------------------------------
# What tracks emission-based dust lifetime?
# τ = load / emission, so correlation with emission or load is partly algebraic.
# The table below is the screen; the figure keeps predictors that are not E or load.
# -----------------------------------------------------------------------------
_TAU_PREDS = ["precip", "AE", "SSA", "SSA_du", "MEC_du", "MEC", "MAC", "MAC_du",
              "dust_frac", "ustar_acrit", "emi_du", "od550dust", "loaddust"]
_tau_rows = []
for (pid, region), g in df_plot.groupby(["ppe_id", "region"]):
    if region not in LIFETIME_REGIONS:
        continue
    y = g["lifetime_du"].astype(float)
    for pred in _TAU_PREDS:
        if pred not in g.columns:
            continue
        x = g[pred].astype(float)
        ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
        if ok.sum() < 20 or float(np.nanstd(x[ok])) == 0:
            continue
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        _tau_rows.append({"ppe_id": pid, "region": region, "predictor": pred,
                          "n": int(ok.sum()), "r_lifetime": r})
tau_corr = pd.DataFrame(_tau_rows)
if not tau_corr.empty:
    tau_corr.to_csv(TABLE_DIR / "ppe_lifetime_du_correlations.csv", index=False)
    show = tau_corr.loc[~tau_corr["predictor"].isin(["emi_du", "loaddust", "od550dust"])]
    preds = ["precip", "AE", "SSA", "MEC_du", "MAC", "dust_frac", "ustar_acrit"]
    regions_present = [r for r in LIFETIME_REGIONS if r in set(show["region"])]
    ppe_present = [p for p in PPE_IDS if p in set(show["ppe_id"])]
    fig, axes = plt.subplots(len(ppe_present), len(regions_present),
                             figsize=(4.2 * len(regions_present), 3.2 * len(ppe_present)),
                             squeeze=False, sharey=True)
    for i, pid in enumerate(ppe_present):
        for j, region in enumerate(regions_present):
            ax = axes[i, j]
            sub = show.loc[(show.ppe_id == pid) & (show.region == region)]
            if sub.empty:
                ax.axis("off")
                continue
            vals = [float(sub.loc[sub.predictor == p, "r_lifetime"].iloc[0])
                    if p in set(sub.predictor) else np.nan for p in preds]
            colors = ["#d62728" if (np.isfinite(v) and v < 0) else "#1f77b4" for v in vals]
            ax.barh(preds, [0 if not np.isfinite(v) else v for v in vals], color=colors)
            ax.axvline(0, color="k", lw=0.6)
            ax.set_xlim(-1, 1)
            ax.set_title(f"{pid}  {SEASON_TITLES.get(region, region)}", fontsize=9)
            if j == 0:
                ax.set_ylabel("predictor")
            if i == len(ppe_present) - 1:
                ax.set_xlabel(r"correlation with $\tau_{DU}$")
    fig.suptitle(r"Emission-based dust lifetime: correlation with other fields", y=1.01)
    fig.tight_layout()
    save_fig(fig, "ppe_lifetime_du_correlations.png")
    top = (tau_corr.assign(abs_r=lambda d: d["r_lifetime"].abs())
           .sort_values("abs_r", ascending=False)
           .groupby(["ppe_id", "region"], as_index=False).head(3))
    print("\n--- Strongest |r| with emission-based dust lifetime ---")
    print(top.sort_values(["ppe_id", "region", "abs_r"], ascending=[True, True, False])
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# %%
# -----------------------------------------------------------------------------
# Region maps (PPE 2024 ensemble-mean DAOD) and dust share of column AOD / AAOD
# Seasons are the months in regions.REGIONS, taken from the Aug 2024–Jul 2025 year.
# -----------------------------------------------------------------------------
import box_sensitivity as boxes

print("\n--- Dust AOD maps and dust share of column optics (PPE 2024) ---")
for region in ANALYSIS_REGIONS:
    months = tuple(reg.REGIONS[region]["season_months"])
    when = f"{_season_label(months)} (within Aug 2024–Jul 2025)"
    clon, ext = boxes.extent_for_regions([region], pad=12.0)
    daod = boxes.mask_to_surface(
        boxes.ensemble_mean_aod("pd2024", ["od550dust"], months=months),
        [region], blank_outside=True)
    aod = boxes.ensemble_mean_aod("pd2024", ["od550aer"], months=months)
    daaod = boxes.ensemble_mean_aod("pd2024", ["abs550dust"], months=months)
    aaod = boxes.ensemble_mean_aod("pd2024", ["abs550aer"], months=months)
    boxes.plot_study_map(
        daod, boxes.region_boxes([region]),
        title=f"PPE 2024 ensemble-mean DAOD$_{{550}}$, {region}, {when}",
        cbar_label=r"DAOD$_{550}$ [-]",
        filename=FIGURE_DIR / f"ppe2024_DAOD550_boxes_{region}.png",
        extent=ext, central_longitude=clon, land_facecolor=None,
    )
    share = (daod / aod).where(aod > 1e-4)
    boxes.plot_study_map(
        share, boxes.region_boxes([region]),
        title=f"PPE 2024 DAOD$_{{550}}$ / AOD$_{{550}}$, {region}, {when}",
        cbar_label=r"DAOD$_{550}$ / AOD$_{550}$ [-]",
        filename=FIGURE_DIR / f"ppe2024_dust_frac_AOD_{region}.png",
        extent=ext, central_longitude=clon, land_facecolor=None,
    )
    ashare = (daaod / aaod).where(aaod > 1e-6)
    boxes.plot_study_map(
        ashare, boxes.region_boxes([region]),
        title=f"PPE 2024 dust AAOD$_{{550}}$ / AAOD$_{{550}}$, {region}, {when}",
        cbar_label=r"dust AAOD$_{550}$ / AAOD$_{550}$ [-]",
        filename=FIGURE_DIR / f"ppe2024_dust_frac_AAOD_{region}.png",
        extent=ext, central_longitude=clon, land_facecolor=None,
    )

# One colour scale for the whole-Earth maps. Mercator is the flat global view;
# Robinson is the curved whole-Earth view so every region stays on the page.
_world = boxes.ensemble_mean_aod("pd2024", ["od550dust"])
_world_boxes = boxes.region_boxes(ANALYSIS_REGIONS)
_finite = np.asarray(_world.values, float)
_finite = _finite[np.isfinite(_finite)]
_z0, _z1 = (float(np.nanpercentile(_finite, 2)), float(np.nanpercentile(_finite, 98)))
import cartopy.crs as ccrs
boxes.plot_world_field(
    _world, _world_boxes, ccrs.Mercator(),
    "PPE 2024 ensemble-mean DAOD$_{550}$, all dust regions",
    r"DAOD$_{550}$ [-]",
    FIGURE_DIR / "ppe2024_DAOD550_global_mercator.png",
    extent=(-180, 180, -70, 70), zmin=_z0, zmax=_z1,
)
boxes.plot_world_field(
    _world, _world_boxes, ccrs.Robinson(),
    "PPE 2024 ensemble-mean DAOD$_{550}$, all dust regions",
    r"DAOD$_{550}$ [-]",
    FIGURE_DIR / "ppe2024_DAOD550_global_sphere.png",
    zmin=_z0, zmax=_z1,
)

# %%
# -----------------------------------------------------------------------------
# Source-region attribution, no fitted constraint.
# DAOD is compared with observed total AOD (dust stands in for the column).
# The absorption track uses column AAOD and dust load for every PPE and AeroCom.
# MAC_du is drawn only where dust absorption is archived (the two ECHAM PPEs).
# SPEXone is the 2024–25 reference; POLDER is the 2010 reference.
# -----------------------------------------------------------------------------
SOURCE_TITLES = {r: SEASON_TITLES[r] for r in SOURCE_REGIONS}
print("\n--- SPEXone for ICON and column-AAOD references ---")
try:
    icon_obs = ppe_io.load_obs_for_ppe("icon2025", ae_pair="440_550")
    obs_means["icon2025"] = ppe_io.obs_region_means(
        icon_obs, SOURCE_REGIONS, ppe_cfg.get_run("icon2025")["period"])
    del icon_obs
except Exception as exc:
    print(f"  ICON SPEXone means failed: {exc}")
    ISSUES.append(f"icon2025: SPEXone regional means failed ({exc}).")

ATTR_LABELS = {
    **AXIS_LABELS,
    "abs550aer": r"Column AAOD$_{550}$ [-]",
    "abs550dust": r"Dust AAOD$_{550}$ [-]",
    "MAC_load": r"Column AAOD / dust load (m$^{2}$ g$^{-1}$)",
    "E_tau_MEC": r"E$\times\tau\times$MEC$_{DU}$ [-]",
    "E_tau_MAC": r"E$\times\tau\times$(column AAOD / dust load) [-]",
}


def _with_products(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    life = out["lifetime_du"] if "lifetime_du" in out.columns else np.nan
    if "lifetime_du_emission" in out.columns:
        life = out["lifetime_du_emission"].where(out["lifetime_du_emission"].notna(), life)
        out["lifetime_du"] = life
    with np.errstate(divide="ignore", invalid="ignore"):
        out["MAC_load"] = out["abs550aer"] / (out["loaddust"] * 1e3)
        out.loc[~(out["loaddust"] > 0), "MAC_load"] = np.nan
        tau_s = out["lifetime_du"] * 86400.0
        out["E_tau_MEC"] = out["emi_du"] * tau_s * out["MEC_du"] * 1000.0
        out["E_tau_MAC"] = out["emi_du"] * tau_s * out["MAC_load"] * 1000.0
    return out


attr = _with_products(df_plot)
attr_emi = attr.loc[~((attr["ppe_id"] == "icon2025") & (attr["region"] == "atacama"))]
attr_aero = _with_products(aer_plot) if not aer_plot.empty else aer_plot
attr_common = dict(saver=save_fig, ppe_ids=PPE_IDS, labels=ATTR_LABELS,
                   region_titles=SOURCE_TITLES, aerocom_df=attr_aero)


def _obs_y(column: str, legend: str):
    lines = {}
    for pid, label in (
        ("pd2024", "SPEXone 2024–25"),
        ("icon2025", "SPEXone, ICON period"),
        ("pd2010", "POLDER 2010"),
    ):
        if pid not in obs_means:
            continue
        for region in SOURCE_REGIONS:
            val = obs_means[pid].get(region, {}).get(column, np.nan)
            lines[(pid, region)] = (val, f"{label} {legend}")
    return lines


def _dust_equals_column(ax, region):
    pts = [c.get_offsets() for c in ax.collections if len(c.get_offsets())]
    if not pts:
        return
    hi = float(np.nanmax(np.vstack(pts)))
    ax.plot([0, hi], [0, hi], color="k", ls=":", lw=1, label="dust = all AOD")
    ax.set_xlim(0, hi * 1.05)
    ax.set_ylim(0, hi * 1.05)


pp.scatter_by_region(
    attr, "od550aer", "od550dust", SOURCE_REGIONS,
    filename="ppe_DAOD_vs_total_AOD.png",
    title="Supplementary: model DAOD$_{550}$ vs total AOD$_{550}$ (sources)",
    extra=_dust_equals_column, **attr_common,
)
pp.scatter_by_region(
    attr_emi, "emi_du", "od550dust", SOURCE_REGIONS,
    filename="ppe_attr_DAOD_vs_emission.png",
    title=r"Dust AOD$_{550}$ vs emission (ICON omitted on Atacama; obs = total AOD)",
    obs_y=_obs_y("AOD_550", "total AOD"), **attr_common,
)
pp.scatter_by_region(
    attr, "lifetime_du", "od550dust", SOURCE_REGIONS,
    filename="ppe_attr_DAOD_vs_lifetime.png",
    title=r"Dust AOD$_{550}$ vs emission lifetime (obs = total AOD)",
    obs_y=_obs_y("AOD_550", "total AOD"), **attr_common,
)
pp.scatter_by_region(
    attr, "MEC_du", "od550dust", SOURCE_REGIONS,
    filename="ppe_attr_DAOD_vs_MEC.png",
    title=r"Dust AOD$_{550}$ vs MEC$_{DU}$ (obs = total AOD)",
    obs_y=_obs_y("AOD_550", "total AOD"), **attr_common,
)
pp.scatter_by_region(
    attr_emi, "E_tau_MEC", "od550dust", SOURCE_REGIONS,
    filename="ppe_attr_DAOD_vs_EtauMEC.png",
    title=r"Dust AOD$_{550}$ vs E$\times\tau\times$MEC$_{DU}$ (obs = total AOD)",
    obs_y=_obs_y("AOD_550", "total AOD"), **attr_common,
)
pp.scatter_by_region(
    attr_emi, "emi_du", "abs550aer", SOURCE_REGIONS,
    filename="ppe_attr_AAOD_vs_emission.png",
    title="Column AAOD$_{550}$ vs dust emission (obs = column AAOD)",
    obs_y=_obs_y("AAOD_550", "column AAOD"), **attr_common,
)
pp.scatter_by_region(
    attr, "lifetime_du", "abs550aer", SOURCE_REGIONS,
    filename="ppe_attr_AAOD_vs_lifetime.png",
    title="Column AAOD$_{550}$ vs dust emission lifetime (obs = column AAOD)",
    obs_y=_obs_y("AAOD_550", "column AAOD"), **attr_common,
)
pp.scatter_by_region(
    attr, "MAC_load", "abs550aer", SOURCE_REGIONS,
    filename="ppe_attr_AAOD_vs_MAC_load.png",
    title="Column AAOD$_{550}$ vs column AAOD / dust load (obs = column AAOD)",
    obs_y=_obs_y("AAOD_550", "column AAOD"), **attr_common,
)
pp.scatter_by_region(
    attr_emi, "E_tau_MAC", "abs550aer", SOURCE_REGIONS,
    filename="ppe_attr_AAOD_vs_EtauMAC.png",
    title=r"Column AAOD$_{550}$ vs E$\times\tau\times$(column AAOD / dust load)",
    obs_y=_obs_y("AAOD_550", "column AAOD"), **attr_common,
)
echam = attr.loc[attr["ppe_id"].isin(["pd2024", "pd2010"])]
pp.scatter_by_region(
    echam, "od550dust", "MAC_du", SOURCE_REGIONS,
    filename="ppe_DAOD_vs_MAC_du.png",
    title="ECHAM only: MAC$_{DU}$ vs DAOD$_{550}$ (no dust absorption in ICON or AeroCom AP3)",
    ppe_ids=["pd2024", "pd2010"], labels=ATTR_LABELS, region_titles=SOURCE_TITLES,
    saver=save_fig,
)
pp.scatter_by_region(
    echam, "emi_du", "abs550dust", SOURCE_REGIONS,
    filename="ppe_dustAAOD_vs_emission.png",
    title="ECHAM only: dust AAOD$_{550}$ vs dust emission",
    ppe_ids=["pd2024", "pd2010"], labels=ATTR_LABELS, region_titles=SOURCE_TITLES,
    saver=save_fig,
)

# %%
# -----------------------------------------------------------------------------
# ICON ensemble median over South America, Aug–Nov 2024.
# Shows the author that dust emission inside the Atacama box is zero.
# -----------------------------------------------------------------------------
import xarray as xr


def _icon_member_mean(path: Path, varname: str, months, year: int) -> np.ndarray:
    ds = xr.open_dataset(path)
    try:
        name = varname if varname in ds.data_vars else next(
            n for n in ds.data_vars if "bnds" not in n and "time" in ds[n].dims)
        stamp = np.asarray(ds["time"].values).astype(np.int64)
        keep = ((stamp // 10000) == year) & np.isin((stamp // 100) % 100, list(months))
        if not np.any(keep):
            return np.full(ds.sizes["ncells"], np.nan)
        return np.asarray(ds[name].isel(time=np.where(keep)[0]).mean("time").values, float)
    finally:
        ds.close()


def icon_median_field(folder: str, varname: str, months, year: int,
                      lon_range, lat_range, dlon: float = 0.5) -> xr.DataArray:
    root = Path(ppe_cfg.get_run("icon2025")["root"]) / folder
    files = sorted(root.glob("ens_*_PD_*.nc"))
    if not files:
        raise FileNotFoundError(root)
    with xr.open_dataset(files[0]) as sample:
        lon = np.degrees(np.asarray(sample["clon"].values, float)) % 360.0
        lat = np.degrees(np.asarray(sample["clat"].values, float))
    window = ((lon >= lon_range[0]) & (lon <= lon_range[1])
              & (lat >= lat_range[0]) & (lat <= lat_range[1]))
    lon_w, lat_w = lon[window], lat[window]

    rows = []
    for i, path in enumerate(files, 1):
        rows.append(_icon_member_mean(path, varname, months, year)[window])
        if i % 80 == 0 or i == len(files):
            print(f"    {folder}: {i}/{len(files)} members", flush=True)
    stacked = np.vstack(rows)
    med = np.nanmedian(stacked, axis=0)
    positive = np.nanmean(stacked > 0, axis=0)
    lon_edges = np.arange(lon_range[0], lon_range[1] + dlon, dlon)
    lat_edges = np.arange(lat_range[0], lat_range[1] + dlon, dlon)
    ix = np.clip(np.digitize(lon_w, lon_edges) - 1, 0, len(lon_edges) - 2)
    iy = np.clip(np.digitize(lat_w, lat_edges) - 1, 0, len(lat_edges) - 2)
    grid = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan)
    pos = np.full_like(grid, np.nan)
    order = iy * grid.shape[1] + ix
    for key in np.unique(order):
        j, i = divmod(int(key), grid.shape[1])
        sel = order == key
        grid[j, i] = np.nanmedian(med[sel])
        pos[j, i] = np.nanmedian(positive[sel])
    field = xr.DataArray(
        grid,
        coords={"lat": 0.5 * (lat_edges[:-1] + lat_edges[1:]),
                "lon": 0.5 * (lon_edges[:-1] + lon_edges[1:])},
        dims=("lat", "lon"),
    )
    field.attrs["n_members"] = int(stacked.shape[0])
    field.attrs["native_positive_fraction"] = float(np.nanmean(positive))
    return field, lon_w, lat_w, med


# ICON-HAM has no Atacama dust source, so it is not drawn on Atacama panels.
print("\n--- ICON Atacama maps skipped (no dust source) ---")
ISSUES.append("ICON-HAM is omitted from every Atacama panel: the member has no dust source there.")

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
