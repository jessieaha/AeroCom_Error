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
# remote_SO_DJF, remote_SPO_box_DJF, SAO_2_DJF, then the other sea-salt basins.
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
PPE_IDS: List[str] = ["pd2024", "pd2010"]
EXCLUDE_MEMBERS: List[str] = []  # e.g. ["Control"] or ["pd2010:17"]

# Pristine DJF boxes first, then the original basins (SAO is the less remote ocean).
_REGION_ORDER = ["remote_SO_DJF", "remote_SPO_box_DJF", "SAO_2_DJF", "SAO_SOND", "ANTAO_JFM", "NPO_JF"]
SOURCE_REGIONS = reg.source_regions(SPECIES)
ANALYSIS_REGIONS = [r for r in _REGION_ORDER if r in SOURCE_REGIONS]

SAVE_FIGURE = True
FIGURE_DIR = project_root / "figure" / "PPE_ss"
TABLE_DIR = ppe_cfg.ANALYSIS_DIR / SPECIES
TABLE_DIR.mkdir(parents=True, exist_ok=True)
save_fig = pp.FigureSaver(FIGURE_DIR, save=SAVE_FIGURE)

HOMOGENIZE_VARS = {"AOD": "AOD_550", "AAOD": "AAOD_550", "SSA": "SSA"}

AXIS_LABELS = {
    "SSA": "SSA (1)",
    "1-SSA": "1 − SSA (1)",
    "MEC": r"MEC (m$^{2}$ g$^{-1}$)",
    "MEC_ss": r"MEC$_{SS}$ (m$^{2}$ g$^{-1}$)",
    "MEC_so4": r"MEC$_{SO4}$ (m$^{2}$ g$^{-1}$)",
    "MAC": r"MAC (m$^{2}$ g$^{-1}$)",
    "AE": "AE 550-865 (1)",
    "precip": r"Precipitation (mm day$^{-1}$)",
    "lifetime_ss": r"Sea-salt lifetime $\tau_{SS}$ (day)",
    "lifetime_mass": r"Lifetime $\tau$ (day)",
    "inv_lifetime_ss": r"1/$\tau_{SS}$ (day$^{-1}$)",
    "inv_lifetime_mass": r"1/$\tau$ (day$^{-1}$)",
    "od550aer": "AOD$_{550}$ (1)",
    "od550ss": "AOD$_{SS,550}$ (1)",
    "ss_burden_frac": "Sea-salt share of SS+SO4 burden (1)",
    "dms_source_frac": "DMS share of SO4-equivalent source (1)",
    "so2_source_frac": "SO2 share of SO4-equivalent source (1)",
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
# What changes sea-salt and sea-salt+sulfate lifetime
# -----------------------------------------------------------------------------
# Emission lifetime only (PPE has no depss/wetss/dryss). Sulfate source is
# primary SO4 plus half of SO2 and DMS, converted to SO4 mass. DMS is the
# remote source; SO2 is the continental-pollution source.
model_seasonal_df = ppe_io.add_lifetime_drivers(model_seasonal_df)
aerocom_df = ppe_io.add_lifetime_drivers(aerocom_df)
df_plot = pp.with_one_minus_ssa(model_seasonal_df)
aer_plot = pp.with_one_minus_ssa(aerocom_df)
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
# Observation maps (raw obs per PPE period)
# -----------------------------------------------------------------------------
for pid, obs in obs_tables.items():
    if obs.empty or not SAVE_FIGURE:
        continue
    period = ppe_cfg.get_run(pid)["period"]
    obs_name = ppe_io.ppe_obs_name(pid)
    fig, axes = plt.subplots(len(ANALYSIS_REGIONS), 2, figsize=(12, 3.5 * len(ANALYSIS_REGIONS)),
                             squeeze=False)
    for i, region in enumerate(ANALYSIS_REGIONS):
        sub = ppe_io.obs_in_region(obs, region, period)
        if sub.empty:
            continue
        grid = sub.assign(lon=sub["lon360"].round(0), lat=sub["latitude"].round(0)) \
            .groupby(["lon", "lat"], as_index=False)[["AOD_550", "SSA"]].mean()
        for j, col in enumerate(["AOD_550", "SSA"]):
            ax = axes[i, j]
            sc = ax.scatter(grid["lon"], grid["lat"], c=grid[col], s=8, cmap="viridis")
            fig.colorbar(sc, ax=ax, shrink=0.8).set_label(f"{col} (1)")
            ax.set_title(f"{region} {obs_name} {col}")
            ax.set_xlabel("Longitude (°E, 0-360)")
            ax.set_ylabel("Latitude (°)")
    fig.tight_layout()
    save_fig(fig, f"{pid}_{obs_name}_regional_maps.png")

# %%
# -----------------------------------------------------------------------------
# Regional map: PPE 2024 ensemble-mean sea-salt AOD, DJF, with the boxes
# -----------------------------------------------------------------------------
print("\n--- Sea-salt regional map (PPE 2024, DJF od550ss) ---")
ss_field = boxes.ensemble_mean_aod("pd2024", ["od550ss"], months=(12, 1, 2))
ss_defs = boxes.load_defs()
boxes.plot_study_map(
    ss_field, boxes.map_boxes(ss_defs, ["remote_SO_DJF", "remote_SPO_box_DJF"]),
    title="PPE 2024 ensemble-mean sea-salt AOD$_{550}$ (DJF), remote Pacific",
    cbar_label="AOD$_{SS,550}$ (1)",
    filename=FIGURE_DIR / "ppe2024_seasalt_AOD_boxes_DJF.png",
    extent=(170, 290, -75, -20),
)
boxes.plot_study_map(
    ss_field, boxes.map_boxes(ss_defs, ["SAO_2_DJF"]),
    title="PPE 2024 ensemble-mean sea-salt AOD$_{550}$ (DJF), SAO_2",
    cbar_label="AOD$_{SS,550}$ (1)",
    filename=FIGURE_DIR / "ppe2024_seasalt_AOD_boxes_SAO_2_DJF.png",
    extent=(-50, 20, -70, -20),
    central_longitude=0.0,
)
boxes.plot_study_map(
    ss_field, boxes.map_boxes(ss_defs, boxes.BASELINES["ss"]),
    title="PPE 2024 ensemble-mean sea-salt AOD$_{550}$ (DJF), global",
    cbar_label="AOD$_{SS,550}$ (1)",
    filename=FIGURE_DIR / "ppe2024_seasalt_AOD_boxes_DJF_global.png",
)
boxes.plot_study_map(
    ss_field, [],
    title="PPE 2024 ensemble-mean sea-salt AOD$_{550}$ (DJF)",
    cbar_label="AOD$_{SS,550}$ (1)",
    filename=FIGURE_DIR / "ppe2024_seasalt_AOD_DJF_curved.png",
)

print("\nDone.")
print(f"Figures in: {FIGURE_DIR}")
print(f"Tables in:  {TABLE_DIR}")
