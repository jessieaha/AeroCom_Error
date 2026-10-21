# %% [markdown]
# # PPE precipitation vs GPCP (AAOD fire-season regions)
#
# Precipitation-only companion to ``AAOD_error_attribution_PPE.py``. Uses the
# regional monthly PPE store (``Data/PPE_regional``) and the gridded PPE
# precipitation on scratch for every registered PPE with a precip stream;
# AeroCom (2010) from ``tables/regional_seasonal_aggregates.parquet``.
# GPCP months missing from ``Data/Prec`` are excluded, never filled.
#
# Run: `sbatch run_notebook_or_py.sbatch notebooks/PPE_precip_vs_GPCP.py`

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
for p in (project_root / "py", project_root / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import PPE_set as ppe_cfg
import ppe_io
import ppe_plots as pp
import ppe_precip_gpcp as pg
import regions as reg

PPE_IDS = [p for p, run in ppe_cfg.PPE_RUNS.items() if run["has_precip"]]
REGION_NAMES = ["africa", "amazon", "outflow_af"]
REGIONS = {r: reg.REGIONS[r] for r in REGION_NAMES}
COLORS = reg.region_colors(REGION_NAMES)
MAP_MONTHS = sorted({m for cfg in REGIONS.values() for m in cfg["season_months"]})

GPCP_DIR = project_root / "Data" / "Prec"
AEROCOM_PARQUET = project_root / "tables" / "regional_seasonal_aggregates.parquet"
FIGURE_DIR = project_root / "figure" / "PPE_bb"
TABLE_DIR = ppe_cfg.ANALYSIS_DIR / "bb"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
save_fig = pp.FigureSaver(FIGURE_DIR)

aerocom_precip, gpcp_2010 = pg.load_aerocom_precip(AEROCOM_PARQUET, REGIONS)
print(f"AeroCom models with precip: {aerocom_precip['model'].nunique()}; GPCP 2010: {gpcp_2010}")
monthly_all = ppe_io.load_regional(PPE_IDS, species="bb", regions=REGION_NAMES, seasonal=False)

# %%
for pid in PPE_IDS:
    print(f"\n=== {pid} ===")
    ym = ppe_io.ppe_year_months(pid)
    pid_save = lambda fig, name, _p=pid: save_fig(fig, f"{_p}_{name}")  # noqa: E731
    gpcp_da, gpcp_missing = pg.load_gpcp_months(GPCP_DIR, ym)
    print(f"GPCP months loaded: {gpcp_da.sizes['time'] if gpcp_da is not None else 0}; "
          f"missing (excluded, not filled): {gpcp_missing}")
    gpcp_obs, gpcp_rows, matched = pg.gpcp_region_tables(gpcp_da, REGIONS, ym)
    gpcp_rows.to_csv(TABLE_DIR / f"{pid}_gpcp_region_monthly.csv", index=False)

    monthly_df = monthly_all.loc[monthly_all["ppe_id"] == pid]
    matched_df = pg.matched_model_precip(monthly_df, matched)
    matched_df.to_csv(TABLE_DIR / f"{pid}_precip_gpcp_matched.csv", index=False)
    check = pg.precip_check_table(matched_df, gpcp_obs, matched)
    check.to_csv(TABLE_DIR / f"{pid}_precip_vs_gpcp.csv", index=False)
    print("PPE (GPCP-matched months) vs GPCP:")
    print(check.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    with ppe_io.open_gridded(pid, ["precip"]) as ds:
        ppe_precip = ds["precip"].drop_vars("experiment", errors="ignore").load()
    print(f"PPE gridded precip: {dict(ppe_precip.sizes)}")

    pg.plot_precip_seasonal_cycle(monthly_df, gpcp_rows, REGIONS, COLORS, ym, pid_save)
    pg.plot_precip_season_box(matched_df, aerocom_precip, gpcp_obs, gpcp_2010, matched,
                              REGIONS, COLORS, pid_save)
    pg.plot_precip_maps(ppe_precip, gpcp_da, REGIONS, pid_save, MAP_MONTHS)
    stats = pg.plot_precip_scatter(ppe_precip, gpcp_da, REGIONS, pid_save)
    stats.to_csv(TABLE_DIR / f"{pid}_precip_ensmean_vs_gpcp_scatter_stats.csv", index=False)
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\nDone.")
