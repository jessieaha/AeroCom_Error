# %% [markdown]
# # PPE precipitation vs GPCP (AAOD fire-season regions)
#
# Precipitation-only companion to ``AAOD_error_attribution_PPE.py``. Reuses the
# regional monthly PPE table written by that script and the gridded PPE
# precipitation; AeroCom (2010) from ``tables/regional_seasonal_aggregates.parquet``.
# GPCP months missing from ``Data/Prec`` are excluded, never filled.
#
# Run: `sbatch run_notebook_or_py.sbatch notebooks/PPE_precip_vs_GPCP.py`

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
for p in (project_root / "py", project_root / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import PPE_set as ppe_cfg
import ppe_precip_gpcp as pg

REGIONS = ppe_cfg.AAOD_REGIONS
COLORS = ppe_cfg.AAOD_REGION_COLORS
YEAR_MONTHS = ppe_cfg.PPE_YEAR_MONTHS
MAP_MONTHS = sorted({m for cfg in REGIONS.values() for m in cfg["season_months"]})

GPCP_DIR = project_root / "Data" / "Prec"
PPE_PRECIP_NC = project_root / "Data" / "PPE_processed_monthly" / "precip_monthly.nc"
MONTHLY_PARQUET = project_root / "Data" / "PPE_AAOD" / "aaod_ppe_monthly_africa_amazon_outflow.parquet"
AEROCOM_PARQUET = project_root / "tables" / "regional_seasonal_aggregates.parquet"
FIGURE_DIR = project_root / "figure" / "PPE_AAOD"
TABLE_DIR = project_root / "Data" / "PPE_AAOD"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# %%
gpcp_da, gpcp_missing = pg.load_gpcp_months(GPCP_DIR, YEAR_MONTHS)
print(f"GPCP months loaded: {gpcp_da.sizes['time'] if gpcp_da is not None else 0}; "
      f"missing (excluded, not filled): {gpcp_missing}")
gpcp_obs, gpcp_rows, matched = pg.gpcp_region_tables(gpcp_da, REGIONS, YEAR_MONTHS)
gpcp_rows.to_csv(TABLE_DIR / "gpcp_region_monthly.csv", index=False)
for region in REGIONS:
    print(f"  {region}: GPCP fire-season mean {gpcp_obs[region]:.3f} mm/day "
          f"over {sorted(matched[region])}")

monthly_df = pd.read_parquet(MONTHLY_PARQUET)
matched_df = pg.matched_model_precip(monthly_df, matched)
matched_df.to_csv(TABLE_DIR / "ppe_precip_gpcp_matched.csv", index=False)
check = pg.precip_check_table(matched_df, gpcp_obs, matched)
check.to_csv(TABLE_DIR / "ppe_precip_vs_gpcp.csv", index=False)
print("\nPPE (GPCP-matched months) vs GPCP:")
print(check.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

aerocom_precip, gpcp_2010 = pg.load_aerocom_precip(AEROCOM_PARQUET, REGIONS)
print(f"\nAeroCom models with precip: {aerocom_precip['model'].nunique()}; GPCP 2010: {gpcp_2010}")

with xr.open_dataset(PPE_PRECIP_NC) as ds:
    ppe_precip = ds["precip"].drop_vars("experiment", errors="ignore").load()
print(f"PPE gridded precip: {dict(ppe_precip.sizes)} ({ppe_precip.attrs.get('units')})")

# %%
pg.plot_precip_seasonal_cycle(monthly_df, gpcp_rows, REGIONS, COLORS, YEAR_MONTHS, save_fig)
pg.plot_precip_season_box(matched_df, aerocom_precip, gpcp_obs, gpcp_2010, matched,
                          REGIONS, COLORS, save_fig)
pg.plot_precip_maps(ppe_precip, gpcp_da, REGIONS, save_fig, MAP_MONTHS)
scatter_stats = pg.plot_precip_scatter(ppe_precip, gpcp_da, REGIONS, save_fig)
scatter_stats.to_csv(TABLE_DIR / "ppe_precip_ensmean_vs_gpcp_scatter_stats.csv", index=False)
print("\nScatter stats (grid-box-months):")
print(scatter_stats.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\nDone.")
