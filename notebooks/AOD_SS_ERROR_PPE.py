# %% [markdown]
# # AOD sea-salt error analysis on Pace PPE
#
# Port of ``AOD_SS_ERROR.ipynb`` onto ``Data/PPE_processed_monthly/``.
# Observations use SPEXone (2024-08 … 2025-07), not POLDER.
# Emission-based lifetimes only (wet/dry/dep streams are not in the PPE extract).
#
# Run: `.venv/bin/python notebooks/AOD_SS_ERROR_PPE.py`
# Or:  `sbatch jobs/run_notebook_or_py.sbatch notebooks/AOD_SS_ERROR_PPE.py`

# %%
import sys
import os
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

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
from scipy.stats import linregress

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Project paths
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
import aerocom_data
import notebook_setup as setup
from aerocom_data import _get_dataarray, _align_da_to_ref

print(f"Project root: {project_root}")

# %%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PPE_DIR = project_root / "Data" / "PPE_processed_monthly"
SPEXONE_GLOB = str(project_root / "Data" / "SPEXone_L2_regridded_{year}_{month:02d}_v4_0.parquet")
SPEXONE_YEARS_MONTHS = [
    (2024, m) for m in range(8, 13)
] + [
    (2025, m) for m in range(1, 8)
]
WVL = "550nm"

EXCLUDE_MEMBERS: List[str] = []  # e.g. ["Control"] or ["250"]
# Always use the full PPE ensemble (no member-count cap).

DERIVED_VAR_AFTER_AGG = {"lifetime_ss", "lifetime"}
USE_DEPOSITION_FOR_LIFETIME = False

# PPE burden_* streams are labeled kg m-2 but are ~1000× below AeroCom column
# burdens (and give MEC_ss ~700 m2 g-1, 1/τ_ss ~300 d-1). Scaling recovers
# MEC_ss ~ O(1) m2 g-1 (axis 0–15) and 1/τ ~ 0.1–1 d-1. Emissions look OK.
BURDEN_MASS_SCALE = 1e3

SAVE_FIGURE = True
FIGURE_DIR = project_root / "figure" / "PPE_SS"
TABLE_DIR = project_root / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Axis labels with units (used by all scatter / lifetime plots)
AXIS_LABELS = {
    "SSA": "SSA (1)",
    "1-SSA": "1 − SSA (1)",
    "MEC": r"MEC (m$^{2}$ g$^{-1}$)",
    "MEC_ss": r"MEC$_{SS}$ (m$^{2}$ g$^{-1}$)",
    "MEC_so4": r"MEC$_{SO4}$ (m$^{2}$ g$^{-1}$)",
    "MAC": r"MAC (m$^{2}$ g$^{-1}$)",
    "AE": "AE (1)",
    "precip": r"Precipitation (mm day$^{-1}$)",
    "lifetime": r"Lifetime $\tau$ (day)",
    "lifetime_ss": r"Sea-salt lifetime $\tau_{SS}$ (day)",
    "inv_lifetime": r"1/$\tau$ (day$^{-1}$)",
    "inv_lifetime_ss": r"1/$\tau_{SS}$ (day$^{-1}$)",
    "od550aer": "AOD$_{550}$ (1)",
    "od550ss": "AOD$_{SS,550}$ (1)",
    "loadss": r"SS burden (kg m$^{-2}$)",
    "load_total": r"Total burden (kg m$^{-2}$)",
    "emiss": r"SS emission (kg m$^{-2}$ s$^{-1}$)",
    "emi_total": r"Total emission (kg m$^{-2}$ s$^{-1}$)",
}

SAVE_AGG_CACHE = True
AGG_PARQUET = TABLE_DIR / "seasalt_ppe_seasonal_NPO_JF_ANTAO_JFM_SAO_SOND.parquet"
USE_AGG_CACHE = AGG_PARQUET.is_file() and False  # force full pipeline by default

SPEXONE_HOMOGENIZE = True
SSA_SPEXONE_HOMOGENIZE = True
AOD_SPEXONE_HOMOGENIZE = True

QUALITY_FILTER_PARAMS = {
    "min_aod": 0.01,
    "min_aaod": 1e-5,
    "max_aaod_fraction": 0.95,
    "min_ssa": 0.5,
    "max_ssa": 1.02,
}

# Ocean basins from AOD_SS_ERROR; time windows remapped onto the PPE year
# (Aug 2024 – Jul 2025), keeping the same month-of-year seasons.
REGIONS = {
    "NPO_JF": {
        "surface_type": "ocean",
        "lon_range": (120, 260),
        "lat_range": (20, 60),
        "time_slice": ("2025-01-01", "2025-02-28"),
        "edge_weighted": False,
    },
    "ANTAO_JFM": {
        "surface_type": "ocean",
        "lon_range": (0, 360),
        "lat_range": (-90, -60),
        "time_slice": ("2025-01-01", "2025-03-01"),
        "edge_weighted": False,
    },
    "SAO_SOND": {
        "surface_type": "ocean",
        "lon_range": (290, 360),
        "lat_range": (-60, -20),
        "time_slice": ("2024-09-01", "2024-12-31"),
        "edge_weighted": False,
    },
}
SOURCE_REGIONS = ["NPO_JF", "ANTAO_JFM", "SAO_SOND"]
ANALYSIS_REGIONS = list(SOURCE_REGIONS)

REGION_COLORS = {
    "NPO_JF": "#2ca02c",
    "ANTAO_JFM": "#ff7f0e",
    "SAO_SOND": "#1f77b4",
}

LOAD_VARS = ["loadbc", "loaddust", "loadoa", "loadso4", "loadss"]
EMI_VARS = ["emibc", "emidust", "emioa", "emiso2", "emiss"]

VARIABLES_NEEDED = [
    "od440aer", "od550aer", "od865aer", "abs550aer", "od550ss", "od550so4",
    "precip",
    "loadbc", "loaddust", "loadoa", "loadso4", "loadss",
    "emibc", "emidust", "emioa", "emiso2", "emiss",
]


def ens_label(ens_id: int) -> str:
    """Stable member name (experiment strings in NetCDF are truncated)."""
    eid = int(ens_id)
    return "Control" if eid < 0 else str(eid)


def save_fig(fig, name: str) -> Path:
    path = FIGURE_DIR / name
    if SAVE_FIGURE:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)
    return path


print("--- Active configuration ---")
print(f"  PPE_DIR              = {PPE_DIR}")
print(f"  ANALYSIS_REGIONS     = {ANALYSIS_REGIONS}")
print(f"  DERIVED_VAR_AFTER_AGG= {DERIVED_VAR_AFTER_AGG}")
print(f"  BURDEN_MASS_SCALE    = {BURDEN_MASS_SCALE}")
print(f"  SPEXONE_HOMOGENIZE   = {SPEXONE_HOMOGENIZE}")
print(f"  SAVE_FIGURE          = {SAVE_FIGURE}")
print(f"  FIGURE_DIR           = {FIGURE_DIR}")
print(f"  EXCLUDE_MEMBERS      = {EXCLUDE_MEMBERS}")

# %%
# -----------------------------------------------------------------------------
# Load PPE monthly ensemble NetCDFs
# -----------------------------------------------------------------------------
GROUP_FILES = {
    "burden": PPE_DIR / "burden_monthly.nc",
    "emi": PPE_DIR / "emi_monthly.nc",
    "optical": PPE_DIR / "optical_monthly.nc",
    "precip": PPE_DIR / "precip_monthly.nc",
}


def load_ppe_as_model_dict(ppe_files: Dict[str, Path]) -> Tuple[Dict[str, Dict[str, xr.Dataset]], List[str]]:
    """Return data[member][var] = Dataset and sorted member labels."""
    for key, path in ppe_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing PPE group file ({key}): {path}")

    opened = {k: xr.open_dataset(p) for k, p in ppe_files.items()}
    try:
        ens_ids = [int(e) for e in opened["burden"].ens.values]
        labels = [ens_label(e) for e in ens_ids]
        if EXCLUDE_MEMBERS:
            keep = [
                (e, lab) for e, lab in zip(ens_ids, labels)
                if lab not in EXCLUDE_MEMBERS and str(e) not in EXCLUDE_MEMBERS
            ]
            ens_ids, labels = zip(*keep) if keep else ([], [])
            ens_ids, labels = list(ens_ids), list(labels)

        data: Dict[str, Dict[str, xr.Dataset]] = {lab: {} for lab in labels}
        for lab, eid in zip(labels, ens_ids):
            for group_ds in opened.values():
                sub = group_ds.sel(ens=eid)
                for var in sub.data_vars:
                    if var in ("experiment",):
                        continue
                    da = sub[var]
                    if "ens" in da.dims:
                        da = da.squeeze("ens", drop=True)
                    elif "ens" in da.coords:
                        da = da.drop_vars("ens")
                    # Drop truncated experiment label coord
                    if "experiment" in da.coords:
                        da = da.drop_vars("experiment")
                    ds_var = da.to_dataset(name=var)
                    data[lab][var] = ds_var
        return data, labels
    finally:
        for ds in opened.values():
            ds.close()


if USE_AGG_CACHE:
    print("USE_AGG_CACHE=True: skipping NetCDF load")
    raw_data = {}
    models = []
else:
    print(f"Loading PPE monthly ensembles from {PPE_DIR} ...")
    raw_data, models = load_ppe_as_model_dict(GROUP_FILES)
    print(f"Ensemble members: {len(models)}")
    if models:
        print(f"  first={models[0]}  last={models[-1]}")

# %%
# -----------------------------------------------------------------------------
# Normalize / sums
# -----------------------------------------------------------------------------
def apply_aerosol_quality_filter(
    aod, aaod=None, ssa=None,
    min_aod=0.01, min_aaod=1e-5,
    max_aaod_fraction=0.95,
    min_ssa=0.5, max_ssa=1.02,
):
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
        abs_mask = (
            abs_mask
            & (aaod > min_aaod)
            & (aaod < aod * max_aaod_fraction)
            & np.isfinite(aaod)
        )
    if ssa is not None:
        abs_mask = abs_mask & (ssa >= min_ssa) & (ssa <= max_ssa) & np.isfinite(ssa)

    aod_out = _where(aod, aod_mask)
    aaod_out = _where(aaod, abs_mask)
    ssa_out = _where(ssa, abs_mask)
    if ssa_out is not None:
        ssa_out = ssa_out.clip(0.0, 1.0)
    return aod_out, aaod_out, ssa_out


if USE_AGG_CACHE:
    print("USE_AGG_CACHE=True: skipping normalize/sums")
    data = {}
    target_models = []
else:
    data = {}
    for m in models:
        normalized = {}
        for var in VARIABLES_NEEDED:
            if raw_data[m].get(var) is None:
                continue
            try:
                normalized[var] = setup.normalize_dataset_time(
                    raw_data[m][var], var_hint=f"{m}/{var}"
                )
            except Exception as e:
                print(f"  Failed to normalize {m}/{var}: {e}")
                normalized[var] = raw_data[m][var]
        # Correct PPE burden magnitude (see BURDEN_MASS_SCALE).
        if BURDEN_MASS_SCALE != 1.0:
            for key in LOAD_VARS:
                ds = normalized.get(key)
                if ds is None:
                    continue
                da = _get_dataarray(ds, key)
                if da is None:
                    continue
                da = da * BURDEN_MASS_SCALE
                da.attrs["units"] = "kg m-2"
                da.attrs["burden_mass_scale"] = float(BURDEN_MASS_SCALE)
                normalized[key] = da.to_dataset(name=key)

        normalized["load_total"] = setup.sum_datasets(
            normalized, LOAD_VARS, "load_total", require_all=True
        )
        normalized["emi_total"] = setup.sum_datasets(
            normalized, EMI_VARS, "emi_total", require_all=True
        )
        data[m] = normalized

    target_models = [
        m for m in models
        if data[m].get("od550aer") is not None
        and data[m].get("od550ss") is not None
        and data[m].get("loadss") is not None
        and data[m].get("emiss") is not None
    ]
    print(f"Target members with od550aer+od550ss+loadss+emiss: {len(target_models)}")
    if target_models and BURDEN_MASS_SCALE != 1.0:
        print(f"Applied BURDEN_MASS_SCALE={BURDEN_MASS_SCALE} to {LOAD_VARS}")

# Free raw handles (datasets already copied into data)
raw_data = None

# %%
# -----------------------------------------------------------------------------
# Pre-aggregation derived variables (MEC, SSA, AE, MAC)
# -----------------------------------------------------------------------------
if USE_AGG_CACHE:
    print("USE_AGG_CACHE=True: skipping derived vars")
    data_derived = {}
else:
    derived_vars_pre = [
        dv for dv in ["MEC", "MEC_ss", "MEC_so4", "SSA", "AE", "MAC"]
        if dv not in DERIVED_VAR_AFTER_AGG
    ]
    derived = {m: {} for m in target_models}

    for m in target_models:
        aerocom_data.align_model_grids(data[m], ref_var="od550aer", model_hint=m)

    print(f"Grid-aligned {len(target_models)} members onto od550aer.")

    for m in target_models:
        model_data = data[m]
        aod550 = _get_dataarray(model_data.get("od550aer"), "od550aer")
        if aod550 is None:
            continue
        aaod = _get_dataarray(model_data.get("abs550aer"), "abs550aer")
        if aaod is not None:
            aaod = _align_da_to_ref(aaod, aod550, model_hint=f"{m}/abs550aer")
            aod550_f, aaod_f, ssa = apply_aerosol_quality_filter(
                aod=aod550, aaod=aaod, ssa=1.0 - aaod / aod550,
                **QUALITY_FILTER_PARAMS,
            )
            aod550 = aod550_f
            aaod = aaod_f
        else:
            ssa = None

        loads = []
        for key in LOAD_VARS:
            da = _get_dataarray(model_data.get(key), key)
            if da is not None:
                loads.append(_align_da_to_ref(da, aod550, model_hint=f"{m}/{key}"))
        total_load = sum(loads) if loads else None

        aod_ss = _get_dataarray(model_data.get("od550ss"), "od550ss")
        load_ss = _get_dataarray(model_data.get("loadss"), "loadss")
        aod_so4 = _get_dataarray(model_data.get("od550so4"), "od550so4")
        load_so4 = _get_dataarray(model_data.get("loadso4"), "loadso4")

        for dv in derived_vars_pre:
            try:
                if dv == "MEC":
                    if total_load is None:
                        raise KeyError("Missing load_total components.")
                    derived[m]["MEC"] = aod550 / (total_load * 1e3)
                elif dv == "MEC_ss":
                    if aod_ss is None or load_ss is None:
                        raise KeyError("Missing od550ss or loadss.")
                    aod_ss_a = _align_da_to_ref(aod_ss, aod550, model_hint=f"{m}/od550ss")
                    load_ss_a = _align_da_to_ref(load_ss, aod550, model_hint=f"{m}/loadss")
                    derived[m]["MEC_ss"] = aod_ss_a / (load_ss_a * 1e3)
                elif dv == "MEC_so4":
                    if aod_so4 is None or load_so4 is None:
                        raise KeyError("Missing od550so4 or loadso4.")
                    aod_so4_a = _align_da_to_ref(aod_so4, aod550, model_hint=f"{m}/od550so4")
                    load_so4_a = _align_da_to_ref(load_so4, aod550, model_hint=f"{m}/loadso4")
                    derived[m]["MEC_so4"] = aod_so4_a / (load_so4_a * 1e3)
                elif dv == "SSA":
                    if ssa is None:
                        raise KeyError("Missing abs550aer for SSA.")
                    derived[m]["SSA"] = ssa
                elif dv == "MAC":
                    if aaod is None or total_load is None:
                        raise KeyError("Missing abs550aer or loads for MAC.")
                    derived[m]["MAC"] = aaod / (total_load * 1e3)
                elif dv == "AE":
                    aod_other = _get_dataarray(model_data.get("od865aer"), "od865aer")
                    other_wvl = 865.0
                    if aod_other is None:
                        aod_other = _get_dataarray(model_data.get("od440aer"), "od440aer")
                        other_wvl = 440.0
                    if aod_other is None:
                        raise KeyError("Missing od865aer/od440aer for AE.")
                    aod_other = _align_da_to_ref(aod_other, aod550, model_hint=f"{m}/spectral_aod")
                    aod_other = aod_other.where(np.isfinite(aod550))
                    derived[m]["AE"] = -np.log(aod550 / aod_other) / np.log(550.0 / other_wvl)
            except Exception as e:
                print(f"  Derived {dv} failed for {m}: {e}")
                derived[m][dv] = None

    # Merge derived into data_derived as Datasets
    data_derived = {}
    for m in target_models:
        data_derived[m] = dict(data[m])
        for dv, val in derived[m].items():
            if val is None:
                data_derived[m][dv] = None
            elif isinstance(val, xr.Dataset):
                data_derived[m][dv] = val
            else:
                data_derived[m][dv] = val.to_dataset(name=dv)

# %%
# -----------------------------------------------------------------------------
# Regional aggregation
# -----------------------------------------------------------------------------
if USE_AGG_CACHE:
    print("USE_AGG_CACHE=True: skipping regional aggregation")
    model_monthly = {}
    model_seasonal = {}
    masks = {}
else:
    template = data[target_models[0]]["od550aer"].isel(time=0)
    masks = setup.create_analysis_masks(
        template, ANALYSIS_REGIONS, region=REGIONS, surface_type=None
    )
    print("Regions created:", list(masks.keys()))

    variables_pre_aggregated = [
        "MEC", "MAC", "SSA", "AE", "MEC_ss", "MEC_so4",
        "od550so4", "od550ss", "loadso4", "loadss", "emiss", "emiso2",
        "precip", "load_total", "emi_total", "od550aer", "abs550aer",
    ]
    variables_to_aggregate = [
        v for v in variables_pre_aggregated if v not in DERIVED_VAR_AFTER_AGG
    ]

    def _as_da_dict(model_dict, var_name):
        """Unwrap Datasets to DataArrays for cameo regional_aggregate."""
        out = {}
        for model, md in model_dict.items():
            if var_name not in md or md[var_name] is None:
                continue
            val = md[var_name]
            if isinstance(val, xr.Dataset):
                val = _get_dataarray(val, var_name)
            out[model] = {var_name: val}
        return out

    def _aggregate_all(return_time_series):
        result = {}
        for region in ANALYSIS_REGIONS:
            result[region] = {}
            for var in variables_to_aggregate:
                result[region][var] = setup.aggregate_region(
                    _as_da_dict(data_derived, var),
                    var,
                    region,
                    masks,
                    region=REGIONS,
                    return_time_series=return_time_series,
                    skipna=True,
                )
        return result

    model_monthly = _aggregate_all(return_time_series=True)
    model_seasonal = _aggregate_all(return_time_series=False)

# %%
# -----------------------------------------------------------------------------
# Post-aggregation lifetimes (emission path only)
# -----------------------------------------------------------------------------
LIFETIME_SPECS = {
    "lifetime_ss": ("loadss", "emiss"),
    "lifetime": ("load_total", "emi_total"),
}

if USE_AGG_CACHE:
    print("USE_AGG_CACHE=True: skipping post-aggregation lifetimes")
else:
    for agg_name, agg in (("monthly", model_monthly), ("seasonal", model_seasonal)):
        for region in ANALYSIS_REGIONS:
            for out_key, (load_key, flux_key) in LIFETIME_SPECS.items():
                load = agg[region].get(load_key, {})
                flux = agg[region].get(flux_key, {})
                agg[region][out_key] = (
                    setup._lifetime_from_load_flux(load, flux, out_key, region)
                    if load and flux else {}
                )
            print(
                f"  {agg_name}/{region}: "
                f"lifetime_ss={len(agg[region]['lifetime_ss'])}, "
                f"lifetime={len(agg[region]['lifetime'])}"
            )

# %%
# -----------------------------------------------------------------------------
# DataFrames
# -----------------------------------------------------------------------------
if USE_AGG_CACHE:
    model_seasonal_df = pd.read_parquet(AGG_PARQUET)
    monthly_path = TABLE_DIR / "seasalt_ppe_monthly_NPO_JF_ANTAO_JFM_SAO_SOND.parquet"
    model_monthly_df = pd.read_parquet(monthly_path) if monthly_path.is_file() else pd.DataFrame()
else:
    model_monthly_df, model_seasonal_df = setup.convert_aggregation_to_dataframes(
        model_monthly, model_seasonal, month_col="month"
    )
    if SAVE_AGG_CACHE:
        monthly_path = TABLE_DIR / "seasalt_ppe_monthly_NPO_JF_ANTAO_JFM_SAO_SOND.parquet"
        model_monthly_df.to_parquet(monthly_path)
        model_seasonal_df.to_parquet(AGG_PARQUET)
        print(f"Wrote {monthly_path}")
        print(f"Wrote {AGG_PARQUET}")

print("\nSeasonal availability:")
availability_columns = [
    var for var in ["MEC_ss", "MAC", "SSA", "AE", "lifetime_ss", "lifetime", "precip", "abs550aer"]
    if var in model_seasonal_df.columns
]
if not model_seasonal_df.empty and availability_columns:
    print(model_seasonal_df.groupby("region")[availability_columns].count())
print(model_seasonal_df.head())

# %%
# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def _axis_label(col: str, override=None) -> str:
    if override:
        return override
    return AXIS_LABELS.get(col, col)


def plot_combined_regions(
    model_seasonal_df,
    x_col="SSA",
    y_col="MEC_ss",
    x_label=None,
    y_label=None,
    force_intercept_zero=False,
    filename="combined_regions.png",
    title=None,
):
    if model_seasonal_df.empty:
        print("model_seasonal_df is empty.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    for idx, region in enumerate(SOURCE_REGIONS):
        cols = ["model", y_col, "SSA" if x_col == "1-SSA" else x_col]
        missing = [c for c in cols if c not in model_seasonal_df.columns and c != "1-SSA"]
        if missing:
            print(f"Skipping {region}: missing columns {missing}")
            continue
        df = model_seasonal_df.loc[model_seasonal_df["region"] == region, [c for c in cols if c in model_seasonal_df.columns]].copy()
        if x_col == "1-SSA":
            df[x_col] = 1.0 - df["SSA"]
        df = df[["model", x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty or len(df) < 2:
            print(f"Skipping {region}: insufficient points ({len(df)}).")
            continue
        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        color = REGION_COLORS.get(region, f"C{idx}")
        ax.scatter(x, y, s=40, alpha=0.65, color=color, edgecolors="k", linewidths=0.4, label=region)
        if force_intercept_zero:
            slope = float(np.nansum(x * y) / np.nansum(x * x)) if np.nansum(x * x) else np.nan
            intercept = 0.0
            r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
        else:
            lr = linregress(x, y)
            slope, intercept, r = lr.slope, lr.intercept, lr.rvalue
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ax.plot(xs, slope * xs + intercept, color=color, lw=1.5,
                label=f"{region}: y={slope:.2f}x+{intercept:.2f} r={r:.2f}")

    ax.set_xlabel(_axis_label(x_col, x_label))
    ax.set_ylabel(_axis_label(y_col, y_label))
    ax.set_title(title or f"PPE ensemble: {_axis_label(y_col)} vs {_axis_label(x_col)}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    save_fig(fig, filename)


def plot_tau_vs_precip(
    df,
    lifetime_col="lifetime_ss",
    x_var="precip",
    source_regions=None,
    filename=None,
    title=None,
    inv_ylabel=None,
):
    source_regions = source_regions or SOURCE_REGIONS
    if lifetime_col not in df.columns or x_var not in df.columns:
        print(f"Skip plot_tau_vs_precip: missing {lifetime_col} or {x_var}")
        return
    inv_key = "inv_lifetime_ss" if lifetime_col.endswith("_ss") else "inv_lifetime"
    fig, axes = plt.subplots(1, len(source_regions), figsize=(5 * len(source_regions), 4), squeeze=False)
    axes = axes.ravel()
    for ax, region in zip(axes, source_regions):
        sub = df.loc[df["region"] == region, ["model", x_var, lifetime_col]].replace([np.inf, -np.inf], np.nan).dropna()
        sub = sub.loc[sub[lifetime_col] > 0]
        if sub.empty:
            ax.set_title(f"{region}: no data")
            continue
        y = 1.0 / sub[lifetime_col].to_numpy(dtype=float)
        x = sub[x_var].to_numpy(dtype=float)
        ax.scatter(x, y, s=35, alpha=0.7, edgecolors="k", linewidths=0.3)
        if len(sub) >= 2:
            lr = linregress(x, y)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 40)
            ax.plot(xs, lr.slope * xs + lr.intercept, "r-", lw=1.5)
            ax.set_title(f"{region}: r={lr.rvalue:.2f} n={len(sub)}")
        else:
            ax.set_title(f"{region}: n={len(sub)}")
        ax.set_xlabel(_axis_label(x_var))
        ax.set_ylabel(inv_ylabel or _axis_label(inv_key))
        ax.grid(True, alpha=0.3)
    fig.suptitle(title or f"PPE: {_axis_label(inv_key)} vs {_axis_label(x_var)}")
    fig.tight_layout()
    save_fig(fig, filename or f"ppe_{lifetime_col}_vs_{x_var}.png")


def plot_scatter_by_region(
    df, x_col, y_col, color_col=None, regions=None,
    x_label=None, y_label=None, color_label=None,
    fit_line=False, filename="scatter_by_region.png", title=None,
):
    regions = regions or SOURCE_REGIONS
    need = [x_col, y_col] + ([color_col] if color_col else [])
    if any(c not in df.columns for c in need):
        print(f"Skip scatter: missing {need}")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = None
    for idx, region in enumerate(regions):
        cols = ["model", x_col, y_col] + ([color_col] if color_col else [])
        sub = df.loc[df["region"] == region, cols].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue
        color = REGION_COLORS.get(region, f"C{idx}")
        sc = ax.scatter(
            sub[x_col], sub[y_col], c=sub[color_col] if color_col else color,
            s=40, alpha=0.7, edgecolors="k", linewidths=0.3, label=region,
            cmap="viridis" if color_col else None,
        )
        if fit_line and len(sub) >= 2:
            lr = linregress(sub[x_col], sub[y_col])
            xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 40)
            ax.plot(xs, lr.slope * xs + lr.intercept, color=color, lw=1.4,
                    label=f"{region} r={lr.rvalue:.2f}")
    if color_col and sc is not None:
        fig.colorbar(sc, ax=ax, label=_axis_label(color_col, color_label))
    ax.set_xlabel(_axis_label(x_col, x_label))
    ax.set_ylabel(_axis_label(y_col, y_label))
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_fig(fig, filename)


def print_derived_range_check(df: pd.DataFrame) -> None:
    """Sanity-check MEC and inverse-lifetime ranges after burden scaling."""
    print("\n--- Derived-variable range check (seasonal) ---")
    checks = [
        ("MEC_ss", (0.0, 15.0)),
        ("MEC", (0.0, 15.0)),
        ("lifetime_ss", None),
        ("lifetime", None),
    ]
    for col, expected in checks:
        if col not in df.columns:
            print(f"  {col}: missing")
            continue
        s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            print(f"  {col}: no finite values")
            continue
        q = s.quantile([0.05, 0.5, 0.95])
        print(
            f"  {col}: median={q.loc[0.5]:.4g}  "
            f"q05={q.loc[0.05]:.4g}  q95={q.loc[0.95]:.4g}  "
            f"n={len(s)}"
        )
        if col.startswith("lifetime"):
            inv = 1.0 / s[s > 0]
            qi = inv.quantile([0.05, 0.5, 0.95])
            print(
                f"    1/{col}: median={qi.loc[0.5]:.4g}  "
                f"q05={qi.loc[0.05]:.4g}  q95={qi.loc[0.95]:.4g}  "
                f"(expect ~0.1–1 day^-1)"
            )
        elif expected is not None:
            lo, hi = expected
            inside = ((s >= lo) & (s <= hi)).mean() * 100
            print(f"    fraction in [{lo}, {hi}]: {inside:.1f}%")


# %%
# -----------------------------------------------------------------------------
# Model-side figures (sea-salt only + total aerosol)
# -----------------------------------------------------------------------------
if not model_seasonal_df.empty:
    print_derived_range_check(model_seasonal_df)

    # --- Total aerosol ---
    plot_combined_regions(
        model_seasonal_df, x_col="SSA", y_col="MEC",
        force_intercept_zero=False,
        filename="ppe_SSA_vs_MEC.png",
        title="PPE total aerosol: MEC vs SSA",
    )
    plot_combined_regions(
        model_seasonal_df, x_col="1-SSA", y_col="MEC",
        force_intercept_zero=True,
        filename="ppe_1SSA_vs_MEC.png",
        title="PPE total aerosol: MEC vs 1−SSA",
    )
    if "MAC" in model_seasonal_df.columns:
        plot_combined_regions(
            model_seasonal_df, x_col="1-SSA", y_col="MAC",
            force_intercept_zero=True,
            filename="ppe_1SSA_vs_MAC.png",
            title="PPE total aerosol: MAC vs 1−SSA",
        )

    # --- Sea salt only ---
    plot_combined_regions(
        model_seasonal_df, x_col="SSA", y_col="MEC_ss",
        force_intercept_zero=False,
        filename="ppe_SSA_vs_MEC_ss.png",
        title="PPE sea salt: MEC$_{SS}$ vs SSA",
    )
    plot_combined_regions(
        model_seasonal_df, x_col="1-SSA", y_col="MEC_ss",
        force_intercept_zero=True,
        filename="ppe_1SSA_vs_MEC_ss.png",
        title="PPE sea salt: MEC$_{SS}$ vs 1−SSA",
    )

    # Inverse lifetime vs precip / AE — sea salt and total
    plot_tau_vs_precip(
        model_seasonal_df, "lifetime_ss", "precip",
        filename="ppe_inv_lifetime_ss_vs_precip.png",
        title=r"PPE sea salt: 1/$\tau_{SS}$ vs precipitation",
    )
    plot_tau_vs_precip(
        model_seasonal_df, "lifetime", "precip",
        filename="ppe_inv_lifetime_vs_precip.png",
        title=r"PPE total aerosol: 1/$\tau$ vs precipitation",
    )
    plot_tau_vs_precip(
        model_seasonal_df, "lifetime_ss", "AE",
        filename="ppe_inv_lifetime_ss_vs_AE.png",
        title=r"PPE sea salt: 1/$\tau_{SS}$ vs AE",
    )
    plot_tau_vs_precip(
        model_seasonal_df, "lifetime", "AE",
        filename="ppe_inv_lifetime_vs_AE.png",
        title=r"PPE total aerosol: 1/$\tau$ vs AE",
    )

    inv = model_seasonal_df.copy()
    if "lifetime_ss" in inv.columns:
        inv["inv_lifetime_ss"] = 1.0 / inv["lifetime_ss"]
        plot_scatter_by_region(
            inv, x_col="precip", y_col="inv_lifetime_ss", color_col="AE",
            regions=SOURCE_REGIONS, fit_line=True,
            filename="ppe_precip_vs_inv_lifetime_ss_colored_AE.png",
            title=r"PPE sea salt: 1/$\tau_{SS}$ vs precip (colored by AE)",
        )
    if "lifetime" in inv.columns:
        inv["inv_lifetime"] = 1.0 / inv["lifetime"]
        plot_scatter_by_region(
            inv, x_col="precip", y_col="inv_lifetime", color_col="AE",
            regions=SOURCE_REGIONS, fit_line=True,
            filename="ppe_precip_vs_inv_lifetime_colored_AE.png",
            title=r"PPE total aerosol: 1/$\tau$ vs precip (colored by AE)",
        )

    # 3D: 1/lifetime vs precip (x) and AE (y) — sea salt and total
    # (ported from AOD_SS_ERROR.ipynb; no member labels)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def plot_lifetime_ae_precip_3d(
        df,
        lifetime_col="lifetime_ss",
        source_regions=None,
        angle=35,
        elev=15,
        inverse=True,
        filename=None,
        title=None,
    ):
        """3D scatter of (inverse) lifetime vs precip and AE, plus OLS plane."""
        source_regions = source_regions or SOURCE_REGIONS
        need = ["region", "model", "AE", "precip", lifetime_col]
        if any(c not in df.columns for c in need):
            print(f"Skip 3D plot: missing columns for {lifetime_col}")
            return None
        plot_df = (
            df.loc[df["region"].isin(source_regions), need]
            .rename(columns={"precip": "model_precip", lifetime_col: "lifetime"})
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .copy()
        )
        plot_df = plot_df.loc[plot_df["lifetime"] > 0]
        plot_df["z"] = (1.0 / plot_df["lifetime"]) if inverse else plot_df["lifetime"]
        inv_key = "inv_lifetime_ss" if lifetime_col.endswith("_ss") else "inv_lifetime"
        z_label = _axis_label(inv_key) if inverse else _axis_label(lifetime_col)
        print(f"Valid AE / precip / {z_label} rows: {len(plot_df)}")
        if plot_df.empty:
            return plot_df

        ncols = 2
        nrows = int(np.ceil(len(source_regions) / ncols))
        fig = plt.figure(figsize=(13, 5.5 * nrows))
        for idx, region in enumerate(source_regions):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
            region_color = REGION_COLORS.get(region, "#333333")
            sub = plot_df[plot_df["region"] == region]
            if len(sub) < 3:
                ax.set_title(f"{region}: insufficient data (n={len(sub)})")
                continue
            x = sub["model_precip"].to_numpy(dtype=float)
            y = sub["AE"].to_numpy(dtype=float)
            z = sub["z"].to_numpy(dtype=float)
            ax.scatter(x, y, z, c=region_color, s=40, edgecolor="black",
                       linewidths=0.3, depthshade=True, alpha=0.75)

            A = np.column_stack([x, y, np.ones_like(x)])
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
            a, b, c = coef
            z_hat = A @ coef
            ss_res = np.sum((z - z_hat) ** 2)
            ss_tot = np.sum((z - z.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            xg = np.linspace(x.min(), x.max(), 12)
            yg = np.linspace(y.min(), y.max(), 12)
            xx, yy = np.meshgrid(xg, yg)
            zz = a * xx + b * yy + c
            ax.plot_surface(xx, yy, zz, alpha=0.25, color=region_color, linewidth=0)
            ax.text2D(
                0.03, 0.97,
                f"1/$\\tau$ = {a:.3g} P + {b:.3g} AE + {c:.3g}\n"
                f"$R^2$ = {r2:.2f}  n={len(sub)}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor=region_color, alpha=0.85),
            )
            ax.set_xlabel(_axis_label("precip"))
            ax.set_ylabel(_axis_label("AE"))
            ax.set_zlabel(z_label)
            ax.set_title(region)
            ax.view_init(elev=elev, azim=angle)

        fig.suptitle(
            title or f"PPE: {z_label} vs precip and AE (3D OLS)",
            y=1.02, fontsize=13,
        )
        fig.tight_layout()
        save_fig(
            fig,
            filename or f"ppe_3d_{'inv_' if inverse else ''}{lifetime_col}_vs_precip_AE.png",
        )
        return plot_df

    if "lifetime_ss" in model_seasonal_df.columns:
        plot_lifetime_ae_precip_3d(
            model_seasonal_df,
            lifetime_col="lifetime_ss",
            angle=35, elev=15, inverse=True,
            filename="ppe_3d_inv_lifetime_ss_vs_precip_AE.png",
            title=r"PPE sea salt: 1/$\tau_{SS}$ vs precip and AE",
        )
    if "lifetime" in model_seasonal_df.columns:
        plot_lifetime_ae_precip_3d(
            model_seasonal_df,
            lifetime_col="lifetime",
            angle=35, elev=15, inverse=True,
            filename="ppe_3d_inv_lifetime_vs_precip_AE.png",
            title=r"PPE total aerosol: 1/$\tau$ vs precip and AE",
        )

# %%
# -----------------------------------------------------------------------------
# SPEXone load + regional means (replaces POLDER)
# -----------------------------------------------------------------------------
def load_spexone() -> pd.DataFrame:
    files = []
    for year, month in SPEXONE_YEARS_MONTHS:
        path = Path(SPEXONE_GLOB.format(year=year, month=month))
        if path.is_file():
            files.append(path)
        else:
            print(f"  SPEXone missing: {path.name}")
    if not files:
        raise FileNotFoundError("No SPEXone monthly parquet files found for PPE overlap.")
    print(f"Loading {len(files)} SPEXone monthly files ...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["month"] = df["date"].dt.to_period("M")
    # SPEXone lon is typically -180..180; PPE / REGIONS use 0..360
    df["lon360"] = (df["lon_bin"].astype(float) + 360.0) % 360.0
    df["latitude"] = df["lat_bin"].astype(float)
    aod_col = f"aod_{WVL}"
    aaod_col = f"aaod_{WVL}"
    ssa_col = f"ssa_{WVL}"
    df["AOD_550"] = df[aod_col] if aod_col in df.columns else np.nan
    df["AAOD_550"] = df[aaod_col] if aaod_col in df.columns else np.nan
    if ssa_col in df.columns:
        df["SSA"] = df[ssa_col]
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["SSA"] = 1.0 - df["AAOD_550"] / df["AOD_550"]
    if df["AAOD_550"].isna().all() and ssa_col in df.columns:
        df["AAOD_550"] = (1.0 - df[ssa_col]) * df["AOD_550"]
    with np.errstate(divide="ignore", invalid="ignore"):
        if "aod_865nm" in df.columns:
            df["AE"] = -np.log(df["AOD_550"] / df["aod_865nm"]) / np.log(550.0 / 865.0)
        else:
            df["AE"] = np.nan
    df = df.replace([np.inf, -np.inf], np.nan)
    aod_f, aaod_f, ssa_f = apply_aerosol_quality_filter(
        aod=df["AOD_550"], aaod=df["AAOD_550"], ssa=df["SSA"], **QUALITY_FILTER_PARAMS
    )
    df["AOD_550"] = aod_f
    df["AAOD_550"] = aaod_f
    df["SSA"] = ssa_f
    print(f"SPEXone rows: {len(df):,}  valid AOD+AAOD+SSA: {df[['AOD_550','AAOD_550','SSA']].notna().all(axis=1).sum():,}")
    return df


def spexone_in_region(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    cfg = REGIONS[region_name]
    lon0, lon1 = cfg["lon_range"]
    lat0, lat1 = cfg["lat_range"]
    work = df.copy()
    if lon0 <= lon1:
        lon_ok = (work["lon360"] >= lon0) & (work["lon360"] <= lon1)
    else:
        lon_ok = (work["lon360"] >= lon0) | (work["lon360"] <= lon1)
    work = work[lon_ok & (work["latitude"] >= lat0) & (work["latitude"] <= lat1)].copy()
    t0, t1 = cfg["time_slice"]
    work = work[(work["date"] >= pd.Timestamp(t0)) & (work["date"] <= pd.Timestamp(t1))].copy()
    return work


def spexone_weighted_mean(sub: pd.DataFrame, col: str) -> float:
    sub = sub.dropna(subset=[col, "latitude"])
    if sub.empty:
        return np.nan
    w = np.cos(np.deg2rad(sub["latitude"].to_numpy(dtype=float)))
    vals = sub[col].to_numpy(dtype=float)
    finite = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if finite.sum() < 3:
        return np.nan
    return float(np.average(vals[finite], weights=w[finite]))


print("\n--- SPEXone ---")
try:
    spex_df = load_spexone()
except Exception as e:
    print(f"SPEXone load failed: {e}")
    spex_df = pd.DataFrame()

spex_obs = {}
if not spex_df.empty:
    for region in ANALYSIS_REGIONS:
        sub = spexone_in_region(spex_df, region)
        spex_obs[region] = {
            "AOD_550": spexone_weighted_mean(sub, "AOD_550"),
            "AAOD_550": spexone_weighted_mean(sub, "AAOD_550"),
            "SSA": spexone_weighted_mean(sub, "SSA"),
            "AE": spexone_weighted_mean(sub, "AE"),
            "n": int(len(sub)),
        }
        print(
            f"  {region}: n={spex_obs[region]['n']}  "
            f"AOD={spex_obs[region]['AOD_550']:.4f}  "
            f"SSA={spex_obs[region]['SSA']:.4f}"
        )

# SPEXone regional maps
if not spex_df.empty and SAVE_FIGURE:
    nreg = len(ANALYSIS_REGIONS)
    fig, axes = plt.subplots(nreg, 2, figsize=(12, 3.5 * nreg), squeeze=False)
    for i, region in enumerate(ANALYSIS_REGIONS):
        sub = spexone_in_region(spex_df, region)
        if sub.empty:
            continue
        grid = sub.groupby(["lon_bin", "lat_bin"], as_index=False)[["AOD_550", "SSA"]].mean()
        for j, col in enumerate(["AOD_550", "SSA"]):
            ax = axes[i, j]
            sc = ax.scatter(grid["lon_bin"], grid["lat_bin"], c=grid[col], s=8, cmap="viridis")
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
            unit = " (1)" if col in ("AOD_550", "SSA") else ""
            cbar.set_label(f"{col}{unit}")
            ax.set_title(f"{region} SPEXone {col}{unit}")
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
    fig.tight_layout()
    save_fig(fig, "ppe_spexone_regional_maps.png")

# %%
# -----------------------------------------------------------------------------
# SPEXone homogenization (same template as POLDER cells 32–34)
# -----------------------------------------------------------------------------
def sample_model_var_at_spexone(model_da, spex_sub: pd.DataFrame) -> float:
    """Sample model monthly field at SPEXone bins; cosine-lat weighted mean."""
    if model_da is None or spex_sub.empty:
        return np.nan
    if isinstance(model_da, xr.Dataset):
        model_da = _get_dataarray(model_da, list(model_da.data_vars)[0])
    lat_vals = model_da.lat.values
    lon_vals = model_da.lon.values
    lat_idx = np.abs(lat_vals[:, None] - spex_sub["latitude"].values[None, :]).argmin(axis=0)
    lon_idx = np.abs(lon_vals[:, None] - spex_sub["lon360"].values[None, :]).argmin(axis=0)
    times = pd.to_datetime(model_da.time.values)
    spex_times = pd.to_datetime(spex_sub["date"].values)
    month_idx = np.array([
        np.argmin(np.abs(times - pd.Timestamp(t).to_period("M").to_timestamp()))
        for t in spex_times
    ])
    vals = np.asarray(model_da.values)[month_idx, lat_idx, lon_idx]
    weights = np.cos(np.deg2rad(spex_sub["latitude"].values.astype(float)))
    finite = np.isfinite(vals) & np.isfinite(weights)
    if finite.sum() < 3:
        return np.nan
    return float(np.average(vals[finite], weights=weights[finite]))


def _regional_dict_from_df(df, region, var_col):
    if var_col not in df.columns:
        return {}
    sub = df.loc[df["region"] == region, ["model", var_col]].dropna()
    return dict(zip(sub["model"].astype(str), sub[var_col].astype(float)))


def homogenize_spexone_variable(
    model_seasonal_df,
    spex_raw: pd.DataFrame,
    obs_col: str,
    model_var_key: str,
    var_label: str,
    data_derived_dict: dict,
):
    homogenized = {}
    details = {}
    rows = []
    print(f"\n--- SPEXone homogenization ({var_label}) ---")
    for region in ANALYSIS_REGIONS:
        sub = spexone_in_region(spex_raw, region)
        if sub.empty:
            print(f"  {region}: no SPEXone samples")
            details[region] = {"skipped": True}
            continue
        obs_sampled = spexone_weighted_mean(sub, obs_col)
        regional = _regional_dict_from_df(model_seasonal_df, region, model_var_key)
        sampled = {}
        for model in regional:
            da = None
            if data_derived_dict and model in data_derived_dict:
                da = data_derived_dict[model].get(model_var_key)
            s = sample_model_var_at_spexone(da, sub)
            if np.isfinite(s) and np.isfinite(regional[model]):
                sampled[model] = s
        common = [m for m in regional if m in sampled]
        if len(common) < 3:
            print(
                f"  {region}: homogenization skipped (n={len(common)}); "
                f"raw SPEXone {var_label}={obs_sampled:.5f}"
            )
            homogenized[region] = obs_sampled
            details[region] = {"skipped": True, "obs_sampled": obs_sampled, "n": len(common)}
            rows.append({
                "region": region, "variable": var_label,
                "raw_mean": obs_sampled, "a": np.nan, "b": np.nan, "r2": np.nan,
                "homogenized": obs_sampled, "n_models": len(common), "skipped": True,
            })
            continue

        x = np.array([regional[m] for m in common], dtype=float)
        y = np.array([sampled[m] for m in common], dtype=float)
        X = np.column_stack([x, np.ones(len(x))])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
        y_pred = a * x + b
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum((y - y_pred) ** 2) / ss_tot if ss_tot > 0 else np.nan
        homogenized_regional = (obs_sampled - b) / a if a != 0 else np.nan
        homogenized[region] = homogenized_regional
        details[region] = {
            "skipped": False, "models": common, "x": x, "y": y,
            "a": a, "b": b, "r2": r2, "obs_sampled": obs_sampled,
            "homogenized": homogenized_regional,
        }
        rows.append({
            "region": region, "variable": var_label,
            "raw_mean": obs_sampled, "a": a, "b": b, "r2": r2,
            "homogenized": homogenized_regional, "n_models": len(common), "skipped": False,
        })
        print(
            f"  {region}: sampled = {a:.4f}*regional + {b:.4f}  "
            f"(R^2={r2:.3f}, n={len(common)})"
        )
        print(
            f"    raw {var_label}={obs_sampled:.5f} -> homogenized={homogenized_regional:.5f}"
        )

        # Homogenization scatter
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x, y, s=30, alpha=0.7, edgecolors="k", linewidths=0.3)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 40)
        ax.plot(xs, a * xs + b, "r-", lw=1.5, label=f"fit R$^2$={r2:.2f}")
        ax.axvline(homogenized_regional, color="g", ls="--", label="homogenized")
        ax.axhline(obs_sampled, color="b", ls=":", label="SPEXone raw")
        ax.set_xlabel(f"PPE regional {var_label} (1)")
        ax.set_ylabel(f"PPE sampled at SPEXone {var_label} (1)")
        ax.set_title(f"{region} SPEXone homogenization ({var_label})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        save_fig(fig, f"ppe_spexone_homog_{var_label}_{region}.png")

    return homogenized, details, rows


homog_rows = []
if spex_df.empty:
    print("SPEXone empty: skipping homogenization")
    aod_homogenized = aaod_homogenized = ssa_homogenized = {}
else:
    # Need data_derived for sampling; rebuild light refs if cache path skipped it
    if USE_AGG_CACHE and not data_derived:
        print("Warning: USE_AGG_CACHE without in-memory grids; homogenization uses regional means only fallback.")
        data_derived = {}

    if AOD_SPEXONE_HOMOGENIZE:
        aod_homogenized, aod_details, rows = homogenize_spexone_variable(
            model_seasonal_df, spex_df, "AOD_550", "od550aer", "AOD", data_derived
        )
        homog_rows.extend(rows)
    else:
        aod_homogenized = {}

    if SPEXONE_HOMOGENIZE:
        aaod_homogenized, aaod_details, rows = homogenize_spexone_variable(
            model_seasonal_df, spex_df, "AAOD_550", "abs550aer", "AAOD", data_derived
        )
        homog_rows.extend(rows)
    else:
        aaod_homogenized = {}

    if SSA_SPEXONE_HOMOGENIZE:
        ssa_homogenized, ssa_details, rows = homogenize_spexone_variable(
            model_seasonal_df, spex_df, "SSA", "SSA", "SSA", data_derived
        )
        homog_rows.extend(rows)
    else:
        ssa_homogenized = {}

if homog_rows:
    homog_df = pd.DataFrame(homog_rows)
    homog_path = TABLE_DIR / "spexone_ppe_homogenized.csv"
    homog_df.to_csv(homog_path, index=False)
    print(f"Wrote {homog_path}")
    print(homog_df)

print("\nDone.")
print(f"Figures in: {FIGURE_DIR}")
