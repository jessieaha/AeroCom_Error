# %% [markdown]
# # AAOD error attribution on Pace PPE (SSA–MAC, then lifetime)
#
# Port of the first-stage relations in ``AAOD_error_attribution.ipynb`` onto
# ``Data/PPE_processed_monthly/``. Observations use SPEXone (not POLDER).
# Emission-based BC+OA lifetimes only (no deposition in the PPE extract).
# No GPCP / constrained E_C / Fig. 3 decomposition in this pass.
#
# Run: `sbatch run_notebook_or_py.sbatch notebooks/AAOD_error_attribution_PPE.py`
# Do not run the full ensemble on the login node.

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
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import linregress

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

import aerocom_data
import notebook_setup as setup
from aerocom_data import _get_dataarray, _align_da_to_ref
import PPE_set as ppe_cfg

print(f"Project root: {project_root}")

# %%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PPE_DIR = project_root / "Data" / "PPE_processed_monthly"
SPEXONE_GLOB = str(project_root / "Data" / "SPEXone_L2_regridded_{year}_{month:02d}_v4_0.parquet")
SPEXONE_YEARS_MONTHS = [(2024, m) for m in range(8, 13)] + [(2025, m) for m in range(1, 8)]
WVL = "550nm"

EXCLUDE_MEMBERS: List[str] = []
DERIVED_VAR_AFTER_AGG = {"lifetime_BC_OA"}
USE_DEPOSITION_FOR_LIFETIME = False  # depbc/depoa absent from PPE

# PPE burden_* labeled kg m-2 but ~1000× low vs AeroCom; same fix as AOD_SS_ERROR_PPE.
BURDEN_MASS_SCALE = 1e3

MAC_X_COL = "1-SSA"  # theory: MAC = A*(1-SSA)
INTERCEPT_0 = True

BROWN_SSA_INTERCEPT = 0.969
BROWN_SSA_SLOPE = -0.779

SAVE_FIGURE = True
FIGURE_DIR = project_root / "figure" / "PPE_AAOD"
TABLE_DIR = project_root / "Data" / "PPE_AAOD"
_SCRATCH_CANDIDATES = [
    Path("/scratch-shared/jzhang1/PPE_AAOD"),
    project_root / "Data" / "PPE_AAOD" / "scratch",
]
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR = None
for _cand in _SCRATCH_CANDIDATES:
    try:
        _cand.mkdir(parents=True, exist_ok=True)
        SCRATCH_DIR = _cand
        break
    except OSError as _e:
        print(f"  Scratch unavailable {_cand}: {_e}")
if SCRATCH_DIR is None:
    raise RuntimeError("Could not create any scratch directory")
print(f"  Using SCRATCH_DIR = {SCRATCH_DIR}")

AXIS_LABELS = {
    "SSA": "SSA (1)",
    "1-SSA": "1 − SSA (1)",
    "MAC": r"MAC (m$^{2}$ g$^{-1}$)",
    "AE": "AE (1)",
    "precip": r"Model precipitation (mm day$^{-1}$)",
    "rBC": "rBC = BC/(BC+OA) (1)",
    "lifetime_BC_OA": r"BC+OA lifetime $\tau$ (day)",
    "inv_lifetime_BC_OA": r"1/$\tau_{\mathrm{BC+OA}}$ (day$^{-1}$)",
    "abs550aer": "AAOD$_{550}$ (1)",
    "od550aer": "AOD$_{550}$ (1)",
    "load_BC_OA": r"BC+OA burden (kg m$^{-2}$)",
    "emi_BC_OA": r"BC+OA emission (kg m$^{-2}$ s$^{-1}$)",
}

SAVE_AGG_CACHE = True
AGG_PARQUET = TABLE_DIR / "aaod_ppe_seasonal_africa_amazon_outflow.parquet"
MONTHLY_PARQUET = TABLE_DIR / "aaod_ppe_monthly_africa_amazon_outflow.parquet"
USE_AGG_CACHE = False

SPEXONE_HOMOGENIZE = True
SSA_SPEXONE_HOMOGENIZE = True
AAOD_SPEXONE_HOMOGENIZE = True

QUALITY_FILTER_PARAMS = {
    "min_aod": 0.01,
    "min_aaod": 1e-5,
    "max_aaod_fraction": 0.95,
    "min_ssa": 0.5,
    "max_ssa": 1.02,
}

# Fire-season boxes from notebook_setup.REGIONS; month-of-year keeps the
# AeroCom seasons while the PPE archive spans Aug 2024 – Jul 2025.
PPE_TIME = ("2024-08-01", "2025-07-31")
REGIONS = {
    "africa": {
        "surface_type": "land",
        "lon_range": (15, 37),
        "lat_range": (-15, 0),
        "time_slice": PPE_TIME,
        "season_months": (6, 7, 8, 9),  # Jun–Sep
        "edge_weighted": False,
    },
    "amazon": {
        "surface_type": "land",
        "lon_range": (287, 317),
        "lat_range": (-17, -3),
        "time_slice": PPE_TIME,
        "season_months": (7, 8, 9, 10),  # Jul–Oct
        "edge_weighted": False,
    },
    "outflow_af": {
        "surface_type": "ocean",
        "lon_range": (350, 15),
        "lat_range": (-15, 0),
        "time_slice": PPE_TIME,
        "season_months": (6, 7, 8, 9),
        "edge_weighted": True,
    },
}
SOURCE_REGIONS = ["africa", "amazon"]
ANALYSIS_REGIONS = SOURCE_REGIONS + ["outflow_af"]
PLOT_REGIONS = list(ANALYSIS_REGIONS)

REGION_COLORS = {
    "africa": "#d62728",
    "amazon": "#2ca02c",
    "outflow_af": "#1f77b4",
}

BC_OA_LOAD = ["loadbc", "loadoa"]
BC_OA_EMI = ["emibc", "emioa"]
LOAD_SCALE_VARS = list(BC_OA_LOAD)

VARIABLES_NEEDED = [
    "od440aer", "od550aer", "od865aer", "abs550aer",
    "precip",
    "loadbc", "loadoa",
    "emibc", "emioa",
]


def ens_label(ens_id: int) -> str:
    eid = int(ens_id)
    return "Control" if eid < 0 else str(eid)


def save_fig(fig, name: str) -> Path:
    path = FIGURE_DIR / name
    if SAVE_FIGURE:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)
    return path


def _axis_label(col: str, override=None) -> str:
    if override:
        return override
    return AXIS_LABELS.get(col, col)


print("--- Active configuration ---")
print(f"  PPE_DIR              = {PPE_DIR}")
print(f"  ANALYSIS_REGIONS     = {ANALYSIS_REGIONS}")
print(f"  DERIVED_VAR_AFTER_AGG= {DERIVED_VAR_AFTER_AGG}")
print(f"  BURDEN_MASS_SCALE    = {BURDEN_MASS_SCALE}")
print(f"  MAC_X_COL            = {MAC_X_COL!r}  INTERCEPT_0={INTERCEPT_0}")
print(f"  USE_DEPOSITION_FOR_LIFETIME = {USE_DEPOSITION_FOR_LIFETIME}")
print(f"  FIGURE_DIR           = {FIGURE_DIR}")
print(f"  TABLE_DIR            = {TABLE_DIR}")
print(f"  SCRATCH_DIR          = {SCRATCH_DIR}")

# Runtime gap tracker (written to potential_issues.md at the end)
ISSUES: List[str] = []

# %%
# -----------------------------------------------------------------------------
# Load PPE monthly ensemble NetCDFs (reuse existing extract)
# -----------------------------------------------------------------------------
GROUP_FILES = {
    "burden": PPE_DIR / "burden_monthly.nc",
    "emi": PPE_DIR / "emi_monthly.nc",
    "optical": PPE_DIR / "optical_monthly.nc",
    "precip": PPE_DIR / "precip_monthly.nc",
}


def load_ppe_as_model_dict(ppe_files: Dict[str, Path]) -> Tuple[Dict[str, Dict[str, xr.Dataset]], List[str]]:
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
            ens_ids, labels = (list(x) for x in zip(*keep)) if keep else ([], [])

        data: Dict[str, Dict[str, xr.Dataset]] = {lab: {} for lab in labels}
        for lab, eid in zip(labels, ens_ids):
            for group_ds in opened.values():
                sub = group_ds.sel(ens=eid)
                for var in sub.data_vars:
                    if var == "experiment":
                        continue
                    da = sub[var]
                    if "ens" in da.dims:
                        da = da.squeeze("ens", drop=True)
                    elif "ens" in da.coords:
                        da = da.drop_vars("ens")
                    if "experiment" in da.coords:
                        da = da.drop_vars("experiment")
                    data[lab][var] = da.to_dataset(name=var)
        return data, labels
    finally:
        for ds in opened.values():
            ds.close()


if USE_AGG_CACHE and AGG_PARQUET.is_file():
    print("USE_AGG_CACHE=True: skipping NetCDF load")
    raw_data: Dict = {}
    models: List[str] = []
else:
    print(f"Loading PPE monthly ensembles from {PPE_DIR} ...")
    raw_data, models = load_ppe_as_model_dict(GROUP_FILES)
    print(f"Ensemble members: {len(models)}")
    if models:
        print(f"  first={models[0]}  last={models[-1]}")

# %%
# -----------------------------------------------------------------------------
# Normalize / BC+OA sums
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


if USE_AGG_CACHE and AGG_PARQUET.is_file():
    print("USE_AGG_CACHE=True: skipping normalize/sums")
    data: Dict = {}
    target_models: List[str] = []
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

        if BURDEN_MASS_SCALE != 1.0:
            for key in LOAD_SCALE_VARS:
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

        normalized["load_BC_OA"] = setup.sum_datasets(
            normalized, BC_OA_LOAD, "load_BC_OA", require_all=True
        )
        normalized["emi_BC_OA"] = setup.sum_datasets(
            normalized, BC_OA_EMI, "emi_BC_OA", require_all=True
        )
        data[m] = normalized

    target_models = [
        m for m in models
        if data[m].get("od550aer") is not None
        and data[m].get("abs550aer") is not None
        and data[m].get("load_BC_OA") is not None
        and data[m].get("emi_BC_OA") is not None
    ]
    print(f"Target members with od550aer+abs550aer+load_BC_OA+emi_BC_OA: {len(target_models)}")
    if target_models and BURDEN_MASS_SCALE != 1.0:
        print(f"Applied BURDEN_MASS_SCALE={BURDEN_MASS_SCALE} to {LOAD_SCALE_VARS}")

raw_data = None

# %%
# -----------------------------------------------------------------------------
# Pre-aggregation derived variables (MAC, SSA, AE, rBC)
# -----------------------------------------------------------------------------
if USE_AGG_CACHE and AGG_PARQUET.is_file():
    print("USE_AGG_CACHE=True: skipping derived vars")
    data_derived: Dict = {}
else:
    derived_vars_pre = [
        dv for dv in ["MAC", "SSA", "AE", "rBC"]
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

        load_bc = _get_dataarray(model_data.get("loadbc"), "loadbc")
        load_oa = _get_dataarray(model_data.get("loadoa"), "loadoa")
        load_bcoa = _get_dataarray(model_data.get("load_BC_OA"), "load_BC_OA")

        for dv in derived_vars_pre:
            try:
                if dv == "SSA":
                    if ssa is None:
                        raise KeyError("Missing abs550aer for SSA.")
                    derived[m]["SSA"] = ssa
                elif dv == "MAC":
                    if aaod is None or load_bcoa is None:
                        raise KeyError("Missing abs550aer or load_BC_OA for MAC.")
                    load_a = _align_da_to_ref(load_bcoa, aod550, model_hint=f"{m}/load_BC_OA")
                    # MAC = AAOD / (load_BC_OA * 1e3)  → m² g⁻¹
                    derived[m]["MAC"] = aaod / (load_a * 1e3)
                elif dv == "rBC":
                    if load_bc is None or load_oa is None:
                        raise KeyError("Missing loadbc/loadoa for rBC.")
                    bc_a = _align_da_to_ref(load_bc, aod550, model_hint=f"{m}/loadbc")
                    oa_a = _align_da_to_ref(load_oa, aod550, model_hint=f"{m}/loadoa")
                    denom = bc_a + oa_a
                    derived[m]["rBC"] = (bc_a / denom).where(denom > 0)
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
# Regional aggregation (monthly over PPE year; seasonal via season_months)
# -----------------------------------------------------------------------------
def _as_da_dict(model_dict, var_name):
    out = {}
    for model, md in model_dict.items():
        if var_name not in md or md[var_name] is None:
            continue
        val = md[var_name]
        if isinstance(val, xr.Dataset):
            val = _get_dataarray(val, var_name)
        out[model] = {var_name: val}
    return out


def _month_number(val) -> Optional[int]:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return None
    if isinstance(val, pd.Period):
        return int(val.month)
    if hasattr(val, "month"):
        try:
            return int(val.month)
        except Exception:
            pass
    try:
        ts = pd.Timestamp(val)
        if pd.notna(ts):
            return int(ts.month)
    except Exception:
        pass
    try:
        return int(val)
    except Exception:
        return None


def seasonal_from_monthly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Mean over season_months per region/model (handles year-boundary seasons)."""
    if monthly_df.empty:
        return monthly_df.copy()
    work = monthly_df.copy()
    work["_mon"] = work["month"].map(_month_number)
    rows = []
    value_cols = [
        c for c in work.columns
        if c not in ("region", "model", "month", "_mon")
    ]
    for region in ANALYSIS_REGIONS:
        months = set(REGIONS[region]["season_months"])
        sub = work.loc[(work["region"] == region) & (work["_mon"].isin(months))]
        if sub.empty:
            continue
        for model, g in sub.groupby("model"):
            row = {"region": region, "model": model}
            for c in value_cols:
                row[c] = pd.to_numeric(g[c], errors="coerce").mean()
            rows.append(row)
    return pd.DataFrame(rows)


if USE_AGG_CACHE and AGG_PARQUET.is_file():
    print("USE_AGG_CACHE=True: skipping regional aggregation")
    model_monthly = {}
    model_seasonal = {}
    masks = {}
    model_monthly_df = (
        pd.read_parquet(MONTHLY_PARQUET) if MONTHLY_PARQUET.is_file() else pd.DataFrame()
    )
    model_seasonal_df = pd.read_parquet(AGG_PARQUET)
else:
    template = data[target_models[0]]["od550aer"].isel(time=0)
    masks = setup.create_analysis_masks(
        template, ANALYSIS_REGIONS, region=REGIONS, surface_type=None
    )
    print("Regions created:", list(masks.keys()))

    variables_pre_aggregated = [
        "MAC", "SSA", "AE", "rBC",
        "precip", "load_BC_OA", "emi_BC_OA",
        "od550aer", "abs550aer", "loadbc", "loadoa", "emibc", "emioa",
    ]
    variables_to_aggregate = [
        v for v in variables_pre_aggregated if v not in DERIVED_VAR_AFTER_AGG
    ]

    model_monthly = {}
    for region in ANALYSIS_REGIONS:
        model_monthly[region] = {}
        for var in variables_to_aggregate:
            model_monthly[region][var] = setup.aggregate_region(
                _as_da_dict(data_derived, var),
                var,
                region,
                masks,
                region=REGIONS,
                return_time_series=True,
                skipna=True,
            )

    # Placeholder seasonal dict (rebuilt from monthly after lifetime)
    model_seasonal = {r: {} for r in ANALYSIS_REGIONS}

    model_monthly_df, _ = setup.convert_aggregation_to_dataframes(
        model_monthly, model_seasonal, month_col="month"
    )
    model_seasonal_df = seasonal_from_monthly(model_monthly_df)

    # Post-aggregation emission lifetime on seasonal means
    model_seasonal_df = model_seasonal_df.copy()
    model_seasonal_df["lifetime_BC_OA"] = np.nan
    for region in ANALYSIS_REGIONS:
        mask = model_seasonal_df["region"] == region
        sub = model_seasonal_df.loc[mask]
        load = {
            str(m): float(v)
            for m, v in zip(sub["model"], sub["load_BC_OA"])
            if np.isfinite(v)
        }
        flux = {
            str(m): float(v)
            for m, v in zip(sub["model"], sub["emi_BC_OA"])
            if np.isfinite(v)
        }
        lt = setup._lifetime_from_load_flux(load, flux, "lifetime_BC_OA", region)
        model_seasonal[region]["lifetime_BC_OA"] = lt
        model_seasonal_df.loc[mask, "lifetime_BC_OA"] = (
            model_seasonal_df.loc[mask, "model"].astype(str).map(lt).to_numpy()
        )
        print(f"  seasonal/{region}: lifetime_BC_OA={len(lt)}")

    # Monthly lifetimes for completeness
    if "load_BC_OA" in model_monthly_df.columns and "emi_BC_OA" in model_monthly_df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            model_monthly_df["lifetime_BC_OA"] = model_monthly_df["load_BC_OA"] / (
                model_monthly_df["emi_BC_OA"] * 86400.0
            )

    if SAVE_AGG_CACHE:
        scratch_monthly = SCRATCH_DIR / MONTHLY_PARQUET.name
        scratch_seasonal = SCRATCH_DIR / AGG_PARQUET.name
        model_monthly_df.to_parquet(scratch_monthly)
        model_seasonal_df.to_parquet(scratch_seasonal)
        model_monthly_df.to_parquet(MONTHLY_PARQUET)
        model_seasonal_df.to_parquet(AGG_PARQUET)
        print(f"Wrote scratch {scratch_monthly}")
        print(f"Wrote scratch {scratch_seasonal}")
        print(f"Wrote {MONTHLY_PARQUET}")
        print(f"Wrote {AGG_PARQUET}")

print("\nSeasonal availability:")
availability_columns = [
    var for var in ["MAC", "SSA", "AE", "rBC", "lifetime_BC_OA", "precip", "abs550aer"]
    if var in model_seasonal_df.columns
]
if not model_seasonal_df.empty and availability_columns:
    print(model_seasonal_df.groupby("region")[availability_columns].count())
print(model_seasonal_df.head())

# %%
# -----------------------------------------------------------------------------
# Range check
# -----------------------------------------------------------------------------
def print_derived_range_check(df: pd.DataFrame) -> None:
    print("\n--- Derived-variable range check (seasonal) ---")
    checks = [
        ("MAC", (0.0, 15.0)),
        ("lifetime_BC_OA", None),
    ]
    for col, expected in checks:
        if col not in df.columns:
            print(f"  {col}: missing")
            ISSUES.append(f"Range check: column {col} missing from seasonal table.")
            continue
        s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            print(f"  {col}: no finite values")
            ISSUES.append(f"Range check: no finite {col} values.")
            continue
        q = s.quantile([0.05, 0.5, 0.95])
        print(
            f"  {col}: median={q.loc[0.5]:.4g}  "
            f"q05={q.loc[0.05]:.4g}  q95={q.loc[0.95]:.4g}  n={len(s)}"
        )
        if col.startswith("lifetime"):
            inv = 1.0 / s[s > 0]
            qi = inv.quantile([0.05, 0.5, 0.95])
            print(
                f"    1/{col}: median={qi.loc[0.5]:.4g}  "
                f"q05={qi.loc[0.05]:.4g}  q95={qi.loc[0.95]:.4g}  "
                f"(expect ~0.1–1 day^-1)"
            )
            if qi.loc[0.5] > 10 or qi.loc[0.5] < 0.01:
                ISSUES.append(
                    f"1/{col} median={qi.loc[0.5]:.4g} outside expected ~0.1–1; "
                    "check BURDEN_MASS_SCALE / emission units."
                )
        elif expected is not None:
            lo, hi = expected
            inside = float(((s >= lo) & (s <= hi)).mean() * 100)
            print(f"    fraction in [{lo}, {hi}]: {inside:.1f}%")
            if inside < 50:
                ISSUES.append(
                    f"{col}: only {inside:.1f}% of values in [{lo}, {hi}] "
                    "(possible unit / burden-scale issue)."
                )


if not model_seasonal_df.empty:
    print_derived_range_check(model_seasonal_df)

# %%
# -----------------------------------------------------------------------------
# Plot helpers (no member labels / legends)
# -----------------------------------------------------------------------------
def plot_mac_vs_x(
    df: pd.DataFrame,
    x_col: str = "SSA",
    force_intercept_zero: bool = False,
    regions=None,
    filename: str = "ppe_MAC_vs_x.png",
    title: str = None,
):
    regions = regions or PLOT_REGIONS
    if df.empty or "MAC" not in df.columns or "SSA" not in df.columns:
        print(f"Skip {filename}: missing MAC/SSA")
        return
    fig, axes = plt.subplots(1, len(regions), figsize=(4.5 * len(regions), 4.2), squeeze=False)
    axes = axes.ravel()
    for ax, region in zip(axes, regions):
        sub = df.loc[df["region"] == region, ["model", "SSA", "MAC"]].copy()
        if x_col == "1-SSA":
            sub[x_col] = 1.0 - sub["SSA"]
        else:
            sub[x_col] = sub["SSA"]
        sub = sub[["model", x_col, "MAC"]].replace([np.inf, -np.inf], np.nan).dropna()
        color = REGION_COLORS.get(region, "#333333")
        if sub.empty:
            ax.set_title(f"{region}: no data")
            continue
        x = sub[x_col].to_numpy(dtype=float)
        y = sub["MAC"].to_numpy(dtype=float)
        ax.scatter(x, y, s=28, alpha=0.65, color=color, edgecolors="k", linewidths=0.3)
        if len(sub) >= 2:
            if force_intercept_zero:
                slope = float(np.nansum(x * y) / np.nansum(x * x)) if np.nansum(x * x) else np.nan
                intercept = 0.0
                r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
            else:
                lr = linregress(x, y)
                slope, intercept, r = lr.slope, lr.intercept, lr.rvalue
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 40)
            ax.plot(xs, slope * xs + intercept, color="k", lw=1.4)
            ax.set_title(f"{region}: y={slope:.2f}x+{intercept:.2f} r={r:.2f} n={len(sub)}")
        else:
            ax.set_title(f"{region}: n={len(sub)}")
        ax.set_xlabel(_axis_label(x_col))
        ax.set_ylabel(_axis_label("MAC"))
        ax.grid(True, alpha=0.3)
    fig.suptitle(title or f"PPE: MAC vs {x_col}")
    fig.tight_layout()
    save_fig(fig, filename)


def plot_ssa_vs_rbc(df: pd.DataFrame, regions=None, filename="ppe_SSA_vs_rBC.png"):
    regions = regions or PLOT_REGIONS
    if "rBC" not in df.columns or "SSA" not in df.columns:
        print("Skip SSA vs rBC: missing columns")
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for idx, region in enumerate(regions):
        sub = df.loc[df["region"] == region, ["rBC", "SSA"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if sub.empty:
            continue
        ax.scatter(
            sub["rBC"], sub["SSA"], s=28, alpha=0.65,
            color=REGION_COLORS.get(region, f"C{idx}"),
            edgecolors="k", linewidths=0.3, label=region,
        )
    # Brown et al. (2021) line (drawn, not fit)
    xs = np.linspace(0, 0.6, 50)
    ax.plot(
        xs, BROWN_SSA_INTERCEPT + BROWN_SSA_SLOPE * xs,
        "k--", lw=1.5, label="Brown et al. 2021",
    )
    ax.set_xlabel(_axis_label("rBC"))
    ax.set_ylabel(_axis_label("SSA"))
    ax.set_title("PPE: SSA vs rBC (Brown line)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    save_fig(fig, filename)


def plot_tau_vs_x(
    df, lifetime_col="lifetime_BC_OA", x_var="precip",
    regions=None, filename=None, title=None,
):
    regions = regions or PLOT_REGIONS
    if lifetime_col not in df.columns or x_var not in df.columns:
        print(f"Skip plot_tau_vs_x: missing {lifetime_col} or {x_var}")
        return
    fig, axes = plt.subplots(1, len(regions), figsize=(4.8 * len(regions), 4), squeeze=False)
    axes = axes.ravel()
    for ax, region in zip(axes, regions):
        sub = df.loc[
            df["region"] == region, ["model", x_var, lifetime_col]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        sub = sub.loc[sub[lifetime_col] > 0]
        color = REGION_COLORS.get(region, "#333333")
        if sub.empty:
            ax.set_title(f"{region}: no data")
            continue
        y = 1.0 / sub[lifetime_col].to_numpy(dtype=float)
        x = sub[x_var].to_numpy(dtype=float)
        ax.scatter(x, y, s=28, alpha=0.7, color=color, edgecolors="k", linewidths=0.3)
        if len(sub) >= 2:
            lr = linregress(x, y)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 40)
            ax.plot(xs, lr.slope * xs + lr.intercept, "k-", lw=1.4)
            ax.set_title(f"{region}: r={lr.rvalue:.2f} n={len(sub)}")
        else:
            ax.set_title(f"{region}: n={len(sub)}")
        ax.set_xlabel(_axis_label(x_var))
        ax.set_ylabel(_axis_label("inv_lifetime_BC_OA"))
        ax.grid(True, alpha=0.3)
    fig.suptitle(title or f"PPE: 1/{lifetime_col} vs {x_var}")
    fig.tight_layout()
    save_fig(fig, filename or f"ppe_inv_{lifetime_col}_vs_{x_var}.png")


def plot_lifetime_ae_precip_3d(
    df,
    lifetime_col="lifetime_BC_OA",
    regions=None,
    angle=35,
    elev=15,
    inverse=True,
    filename=None,
    title=None,
):
    regions = regions or PLOT_REGIONS
    need = ["region", "model", "AE", "precip", lifetime_col]
    if any(c not in df.columns for c in need):
        print(f"Skip 3D plot: missing columns for {lifetime_col}")
        return None
    plot_df = (
        df.loc[df["region"].isin(regions), need]
        .rename(columns={"precip": "model_precip", lifetime_col: "lifetime"})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    plot_df = plot_df.loc[plot_df["lifetime"] > 0]
    plot_df["z"] = (1.0 / plot_df["lifetime"]) if inverse else plot_df["lifetime"]
    z_label = _axis_label("inv_lifetime_BC_OA") if inverse else _axis_label(lifetime_col)
    print(f"Valid AE / precip / {z_label} rows: {len(plot_df)}")
    if plot_df.empty:
        return plot_df

    ncols = 2
    nrows = int(np.ceil(len(regions) / ncols))
    fig = plt.figure(figsize=(12, 5.0 * nrows))
    for idx, region in enumerate(regions):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        region_color = REGION_COLORS.get(region, "#333333")
        sub = plot_df[plot_df["region"] == region]
        if len(sub) < 3:
            ax.set_title(f"{region}: insufficient data (n={len(sub)})")
            continue
        x = sub["model_precip"].to_numpy(dtype=float)
        y = sub["AE"].to_numpy(dtype=float)
        z = sub["z"].to_numpy(dtype=float)
        ax.scatter(x, y, z, c=region_color, s=28, edgecolor="black",
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
    fig.suptitle(title or f"PPE: {z_label} vs precip and AE (3D OLS)", y=1.02, fontsize=13)
    fig.tight_layout()
    save_fig(fig, filename or "ppe_3d_inv_lifetime_bcoa_vs_precip_AE.png")
    return plot_df


def plot_region_map(filename="ppe_aaod_region_map.png"):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("PPE AAOD analysis regions")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#e8f4fc")
    for name, cfg in REGIONS.items():
        lon0, lon1 = cfg["lon_range"]
        lat0, lat1 = cfg["lat_range"]
        color = REGION_COLORS.get(name, "#333333")
        if lon0 <= lon1:
            ax.add_patch(Rectangle(
                (lon0, lat0), lon1 - lon0, lat1 - lat0,
                fill=False, edgecolor=color, lw=2, label=name,
            ))
        else:
            # wraps 0° (outflow_af)
            ax.add_patch(Rectangle(
                (lon0, lat0), 360 - lon0, lat1 - lat0,
                fill=False, edgecolor=color, lw=2, label=name,
            ))
            ax.add_patch(Rectangle(
                (0, lat0), lon1, lat1 - lat0,
                fill=False, edgecolor=color, lw=2,
            ))
    ax.legend(fontsize=8, loc="lower left")
    save_fig(fig, filename)


# %%
# -----------------------------------------------------------------------------
# Model-side figures (SSA–MAC, then lifetime)
# -----------------------------------------------------------------------------
if not model_seasonal_df.empty:
    plot_region_map()

    plot_mac_vs_x(
        model_seasonal_df, x_col="SSA", force_intercept_zero=False,
        filename="ppe_MAC_vs_SSA.png",
        title="PPE: MAC vs SSA",
    )
    plot_mac_vs_x(
        model_seasonal_df, x_col="1-SSA", force_intercept_zero=INTERCEPT_0,
        filename="ppe_MAC_vs_1SSA.png",
        title="PPE: MAC vs 1−SSA (intercept forced through origin)"
        if INTERCEPT_0 else "PPE: MAC vs 1−SSA",
    )
    plot_ssa_vs_rbc(model_seasonal_df)

    plot_tau_vs_x(
        model_seasonal_df, "lifetime_BC_OA", "precip",
        filename="ppe_inv_lifetime_bcoa_vs_precip.png",
        title=r"PPE BC+OA: 1/$\tau$ vs model precipitation",
    )
    plot_tau_vs_x(
        model_seasonal_df, "lifetime_BC_OA", "AE",
        filename="ppe_inv_lifetime_bcoa_vs_AE.png",
        title=r"PPE BC+OA: 1/$\tau$ vs AE",
    )
    plot_lifetime_ae_precip_3d(
        model_seasonal_df,
        lifetime_col="lifetime_BC_OA",
        angle=35, elev=15, inverse=True,
        filename="ppe_3d_inv_lifetime_bcoa_vs_precip_AE.png",
        title=r"PPE BC+OA: 1/$\tau$ vs precip and AE",
    )

# %%
# -----------------------------------------------------------------------------
# SPEXone load + regional means
# -----------------------------------------------------------------------------
def load_spexone() -> pd.DataFrame:
    files = []
    for year, month in SPEXONE_YEARS_MONTHS:
        path = Path(SPEXONE_GLOB.format(year=year, month=month))
        if path.is_file():
            files.append(path)
        else:
            print(f"  SPEXone missing: {path.name}")
            ISSUES.append(f"Missing SPEXone file: {path.name}")
    if not files:
        raise FileNotFoundError("No SPEXone monthly parquet files found for PPE overlap.")
    print(f"Loading {len(files)} SPEXone monthly files ...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["month"] = df["date"].dt.to_period("M")
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
    print(
        f"SPEXone rows: {len(df):,}  valid AOD+AAOD+SSA: "
        f"{df[['AOD_550','AAOD_550','SSA']].notna().all(axis=1).sum():,}"
    )
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
    months = set(cfg["season_months"])
    work = work[work["date"].dt.month.isin(months)].copy()
    # Restrict to PPE archive window
    t0, t1 = PPE_TIME
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
    ISSUES.append(f"SPEXone load failed: {e}")
    spex_df = pd.DataFrame()

spex_obs = {}
spex_coverage_rows = []
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
            f"AAOD={spex_obs[region]['AAOD_550']:.5f}  "
            f"SSA={spex_obs[region]['SSA']:.4f}"
        )
        if spex_obs[region]["n"] == 0:
            ISSUES.append(f"SPEXone: zero rows in {region} after season/quality filter.")
        for mon in REGIONS[region]["season_months"]:
            n_mon = int((sub["date"].dt.month == mon).sum()) if not sub.empty else 0
            spex_coverage_rows.append({"region": region, "month": mon, "n": n_mon})
            if n_mon == 0:
                ISSUES.append(
                    f"SPEXone coverage: {region} month={mon} has 0 rows "
                    "(fire season split across PPE year boundary)."
                )

# %%
# -----------------------------------------------------------------------------
# SPEXone homogenization (AAOD, SSA)
# -----------------------------------------------------------------------------
def sample_model_var_at_spexone(model_da, spex_sub: pd.DataFrame) -> float:
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
    rows = []
    print(f"\n--- SPEXone homogenization ({var_label}) ---")
    for region in ANALYSIS_REGIONS:
        sub = spexone_in_region(spex_raw, region)
        if sub.empty:
            print(f"  {region}: no SPEXone samples")
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

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x, y, s=20, alpha=0.6, edgecolors="k", linewidths=0.2)
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

    return homogenized, rows


homog_rows = []
if spex_df.empty:
    print("SPEXone empty: skipping homogenization")
    ISSUES.append("SPEXone empty: homogenization skipped.")
else:
    if USE_AGG_CACHE and not data_derived:
        print("Warning: USE_AGG_CACHE without grids; homogenization may skip sampling.")
        data_derived = {}
        ISSUES.append("Homogenization ran without in-memory grids (cache path).")

    if AAOD_SPEXONE_HOMOGENIZE:
        _, rows = homogenize_spexone_variable(
            model_seasonal_df, spex_df, "AAOD_550", "abs550aer", "AAOD", data_derived
        )
        homog_rows.extend(rows)
    if SSA_SPEXONE_HOMOGENIZE:
        _, rows = homogenize_spexone_variable(
            model_seasonal_df, spex_df, "SSA", "SSA", "SSA", data_derived
        )
        homog_rows.extend(rows)

if homog_rows:
    homog_df = pd.DataFrame(homog_rows)
    homog_scratch = SCRATCH_DIR / "spexone_ppe_aaod_homogenized.csv"
    homog_path = TABLE_DIR / "spexone_ppe_aaod_homogenized.csv"
    homog_df.to_csv(homog_scratch, index=False)
    homog_df.to_csv(homog_path, index=False)
    print(f"Wrote {homog_path}")
    print(homog_df)

if spex_coverage_rows:
    cov = pd.DataFrame(spex_coverage_rows)
    cov.to_csv(TABLE_DIR / "spexone_region_month_coverage.csv", index=False)

# %%
# -----------------------------------------------------------------------------
# Potential issues / gap list
# -----------------------------------------------------------------------------
# Static structural gaps
ISSUES.extend([
    "Missing PPE variables (py/PPE_set.MISSING_VARS): "
    + ", ".join(ppe_cfg.MISSING_VARS)
    + ". Deposition-based lifetime_BC_OA (notebook default USE_DEPOSITION_FOR_LIFETIME=True) "
    "is unavailable; this script uses emission lifetime only.",
    "Missing observed precipitation (GPCP not downloaded). Lifetime plots use model precip only; "
    "no Fig. S3-style observed-precip overlay or constrained τ from precip.",
    "Missing POLDER-GRASP: SPEXone (Aug 2024–Jul 2025) substitutes for AAOD/SSA homogenization.",
    "Missing OMI HCHO: Fig. 4-style AOD–HCHO scatter not implemented.",
    "Fig. 3 error decomposition / constrained E_C / African outflow meta-model deferred "
    "(need observed precip and/or deposition for full Zhong pipeline).",
    f"Burden unit risk: PPE burden_* streams labeled kg m-2 but appear ~1000× low; "
    f"BURDEN_MASS_SCALE={BURDEN_MASS_SCALE} applied to loadbc/loadoa before MAC and lifetime.",
    "MAC formula uses ×1e3 assuming load in kg m-2 → g m-2 (m² g⁻¹). "
    "Lifetime uses τ = load / (emi * 86400) with emi in kg m-2 s-1.",
    "SPEXone lon_bin is typically −180…180; converted to lon360 for PPE 0…360 grids.",
    "NetCDF experiment strings are truncated; members keyed by ens id (Control / integer).",
    "Fire-season months for africa/amazon/outflow_af span the PPE year boundary "
    "(Aug 2024–Jul 2025): seasonal means use month-of-year filtering "
    f"(africa/outflow={REGIONS['africa']['season_months']}, "
    f"amazon={REGIONS['amazon']['season_months']}).",
])

# Deduplicate while preserving order
seen = set()
unique_issues = []
for item in ISSUES:
    if item not in seen:
        seen.add(item)
        unique_issues.append(item)

issues_path = TABLE_DIR / "potential_issues.md"
with open(issues_path, "w", encoding="utf-8") as f:
    f.write("# PPE AAOD attribution — potential issues\n\n")
    f.write(
        "Generated by `notebooks/AAOD_error_attribution_PPE.py`. "
        "Scope: SSA–MAC and emission BC+OA lifetime only.\n\n"
    )
    for i, item in enumerate(unique_issues, 1):
        f.write(f"{i}. {item}\n")
print(f"Wrote {issues_path} ({len(unique_issues)} items)")

print("\nDone.")
print(f"Figures in: {FIGURE_DIR}")
print(f"Tables in:  {TABLE_DIR}")
print(f"Scratch:    {SCRATCH_DIR}")
