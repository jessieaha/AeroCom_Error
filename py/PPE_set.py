"""PPE ensemble monthly extract configuration.

Maps AeroCom-style names used by the analysis notebooks onto HAM variable
names in Pace PPE NetCDF streams. Several PPEs are registered in ``PPE_RUNS``;
each has its own read-only source root, file globs, period and burden scale.

Storage:
  * large, regenerable gridded data -> ``/scratch-shared/jzhang1/PPE/<ppe_id>/``
    (``members/`` per-member extracts, ``ensemble/`` group NetCDFs with
    dims ens, time, lat, lon, ``provenance.csv``)
  * small regional tables -> ``Data/PPE_regional/`` in the project (see
    ``py/ppe_regional_aggregates.py``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import regions as _regions

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRON_DRIVE = Path("/gpfs/work3/0/prjs1474")
PACE_PPE_OUTPUT = SRON_DRIVE / "ybhatti" / "Pace_PPE_Output"
PPE_2024_ROOT = PACE_PPE_OUTPUT / "PPE_Experiments"
PPE_2010_ROOT = PACE_PPE_OUTPUT / "2010_PPE" / "Collaboration" / "Processing"
PPE_PATTERN = "PPE_ENS_*"

SCRATCH_ROOT = Path("/scratch-shared/jzhang1/PPE")
SCRATCH_LOGS = SCRATCH_ROOT / "logs"

REGIONAL_DIR = PROJECT_ROOT / "Data" / "PPE_regional"
REGIONAL_MONTHLY = REGIONAL_DIR / "ppe_regional_monthly.parquet"
REGIONAL_SEASONAL = REGIONAL_DIR / "ppe_regional_seasonal.parquet"
REGIONAL_MANIFEST = REGIONAL_DIR / "manifest.json"

ANALYSIS_DIR = PROJECT_ROOT / "Data" / "PPE_analysis"

# Skip member files that are newer than their sources when False.
RENEW = False

# Classic CDI streams (burden, rad_monthly) need netCDF4; netCDF-4 streams also
# open with it. h5netcdf alone cannot read classic files.
XR_ENGINE = "netcdf4"
XR_WRITE_ENGINE = "netcdf4"

# ---------------------------------------------------------------------------
# Source file globs (relative to one PPE_ENS_* directory)
# Exclude restart_* and input symlinks such as megan_emission_factors.nc.
# ---------------------------------------------------------------------------
FILE_GLOBS_2024 = {
    "burden": "PPE_ENS_*_burden.nc",
    "emi": "PPE_ENS_*_emi.nc",
    "hifreq_monthly": "PPE_ENS_*_hifreq_monthly.nc",
    "rad_monthly": "PPE_ENS_*_rad_monthly.nc",
    "vphysc": "PPE_ENS_*_vphysc.nc",
    "echam": "PPE_ENS_*_echam.nc",
    # 3-hourly fallback only; never preferred. Pattern excludes *_hifreq_monthly.nc
    # and standalone ABS/TAU_*_hifreq.nc extracts.
    "hifreq": "PPE_ENS_*[0-9]_hifreq.nc",
}

# 2010 PPE: cdo monmean of the 2010 streams; no hifreq / vphysc / echam.
FILE_GLOBS_2010 = {
    "burden": "PPE_ENS_*_2010_monthly_burden.nc",
    "emi": "PPE_ENS_*_2010_monthly_emi.nc",
    "rad_monthly": "PPE_ENS_*_2010_monthly_rad.nc",
}

# Back-compat alias (2024 layout).
FILE_GLOBS = FILE_GLOBS_2024

EXCLUDE_NAME_SUBSTR = (
    "restart_",
    "megan_emission_factors",
)

# ---------------------------------------------------------------------------
# PPE run registry
# ---------------------------------------------------------------------------
# Aerosol burdens. The "burden" stream (burden_* in *_burden.nc) is labelled
# kg m-2 but is a constant factor below the monthly mean of the 3-hourly
# output (cdo monmean -> *_hifreq_monthly.nc, which also carries burden_*).
# In pd2024 the hifreq_monthly burdens correlate 0.975 with the dust optical
# depth per cell (burden stream: 0.51) and give AeroCom-like global burdens,
# lifetimes and dust MEC without any scaling, so pd2024 reads them directly.
# hifreq_monthly / burden-stream ratio, median over 33 pd2024 members (5-95%
# range within +-4%); used to calibrate runs that only have the burden stream:
AEROSOL_LOADS = ("loadss", "loadso4", "loadbc", "loadoa", "loaddust")
BURDEN_STREAM_FACTOR = {
    "loaddust": 227.4,
    "loadss": 257.4,
    "loadbc": 255.6,
    "loadoa": 250.8,
    "loadso4": 255.8,
}
# burden_scale: float (all aerosol loads) or {load var: factor}; applied to
# LOAD_SCALE_VARS at aggregation. None = not verified (builder refuses to run).
# Gas burdens (burden_SO2 / burden_DMS) are only in the burden stream and give
# realistic global SO2/DMS burdens unscaled. Checked by py/ppe_validate.py
# (Data/PPE_regional/validation_<ppe>.csv).
PPE_RUNS: Dict[str, dict] = {
    "pd2024": dict(
        label="PPE 2024-25",
        root=PPE_2024_ROOT,
        period=("2024-08-01", "2025-07-31"),
        year_months=[(2024, m) for m in range(8, 13)] + [(2025, m) for m in range(1, 8)],
        file_globs=FILE_GLOBS_2024,
        group_preference={v: "hifreq_monthly" for v in AEROSOL_LOADS},
        has_precip=True,
        burden_scale=1.0,
        obs="SPEXone",
        color="#d62728",
        marker="o",
    ),
    "pd2010": dict(
        label="PPE 2010",
        root=PPE_2010_ROOT,
        period=("2010-01-01", "2010-12-31"),
        year_months=[(2010, m) for m in range(1, 13)],
        file_globs=FILE_GLOBS_2010,
        # No precip stream in the 2010 extract; per-member vphysc/echam precip
        # requested from the PPE owner. Once delivered (any root with the same
        # PPE_ENS_* member directories), add e.g.
        #   file_globs["vphysc"] = "PPE_ENS_*_2010*vphysc*.nc"
        #   group_roots = {"vphysc": Path("/path/to/2010/precip/root")}
        # set has_precip=True, then rerun
        #   sbatch ppe_extract_monthly.sbatch --ppe pd2010 --renew
        #   python py/ppe_regional_aggregates.py --ppe pd2010
        has_precip=False,
        # Burden stream only (cdo monmean; correlates 0.997 with dust AOD):
        # calibrated with the pd2024 stream factor.
        burden_scale=dict(BURDEN_STREAM_FACTOR),
        obs="POLDER",
        color="#1f77b4",
        marker="s",
    ),
}
DEFAULT_PPE = "pd2024"


def get_run(ppe_id: str) -> dict:
    if ppe_id not in PPE_RUNS:
        raise KeyError(f"Unknown PPE {ppe_id!r}; known: {sorted(PPE_RUNS)}")
    return PPE_RUNS[ppe_id]


def run_dirs(ppe_id: str) -> Dict[str, Path]:
    """Scratch layout for one PPE run."""
    get_run(ppe_id)
    base = SCRATCH_ROOT / ppe_id
    return {
        "base": base,
        "members": base / "members",
        "ensemble": base / "ensemble",
        "provenance": base / "provenance.csv",
        "logs": SCRATCH_LOGS,
    }


def ensemble_file(ppe_id: str, group: str) -> Path:
    return run_dirs(ppe_id)["ensemble"] / f"{group}_monthly.nc"


def burden_scale_for(ppe_id: str, var: str):
    """Factor applied to aerosol load ``var`` of a run (None if unverified)."""
    scale = get_run(ppe_id)["burden_scale"]
    if scale is None or var not in LOAD_SCALE_VARS:
        return None if scale is None else 1.0
    if isinstance(scale, dict):
        return float(scale.get(var, 1.0))
    return float(scale)


def preferred_group(ppe_id: str, aerocom_name: str) -> str:
    """Source group searched first for a variable (run override or VAR_MAP)."""
    override = get_run(ppe_id).get("group_preference", {})
    return override.get(aerocom_name, VAR_MAP[aerocom_name][1])


# ---------------------------------------------------------------------------
# Variable map: AeroCom name -> (HAM name, preferred source group, units)
# Names in the output NetCDFs keep the AeroCom spelling so notebook formulas
# can be reused with an ``ens`` dimension.
# ---------------------------------------------------------------------------
# Burden / load (kg m-2); see burden_scale / group_preference above. HAM names
# are identical in the burden and hifreq_monthly streams.
BURDEN_VARS = {
    "loadss": ("burden_SS", "burden", "kg m-2"),
    "loadso4": ("burden_SO4", "burden", "kg m-2"),
    "loadbc": ("burden_BC", "burden", "kg m-2"),
    "loadoa": ("burden_OC", "burden", "kg m-2"),
    "loaddust": ("burden_DU", "burden", "kg m-2"),
    "burden_DMS": ("burden_DMS", "burden", "kg m-2"),
    "burden_SO2": ("burden_SO2", "burden", "kg m-2"),
}

# Aerosol loads that burden_scale applies to (see AEROSOL_LOADS above).
LOAD_SCALE_VARS = AEROSOL_LOADS

# Emissions (kg m-2 s-1). AeroCom sea-salt emission is ``emiss``.
EMI_VARS = {
    "emiss": ("emi_SS", "emi", "kg m-2 s-1"),
    "emiso2": ("emi_SO2", "emi", "kg m-2 s-1"),
    "emibc": ("emi_BC", "emi", "kg m-2 s-1"),
    "emioa": ("emi_OC", "emi", "kg m-2 s-1"),
    "emidust": ("emi_DU", "emi", "kg m-2 s-1"),
    "emi_SO4": ("emi_SO4", "emi", "kg m-2 s-1"),
    "emi_DMS": ("emi_DMS", "emi", "kg m-2 s-1"),
    # Time fraction with u* above the dust emission threshold (dust driver).
    "ustar_acrit": ("ustar_acrit", "emi", "%"),
}

# Optics. Prefer hifreq_monthly for column totals; rad_monthly for components.
# The 2010 PPE has no hifreq_monthly, so totals fall back to rad_monthly.
OPTICAL_VARS = {
    "od550aer": ("TAU_2D_550nm", "hifreq_monthly", "1"),
    "abs550aer": ("ABS_2D_550nm", "hifreq_monthly", "1"),
    "ANG_550nm_865nm": ("ANG_550nm_865nm", "hifreq_monthly", "1"),
    "od550ss": ("TAU_COMP_SS_550nm", "rad_monthly", "1"),
    "abs550ss": ("ABS_COMP_SS_550nm", "rad_monthly", "1"),
    "od550so4": ("TAU_COMP_SO4_550nm", "rad_monthly", "1"),
    "od550dust": ("TAU_COMP_DU_550nm", "rad_monthly", "1"),
    "abs550dust": ("ABS_COMP_DU_550nm", "rad_monthly", "1"),
    "od550bc": ("TAU_COMP_BC_550nm", "rad_monthly", "1"),
    "abs550bc": ("ABS_COMP_BC_550nm", "rad_monthly", "1"),
    "od550oa": ("TAU_COMP_OC_550nm", "rad_monthly", "1"),
    "abs550oa": ("ABS_COMP_OC_550nm", "rad_monthly", "1"),
    "od440aer": ("TAU_2D_440nm", "rad_monthly", "1"),
    "abs440aer": ("ABS_2D_440nm", "rad_monthly", "1"),
    "od865aer": ("TAU_2D_865nm", "rad_monthly", "1"),
}

# Precipitation. Store rate + mm day-1; keep echam large-scale/convective.
# vphysc "precip" is accumulated over the output interval (mm); the rate is
# "precip_na" (non-accumulated, kg m-2 s-1).
PRECIP_VARS = {
    "precip_na": ("precip_na", "vphysc", "kg m-2 s-1"),
    "aprl": ("aprl", "echam", "kg m-2 s-1"),
    "aprc": ("aprc", "echam", "kg m-2 s-1"),
}

# Derived precip (built in the extract script, not read from source).
PRECIP_DERIVED = {
    "precip": "mm day-1",  # precip_na * 86400
}

# Deposition / wet / dry: required by the AeroCom notebooks but not posted in
# the PPE. Recorded as missing in provenance; never invent values.
MISSING_VARS = (
    "depss",
    "depso4",
    "depbc",
    "depoa",
    "depdust",
    "wetss",
    "dryss",
    "wetso4",
    "dryso4",
    "wetdu",
)

# Group -> AeroCom names written into that ensemble NetCDF.
GROUP_VARS = {
    "burden": list(BURDEN_VARS),
    "emi": list(EMI_VARS),
    "optical": list(OPTICAL_VARS),
    "precip": list(PRECIP_VARS) + list(PRECIP_DERIVED),
}

# Flat lookup: aerocom_name -> (ham_name, preferred_group, units)
VAR_MAP = {}
VAR_MAP.update(BURDEN_VARS)
VAR_MAP.update(EMI_VARS)
VAR_MAP.update(OPTICAL_VARS)
VAR_MAP.update(PRECIP_VARS)

# Fallback search order when the preferred file is missing the variable.
# Groups absent from a run's file_globs are skipped.
SOURCE_FALLBACK = (
    "burden",
    "emi",
    "hifreq_monthly",
    "rad_monthly",
    "vphysc",
    "echam",
    "hifreq",  # last resort: 3-hourly -> monthly mean
)

# NetCDF encoding defaults for durable ensemble files.
ENS_CHUNK = {"ens": 32, "time": 12, "lat": 96, "lon": 192}

# Control member numeric id for the ``ens`` coordinate.
CONTROL_ENS_ID = -1

# ---------------------------------------------------------------------------
# Back-compat names for scripts written against the single 2024 PPE
# (PPE_precip_vs_GPCP.py, ppe_precip_gpcp.py).
# ---------------------------------------------------------------------------
PPE_ROOT = PPE_2024_ROOT
PPE_TIME: Tuple[str, str] = PPE_RUNS["pd2024"]["period"]
PPE_YEAR_MONTHS: List[Tuple[int, int]] = PPE_RUNS["pd2024"]["year_months"]
AAOD_REGIONS = _regions.ppe_view(["africa", "amazon", "outflow_af"], time_slice=PPE_TIME)
AAOD_REGION_COLORS = _regions.region_colors(["africa", "amazon", "outflow_af"])
