"""PPE ensemble monthly extract configuration.

Maps AeroCom-style names used by notebooks/AOD_SS_ERROR.ipynb onto HAM
variable names in Pace PPE NetCDF streams. Source directories under
``PPE_ROOT`` are read-only; scratch holds per-member temps; durable
ensemble NetCDFs land in ``OUTPUT_DIR``.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRON_DRIVE = Path("/gpfs/work3/0/prjs1474")
PPE_ROOT = SRON_DRIVE / "ybhatti" / "Pace_PPE_Output" / "PPE_Experiments"
PPE_PATTERN = "PPE_ENS_*"

SCRATCH_DIR = Path("/scratch-shared/jzhang1/PPE_monthly")
SCRATCH_MEMBERS = SCRATCH_DIR / "members"
SCRATCH_LOGS = SCRATCH_DIR / "logs"

OUTPUT_DIR = PROJECT_ROOT / "Data" / "PPE_processed_monthly"
PROVENANCE_CSV = OUTPUT_DIR / "provenance.csv"

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
FILE_GLOBS = {
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

EXCLUDE_NAME_SUBSTR = (
    "restart_",
    "megan_emission_factors",
)

# ---------------------------------------------------------------------------
# Variable map: AeroCom name -> (HAM name, preferred source group, units)
# Names in the output NetCDFs keep the AeroCom spelling so AOD_SS_ERROR
# formulas can be reused with an ``ens`` dimension.
# ---------------------------------------------------------------------------
# Burden / load (kg m-2)
# NOTE: Pace PPE burden_* values are ~1000× below AeroCom kg m-2 column burdens
# (likely a tracer unit inconsistency in the source stream). AOD_SS_ERROR_PPE.py
# applies BURDEN_MASS_SCALE=1e3 at analysis time to recover MEC ~ O(1) m2 g-1
# and 1/lifetime ~ 0.1–1 d-1. Revisit here if the source files are corrected.
BURDEN_VARS = {
    "loadss": ("burden_SS", "burden", "kg m-2"),
    "loadso4": ("burden_SO4", "burden", "kg m-2"),
    "loadbc": ("burden_BC", "burden", "kg m-2"),
    "loadoa": ("burden_OC", "burden", "kg m-2"),
    "loaddust": ("burden_DU", "burden", "kg m-2"),
    "burden_DMS": ("burden_DMS", "burden", "kg m-2"),
    "burden_SO2": ("burden_SO2", "burden", "kg m-2"),
}

# Emissions (kg m-2 s-1). AeroCom sea-salt emission is ``emiss``.
EMI_VARS = {
    "emiss": ("emi_SS", "emi", "kg m-2 s-1"),
    "emiso2": ("emi_SO2", "emi", "kg m-2 s-1"),
    "emibc": ("emi_BC", "emi", "kg m-2 s-1"),
    "emioa": ("emi_OC", "emi", "kg m-2 s-1"),
    "emidust": ("emi_DU", "emi", "kg m-2 s-1"),
    "emi_SO4": ("emi_SO4", "emi", "kg m-2 s-1"),
    "emi_DMS": ("emi_DMS", "emi", "kg m-2 s-1"),
}

# Optics. Prefer hifreq_monthly for column totals; rad_monthly for components.
OPTICAL_VARS = {
    "od550aer": ("TAU_2D_550nm", "hifreq_monthly", "1"),
    "abs550aer": ("ABS_2D_550nm", "hifreq_monthly", "1"),
    "ANG_550nm_865nm": ("ANG_550nm_865nm", "hifreq_monthly", "1"),
    "od550ss": ("TAU_COMP_SS_550nm", "rad_monthly", "1"),
    "od550so4": ("TAU_COMP_SO4_550nm", "rad_monthly", "1"),
    "od440aer": ("TAU_2D_440nm", "rad_monthly", "1"),
    "od865aer": ("TAU_2D_865nm", "rad_monthly", "1"),
}

# Precipitation. Store rate + mm day-1; keep echam large-scale/convective.
PRECIP_VARS = {
    "precip_na": ("precip", "vphysc", "kg m-2 s-1"),
    "aprl": ("aprl", "echam", "kg m-2 s-1"),
    "aprc": ("aprc", "echam", "kg m-2 s-1"),
}

# Derived precip (built in the extract script, not read from source).
PRECIP_DERIVED = {
    "precip": "mm day-1",  # precip_na * 86400
}

# Deposition / wet / dry: required by AOD_SS_ERROR but not posted in PPE.
# Recorded as missing in provenance; never invent values.
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
