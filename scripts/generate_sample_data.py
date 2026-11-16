"""Generate synthetic AeroCom monthly pickle files for local development/testing.

The real AeroCom data is downloaded from a credentialed met.no HPC server
(see extract_data_ap3.sh) and is NOT included in the repo (Data/ is gitignored).
This script produces small, physically-plausible synthetic monthly fields with
the exact structure the analysis notebooks expect, so the pipeline can be run
end-to-end without access to the HPC data.

Pickle structure produced (matches aerocom_data.get_data output):
    {model_name: {var_name: xarray.DataArray(time, lat, lon)}}

Files written to Data/var_files/original/monthly/:
    od550.pickle      -> {'AOD550'}
    abs550.pickle     -> {'AAOD550'}
    load_total.pickle -> {'load_total'}
    load_BC_OA.pickle -> {'load_BC_OA'}
    emi_total.pickle  -> {'emi_total'}
    emi_BC_OA.pickle  -> {'emi_BC_OA'}

Usage:
    python scripts/generate_sample_data.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# A representative subset of AeroCom Phase III models (names match aerocom_data.py).
MODELS = [
    "CAM5.3-Oslo_AP3-CTRL2016-PD",
    "ECHAM6-HAM2_AP3-CTRL2016-PD",
    "ECHAM6-SALSA_CTRL2016-PD",
    "ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD",
    "TM5_AP3-CTRL2016",
    "GFDL-AM4-met2010_AP3-CTRL",
    "GEOS-i33p2-met2010_AP3-CTRL",
    "INCA_AP3-CTRL",
    "MIROC-SPRINTARS_AP3-CTRL",
    "TM5-met2010_AP3-CTRL2019",
]

# Coarse global grid (keeps pickles small but exercises the full weighting logic).
LAT = np.arange(-88.0, 89.0, 4.0)
LON = np.arange(0.0, 360.0, 5.0)
TIME = pd.date_range("2010-01-01", periods=12, freq="MS")


def _field(rng, base, spatial_amp, seasonal_amp, floor=0.0):
    """Build a (time, lat, lon) field with a smooth spatial + seasonal pattern."""
    lat2d, lon2d = np.meshgrid(LAT, LON, indexing="ij")
    # Tropics-peaked spatial pattern (aerosol loading is higher near the equator).
    spatial = np.exp(-((lat2d / 40.0) ** 2)) * (
        1.0 + 0.3 * np.cos(np.deg2rad(lon2d))
    )
    spatial = spatial / spatial.max()
    field = np.empty((len(TIME), len(LAT), len(LON)))
    for t in range(len(TIME)):
        seasonal = 1.0 + seasonal_amp * np.sin(2 * np.pi * (t + 1) / 12.0)
        noise = 1.0 + 0.05 * rng.standard_normal((len(LAT), len(LON)))
        field[t] = base * (1.0 + spatial_amp * spatial) * seasonal * noise
    return np.clip(field, floor, None)


def _da(values, name):
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": TIME, "lat": LAT, "lon": LON},
        name=name,
    )


def build():
    od550, abs550, load_total, load_bc_oa, emi_total, emi_bc_oa = ({} for _ in range(6))
    for i, model in enumerate(MODELS):
        rng = np.random.default_rng(1000 + i)
        # Per-model bias so models spread out on the SSA/MEC scatter.
        bias = 0.85 + 0.3 * (i / max(len(MODELS) - 1, 1))

        aod = _field(rng, base=0.15 * bias, spatial_amp=1.8, seasonal_amp=0.25, floor=1e-4)
        # Absorbing AOD ~5-15% of AOD -> SSA in ~0.85-0.95 range.
        aaod = aod * (0.05 + 0.10 * rng.random())
        load = _field(rng, base=1.5e-3 * bias, spatial_amp=1.8, seasonal_amp=0.25, floor=1e-7)
        load_bcoa = load * (0.08 + 0.05 * rng.random())
        emi = _field(rng, base=2.0e-11 * bias, spatial_amp=1.8, seasonal_amp=0.30, floor=0.0)
        emi_bcoa = emi * (0.10 + 0.05 * rng.random())

        od550[model] = {"AOD550": _da(aod, "AOD550")}
        abs550[model] = {"AAOD550": _da(aaod, "AAOD550")}
        load_total[model] = {"load_total": _da(load, "load_total")}
        load_bc_oa[model] = {"load_BC_OA": _da(load_bcoa, "load_BC_OA")}
        emi_total[model] = {"emi_total": _da(emi, "emi_total")}
        emi_bc_oa[model] = {"emi_BC_OA": _da(emi_bcoa, "emi_BC_OA")}

    return {
        "od550.pickle": od550,
        "abs550.pickle": abs550,
        "load_total.pickle": load_total,
        "load_BC_OA.pickle": load_bc_oa,
        "emi_total.pickle": emi_total,
        "emi_BC_OA.pickle": emi_bc_oa,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "Data" / "var_files" / "original" / "monthly"
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, data in build().items():
        with open(out_dir / filename, "wb") as f:
            pickle.dump(data, f)
        print(f"wrote {out_dir / filename}  ({len(data)} models)")

    print(f"\nSample monthly pickles ready in {out_dir}")


if __name__ == "__main__":
    main()
