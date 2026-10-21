#!/usr/bin/env python3
"""Sanity checks for a PPE extract before analysis.

For each requested PPE (scratch ensemble files, raw = unscaled burdens):
  * global annual-mean aerosol burdens (Tg), emissions (Tg yr-1) and
    emission lifetimes (days) for Control and the ensemble median;
  * the same quantities for AeroCom AP3 models (Data/AP3_processed_monthly),
    with the ECHAM-HAM entries singled out;
  * implied burden scale = AeroCom ECHAM-HAM burden / PPE Control raw burden;
  * dust MEC (m2 g-1) after the configured burden_scale;
  * SO4 lifetime using the SO2 and DMS source in ppe_io.LIFETIME_DEFS;
  * time axis (12 distinct months inside the run period) and provenance
    completeness (variables missing for any member, precip excluded when the
    run has no precip stream).

Writes Data/PPE_regional/validation_<ppe_id>.csv and prints a summary.

    python py/ppe_validate.py --ppe pd2010 pd2024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import PPE_set as cfg  # noqa: E402

EARTH_R = 6.371e6
SPECIES = {
    "du": ("loaddust", "emidust", "od550dust"),
    "ss": ("loadss", "emiss", "od550ss"),
    "bc": ("loadbc", "emibc", "od550bc"),
    "oa": ("loadoa", "emioa", "od550oa"),
    "so4": ("loadso4", None, "od550so4"),
}
AEROCOM_DIR = cfg.PROJECT_ROOT / "Data" / "AP3_processed_monthly"
ECHAM_HAM_TAGS = ("ECHAM6.3-HAM2.3", "ECHAM6-HAM2")


def cell_area(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    dlat = np.deg2rad(abs(float(np.median(np.diff(lat)))))
    dlon = np.deg2rad(abs(float(np.median(np.diff(lon)))))
    return (EARTH_R ** 2 * dlat * dlon * np.cos(np.deg2rad(lat)))[:, None] * np.ones(len(lon))[None, :]


def global_total(da: xr.DataArray) -> xr.DataArray:
    """Area integral over lat/lon (kg for loads, kg s-1 for fluxes), time-mean."""
    area = xr.DataArray(cell_area(da.lat.values, da.lon.values), dims=("lat", "lon"),
                        coords={"lat": da.lat, "lon": da.lon})
    tot = (da * area).sum(("lat", "lon"))
    return tot.mean("time") if "time" in tot.dims else tot


def ppe_globals(ppe_id: str) -> pd.DataFrame:
    rows = []
    burden = xr.open_dataset(cfg.ensemble_file(ppe_id, "burden"), engine=cfg.XR_WRITE_ENGINE)
    emi = xr.open_dataset(cfg.ensemble_file(ppe_id, "emi"), engine=cfg.XR_WRITE_ENGINE)
    opt = xr.open_dataset(cfg.ensemble_file(ppe_id, "optical"), engine=cfg.XR_WRITE_ENGINE)
    try:
        for sp, (lv, ev, ov) in SPECIES.items():
            scale = cfg.burden_scale_for(ppe_id, lv)
            load_tg = global_total(burden[lv].load()) / 1e9
            row = {"ppe_id": ppe_id, "species": sp}
            ctrl = load_tg.sel(ens=cfg.CONTROL_ENS_ID) if cfg.CONTROL_ENS_ID in load_tg.ens else None
            row["burden_raw_Tg_control"] = float(ctrl) if ctrl is not None else np.nan
            row["burden_raw_Tg_median"] = float(load_tg.median("ens"))
            if ev is not None:
                emi_tg_yr = global_total(emi[ev].load()) * 86400 * 365 / 1e9
                row["emission_Tg_yr_median"] = float(emi_tg_yr.median("ens"))
                tau = (load_tg / (emi_tg_yr / 365.0))
                row["lifetime_raw_d_median"] = float(tau.median("ens"))
                if scale is not None:
                    row["lifetime_scaled_d_median"] = float((tau * scale).median("ens"))
            if ov in opt and scale is not None:
                od = opt[ov].load()
                area = cell_area(od.lat.values, od.lon.values)
                aod_global = float((od.mean("time") * area).sum(("lat", "lon")).median("ens")
                                   / area.sum())
                load_g_m2 = row["burden_raw_Tg_median"] * scale * 1e12 / area.sum()
                row["MEC_global_m2_g"] = aod_global / load_g_m2 if load_g_m2 > 0 else np.nan
            rows.append(row)
    finally:
        burden.close(); emi.close(); opt.close()
    return pd.DataFrame(rows)


def _open_aerocom(var: str, model: str) -> Optional[xr.DataArray]:
    path = AEROCOM_DIR / var / f"{model}_{var}_processed.nc"
    if not path.is_file():
        return None
    ds = xr.open_dataset(path)
    da = ds[list(ds.data_vars)[0]]
    ren = {k: v for k, v in (("latitude", "lat"), ("longitude", "lon")) if k in da.dims}
    return da.rename(ren).load()


def aerocom_globals() -> pd.DataFrame:
    rows = []
    for sp, (lv, ev, _ov) in SPECIES.items():
        vdir = AEROCOM_DIR / lv
        if not vdir.is_dir():
            continue
        for f in sorted(vdir.glob(f"*_{lv}_processed.nc")):
            model = f.name[: -len(f"_{lv}_processed.nc")]
            try:
                load = _open_aerocom(lv, model)
                row = {"model": model, "species": sp,
                       "burden_Tg": float(global_total(load)) / 1e9}
                if ev is not None:
                    e = _open_aerocom(ev, model)
                    if e is not None:
                        etot = float(global_total(e)) * 86400 * 365 / 1e9
                        row["emission_Tg_yr"] = etot
                        row["lifetime_d"] = row["burden_Tg"] / (etot / 365.0) if etot else np.nan
                rows.append(row)
            except Exception as exc:
                print(f"  AeroCom {model} {lv}: {exc}")
    return pd.DataFrame(rows)


def sulfur_lifetime_check(ppe_id: str) -> pd.DataFrame:
    """Global check of the SO4 lifetime that uses SO2 and DMS emissions.

    Same weights as ``ppe_io.LIFETIME_DEFS['lifetime_SO4']`` and ``lifetime_S``:

    * ``lifetime_SO4``: sulfate burden / SO4-equivalent source, with yield
      ``SO2_TO_SO4_FRAC`` on SO2 and DMS and molar conversion M_SO4/M_SO2
      and M_SO4/M_DMS (one S atom each).
    * ``lifetime_S``: elemental-S burden (SO4 + SO2 + DMS) / elemental-S
      emission, with no yield factor.

    Area-weighted global means, ensemble median. A yield of 1 is printed
    beside the configured yield so the factor is visible.
    """
    import ppe_io

    burden = xr.open_dataset(cfg.ensemble_file(ppe_id, "burden"), engine=cfg.XR_WRITE_ENGINE)
    emi = xr.open_dataset(cfg.ensemble_file(ppe_id, "emi"), engine=cfg.XR_WRITE_ENGINE)
    try:
        def gmean(da):
            area = xr.DataArray(
                cell_area(da.lat.values, da.lon.values),
                dims=("lat", "lon"), coords={"lat": da.lat, "lon": da.lon},
            )
            w = area / area.sum()
            out = (da * w).sum(("lat", "lon"))
            return out.mean("time") if "time" in out.dims else out

        scale = cfg.burden_scale_for(ppe_id, "loadso4") or 1.0
        load_so4 = gmean(burden["loadso4"].load()) * scale
        b_so2 = gmean(burden["burden_SO2"].load()) if "burden_SO2" in burden else None
        b_dms = gmean(burden["burden_DMS"].load()) if "burden_DMS" in burden else None
        terms = {}
        for name in ("emi_SO4", "emiso2", "emi_DMS"):
            if name not in emi:
                print(f"  [{ppe_id}] missing {name}; SO4-equivalent lifetime is undefined")
                return pd.DataFrame()
            terms[name] = gmean(emi[name].load())
        f = ppe_io.SO2_TO_SO4_FRAC
        w_so2 = f * ppe_io.M_SO4 / ppe_io.M_SO2
        w_dms = f * ppe_io.M_SO4 / ppe_io.M_DMS
        src = terms["emi_SO4"] + w_so2 * terms["emiso2"] + w_dms * terms["emi_DMS"]
        src1 = terms["emi_SO4"] + (w_so2 / f) * terms["emiso2"] + (w_dms / f) * terms["emi_DMS"]
        s_load = (load_so4 * ppe_io.M_S / ppe_io.M_SO4
                  + b_so2 * ppe_io.M_S / ppe_io.M_SO2
                  + b_dms * ppe_io.M_S / ppe_io.M_DMS)
        s_emi = (terms["emi_SO4"] * ppe_io.M_S / ppe_io.M_SO4
                 + terms["emiso2"] * ppe_io.M_S / ppe_io.M_SO2
                 + terms["emi_DMS"] * ppe_io.M_S / ppe_io.M_DMS)

        def med(da):
            return float(da.median("ens"))

        parts = {
            "emi_SO4": med(terms["emi_SO4"]),
            "emiso2_as_SO4": med(w_so2 * terms["emiso2"]),
            "emi_DMS_as_SO4": med(w_dms * terms["emi_DMS"]),
        }
        total = sum(parts.values())
        row = {
            "ppe_id": ppe_id,
            "loadso4_kg_m-2": med(load_so4),
            "lifetime_SO4_d": med(load_so4 / (src * 86400.0)),
            "lifetime_SO4_yield1_d": med(load_so4 / (src1 * 86400.0)),
            "lifetime_S_d": med(s_load / (s_emi * 86400.0)),
            "frac_emi_SO4": parts["emi_SO4"] / total,
            "frac_emiso2": parts["emiso2_as_SO4"] / total,
            "frac_emi_DMS": parts["emi_DMS_as_SO4"] / total,
            "w_SO2": w_so2,
            "w_DMS": w_dms,
            "yield": f,
        }
        print(f"  [{ppe_id}] SO4-equivalent source fractions "
              f"(yield={f}): primary SO4 {row['frac_emi_SO4']:.2f}, "
              f"SO2 {row['frac_emiso2']:.2f}, DMS {row['frac_emi_DMS']:.2f}")
        print(f"           lifetime_SO4 {row['lifetime_SO4_d']:.2f} d "
              f"(yield 1 would be {row['lifetime_SO4_yield1_d']:.2f} d); "
              f"lifetime_S {row['lifetime_S_d']:.2f} d")
        return pd.DataFrame([row])
    finally:
        burden.close()
        emi.close()


def time_axis_check(ppe_id: str) -> Dict[str, object]:
    ds = xr.open_dataset(cfg.ensemble_file(ppe_id, "burden"), engine=cfg.XR_WRITE_ENGINE)
    times = pd.to_datetime(ds.time.values)
    ds.close()
    t0, t1 = (pd.Timestamp(t) for t in cfg.get_run(ppe_id)["period"])
    return {
        "n_months": int(len(times)),
        "first": str(times.min())[:10],
        "last": str(times.max())[:10],
        "inside_period": bool(((times >= t0) & (times <= t1)).all()),
        "unique_months": int(times.to_period("M").nunique()),
    }


def provenance_check(ppe_id: str) -> pd.DataFrame:
    prov = pd.read_csv(cfg.run_dirs(ppe_id)["provenance"])
    prov = prov.loc[~prov["aerocom_name"].isin(cfg.MISSING_VARS)]
    if not cfg.get_run(ppe_id)["has_precip"]:
        prov = prov.loc[~prov["aerocom_name"].isin(list(cfg.PRECIP_VARS) + ["precip"])]
    miss = prov.loc[prov["status"] == "missing"]
    return (miss.groupby("aerocom_name")["ens_id"].nunique().rename("n_members_missing")
            .reset_index())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ppe", nargs="+", default=list(cfg.PPE_RUNS))
    args = p.parse_args(argv)

    print("--- AeroCom AP3 global budgets ---")
    aer = aerocom_globals()
    aer_med = aer.groupby("species")[["burden_Tg", "emission_Tg_yr", "lifetime_d"]].median()
    ham = aer.loc[aer["model"].str.contains("|".join(ECHAM_HAM_TAGS), regex=True)]
    ham_med = ham.groupby("species")[["burden_Tg", "emission_Tg_yr", "lifetime_d"]].median()
    print("AeroCom multi-model median:\n", aer_med.round(3))
    print("AeroCom ECHAM-HAM median:\n", ham_med.round(3))

    cfg.REGIONAL_DIR.mkdir(parents=True, exist_ok=True)
    for ppe_id in args.ppe:
        if not cfg.ensemble_file(ppe_id, "burden").is_file():
            print(f"\n[{ppe_id}] no ensemble files; skipped")
            continue
        print(f"\n=== {ppe_id} (burden_scale={cfg.get_run(ppe_id)['burden_scale']}) ===")
        g = ppe_globals(ppe_id)
        g["aerocom_burden_Tg_median"] = g["species"].map(aer_med["burden_Tg"])
        g["aerocom_ham_burden_Tg"] = g["species"].map(ham_med["burden_Tg"])
        g["aerocom_lifetime_d_median"] = g["species"].map(aer_med["lifetime_d"])
        g["aerocom_ham_lifetime_d"] = g["species"].map(ham_med["lifetime_d"])
        g["aerocom_emission_Tg_yr_median"] = g["species"].map(aer_med["emission_Tg_yr"])
        g["implied_scale_vs_ham"] = g["aerocom_ham_burden_Tg"] / g["burden_raw_Tg_control"]
        g["implied_scale_vs_median"] = g["aerocom_burden_Tg_median"] / g["burden_raw_Tg_median"]
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(g.round(4).to_string(index=False))
        sulfur = sulfur_lifetime_check(ppe_id)
        if not sulfur.empty:
            for col, val in sulfur.iloc[0].items():
                if col != "ppe_id":
                    g[col] = val
        ta = time_axis_check(ppe_id)
        print("time axis:", ta)
        pv = provenance_check(ppe_id)
        print("variables missing for >=1 member:",
              "none" if pv.empty else pv.to_dict("records"))
        out = cfg.REGIONAL_DIR / f"validation_{ppe_id}.csv"
        g.assign(**{f"time_{k}": v for k, v in ta.items()},
                 n_vars_missing=len(pv)).to_csv(out, index=False)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
