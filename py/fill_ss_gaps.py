#!/usr/bin/env python3
"""Fill sea-salt rows that the main extract does not already hold.

* pd2010 seasonal rows for ``SPO_DJF`` (the remote South Pacific box). Read
  the 2010 burden and emission files (scaled like the other 2010 loads) and
  the optical ensemble, and append that region to the seasonal table.
* AeroCom sea-salt AE for the same boxes. Most models archive 870 nm rather
  than 865 nm. Rebuild each named region for every model that has ``emiss``,
  using 550–865 nm when it exists and 550–870 nm otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import PPE_set as cfg
import regions as reg
from regional_aggregates import weights_from_defs

ROOT = Path(cfg.get_run("pd2010")["root"])
PD2010 = ROOT / "PD" if (ROOT / "PD").is_dir() else ROOT
AERO_DIR = cfg.PROJECT_ROOT / "Data" / "AP3_processed_monthly"
AERO_OUT = cfg.PROJECT_ROOT / "tables" / "aerocom_ss_seasonal.parquet"
SEASONAL = cfg.REGIONAL_SEASONAL
SS_REGIONS = ("SPO_DJF", "SAO_SOND")
LOADSS_SCALE = cfg.BURDEN_STREAM_FACTOR["loadss"]


def _wmean(da: xr.DataArray, weight: xr.DataArray) -> xr.DataArray:
    aligned = da
    if not (np.array_equal(da.lat.values, weight.lat.values) and np.array_equal(da.lon.values, weight.lon.values)):
        aligned = da.interp(lat=weight.lat, lon=weight.lon, method="nearest")
    finite = np.isfinite(aligned)
    num = (aligned.where(finite) * weight).sum(("lat", "lon"), skipna=True)
    den = (weight * finite).sum(("lat", "lon"))
    return num / den


def _season_mean(series: pd.Series, months) -> float:
    mon = pd.to_datetime(series.index).month
    keep = series.loc[np.isin(mon, list(months))]
    return float(keep.mean()) if len(keep) else np.nan


def fill_pd2010_region(region: str = "SPO_DJF") -> None:
    """Append one sea-salt region for every pd2010 member already in the table."""
    if region not in reg.REGIONS:
        raise SystemExit(f"{region} is not in regions.REGIONS")
    months = tuple(reg.REGIONS[region]["season_months"])
    existing = pd.read_parquet(SEASONAL)
    ident = existing.loc[
        (existing.ppe_id == "pd2010") & (existing.region == "SAO_SOND"),
        ["ens", "model", "experiment"],
    ].drop_duplicates()
    if ident.empty:
        raise SystemExit("pd2010 SAO_SOND rows missing; cannot align member ids")

    template = xr.open_dataset(next(PD2010.glob("PPE_ENS_1/*burden.nc")))
    weights = weights_from_defs(template["burden_SS"].isel(time=0), {region: reg.REGIONS[region]})
    w = weights.sel(region=region)
    template.close()

    rows = []
    for _, meta in ident.iterrows():
        label = str(meta["model"])
        folder = PD2010 / ("PPE_ENS_Control" if label == "Control" else f"PPE_ENS_{label}")
        burden = next(folder.glob("*burden.nc"))
        emi = next(folder.glob("*emi.nc"))
        with xr.open_dataset(burden) as ds:
            loads = {}
            for src, dest, scale in (
                ("burden_SS", "loadss", LOADSS_SCALE),
                ("burden_BC", "loadbc", cfg.BURDEN_STREAM_FACTOR["loadbc"]),
                ("burden_OC", "loadoa", cfg.BURDEN_STREAM_FACTOR["loadoa"]),
                ("burden_DU", "loaddust", cfg.BURDEN_STREAM_FACTOR["loaddust"]),
                ("burden_SO4", "loadso4", cfg.BURDEN_STREAM_FACTOR["loadso4"]),
            ):
                if src not in ds:
                    continue
                series = pd.Series(
                    _wmean(ds[src] * scale, w).values,
                    index=pd.to_datetime(ds.time.values),
                )
                loads[dest] = _season_mean(series, months)
        with xr.open_dataset(emi) as ds:
            fluxes = {}
            for src, dest in (
                ("emi_SS", "emiss"),
                ("emi_SO2", "emiso2"),
                ("emi_BC", "emibc"),
                ("emi_OC", "emioa"),
                ("emi_DU", "emidust"),
                ("emi_SO4", "emi_SO4"),
                ("emi_DMS", "emi_DMS"),
            ):
                if src not in ds:
                    continue
                flux = _wmean(ds[src], w)
                flux_s = pd.Series(flux.values, index=pd.to_datetime(flux.time.values))
                fluxes[dest] = _season_mean(flux_s, months)
        rec = {
            "ppe_id": "pd2010",
            "ens": int(meta["ens"]),
            "model": label,
            "experiment": meta["experiment"],
            "region": region,
            "n_months": len(months),
        }
        rec.update(fluxes)
        rec.update(loads)
        rows.append(rec)
        if len(rows) % 40 == 0:
            print(f"  pd2010 {region} {len(rows)}/{len(ident)}", flush=True)

    # Angstrom from the ensemble file, same season.
    opt = xr.open_dataset(cfg.ensemble_file("pd2010", "optical"))
    ang = _wmean(opt["ANG_550nm_865nm"], w)
    opt.close()
    ang_s = pd.DataFrame({
        "time": pd.to_datetime(ang.time.values),
        "ens": np.repeat(ang.ens.values, ang.sizes["time"]) if False else None,
    })
    # ang is (ens, time)
    ae_rows = []
    for ens_id in ang.ens.values:
        series = pd.Series(ang.sel(ens=ens_id).values, index=pd.to_datetime(ang.time.values))
        ae_rows.append({"ens": int(ens_id), "AE_550_865": _season_mean(series, months)})
    ae = pd.DataFrame(ae_rows)
    out = pd.DataFrame(rows).merge(ae, on="ens", how="left")

    # Column optics from the ensemble so SAO_2 has the same MAC / MEC fields.
    opt = xr.open_dataset(cfg.ensemble_file("pd2010", "optical"))
    opt_rows = []
    series = {}
    for var in ("od550aer", "abs550aer", "od550ss"):
        reg_mean = _wmean(opt[var], w)
        series[var] = reg_mean
    opt.close()
    for ens_id in series["od550aer"].ens.values:
        rec = {"ens": int(ens_id)}
        for var, da in series.items():
            s = pd.Series(da.sel(ens=ens_id).values, index=pd.to_datetime(da.time.values))
            rec[var] = _season_mean(s, months)
        opt_rows.append(rec)
    out = out.merge(pd.DataFrame(opt_rows), on="ens", how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["MEC_ss"] = out["od550ss"] / (out["loadss"] * 1e3)
        load_total = out[["loadbc", "loadoa", "loaddust", "loadss", "loadso4"]].sum(axis=1, min_count=1)
        out["MEC"] = out["od550aer"] / (load_total * 1e3)
        out["MAC_total"] = out["abs550aer"] / (load_total * 1e3)
        out["SSA"] = 1.0 - out["abs550aer"] / out["od550aer"]
    out["MEC_ss"] = out["MEC_ss"].where(out["loadss"] > 0)
    out["MEC"] = out["MEC"].where(load_total > 0)
    out["MAC_total"] = out["MAC_total"].where(load_total > 0)
    out["SSA"] = out["SSA"].where((out["od550aer"] > 0.01) & out["SSA"].between(0, 1))

    keep = existing.loc[~((existing.ppe_id == "pd2010") & (existing.region == region))]
    merged = pd.concat([keep, out], ignore_index=True, sort=False)
    merged.to_parquet(SEASONAL, index=False)
    print(f"Appended {len(out)} pd2010 {region} rows -> {SEASONAL.name}")


def _open_var(model: str, var: str):
    path = AERO_DIR / var / f"{model}_{var}_processed.nc"
    if not path.is_file():
        return None
    ds = xr.open_dataset(path)
    name = var if var in ds.data_vars else next(iter(ds.data_vars))
    da = ds[name].load()
    ds.close()
    return da


def fill_aerocom(regions=SS_REGIONS) -> None:
    models = sorted(p.name[: -len("_emiss_processed.nc")] for p in (AERO_DIR / "emiss").glob("*_emiss_processed.nc"))
    defs = {name: reg.REGIONS[name] for name in regions}
    rows = []
    weight_cache = {}
    for i, model in enumerate(models, 1):
        fields = {var: _open_var(model, var) for var in
                  ("emiss", "loadss", "od550aer", "od865aer", "od870aer")}
        template = next(da for da in fields.values() if da is not None)
        key = (template.lat.values.tobytes(), template.lon.values.tobytes())
        if key not in weight_cache:
            weight_cache[key] = weights_from_defs(template.isel(time=0), defs)
        weights = weight_cache[key]
        for region, cfg in defs.items():
            w = weights.sel(region=region)
            rec = {"model": model, "region": region, "ppe_id": "AeroCom"}
            monthly = {}
            for var, da in fields.items():
                if da is None:
                    rec[var] = np.nan
                    continue
                series = _wmean(da, w)
                monthly[var] = pd.Series(np.asarray(series.values, float), index=pd.to_datetime(series.time.values))
                rec[var] = _season_mean(monthly[var], cfg["season_months"])
            od = monthly.get("od550aer")
            ae = np.nan
            if od is not None and "od865aer" in monthly:
                ratio = od / monthly["od865aer"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    per = -np.log(ratio) / np.log(550.0 / 865.0)
                ae = _season_mean(per.replace([np.inf, -np.inf], np.nan), cfg["season_months"])
                rec["AE_pair"] = "550-865"
            if not np.isfinite(ae) and od is not None and "od870aer" in monthly:
                ratio = od / monthly["od870aer"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    per = -np.log(ratio) / np.log(550.0 / 870.0)
                ae = _season_mean(per.replace([np.inf, -np.inf], np.nan), cfg["season_months"])
                rec["AE_pair"] = "550-870"
            rec["AE"] = ae
            rec["AE_550_865"] = rec["AE"] if rec.get("AE_pair") == "550-865" else np.nan
            rec["AE_550_870"] = rec["AE"] if rec.get("AE_pair") == "550-870" else np.nan
            rows.append(rec)
        print(f"  AeroCom {i}/{len(models)} {model}", flush=True)
    out = pd.DataFrame(rows)
    if AERO_OUT.is_file():
        old = pd.read_parquet(AERO_OUT)
        old = old.loc[~old["region"].isin(list(regions))]
        out = pd.concat([old, out], ignore_index=True, sort=False)
    AERO_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(AERO_OUT, index=False)
    have = out.dropna(subset=["emiss", "loadss"])
    print(f"Wrote {AERO_OUT.name}: {out.model.nunique()} emission models, "
          f"{have.model.nunique()} with load+emission, "
          f"AE finite {out.AE.notna().sum()} rows")
    print(out.groupby(["region", "AE_pair"]).model.nunique().to_string())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "pd2010"):
        for region in SS_REGIONS:
            fill_pd2010_region(region)
    if which in ("both", "aerocom"):
        fill_aerocom(SS_REGIONS)
