"""Seasonal regional dry deposition fraction for AeroCom AP3 2010.

    f_dry = dry / (dry + wet)

One panel figure per species. Regions are the analysis boxes used by the
species notebooks:
  * BC+OA:    africa, amazon, outflow_af          (AAOD_error_attribution)
  * sea salt: SAO_SOND, SAO_2_DJF (highest 1/τ vs AE; NPO removed)
  * sulfate:  the same sea-salt basins            (sulfate is diagnosed there)
  * dust:     sahara, australia, atacama, outflow_sahara
                                                 (AOD_DUST_ERROR_PPE)

A flux is the sum of its components (BC+OA is bc+oa). When a model has no dry
file, dry = dep - wet. When it has no wet file, wet = dep - dry. Both the
reported flux and the total deposition must be complete; a missing component
is treated as that flux not existing. Models with neither dry nor wet cannot
be split from total deposition alone.

Run: `.venv-1/bin/python notebooks/drywet_deposition_fraction.py`
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

project_root = Path(__file__).resolve().parent.parent
py_dir = project_root / "py"
nb_dir = project_root / "notebooks"
for p in (str(project_root), str(py_dir), str(nb_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import aerocom_data
import notebook_setup as setup
import regions as reg

# Same exclusions as AAOD_error_attribution.ipynb and AOD_SS_ERROR.ipynb.
EXCLUDE_MODELS = [ 
    "GEOS-i33p2-met2010_AP3-CTRL", 
    "GISS-ModelE2p1p1-OMA_AP3-CTRL", 
    "GISS-ModelE2p1p1-MATRIX_AP3-CTRL",
    "NorESM2-met2010_AP3-CTRL-v3",
]

_SS_REGIONS = ["SAO_SOND", "SAO_2_DJF"]

# dry / wet / dep are AeroCom variable names summed before the fraction.
# An empty tuple means that flux is not archived (reconstruct from dep).
SPECIES: Dict[str, dict] = {
    "bc_oa": dict(
        label="BC+OA",
        dry=("drybc", "dryoa"),
        wet=("wetbc", "wetoa"),
        dep=("depbc", "depoa"),
        regions=["africa", "amazon", "outflow_af"],
        legacy_csv="dry_wet_deposition_fraction_BC_OA.csv",
        legacy_fig="drywet_fraction_BC_OA_bars.png",
    ),
    "ss": dict(
        label="Sea salt",
        dry=("dryss",),
        wet=("wetss",),
        dep=("depss",),
        regions=list(_SS_REGIONS),
    ),
    "du": dict(
        label="Dust",
        dry=("drydu",),
        wet=("wetdu",),
        dep=("depdust",),
        regions=["sahara", "australia", "atacama", "outflow_sahara"],
    ),
    "so4+ss": dict(
        label="Sulfate + Sea salt",
        dry=("dryso4", "dryss"),
        wet=("wetso4", "wetss"),
        dep=("depso4", "depss"),
        regions=list(_SS_REGIONS),
    ),
}

FIGURE_DIR = project_root / "figure"/"dep"
TABLE_DIR = project_root / "tables"
DATA_DIR = project_root / "Data" / "AP3_processed_monthly"
REGION_VIEW = setup.ALL_REGIONS


def _column_flux(da):
    """Drop singleton extra dims. Return None when a real vertical axis remains."""
    extra = [d for d in da.dims if d not in ("time", "lat", "lon")]
    for dim in extra:
        if da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)
        else:
            return None
    return da


def _positive_flux(ds, model: str, var: str):
    if ds is None:
        return None
    try:
        ds_n = aerocom_data.normalize_dataset_time(ds, var_hint=f"{model}/{var}", year=2010)
        if ds_n is None or not list(ds_n.data_vars):
            return None
        da = setup.positive_deposition(ds_n[list(ds_n.data_vars)[0]])
        da = _column_flux(da)
        return None if da is None else da.load()
    finally:
        try:
            ds.close()
        except Exception:
            pass


def _sum_on(parts: Sequence, ref):
    """Sum DataArrays after aligning each onto ``ref``. None if any part is missing."""
    if not parts or any(p is None for p in parts):
        return None
    total = None
    for part in parts:
        part = aerocom_data._align_da_to_ref(part, ref)
        total = part if total is None else total + part
    return total


def resolve_dry_wet(fields: Dict[str, object], spec: dict):
    """Return (dry, wet, source) for one model.

    ``source`` is ``dry+wet``, ``dep-wet``, or ``dep-dry``.
    """
    names = list(spec["dry"]) + list(spec["wet"]) + list(spec["dep"])
    ref = next((fields[n] for n in names if fields.get(n) is not None), None)
    if ref is None:
        return None, None, "no fluxes"

    dry = _sum_on([fields.get(n) for n in spec["dry"]], ref) if spec["dry"] else None
    wet = _sum_on([fields.get(n) for n in spec["wet"]], ref) if spec["wet"] else None
    dep = _sum_on([fields.get(n) for n in spec["dep"]], ref) if spec["dep"] else None
    if dry is not None and wet is not None:
        return dry, wet, "dry+wet"
    if dry is None and wet is not None and dep is not None:
        return dep - wet, wet, "dep-wet"
    if wet is None and dry is not None and dep is not None:
        return dry, dep - dry, "dep-dry"
    missing = []
    if dry is None:
        missing.append("dry " + "+".join(spec["dry"]) if spec["dry"] else "dry (not archived)")
    if wet is None:
        missing.append("wet " + "+".join(spec["wet"]) if spec["wet"] else "wet (not archived)")
    if dep is None:
        missing.append("dep " + "+".join(spec["dep"]))
    return None, None, "incomplete: " + ", ".join(missing)


def _scalar(value) -> float:
    try:
        if hasattr(value, "values"):
            return float(np.nanmean(np.asarray(value.values)))
        return float(value)
    except Exception:
        return np.nan


def seasonal_table(model_fields: Dict[str, dict], spec: dict) -> pd.DataFrame:
    """Regional seasonal means and dry fraction for one species."""
    regions = [r for r in spec["regions"] if r in REGION_VIEW]
    missing = [r for r in spec["regions"] if r not in REGION_VIEW]
    if missing:
        raise KeyError(f"{spec['label']}: regions not in the registry: {missing}")
    template = next(iter(model_fields.values()))["wet"].isel(time=0, drop=True)
    aligned = {}
    for model, pair in model_fields.items():
        aligned[model] = {
            "dry": aerocom_data._align_da_to_ref(pair["dry"], template, model_hint=model),
            "wet": aerocom_data._align_da_to_ref(pair["wet"], template, model_hint=model),
            "flux_source": pair["flux_source"],
        }
    masks = setup.create_analysis_masks(template, regions, region=REGION_VIEW)
    rows = []
    for region in regions:
        dry_map = setup.aggregate_region(
            aligned, "dry", region, masks, region=REGION_VIEW,
            return_time_series=False, skipna=True,
        )
        wet_map = setup.aggregate_region(
            aligned, "wet", region, masks, region=REGION_VIEW,
            return_time_series=False, skipna=True,
        )
        for model in sorted(set(dry_map) & set(wet_map)):
            dry, wet = _scalar(dry_map[model]), _scalar(wet_map[model])
            if not (np.isfinite(dry) and np.isfinite(wet)) or dry <= 0 or wet <= 0:
                print(f"  {spec['label']} {model} {region}: non-positive seasonal "
                      f"dry={dry:.3g} wet={wet:.3g}; skipped")
                continue
            tot = dry + wet
            rows.append({
                "species": spec["label"],
                "region": region,
                "model": model,
                "dry": dry,
                "wet": wet,
                "dry_frac": dry / tot,
                "wet_frac": wet / tot,
                "flux_source": aligned[model]["flux_source"],
            })
    return pd.DataFrame(rows, columns=[
        "species", "region", "model", "dry", "wet", "dry_frac", "wet_frac", "flux_source",
    ])


def plot_fraction_panels(df: pd.DataFrame, spec: dict, save_path: Path,
                         also: Optional[Path] = None) -> None:
    """One panel per analysis region. Hatched bars were reconstructed from dep."""
    regions = [r for r in spec["regions"] if r in set(df["region"])]
    if df.empty or not regions:
        print(f"  {spec['label']}: nothing to plot")
        return
    models = sorted(df["model"].unique())
    labels = [setup.short_model_name(m) for m in models]
    source = df.drop_duplicates("model").set_index("model")["flux_source"]
    colors = reg.region_colors(regions)
    x = np.arange(len(models))
    fig, axes = plt.subplots(
        len(regions), 1,
        figsize=(max(9.0, 0.48 * len(models) + 3.0), 3.5 * len(regions)),
        squeeze=False, sharex=True,
    )
    hatched = False
    for ax, region in zip(axes[:, 0], regions):
        sub = df[df["region"] == region].set_index("model").reindex(models)
        reconstructed = source.reindex(models).ne("dry+wet").fillna(False).to_numpy()
        hatched = hatched or bool(reconstructed.any())
        bars = ax.bar(
            x, sub["dry_frac"].to_numpy(),
            color=colors.get(region, "tab:blue"), edgecolor="k", linewidth=0.6,
        )
        for bar, recon in zip(bars, reconstructed):
            if recon:
                bar.set_hatch("///")
        med = np.nanmedian(sub["dry_frac"].to_numpy())
        if np.isfinite(med):
            ax.axhline(med, color="k", ls="--", lw=1, label=f"median = {med:.2f}")
            ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("dry / (dry + wet)")
        n = int(sub["dry_frac"].notna().sum())
        ax.set_title(f"{region} (n={n})", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    title = f"{spec['label']} dry deposition fraction"
    if hatched:
        title += "\n(hatched: dry or wet missing, filled as total deposition − the other flux)"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (save_path, also):
        if path is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def _variables() -> List[str]:
    names = []
    for spec in SPECIES.values():
        names.extend(spec["dry"])
        names.extend(spec["wet"])
        names.extend(spec["dep"])
    return sorted(set(names))


def main() -> pd.DataFrame:
    print(f"Project root: {project_root}")
    print(f"Loading { _variables() } from {DATA_DIR}")
    raw = aerocom_data.load_monthly_data_from_netcdf(
        output_base_dir=str(DATA_DIR), variables=_variables(),
    )
    models = sorted(m for m in raw if m not in EXCLUDE_MODELS)
    print(f"{len(models)} models ({len(raw) - len(models)} excluded)")

    # Materialize positive column fluxes once; species then only sum and subtract.
    fluxes: Dict[str, Dict[str, object]] = {}
    for model in models:
        fluxes[model] = {}
        for var in _variables():
            fluxes[model][var] = _positive_flux(raw[model].get(var), model, var)

    frames = []
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for key, spec in SPECIES.items():
        print(f"\n=== {spec['label']}  regions={spec['regions']} ===")
        fields = {}
        for model in models:
            dry, wet, source = resolve_dry_wet(fluxes[model], spec)
            if dry is None or wet is None:
                print(f"  {model}: {source}; skipped")
                continue
            fields[model] = {"dry": dry, "wet": wet, "flux_source": source}
            print(f"  {model}: {source}")
        if not fields:
            print(f"  {spec['label']}: no model has a usable dry/wet pair")
            continue
        df = seasonal_table(fields, spec)
        frames.append(df)
        if df.empty:
            continue
        print(df.pivot(index="model", columns="region", values="dry_frac").round(3).to_string())
        counts = df.drop_duplicates("model")["flux_source"].value_counts()
        print("Flux source (models): " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        fig_path = FIGURE_DIR / f"drywet_fraction_{key}.png"
        legacy_fig = spec.get("legacy_fig")
        plot_fraction_panels(
            df, spec, fig_path,
            also=(FIGURE_DIR / legacy_fig) if legacy_fig else None,
        )
        csv_path = TABLE_DIR / f"dry_wet_deposition_fraction_{key}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        legacy_csv = spec.get("legacy_csv")
        if legacy_csv:
            df.to_csv(TABLE_DIR / legacy_csv, index=False)
            print(f"Saved: {TABLE_DIR / legacy_csv}")

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = TABLE_DIR / "dry_wet_deposition_fraction.csv"
    out.to_csv(combined, index=False)
    print(f"\nSaved: {combined} ({len(out)} rows)")
    return out


if __name__ == "__main__":
    main()
