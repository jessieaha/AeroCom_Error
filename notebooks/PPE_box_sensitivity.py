# %% [markdown]
# # Box size and location sensitivity (sea salt and biomass burning)
#
# Tests how the lifetime, MEC/MAC and SSA relations move when the analysis
# box is shifted by ±5° in longitude and latitude and resized by ±5° in both
# directions. Every box is at least 7° in latitude and 15° in longitude.
#
# Sea salt starts from a large remote Southern Ocean box in the peak DMS
# season (DJF, far from land) and from an Amazon-sized box nested inside it.
# Biomass burning uses the Africa and Amazon source boxes.
#
# Background of the maps: PPE 2024 ensemble-mean species AOD at 550 nm
# (sea salt, or BC+OA), same colour scale as the regional map in
# ``AAOD_error_attribution``.
#
# Run: `.venv-1/bin/python notebooks/PPE_box_sensitivity.py`
# Tables are built by `python py/box_sensitivity.py` if they are missing.

# %%
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    get_ipython()  # noqa: F821
    _IN_IPYTHON = True
except NameError:
    _IN_IPYTHON = False

import matplotlib
if not _IN_IPYTHON:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

for p in (str(project_root), str(project_root / "py"), str(project_root / "notebooks")):
    if p not in sys.path:
        sys.path.insert(0, p)

import box_sensitivity as boxes
import ppe_plots as pp

print(f"Project root: {project_root}")

# %%
if not boxes.PPE_SEASONAL.is_file() or not boxes.AEROCOM_SEASONAL.is_file():
    print("Building box-sensitivity tables (PPE + AeroCom)...")
    boxes.build()

FIGURE_DIR = project_root / "figure" / "PPE_box_sensitivity"
TABLE_DIR = project_root / "Data" / "PPE_analysis" / "box_sensitivity"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
save_fig = pp.FigureSaver(FIGURE_DIR, save=True)

defs = boxes.load_defs()
catalog = pd.DataFrame([
    {"region": n, "baseline": c["baseline"], "species": ",".join(c["species"]),
     "dlon": c["dlon"], "dlat": c["dlat"], "dsize": c["dsize"],
     "lon0": c["lon_range"][0], "lon1": c["lon_range"][1],
     "lat0": c["lat_range"][0], "lat1": c["lat_range"][1],
     "dlat_span": abs(c["lat_range"][1] - c["lat_range"][0]),
     "dlon_span": (c["lon_range"][1] - c["lon_range"][0]) % 360 or 360,
     "season": ",".join(str(m) for m in c["season_months"]),
     "n_cells": c.get("n_cells", np.nan)}
    for n, c in defs.items()
    if c["baseline"] in {b for names in boxes.BASELINES.values() for b in names}
])
catalog.to_csv(TABLE_DIR / "boxes.csv", index=False)
print(f"Boxes: {len(catalog)}")
print(catalog.groupby("baseline")[["dlat_span", "dlon_span", "n_cells"]].agg(["min", "median", "max"]))
too_small = catalog.loc[(catalog["dlat_span"] < boxes.MIN_DLAT) | (catalog["dlon_span"] < boxes.MIN_DLON)]
print(f"Boxes below the minimum size: {len(too_small)}")

# %%
# -----------------------------------------------------------------------------
# Maps. Solid = nominal box, dashed = ±5° shifts, dotted = ±5° size at the
# same centre. PPE 2024 ensemble mean is the background.
# -----------------------------------------------------------------------------
print("\n--- Maps ---")
ss_field = boxes.ensemble_mean_aod("pd2024", ["od550ss"], months=(12, 1, 2))
boxes.plot_study_map(
    ss_field, boxes.map_boxes(defs, ["SAO_SOND", "SAO_2_DJF"]),
    title="Sea salt: PPE 2024 ensemble-mean AOD$_{SS,550}$ (DJF), South Atlantic",
    cbar_label=r"AOD$_{SS,550}$ [-]",
    filename=FIGURE_DIR / "map_ss_SAO.png",
    extent=(-50, 20, -70, -20),
    central_longitude=0.0,
)
boxes.plot_study_map(
    ss_field, boxes.map_boxes(defs, boxes.BASELINES["ss"]),
    title="Sea salt: PPE 2024 ensemble-mean AOD$_{SS,550}$ (DJF)",
    cbar_label=r"AOD$_{SS,550}$ [-]",
    filename=FIGURE_DIR / "map_ss_DJF_global.png",
)

for baseline, months, extent in (
    ("africa", (6, 7, 8, 9), (0, 55, -35, 20)),
    ("amazon", (7, 8, 9, 10), (265, 340, -35, 15)),
):
    field = boxes.ensemble_mean_aod("pd2024", ["od550bc", "od550oa"], months=months)
    draw = boxes.map_boxes(defs, [baseline])
    boxes.plot_study_map(
        field, draw,
        title=f"Biomass burning: PPE 2024 ensemble-mean AOD$_{{BC+OA,550}}$ ({baseline})",
        cbar_label=r"AOD$_{BC+OA,550}$ [-]",
        filename=FIGURE_DIR / f"map_bb_{baseline}.png",
        extent=extent,
    )
bb_annual = boxes.ensemble_mean_aod("pd2024", ["od550bc", "od550oa"], months=None)
boxes.plot_study_map(
    bb_annual, boxes.map_boxes(defs, boxes.BASELINES["bb"]),
    title="Biomass burning: PPE 2024 ensemble-mean AOD$_{BC+OA,550}$ (annual)",
    cbar_label=r"AOD$_{BC+OA,550}$ [-]",
    filename=FIGURE_DIR / "map_bb_global.png",
)

# %%
def _relation_table(species: str) -> pd.DataFrame:
    """Slope and R² of each relation in every box, for each PPE and AeroCom."""
    ppe = pp.with_one_minus_ssa(boxes.ppe_frame(species))
    aero = boxes.screen_lifetime(
        boxes.aerocom_frame(species),
        ["lifetime_ss", "lifetime_mass", "lifetime_BC_OA", "lifetime"],
    )
    aero = pp.with_one_minus_ssa(aero)
    if not aero.empty:
        aero = aero.loc[aero["baseline"].isin(boxes.BASELINES[species])]
    frames = [ppe]
    if aero is not None and not aero.empty:
        frames.append(aero)
    df = pd.concat(frames, ignore_index=True, sort=False)
    rows = []
    for xcol, ycol, inverse_y, through_origin in boxes.RELATIONS[species]:
        for (ppe_id, region), sub in df.groupby(["ppe_id", "region"]):
            if ycol == "precip" or xcol == "precip":
                if not np.isfinite(sub["precip"].astype(float)).any():
                    continue
            x, y = pp.xy(sub, region, xcol, ycol, inverse_y)
            slope = intercept = r2 = np.nan
            if len(x) >= 8:
                slope, intercept, r2 = pp.fit_line(x, y, through_origin)
            meta = sub.iloc[0]
            rows.append({
                "species": species, "ppe_id": ppe_id, "region": region,
                "baseline": meta["baseline"], "dlon": int(meta["dlon"]),
                "dlat": int(meta["dlat"]), "dsize": int(meta["dsize"]),
                "x": xcol, "y": ycol, "inverse_y": inverse_y,
                "through_origin": through_origin,
                "slope": slope, "intercept": intercept, "r2": r2, "n": int(len(x)),
            })
    return pd.DataFrame(rows)


def _summary(stats: pd.DataFrame) -> pd.DataFrame:
    """Nominal R² against the min/max R² over the ±5° set."""
    rows = []
    keys = ["species", "ppe_id", "baseline", "x", "y"]
    for key, sub in stats.groupby(keys):
        nominal = sub.loc[(sub["dlon"] == 0) & (sub["dlat"] == 0) & (sub["dsize"] == 0)]
        if nominal.empty or not np.isfinite(nominal["r2"].iloc[0]):
            continue
        r0 = float(nominal["r2"].iloc[0])
        s0 = float(nominal["slope"].iloc[0])
        finite = sub.loc[np.isfinite(sub["r2"])]
        worst = finite.loc[(finite["r2"] - r0).abs().idxmax()]
        rows.append({
            "species": key[0], "ppe_id": key[1], "baseline": key[2],
            "x": key[3], "y": key[4],
            "r2_nominal": r0, "slope_nominal": s0,
            "r2_min": float(finite["r2"].min()), "r2_max": float(finite["r2"].max()),
            "r2_span": float(finite["r2"].max() - finite["r2"].min()),
            "slope_min": float(finite["slope"].min()), "slope_max": float(finite["slope"].max()),
            "worst_dlon": int(worst["dlon"]), "worst_dlat": int(worst["dlat"]),
            "worst_dsize": int(worst["dsize"]), "worst_r2": float(worst["r2"]),
            "n_boxes": int(len(finite)),
        })
    return pd.DataFrame(rows)


def _heatmaps(stats: pd.DataFrame, species: str, saver) -> None:
    """R² over the 3×3 location grid (size held at 0) and R² versus size."""
    shifts = list(boxes.SHIFT_DEG)
    relations = list(stats.groupby(["x", "y", "inverse_y"], sort=False).groups)
    datasets = [d for d in ("pd2024", "pd2010", "AeroCom") if d in set(stats["ppe_id"])]
    baselines = [b for b in boxes.BASELINES[species] if b in set(stats["baseline"])]
    for xcol, ycol, inverse_y in relations:
        rel = stats.loc[(stats["x"] == xcol) & (stats["y"] == ycol)]
        ylab = f"1/{ycol}" if inverse_y else ycol
        fig, axes = plt.subplots(
            len(baselines), len(datasets) + 1,
            figsize=(3.4 * (len(datasets) + 1), 3.3 * len(baselines)),
            squeeze=False,
        )
        for i, baseline in enumerate(baselines):
            for j, dataset in enumerate(datasets):
                ax = axes[i, j]
                grid = np.full((len(shifts), len(shifts)), np.nan)
                sub = rel.loc[(rel["baseline"] == baseline) & (rel["ppe_id"] == dataset)
                              & (rel["dsize"] == 0)]
                for _, row in sub.iterrows():
                    grid[shifts.index(int(row["dlat"])), shifts.index(int(row["dlon"]))] = row["r2"]
                im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", origin="lower")
                for ii in range(len(shifts)):
                    for jj in range(len(shifts)):
                        val = grid[ii, jj]
                        if np.isfinite(val):
                            ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                                    color="white" if val < 0.55 else "black", fontsize=8)
                ax.set_xticks(range(len(shifts)), [f"{s:+d}" for s in shifts])
                ax.set_yticks(range(len(shifts)), [f"{s:+d}" for s in shifts])
                ax.set_xlabel("Δlon (°)")
                ax.set_ylabel("Δlat (°)")
                ax.set_title(f"{baseline}\n{dataset}, size 0")
            ax = axes[i, -1]
            size = rel.loc[(rel["baseline"] == baseline) & (rel["dlon"] == 0) & (rel["dlat"] == 0)]
            for dataset in datasets:
                s = size.loc[size["ppe_id"] == dataset].sort_values("dsize")
                if s.empty:
                    continue
                ax.plot(s["dsize"], s["r2"], marker="o", label=dataset)
            ax.set_xlabel("Δsize (°)")
            ax.set_ylabel("R²")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{baseline}\nR² vs size (no shift)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        fig.colorbar(im, ax=axes[:, :-1].ravel().tolist(), shrink=0.6, label="R²", pad=0.02)
        fig.suptitle(f"{species}: {ylab} vs {xcol}")
        fig.subplots_adjust(top=0.88, wspace=0.45, hspace=0.45)
        saver(fig, f"sensitivity_{species}_{ycol}_vs_{xcol}.png")


# %%
print("\n--- Relation sensitivity ---")
all_stats = []
for species in ("ss", "bb"):
    stats = _relation_table(species)
    all_stats.append(stats)
    print(f"{species}: {len(stats)} fits")
stats_df = pd.concat(all_stats, ignore_index=True)
stats_df.to_csv(TABLE_DIR / "relation_fits.csv", index=False)
summary = _summary(stats_df)
summary.to_csv(TABLE_DIR / "relation_sensitivity.csv", index=False)
print("\nR² span across the ±5° boxes (nominal, min, max):")
show = summary.sort_values(["species", "baseline", "y", "ppe_id"])
print(show[["species", "ppe_id", "baseline", "x", "y", "r2_nominal", "r2_min",
            "r2_max", "r2_span", "worst_dlon", "worst_dlat", "worst_dsize"]].to_string(index=False))

for species in ("ss", "bb"):
    _heatmaps(stats_df.loc[stats_df["species"] == species], species, save_fig)

print("\nDone.")
print(f"Figures in: {FIGURE_DIR}")
print(f"Tables in:  {TABLE_DIR}")
