# AeroCom and PPE error attribution

Analysis of AeroCom Phase III (2010) and the Pace ECHAM-HAM perturbed-parameter ensembles (2010 and Aug 2024–Jul 2025). The project compares regional aerosol optical depth, absorption, lifetime and mass extinction or absorption against SPEXone, POLDER-GRASP and GPCP.

Region geometry and the land/sea mask live in one place: [`py/regions.py`](py/regions.py). Each box has `surface_type` (`land`, `ocean` or `all`). A notebook can pass a different `region_defs` dict into `build_regional` and recompute without editing that file.

## Data flow

Three PPEs plus AeroCom feed the same regional tables and BB / SS / DU notebooks.
Figure inventory by aerosol type: [`docs/FIGURE_MATRIX_BB_SS_DU.md`](docs/FIGURE_MATRIX_BB_SS_DU.md).

```mermaid
flowchart TB
  subgraph sources [Read-only sources]
    echam24["ECHAM PPE 2024-25<br/>YBHATTI / PPE_Experiments"]
    echam10["ECHAM PPE 2010<br/>YBHATTI / 2010_PPE/.../PD"]
    icon25["ICON-HAM PPE 2024-25<br/>hjia / ICON-HAM-PPE_24_25/PD"]
    aero["AeroCom AP3 monthly<br/>Data/AP3_processed_monthly"]
  end
  subgraph extract [Extract]
    ex24["ppe_extract_monthly<br/>pd2024"]
    ex10["ppe_extract_monthly<br/>pd2010"]
    exIcon["icon_regional.py<br/>native R2B4 cell weights"]
  end
  subgraph tables [Regional tables]
    reg["regions.py boxes"]
    build["regional_aggregates / box_sensitivity"]
    pq["ppe_regional_*.parquet<br/>+ aerocom overlay"]
  end
  subgraph figs [Standard figures — no homog]
    bb["BB: figure/PPE_bb"]
    ss["SS: figure/PPE_ss"]
    du["DU: figure/PPE_du"]
    dep["dry/wet: figure/dep"]
  end
  echam24 --> ex24
  echam10 --> ex10
  icon25 --> exIcon
  ex24 --> build
  ex10 --> build
  exIcon --> build
  aero --> build
  reg --> build
  build --> pq
  pq --> bb
  pq --> ss
  pq --> du
  aero --> dep
```

| Ensemble | `ppe_id` | Root | In plots today |
| --- | --- | --- | --- |
| ECHAM-HAM 2024-25 (YBHATTI) | `pd2024` | `…/Pace_PPE_Output/PPE_Experiments` | yes |
| ECHAM-HAM 2010 (YBHATTI) | `pd2010` | `…/2010_PPE/Collaboration/Processing/PD` | yes |
| ICON-HAM 2024-25 (hjia) | `icon2025` | `…/ICON-HAM-PPE_24_25/PD` | yes (native-grid regional means) |
| AeroCom AP3 | — | `Data/AP3_processed_monthly/` | diamonds on combined panels |

| Step | Where |
| --- | --- |
| PPE source files | Read-only ensembles above. Monthly 2-D extracts → `/scratch-shared/jzhang1/PPE/<ppe_id>/`. |
| AeroCom source files | `Data/AP3_processed_monthly/` |
| Build | `python py/ppe_regional_aggregates.py` (ECHAM); `python py/icon_regional.py` (ICON-HAM); AeroCom overlay via `python py/box_sensitivity.py` |
| PPE tables | `Data/PPE_regional/ppe_regional_{monthly,seasonal}.parquet` |
| AeroCom wide table | `tables/aerocom_regional_seasonal.parquet` |
| Legacy long AeroCom table | `python py/build_regional_aggregates.py --legacy` |
| Notebooks | `notebooks/AOD_SS_ERROR_PPE.py`, `AAOD_error_attribution_PPE.py`, `AOD_DUST_ERROR_PPE.py`, and the AeroCom attribution scripts |
| Figure matrix | [`docs/FIGURE_MATRIX_BB_SS_DU.md`](docs/FIGURE_MATRIX_BB_SS_DU.md) |

`py/ppe_regional_aggregates.py` re-exports `regional_aggregates`. Quality filter `"ppe"` (AOD > 0.01) is the default. Pass `quality="aerocom"` for the stricter AOD > 0.05 filter.

Lifetimes are burden divided by emission after the regional mean. Sea salt plus sulfate uses primary sea salt and sulfate plus half of the SO2 and DMS emissions, converted to sulfate mass (`ppe_io.add_lifetime_drivers`). The PPE extract has no deposition fluxes. AeroCom deposition lifetimes are drawn only when those columns exist.

## Maps

Boxed regional maps use `cameo_toolbox.fake_uba_map` through `plot_boxed_field` (Plate Carrée, so the rectangles stay straight). Maps with no analysis box use `uba_map_flex(..., projection_type="curved")` through `plot_curved_field`.

## Notebook efficiency

The older attribution scripts used to reopen every model file and repeat time handling, masks and regional means in each file. Those helpers now call `aerocom_data`, `notebook_setup.aggregate_region` and `ppe_io`. The PPE notebooks read the parquet and only plot. An older notebook can do the same: call `build_regional` once, then keep its attribution algebra.
