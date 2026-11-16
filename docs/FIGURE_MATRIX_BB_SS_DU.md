# Standard figure matrix — BB / SS / DU

Summary of analysis figures by aerosol type. **Standard pack** excludes homogenization
diagnostics (`*_homog_*`). Dimensionless quantities use **`[-]`** (not `(1)`).

## Ensembles and observations

| ID | Model / campaign | Source root | Period | Obs | Status in plots |
| --- | --- | --- | --- | --- | --- |
| `pd2024` | ECHAM-HAM PPE (YBHATTI) | `/gpfs/work3/0/prjs1474/ybhatti/Pace_PPE_Output/PPE_Experiments` | Aug 2024 – Jul 2025 | SPEXone | **In** regional parquet + combined figures |
| `pd2010` | ECHAM-HAM PPE 2010 (YBHATTI) | `/gpfs/work3/0/prjs1474/ybhatti/Pace_PPE_Output/2010_PPE/Collaboration/Processing/PD` | 2010 | POLDER-GRASP | **In** (no precip stream yet) |
| `icon2025` | ICON-HAM PPE (hjia) | `/projects/0/prjs1474/shared/tmp/for_jessie/ICON-HAM-PPE_24_25/PD` | Jun 2024 – Aug 2025 (15 monthly steps, R2B4) | SPEXone period, no T63 collocation | **In** regional tables from `py/icon_regional.py` (native cell weights) |
| AeroCom | AP3 CTRL (~27 models) | `Data/AP3_processed_monthly/` | 2010 | POLDER / GPCP | Open diamonds on combined panels |

Plot markers (from `PPE_set.PPE_RUNS` / `ppe_plots.AEROCOM_STYLE`):

- ECHAM-HAM 2024-25 (YBHATTI) — red circles  
- ECHAM-HAM 2010 (YBHATTI) — blue squares  
- ICON-HAM 2024-25 (hjia) — green triangles  
- AeroCom AP3 — white diamonds with black edge  

ICON BB/DU Angstrom is the archived **440–670 nm** exponent, stored in `AE_440_550`. Sea salt uses **550–865 nm**. ICON has dry deposition only (no wetdep), so dry/total bars stay AeroCom. ICON has no dust absorption, so `MAC_du` / dust SSA points are absent.

---

## Standard figure pack (no homogenization)

Columns = aerosol type. Paths are under `figure/PPE_bb`, `figure/PPE_ss`, `figure/PPE_du`
(and `figure/dep` for dry/wet). “combined_*” panels draw **PPE + AeroCom**.

| Relation | BB (BC+OA) | SS | DU | Notes |
| --- | --- | --- | --- | --- |
| **1−SSA vs MAC** | `combined_MAC_vs_1SSA.png` | `ppe_1SSA_vs_MAC.png` | `ppe_MAC_du_vs_1SSA_du.png` | Through-origin fit. ICON has no dust absorption |
| **MEC vs AE** | `combined_MEC_vs_AE.png` | `ppe_MEC_ss_vs_AE.png` | `combined_MEC_du_vs_AE.png` | ICON BB/DU AE is 440–670 nm |
| **1/τ (emission) vs AE** | `combined_inv_lifetime_bcoa_vs_AE.png` | `ppe_inv_lifetime_ss_vs_AE.png` | `combined_inv_lifetime_du_vs_AE.png` | AeroCom diamonds are load/emission. Deposition overlay is `combined_inv_lifetime_bcoa_dep_vs_AE.png` |
| **1/τ vs precip** *(optional)* | `combined_inv_lifetime_bcoa_vs_precip.png` | `ppe_inv_lifetime_ss_vs_precip.png` | `combined_inv_lifetime_du_vs_precip.png` | `pd2024` (+ AeroCom); skip `pd2010` until precip |
| **1/τ vs precip + AE** *(optional)* | `ppe_3d_inv_lifetime_BC_OA_vs_precip_AE.png` | `ppe_3d_inv_lifetime_ss_vs_precip_AE.png` | `ppe_3d_inv_lifetime_du_vs_precip_AE.png` | 3-D OLS plane |
| **SSA vs rBC** *(BB)* | `combined_SSA_vs_rBC.png` | — | — | BB only |
| **Dry / total deposition** | `figure/dep/drywet_fraction_bc_oa.png` | `figure/dep/drywet_fraction_ss.png` | `figure/dep/drywet_fraction_du.png` | AeroCom (+ ICON later: has `ddep_*`) |

Also useful but not in the core pack: SSA vs MEC (`ppe_SSA_vs_MEC*.png`), marine SS+SO4 lifetime, dust `ustar_acrit` emission, box-sensitivity under `figure/PPE_box_sensitivity/`.

### Explicitly **out** of the standard pack

- All `*_homog_*` SPEXone/POLDER homogenization diagnostics  
- Obs-only regional point maps (`*_SPEXone_*`, `*_POLDER_*`) unless used as map insets  
- Nested RH/stress lifetime constraint (SS exploration only)

---

## Per-type notebooks (regenerate pack)

| Type | Script | Figure dir | Regions |
| --- | --- | --- | --- |
| BB | `notebooks/AAOD_error_attribution_PPE.py` | `figure/PPE_bb` | africa, amazon, outflow_af |
| SS | `notebooks/AOD_SS_ERROR_PPE.py` | `figure/PPE_ss` | SAO_SOND, SPO_DJF (remote South Pacific, formerly remote_SO_DJF) |
| DU | `notebooks/AOD_DUST_ERROR_PPE.py` | `figure/PPE_du` | sahara, australia, atacama, outflow_sahara |
| Dry/wet | `notebooks/drywet_deposition_fraction.py` | `figure/dep` | AeroCom regional |

Rebuild tables if VAR_MAP / members change:

```bash
python py/ppe_extract_monthly.py --ppe pd2024   # and pd2010 when needed
python py/ppe_regional_aggregates.py --ppe pd2024 pd2010
python py/icon_regional.py                      # ICON-HAM native-grid regional means
python py/box_sensitivity.py                    # AeroCom overlay parquet
.venv-1/bin/python notebooks/AAOD_error_attribution_PPE.py
.venv-1/bin/python notebooks/AOD_SS_ERROR_PPE.py
.venv-1/bin/python notebooks/AOD_DUST_ERROR_PPE.py
.venv-1/bin/python notebooks/drywet_deposition_fraction.py
```

---

## ICON-HAM PPE (`icon2025`)

`py/icon_regional.py` area-weights native R2B4 cells inside each `regions.py` box
(spherical triangle area, Natural Earth land/ocean mask). Outputs:

- `Data/PPE_regional/icon2025_regional_monthly.parquet`
- `Data/PPE_regional/icon2025_regional_seasonal.parquet`

`ppe_io.load_regional` appends those tables when `icon2025` is requested. The
ECHAM scratch ensemble is unchanged. Homogenization is skipped for ICON because
there is no T63 gridded ensemble to sample at SPEXone/POLDER points.

---

## Axis unit convention

| Quantity | Label form |
| --- | --- |
| SSA, AE, fractions, AOD, AAOD | `SSA [-]`, `AE 550–865 [-]`, `AOD$_{550}$ [-]` |
| MAC / MEC | `MAC (m$^{2}$ g$^{-1}$)` |
| Lifetime | `$\tau$ (day)`, `1/$\tau$ (day$^{-1}$)` |
| Precipitation | `mm day$^{-1}$` |
| Stress | `Pa` |

---

## Related

- Data-flow diagram: [README.md](../README.md#data-flow)  
- Region registry: `py/regions.py`  
- Shared scatters: `py/ppe_plots.py` (`scatter_by_region` marks each PPE + AeroCom)
