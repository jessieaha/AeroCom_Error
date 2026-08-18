# AOD error attribution — gap summary vs Zhong et al. 2022 Table 1

Ensembles: full=28 models, paper=17 models.
Gap threshold: |gap_pct| > 20.0%.

## Out of scope (no data in this repo)

- Fig. 3 emission-inventory comparison (GFED/QFED/FEER/GFAS)
- Fig. 6 ECHAM-HAM EC/MFC climate-model rerun
- AERONET cross-check of regional AOD/AE

## Known methodological caveats

- GPCP SE Asia / Eastern Siberia precip in our boxes is much wetter than
  Zhong Table 1 (our land-masked GPCP vs their regional means). This
  propagates into constrained τ and E.
- Homogenized AOD for SE Asia / Boreal NA can diverge strongly from
  Table 1 when sampling/coverage differs from the paper.
- LOO validation uses per-region leave-one-model-out, pooled across the
  five SOURCE_REGIONS; metrics report Pearson R (as in Fig. 2c/d).

## Ensemble: full

| region | variable | computed | paper | gap | gap_pct |
| --- | --- | --- | --- | --- | --- |
| africa | E (10^-11) | 13.99 | 27.9 | -13.91 | -49.9% |
| africa | tau | 9.126 | 4 | 5.126 | 128.2% |
| amazon | E (10^-11) | 11.77 | 18.2 | -6.427 | -35.3% |
| amazon | tau | 7.04 | 4.3 | 2.74 | 63.7% |
| boreal_na | AOD | 0.4874 | 0.16 | 0.3274 | 204.7% |
| boreal_na | E (10^-11) | 20.47 | 10.3 | 10.17 | 98.7% |
| boreal_na | precip | 2.21 | 1.8 | 0.4101 | 22.8% |
| boreal_na | tau | 4.776 | 3 | 1.776 | 59.2% |
| eastern_siberia | E (10^-11) | 13.25 | 8.3 | 4.953 | 59.7% |
| eastern_siberia | precip | 2.804 | 0.6 | 2.204 | 367.4% |
| se_asia | AE | 1.512 | 1.2 | 0.3115 | 26.0% |
| se_asia | AOD | 0.2711 | 0.88 | -0.6089 | -69.2% |
| se_asia | E (10^-11) | 25.1 | 47.6 | -22.5 | -47.3% |
| se_asia | precip | 7.325 | 0.8 | 6.525 | 815.7% |
| se_asia | tau | 2.698 | 3.9 | -1.202 | -30.8% |

- LOO MEC: R=0.160 (paper 0.72), NMB=-0.8%, RMSE=1.397, n=32
- LOO inv_lifetime: R=0.341 (paper 0.78), NMB=-5.3%, RMSE=0.1447, n=32
- Fig.4 mean |error| %: E=33.6 / tau=18.4 / MEC=12.1 / cross=35.8 (paper 38/22/27/13)
- Outflow meta-model: R2=0.958, NMB=0.0%, RMSE=0.0147, n=6

## Ensemble: paper

| region | variable | computed | paper | gap | gap_pct |
| --- | --- | --- | --- | --- | --- |
| africa | E (10^-11) | 13.49 | 27.9 | -14.41 | -51.6% |
| africa | tau | 9.085 | 4 | 5.085 | 127.1% |
| amazon | E (10^-11) | 11.74 | 18.2 | -6.465 | -35.5% |
| amazon | tau | 7.038 | 4.3 | 2.738 | 63.7% |
| boreal_na | AOD | 0.4613 | 0.16 | 0.3013 | 188.3% |
| boreal_na | E (10^-11) | 19.47 | 10.3 | 9.174 | 89.1% |
| boreal_na | precip | 2.21 | 1.8 | 0.4101 | 22.8% |
| boreal_na | tau | 4.775 | 3 | 1.775 | 59.2% |
| eastern_siberia | AOD | 0.2533 | 0.21 | 0.0433 | 20.6% |
| eastern_siberia | E (10^-11) | 14.65 | 8.3 | 6.349 | 76.5% |
| eastern_siberia | precip | 2.804 | 0.6 | 2.204 | 367.4% |
| se_asia | AOD | 0.2553 | 0.88 | -0.6247 | -71.0% |
| se_asia | E (10^-11) | 23.33 | 47.6 | -24.27 | -51.0% |
| se_asia | precip | 7.325 | 0.8 | 6.525 | 815.7% |
| se_asia | tau | 2.678 | 3.9 | -1.222 | -31.3% |

- LOO MEC: R=0.160 (paper 0.72), NMB=-0.8%, RMSE=1.397, n=32
- LOO inv_lifetime: R=0.341 (paper 0.78), NMB=-5.3%, RMSE=0.1447, n=32
- Fig.4 mean |error| %: E=32.0 / tau=16.4 / MEC=10.7 / cross=40.9 (paper 38/22/27/13)
