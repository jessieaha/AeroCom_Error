#!/usr/bin/env python3
"""MFC diagnostics for African outflow meta-model (Zhong Sci. Adv. Fig. S7).

Reports R², MFC, POLDER obs, gap, Default/EC means, and MFC term decomposition
for box-budget / paper-like ensembles. Uses saved figure3_*.csv artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NB = Path(__file__).resolve().parent
ROOT = NB.parent
TABLE_DIR = ROOT / 'tables'
FIGURE_DIR = ROOT / 'figure'
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
KG_TO_G_DAY = 1000.0 * 86400.0

PAPERLIKE_DROP = [
    "HadGEM3-GA7.1_AP3-CTRL2016-PD",
    "GEOS-Chem-v11-01_AP3-CTRL2016-PD",
    "NorESM2-met2010_AP3-CTRL",
    "OsloCTM3v1.01-met2010_AP3-CTRL",
    "ECMWF-IFS-CY45R1-CAMS-CTRL-met2010_AP3-CTRL",
    "ECMWF-IFS-CY46R1-CAMS-CTRL-met2010_AP3-CTRL",
]


def _terms(e_kg_m2_s, tau_days, mac):
    e_g = float(e_kg_m2_s) * KG_TO_G_DAY
    e_tau = e_g * float(tau_days)
    mac = float(mac)
    return e_tau * mac, e_tau, mac


def fit_eq6(df):
    rows = []
    for _, r in df.iterrows():
        x1, x2, x3 = _terms(r["E"], r["tau"], r["MAC"])
        rows.append((x1, x2, x3, float(r["AAOD_out"])))
    arr = np.asarray(rows, dtype=float)
    X = np.column_stack([arr[:, :3], np.ones(len(arr))])
    y = arr[:, 3]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    return {
        "n": int(len(df)),
        "r2": float(r2),
        "rmse": rmse,
        "A": float(coeffs[0]),
        "B": float(coeffs[1]),
        "C": float(coeffs[2]),
        "D": float(coeffs[3]),
        "y": y,
        "y_pred": y_pred,
        "models": df["model"].tolist(),
    }


def build_meta():
    pred = pd.read_csv(TABLE_DIR / "figure3_outflow_meta_predictions.csv")
    mrd = pd.read_csv(TABLE_DIR / "figure3_model_region_data.csv")
    africa = mrd[mrd["region"] == "africa"].set_index("model")
    outflow = mrd[mrd["region"] == "outflow_af"].set_index("model")
    polder = float(pred["AAOD_obs"].iloc[0])

    rows = []
    for model in sorted(set(africa.index) & set(outflow.index)):
        e = africa.loc[model, "emi_BC_OA"]
        mac = africa.loc[model, "MAC"]
        tau = africa.loc[model, "lifetime_BC_OA"]
        aaod = outflow.loc[model, "abs550aer"]
        aaod_src = africa.loc[model, "abs550aer"]
        if not all(np.isfinite(float(v)) for v in [e, mac, tau, aaod]):
            continue
        rows.append(
            {
                "model": model,
                "AAOD_out": float(aaod),
                "AAOD_src": float(aaod_src) if np.isfinite(float(aaod_src)) else np.nan,
                "E": float(e),
                "tau": float(tau),
                "MAC": float(mac),
            }
        )
    return pd.DataFrame(rows), polder


def mfc_row(name, fit, e_c, tau_c, mac_c, aaod_src_obs, polder, notes=""):
    A, B, C, D = fit["A"], fit["B"], fit["C"], fit["D"]
    x1, x2, x3 = _terms(e_c, tau_c, mac_c)
    mfc = A * x1 + B * x2 + C * x3 + D
    # EC: scale each model's E so source AAOD matches POLDER africa, then mean
    ec_vals = []
    for i, model in enumerate(fit["models"]):
        # recover E,tau,MAC from meta by re-reading — pass via closure in caller
        pass
    return {
        "ensemble": name,
        "n": fit["n"],
        "R2": fit["r2"],
        "RMSE": fit["rmse"],
        "POLDER_obs": polder,
        "MFC": mfc,
        "MFC_minus_obs": mfc - polder,
        "MFC_over_Africa_AAOD": mfc / aaod_src_obs if aaod_src_obs else np.nan,
        "term_A_X1": A * x1,
        "term_B_X2": B * x2,
        "term_C_X3": C * x3,
        "term_D": D,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "notes": notes,
    }


def mean_ec(fit, meta, e_c_unused, tau_c, mac_c, aaod_src_obs, A, B, C, D):
    vals = []
    meta_i = meta.set_index("model")
    for model in fit["models"]:
        r = meta_i.loc[model]
        if np.isfinite(r["AAOD_src"]) and r["AAOD_src"] != 0:
            e_ec = r["E"] * (aaod_src_obs / r["AAOD_src"])
        else:
            e_ec = r["E"]
        x1, x2, x3 = _terms(e_ec, r["tau"], r["MAC"])
        vals.append(A * x1 + B * x2 + C * x3 + D)
    return float(np.mean(vals)) if vals else np.nan


def main():
    meta, polder = build_meta()
    c = pd.read_csv(TABLE_DIR / "figure3_constrained_estimates.csv").query("region == 'africa'").iloc[0]
    e_c, tau_c, mac_c = float(c["E_c"]), float(c["tau_c"]), float(c["MAC_c"])
    aaod_src_obs = float(c["AAOD_obs"])
    mac_x = c.get("MAC_x_col", "1-SSA")

    ensembles = [
        ("box_budget_n18", meta, "Notebook default: physical box-budget τ only"),
        (
            "paperlike_box_budget",
            meta[~meta["model"].isin(PAPERLIKE_DROP)],
            "Best-effort paper-like ∩ box-budget",
        ),
        (
            "box_budget_drop_OsloCTM3",
            meta[meta["model"] != "OsloCTM3v1.01-met2010_AP3-CTRL"],
            "n≈17: box-budget minus OsloCTM3",
        ),
    ]

    # MAC_X_COL impact: rescale E_c so EτMAC stays = AAOD_obs for alternate MAC_c
    mac_alts = []
    for mac_alt in (0.88, 1.2, 1.5, 2.0):
        e_g = aaod_src_obs / (tau_c * mac_alt)
        e_alt = e_g / KG_TO_G_DAY
        mac_alts.append((mac_alt, e_alt))

    rows = []
    for name, df, notes in ensembles:
        if len(df) < 4:
            continue
        fit = fit_eq6(df)
        row = mfc_row(name, fit, e_c, tau_c, mac_c, aaod_src_obs, polder, notes)
        row["mean_Default"] = float(np.mean(fit["y"]))
        row["mean_EC"] = mean_ec(
            fit, meta, e_c, tau_c, mac_c, aaod_src_obs, fit["A"], fit["B"], fit["C"], fit["D"]
        )
        row["Africa_AAOD_obs"] = aaod_src_obs
        row["Africa_E_c"] = e_c
        row["Africa_tau_c"] = tau_c
        row["Africa_MAC_c"] = mac_c
        row["MAC_x_col"] = mac_x
        rows.append(row)

    # MAC sensitivity on box-budget fit
    fit0 = fit_eq6(meta)
    for mac_alt, e_alt in mac_alts:
        row = mfc_row(
            f"box_budget_MAC_c={mac_alt:.2f}",
            fit0,
            e_alt,
            tau_c,
            mac_alt,
            aaod_src_obs,
            polder,
            "E_c rescaled so E·τ·MAC = Africa AAOD_obs; same box-budget coeffs",
        )
        row["mean_Default"] = float(np.mean(fit0["y"]))
        row["mean_EC"] = np.nan
        row["Africa_AAOD_obs"] = aaod_src_obs
        row["Africa_E_c"] = e_alt
        row["Africa_tau_c"] = tau_c
        row["Africa_MAC_c"] = mac_alt
        row["MAC_x_col"] = f"alt_MAC_c={mac_alt}"
        rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = TABLE_DIR / "figure3_outflow_mfc_diagnostics.csv"
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    cols = [
        "ensemble",
        "n",
        "R2",
        "POLDER_obs",
        "MFC",
        "MFC_minus_obs",
        "mean_Default",
        "mean_EC",
        "MFC_over_Africa_AAOD",
        "term_A_X1",
        "term_B_X2",
        "term_C_X3",
        "term_D",
    ]
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Africa constraint cross-check note
    note = TABLE_DIR / "figure3_outflow_africa_constraint_crosscheck.md"
    note.write_text(
        f"""# Africa constrained values vs Zhong Sci. Adv. 2023 (Fig. S7 context)

## Notebook (current)
| Quantity | Value | Notes |
|----------|------:|-------|
| AAOD_obs (homogenized) | {aaod_src_obs:.5f} | Raw sampled africa ≈ 0.08185 → homogenized |
| MAC_c | {mac_c:.4f} m² g⁻¹ | From `MAC_X_COL={mac_x}`, INTERCEPT_0 as in cell 1 |
| tau_c | {tau_c:.3f} d | From 1/τ = A + αPr + βAE at GPCP/AE obs |
| E_c | {e_c:.3e} kg m⁻² s⁻¹ ({e_c * KG_TO_G_DAY:.4f} g m⁻² day⁻¹) | E_c = AAOD_obs / (τ_c · MAC_c) in consistent units |

## Paper
Sci. Adv. 2023 reports constrained E, τ, MAC in **figures** (Fig. 2–3 / SM) rather than
a single numeric table for Africa MAC_c / τ_c / E_c in extractable text. SM Text 2
states that plugging constrained (E_s, τ_s, MAC_s) into Eq. 6 yields an outflow AAOD
close to the POLDER dashed line in Fig. S7B.

Paper MAC constraint uses a linear MAC–SSA relationship from AeroCom (not necessarily
through-origin `MAC = A(1−SSA)`). Notebook `MAC_X_COL='1-SSA'` with `INTERCEPT_0=True`
can yield a lower MAC_c (~0.88) than an intercept-enabled SSA fit; however MFC
diagnostics show rescaling MAC_c while keeping E·τ·MAC = AAOD_obs **does not** close
the MFC–POLDER gap (see `figure3_outflow_mfc_diagnostics.csv` MAC_c rows).

## Outflow POLDER after bbox fix
Notebook `REGIONS['outflow_af']` now matches `cameo_toolbox` / AOD notebook:
`lon=(350, 8)`, `lat=(-15, 3)`. Sampled POLDER AAOD ≈ **{polder:.5f}** (was 0.076 with
the old wider box). Model Default was already on the Nature box via cameo_toolbox.

## Reporting
- Obs line: raw sampled outflow AAOD (no homogenization) — paper-consistent.
- MFC residual below POLDER reflects AeroCom underestimation of outflow AAOD
  embedded in Eq. 6 coefficients (mean Default ≪ POLDER).
- Prefer reporting EC alongside MFC (Nature Fig. 5 style); do not inflate MFC with
  unphysical τ.
"""
    )
    print(f"Wrote {note}")
    return out


if __name__ == "__main__":
    main()
