#!/usr/bin/env python3
"""Outflow meta-model R² sensitivity suite (Zhong Eq. 6 / Sci. Adv. Fig. S7).

Loads saved figure3_*.csv artifacts — does not re-run the full AAOD notebook.
Writes tables/figure3_outflow_sensitivity_summary.csv (+ optional PNG under figure/).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NB = Path(__file__).resolve().parent
ROOT = NB.parent
TABLE_DIR = ROOT / 'tables'
FIGURE_DIR = ROOT / 'figure'
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
KG_TO_G_DAY = 1000.0 * 86400.0  # kg m^-2 s^-1 -> g m^-2 day^-1

REGRESSION_TAU_MODELS = [
    "CAM5_CTRL2016",
    "EC-Earth3-AerChem-met2010_AP3-CTRL2019",
    "GFDL-AM4-met2010_AP3-CTRL",
    "TM5-met2010_AP3-CTRL2019",
]

# Checklist extras relative to a typical Zhong / Fig. S7B-style ensemble.
PAPERLIKE_DROP = [
    "HadGEM3-GA7.1_AP3-CTRL2016-PD",
    "GEOS-Chem-v11-01_AP3-CTRL2016-PD",
    "NorESM2-met2010_AP3-CTRL",
    "OsloCTM3v1.01-met2010_AP3-CTRL",
    "ECMWF-IFS-CY45R1-CAMS-CTRL-met2010_AP3-CTRL",
    "ECMWF-IFS-CY46R1-CAMS-CTRL-met2010_AP3-CTRL",
]


def _es_tau_mac_terms(e_kg_m2_s, tau_days, mac, convert_e=True):
    e = float(e_kg_m2_s) * KG_TO_G_DAY if convert_e else float(e_kg_m2_s)
    e_tau = e * float(tau_days)
    mac = float(mac)
    return e_tau * mac, e_tau, mac


def fit_eq6(df, e_col="E", convert_e=True):
    """OLS: AAOD = A*(E*τ*MAC) + B*(E*τ) + C*MAC + D."""
    rows = []
    for _, r in df.iterrows():
        x1, x2, x3 = _es_tau_mac_terms(r[e_col], r["tau"], r["MAC"], convert_e=convert_e)
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
    nmb = float(100.0 * np.mean(y_pred - y) / np.mean(y)) if np.mean(y) != 0 else np.nan
    resid = y_pred - y
    return {
        "n": int(len(df)),
        "r2": float(r2),
        "rmse": rmse,
        "nmb": nmb,
        "A": float(coeffs[0]),
        "B": float(coeffs[1]),
        "C": float(coeffs[2]),
        "D": float(coeffs[3]),
        "y": y,
        "y_pred": y_pred,
        "resid": resid,
        "models": df["model"].tolist(),
    }


def build_meta_df():
    pred = pd.read_csv(TABLE_DIR / "figure3_outflow_meta_predictions.csv")
    mrd = pd.read_csv(TABLE_DIR / "figure3_model_region_data.csv")
    reg = pd.read_csv(TABLE_DIR / "figure3_regression_data.csv")
    decomp = pd.read_csv(TABLE_DIR / "figure3_error_decomposition.csv")

    africa = mrd[mrd["region"] == "africa"].set_index("model")
    outflow = mrd[mrd["region"] == "outflow_af"].set_index("model")

    # Reconstruct inv_lt params from africa regression rows (Zhong Methods).
    af_reg = reg[reg["region"] == "africa"].dropna(subset=["inv_lifetime", "precip", "AE"])
    Xols = np.column_stack(
        [np.ones(len(af_reg)), af_reg["precip"].values, af_reg["AE"].values]
    )
    inv_coeffs, _, _, _ = np.linalg.lstsq(Xols, af_reg["inv_lifetime"].values, rcond=None)
    intercept, alpha, beta = [float(v) for v in inv_coeffs]

    raw_map = (
        decomp.loc[decomp["region"] == "africa"]
        .set_index("model")["lifetime_raw"]
        .to_dict()
    )

    rows = []
    for _, pr in pred.iterrows():
        model = pr["model"]
        if model not in africa.index or model not in outflow.index:
            continue
        a = africa.loc[model]
        o = outflow.loc[model]
        e = a["emi_BC_OA"]
        mac = a["MAC"]
        aaod_out = o["abs550aer"]
        if not all(np.isfinite(float(v)) for v in [e, mac, aaod_out]):
            continue

        tau_origin = pr["tau_origin"]
        if tau_origin == "box_budget" and np.isfinite(float(a["lifetime_BC_OA"])):
            tau = float(a["lifetime_BC_OA"])
        else:
            # Reconstruct regression τ̂
            precip, ae = a["precip"], a["AE"]
            if not (np.isfinite(float(precip)) and np.isfinite(float(ae))):
                continue
            inv_lt = intercept + alpha * float(precip) + beta * float(ae)
            if not np.isfinite(inv_lt) or inv_lt <= 0:
                continue
            tau = float(1.0 / inv_lt)
            tau_origin = "regression"

        raw_tau = raw_map.get(model, np.nan)
        rows.append(
            {
                "model": model,
                "AAOD_out": float(aaod_out),
                "E": float(e),
                "tau": float(tau),
                "MAC": float(mac),
                "tau_origin": tau_origin,
                "tau_raw": float(raw_tau) if np.isfinite(float(raw_tau)) else np.nan,
                "AAOD_meta_fit_saved": float(pr["AAOD_meta_fit"]),
            }
        )

    meta = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    meta.attrs["inv_lt"] = {
        "intercept": intercept,
        "alpha": alpha,
        "beta": beta,
        "n_fit": int(len(af_reg)),
    }
    return meta


def summarize(name, fit, notes=""):
    top = ""
    if fit is not None and "resid" in fit and fit["n"] >= 4:
        order = np.argsort(-np.abs(fit["resid"]))
        tops = [
            f"{fit['models'][i]}({fit['resid'][i]:+.4f})"
            for i in order[:3]
        ]
        top = "; ".join(tops)
    return {
        "experiment": name,
        "n": fit["n"] if fit else np.nan,
        "R2": fit["r2"] if fit else np.nan,
        "RMSE": fit["rmse"] if fit else np.nan,
        "NMB_pct": fit["nmb"] if fit else np.nan,
        "top_residuals": top,
        "notes": notes,
    }


def main():
    meta = build_meta_df()
    inv = meta.attrs["inv_lt"]
    print(
        f"Built meta_df n={len(meta)} "
        f"(box_budget={(meta.tau_origin=='box_budget').sum()}, "
        f"regression={(meta.tau_origin=='regression').sum()})"
    )
    print(
        f"  inv_lt africa: 1/tau = {inv['intercept']:.4f} "
        f"+ {inv['alpha']:.4f}*Pr + {inv['beta']:.4f}*AE "
        f"(n={inv['n_fit']})"
    )

    # Sanity: reproduce notebook baseline R²
    base = fit_eq6(meta)
    saved_r2 = 1 - np.sum((meta["AAOD_out"] - meta["AAOD_meta_fit_saved"]) ** 2) / np.sum(
        (meta["AAOD_out"] - meta["AAOD_out"].mean()) ** 2
    )
    print(f"  Baseline refit R²={base['r2']:.4f} (saved meta-fit R²={saved_r2:.4f})")

    results = []
    results.append(
        summarize(
            "0_baseline_all",
            base,
            "Full n=22 ensemble (18 box_budget + 4 regression-τ̂); notebook default",
        )
    )

    # 1. box_budget only
    bb = meta[meta["tau_origin"] == "box_budget"].copy()
    fit_bb = fit_eq6(bb)
    results.append(
        summarize(
            "1_box_budget_only",
            fit_bb,
            "Exclude 4 regression-τ̂ models (CAM5_CTRL2016, EC-Earth3, GFDL-AM4, TM5-met2010)",
        )
    )

    # 2. Paper-like subsets (exact Fig. S7B list unknown)
    drop_set = set(PAPERLIKE_DROP)
    paper_from_all = meta[~meta["model"].isin(drop_set)].copy()
    results.append(
        summarize(
            "2a_paperlike_drop_extras",
            fit_eq6(paper_from_all),
            "Best-effort: drop checklist extras (HadGEM, GEOS-Chem, NorESM2, "
            "OsloCTM3, CAMS-CY45/46); still includes τ̂ models",
        )
    )
    paper_bb = bb[~bb["model"].isin(drop_set)].copy()
    results.append(
        summarize(
            "2b_paperlike_box_budget",
            fit_eq6(paper_bb),
            "Best-effort paper-comparable: box_budget ∩ not in checklist extras "
            f"(n={len(paper_bb)}; exact S7B list unknown)",
        )
    )
    # Closest n≈17: box_budget minus OsloCTM3 (common high-residual / checklist name)
    n17 = bb[bb["model"] != "OsloCTM3v1.01-met2010_AP3-CTRL"].copy()
    results.append(
        summarize(
            "2c_n17_box_budget_drop_OsloCTM3",
            fit_eq6(n17),
            "Closest n≈17 match: all box_budget except OsloCTM3v1.01",
        )
    )

    # 3. Leave-one-out
    loo_focus = [
        "ECHAM6-SALSA_CTRL2016-PD",
        "OsloCTM3v1.01-met2010_AP3-CTRL",
    ]
    for m in loo_focus:
        sub = meta[meta["model"] != m].copy()
        results.append(
            summarize(
                f"3_LOO_{m}",
                fit_eq6(sub),
                f"Leave-one-out from full ensemble; dropped {m}",
            )
        )
    # LOO on box_budget for the same focus models
    for m in loo_focus:
        if m not in bb["model"].values:
            continue
        sub = bb[bb["model"] != m].copy()
        results.append(
            summarize(
                f"3b_LOO_box_{m}",
                fit_eq6(sub),
                f"Leave-one-out from box_budget-only; dropped {m}",
            )
        )
    # Full LOO ranking on baseline (report worst / best)
    loo_rows = []
    for m in meta["model"]:
        sub = meta[meta["model"] != m]
        f = fit_eq6(sub)
        loo_rows.append({"left_out": m, "R2": f["r2"], "RMSE": f["rmse"], "n": f["n"]})
    loo_df = pd.DataFrame(loo_rows).sort_values("R2", ascending=False)
    loo_csv = TABLE_DIR / "figure3_outflow_sensitivity_loo.csv"
    loo_df.to_csv(loo_csv, index=False)
    best = loo_df.iloc[0]
    worst = loo_df.iloc[-1]
    results.append(
        summarize(
            "3c_LOO_best_gain",
            fit_eq6(meta[meta["model"] != best["left_out"]]),
            f"Largest R² gain by dropping {best['left_out']} "
            f"(LOO table: {loo_csv.name})",
        )
    )
    results.append(
        {
            "experiment": "3d_LOO_summary",
            "n": len(meta) - 1,
            "R2": np.nan,
            "RMSE": np.nan,
            "NMB_pct": np.nan,
            "top_residuals": "",
            "notes": (
                f"Best LOO R²={best['R2']:.3f} drop {best['left_out']}; "
                f"worst LOO R²={worst['R2']:.3f} drop {worst['left_out']}; "
                f"full table {loo_csv.name}"
            ),
        }
    )

    # 4. E unit swap (should not change R²)
    fit_kg = fit_eq6(meta, convert_e=False)
    results.append(
        summarize(
            "4_E_units_kg_m2_s",
            fit_kg,
            f"Raw E [kg m^-2 s^-1] vs baseline g m^-2 d^-1; "
            f"ΔR²={fit_kg['r2'] - base['r2']:.2e} (expect ~0)",
        )
    )

    # 5. Target swap — not available without recompute
    results.append(
        {
            "experiment": "5_target_regional_mean",
            "n": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "NMB_pct": np.nan,
            "top_residuals": "",
            "notes": (
                "SKIPPED: saved CSVs only have sampled clear-sky outflow abs550aer; "
                "regional-mean AAOD would need full notebook recompute"
            ),
        }
    )

    # 6. BC+OA vs total E — not in AAOD saved artifacts
    results.append(
        {
            "experiment": "6_E_total_vs_BC_OA",
            "n": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "NMB_pct": np.nan,
            "top_residuals": "",
            "notes": (
                "SKIPPED: figure3_model_region_data.csv has emi_BC_OA only "
                "(no emi_total); AOD notebook uses total E + MEC separately"
            ),
        }
    )

    # 7. Unphysical raw τ for the 4 filtered models
    raw4 = meta.copy()
    for m in REGRESSION_TAU_MODELS:
        mask = raw4["model"] == m
        if not mask.any():
            continue
        rt = float(raw4.loc[mask, "tau_raw"].iloc[0])
        if np.isfinite(rt):
            raw4.loc[mask, "tau"] = rt
            raw4.loc[mask, "tau_origin"] = "raw_unphysical"
    fit_raw = fit_eq6(raw4)
    raw_vals = {
        m: float(meta.loc[meta.model == m, "tau_raw"].iloc[0])
        for m in REGRESSION_TAU_MODELS
        if m in meta.model.values
    }
    results.append(
        summarize(
            "7_raw_unphysical_tau_4models",
            fit_raw,
            "Replace τ̂ with raw box-budget τ for 4 filtered models: "
            + ", ".join(f"{k}={v:.1f}d" for k, v in raw_vals.items()),
        )
    )
    # Also: those 4 alone are underdetermined (n=4, p=4) — report residual if forced
    only4 = raw4[raw4["model"].isin(REGRESSION_TAU_MODELS)].copy()
    if len(only4) >= 4:
        # Use raw τ
        for m in REGRESSION_TAU_MODELS:
            mask = only4["model"] == m
            rt = float(meta.loc[meta.model == m, "tau_raw"].iloc[0])
            only4.loc[mask, "tau"] = rt
        fit4 = fit_eq6(only4)
        results.append(
            summarize(
                "7b_raw_tau_only4_models",
                fit4,
                "Diagnostic OLS on only the 4 filtered models with raw τ "
                "(n=p=4 → near-perfect in-sample R² is uninformative)",
            )
        )

    summary = pd.DataFrame(results)
    out_csv = TABLE_DIR / "figure3_outflow_sensitivity_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(summary[["experiment", "n", "R2", "RMSE", "NMB_pct"]].to_string(index=False))

    # Small summary figure
    plot_exps = summary.dropna(subset=["R2"]).copy()
    plot_exps = plot_exps[~plot_exps["experiment"].str.startswith("3d")]
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(plot_exps))
    ax.barh(y, plot_exps["R2"], color="#4C72B0", alpha=0.85)
    ax.axvline(0.97, color="crimson", ls="--", lw=1.2, label="Zhong Fig. S7A R²≈0.97")
    ax.axvline(base["r2"], color="gray", ls=":", lw=1.2, label=f"Baseline R²={base['r2']:.2f}")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_exps["experiment"], fontsize=8)
    ax.set_xlabel("R²")
    ax.set_xlim(0, 1.05)
    ax.set_title("Outflow meta-model R² sensitivity (Zhong Eq. 6)")
    ax.legend(fontsize=8, loc="lower right")
    for yi, r2, n in zip(y, plot_exps["R2"], plot_exps["n"]):
        ax.text(min(r2 + 0.02, 1.0), yi, f"n={int(n)}  R²={r2:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    out_png = FIGURE_DIR / "figure3_outflow_sensitivity_summary.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    # Write a results markdown snippet for the notebook
    md_path = TABLE_DIR / "figure3_outflow_sensitivity_results.md"
    lines = [
        "### Outflow meta-fit R² sensitivity — results",
        "",
        f"Generated by `outflow_meta_r2_sensitivity.py` from saved CSVs "
        f"(baseline refit R²={base['r2']:.3f}, n={base['n']}; "
        f"Zhong Fig. S7A target R²≈0.97, n≈17).",
        "",
        "| Experiment | n | R² | RMSE | NMB (%) | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, r in summary.iterrows():
        def fmt(v, nd=3):
            return "—" if pd.isna(v) else f"{v:.{nd}f}"

        lines.append(
            f"| `{r['experiment']}` | {fmt(r['n'], 0)} | {fmt(r['R2'])} | "
            f"{fmt(r['RMSE'], 5)} | {fmt(r['NMB_pct'], 1)} | {r['notes']} |"
        )
    lines += [
        "",
        "**Recommendation**",
        "",
        "- **Paper-comparable Fig. S7:** report `1_box_budget_only` and/or "
        "`2b_paperlike_box_budget` / `2c_n17_*` (physical post-agg τ only; "
        "exact S7B membership unknown).",
        "- **All-models reporting:** keep `0_baseline_all` (n=22) but state that "
        "4 regression-τ̂ models lower R² vs the paper’s physical-τ ensemble.",
        "- E unit choice does not affect R²; target / total-E swaps need recompute.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    print(f"Saved {md_path}")
    return summary


if __name__ == "__main__":
    main()
