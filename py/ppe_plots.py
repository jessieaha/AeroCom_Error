"""Plot helpers shared by the PPE analysis notebooks (SS, AAOD, dust).

Every panel draws one cloud and one fit per PPE (style from
``ppe_io.ppe_style``), optionally AeroCom models as open diamonds and
observation lines in the PPE colour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress

import ppe_io

AEROCOM_STYLE = dict(s=55, marker="D", facecolors="white", edgecolors="black",
                     linewidths=1.2, zorder=4)
AEROCOM_LABEL = "AeroCom AP3"


def show_inline(fig) -> bool:
    """Embed ``fig`` in the notebook output when a Jupyter frontend is running.

    Batch scripts use the Agg backend and get no display. Closing the figure
    afterwards is safe: ``display`` has already copied the PNG into the cell.
    """
    try:
        ip = get_ipython()  # noqa: F821
    except NameError:
        return False
    if ip is None or matplotlib.get_backend().lower() == "agg":
        return False
    from IPython.display import display
    display(fig)
    return True


class FigureSaver:
    """``saver(fig, name)`` writes to ``directory`` and, in a notebook, shows the figure."""

    def __init__(self, directory: Path, save: bool = True, dpi: int = 300):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.save = save
        self.dpi = dpi

    def __call__(self, fig, name: str) -> Path:
        path = self.directory / name
        if self.save:
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            print(f"  Saved {path}")
        showed = show_inline(fig)
        if self.save or not showed:
            plt.close(fig)
        return path


def fit_line(x, y, through_origin: bool = False) -> Tuple[float, float, float]:
    """(slope, intercept, R²). Through the origin R² is uncentered."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or not np.any(x):
        return np.nan, np.nan, np.nan
    if through_origin:
        slope = float(np.sum(x * y) / np.sum(x * x))
        return slope, 0.0, float(1.0 - np.sum((y - slope * x) ** 2) / np.sum(y ** 2))
    lr = linregress(x, y)
    return float(lr.slope), float(lr.intercept), float(lr.rvalue ** 2)


def fit_label(slope: float, intercept: float, r2: float) -> str:
    eq = f"y={slope:.3g}x" if intercept == 0 else f"y={slope:.3g}x{intercept:+.3g}"
    return f"{eq}, R²={r2:.2f}"


def xy(df: pd.DataFrame, region: str, xcol: str, ycol: str,
       inverse_y: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if df is None or df.empty or xcol not in df.columns or ycol not in df.columns:
        return np.array([]), np.array([])
    sub = df.loc[df["region"] == region, [xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    if inverse_y:
        sub = sub.loc[sub[ycol] > 0]
    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    return x, (1.0 / y if inverse_y else y)


def with_one_minus_ssa(df: Optional[pd.DataFrame], ssa_col: str = "SSA") -> Optional[pd.DataFrame]:
    if df is None or df.empty or ssa_col not in df.columns:
        return df
    df = df.copy()
    df[f"1-{ssa_col}"] = 1.0 - df[ssa_col]
    return df


def scatter_by_region(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    regions: Sequence[str],
    saver: Callable,
    filename: str,
    title: str,
    ppe_ids: Optional[Sequence[str]] = None,
    labels: Optional[Mapping[str, str]] = None,
    inverse_y: bool = False,
    through_origin: bool = False,
    aerocom_df: Optional[pd.DataFrame] = None,
    aerocom_ycol: Optional[str] = None,
    obs_x: Optional[Mapping[Tuple[str, str], Tuple[float, str]]] = None,
    obs_y: Optional[Mapping[Tuple[str, str], Tuple[float, str]]] = None,
    ylabel: Optional[str] = None,
    extra: Optional[Callable] = None,
    region_titles: Optional[Mapping[str, str]] = None,
):
    """One panel per region; PPE clouds + fits, AeroCom diamonds, obs lines.

    ``obs_x`` / ``obs_y`` map (ppe_id, region) -> (value, label); use
    ppe_id "AeroCom" for lines that belong to the AeroCom year.
    """
    labels = labels or {}
    ppe_ids = list(ppe_ids or df["ppe_id"].unique())
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 4.4), squeeze=False)
    for ax, region in zip(axes.ravel(), regions):
        n_drawn = 0
        for ppe_id in ppe_ids:
            st = ppe_io.ppe_style(ppe_id)
            x, y = xy(df.loc[df["ppe_id"] == ppe_id], region, xcol, ycol, inverse_y)
            if len(x) == 0:
                continue
            n_drawn += len(x)
            ax.scatter(x, y, s=16, alpha=0.45, color=st["color"], marker=st["marker"],
                       edgecolors="none", label=f"{st['label']} (n={len(x)})")
            if len(x) >= 3:
                s, i, r2 = fit_line(x, y, through_origin)
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 40)
                ax.plot(xs, s * xs + i, color=st["color"], lw=1.6,
                        label=f"{st['label']}: {fit_label(s, i, r2)}")
        if aerocom_df is not None and not aerocom_df.empty:
            xa, ya = xy(aerocom_df, region, xcol, aerocom_ycol or ycol, inverse_y)
            if len(xa):
                n_drawn += len(xa)
                ax.scatter(xa, ya, label=f"{AEROCOM_LABEL} (n={len(xa)})", **AEROCOM_STYLE)
                if len(xa) >= 3:
                    s, i, r2 = fit_line(xa, ya, through_origin)
                    xs = np.linspace(np.nanmin(xa), np.nanmax(xa), 40)
                    ax.plot(xs, s * xs + i, color="black", ls="--", lw=1.4,
                            label=f"{AEROCOM_LABEL}: {fit_label(s, i, r2)}")
        for lines, axline in ((obs_x, ax.axvline), (obs_y, ax.axhline)):
            for (pid, r), (val, lab) in (lines or {}).items():
                if r != region or not np.isfinite(val):
                    continue
                color = "gray" if pid == "AeroCom" else ppe_io.ppe_style(pid)["color"]
                axline(val, color=color, ls="--" if axline == ax.axvline else ":",
                       lw=1.2, label=lab)
        if extra is not None:
            extra(ax, region)
        shown = (region_titles or {}).get(region, region)
        ax.set_title(shown if n_drawn else f"{shown}: no data", fontsize=11)
        ax.set_xlabel(labels.get(xcol, xcol))
        ax.set_ylabel(ylabel or labels.get(f"inv_{ycol}" if inverse_y else ycol, ycol))
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    saver(fig, filename)


def density_by_region(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    regions: Sequence[str],
    saver: Callable,
    filename: str,
    title: str,
    ppe_ids: Sequence[str] = ("pd2024", "pd2010", "icon2025"),
    labels: Optional[Mapping[str, str]] = None,
    inverse_y: bool = False,
    aerocom_df: Optional[pd.DataFrame] = None,
    aerocom_ycol: Optional[str] = None,
    n_shades: int = 3,
):
    """Joint density of the three emission-lifetime PPEs, AeroCom as points.

    Each PPE is a few filled shades plus contour lines in its own colour.
    Marginal densities sit above and to the right of each panel.
    """
    from matplotlib.gridspec import GridSpec
    from scipy.stats import gaussian_kde

    labels = labels or {}
    ylabel = labels.get(f"inv_{ycol}" if inverse_y else ycol, ycol)
    xlabel = labels.get(xcol, xcol)
    fig = plt.figure(figsize=(5.4 * len(regions), 5.2))
    outer = GridSpec(1, len(regions), figure=fig, wspace=0.42)

    def _kde_grid(x, y, xlim, ylim, nx=70, ny=70):
        if len(x) < 8 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return None
        kde = gaussian_kde(np.vstack([x, y]))
        xx, yy = np.meshgrid(
            np.linspace(*xlim, nx),
            np.linspace(*ylim, ny),
        )
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        return xx, yy, zz

    for i, region in enumerate(regions):
        inner = outer[i].subgridspec(2, 2, height_ratios=(1.0, 4.0), width_ratios=(4.0, 1.05),
                                      hspace=0.05, wspace=0.05)
        ax_top = fig.add_subplot(inner[0, 0])
        ax = fig.add_subplot(inner[1, 0])
        ax_right = fig.add_subplot(inner[1, 1])
        clouds = []
        for ppe_id in ppe_ids:
            x, y = xy(df.loc[df["ppe_id"] == ppe_id], region, xcol, ycol, inverse_y)
            if len(x):
                clouds.append((ppe_id, x, y))
        if not clouds:
            ax.set_title(f"{region}: no PPE data")
            ax_top.axis("off")
            ax_right.axis("off")
            continue
        xa, ya = (np.array([]), np.array([]))
        if aerocom_df is not None and not aerocom_df.empty:
            xa, ya = xy(aerocom_df, region, xcol, aerocom_ycol or ycol, inverse_y)
        xs = np.concatenate([c[1] for c in clouds])
        ys = np.concatenate([c[2] for c in clouds])
        lim_x = np.concatenate([xs, xa]) if len(xa) else xs
        lim_y = np.concatenate([ys, ya]) if len(ya) else ys
        pad_x = 0.08 * (np.nanmax(lim_x) - np.nanmin(lim_x) or 1.0)
        pad_y = 0.08 * (np.nanmax(lim_y) - np.nanmin(lim_y) or 1.0)
        xlim = (np.nanmin(lim_x) - pad_x, np.nanmax(lim_x) + pad_x)
        ylim = (np.nanmin(lim_y) - pad_y, np.nanmax(lim_y) + pad_y)
        gx = np.linspace(*xlim, 200)
        gy = np.linspace(*ylim, 200)
        for ppe_id, x, y in clouds:
            st = ppe_io.ppe_style(ppe_id)
            color = st["color"]
            grid = _kde_grid(x, y, xlim, ylim)
            if grid is not None:
                xx, yy, zz = grid
                edges = np.linspace(zz.max() * 0.12, zz.max(), n_shades + 1)
                shades = [matplotlib.colors.to_rgba(color, a) for a in np.linspace(0.12, 0.42, n_shades)]
                ax.contourf(xx, yy, zz, levels=edges, colors=shades, zorder=1)
                ax.contour(xx, yy, zz, levels=edges, colors=color, linewidths=1.0, zorder=2)
            if len(x) >= 8 and np.nanstd(x) > 0:
                kx = gaussian_kde(x)(gx)
                ax_top.fill_between(gx, kx, color=color, alpha=0.2)
                ax_top.plot(gx, kx, color=color, lw=1.3, label=st["label"])
            if len(y) >= 8 and np.nanstd(y) > 0:
                ky = gaussian_kde(y)(gy)
                ax_right.fill_betweenx(gy, ky, color=color, alpha=0.2)
                ax_right.plot(ky, gy, color=color, lw=1.3)
            ax.plot([], [], color=color, lw=1.4, label=st["label"])
        if len(xa):
            ax.scatter(xa, ya, label=f"{AEROCOM_LABEL} (n={len(xa)})", **AEROCOM_STYLE)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax_top.set_xlim(xlim)
        ax_right.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax_top.set_title(region, fontsize=11)
        ax.grid(True, alpha=0.25, zorder=0)
        ax.legend(fontsize=7, loc="best")
        ax_top.tick_params(labelbottom=False, length=0)
        ax_right.tick_params(labelleft=False, length=0)
        for side in ("top", "right", "left"):
            ax_top.spines[side].set_visible(False)
        ax_top.spines["bottom"].set_visible(False)
        ax_top.set_yticks([])
        for side in ("top", "right", "bottom"):
            ax_right.spines[side].set_visible(False)
        ax_right.spines["left"].set_visible(False)
        ax_right.set_xticks([])
    fig.suptitle(title, y=0.98)
    fig.subplots_adjust(top=0.88, wspace=0.42)
    saver(fig, filename)


def lifetime_3d(
    df: pd.DataFrame,
    lifetime_col: str,
    regions: Sequence[str],
    saver: Callable,
    filename: str,
    title: str,
    labels: Optional[Mapping[str, str]] = None,
    x_col: str = "precip",
    ae_col: str = "AE",
    ppe_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """1/τ vs precip and AE with an OLS plane per PPE (PPEs without precip skipped).

    Returns the fitted coefficients (one row per PPE x region).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    labels = labels or {}
    ppe_ids = [p for p in (ppe_ids or df["ppe_id"].unique()) if ppe_io.has_precip(df, p)]
    rows = []
    if not ppe_ids:
        print(f"Skip {filename}: no PPE with precipitation")
        return pd.DataFrame()
    ncols = min(2, len(regions))
    nrows = int(np.ceil(len(regions) / ncols))
    fig = plt.figure(figsize=(6.5 * ncols, 5.2 * nrows))
    for idx, region in enumerate(regions):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        texts = []
        for ppe_id in ppe_ids:
            st = ppe_io.ppe_style(ppe_id)
            need = [x_col, ae_col, lifetime_col]
            sub = df.loc[(df["ppe_id"] == ppe_id) & (df["region"] == region), need]
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
            sub = sub.loc[sub[lifetime_col] > 0]
            if len(sub) < 4:
                continue
            x = sub[x_col].to_numpy(float)
            y = sub[ae_col].to_numpy(float)
            z = 1.0 / sub[lifetime_col].to_numpy(float)
            ax.scatter(x, y, z, s=14, color=st["color"], marker=st["marker"], alpha=0.55,
                       label=f"{st['label']} (n={len(x)})")
            A = np.column_stack([x, y, np.ones_like(x)])
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
            ss_tot = np.sum((z - z.mean()) ** 2)
            r2 = 1 - np.sum((z - A @ coef) ** 2) / ss_tot if ss_tot > 0 else np.nan
            xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10),
                                 np.linspace(y.min(), y.max(), 10))
            ax.plot_surface(xx, yy, coef[0] * xx + coef[1] * yy + coef[2], alpha=0.2,
                            color=st["color"], linewidth=0)
            texts.append(f"{st['label']}: {coef[0]:.3g}P {coef[1]:+.3g}AE {coef[2]:+.3g}, "
                         f"R²={r2:.2f}")
            rows.append({"ppe_id": ppe_id, "region": region, "lifetime": lifetime_col,
                         "alpha_pr": coef[0], "beta_ae": coef[1], "A": coef[2], "r2": r2,
                         "n": len(x)})
        if texts:
            ax.text2D(0.02, 0.98, "\n".join(texts), transform=ax.transAxes, va="top",
                      fontsize=7, bbox=dict(facecolor="white", alpha=0.8))
            ax.legend(fontsize=7, loc="lower left")
        ax.set_xlabel(labels.get(x_col, x_col))
        ax.set_ylabel(labels.get(ae_col, ae_col))
        ax.set_zlabel(labels.get(f"inv_{lifetime_col}", f"1/{lifetime_col}"))
        ax.set_title(region)
        ax.view_init(elev=15, azim=35)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    saver(fig, filename)
    return pd.DataFrame(rows)


def homogenization_scatter(result: dict, region: str, var_label: str, ppe_id: str,
                           obs_name: str, saver: Callable) -> None:
    if result.get("skipped") or len(result.get("x", [])) < 3:
        return
    st = ppe_io.ppe_style(ppe_id)
    x, y = result["x"], result["y"]
    a, b = result["a"], result["b"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=18, alpha=0.6, color=st["color"], edgecolors="k", linewidths=0.2)
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 40)
    ax.plot(xs, a * xs + b, "k-", lw=1.5, label=f"fit R$^2$={result['r2']:.2f}")
    ax.axvline(result["homogenized"], color="g", ls="--", label="homogenized")
    ax.axhline(result["raw_mean"], color="b", ls=":", label=f"{obs_name} raw")
    ax.set_xlabel(f"{st['label']} regional {var_label}")
    ax.set_ylabel(f"{st['label']} sampled at {obs_name} {var_label}")
    ax.set_title(f"{region}: {obs_name} homogenization ({var_label})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    saver(fig, f"{ppe_id}_homog_{var_label}_{region}.png")


def range_check(df: pd.DataFrame, cols: Sequence[str],
                expected: Optional[Mapping[str, Tuple[float, float]]] = None) -> pd.DataFrame:
    """Quantiles per PPE x column; prints and returns a tidy table."""
    expected = expected or {}
    rows = []
    for ppe_id, g in df.groupby("ppe_id"):
        for col in cols:
            if col not in g.columns:
                continue
            s = g[col].replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                rows.append({"ppe_id": ppe_id, "var": col, "n": 0})
                continue
            q = s.quantile([0.05, 0.5, 0.95])
            row = {"ppe_id": ppe_id, "var": col, "n": len(s), "q05": q.loc[0.05],
                   "median": q.loc[0.5], "q95": q.loc[0.95]}
            if col.startswith("lifetime"):
                row["inv_median_per_d"] = 1.0 / q.loc[0.5] if q.loc[0.5] > 0 else np.nan
            if col in expected:
                lo, hi = expected[col]
                row["frac_in_expected"] = float(((s >= lo) & (s <= hi)).mean())
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return out
