"""Map figure for a dashboard field, with the drawn polygon and a colorbar below."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

import PPE_set as cfg  # noqa: E402
from dashboard import data as D  # noqa: E402

LETTERS = "JFMAMJJASOND"
STOPS = [
    (0.0, "#dbd6d6"),
    (0.2, "#9d579a"),
    (0.4, "#009bd5"),
    (0.6, "#5ead35"),
    (0.8, "#fabb00"),
    (1.0, "#ce1f5e"),
]
CMAP = LinearSegmentedColormap.from_list("dashboard", STOPS)
COAST = Path(__file__).resolve().parent / "static" / "geo" / "coastline.geojson"


def _plain(text: str) -> str:
    return str(text).replace("$", "").replace("{", "").replace("}", "").replace("_", " ")


def _model_label(dataset: str, model: Optional[str]) -> str:
    if dataset in ("pd2024", "pd2010", "icon2025"):
        return cfg.get_run(dataset)["label"]
    if dataset == "aerocom":
        return model or "AeroCom AP3"
    if dataset == "spexone":
        return "SPEXone"
    if dataset == "polder":
        return "POLDER-GRASP"
    return dataset


def _year_months(dataset: str) -> List[tuple]:
    if dataset in ("pd2024", "icon2025", "pd2010"):
        return [(int(y), int(m)) for y, m in cfg.get_run(dataset)["year_months"]]
    if dataset in ("polder", "aerocom"):
        return [(2010, m) for m in range(1, 13)]
    # SPEXone archive covers Jul 2024 through Dec 2025.
    pairs = [(2024, m) for m in range(7, 13)] + [(2025, m) for m in range(1, 13)]
    return pairs


def month_caption(dataset: str, months: Sequence[int]) -> str:
    """Initial letters grouped by year, e.g. 'A 2024, JJ 2025'."""
    chosen = {int(m) for m in months}
    by_year = {}
    for year, month in _year_months(dataset):
        if month in chosen:
            by_year.setdefault(year, []).append(month)
    if not by_year:
        letters = "".join(LETTERS[m - 1] for m in sorted(chosen))
        return letters or "annual"
    parts = []
    for year in sorted(by_year):
        months_y = sorted(set(by_year[year]))
        run = [months_y[0]]
        runs = [run]
        for month in months_y[1:]:
            if month == run[-1] + 1:
                run.append(month)
            else:
                run = [month]
                runs.append(run)
        for group in runs:
            letters = "".join(LETTERS[m - 1] for m in group)
            parts.append(f"{letters} {year}")
    return ", ".join(parts)


def _signed_ring(ring: Sequence[Sequence[float]]) -> np.ndarray:
    verts = np.asarray(ring, dtype=float)
    if verts.shape[0] < 2:
        return verts.reshape(0, 2)
    lon = np.where(verts[:, 0] > 180.0, verts[:, 0] - 360.0, verts[:, 0])
    # Keep consecutive vertices continuous so a box across 0° stays one outline.
    for i in range(1, len(lon)):
        while lon[i] - lon[i - 1] > 180.0:
            lon[i] -= 360.0
        while lon[i] - lon[i - 1] < -180.0:
            lon[i] += 360.0
    return np.column_stack([lon, verts[:, 1]])


def _plot_ring(ax, ring, color: str, name: str = "") -> None:
    verts = _signed_ring(ring)
    if verts.shape[0] < 2:
        return
    # Break a segment that still jumps across the date line.
    xs, ys = [], []
    for i, (lon, lat) in enumerate(verts):
        if i and abs(lon - verts[i - 1, 0]) > 180:
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(lon)
        ys.append(lat)
    ax.plot(xs, ys, color=color, lw=1.4, solid_capstyle="round", zorder=4, label=name or None)


def _coast(ax) -> None:
    if not COAST.is_file():
        return
    fc = json.loads(COAST.read_text())
    for feat in fc.get("features", []):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        kind = geom.get("type")
        lines = coords if kind == "MultiLineString" else [coords] if kind == "LineString" else []
        for line in lines:
            if not line:
                continue
            arr = np.asarray(line, dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], color="#4a5568", lw=0.35, zorder=3)


def render_figure(
    dataset: str,
    var: str,
    months: Sequence[int],
    rings: Sequence[dict],
    vmin: float,
    vmax: float,
    ens: Optional[int] = None,
    model: Optional[str] = None,
) -> bytes:
    payload = D.field_payload(dataset, var, months=list(months), ens=ens, model=model)
    raw = base64.b64decode(payload["values_b64"])
    data = np.frombuffer(raw, dtype="<f4").reshape(payload["shape"]).astype(float)
    lon = np.asarray(payload["lon"], dtype=float)
    lat = np.asarray(payload["lat"], dtype=float)
    # Matplotlib wants latitude increasing upward.
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[::-1]
    var_label = _plain(payload.get("label") or var)
    model_label = _model_label(dataset, model)
    when = month_caption(dataset, months)
    legend = f"{var_label} — {model_label}"

    fig = plt.figure(figsize=(8.2, 4.6))
    ax = fig.add_axes([0.06, 0.20, 0.90, 0.70])
    cax = fig.add_axes([0.18, 0.07, 0.64, 0.035])
    norm = Normalize(vmin=float(vmin), vmax=float(vmax))
    mesh = ax.pcolormesh(lon, lat, data, cmap=CMAP, norm=norm, shading="auto")
    _coast(ax)
    for item in rings:
        _plot_ring(ax, item.get("ring") or [], item.get("color") or "#ffffff", item.get("name") or "")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("auto")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{legend}\n{when}", fontsize=11)
    ax.set_facecolor("#0a0e14")
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
    cbar.ax.set_xlabel(f"{legend}    {when}", fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return buf.getvalue()
