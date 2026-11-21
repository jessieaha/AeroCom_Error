"""Local FastAPI app for the AeroCom / PPE map dashboard.

Run on the login node (bound to localhost only):

    cd /home/jzhang1/AeroCom_Error
    PYTHONPATH=py:. .venv-1/bin/python -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8765

Then from your laptop:

    ssh -L 8765:127.0.0.1:8765 <user>@<host>
    # open http://127.0.0.1:8765
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "py"), str(ROOT)]

from dashboard import data as D  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="AeroCom / PPE map dashboard", version="0.1")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/figures", StaticFiles(directory=ROOT / "figure"), name="figures")


class RegionMeanRequest(BaseModel):
    dataset: str
    var: str
    ring: List[List[float]] = Field(..., description="[[lon, lat], ...] in degrees")
    months: Optional[List[int]] = None
    surface_type: str = "all"
    model: Optional[str] = None
    ens: Optional[int] = None


class PolygonRequest(BaseModel):
    name: str
    ring: List[List[float]]
    surface_type: str = "all"


class AnnualRequest(BaseModel):
    dataset: str
    var: str
    ring: List[List[float]]
    surface_type: str = "all"
    ens: Optional[int] = None


class JobRequest(BaseModel):
    slug: str
    where: str = "slurm"
    mem_gb: int = 32


class FigureRequest(BaseModel):
    dataset: str
    var: str
    months: List[int]
    rings: List[dict] = Field(default_factory=list)
    vmin: float
    vmax: float
    ens: Optional[int] = None
    model: Optional[str] = None


@app.post("/api/figure.png")
def figure_png(body: FigureRequest):
    from dashboard.figure_export import render_figure
    try:
        png = render_figure(
            body.dataset, body.var, body.months, body.rings,
            body.vmin, body.vmax, ens=body.ens, model=body.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=png, media_type="image/png")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/plots")
def plots_page():
    return FileResponse(STATIC / "plots.html")


@app.get("/validate")
def validate_page():
    return FileResponse(STATIC / "validate.html")


class CompareRequest(BaseModel):
    var: str
    month: int
    region: str = ""
    ring: Optional[List[List[float]]] = None
    surface_type: str = "all"


@app.post("/api/compare")
def compare_month(body: CompareRequest):
    from dashboard.compare import compare_month as _compare
    try:
        return _compare(body.var, body.month, body.region, body.ring, body.surface_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/validate")
def validate_json(
    region: str = Query(...),
    var: str = Query("od550aer"),
    month: int = Query(1),
):
    from dashboard.compare import validation
    try:
        return validation(region, var, month)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


_PLOT_LOCK = threading.Lock()


@app.get("/api/notebook_plot.png")
def notebook_plot(
    kind: str = Query(...),
    species: str = Query(...),
    style: str = Query("scatter"),
):
    """Notebook PNG already stored under figure/PPE_*, when this plot matches one."""
    from dashboard import aggplots
    path = aggplots.notebook_figure(species, kind, style)
    if path is None:
        raise HTTPException(status_code=404, detail="no saved notebook figure for this plot")
    return FileResponse(path, media_type="image/png", headers={"X-Plot-Note": str(path)})


@app.get("/api/plot_columns")
def plot_columns(species: str = Query("ss")):
    from dashboard import aggplots
    return {"columns": aggplots.list_plot_columns(species)}


@app.get("/api/figures")
def figure_list(species: str = Query("ss")):
    from dashboard import aggplots
    return {"species": species, "figures": aggplots.list_saved_figures(species)}


@app.get("/api/plot.png")
def plot_png(
    kind: str = Query("xy", description="preset kind, or xy when x and y are set"),
    species: str = Query(..., description="bb | ss | du | total | marine"),
    regions: str = Query(..., description="comma-separated region names"),
    months: Optional[str] = Query(None, description="comma-separated months 1-12"),
    style: str = Query("scatter"),
    lifetime_source: str = Query("emission"),
    timeseries_var: Optional[str] = Query(None),
    save: int = Query(0),
    x: Optional[str] = Query(None, description="x-axis column"),
    y: Optional[str] = Query(None, description="y-axis column"),
):
    from dashboard import aggplots
    names = [r.strip() for r in regions.split(",") if r.strip()]
    month_list = [int(m) for m in months.split(",") if m.strip()] if months else None
    legacy = {"optics": "mec_ae", "lifetime": "lifetime_ae"}
    kind_use = legacy.get(kind, kind)
    try:
        with _PLOT_LOCK:
            png, saved, note = aggplots.render(
                kind_use, species, names,
                months=month_list, style=style,
                lifetime_source=lifetime_source,
                timeseries_var=timeseries_var,
                save=bool(save),
                x=x, y=y,
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    headers = {}
    if saved is not None:
        headers["X-Saved-Path"] = str(saved)
    if note:
        headers["X-Plot-Note"] = note[:500].encode("latin-1", "replace").decode("latin-1")
    return Response(content=png, media_type="image/png", headers=headers)


@app.get("/api/resources")
def resources():
    from dashboard.jobs import MEMORY
    return MEMORY


@app.post("/api/polygons")
def save_polygon(body: PolygonRequest):
    from dashboard.polygons import save_polygon as _save
    try:
        return _save(body.name, body.ring, body.surface_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/annual_cycle")
def annual_cycle(body: AnnualRequest):
    from dashboard.annual import annual_cycle as _cycle
    try:
        return _cycle(body.dataset, body.var, body.ring, body.surface_type, body.ens)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs")
def start_job(body: JobRequest):
    from dashboard import jobs
    try:
        if body.where == "local":
            return jobs.start_local(body.slug)
        if body.where == "slurm":
            return jobs.submit_slurm(body.slug, body.mem_gb)
        raise ValueError("where must be local or slurm")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{slug}")
def job_status(slug: str):
    from dashboard.polygons import read_status
    return read_status(slug)


@app.get("/api/meta")
def meta():
    try:
        return D.meta_payload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/field")
def field(
    dataset: str = Query(..., description="pd2024 | pd2010 | aerocom | spexone | polder | icon2025"),
    var: str = Query(...),
    months: Optional[str] = Query(None, description="comma-separated months, e.g. 6,7,8"),
    ens: Optional[int] = Query(None, description="PPE member id; omit for ensemble mean"),
    model: Optional[str] = Query(None, description="AeroCom model id"),
    spex_mode: str = Query("regridded", description="regridded | native"),
    spex_res: float = Query(1.0, description="SPEXone regrid resolution in degrees"),
    spex_year: Optional[int] = Query(None, description="Year for native SPEXone overpass"),
):
    month_list = None
    if months:
        month_list = [int(x) for x in months.split(",") if x.strip()]
    try:
        return D.field_payload(
            dataset, var, months=month_list, ens=ens, model=model,
            spex_mode=spex_mode, spex_res=spex_res, spex_year=spex_year,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class SpexRegridRequest(BaseModel):
    year: int = 2025
    res: float = 1.0


class RegisterRegionRequest(BaseModel):
    slug: str
    species: List[str] = Field(default_factory=lambda: ["du", "ss", "bb", "total", "marine"])
    season_months: Optional[List[int]] = None


@app.post("/api/pipeline/register")
def register_region(body: RegisterRegionRequest):
    """Append a finished drawn polygon to extra_regions.csv and the main parquet tables."""
    from dashboard.extra_regions import register_polygon
    try:
        out = register_polygon(body.slug, body.species, body.season_months)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    D.meta_payload.cache_clear()
    return out


@app.post("/api/spex_regrid")
def spex_regrid(body: SpexRegridRequest):
    from dashboard import jobs
    try:
        return jobs.submit_spex_regrid(body.year, body.res)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/pipeline")
def pipeline_paths():
    """Absolute paths shown on the pipeline map overlay."""
    root = ROOT
    try:
        from dashboard import iconmap
        icon_path = str(iconmap._root())
    except Exception:
        icon_path = str(root / "Data" / "ICON")
    return {
        "nodes": [
            {"id": "spex_l2", "label": "SPEXone L2", "path": str(root / "Data" / "SPEXone")},
            {"id": "spex_regrid", "label": "SPEXone regridded", "path": str(root / "Data" / "SPEXone_regridded")},
            {"id": "polder", "label": "POLDER", "path": str(root / "Data" / "AP3_POLDER_Collocated")},
            {"id": "ensemble", "label": "PPE ensemble NetCDF", "path": "/scratch-shared/jzhang1/PPE"},
            {"id": "icon", "label": "ICON-HAM", "path": icon_path},
            {"id": "regional", "label": "Regional parquet", "path": str(root / "Data" / "PPE_regional")},
            {"id": "polygons", "label": "Drawn-box tables", "path": str(root / "dashboard" / "cache" / "polygons")},
            {"id": "figures", "label": "Figures", "path": str(root / "figure")},
        ],
    }


@app.post("/api/region_mean")
def region_mean(body: RegionMeanRequest):
    try:
        return D.region_mean_payload(
            dataset=body.dataset,
            var=body.var,
            ring=body.ring,
            months=body.months,
            surface_type=body.surface_type,
            model=body.model,
            ens=body.ens,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
