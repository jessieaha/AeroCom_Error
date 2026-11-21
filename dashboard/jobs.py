"""Run a drawn-box aggregate on this node or hand it to SLURM.

Memory, from the ensemble layout (float32, chunks of 32 members):

* Dashboard and one-variable annual cycle: about 2 GB.
* One ECHAM box through ``regional_columns`` (8 members at a time): about 4–8 GB.
* ICON native-grid box, one file at a time: about 2–4 GB.
* pd2024 + pd2010 + ICON in one job: reserve 32 GB. Use 64 GB if a notebook
  shares the allocation. The NetCDF files are about 5 GB on disk for pd2024;
  that is page cache, not extra RSS, but a 16 GB node is tight.

There is no Apptainer image in this project. Set ``APPTAINER_IMAGE`` to run
the same script inside a container; otherwise the job uses ``.venv-1``.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

from dashboard.polygons import read_status, slugify, status_path

ROOT = Path(__file__).resolve().parent.parent
SBATCH = Path(__file__).resolve().parent / "aggregate_box.sbatch"
LOG_DIR = Path("/scratch-shared/jzhang1/logs")

# What to ask SLURM for. Peak RSS of the aggregate itself is lower; the
# reservation leaves room for HDF5 decompression and all three ensembles.
RECOMMEND_MEM_GB = 32
COMFORT_MEM_GB = 64

MEMORY = {
    "dashboard_gb": 2,
    "annual_cycle_gb": 2,
    "echam_box_gb": 8,
    "icon_box_gb": 4,
    "slurm_request_gb": RECOMMEND_MEM_GB,
    "slurm_comfortable_gb": COMFORT_MEM_GB,
    "pd2024_files_gb": 5,
    "note": (
        "Reserve 32 GB for a drawn-box aggregate (pd2024, pd2010, ICON). "
        "The job's own peak is about 8 GB; 32 GB matches the other PPE jobs "
        "and leaves room for the 5 GB of pd2024 NetCDF. Ask for 64 GB if a "
        "notebook runs in the same allocation."
    ),
}

_LOCAL: dict = {}
_LOCK = threading.Lock()


def _mark(slug: str, **fields) -> None:
    path = status_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if path.is_file():
        current = json.loads(path.read_text())
    current.update(fields)
    current["slug"] = slug
    path.write_text(json.dumps(current, indent=2))


def start_local(slug: str) -> dict:
    slug = slugify(slug)
    with _LOCK:
        thread = _LOCAL.get(slug)
        if thread is not None and thread.is_alive():
            return read_status(slug)
        _mark(slug, state="queued", where="local", message="queued on this node")

        def _run():
            from dashboard.aggregate_box import run
            try:
                run(slug)
            except Exception as exc:
                _mark(slug, state="failed", message=str(exc))

        thread = threading.Thread(target=_run, name=f"box-{slug}", daemon=True)
        _LOCAL[slug] = thread
        thread.start()
    return read_status(slug)


def submit_slurm(slug: str, mem_gb: int = RECOMMEND_MEM_GB) -> dict:
    slug = slugify(slug)
    mem_gb = int(mem_gb)
    if mem_gb < 8 or mem_gb > 256:
        raise ValueError("memory must be between 8 and 256 GB")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _mark(slug, state="queued", where="slurm", message="submitting", memory_gb=mem_gb)
    cmd = [
        "sbatch", "--parsable",
        f"--mem={mem_gb}G",
        f"--job-name=box_{slug[:12]}",
        f"--output={LOG_DIR}/dashbox-%j.out",
        f"--error={LOG_DIR}/dashbox-%j.err",
        str(SBATCH), slug,
    ]
    env = os.environ.copy()
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "sbatch failed").strip()
        _mark(slug, state="failed", message=message)
        raise RuntimeError(message) from exc
    job_id = proc.stdout.strip().split(";")[0]
    _mark(slug, state="queued", where="slurm", slurm_id=job_id,
          message=f"SLURM job {job_id}", memory_gb=mem_gb)
    return read_status(slug)


SPEX_SBATCH = Path(__file__).resolve().parent / "spex_regrid.sbatch"


def submit_spex_regrid(year: int = 2025, res: float = 1.0) -> dict:
    """Submit a 1–12 SLURM array that regrids SPEXone L2 at ``res`` degrees."""
    res = float(res)
    if res not in (0.1, 0.5, 1.0):
        raise ValueError("SPEXone resolution must be 0.1, 0.5 or 1")
    year = int(year)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SPEX_YEAR"] = str(year)
    env["SPEX_RES"] = f"{res:g}"
    cmd = [
        "sbatch", "--parsable",
        f"--job-name=spex_{res:g}d_{year}",
        f"--output={LOG_DIR}/spexregrid-%A_%a.out",
        f"--error={LOG_DIR}/spexregrid-%A_%a.err",
        str(SPEX_SBATCH),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "sbatch failed").strip()
        raise RuntimeError(message) from exc
    job_id = proc.stdout.strip().split(";")[0]
    return {
        "state": "queued",
        "where": "slurm",
        "slurm_id": job_id,
        "year": year,
        "res": res,
        "message": f"SPEXone regrid array {job_id} ({res:g}°, {year})",
    }
