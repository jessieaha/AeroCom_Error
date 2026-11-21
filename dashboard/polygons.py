"""Saved drawn polygons and where their tables live."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

CACHE = Path(__file__).resolve().parent / "cache" / "polygons"


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    if not slug:
        raise ValueError("polygon needs a name")
    return slug[:80]


def polygon_path(slug: str) -> Path:
    return CACHE / f"{slug}.json"


def status_path(slug: str) -> Path:
    return CACHE / f"{slug}_status.json"


def seasonal_path(slug: str) -> Path:
    return CACHE / f"{slug}_seasonal.parquet"


def monthly_path(slug: str) -> Path:
    return CACHE / f"{slug}_monthly.parquet"


def save_polygon(name: str, ring: List[List[float]], surface_type: str = "all") -> dict:
    if len(ring) < 4:
        raise ValueError("polygon needs at least 3 vertices")
    slug = slugify(name)
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name.strip(),
        "slug": slug,
        "ring": [[float(lon), float(lat)] for lon, lat in ring],
        "surface_type": surface_type or "all",
    }
    polygon_path(slug).write_text(json.dumps(payload))
    return payload


def read_polygon(slug: str) -> dict:
    path = polygon_path(slugify(slug))
    if not path.is_file():
        raise FileNotFoundError(f"no saved polygon {slug}")
    return json.loads(path.read_text())


def read_status(slug: str) -> dict:
    path = status_path(slugify(slug))
    if not path.is_file():
        return {"slug": slugify(slug), "state": "absent"}
    return json.loads(path.read_text())
