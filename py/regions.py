"""Single registry of analysis regions and aerosol species.

Seasons are stored as month-of-year tuples so one definition serves the
AeroCom / POLDER year (2010), the 2010 PPE and the Aug 2024 - Jul 2025 PPE.
``notebook_setup.REGIONS`` and ``PPE_set.AAOD_REGIONS`` are views of
``REGIONS`` (see ``aerocom_view`` / ``ppe_view``).

Longitudes are 0..360; a lon_range with lon0 > lon1 wraps across 0E.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ALL_MONTHS = tuple(range(1, 13))

# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
# Keys per region:
#   surface_type   'land' | 'ocean' | 'all'
#   lon_range      (lon0, lon1) in 0..360
#   lat_range      (lat0, lat1)
#   season_months  month-of-year tuple (may wrap the calendar year)
#   edge_weighted  fractional boundary weights (outflow boxes)
#   species        species keys this region is analysed for
#   role           'source' | 'outflow' | 'reference'
#   time_slice_2010  optional exact AeroCom 2010 window when the season starts
#                    or ends mid-month (kept from the original notebook boxes)
#   color          plot colour
#   note           free text caveat
REGIONS: Dict[str, dict] = {
    "global": dict(
        surface_type="all", lon_range=(0, 360), lat_range=(-90, 90),
        season_months=ALL_MONTHS, edge_weighted=False,
        species=("bb", "ss", "du"), role="reference", color="#7f7f7f",
    ),
    # --- Biomass burning (fire seasons; notebook_setup / AAOD notebooks) ---
    "africa": dict(
        surface_type="land", lon_range=(15, 37), lat_range=(-15, 0),
        season_months=(6, 7, 8, 9), edge_weighted=False,
        species=("bb",), role="source", color="#d62728",
    ),
    "amazon": dict(
        surface_type="land", lon_range=(287, 317), lat_range=(-17, -3),
        season_months=(7, 8, 9, 10), edge_weighted=False,
        species=("bb",), role="source", color="#2ca02c",
    ),
    # Nature / AOD paper box (Zhong et al.); same as cameo_toolbox._REGION_DEFINITIONS.
    "outflow_af": dict(
        surface_type="ocean", lon_range=(350, 15), lat_range=(-15, 0),
        season_months=(6, 7, 8, 9), edge_weighted=True,
        species=("bb",), role="outflow", source="africa", color="#1f77b4",
    ),
    # "se_asia": dict(
    #     surface_type="land", lon_range=(95, 125), lat_range=(-5, 15),
    #     season_months=(3,), edge_weighted=False,
    #     species=("bb",), role="source", color="#9467bd",
    # ),
    # "boreal_na": dict(
    #     surface_type="land", lon_range=(242, 258), lat_range=(52, 63),
    #     season_months=(6, 7, 8), edge_weighted=False,
    #     species=("bb",), role="source", color="#8c564b",
    # ),
    # "eastern_siberia": dict(
    #     surface_type="land", lon_range=(120, 170), lat_range=(50, 65),
    #     season_months=(7,), edge_weighted=False,
    #     species=("bb",), role="source", color="#e377c2",
    # ),
    # "west_russia": dict(
    #     surface_type="land", lon_range=(33, 52), lat_range=(51, 60),
    #     season_months=(7, 8, 9), edge_weighted=False,
    #     time_slice_2010=("2010-07-15", "2010-09-10"),
    #     species=("bb",), role="source", color="#bcbd22",
    # ),
    # "boreal_na_west": dict(
    #     surface_type="land", lon_range=(242, 258), lat_range=(52, 63),
    #     season_months=(6, 7, 8), edge_weighted=False,
    #     time_slice_2010=("2010-06-15", "2010-08-31"),
    #     species=("bb",), role="source", color="#17becf",
    # ),
    # "indonesia_peatland": dict(
    #     surface_type="land", lon_range=(98, 118), lat_range=(-5, 5),
    #     season_months=(8, 9, 10, 11), edge_weighted=False,
    #     time_slice_2010=("2010-08-01", "2010-11-15"),
    #     species=("bb",), role="source", color="#ff7f0e",
    # ),
    # Sea salt. North Pacific (NPO) is not used. The second box is the remote
    # South Pacific (formerly remote_SO_DJF), not the smaller South Atlantic DJF box.
    "SAO_SOND": dict(
        surface_type="ocean", lon_range=(320, 10), lat_range=(-60, -25),
        season_months=(9, 10, 11, 12), edge_weighted=False,
        species=("ss",), role="source", color="#1f77b4",
        note="South Atlantic, SON+December.",
    ),
    "SPO_DJF": dict(
        surface_type="ocean", lon_range=(200, 260), lat_range=(-62, -42),
        season_months=(12, 1, 2), edge_weighted=False,
        species=("ss",), role="source", color="#17becf",
        note="Remote South Pacific Southern Ocean, DJF. Formerly remote_SO_DJF.",
    ),
    # --- Mineral dust (primary dust seasons) ---
    "sahara": dict(
        surface_type="land", lon_range=(342, 30), lat_range=(13, 30),
        season_months=(6, 7, 8), edge_weighted=False,
        species=("du",), role="source", color="#d4a017",
        note="Sahara 18°W–30°E (lon 342–30), 13–30°N, land only. JJA.",
    ),
    "australia": dict(
        surface_type="land", lon_range=(115, 145), lat_range=(-32, -18),
        season_months=(9, 10, 11, 12, 1, 2), edge_weighted=False,
        species=("du",), role="source", color="#8c564b",
        note="Australia 115–145°E, 32–18°S, land only. Austral spring–summer.",
    ),
    "atacama": dict(
        surface_type="land", lon_range=(288, 300), lat_range=(-32, -20),
        season_months=(8, 9, 10, 11), edge_weighted=False,
        species=("du",), role="source", color="#9467bd",
        note="Atacama / Altiplano, 72–60°W (lon 288–300), 32–20°S, land only. ASON.",
    ),
    "outflow_sahara": dict(
        surface_type="ocean", lon_range=(310, 342), lat_range=(13, 30),
        season_months=(6, 7, 8), edge_weighted=True,
        species=("du",), role="outflow", source="sahara", color="#1f77b4",
        note="Atlantic 50°W–18°W (lon 310–342), 13–30°N, meeting the Sahara box at lon 342. Ocean only. JJA.",
    ),
}

# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------
# Variable names follow AeroCom spelling (PPE extract keeps the same names).
#   key                           suffix of the summed columns load_<key>, emi_<key>,
#                                 lifetime_<key>
#   load / emi / dep / wet / dry   components summed into those columns
#   od / abs                      species optical depth / absorption at 550 nm
#                                 (None -> total-column od550aer / abs550aer)
#   mass_coeffs                   per-cell ratios built by ppe_regional_aggregates
#   aliases                       notebook column -> regional-table column
#                                 (e.g. the BB notebook's "MAC" is MAC_BC_OA)
#   kernel                        (optical coefficient, SSA column) regressed
SPECIES: Dict[str, dict] = {
    "bb": dict(
        label="Biomass burning (BC+OA)",
        key="BC_OA",
        load=("loadbc", "loadoa"),
        emi=("emibc", "emioa"),
        dep=("depbc", "depoa"),
        wet=("wetbc", "wetoa"),
        dry=(),
        od=None,
        abs=None,
        mass_coeffs=("MAC_BC_OA",),
        aliases={"MAC": "MAC_BC_OA", "AE": "AE_440_550"},
        kernel=("MAC", "SSA"),
    ),
    "ss": dict(
        label="Sea salt",
        key="ss",
        load=("loadss",),
        emi=("emiss",),
        dep=("depss",),
        wet=("wetss",),
        dry=("dryss",),
        od="od550ss",
        abs="abs550ss",
        mass_coeffs=("MEC_ss",),
        aliases={"MAC": "MAC_total", "AE": "AE_550_865"},
        kernel=("MEC_ss", "SSA"),
    ),
    "du": dict(
        label="Mineral dust",
        key="du",
        load=("loaddust",),
        emi=("emidust",),
        dep=("depdust",),
        wet=("wetdu",),
        dry=("drydu",),
        od="od550dust",
        abs="abs550dust",
        mass_coeffs=("MEC_du", "MAC_du"),
        aliases={"MAC": "MAC_total", "AE": "AE_440_550"},
        kernel=("MEC_du", "SSA_du"),
    ),
}

TOTAL_LOAD = ("loadbc", "loaddust", "loadoa", "loadso4", "loadss")
TOTAL_EMI = ("emibc", "emidust", "emioa", "emiso2", "emiss")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def regions_for_species(species: str, roles: Optional[Iterable[str]] = None) -> List[str]:
    """Region names tagged with ``species`` (optionally filtered by role)."""
    roles = set(roles) if roles is not None else None
    return [
        name for name, cfg in REGIONS.items()
        if species in cfg["species"]
        and name != "global"
        and (roles is None or cfg["role"] in roles)
    ]


def source_regions(species: str) -> List[str]:
    return regions_for_species(species, roles=("source",))


def outflow_regions(species: str) -> List[str]:
    return regions_for_species(species, roles=("outflow",))


def region_colors(names: Optional[Sequence[str]] = None) -> Dict[str, str]:
    names = list(REGIONS) if names is None else list(names)
    return {n: REGIONS[n]["color"] for n in names}


def season_window(season_months: Sequence[int], year: int) -> Tuple[str, str]:
    """Contiguous (start, end) ISO dates of a non-wrapping season in ``year``."""
    months = sorted(season_months)
    m0, m1 = months[0], months[-1]
    last_day = calendar.monthrange(year, m1)[1]
    return f"{year}-{m0:02d}-01", f"{year}-{m1:02d}-{last_day:02d}"


def season_wraps(season_months: Sequence[int]) -> bool:
    """True when the season is not one contiguous run inside a calendar year."""
    months = sorted(set(season_months))
    return months != list(range(months[0], months[-1] + 1))


def aerocom_view(names: Optional[Sequence[str]] = None, year: int = 2010) -> Dict[str, dict]:
    """Regions in the ``notebook_setup.REGIONS`` format (``time_slice`` for ``year``).

    Seasons that wrap the calendar year (e.g. australia Sep-Feb) get the full
    year as ``time_slice`` plus ``season_months``; ``notebook_setup.aggregate_region``
    filters by month when ``season_months`` is present.
    """
    out = {}
    for name in (names or REGIONS):
        cfg = REGIONS[name]
        months = tuple(cfg["season_months"])
        if year == 2010 and cfg.get("time_slice_2010"):
            ts = tuple(cfg["time_slice_2010"])
        elif season_wraps(months):
            ts = (f"{year}-01-01", f"{year}-12-31")
        else:
            ts = season_window(months, year)
        view = dict(
            surface_type=cfg["surface_type"],
            lon_range=tuple(cfg["lon_range"]),
            lat_range=tuple(cfg["lat_range"]),
            time_slice=ts,
            edge_weighted=cfg["edge_weighted"],
        )
        if season_wraps(months):
            view["season_months"] = months
        out[name] = view
    return out


def ppe_view(names: Optional[Sequence[str]] = None,
             time_slice: Optional[Tuple[str, str]] = None) -> Dict[str, dict]:
    """Regions in the ``PPE_set.AAOD_REGIONS`` format (time_slice + season_months)."""
    out = {}
    for name in (names or REGIONS):
        cfg = REGIONS[name]
        out[name] = dict(
            surface_type=cfg["surface_type"],
            lon_range=tuple(cfg["lon_range"]),
            lat_range=tuple(cfg["lat_range"]),
            time_slice=time_slice,
            season_months=tuple(cfg["season_months"]),
            edge_weighted=cfg["edge_weighted"],
        )
    return out


def box_mask(lon, lat, lon_range, lat_range):
    """Boolean mask of points inside a lon/lat box.

    Longitude may be −180..180 or 0..360. A range with lon0 > lon1 wraps 0°E.
    """
    import numpy as np

    lon0, lon1 = float(lon_range[0]), float(lon_range[1])
    lat0, lat1 = float(lat_range[0]), float(lat_range[1])
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon360 = np.mod(lon, 360.0)
    if lon0 <= lon1:
        lon_ok = (lon360 >= lon0) & (lon360 <= lon1)
    else:
        lon_ok = (lon360 >= lon0) | (lon360 <= lon1)
    return lon_ok & (lat >= min(lat0, lat1)) & (lat <= max(lat0, lat1))


def in_box(lon360, lat, region: str):
    """Boolean mask of points inside a registered region box (no land/sea mask)."""
    cfg = REGIONS[region]
    return box_mask(lon360, lat, cfg["lon_range"], cfg["lat_range"])


def points_in_region(lon, lat, region: str):
    """Points inside the box and on the region's land or ocean mask.

    Land/ocean uses the same Natural Earth 110 m mask as gridded AeroCom and PPE
    aggregates (``cameo_toolbox.point_is_land``).
    """
    cfg = REGIONS[region]
    inside = box_mask(lon, lat, cfg["lon_range"], cfg["lat_range"])
    surface = cfg.get("surface_type", "all")
    if surface not in ("land", "ocean"):
        return inside
    import cameo_toolbox as ct
    land = ct.point_is_land(lon, lat)
    if surface == "land":
        return inside & land
    return inside & ~land


def definition_hash(names: Optional[Sequence[str]] = None) -> str:
    """Stable short hash of region geometry + seasons (for cache manifests)."""
    names = sorted(names or REGIONS)
    payload = {
        n: {k: REGIONS[n][k] for k in
            ("surface_type", "lon_range", "lat_range", "season_months", "edge_weighted")}
        for n in names
    }
    blob = json.dumps(payload, sort_keys=True, default=list).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


EXTRA_REGIONS_CSV = (
    Path(__file__).resolve().parent.parent / "Data" / "PPE_regional" / "extra_regions.csv"
)


def load_extra_regions(path: Optional[Path] = None) -> None:
    """Merge dashboard-registered boxes into ``REGIONS``.

    The CSV is the place a drawn polygon is stored. Built-in boxes stay in this
    file; a row with the same name replaces only that extra entry on reload.
    """
    path = Path(path) if path else EXTRA_REGIONS_CSV
    if not path.is_file():
        return
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("name") or "").strip()
            if not name:
                continue
            months_raw = (row.get("season_months") or "").replace(";", ",")
            months = tuple(int(x) for x in months_raw.split(",") if x.strip())
            species = tuple(
                s.strip() for s in (row.get("species") or "du,ss,bb").replace(";", ",").split(",")
                if s.strip()
            )
            cfg = dict(
                surface_type=(row.get("surface_type") or "all").strip() or "all",
                lon_range=(float(row["lon0"]), float(row["lon1"])),
                lat_range=(float(row["lat0"]), float(row["lat1"])),
                season_months=months or ALL_MONTHS,
                edge_weighted=False,
                species=species or ("du", "ss", "bb"),
                role=(row.get("role") or "source").strip() or "source",
                color=(row.get("color") or "#ff00aa").strip() or "#ff00aa",
                note=(row.get("note") or "drawn polygon from the dashboard").strip(),
            )
            ring = (row.get("ring") or "").strip()
            if ring:
                cfg["ring"] = json.loads(ring)
            REGIONS[name] = cfg


load_extra_regions()
