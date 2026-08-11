"""OMI-Aura L3 OMHCHOd daily regional means + optional GES DISC download helper."""
from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr


def parse_omhchod_date(name: str) -> Optional[pd.Timestamp]:
    """Parse date from ``OMI-Aura_L3-OMHCHOd_YYYYmMMDD_*.nc`` filename."""
    m = re.search(r'_(\d{4})m(\d{2})(\d{2})_', name)
    if not m:
        return None
    try:
        return pd.Timestamp(f'{m.group(1)}-{m.group(2)}-{m.group(3)}')
    except Exception:
        return None


def lon_bounds_180(lon_min: float, lon_max: float) -> Tuple[float, float]:
    """Convert 0–360 region lon bounds to −180…180 for OMI grids."""
    lo = lon_min - 360.0 if lon_min > 180.0 else lon_min
    hi = lon_max - 360.0 if lon_max > 180.0 else lon_max
    return lo, hi


def load_omi_hcho_daily_region(
    region_name: str,
    regions: Dict,
    project_root: Path,
    years: Sequence[int] = (2008, 2010),
    time_slice: Optional[Tuple[str, str]] = None,
    min_good_pixels: int = 5,
) -> pd.DataFrame:
    """Area-weighted daily regional OMI HCHO column (QA flag == 0).

    If ``time_slice`` is None, use ``regions[region_name]['time_slice']``.
    Pass an explicit wide window (e.g. full year) for QA timeseries outside
    the fire season.
    """
    cfg = regions[region_name]
    lon_min, lon_max = cfg['lon_range']
    lat_min, lat_max = cfg['lat_range']
    if time_slice is None:
        time_slice = cfg['time_slice']
    t0, t1 = pd.Timestamp(time_slice[0]), pd.Timestamp(time_slice[1])
    lon_min_180, lon_max_180 = lon_bounds_180(lon_min, lon_max)

    rows = []
    root = Path(project_root) / 'Data' / 'OMI_HCHO'
    for year in years:
        year_dir = root / str(year)
        if not year_dir.exists():
            continue
        for fpath in sorted(year_dir.glob('OMI-Aura_L3-OMHCHOd_*.nc')):
            date = parse_omhchod_date(fpath.name)
            if date is None or not (t0 <= date <= t1):
                continue
            try:
                ds_root = xr.open_dataset(fpath)
                lon = ds_root['longitude'].values
                lat = ds_root['latitude'].values
                ds_ksd = xr.open_dataset(fpath, group='key_science_data')
                ds_qa = xr.open_dataset(fpath, group='qa_statistics')
                col = ds_ksd['column_amount'].values
                qa = ds_qa['data_quality_flag'].values
                ds_root.close()
                ds_ksd.close()
                ds_qa.close()
            except Exception as e:
                print(f'  Failed to read {fpath.name}: {e}')
                continue

            if lon_min_180 > lon_max_180:
                lon_mask = (lon >= lon_min_180) | (lon <= lon_max_180)
            else:
                lon_mask = (lon >= lon_min_180) & (lon <= lon_max_180)
            lat_mask = (lat >= lat_min) & (lat <= lat_max)
            sub = col[np.ix_(lat_mask, lon_mask)]
            qa_sub = qa[np.ix_(lat_mask, lon_mask)]
            good = (qa_sub == 0) & np.isfinite(sub) & (sub > -1e29)
            if int(good.sum()) < min_good_pixels:
                continue
            latg = lat[lat_mask]
            weights = np.repeat(np.cos(np.deg2rad(latg))[:, None], int(lon_mask.sum()), axis=1)
            hcho_mean = float(np.average(sub[good], weights=weights[good]))
            rows.append({'date': date, 'HCHO': hcho_mean, 'n_pixels': int(good.sum()),
                         'region': region_name})
    return pd.DataFrame(rows)


def list_on_disk_dates(project_root: Path) -> List[pd.Timestamp]:
    dates = []
    root = Path(project_root) / 'Data' / 'OMI_HCHO'
    if not root.exists():
        return dates
    for year_dir in sorted(root.glob('*')):
        if not year_dir.is_dir():
            continue
        for f in year_dir.glob('OMI-Aura_L3-OMHCHOd_*.nc'):
            d = parse_omhchod_date(f.name)
            if d is not None:
                dates.append(d)
    return sorted(dates)


def cache_path(table_dir: Path, region: str) -> Path:
    d = Path(table_dir) / 'cache'
    d.mkdir(parents=True, exist_ok=True)
    return d / f'OMI_HCHO_daily_{region}.parquet'


def save_region_cache(table_dir: Path, region: str, df: pd.DataFrame) -> Path:
    path = cache_path(table_dir, region)
    out = df.copy()
    if not out.empty and 'date' in out.columns:
        out['date'] = pd.to_datetime(out['date'])
    out.to_parquet(path, index=False)
    return path


def load_region_cache(table_dir: Path, region: str) -> Optional[pd.DataFrame]:
    path = cache_path(table_dir, region)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df


# GES DISC HTTPS layout for OMHCHOd.003 (Earthdata Login required).
GESDISC_OMHCHOD_003 = (
    'https://acdisc.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level3/OMHCHOd.003'
)


def daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def expected_omhchod_url(day: date) -> str:
    """Best-effort directory URL; exact filename suffix varies — use listing or CMR."""
    y, m, d = day.year, day.month, day.day
    # Files are listed under year/ only (flat), not year/month.
    return f'{GESDISC_OMHCHOD_003}/{y}/'


def build_wget_url_list(start: str, end: str, project_root: Path) -> Path:
    """Write a wget URL list by scraping year directory HTML (needs Earthdata cookies).

    Returns path to the URL list file. Caller runs wget with ~/.netrc / cookies.
    """
    import urllib.request

    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    years = sorted({start_d.year, end_d.year})
    urls: List[str] = []
    out_dir = Path(project_root) / 'Data' / 'OMI_HCHO'
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / f'omhchod_urls_{start_d}_{end_d}.txt'

    for year in years:
        index_url = f'{GESDISC_OMHCHOD_003}/{year}/'
        try:
            with urllib.request.urlopen(index_url, timeout=60) as resp:
                html = resp.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f'Could not list {index_url}: {e}')
            continue
        for name in re.findall(r'href="(OMI-Aura_L3-OMHCHOd_[^"]+\.nc)"', html):
            d = parse_omhchod_date(name)
            if d is None:
                continue
            day = d.date()
            if start_d <= day <= end_d:
                urls.append(f'{GESDISC_OMHCHOD_003}/{year}/{name}')

    list_path.write_text('\n'.join(urls) + ('\n' if urls else ''))
    print(f'Wrote {len(urls)} URLs -> {list_path}')
    return list_path
