#!/usr/bin/env python3
"""Download OMI-Aura L3 OMHCHOd daily files (GES DISC) for a date range.

Requires Earthdata Login via ``~/.netrc``::

    machine urs.earthdata.nasa.gov
        login YOUR_USER
        password YOUR_PASS

Authorize the GES DISC application in your Earthdata profile, then::

    python py/download_omi_hcho.py --start 2010-06-01 --end 2010-11-15
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'py'))
import omi_hcho  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', default='2010-06-01')
    p.add_argument('--end', default='2010-11-15')
    p.add_argument('--dry-run', action='store_true', help='Only write URL list')
    args = p.parse_args()

    out_year = ROOT / 'Data' / 'OMI_HCHO' / str(pd_year := __import__('pandas').Timestamp(args.start).year)
    out_year.mkdir(parents=True, exist_ok=True)

    list_path = omi_hcho.build_wget_url_list(args.start, args.end, ROOT)
    n_urls = len([ln for ln in list_path.read_text().splitlines() if ln.strip()])
    if n_urls == 0:
        print('No URLs listed (listing may require Earthdata auth). '
              'Create ~/.netrc and re-run, or paste GES DISC links into the list file.')
        return 1
    if args.dry_run:
        return 0

    cookie = Path.home() / '.urs_cookies'
    cmd = [
        'wget', '-c', '--content-disposition',
        f'--load-cookies={cookie}', f'--save-cookies={cookie}',
        '--auth-no-challenge=on', '--keep-session-cookies',
        '-P', str(out_year),
        '-i', str(list_path),
    ]
    print('Running:', ' '.join(cmd))
    rc = subprocess.call(cmd)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
