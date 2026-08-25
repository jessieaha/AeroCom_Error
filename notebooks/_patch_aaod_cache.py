#!/usr/bin/env python3
"""Patch AAOD_error_attribution.ipynb for USE_AGG_CACHE workflow."""
from __future__ import annotations

import json
from pathlib import Path

NB = Path('/scistor/guest/gbb083/AeroCom/notebooks/AAOD_error_attribution.ipynb')
nb = json.loads(NB.read_text())


def get(i: int) -> str:
    return ''.join(nb['cells'][i].get('source', []))


def set_src(i: int, text: str) -> None:
    if not text.endswith('\n'):
        text += '\n'
    nb['cells'][i]['source'] = [text]


def indent_block(src: str, spaces: int = 4) -> str:
    pad = ' ' * spaces
    return '\n'.join((pad + ln) if ln.strip() else '' for ln in src.splitlines())


def wrap_skip(src: str, skip_msg: str) -> str:
    if src.lstrip().startswith('if USE_AGG_CACHE') or 'USE_AGG_CACHE=True: skipping' in src[:200]:
        return src
    return (
        'if USE_AGG_CACHE:\n'
        f"    print({skip_msg!r})\n"
        'else:\n'
        + indent_block(src)
        + '\n'
    )


# --- Config cell 2 ---
src2 = get(2)
if 'USE_AGG_CACHE' not in src2:
    needle = 'USE_PICKLE = False\n'
    insert = (
        'USE_PICKLE = False\n'
        '\n'
        '# Shared regional seasonal cache (see py/build_regional_aggregates.py).\n'
        '# True  → skip NetCDF load / aggregate / POLDER / GPCP / homogenize;\n'
        '#         rebuild model_seasonal + obs from parquet.\n'
        '# False → full pipeline (optional SAVE_AGG_CACHE refreshes the parquet).\n'
        'USE_AGG_CACHE = True\n'
        'SAVE_AGG_CACHE = False\n'
        "AGG_PARQUET = TABLE_DIR / 'regional_seasonal_aggregates.parquet'\n"
    )
    if needle not in src2:
        raise SystemExit('USE_PICKLE not found in config')
    src2 = src2.replace(needle, insert, 1)
    src2 = src2.replace(
        "print(f'  SAVE_FIGURE          = {SAVE_FIGURE}')\n",
        "print(f'  USE_AGG_CACHE         = {USE_AGG_CACHE}')\n"
        "print(f'  AGG_PARQUET           = {AGG_PARQUET}')\n"
        "print(f'  SAVE_FIGURE          = {SAVE_FIGURE}')\n",
        1,
    )
    set_src(2, src2)
    print('config updated')
else:
    print('config already has USE_AGG_CACHE')

# --- Load cell 4 ---
src4 = get(4)
if 'USE_AGG_CACHE' not in src4[:120]:
    set_src(4, wrap_skip(src4, 'USE_AGG_CACHE=True: skipping NetCDF load'))
    print('cell 4 wrapped')
else:
    print('cell 4 already gated')

for idx, msg in [
    (6, 'USE_AGG_CACHE=True: skipping normalize/sums'),
    (8, 'USE_AGG_CACHE=True: skipping pre-aggregation derived vars'),
    (10, 'USE_AGG_CACHE=True: skipping regional aggregation'),
    (17, 'USE_AGG_CACHE=True: skipping POLDER load (from parquet)'),
    (19, 'USE_AGG_CACHE=True: skipping GPCP load (from parquet)'),
    (24, 'USE_AGG_CACHE=True: skipping homogenization (from parquet)'),
]:
    src = get(idx)
    if 'USE_AGG_CACHE=True: skipping' in src[:250]:
        print(f'cell {idx} already gated')
        continue
    set_src(idx, wrap_skip(src, msg))
    print(f'cell {idx} wrapped')

CACHE_CELL = r'''# ---------------------------------------------------------------------------
# Optional: load shared regional seasonal cache
# ---------------------------------------------------------------------------
# Built by:  python py/build_regional_aggregates.py
# Columns: region, dataset, var, source, value, unit, area, ...
import sys
sys.path.insert(0, str(project_root / 'py'))
from build_regional_aggregates import (
    load_agg_cache, rows_to_model_seasonal, rows_to_obs_dicts,
    gpcp_seasonal_to_region_series, DEFAULT_OUT,
)

if USE_AGG_CACHE:
    _agg_path = Path(AGG_PARQUET) if 'AGG_PARQUET' in dir() else Path(DEFAULT_OUT)
    print(f'Loading aggregate cache: {_agg_path}')
    agg_df = load_agg_cache(_agg_path)
    model_seasonal = rows_to_model_seasonal(agg_df, regions=list(ANALYSIS_REGIONS))
    _extra = [r for r in agg_df['region'].unique() if r not in model_seasonal]
    if _extra:
        model_seasonal.update(rows_to_model_seasonal(agg_df, regions=_extra))
    models = sorted({
        m for reg in model_seasonal.values()
        for var_map in reg.values() for m in var_map
    })
    aaod_obs_by_region, ssa_obs_by_region, _gpcp_seasonal, polder_sampled = rows_to_obs_dicts(
        agg_df, regions=list(ANALYSIS_REGIONS)
    )
    gpcp_region = gpcp_seasonal_to_region_series(_gpcp_seasonal)
    polder_monthly = {
        region: pd.DataFrame([{
            'AAOD_550': polder_sampled.get(region, {}).get('AAOD_550'),
            'AOD_550': polder_sampled.get(region, {}).get('AOD_550'),
            'SSA': polder_sampled.get(region, {}).get('SSA'),
            'AE': polder_sampled.get(region, {}).get('AE'),
        }])
        for region in ANALYSIS_REGIONS
    }
    model_monthly = {}  # seasonal-only cache
    print(f'  models={len(models)}, regions={list(model_seasonal)}')
    for region in ANALYSIS_REGIONS:
        print(
            f'  {region}: AAOD_obs={aaod_obs_by_region.get(region, float("nan")):.5f}, '
            f'SSA_obs={ssa_obs_by_region.get(region, float("nan")):.5f}, '
            f'GPCP={_gpcp_seasonal.get(region, float("nan")):.3f}'
        )
        for var in ['MAC', 'SSA', 'AE', 'lifetime_BC_OA', 'precip', 'abs550aer',
                    'emi_BC_OA', 'dep_BC_OA']:
            n = len(model_seasonal.get(region, {}).get(var, {}))
            print(f'    {var}: {n} models')
else:
    print('USE_AGG_CACHE=False: using in-notebook full pipeline results')
    if SAVE_AGG_CACHE:
        print('SAVE_AGG_CACHE=True: re-run `python py/build_regional_aggregates.py` '
              'to refresh the parquet (notebook does not write full multi-region cache).')
'''

already = any('load_agg_cache' in ''.join(c.get('source', [])) for c in nb['cells'])
if not already:
    # After homogenize cell (index 24)
    insert_at = 25
    nb['cells'].insert(insert_at, {
        'cell_type': 'code',
        'execution_count': None,
        'id': 'agg-cache-load',
        'metadata': {},
        'outputs': [],
        'source': [CACHE_CELL if CACHE_CELL.endswith('\n') else CACHE_CELL + '\n'],
    })
    print(f'inserted cache cell at {insert_at}')
else:
    print('cache cell already present')

NB.write_text(json.dumps(nb, indent=1) + '\n')
print('AAOD notebook saved, n_cells=', len(nb['cells']))
PY
