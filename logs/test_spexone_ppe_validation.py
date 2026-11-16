"""Smoke-test SPEXone_check PPE validation cells (0 setup + load + 3/4/5)."""
import json
import traceback
from pathlib import Path

nb = json.loads(Path("notebooks/SPEXone_check.ipynb").read_text())
ns = {"__name__": "__main__"}

exec("".join(nb["cells"][0]["source"]), ns)
print("cell0 ok", ns["project_root"])

code1 = """
import glob
import cameo_toolbox as ct
import regions
from regions import REGIONS
spex_grid_files = sorted(glob.glob(str(project_root / 'Data' / 'SPEXone_regridded' / '*.parquet')))
print(f'loading {len(spex_grid_files)} spex files...')
l3_spex = pd.concat((pd.read_parquet(fp) for fp in spex_grid_files), ignore_index=True)
print('l3_spex', len(l3_spex))
region_boxes = {
    name: (cfg['lon_range'][0], cfg['lon_range'][1], cfg['lat_range'][0], cfg['lat_range'][1])
    for name, cfg in REGIONS.items()
    if 'lon_range' in cfg and 'lat_range' in cfg
}
"""
exec(code1, ns)

for i in (3, 4, 5):
    print(f"\n=== running cell {i} ===")
    try:
        exec("".join(nb["cells"][i]["source"]), ns)
        print(f"cell {i} ok")
    except Exception:
        traceback.print_exc()
        raise

print("done")
