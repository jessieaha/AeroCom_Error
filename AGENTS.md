# AGENTS.md

## Cursor Cloud specific instructions

### What this project is
A scientific data-analysis codebase for the AeroCom aerosol model
intercomparison (climate/atmospheric research). It is **not** a client/server
app: there are no web servers, APIs, databases, or ports. The pipeline is
file-based: NetCDF model output → Python `pickle` intermediates → figures/CSVs
produced by Jupyter notebooks. Python 3 (tested on 3.12). No package manager
config exists upstream; dependencies are installed by the update script (see
`requirements.txt` for the list).

### Services / how to run
There are no long-running services. "Running the app" means executing the
analysis scripts/notebooks:
- Core modules live in `py/` (`functions.py`, `aerocom_data.py`, `get_data.py`).
  Notebooks and scripts add `py/` to `sys.path`; imports are flat
  (e.g. `import functions`), so run/import from within `py/` or after the
  `sys.path` setup the notebooks already do.
- Primary notebook: `notebooks/SSA_MEC_scatter_plot.ipynb`. Execute headless with
  `python3 -m nbconvert --to notebook --execute notebooks/SSA_MEC_scatter_plot.ipynb`.
  It writes `SSA_MEC_*.png` / `SSA_MAC_*.png` and `SSA_MEC_*.csv` into `notebooks/`.
- Jupyter console scripts install to `~/.local/bin` (not on PATH). Invoke via
  `python3 -m jupyter ...` / `python3 -m nbconvert ...` instead of the bare
  `jupyter`/`jupyter-nbconvert` commands.

### Data (important gotcha)
Real input data is **not** in the repo (`Data/` is gitignored) and is downloaded
from a credentialed met.no HPC server via SSH/SCP (`extract_data_ap3.sh`,
consumed by `py/get_data.py`). Those credentials are not available in this
environment, so `py/get_data.py` cannot fetch real data here.

For local development/testing without HPC access, generate physically-plausible
synthetic monthly pickles with:
`python3 scripts/generate_sample_data.py`
This writes `Data/var_files/original/monthly/{od550,abs550,load_total,load_BC_OA,emi_total,emi_BC_OA}.pickle`
in the exact `{model: {var: xarray.DataArray(time,lat,lon)}}` structure the
notebooks expect, letting the full analysis run end-to-end.

### Known pre-existing issue (not an environment problem)
The final summary cell of `notebooks/SSA_MEC_scatter_plot.ipynb` references an
undefined variable `pickle_dir` and raises `NameError`. This is unrelated to
setup; every analysis and plotting cell before it runs successfully. Execute
with `--allow-errors` when running headless if you need all cells to attempt.

### Missing side-analysis modules
`py/linear_ref.py` imports `general_scripts` and `py/sea_salt.py` imports
`funct_geo_tools`; these modules are not in the repo, so those two scripts are
not runnable here. The core SSA/MEC/MAC pipeline does not need them.
