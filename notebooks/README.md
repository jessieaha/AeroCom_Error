# AeroCom Analysis Notebooks

## Overview

This directory contains Jupyter notebooks for analyzing AeroCom aerosol data, specifically focusing on SSA (Single Scattering Albedo) and MEC (Mass Extinction Coefficient) analysis across multiple geographic regions.

## System Path Configuration

### Why `sys.path` is Needed

The AeroCom project structure uses a modular approach with separate directories for Python scripts:
- **`py/`** - Main analysis scripts (functions.py, aerocom_data.py, etc.)
- **`Data/`** - Input and output data files
- **`notebooks/`** - Jupyter notebooks for analysis

To enable proper module imports from notebooks and scripts, we need to:
1. Add the project root to `sys.path`
2. Add the `py/` directory to `sys.path`

This allows imports like:
```python
from py import functions  # Import from py directory
from AeroCom import aerocom_data  # Import from parent package
```

### How It's Done

Both scripts and notebooks include this setup:

```python
import sys
import os
from pathlib import Path

# Add parent directory to system path
notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(notebook_dir)
py_dir = os.path.join(project_root, 'py')

# Add paths to sys.path if not already present
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if py_dir not in sys.path:
    sys.path.insert(0, py_dir)
```

## Available Notebooks

### 1. SSA_MEC_scatter_plot.ipynb

**Purpose:** Create scatter plots of MEC vs SSA across different geographic regions.

**Features:**
- Loads pickled data from preprocessed AeroCom models
- Creates scatter plots for four regions:
  - **AFRICA** - African continent (June-September fire season)
  - **GLOBAL** - Global annual mean
  - **AMAZON** - Amazon region (July-October fire season)
  - **OUTFLOW_AF** - African outflow region (June-September)
- Generates both combined 2×2 grid and individual high-resolution plots
- Computes statistics: mean, std, correlation coefficients
- Exports data to CSV files for further analysis
- Creates cross-regional comparison plots

**Key Functions:**
- `load_pickle_file()` - Load pickled data safely
- `extract_region_data()` - Extract specific region data for all models

**Outputs:**
- `SSA_MEC_scatter_plot.png` - Combined 2×2 grid plot
- `SSA_MEC_scatter_{region}.png` - Individual plots for each region
- `SSA_MEC_cross_regional_comparison.png` - Box plots comparing regions
- `SSA_MEC_summary_statistics.csv` - Summary stats across regions
- `SSA_MEC_data_{region}.csv` - Detailed model data per region

## Updated Scripts

### py/get_data.py

**Updated:** Added proper `sys.path` configuration at the top of the script.

**Changes:**
- Imports `sys`, `os`, and `Path`
- Automatically detects script location and adds project root to `sys.path`
- Enables proper module imports without manual path manipulation

## Data Files Expected

The notebooks expect pickle files in the following structure:

```
AeroCom/
└── Data/
    └── var_files/
        └── regrid/
            └── 0average/
                ├── SSA.pickle
                ├── MEC.pickle
                ├── od550cs.pickle
                ├── load_total.pickle
                └── ... (other pickle files)
```

Pickle files should contain dictionaries with structure:
```python
{
    'region_name': {
        'model_name': value,  # float value for the region/model
        ...
    },
    ...
}
```

## How to Use

### Prerequisites

1. Python 3.7+ with Jupyter installed
2. Required packages:
   - numpy
   - matplotlib
   - pandas
   - xarray
   - scikit-learn
   - scipy

### Running a Notebook

1. Navigate to the AeroCom directory:
```bash
cd /scistor/guest/gbb083/AeroCom
```

2. Start Jupyter:
```bash
jupyter notebook notebooks/SSA_MEC_scatter_plot.ipynb
```

3. Run cells sequentially (Shift+Enter)

### Cell Descriptions

| Cell | Purpose |
|------|---------|
| 1 | Setup sys.path and configure environment |
| 2 | Import required libraries |
| 3 | Define and verify data paths |
| 4 | Load pickle files (SSA, MEC, etc.) |
| 5 | Examine and display data structure |
| 6 | Prepare and organize data by region |
| 7 | Create combined scatter plots |
| 8 | Create individual region plots with trends |
| 9 | Display summary statistics |
| 10 | Export detailed model-by-model data |
| 11 | Create cross-regional comparison plots |
| 12 | Final summary and file listing |

## Region Definitions

- **AFRICA**: Global box covering Africa (full year, subset JJASO)
- **GLOBAL**: Global annual mean (Jan-Dec)
- **AMAZON**: Amazon basin region (JJA months subset to July-Oct)
- **OUTFLOW_AF**: African outflow region (June-September)

**Note:** The `apply_global_mean()` function in `functions.py` applies region-specific fire season time windows as defined in lines 82-89.

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:
1. Verify the project structure is intact
2. Check that `sys.path` is correctly configured in the notebook
3. Ensure you're running from the correct directory

### Missing Data Files

If pickle files are not found:
1. Run `py/get_data.py` from the project root to generate pickle files
2. Verify the Data directory structure exists
3. Check file paths in cell 3 of the notebook

### Correlation NaN Values

This may occur if:
- All SSA or MEC values are identical (no variance)
- Data contains NaN or infinite values
- Less than 2 data points in the region

## References

- **apply_global_mean()**: `py/functions.py`, lines 67-106
- **global_mean()**: `py/functions.py`, lines 32-65
- **Data preprocessing**: `py/get_data.py`

## Future Enhancements

- [ ] Time series analysis for seasonal variations
- [ ] Statistical significance testing
- [ ] Uncertainty quantification
- [ ] Integration with observational datasets
- [ ] Interactive visualization with Plotly

## Contact

For questions or issues related to this analysis, refer to the project documentation or contact the AeroCom team.

---

*Last Updated: 2026-06-17*
