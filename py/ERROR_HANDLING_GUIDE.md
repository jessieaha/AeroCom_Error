# Error Handling Guide for `aerocom_data.py`

## Overview

The `get_data()` function in `aerocom_data.py` now includes comprehensive error handling for common data loading issues. This guide explains the error checks and how to troubleshoot them.

---

## Error Categories

### 1. **Input Validation Errors**

These errors occur before attempting to load data:

#### Path Template Validation
```
❌ ERROR: Path template is invalid.
   Expected format string with {} placeholders...
```
**Solution:** Ensure your path contains `{}` placeholders for formatting.

**Example (correct):**
```python
path = './Data/AEROCOM_III_regrid/{}/aerocom3_{}_{}_{}__{}.nc'
```

#### Frequency Parameter Validation
```
❌ ERROR: Invalid frequency for od550_freq.
   Valid options: monthly, daily, 3hourly
   Received: yearly
```
**Solution:** Use only `'monthly'`, `'daily'`, or `'3hourly'` for frequency parameters.

#### Type Validation
```
❌ ERROR: models_names must be a list or tuple
```
**Solution:** Ensure parameters are of correct type:
- `models_names`: list or tuple of strings
- `path`: string
- `var_name`: string

---

### 2. **File System Errors**

These errors occur when accessing the filesystem:

#### Directory Not Found
```
❌ ERROR: Data directory does not exist.
   Model: CAM5.3-Oslo_AP3-CTRL2016-PD
   Variable: emi_total
   Directory: ./Data/AEROCOM_III_regrid/
   Please check the data path configuration.
```
**Solution:**
- Verify the base data directory exists
- Check path spelling and format
- Ensure directory is readable

**Debug:**
```bash
ls -la ./Data/AEROCOM_III_regrid/
```

#### File Not Found
```
❌ ERROR: Data file not found.
   Model: CAM5.3-Oslo_AP3-CTRL2016-PD
   Variable: emi_total
   Expected file: ./Data/AEROCOM_III_regrid/CAM5.3-Oslo_AP3-CTRL2016-PD/aerocom3_...nc
   Please verify:
     - File path is correct
     - File has been downloaded/created
     - File permissions allow reading
```
**Solution:**
- Check if the file exists: `ls -la <filepath>`
- Verify file permissions: `chmod 644 <filepath>`
- Re-download or regenerate the file if corrupted
- Check if model/variable combination is available

**Debug:**
```bash
find ./Data -name "*CAM5.3*" -type f
file ./Data/.../aerocom3_*.nc  # Check file type
```

---

### 3. **Variable Access Errors**

These errors occur when accessing variables within netCDF files:

#### Variable Not Found in Dataset
```
❌ ERROR: Variable not found in dataset.
   Model: CAM5.3-Oslo_AP3-CTRL2016-PD
   Requested variable: emibc
   File: ./Data/AEROCOM_III_regrid/.../aerocom3_CAM5.3-Oslo_AP3-CTRL2016-PD_emibc_Surface_monthly__2010.nc
   Available variables (5): emibc, emioa, emiss, emiso2, emidust
   Please check variable name spelling or file contents.
```
**Solution:**
- Check variable name spelling
- Use the listed available variables
- Some models may have different variable names
- Inspect file contents with `ncdump` or `xarray`

**Debug:**
```python
import xarray as xr
ds = xr.open_dataset('path/to/file.nc')
print(ds.data_vars)  # List all variables
```

---

### 4. **Variable Request Errors**

#### Unknown Variable Requested
```
❌ ERROR: Unknown variable requested.
   Variable: my_custom_var
   Model: CAM5.3-Oslo_AP3-CTRL2016-PD
   Valid variables: emi_total, emi_BC_OA, emi_ss, emi_bc, emi_oa,
                    load_total, load_BC_OA, load_ss, load_bc, load_oa,
                    optical_depth_1, optical_depth_2, abs_optical_depth, prect
```
**Solution:** Use one of the valid variables listed in the error message.

---

### 5. **Complete Data Loading Failure**

#### No Models Successfully Loaded
```
❌ ERROR: No models were successfully loaded.
   Requested variable: emi_total
   Requested models: CAM5.3-Oslo_AP3-CTRL2016-PD, ECHAM6-HAM2_AP3-CTRL2016-PD, ...
   Path template: ./Data/AEROCOM_III_regrid/{}/aerocom3_{}_{}_{}__{}.nc
   Please check:
     1. Data files exist in the specified path
     2. Path template is correct
     3. Model names are spelled correctly
     4. Variable name is valid
```
**Solution:** Investigate and fix all the issues listed in the error message.

---

## Data Loading Summary

After processing all models, the function prints a summary:

```
📊 Data Loading Summary:
   ✓ Successful: 12 / 16 models
   ✗ Failed: 4 models

   Failed models:
     - CAM5_CTRL2016: emibc
     - GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010: emidust
     ... and 2 more
```

This helps identify which models failed and why.

---

## Troubleshooting Workflow

### Step 1: Verify Path Configuration
```python
from pathlib import Path

path_template = './Data/AEROCOM_III_regrid/{}/aerocom3_{}_{}_{}__{}.nc'
print(f"Path template valid: {'{}' in path_template}")
```

### Step 2: Check Directory Structure
```bash
tree ./Data/AEROCOM_III_regrid -L 2  # Or use find
```

### Step 3: Verify File Existence
```python
from pathlib import Path

model = "CAM5.3-Oslo_AP3-CTRL2016-PD"
var = "emibc"
freq = "monthly"

test_path = path.format(model, model, var, "Surface", freq)
print(f"File exists: {Path(test_path).exists()}")
print(f"File size: {Path(test_path).stat().st_size if Path(test_path).exists() else 'N/A'}")
```

### Step 4: Inspect File Contents
```python
import xarray as xr

ds = xr.open_dataset(test_path)
print("Variables:", list(ds.data_vars.keys()))
print("Coordinates:", list(ds.coords.keys()))
print(ds)  # Full dataset info
```

### Step 5: Test with Single Model
```python
from py import aerocom_data

# Test with just one model first
test_models = ["CAM5.3-Oslo_AP3-CTRL2016-PD"]
data = aerocom_data.get_data(path, test_models, 'emi_total')
print(f"Successfully loaded {len(data)} model(s)")
```

---

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| All models fail | Path is incorrect | Check path template matches file locations |
| Some models fail | File doesn't exist for that model | Download/regenerate missing files |
| Variable not found | Wrong variable name | Check available variables in file |
| Permission denied | File not readable | `chmod 644 filename` |
| Memory error | Data too large | Process fewer models or variables |
| Format error | Corrupted file | Re-download the file |

---

## Error Handling Implementation

### Helper Functions

The `get_data()` function includes two helper functions:

#### `safe_open_dataset(filepath, model, variable_name, location)`
- Checks directory existence
- Checks file existence
- Handles file opening errors
- Provides detailed error messages with context

#### `get_variable_safely(dataset, variable_name, model, filepath)`
- Checks if variable exists in dataset
- Lists available variables on failure
- Provides helpful suggestions

### Try-Except Blocks

Each variable loading block is wrapped in try-except:
```python
try:
    file_path = path.format(model, model, var, ...)
    dataset = safe_open_dataset(file_path, model, var)
    data = functions.shift360(...)
    # Process data
except (FileNotFoundError, KeyError) as e:
    print(f"⚠️  WARNING: {str(e)}")
    failed_models.append((model, var, str(e)))
    raise  # Re-raise to trigger outer exception handler
except Exception as e:
    # Outer handler logs and removes model from results
```

This two-layer approach ensures:
1. Detailed error info is printed
2. Model is removed from results if any variable fails
3. Other models continue processing

---

## Example Usage with Error Handling

```python
from py import aerocom_data

path = './Data/AEROCOM_III_regrid/{}/aerocom3_{}_{}_{}__{}.nc'
models = ['CAM5.3-Oslo_AP3-CTRL2016-PD', 'ECHAM6-HAM2_AP3-CTRL2016-PD']

try:
    data = aerocom_data.get_data(path, models, 'emi_total')
    print(f"Successfully loaded {len(data)} models")
    
    # Use data
    for model, model_data in data.items():
        print(f"{model}: {model_data['emi_total'].shape}")
        
except ValueError as e:
    print(f"Configuration error: {e}")
except FileNotFoundError as e:
    print(f"Data file error: {e}")
except KeyError as e:
    print(f"Variable error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Notes

- **Graceful Degradation**: If some models fail, others continue processing
- **Detailed Logging**: Each error includes context about which model/variable failed
- **Summary Report**: End summary shows how many models succeeded/failed
- **User Guidance**: Error messages include suggestions for fixing issues

---

## Related Documentation

- `functions.py`: Data processing helper functions
- `get_data.py`: Main script using `get_data()`
- `README.md`: General project documentation

---

*Last Updated: 2026-06-17*
*Error Handling Version: 1.0*
