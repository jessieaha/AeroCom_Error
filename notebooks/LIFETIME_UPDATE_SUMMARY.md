# Lifetime Calculation Updates - Summary

**Date:** 2026-08-11  
**Status:** ✅ COMPLETE

## Changes Made

### 1. POST_AGG_LIFETIME Filtering (AOD notebook only)

**File:** `AOD_error_attribution.ipynb`

**Added configuration:**
```python
POST_AGG_LIFETIME_MAX_DAYS = 15.0  # Filter lifetimes > 15 days
POST_AGG_LIFETIME_MIN_DAYS = 0.3   # Filter lifetimes < 0.3 days
```

**Purpose:**  
Prevent model blow-ups (especially models 3, 10, 16) caused by extreme post-aggregation lifetimes in transport-dominated regional boxes. AAOD notebook already had this (set to 20 days max).

**Implementation:**  
Updated `compute_derived_after_aggregation()` function to filter lifetime values after calculation:
- Values > `POST_AGG_LIFETIME_MAX_DAYS` → set to None
- Values < `POST_AGG_LIFETIME_MIN_DAYS` → set to None

### 2. USE_DEPOSITION_FOR_LIFETIME Option (Both notebooks)

**Files:** Both `AOD_error_attribution.ipynb` and `AAOD_error_attribution.ipynb`

**Added configuration:**
```python
USE_DEPOSITION_FOR_LIFETIME = False  # True: tau=load/deposition, False: tau=load/emission
```

**Purpose:**  
Allow switching between emission-based and deposition-based lifetime calculations:

- **Emission-based (default):**
  - AAOD: `lifetime = (loadoa + loadbc) / (emioa + emibc)`
  - AOD: `lifetime = load_total / emi_total`

- **Deposition-based (new option):**
  - AAOD: `lifetime = (loadoa + loadbc) / (depoa + depbc)`
  - AOD: `lifetime = load_total / dep_total`

**Implementation:**  
Updated lifetime calculations in `compute_derived_after_aggregation()` to check the flag:
```python
if USE_DEPOSITION_FOR_LIFETIME:
    flux = agg[region].get('dep_total', {})  # or dep_BC_OA
else:
    flux = agg[region].get('emi_total', {})  # or emi_BC_OA
out[model] = load[model] / (flux[model] * 3600 * 24)
```

## Files Modified

1. **AOD_error_attribution.ipynb**
   - Cell 3: Added POST_AGG config + USE_DEPOSITION config + print statements
   - Cell 9: Updated `compute_derived_after_aggregation()` with both features

2. **AAOD_error_attribution.ipynb**
   - Cell 2: Added USE_DEPOSITION config + print statement  
     (POST_AGG already existed at 20 days)
   - Cell 10: Updated `compute_derived_after_aggregation()` with deposition option

## Backups Created

- `AOD_error_attribution.ipynb.backup_pre_lifetime_fix`
- `AAOD_error_attribution.ipynb.backup_pre_deposition_opt`

## Required Data Variables

For deposition-based lifetime to work, the following variables must exist in NetCDF files:

### AAOD notebook:
- `depoa` - OA deposition flux
- `depbc` - BC deposition flux  
  (or alternatives: `drydepoa`, `wetdepoa`, etc.)

### AOD notebook:
- `dep_total` - Total aerosol deposition flux  
  (or alternatives: `drydepaer`, `wetdepaer`, etc.)

**Note:** If these variables don't exist, the deposition option will silently fall back to empty dictionaries (no models will have lifetime calculated).

## Testing Needed

1. **Run AOD notebook with default settings:**
   ```python
   POST_AGG_LIFETIME_MAX_DAYS = 15.0
   USE_DEPOSITION_FOR_LIFETIME = False
   ```
   - Verify models 3, 10, 16 no longer blow up
   - Check error decomposition percentages are reasonable

2. **Test deposition option (both notebooks):**
   ```python
   USE_DEPOSITION_FOR_LIFETIME = True
   ```
   - Verify deposition variables exist in data
   - Check lifetime values are sensible
   - Compare emission-based vs deposition-based results

3. **Verify print output shows config:**
   ```
   POST_AGG_LIFETIME    = [0.3, 15.0] days
   USE_DEPOSITION_FOR_LIFETIME = False  (load/emission)
   ```

## Expected Behavior

- **AOD notebook:** Extreme lifetimes (>15 days or <0.3 days) after regional aggregation are now masked, preventing error decomposition blow-ups
- **Both notebooks:** Can now calculate lifetime using deposition flux instead of emission flux when flag is enabled
- **AAOD notebook:** Already had POST_AGG filtering at 20 days (now also has deposition option)

## Next Steps

1. Verify deposition variable availability in actual NetCDF files
2. Test notebooks end-to-end with both flag settings
3. Document which models have deposition data available
4. Consider adding diagnostic output showing which method was used per model

