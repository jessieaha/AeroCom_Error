"""
Script to convert POLDER GRASP NetCDF files to Parquet, and subsequently
collocate AP3 Model data (AOD/AAOD) to the observational spatial-temporal tracks.

Author: Jessie Zhang 07 2026 
"""

import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
import cftime
from scipy.interpolate import RegularGridInterpolator
import pyarrow as pa
import pyarrow.parquet as pq

# Reuse the central longitude normalization so every model shares the POLDER grid
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from py.aerocom_data import normalize_longitude

# =============================================================================
# SETTINGS & CONFIGURATIONS
# =============================================================================
YEAR = 2010
ROUND_TIME = '3h'  
spatial_res_lat = 1.0          # Choose your grid box size (e.g., 0.1, 0.25, 0.5, 1.0)
spatial_res_lon = 1.0          # Choose your grid box size (e.g., 0.1, 0.25, 0.5, 1.0)
project_root = '.'  # Set your project root directory path here

# control flags
Convert_POLDER_Parquet = False  # True: Convert raw NC files to Parquet | False: Direct load existing parquets
spatial_temp_resample = True       # True: Aggregate observations inside the same space-time window
temporal_err = 'sd'            # Error calculation mode: 'sd' (standard deviation) or 'range' (max - min)
SUBSET_N_MONTHS = int(os.environ.get('SUBSET_N_MONTHS', 0)) or None  # For quick tests: process only first N months

polder_dir = f'{project_root}/Data/POLDER_GRASP'
model_dir = f'{project_root}/Data/AP3_processed_3hourly/'     
output_folder = f'{project_root}/Data/AP3_POLDER_Collocated'
os.makedirs(output_folder, exist_ok=True)
if spatial_temp_resample:
    output_parquet = f'{output_folder}/POLDER_GRASP_coloc_{ROUND_TIME}_AP3_{YEAR}_lon{spatial_res_lon}_lat{spatial_res_lat}.parquet'
else:
    output_parquet = f'{output_folder}/POLDER_GRASP_coloc_{YEAR}.parquet'
# Mapping input NetCDF variables to final DataFrame columns
obs_rename_map = {
    'od440aer': 'AOD_440',
    'abs550aer': 'AAOD_550',
    'od550aer': 'AOD_550',
    'od865aer': 'AOD_870',
    'od870aer': 'AOD_870'
}

# =============================================================================
# STEP 1 & 2: READ/LOAD POLDER TRACKING DATA
# =============================================================================
df_list = []

if Convert_POLDER_Parquet:
    print("--- [STEP 1 & 2] CONVERTING RAW POLDER NC FILES TO PARQUET ---")
    polder_nc_files = glob.glob(os.path.join(polder_dir, f'*{YEAR}*.nc'))
    if not polder_nc_files:
        raise FileNotFoundError(f"No valid POLDER NetCDF files found in {polder_dir} containing '{YEAR}'.")
        
    for nc_file in sorted(polder_nc_files):
        parq_file = nc_file.replace('.nc', '.parquet')
        print(f"Processing & Converting: {os.path.basename(nc_file)}")
        try:
            with xr.open_dataset(nc_file) as ds:
                df = ds.to_dataframe().reset_index()
                if 'data_length' in df.columns:
                    df = df.drop(columns=['data_length'])
                
                rename_dict = {k: v for k, v in obs_rename_map.items() if k in df.columns}
                df = df.rename(columns=rename_dict)
                df = df.dropna(subset=['longitude', 'latitude', 'time'])
                
                df.to_parquet(parq_file, index=False)
                df_list.append(df)
        except Exception as e:
            print(f"  ❌ Error processing {nc_file}: {e}")
else:
    print("--- [STEP 1 & 2] BYPASSING CONVERSION: LOADING EXISTING PARQUETS ---")
    polder_parq_files = glob.glob(os.path.join(polder_dir, f'*{YEAR}*.parquet'))
    # Safety Check: filter out the master collocation summary file if it sits in the same directory
    polder_parq_files = [f for f in polder_parq_files if "coloc" not in os.path.basename(f)]
    
    if not polder_parq_files:
        raise FileNotFoundError(f"No pre-existing POLDER Parquet files found in {polder_dir} matching '{YEAR}'.")
        
    for parq_file in sorted(polder_parq_files):
        print(f"Loading Parquet track: {os.path.basename(parq_file)}")
        try:
            df = pd.read_parquet(parq_file)
            df_list.append(df)
        except Exception as e:
            print(f"  ❌ Error loading {parq_file}: {e}")

# Combine all loaded data blocks
df_obs = pd.concat(df_list, ignore_index=True)

# Generate tracking rounded timestamps
df_obs['rounded_time'] = pd.to_datetime(df_obs['time']).dt.round(ROUND_TIME)

# Optional subset for quick testing without overwriting the full-year output
if SUBSET_N_MONTHS:
    df_obs = df_obs[df_obs['rounded_time'].dt.month <= SUBSET_N_MONTHS].copy()
    output_parquet = output_parquet.replace('.parquet', f'_subset{SUBSET_N_MONTHS}mo.parquet')
    print(f"SUBSET MODE: processing first {SUBSET_N_MONTHS} month(s) -> {len(df_obs)} rows")

# =============================================================================
# SPATIO-TEMPORAL AGGREGATION & ERROR METRICS ENGINE  
# =============================================================================
if spatial_temp_resample:
    print(f"\n Resampling tracks to {ROUND_TIME} and Spatial Res: ({spatial_res_lat}x{spatial_res_lon} degrees)...")
    
    # Mathematically snap coordinates to the nearest custom grid center
    df_obs['grid_lat'] = np.round(df_obs['latitude'] / spatial_res_lat) * spatial_res_lat
    df_obs['grid_lon'] = np.round(df_obs['longitude'] / spatial_res_lon) * spatial_res_lon
    
    val_cols = list(dict.fromkeys(v for v in obs_rename_map.values() if v in df_obs.columns))
    groupby_keys = ['rounded_time', 'grid_lat', 'grid_lon']

    # Treat zero AOD/AAOD as missing/invalid (aerosol optical depths are positive)
    for col in val_cols:
        df_obs[col] = df_obs[col].mask(df_obs[col] == 0)

    if temporal_err == 'sd':
        err_func = 'std'
    elif temporal_err == 'range':
        err_func = lambda x: (x.max() - x.min()) if x.notna().sum() > 1 else np.nan
    else:
        raise ValueError("Invalid setting for 'temporal_err'. Choose either 'sd' or 'range'.")
        
    # Aggregate data inside the custom spatio-temporal grid boxes
    df_grouped_mean = df_obs.groupby(groupby_keys)[val_cols].mean()
    df_grouped_err = df_obs.groupby(groupby_keys)[val_cols].agg(err_func)
    df_grouped_err = df_grouped_err.rename(columns={c: f"{c}_{ROUND_TIME}_err" for c in val_cols})

    # Zero spread is not a valid uncertainty estimate; treat as missing
    df_grouped_err = df_grouped_err.mask(df_grouped_err == 0)

    # Merge means and errors together
    df_obs = pd.concat([df_grouped_mean, df_grouped_err], axis=1).reset_index()
    
    # Map back to standard headers so downstream engines know where to interpolate
    df_obs = df_obs.rename(columns={'grid_lat': 'latitude', 'grid_lon': 'longitude'})
    df_obs['time'] = df_obs['rounded_time']  

# Final drop of any duplicated columns
df_obs = df_obs.loc[:, ~df_obs.columns.duplicated()]
unique_times = df_obs['rounded_time'].unique()

print(f"Total aggregated tracking points to collocate: {len(df_obs)}")
print(f"Total unique timesteps to match: {len(unique_times)}")

if os.path.exists(output_parquet):
    os.remove(output_parquet)

# =============================================================================
# STEP 3: DYNAMICALLY DISCOVER & LAZY-LOAD MODEL DATASETS
# =============================================================================
print("\n--- [STEP 3] DISCOVERING & LAZY-LOADING AP3 MODEL DATA ---")
model_datasets = {}
all_discovered_models = set()

for var in obs_rename_map.keys():
    var_files = glob.glob(os.path.join(model_dir, var, f"*_{var}_processed.nc"))
    
    for mod_file in var_files:
        filename = os.path.basename(mod_file)
        suffix = f"_{var}_processed.nc"
        
        if filename.endswith(suffix):
            model_name = filename[:-len(suffix)]
            if model_name not in model_datasets:
                model_datasets[model_name] = {}
            
            all_discovered_models.add(model_name)
            try:
                model_datasets[model_name][var] = xr.open_dataset(mod_file, chunks='auto')
                print(f" Loaded model pointer: {model_name} -> {var}")
            except Exception as e:
                print(f" Failed to load model file {mod_file}: {e}")
                model_datasets[model_name][var] = None

all_discovered_models = sorted(list(all_discovered_models))
print(f"\n=> Total unique models found for collocation: {len(all_discovered_models)}")
# =============================================================================
# EXPORT MODEL NAMES TO A TEXT FILE
# =============================================================================
models_txt_path = os.path.join(output_folder, f"discovered_models_{YEAR}.txt")
with open(models_txt_path, "w") as f:
    for m in all_discovered_models:
        f.write(f"{m}\n")
print(f"Exported discovered model names to text file: {models_txt_path}")

if os.path.exists(output_parquet):
    os.remove(output_parquet)
# =============================================================================
# STEP 4: INTERPOLATION ENGINE
# =============================================================================

print("\n--- [STEP 4] RUNNING INTERPOLATION ENGINE ---")
pqwriter = None

for obs_time in unique_times:
    mask = df_obs['rounded_time'] == obs_time
    df_sub = df_obs[mask].copy()
    
    allocated_cols = set()
    for model in all_discovered_models:
        for var, out_var in obs_rename_map.items():
            if model_datasets[model].get(var) is not None:
                col_name = f"{out_var}_{model}"
                if col_name not in allocated_cols:
                    df_sub[col_name] = np.nan
                    allocated_cols.add(col_name)

    # Base track points from POLDER data (Longitude is 0 to 360)
    base_points = np.vstack([df_sub['latitude'], df_sub['longitude']]).T

    for model in all_discovered_models:
        for var, out_var in obs_rename_map.items():
            ds_model = model_datasets[model].get(var)
            if ds_model is None:
                continue
                
            try:
                # Some models use non-standard calendars (e.g. noleap, julian). Convert
                # the target POLDER time to the model's calendar before selection.
                time_cal = ds_model['time'].encoding.get('calendar')
                if time_cal:
                    target_time = cftime.datetime(
                        obs_time.year, obs_time.month, obs_time.day,
                        obs_time.hour, obs_time.minute, obs_time.second,
                        calendar=time_cal
                    )
                else:
                    target_time = obs_time

                model_slice = ds_model[var].sel(time=target_time, method='nearest').compute()
                actual_time = np.asarray(model_slice.time.values).item()

                # Robust time-difference check across standard and cftime calendars
                if hasattr(actual_time, 'calendar'):
                    obs_cftime = cftime.datetime(
                        obs_time.year, obs_time.month, obs_time.day,
                        obs_time.hour, obs_time.minute, obs_time.second,
                        calendar=actual_time.calendar
                    )
                    time_diff = abs((obs_cftime - actual_time).total_seconds())
                else:
                    actual_time = pd.to_datetime(actual_time)
                    time_diff = abs((actual_time - obs_time).total_seconds())

                if time_diff > 86400 * 7:
                    continue

                # Normalize the model slice to the same 0-360 longitude grid as POLDER
                model_slice = normalize_longitude(model_slice)
                model_points = base_points.copy()
                # Wrap any observation longitudes at 360 back to 0
                model_points[:, 1] = np.mod(model_points[:, 1], 360.0)

                interpolator = RegularGridInterpolator(
                    (model_slice['lat'].values, model_slice['lon'].values),
                    model_slice.values,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )

                # Run the interpolator on the normalized 0-360 grid
                df_sub[f"{out_var}_{model}"] = interpolator(model_points)

            except Exception as e:
                pass

    # Clean up processing loops and remove duplicate schema names
    df_sub = df_sub.drop(columns=['rounded_time'], errors='ignore')
    df_sub = df_sub.loc[:, ~df_sub.columns.duplicated()]

    # Mask any remaining zero data values (observations, errors, or interpolated model)
    # as missing. Aerosol optical depths are strictly positive; zero = fill/invalid.
    data_cols = [c for c in df_sub.columns if c not in ['time', 'longitude', 'latitude']]
    df_sub[data_cols] = df_sub[data_cols].mask(df_sub[data_cols] == 0)

    # Force specific column arrangement (time -> longitude -> latitude)
    leading_cols = ['time', 'longitude', 'latitude']
    existing_lead = [c for c in leading_cols if c in df_sub.columns]
    remaining_cols = [c for c in df_sub.columns if c not in existing_lead]
    df_sub = df_sub[existing_lead + remaining_cols]

    if not df_sub.empty:
        table = pa.Table.from_pandas(df_sub)
        if pqwriter is None:
            pqwriter = pq.ParquetWriter(output_parquet, table.schema, use_dictionary=True, compression='snappy')
        pqwriter.write_table(table)

if pqwriter:
    pqwriter.close()

for model_dict in model_datasets.values():
    for ds in model_dict.values():
        if ds is not None:
            ds.close()

print("\n=============================================")
print(f"PROCESS COMPLETE!")
print(f"Collocated Master Year File Saved to: {output_parquet}")
print("=============================================\n")