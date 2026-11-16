# AUTHOR : J.ZHANG TNO 2025
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from dask import delayed
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import sys
##########INPUT##########

run = 'DUc20'
var_key = ['aod', 'ssa',  'aod_uncertainty','ssa_uncertainty','bcaod','bcssa','omaod','omssa','duaod','ssaod','suaod']
# var_key = [ 'aod_uncertainty','ssa_uncertainty','aod', 'ssa',]
wvl = [ 340, 355, 380, 440, 500, 532, 550, 670, 865, 1020, 1064]  
# wvl = [ 340, 440, 550, 865 ]
Flag = 'OCEAN'
########START###########
OOPP_OUT = '/tsn.tno.nl/RA-Data/Express/modascameo_unix/CAMEO_WP1/users/tokayajp/OOPP'
exp_oopp= {
    'standard' : f'{OOPP_OUT}/experiment_woodward_grid/output__input_cells_all',
    '2xBC'    : f'{OOPP_OUT}/experiment_double_BC_SSA_grid/output__input_cells_all',
    'OMP002'   : f'{OOPP_OUT}/experiment_OM_p002_grid/output__input_cells_all',
    'OMP004'   : f'{OOPP_OUT}/experiment_OM_p004_grid/output__input_cells_all',
    'BC_OMp004': f'{OOPP_OUT}/experiment_double_BC_SSA_OM_p004_grid/output__input_cells_all',
    'DU_dubovik'  : f'{OOPP_OUT}/experiment_dubovik_grid/output__input_cells_all',
    'Composite' : f'{OOPP_OUT}/experiment_composite_5wvl_grid/output__input_cells_all',
    'BCOMp004_dubovik':  f'{OOPP_OUT}/experiment_double_BC_SSA_OM_p004_grid_dubovik/output__input_cells_all',
    'BC2OC'     : f'{OOPP_OUT}/experiment_25percEC2OC/output__input_cells_all',
    'all_comp'  :f'{OOPP_OUT}/experiment_spexwvl_perspecie_*/output__input_cells_all',
    'SdOMU20'      :f'{OOPP_OUT}/experiment_20su/output__input_cells_all',
    'DUu20'     :f'{OOPP_OUT}/experiment_dust_u20/output__input_cells_all',
    'DUf20'     :f'{OOPP_OUT}/experiment_dust_f20/output__input_cells_all',
    'DUc20'     :f'{OOPP_OUT}/experiment_dust_c20/output__input_cells_all',
    'SSu20'     :f'{OOPP_OUT}/experiment_ss_u20/output__input_cells_all',
    'SSf20'     :f'{OOPP_OUT}/experiment_ss_f20/output__input_cells_all',
    'SSc20'     :f'{OOPP_OUT}/experiment_ss_c20/output__input_cells_all',
    'dBC20'     :f'{OOPP_OUT}/experiment_dbc20/output__input_cells_all',
    'wBC20'     :f'{OOPP_OUT}/experiment_wbc20/output__input_cells_all',
    'dOM20'     :f'{OOPP_OUT}/experiment_dom20/output__input_cells_all',
    'wOM20'     :f'{OOPP_OUT}/experiment_wom20/output__input_cells_all'
}
oopp_output_dir = exp_oopp[run]
oopp_files = glob.glob(f'{oopp_output_dir}/OOPP_optics_*')

spexONE_folder ='/tsn.tno.nl/Data/SV/sv-059025_unix/projects/EU/CAMEO/WP1/users/zhangj1/SPEXone'
if len(sys.argv) != 2:
    print("Usage: python script.py <month_number>")
    sys.exit(1)
month = int(sys.argv[1])
round_time = '3H'

spex_files = glob.glob(f'{spexONE_folder}/Extract_PACE_SPEXONE_L2_AER_RTAP{Flag}_2024_{month:02d}*')

model_vars = [f"{key}_{wl}nm" for key in var_key for wl in wvl]

for spex_file in spex_files:
    # spex_file = spex_files[0]
    output_parquet = f'//tsn.tno.nl/Data/SV/sv-059025_unix/projects/EU/CAMEO/WP1/users/zhangj1/SPEXone/OOPP_coloc_{round_time}_{run}_all_SPEXONE_L2_AER_RTAP{Flag}_2024_{month:02d}.parquet'
    # Remove output if exists
    if os.path.exists(output_parquet):
        os.remove(output_parquet)

    df_obs = pd.read_parquet(spex_file)
    unique_times = pd.to_datetime(df_obs['date']).dt.round(round_time).unique()
    pqwriter = None
    unique_dates=np.unique(unique_times.strftime('%Y%m%d'))
    in_file_1= oopp_files[0]
    in_data_1 = xr.open_dataset(in_file_1, engine='netcdf4')
    # Get bounds from NetCDF file
    min_lon = float(in_data_1['longitude'].min().values)
    max_lon = float(in_data_1['longitude'].max().values)
    min_lat = float(in_data_1['latitude'].min().values)
    max_lat = float(in_data_1['latitude'].max().values)

    # Filter the dataframe
    df_obs = df_obs[
        (df_obs['lon'] >= min_lon) & (df_obs['lon'] <= max_lon) &
        (df_obs['lat'] >= min_lat) & (df_obs['lat'] <= max_lat)
    ]
    in_data_1.close()
    # if os.path.exists(output_parquet):
    #     os.remove(output_parquet)
    for date in unique_dates:
        # Ensure schema consistency by predefining output columns
        print(f"Running {date}")
        for obs_time in unique_times:
            date_str = obs_time.strftime('%Y%m%d')
            if date_str != date:
                continue
            # Ensure obs_time is UTC and tz-naive
            if obs_time.tzinfo is not None:  # tz-aware
                obs_time_naive = obs_time.tz_convert("UTC").tz_localize(None)
            else:  # tz-naive
                obs_time_naive = obs_time

            # Ensure df_obs['date'] is UTC and tz-naive
            df_times = pd.to_datetime(df_obs['date']).dt.round(round_time)
            if df_times.dt.tz is not None:  # tz-aware
                df_times = df_times.dt.tz_convert("UTC").dt.tz_localize(None)

            # Apply mask
            mask = df_times == obs_time_naive
            # print(f"Running {date_str}")
            # mask = pd.to_datetime(df_obs['date']).dt.round('H').dt.tz_convert('UTC') == obs_time.tz_localize('UTC')
            # Skip if no matches
            if not mask.any():
                # print('no match')
                continue  

            df_sub = df_obs[mask].copy()

            # Pre-fill all model columns with NaNs to ensure consistent schema
            for var in model_vars:
                df_sub[f"oopp_{var}"] = np.nan

            points = np.vstack([df_sub['lat'], df_sub['lon']]).T
            valid = np.ones(len(points), dtype=bool)

            # for var in model_vars:
            for v in var_key:
                for w in wvl: 
                    mod_f_key = f'OOPP_optics_{date_str}.nc' 
                    var = f"{v}_{w}nm"
                    
                    if run == 'all_comp' and w <= 440:
                        oopp_files = glob.glob(f'{OOPP_OUT}/experiment_spexwvl_perspecie_swl/output__input_cells_all/{mod_f_key}')
                    elif run == 'all_comp' and w <= 670:
                        oopp_files = glob.glob(f'{OOPP_OUT}/experiment_spexwvl_perspecie_mwl/output__input_cells_all/{mod_f_key}')
                    elif run == 'all_comp' and w > 670:
                        oopp_files =glob.glob( f'{OOPP_OUT}/experiment_spexwvl_perspecie_lwl/output__input_cells_all/{mod_f_key}')
                    else: 
                        oopp_files = glob.glob(f'{oopp_output_dir}/{mod_f_key}')

                    if len(oopp_files) != 1:
                        print(f"Missing or multiple model file(s): {mod_f_key}")
                        continue
                    try:
                        ds_model = xr.open_dataset(oopp_files[0], engine='netcdf4')
                        model_time = pd.to_datetime(ds_model['time'].values).round('S').tz_localize('UTC')

                        if obs_time not in model_time:
                            print(f"No matching time for {obs_time} in {oopp_files[0]}")
                            ds_model.close()
                            continue

                        model_data = ds_model[var].sel(time=np.datetime64(obs_time.tz_convert(None)), method='nearest')

                        interpolator = RegularGridInterpolator(
                            (model_data['latitude'].values, model_data['longitude'].values),
                            model_data.values,
                            method='linear',
                            bounds_error=False,
                            fill_value=np.nan
                        )

                        interp_vals = interpolator(points)
                        df_sub[f"oopp_{var}"] = interp_vals
                        valid &= ~np.isnan(interp_vals)

                        ds_model.close()

                    except Exception as e:
                        print(f"Error processing {oopp_files[0]}: {e}")
                        continue

            df_final = df_sub[valid]

            if df_final.empty:
                continue

            # Always same schema now
            table = pa.Table.from_pandas(df_final)

            if pqwriter is None:
                pqwriter = pq.ParquetWriter(output_parquet, table.schema, use_dictionary=True, compression='snappy')

            pqwriter.write_table(table)

    if pqwriter:
        pqwriter.close()
    print("SPEXone observation data saved to:", output_parquet)
