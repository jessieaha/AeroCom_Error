""" Script to plot linear relationships between aerosol variables (Fig. 2 of Qirui's paper from 2022).

FRdM, 8th of April 2024 """


###################### IMPORT MODULES ######################

import xarray as xr
import matplotlib.pyplot as plt
from general_scripts import functions

import importlib
importlib.reload(functions)


###################### GET VARIABLES ######################

path_original = '/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/var_files/original/0average/'
path_regrid = '/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/var_files/regrid/0average/'

prect = functions.open_pickle_files(path_original, 'prect.pickle')
lifetime_inv = functions.open_pickle_files(path_original, 'lifetime_inv.pickle')
MEC = functions.open_pickle_files(path_original, 'MEC.pickle')
AE = functions.open_pickle_files(path_original, 'AE.pickle')
MAC = functions.open_pickle_files(path_regrid, 'MAC.pickle')
SSA = functions.open_pickle_files(path_regrid, 'SSA.pickle')


###################### PRECIPITATION OBSERVATIONS: GPCP DATA ######################

# units: mm/day, var: PREC, freq: daily
prect_obs = xr.open_dataset('/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/GPCP/GPCP_1DD_v1.2_199610-201510.nc')

# create masks
masks_prect_obs = {'global': 0, 'africa': 0, 'amazon': 0, 'outflow_af': 0} #, 'north_am': 0, 'siberia': 0, 'se_asia': 0}

masks_prect_obs['global'] = xr.ones_like(prect_obs.isel(time=0))
masks_prect_obs['africa'] = functions.create_mask_weigth(prect_obs.isel(time=0), 15, 37, -15, 0)
masks_prect_obs['amazon'] = functions.create_mask_weigth(prect_obs.isel(time=0), 287, 317, -17, -3) # lon:-73,-43
masks_prect_obs['outflow_af'] = functions.create_mask_weigth(prect_obs.isel(time=0), 350, 8, -15,3)
# masks_prect_obs['north_am'] = functions.create_mask_old(prect_obs.isel(time=0), 232, 265, 55,67) # lon:-128,-95
# masks_prect_obs['siberia'] = functions.create_mask(prect_obs.isel(time=0), 156, 176, 62, 71)
# masks_prect_obs['se_asia'] = functions.create_mask(prect_obs.isel(time=0), 93, 109, 11, 22)

# select time depending on the mask + do regional mean
prect_obs_mean = {}
time_span_list = {
        'global': slice('2010-01-01', '2010-12-31'),
        'africa': slice('2010-06-01', '2010-09-30'),
        'amazon': slice('2010-07-01', '2010-10-31'),
        'outflow_af': slice('2010-06-01', '2010-09-30')}
    # 'north_am': slice('2010-06-01', '2010-08-31'),
    # 'siberia': slice('2010-07-01', '2010-07-31'),
    # 'se_asia': slice('2010-03-01', '2010-03-31')}

for mask in masks_prect_obs.keys():
    time_span = time_span_list[mask]
    prect_obs_mean[mask] = functions.global_mean(prect_obs.PREC.sel(time=time_span), masks_prect_obs[mask])

    # result: prect_obs_mean[region] = TIME SERIES of the precipitation over that region


###################### MAKE FIGURE ######################

colors_AOD = functions.open_pickle_files('/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/var_files/', 'colors_AOD.pickle')
colors_AAOD = functions.open_pickle_files('/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/var_files/', 'colors_AAOD.pickle')

b = functions.make_scatter_plot(prect['amazon'], lifetime_inv['amazon'], prect_obs_mean['amazon'].PREC.mean(), 0.15, 0.4, 0.1, 1, 'Precipitation [$mm day^{-1}$]', '1/lifetime [$day^{-1}$]', colors_AOD, 'Lifetime vs precipitation outflow africa')
plt.show(block=True)
1/b

a = functions.make_scatter_plot(SSA['outflow_af'], MAC['outflow_af'], 0.8424, 0.2, 1.6, 0.8, 1, 'SSA', 'MAC [$m^2 g^{-1}]$', colors_AAOD, 'MAC vs SSA outflow africa')
plt.show(block=True)

"""import pandas as pd
df1 = {'emi': emissions['outflow_af'],
       'lifetime': lifetime['outflow_af'],
       'MEC': MEC['outflow_af'],
       'od550': od550['outflow_af'],
       'Pr': prect['outflow_af'],
       'AE': AE['outflow_af']}
df2 = {'emi_BC_OA': emissions_BC_OA['outflow_af'],
       'lifetime_BC_OA': lifetime_BC_OA['outflow_af'],
       'MAC': MAC['outflow_af'],
       'AAOD': abs550['outflow_af'],
       'SSA': SSA['outflow_af']}
df = pd.DataFrame(df1)
dff = pd.DataFrame(df2)
"""


# HadGEM model gridcell sizes, creation of netcdf file
"""a = xr.open_dataset('/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/new_data/data_original/HadGEM3-GA7.1_AP3-CTRL2016-PD/aerocom3_HadGEM3-GA7.1_AP3-CTRL2016-PD_abs550aer_Column_2010_monthly.nc')
da = a.abs550aer
dlat = np.abs(np.deg2rad(np.diff(da.lat.data))).mean()
dlon = np.abs(np.deg2rad(np.diff(da.lon.data))).mean()

area = np.cos(np.deg2rad(da.lat))*dlat*dlon*6.37e6**2
c = np.zeros((len(a.lat), len(a.lon)))
for i in range(len(a.lat)):
    c[i, :] = area[i]

plt.contourf(da.lon, da.lat, c, levels=100)
plt.colorbar()
plt.show(block=True)
# determine normalization factor
norm_factor = (np.cos(np.deg2rad(da.lat))*dlat*dlon).sum()

# determine weight
weight = np.cos(np.deg2rad(da.lat))*dlat*dlon / norm_factor


import netCDF4
ncfile = netCDF4.Dataset('/Users/fionaromandemiguel/Desktop/grid_cell.nc',mode='w')
ncfile.title = 'Gridbox sizes'
ncfile.summary = ('Gridbox sizes for the HadGEM3-GA7.1_AP3-CTRL2016-PD model')
lat_dim = ncfile.createDimension('latitude', len(a.lat))
lon_dim = ncfile.createDimension('longitude', len(a.lon))
# tim_dim = ncfile.createDimension('time', 1)
lat = ncfile.createVariable('lat', np.float64, ('latitude'))
lat.units = 'degrees_north'
lon = ncfile.createVariable('lon', np.float64, ('longitude',))
lon.units = 'degrees_east'
# time = ncfile.createVariable('time', np.float64, ('time',))
# time.units = 'hours since 1800-01-01'
area = ncfile.createVariable('area', np.float64,(lat_dim,lon_dim))
area.units = 'm2'
lat[:] = a.lat.values
lon[:] = a.lon.values
area[:,:] = c
ncfile.close()"""


###################### COMPARE NUMERICAL VALUES ######################

