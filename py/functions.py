""" Script containing functions to treat, handle and visualize the data.

FRdM, 10th of January 2024 """


###################### IMPORT MODULES ######################

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cftime
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from scipy import stats
import pickle
from pathlib import Path


###################### PICKLE FILES ######################

def save_pickle_files(path: str, file_name: str, var_name: object) -> None:
    # Convert string path to a Path object
    dir_path = Path(path)
    
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save the pickle file
    with open(dir_path / file_name, 'wb') as f:
        pickle.dump(var_name, f)

def open_pickle_files(file_path, file_name):
    with open(file_path+file_name, 'rb') as file:
        var_name = pickle.load(file)
    return var_name


###################### GLOBAL MEAN ######################

def global_mean(da, mask):

    """ Calculates the global annual mean computed with correct weight factors.
        If mask is provided, it performs a regional analysis.

    :param da: data array containing a field defined over latitudes, longitudes and time.
                Select the variable we are interested in (eg: dictionary_name.emibc).
    :param mask: data array containing the corresponding mask.

    :returns: data array with the corresponding annual mean of the field. If plotted, it is a time series.
    """

    # detect if data has "extra" dimensions (dimensions that have no coordinates) and drop them
    if 'bnds' in list(da.dims):
        da = da.drop_dims('bnds', errors='raise')
    elif 'nbnd' in list(da.dims):
        da = da.drop_dims('nbnd', errors='raise')
    else:
        da = da

    # determine the grid size
    dlat = np.abs(np.deg2rad(np.diff(da.lat.data))).mean()
    dlon = np.abs(np.deg2rad(np.diff(da.lon.data))).mean()

    # determine normalization factor
    norm_factor = (np.cos(np.deg2rad(da.lat))*mask*dlat*dlon).sum()

    # determine weight
    weight = np.cos(np.deg2rad(da.lat))*mask*dlat*dlon / norm_factor

    # put everything together
    da_mean = (da*weight).sum(dim=['lat', 'lon'])

    return da_mean

def apply_global_mean(dict, virtual_var, time_series=False):

    """ Applies the function 'global_mean' to every model in the dictionary 'dict'.
        Depending on the region (i.e. the mask), it selects the corresponding fire season.
        If requested, it gives the mean of the time series.

    :param dict: dictionary containing the data and the masks. Every key has to be a different model.
    :param virtual_var: name of the virtual variable to average (string).
    :param time_series: dictionary containing the months to look at.
    :param time_series: True --> one value per month (if plotted, it's a time series)
                        False --> mean value over all the months

    :returns: dictionary with the corresponding averaged values for each region (result1[region][model] = xarray)
    """

    time_span_list = {
        'global': slice('2010-01-01', '2010-12-31'),
        'africa': slice('2010-06-01', '2010-09-30'),
        'amazon': slice('2010-07-01', '2010-10-31'),
        'outflow_af': slice('2010-06-01', '2010-09-30')}
        # 'north_am': slice('2010-06-01', '2010-08-31'),
        # 'siberia': slice('2010-07-01', '2010-07-31'),
        # 'se_asia': slice('2010-03-01', '2010-03-31')}

    # Initialize the result dictionary
    result = {mask: {} for model in dict for mask in dict[model]['masks'].keys()}
    result_mean = {mask: {} for model in dict for mask in dict[model]['masks'].keys()}

    # Loop through each model and mask to calculate the mean
    for model, model_data in dict.items():
        for mask in model_data['masks'].keys():

            result[mask][model] = global_mean(model_data[virtual_var].sel(time=time_span_list[mask]), model_data['masks'][mask])
            result_mean[mask][model] = float(result[mask][model].mean())

    if time_series is True:
        return result

    else:
        return result_mean


###################### MASKS ######################

def shift360(data):

    """ Shifts the longitude from -180,180 to 0,360.
    Useful for the AeroCom data.

    :param data: data array defined over a field of latitude and longitude.

    :returns: original data array but with the 180 longitude at the center.
    """

    # data_shifted = data.copy(deep=True)
    data_shifted = data.copy()

    # update the 'lon' coordinate values (change the actual lon values)
    data_shifted['lon'] = (data_shifted['lon']+360) % 360

    # sort 'lon' values in ascending order
    data_shifted = data_shifted.sortby('lon')

    return data_shifted


# The region-mask, box-mask and land/sea helpers below are now maintained in
# cameo_toolbox.py.  They are re-exported here for backward compatibility with
# notebooks that import from functions.py.
from cameo_toolbox import (
    create_mask_weigth,
    create_mask,
    create_region_mask,
    regional_aggregate,
    aggregate_models,
    land_mask_from_cartopy,
)

###################### DATA HANDELING ######################

def normalize_monthly_time(da):
    """Normalize the time coordinate of a monthly DataArray to first-of-month.

    AeroCom models use different timestamp conventions for monthly averages:
    - mid-month  (e.g. 2010-01-16): represents January 2010
    - end-of-month (e.g. 2010-01-31): represents January 2010
    - next-month-start (e.g. 2010-02-01 for January): represents January 2010
    - numeric year-month (e.g. 201001): represents January 2010

    All are mapped to the first day of the represented month (2010-01-01).
    This ensures consistent time coordinates when dividing DataArrays from
    different variables or models (e.g. AOD / load for MEC, or abs / AOD for SSA).

    Parameters
    ----------
    da : xr.DataArray
        Monthly data with a 'time' dimension using datetime64[ns] coordinates.

    Returns
    -------
    xr.DataArray
        Same data with time coordinate normalised to first-of-month.
    """
    import pandas as pd

    if 'time' not in da.dims or len(da.time) == 0:
        return da

    raw = da.time.values
    if raw.dtype.kind in 'iuf' and np.all(raw > 100000) and np.all(raw < 999999):
        # Numeric year-month convention (e.g. GEOS-i33p2 stores 201001 ... 201012)
        years = (raw // 100).astype(int)
        months = (raw % 100).astype(int)
        times = pd.to_datetime([f'{y:04d}-{m:02d}-01' for y, m in zip(years, months)])
    else:
        times = pd.DatetimeIndex(raw)

    # Detect "next-month-start" convention:
    # All timestamps land on the 1st (possibly with sub-day offsets), and the
    # series covers exactly 12 months ending in January of the following year
    # (e.g. Feb 2010 → Jan 2011 represents Jan–Dec 2010).
    # Files with 13 or more time steps are NOT shifted — the extra boundary
    # value is simply excluded by the time_slice in regional_aggregate.
    all_first = all(t.day == 1 for t in times)
    is_next_month_start = (
        all_first
        and len(times) == 12
        and times[-1].year == times[0].year + 1
        and times[-1].month == 1
    )

    if is_next_month_start:
        # Shift back one month and truncate to midnight.
        new_times = [pd.Timestamp(t.year, t.month, 1) - pd.DateOffset(months=1) for t in times]
    else:
        # Mid-month, end-of-month, or start-of-month with sub-day offsets:
        # truncate to midnight of the 1st of the same month.
        new_times = [pd.Timestamp(t.year, t.month, 1) for t in times]

    return da.assign_coords(
        time=pd.DatetimeIndex(new_times).values.astype('datetime64[ns]')
    )


def convert_cftime_to_datetime(data_set):

    """ Converts the cftime time axis to datetime64 objects.

    :param data_set: data set containing the data.

    :returns: same data as the original one but with a datetime64 time axis.
    """

    if isinstance(data_set.time.data[0], cftime.datetime):
        data_set['time'] = xr.DataArray(np.array(data_set.time.values, dtype='datetime64[ns]'),
                                        dims=data_set.time.dims, coords={data_set.time.dims[0]: data_set.time.values})

    return data_set

def fit_data(x_data, y_data):

    """Fits the data to a linear regression model and returns all regression parameters,
    confidence intervals, and prediction intervals in a single function.

    :param x_data: ndarray of x values
    :param y_data: ndarray of y values

    :returns: dictionary with regression parameters and statistics """

    # Filter NaN values
    idx = ~np.isnan(x_data + y_data)
    x_nonan = x_data[idx]
    y_nonan = y_data[idx]

    # Fit linear model
    model = LinearRegression().fit(x_nonan.reshape(-1, 1), y_nonan)
    slope = model.coef_[0]
    intercept = model.intercept_

    # R² and Pearson coefficient
    r2 = model.score(x_nonan.reshape(-1, 1), y_nonan)
    pear_coef = pearsonr(x_nonan, y_nonan)[0]

    # Predictions and residuals
    y_pred = model.predict(x_nonan.reshape(-1, 1))
    residuals = y_nonan.flatten() - y_pred

    # Sampling x for prediction bands
    x_sampled = np.linspace(np.nanmin(x_data), np.nanmax(x_data), 100)
    y_sampled_pred = model.predict(x_sampled.reshape(-1, 1))

    # Residual sum of squares and sigma²
    rss = np.sum(residuals**2)
    sigma_squared = rss / (len(x_nonan) - 2)

    # Deviation of x from mean
    x_dev = np.sum((x_nonan - np.mean(x_nonan))**2)

    # Standard errors
    se_fit = np.sqrt(sigma_squared) * np.sqrt(1 / len(x_nonan) + (x_sampled - np.mean(x_nonan)) ** 2 / x_dev)
    se_pred = np.sqrt(sigma_squared) * np.sqrt(1 + 1 / len(x_nonan) + (x_sampled - np.mean(x_nonan)) ** 2 / x_dev)

    # Confidence and prediction intervals
    t_val = stats.t.ppf(0.975, len(x_nonan) - 2)  # 95% CI
    ci_lower = y_sampled_pred - t_val * se_fit
    ci_upper = y_sampled_pred + t_val * se_fit
    pi_lower = y_sampled_pred - t_val * se_pred
    pi_upper = y_sampled_pred + t_val * se_pred

    # Standard errors of slope and intercept
    sem = np.sqrt(sigma_squared / x_dev)
    sen = np.sqrt(sigma_squared * np.sum(x_nonan**2) / (len(x_nonan) * x_dev))

    return {
        'slope': slope, 'inter': intercept,
        'r2': r2, 'pear_coef': pear_coef,
        'pi_lower': pi_lower, 'pi_upper': pi_upper,
        'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'x_sampled': x_sampled,
        'sem': sem, 'sen': sen }

def fit_poly_data(x_data, y_data, degree):

    # Filter NaN values
    idx = ~np.isnan(x_data + y_data)
    x_nonan = x_data[idx]
    y_nonan = y_data[idx]

    # Fit polynomial model
    params = np.polyfit(x_nonan, y_nonan, degree)

    # Calculate correlation coefficient R²
    y_pred = np.polyval(params, x_nonan)
    ss_res = np.sum((y_nonan - y_pred) ** 2)
    ss_tot = np.sum((y_nonan - np.mean(y_nonan)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {'x_nonan': x_nonan, 'y_nonan': y_nonan, 'param': params, 'r2': r2}
