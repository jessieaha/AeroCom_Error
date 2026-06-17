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


###################### PICKLE FILES ######################

def save_pickle_files(path: object, file_name: object, var_name: object) -> object:
    with open(path+file_name, 'wb') as f:
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

def create_mask_weigth(data, lon_initial, lon_final, lat_initial, lat_final):

    """ Creates a mask given a certain condition (1 condition fulfilled, 0 otherwise).
        For the cells at the edges, takes into account the % of the area that is inside the domain.
        The function works with lon values between -180,180 but the input values have to be between 0,360.

    :param data: data array containing a field defined over latitudes, longitudes and time.
    :param lon_initial: initial longitude of the considered region, starting from 0º (float).
    :param lon_final: final longitude of the considered region, ending at 360º (float).
    :param lat_initial: initial latitude of the considered region, starting from -90N (float).
    :param lat_final: final latitude of the considered region, ending at 90N (float).

    :returns: data array with the same shape as the original one but containing only 1 or 0.
                It is only defined over latitude and longitude (does not have the time dimension).
                This array should be multiplied with the data array to apply the mask.
    """

    data_ones = xr.ones_like(data)

    dlat = np.abs(np.diff(data.lat.data)).mean()
    dlon = np.abs(np.diff(data.lon.data)).mean()

    # Case where the range wraps around 360
    if lon_initial > lon_final:
        mask_lon = data_ones.where((data.lon >= lon_initial) | (data.lon <= lon_final), 0)

    else:
        mask_lon = data_ones.where((data.lon >= lon_initial) & (data.lon <= lon_final), 0)

    mask_lat = mask_lon.where((data.lat >= lat_initial) & (data.lat <= lat_final), 0)

    # special case: edges
    mask1 = mask_lat.where(~((np.abs(lat_initial - mask_lat.lat) < dlat / 2) &
                             (np.abs(lon_initial - mask_lat.lon) < dlon / 2)),
                           ((mask_lat.lat + dlat / 2 - lat_initial) / dlat) * (dlon / 2 + mask_lat.lon - lon_initial) / dlon)

    mask2 = mask1.where(~((np.abs(lat_initial - mask1.lat) < dlat / 2) &
                          (np.abs(lon_final - mask1.lon) < dlon / 2)),
                        ((mask1.lat + dlat / 2 - lat_initial) / dlat) * ((lon_final + dlon / 2 - mask1.lon) / dlon))

    mask3 = mask2.where(~((np.abs(lat_final - mask2.lat) < dlat / 2) &
                          (np.abs(lon_initial - mask2.lon) < dlon / 2)),
                        ((-mask2.lat + dlat / 2 + lat_final) / dlat) * ((dlon / 2 + mask2.lon - lon_initial) / dlon))

    mask4 = mask3.where(~((np.abs(lat_final - mask3.lat) < dlat / 2) &
                          (np.abs(lon_final - mask3.lon) < dlon / 2)),
                        ((-mask3.lat + dlat / 2 + lat_final) / dlat) * ((lon_final + dlon / 2 - mask3.lon) / dlon))

    mask_low = mask4.where(~((np.abs(lat_initial - mask4.lat) < dlat / 2) &
                             (lon_initial <= mask4.lon - dlon / 2) & (lon_final >= mask4.lon + dlon / 2)),
                           ((mask4.lat + dlat / 2 - lat_initial) / dlat))

    mask_up = mask_low.where(~((np.abs(lat_final - mask_low.lat) < dlat / 2) &
                               (lon_initial <= mask_low.lon - dlon / 2) & (lon_final >= mask_low.lon + dlon / 2)),
                             ((-mask_low.lat + dlat / 2 + lat_final) / dlat))

    mask_left = mask_up.where(~((lat_initial <= mask_up.lat - dlat / 2) & (lat_final >= mask_up.lat + dlat / 2) &
                                (np.abs(lon_initial - mask_up.lon) < dlon / 2)),
                              ((dlon / 2 + mask_up.lon - lon_initial) / dlon))

    mask_right = mask_left.where(~((lat_initial <= mask_left.lat - dlat / 2) & (lat_final >= mask_left.lat + dlat / 2) &
                                   (np.abs(lon_final - mask_left.lon) < dlon / 2)),
                                 ((lon_final + dlon / 2 - mask_left.lon) / dlon))

    return mask_right

def create_mask(data, lon_initial, lon_final, lat_initial, lat_final):

    """ Same as 'create_mask_weigth(data, lon_initial, lon_final, lat_initial, lat_final)',
    but now the cells at the edges are either 0 (the cell center is outside the region)
    or 1 (the cell center is inside the region).
    """

    data_ones = xr.ones_like(data)

    if lon_initial > lon_final:
        mask_lon = data_ones.where((data.lon >= lon_initial) | (data.lon <= lon_final), 0)

    else:
        mask_lon = data_ones.where((data.lon >= lon_initial) & (data.lon <= lon_final), 0)

    mask_lat = mask_lon.where((data.lat >= lat_initial) & (data.lat <= lat_final), 0)

    return mask_lat


###################### DATA HANDELING ######################

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