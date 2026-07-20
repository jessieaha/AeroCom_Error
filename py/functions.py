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

import pickle
from pathlib import Path

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


###################### REGION MASKS AND AGGREGATION (NOTEBOOK POST-PROCESSING) ######################

def _normalize_lon_range(lon_range):
    """Accept (lon_min, lon_max) in 0–360; None means full longitude."""
    if lon_range is None:
        return None
    lon_min, lon_max = lon_range
    return float(lon_min), float(lon_max)


def _normalize_lat_range(lat_range):
    """Accept (lat_min, lat_max); None means full latitude."""
    if lat_range is None:
        return None
    lat_min, lat_max = lat_range
    return float(lat_min), float(lat_max)


def _parse_time_slice(time_slice):
    """Convert None, slice, or (start, end) strings into an xarray time slice."""
    if time_slice is None:
        return slice(None)
    if isinstance(time_slice, slice):
        return time_slice
    if isinstance(time_slice, (list, tuple)) and len(time_slice) == 2:
        return slice(time_slice[0], time_slice[1])
    raise TypeError(
        "time_slice must be None, a slice, or a (start, end) pair of time labels"
    )


def _grid_cell_sizes(da):
    """Mean grid spacing in radians for lat and lon."""
    dlat = np.abs(np.deg2rad(np.diff(da.lat.data))).mean()
    dlon = np.abs(np.deg2rad(np.diff(da.lon.data))).mean()
    return dlat, dlon


def _spatial_template(da):
    """Reduce da to its lat/lon field for mask construction."""
    template = da
    if 'time' in template.dims:
        template = template.isel(time=0, drop=True)
    for dim in list(template.dims):
        if dim not in ('lat', 'lon'):
            template = template.isel({dim: 0}, drop=True)
    return template


def _spatial_weights(da, mask, edge_weighted=False):
    """
    Area weights for lat/lon aggregation.
    mask may be binary or fractional; edge_weighted rebuilds fractional box weights
    from mask.attrs when lon_range/lat_range are stored.

    When the mask grid does not match the model's grid (different resolutions),
    the mask is automatically rebuilt on the model's grid using the stored
    lon_range/lat_range attrs, or interpolated if those attrs are absent.
    """
    dlat, dlon = _grid_cell_sizes(da)
    data_template = _spatial_template(da)

    def _grids_match(m, t):
        return (
            len(m.lat) == len(t.lat)
            and len(m.lon) == len(t.lon)
            and np.array_equal(m.lat.values, t.lat.values)
            and np.array_equal(m.lon.values, t.lon.values)
        )

    def _rebuild_mask(template):
        """Rebuild the region mask on an arbitrary spatial template."""
        lon_min, lon_max = mask.attrs['lon_range']
        lat_min, lat_max = mask.attrs['lat_range']
        if edge_weighted:
            rm = create_mask_weigth(template, lon_min, lon_max, lat_min, lat_max)
        else:
            rm = create_mask(template, lon_min, lon_max, lat_min, lat_max)
        if 'surface' in mask.attrs and mask.attrs['surface'] not in (None, 'all'):
            surface_mask = _land_sea_mask(
                rm,
                mask.attrs['surface'],
                land_mask=mask.attrs.get('land_mask'),
                land_mask_path=mask.attrs.get('land_mask_path'),
            )
            rm = rm * surface_mask
        return rm

    if edge_weighted and 'lon_range' in mask.attrs and 'lat_range' in mask.attrs:
        region_mask = _rebuild_mask(data_template)
    elif not _grids_match(mask, data_template):
        # Model grid differs from the template used to build the mask — rebuild.
        if 'lon_range' in mask.attrs and 'lat_range' in mask.attrs:
            region_mask = _rebuild_mask(data_template)
        else:
            region_mask = mask.interp(
                lat=data_template.lat, lon=data_template.lon, method='nearest'
            )
    else:
        region_mask = mask

    cell_area = np.cos(np.deg2rad(da.lat)) * dlat * dlon
    weights = cell_area * region_mask
    return weights, region_mask


def land_mask_from_cartopy(template, resolution='110m'):
    """
    Build a 0/1 land mask from the Cartopy Natural Earth land feature.

    Returns 1 for land and 0 for ocean on the template lat/lon grid.  The
    template may use either 0--360 or -180--180 longitude; the polygons are
    tested in the -180--180 frame and mapped back to the original grid.
    """
    import numpy as np
    import xarray as xr
    import cartopy.feature as cfeature
    from shapely.vectorized import contains
    from shapely.ops import unary_union

    lat = np.asarray(template.lat.values)
    lon = np.asarray(template.lon.values)

    # Natural Earth land shapes are defined on -180..180
    lon_wrapped = (lon + 180.0) % 360.0 - 180.0
    lon2d, lat2d = np.meshgrid(lon_wrapped, lat)

    land = cfeature.NaturalEarthFeature('physical', 'land', resolution)
    geoms = list(land.geometries())
    if not geoms:
        raise ValueError(f"No land geometries loaded from Natural Earth '{resolution}'.")

    union = unary_union(geoms)
    mask = contains(union, lon2d, lat2d).astype(float)

    return xr.DataArray(
        mask,
        dims=('lat', 'lon'),
        coords={'lat': lat, 'lon': lon},
        name='land_mask',
    )


def _land_sea_mask(template, surface, land_mask=None, land_mask_path=None):
    """
    Return a 0/1 mask on the template grid for land or sea.

    surface: 'land', 'sea', 'ocean', or 'all' (ocean is treated as sea)
    land_mask: xr.DataArray with 1=land, 0=sea (or fractional)
    land_mask_path: netCDF path; looks for lsm, land, or land_mask variables
    """
    if surface in (None, 'all'):
        return xr.ones_like(template)

    # Accept 'ocean' as an alias for 'sea'
    if surface == 'ocean':
        surface = 'sea'

    if surface not in ('land', 'sea'):
        raise ValueError("surface must be 'all', 'land', 'sea', or 'ocean'")

    if land_mask is not None:
        lsm = land_mask
    elif land_mask_path is not None:
        with xr.open_dataset(land_mask_path) as ds:
            for var in ('lsm', 'land', 'land_mask', 'LAND_MASK', 'landf'):
                if var in ds:
                    lsm = ds[var].load()
                    break
            else:
                raise KeyError(
                    f"No land-mask variable found in {land_mask_path}. "
                    "Expected one of: lsm, land, land_mask, LAND_MASK"
                )
    else:
        try:
            import regionmask
            land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
            lsm = land.mask(template.lon, template.lat) == 0
        except ImportError:
            try:
                lsm = land_mask_from_cartopy(template)
            except Exception as exc:
                raise ValueError(
                    "Land/sea masking requires land_mask, land_mask_path, "
                    "the 'regionmask' package, or cartopy+shapely."
                ) from exc

    if not isinstance(lsm, xr.DataArray):
        lsm = xr.DataArray(lsm, dims=('lat', 'lon'), coords={'lat': template.lat, 'lon': template.lon})

    if 'lat' in lsm.dims and 'lon' in lsm.dims:
        if not np.array_equal(lsm.lat.values, template.lat.values) or not np.array_equal(lsm.lon.values, template.lon.values):
            lsm = lsm.interp(lat=template.lat, lon=template.lon, method='nearest')

    lsm = (lsm >= 0.5).astype(float)

    if surface == 'land':
        return lsm
    return 1.0 - lsm


def create_region_mask(
    template,
    name,
    lon_range=(0, 360),
    lat_range=(-90, 90),
    surface='all',
    land_mask=None,
    land_mask_path=None,
    mask_registry=None,
):
    """
    Build a named regional mask on the same lat/lon grid as template.

    Parameters
    ----------
    template : xr.DataArray or xr.Dataset
        Reference field (only lat/lon coordinates are used). Use any loaded
        AeroCom variable, e.g. model_data['AOD550'].isel(time=0).
    name : str
        Name for this region (stored in mask.attrs and optionally in mask_registry).
    lon_range : tuple or None
        (lon_min, lon_max) in 0–360. None selects the full longitude range.
        Supports wrap-around when lon_min > lon_max (e.g. 350, 8 for outflow).
    lat_range : tuple or None
        (lat_min, lat_max). None selects the full latitude range.
    surface : str
        'all', 'land', or 'sea'.
    land_mask : xr.DataArray, optional
        1=land, 0=sea on the same or interpolatable grid.
    land_mask_path : str, optional
        Path to a netCDF land-sea mask file.
    mask_registry : dict, optional
        If given, stores the mask as mask_registry[name] = mask.

    Returns
    -------
    xr.DataArray
        Mask with values 0/1 (lat, lon). Metadata is stored in .attrs for use
        with regional_aggregate(..., edge_weighted=True).
    """
    if isinstance(template, xr.Dataset):
        template = template[list(template.data_vars)[0]]

    lon_range = _normalize_lon_range(lon_range)
    lat_range = _normalize_lat_range(lat_range)

    spatial_template = _spatial_template(template)

    if lon_range is None and lat_range is None:
        box_mask = xr.ones_like(spatial_template)
    else:
        lon_min, lon_max = lon_range if lon_range is not None else (0.0, 360.0)
        lat_min, lat_max = lat_range if lat_range is not None else (-90.0, 90.0)
        box_mask = create_mask(spatial_template, lon_min, lon_max, lat_min, lat_max)

    surface_mask = _land_sea_mask(
        spatial_template,
        surface,
        land_mask=land_mask,
        land_mask_path=land_mask_path,
    )
    mask = (box_mask * surface_mask).astype(float)
    mask.name = name
    mask.attrs.update({
        'name': name,
        'lon_range': lon_range if lon_range is not None else (0.0, 360.0),
        'lat_range': lat_range if lat_range is not None else (-90.0, 90.0),
        'surface': surface,
        'land_mask_path': land_mask_path,
    })

    if mask_registry is not None:
        mask_registry[name] = mask

    return mask


def regional_aggregate(
    da,
    mask,
    spatial='mean',
    edge_weighted=False,
    time_slice=None,
    temporal='mean',
    return_time_series=False,
):
    """
    Spatially and/or temporally aggregate data using a mask from create_region_mask.

    Parameters
    ----------
    da : xr.DataArray
        Field with lat, lon, and optionally time dimensions.
    mask : xr.DataArray
        Mask from create_region_mask (global mask = ones on full grid).
    spatial : str
        'mean' (area-weighted average) or 'total' (weighted sum over the region).
    edge_weighted : bool
        If True, use fractional grid-cell weights at the region boundaries.
    time_slice : slice, (start, end), or None
        Temporal subset applied before aggregation. None keeps all times.
    temporal : str
        'mean' or 'total' over the remaining time dimension after spatial reduction.
        Ignored when return_time_series=True.
    return_time_series : bool
        If True, return the spatial aggregate at each time step without collapsing time.

    Returns
    -------
    float or xr.DataArray
        Scalar when temporal reduction is applied; otherwise a time series.
    """
    if spatial not in ('mean', 'total'):
        raise ValueError("spatial must be 'mean' or 'total'")
    if temporal not in ('mean', 'total'):
        raise ValueError("temporal must be 'mean' or 'total'")

    da_work = da
    if 'bnds' in da_work.dims:
        da_work = da_work.drop_dims('bnds', errors='ignore')
    elif 'nbnd' in da_work.dims:
        da_work = da_work.drop_dims('nbnd', errors='ignore')

    if 'time' in da_work.dims:
        da_work = da_work.sel(time=_parse_time_slice(time_slice))

    weights, _ = _spatial_weights(da_work, mask, edge_weighted=edge_weighted)
    weighted_field = da_work * weights

    if spatial == 'mean':
        norm = weights.sum(dim=['lat', 'lon'])
        spatial_result = weighted_field.sum(dim=['lat', 'lon']) / norm
    else:
        spatial_result = weighted_field.sum(dim=['lat', 'lon'])

    if return_time_series or 'time' not in spatial_result.dims:
        return spatial_result

    if temporal == 'mean':
        return float(spatial_result.mean('time').values)
    return float(spatial_result.sum('time').values)


def aggregate_models(
    model_dict,
    var_name,
    mask,
    spatial='mean',
    edge_weighted=False,
    time_slice=None,
    temporal='mean',
    return_time_series=False,
):
    """
    Apply regional_aggregate to every model in an AeroCom get_data() dictionary.

    Returns
    -------
    dict
        {model_name: aggregated value or time series}
    """
    result = {}
    for model, model_data in model_dict.items():
        if var_name not in model_data:
            raise KeyError(f"Variable '{var_name}' not found for model '{model}'")
        result[model] = regional_aggregate(
            model_data[var_name],
            mask,
            spatial=spatial,
            edge_weighted=edge_weighted,
            time_slice=time_slice,
            temporal=temporal,
            return_time_series=return_time_series,
        )
    return result


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