import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
import matplotlib.ticker as mticker
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

import warnings
warnings.filterwarnings('ignore', category=cartopy.io.DownloadWarning)

from typing import List, Optional, Union, Tuple, Literal
from mpl_toolkits.axes_grid1 import ImageGrid

def fake_uba_map(
    lon, lat, c_array, num_levels=None, zmin=None, zmax=None, labels="", title="",
    mycolor=None,
    region_boxes=None,            # dict {"Name": (lon_min, lon_max, lat_min, lat_max)}
    region_edgecolor='magenta',
    region_linewidth=1.5,
    region_linestyle='--',
    show_region_labels=True,
    cbar_orientation='vertical',  # NEW: 'vertical' or 'horizontal'
    cbar_extend='auto'            # NEW: 'auto' | 'neither' | 'min' | 'max' | 'both'
):
    """
    Plot a pcolormesh with cartopy and a flexible colorbar.

    - Automatically sets colorbar extend when cbar_extend='auto' by checking if data
      falls outside [zmin, zmax].
    - cbar_orientation can be 'vertical' or 'horizontal'.
    """



    # Show cartopy data dir (as in your version)
    print(cartopy.config['data_dir'])
    cartopy_dir = '/uhome/zhangj1/.local/share/cartopy'
    os.environ['CARTOPY_USER_BACKGROUNDS'] = cartopy_dir
    os.environ['CARTOPY_SHAPEDIR'] = cartopy_dir

    # Ensure arrays
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    c_array = np.asarray(c_array)

    # Create figure/axis
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Meshgrid from 1D lon/lat
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Compute data min/max (ignore NaN)
    data_min = np.nanmin(c_array) if np.size(c_array) > 0 else np.nan
    data_max = np.nanmax(c_array) if np.size(c_array) > 0 else np.nan

    # If zmin/zmax not provided, derive from data
    if zmin is None:
        zmin = data_min
    if zmax is None:
        zmax = data_max

    # Helpers
    def _auto_extend(data_min, data_max, zmin, zmax):
        below = np.isfinite(zmin) and np.isfinite(data_min) and (data_min < zmin)
        above = np.isfinite(zmax) and np.isfinite(data_max) and (data_max > zmax)
        if below and above:
            return 'both'
        elif below:
            return 'min'
        elif above:
            return 'max'
        else:
            return 'neither'

    # Colormap / normalization
    # - Set up defaults
    boundaries = None
    cmap = None
    norm = None

    if mycolor is None:
        # Default discrete scheme
        mycolor = ["#DBD6D6", "#9D579A", "#009BD5", "#5EAD35", "#FABB00", "#CE1F5E"]
        if num_levels is None:
            num_levels = 12
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", mycolor, N=num_levels)
        boundaries = np.linspace(zmin, zmax, num_levels + 1)
        # Use clip=True to keep values within [zmin,zmax] for mapping;
        # colorbar 'extend' will still show triangles for out-of-range
        norm = mcolors.BoundaryNorm(boundaries, ncolors=num_levels, clip=True)

    elif mycolor == 'diff':
        # Diverging continuous colormap centered at 0
        cmap = plt.get_cmap('RdBu_r')
        # If user didn't pass balanced z-limits, make them symmetric
        if (zmin is None) or (zmax is None) or (zmin >= 0) or (zmax <= 0):
            m = np.nanmax(np.abs([data_min, data_max]))
            zmin, zmax = -m, m
        norm = mcolors.TwoSlopeNorm(vmin=zmin, vcenter=0.0, vmax=zmax)

    else:
        # Custom discrete color list
        if num_levels is None:
            num_levels = len(mycolor) if hasattr(mycolor, '__len__') else 12
        cmap = plt.get_cmap(mycolor)
        boundaries = np.linspace(zmin, zmax, num_levels + 1)  # Create boundaries for the color levels
        norm = plt.Normalize(vmin=zmin, vmax=zmax, clip=True)

    # Draw the field
    # NOTE: 'extend' is NOT a valid kwarg for pcolormesh; it's set on colorbar.
    c = ax.pcolormesh(lon_grid, lat_grid, c_array, cmap=cmap, norm=norm, shading='auto')

    # Coastlines/borders
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    # ax.add_feature(cfeature.BORDERS.with_scale('50m'), facecolor='none', edgecolor='gray')

    # Region boxes & labels
    if region_boxes:
        for name, (lonmin, lonmax, latmin, latmax) in region_boxes.items():
            rect = Rectangle(
                (lonmin, latmin),
                lonmax - lonmin,
                latmax - latmin,
                fill=False,
                edgecolor=region_edgecolor,
                linewidth=region_linewidth,
                linestyle=region_linestyle,
                transform=ccrs.PlateCarree(),
                zorder=5
            )
            ax.add_patch(rect)

            if show_region_labels:
                ax.text(
                    (lonmin + lonmax) / 2.0, latmax + 0.5, name,
                    transform=ccrs.PlateCarree(),
                    ha='center', va='bottom', fontsize=10, color=region_edgecolor,
                    zorder=6,
                    path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white')]
                )

    # Decide colorbar 'extend'
    if cbar_extend == 'auto':
        extend_mode = _auto_extend(data_min, data_max, zmin, zmax)
    else:
        extend_mode = cbar_extend  # trust user: 'neither'|'min'|'max'|'both'

    # Colorbar kwargs adapted to orientation
    if cbar_orientation not in ('vertical', 'horizontal'):
        raise ValueError("cbar_orientation must be 'vertical' or 'horizontal'")

    cbar_kwargs = dict(orientation=cbar_orientation, extend=extend_mode)
    if cbar_orientation == 'vertical':
        # Tall, slimmer bar
        cbar_kwargs.update(pad=0.02, aspect=30, shrink=0.7)
    else:
        # Horizontal needs more padding and uses fraction instead of shrink
        cbar_kwargs.update(pad=0.08, fraction=0.046, aspect=40)

    # Add colorbar
    # For discrete maps, pass boundaries so ticks align nicely
    if boundaries is not None:
        cbar = plt.colorbar(c, ax=ax, boundaries=boundaries, **cbar_kwargs)
        # Ticks (avoid too many labels)
        max_ticks = 20
        if len(boundaries) <= max_ticks:
            cbar.set_ticks(boundaries)
            cbar.set_ticklabels([f"{b:.2f}" for b in boundaries])
        else:
            # Reduce tick density if too many levels
            step = int(np.ceil(len(boundaries) / max_ticks))
            ticks = boundaries[::step]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{b:.2f}" for b in ticks])
    else:
        cbar = plt.colorbar(c, ax=ax, **cbar_kwargs)

    cbar.set_label(labels)
    cbar.ax.tick_params(labelsize=10)

    # Title and clean axes
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from typing import List, Optional, Union, Tuple, Literal
from mpl_toolkits.axes_grid1 import ImageGrid

def uba_map_flex(
    data: Optional[Union[xr.Dataset, xr.DataArray, dict]] = None, # Added dict support
    lon: Optional[np.ndarray] = None, 
    lat: Optional[np.ndarray] = None, 
    c_array: Optional[Union[np.ndarray, List[np.ndarray], str, List[str]]] = None,
    num_levels: Optional[int] = None, 
    vmin: Optional[Union[float, List[float], Tuple[float, float]]] = None, 
    vmax: Optional[float] = None,   
    labels: str = "",            
    cbar_title: str = "",        
    cbar_tick_format: Optional[str] = None,
    title: Union[str, List[str]] = "",
    mycolor: Union[str, List[str], None] = None,
    projection_type: Literal["regular", "curved"] = "curved",
    n_colors: Optional[int] = None,  
    extent: Union[Literal["data", "global"], List[float], dict] = "data",
    gridline: bool = False,
    cbar_tick_mode: Literal["bounds", "centers"] = "bounds",
    cbar_mode: Literal["single", "each"] = "single",
    cbar_size: str = "2.5%",     
    cbar_pad: str = "2%",      
    figsize_per_panel: Tuple[float, float] = (8.0, 5.0),
    province_border : bool = False,
    coast_alpha: float = 0.6,
    borders_alpha: float = 0.4,
    land_alpha: float = 0.2,
    ocean_alpha: float = 0.1,
    edge_lw: float = 0.5,
    title_fontsize: int = 12,
    label_fontsize: int = 9,
    show: bool = True,
    savefile: Optional[str] = None,
    cbar_orientation: Literal['vertical', 'horizontal'] = 'vertical',
    cbar_extend: Literal['auto', 'neither', 'min', 'max', 'both'] = 'auto'
):
    # --- 1. Robust Data Extraction ---
    data_list = []
    
    # Handle Dictionary Input
    if isinstance(data, dict):
        keys = list(data.keys())
        first_val = data[keys[0]]
        
        # Pull coordinates from first item if it's an xarray object
        if lon is None or lat is None:
            if hasattr(first_val, 'coords'):
                lon = first_val.coords.get('lon', first_val.coords.get('longitude')).values
                lat = first_val.coords.get('lat', first_val.coords.get('latitude')).values
        
        # Convert all dict values to numpy
        for k in keys:
            val = data[k]
            data_list.append(val.values if hasattr(val, 'values') else np.asarray(val))
        
        # Use keys as titles if no title provided
        if not title:
            title = [str(k) for k in keys]

    # Handle Standard Xarray Input
    elif data is not None and isinstance(data, (xr.Dataset, xr.DataArray)):
        if lon is None:
            lon = data.coords.get('lon', data.coords.get('longitude')).values
        if lat is None:
            lat = data.coords.get('lat', data.coords.get('latitude')).values
        
        if c_array is None:
            if isinstance(data, xr.Dataset):
                data_list = [data[v].values for v in data.data_vars]
                if not title: title = list(data.data_vars)
            else:
                data_list = [data.values]
        elif isinstance(c_array, str):
            data_list = [data[c_array].values]
        else:
            data_list = [data[v].values if hasattr(data[v], 'values') else data[v] for v in c_array]
    
    # Fallback to c_array for raw numpy
    else:
        if isinstance(c_array, list):
            data_list = [np.asarray(a) for a in c_array]
        else:
            data_list = [np.asarray(c_array)]

    # --- 2. Color Scaling Logic ---
    # Filter for finite numbers to avoid the TypeError
    all_data_stack = np.concatenate([np.asarray(a).ravel() for a in data_list])
    valid_data = all_data_stack[np.isfinite(all_data_stack)]
    
    d_min = valid_data.min() if valid_data.size > 0 else 0.0
    d_max = valid_data.max() if valid_data.size > 0 else 1.0

    if isinstance(vmin, (list, tuple)) and len(vmin) == 2:
        plot_vmin, plot_vmax = np.nanquantile(valid_data, vmin[0]), np.nanquantile(valid_data, vmin[1])
    else:
        plot_vmin = vmin if vmin is not None else d_min
        plot_vmax = vmax if vmax is not None else d_max

    # Custom Default Palette
    if mycolor is None:
        mycolor = ["#DBD6D6", "#9D579A", "#009BD5", "#5EAD35", "#FABB00", "#CE1F5E"]
        num_levels = num_levels or 12
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", mycolor, N=num_levels)
        boundaries = np.linspace(plot_vmin, plot_vmax, num_levels + 1)
        norm = mcolors.BoundaryNorm(boundaries, ncolors=num_levels, clip=True)
    else:
        if mycolor == 'diff':
            cmap = plt.get_cmap('RdBu_r')
            limit = max(abs(plot_vmin), abs(plot_vmax))
            plot_vmin, plot_vmax = -limit, limit
        else:
            cmap = plt.get_cmap(mycolor if isinstance(mycolor, str) else 'viridis')
        
        if num_levels:
            boundaries = np.linspace(plot_vmin, plot_vmax, num_levels + 1)
            norm = mcolors.BoundaryNorm(boundaries, cmap.N, extend='both')
        else:
            norm = mcolors.Normalize(vmin=plot_vmin, vmax=plot_vmax)
            boundaries = None

    # --- 3. Figure Setup ---
    proj = ccrs.Robinson() if projection_type == "curved" else ccrs.PlateCarree()
    num_panels = len(data_list)
    titles = [title] * num_panels if isinstance(title, str) else title

    # Remember dict keys so per-panel extent can be keyed by data name.
    data_keys = None
    if isinstance(data, dict):
        data_keys = list(data.keys())

    ncols = min(3, num_panels)
    nrows = int(np.ceil(num_panels / ncols))
    
    fig = plt.figure(figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.5,
                     cbar_mode=cbar_mode if num_panels > 1 else "single",
                     cbar_location="right" if cbar_orientation == "vertical" else "bottom",
                     cbar_size=cbar_size, cbar_pad=cbar_pad, share_all=True,
                     axes_class=(GeoAxes, dict(projection=proj)))

    # --- 4. Plotting Loop ---
    # Use a local Cartopy cache if one exists, otherwise rely on the default.
    local_cartopy_dir = Path.home() / '.local' / 'share' / 'cartopy'
    if local_cartopy_dir.exists():
        cartopy.config['data_dir'] = str(local_cartopy_dir)

    for i, data_val in enumerate(data_list):
        ax = grid[i]
        mesh = ax.pcolormesh(lon, lat, data_val, transform=ccrs.PlateCarree(), 
                             cmap=cmap, norm=norm, shading='auto')

        try:
            # Force Cartopy to resolve the land feature now so any missing-data
            # error is caught here rather than at draw/save time.  Use the 110m
            # resolution so a local cache can be used without auto-scaling.
            land_feat = cfeature.LAND.with_scale('110m')
            next(iter(land_feat.geometries()))
            ax.add_feature(land_feat, facecolor='lightgray', alpha=land_alpha)
        except Exception:
            pass
        try:
            coast_feat = cfeature.COASTLINE.with_scale('110m')
            next(iter(coast_feat.geometries()))
            ax.add_feature(coast_feat, linewidth=edge_lw, alpha=coast_alpha)
        except Exception:
            pass
        if province_border:
            try:
                borders_feat = cfeature.BORDERS.with_scale('110m')
                next(iter(borders_feat.geometries()))
                ax.add_feature(borders_feat, linewidth=edge_lw, alpha=borders_alpha)
                prov_lines = cfeature.NaturalEarthFeature(
                    category='cultural',
                    name='admin_1_states_provinces_lines',
                    scale='110m',
                    facecolor='none'
                )
                next(iter(prov_lines.geometries()))
                ax.add_feature(prov_lines, edgecolor='black', linewidth=edge_lw, alpha=borders_alpha)
            except Exception:
                pass

        # Resolve per-panel extent (dict extent keyed by data key or title)
        panel_extent = extent
        if isinstance(extent, dict):
            key = data_keys[i] if data_keys is not None and i < len(data_keys) else titles[i]
            panel_extent = extent.get(key, 'data')

        if panel_extent == 'global':
            ax.set_global()
        elif panel_extent == 'data':
            ax.set_extent([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], crs=ccrs.PlateCarree())
        else:
            ext = list(panel_extent)
            # Cartopy expects monotonic longitudes; unwrap dateline-crossing windows.
            if ext[0] > ext[1]:
                ext[0] -= 360
            ax.set_extent(ext, crs=ccrs.PlateCarree())

        ax.set_title(titles[i], fontsize=title_fontsize)

        # --- NEW: axis labels (optional) ---
        # `labels` can be a string (applies to both x and y for quick annotation),
        # or a tuple/list like (xlabel, ylabel).
        if labels:
            if isinstance(labels, (tuple, list)) and len(labels) == 2:
                ax.set_xlabel(labels[0], fontsize=label_fontsize)
                ax.set_ylabel(labels[1], fontsize=label_fontsize)
            elif isinstance(labels, str):
                ax.set_xlabel(labels, fontsize=label_fontsize)
                ax.set_ylabel(labels, fontsize=label_fontsize)

        # colorbar
        if cbar_mode == "each" or (cbar_mode == "single" and i == num_panels - 1):
            cb_ax = grid.cbar_axes[i]
            ext_type = cbar_extend if cbar_extend != 'auto' else 'both'
            cb = plt.colorbar(mesh, cax=cb_ax, orientation=cbar_orientation, extend=ext_type)

            # --- NEW: colorbar label/title ---
            if cbar_title:
                # For horizontal cbar, the label sits below and benefits from a bit more pad.
                if cbar_orientation == 'horizontal':
                    cb.set_label(cbar_title, fontsize=label_fontsize, labelpad=6)
                else:
                    cb.set_label(cbar_title, fontsize=label_fontsize, labelpad=6)

            # --- correct axis for tick formatting ---
            if cbar_tick_format:
                formatter = mticker.FormatStrFormatter(f'%{cbar_tick_format}')
                if cbar_orientation == 'horizontal':
                    cb.ax.xaxis.set_major_formatter(formatter)
                else:
                    cb.ax.yaxis.set_major_formatter(formatter)

            if boundaries is not None and cbar_tick_mode == "bounds":
                cb.set_ticks(boundaries)

        # if cbar_mode == "each" or (cbar_mode == "single" and i == num_panels - 1):
        #     cb_ax = grid.cbar_axes[i]
        #     ext_type = cbar_extend if cbar_extend != 'auto' else 'both'
        #     cb = plt.colorbar(mesh, cax=cb_ax, orientation=cbar_orientation, extend=ext_type)
        #     if cbar_tick_format:
        #         cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f'%{cbar_tick_format}'))
        #     if boundaries is not None and cbar_tick_mode == "bounds":
        #         cb.set_ticks(boundaries)

    if savefile: plt.savefig(savefile, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
        return None
    return fig
from typing import Optional, Union, Tuple, List, Literal 
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import ImageGrid
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt 
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def make_alpha_fade_cmap(
    base_cmap='RdBu_r',
    vmin=-100.0, vmax=100.0,
    center=0.0,
    min_alpha=0.05,   # almost transparent near center (0)
    max_alpha=0.9,    # fairly opaque at extremes
    gamma=1.0,        # >1 makes alpha rise slower near center, faster near ends
    N=256
):
    """
    Returns (ListedColormap_with_alpha, Normalize) so that colors are transparent near `center`
    and become more opaque as |value - center| grows.
    For diverging data around 0, pass vmin<0, center=0, vmax>0 (TwoSlopeNorm).
    For strictly positive data (e.g., 0..max), set center=0 with vmin=0 (Normalize).
    """
    base = cm.get_cmap(base_cmap, N)

    # Choose norm: diverging vs single-sided
    if vmin < center < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Sample the value range and compute alpha as a function of distance to center
    vals = np.linspace(vmin, vmax, N)
    t = np.abs(norm(vals) - norm(center))           # 0 at center, up to ~1 at extremes
    if t.max() == 0:
        alpha = np.full(N, max_alpha)
    else:
        alpha = min_alpha + (max_alpha - min_alpha) * (t / t.max())**gamma

    colors = base(np.linspace(0, 1, N))
    colors[:, 3] = np.clip(alpha, 0, 1)            # set per-color alpha

    cmap_alpha = mcolors.ListedColormap(colors)
    # Make NaNs fully transparent so OSM shows through
    cmap_alpha.set_bad(alpha=0.0)
    return cmap_alpha, norm

def uba_map_city(
    data: Optional[Union[xr.Dataset, xr.DataArray, dict]] = None,
    lon: Optional[np.ndarray] = None,
    lat: Optional[np.ndarray] = None,
    c_array: Optional[Union[np.ndarray, List[np.ndarray], str, List[str]]] = None,
    num_levels: Optional[int] = None,
    vmin: Optional[Union[float, List[float], Tuple[float, float]]] = None,
    vmax: Optional[float] = None,
    labels: str = "",
    cbar_title: str = "",
    cbar_tick_format: Optional[str] = None,
    title: Union[str, List[str]] = "",
    mycolor: Union[str, List[str], None] = None,
    projection_type: Literal["regular", "curved"] = "regular",  # ignored when use_osm=True
    n_colors: Optional[int] = None,
    extent: Union[Literal["data", "global"], List[float], dict] = "data",
    gridline: bool = False,
    cbar_tick_mode: Literal["bounds", "centers"] = "bounds",
    cbar_mode: Literal["single", "each"] = "single",
    cbar_size: str = "2.5%",
    cbar_pad: str = "2%",
    figsize_per_panel: Tuple[float, float] = (8.0, 5.0),
    province_border: bool = False,
    coast_alpha: float = 0.6,
    borders_alpha: float = 0.4,
    land_alpha: float = 0.2,
    ocean_alpha: float = 0.1,
    edge_lw: float = 0.5,
    title_fontsize: int = 12,
    label_fontsize: int = 9,
    show: bool = True,
    savefile: Optional[str] = None,
    cbar_orientation: Literal['vertical', 'horizontal'] = 'vertical',
    cbar_extend: Literal['auto', 'neither', 'min', 'max', 'both'] = 'auto',
    color_alpha: float = 0.5, 
    # --- OSM / city controls ---
    use_osm: bool = True,
    osm_source: Literal['OSM', 'Stamen-terrain', 'Stamen-toner', 'Stamen-watercolor'] = 'OSM',
    osm_zoom: int = 12,                   # ~12–14 city scale, 7–9 regional
    osm_alpha: float = 1.0,
    city_center: Optional[Tuple[float, float]] = None,  # (lon, lat)
    city_radius_km: float = 15.0,         # radius for city extent
    draw_features_over_osm: bool = False, # coastlines/borders over tiles (usually off for city maps)
    add_osm_attribution: bool = True,
    #--- color tranparency control  mycolor = alpha_fade_around_zero ---
    alpha_vmin: float = -80,    # set to your data range
    alpha_vmax: float = 80,
    alpha_center: float = 0.0,
    alpha_min: float = 0.05,
    alpha_max: float = 0.9,
    alpha_gamma: float = 1.2,
    alpha_base_cmap: str = 'RdBu_r',

):
    """
    City-scale map with an OpenStreetMap (or Stamen) basemap.
    Mirrors uba_map_flex inputs; draws pcolormesh on top of tiles.
    If show=True -> plt.show() and return None; else returns fig.
    """
    # --- 1) Robust data extraction (same logic pattern as your uba_map_flex) ---
    data_list = []

    if isinstance(data, dict):
        keys = list(data.keys())
        first_val = data[keys[0]]

        if lon is None or lat is None:
            if hasattr(first_val, 'coords'):
                lon = first_val.coords.get('lon', first_val.coords.get('longitude')).values
                lat = first_val.coords.get('lat', first_val.coords.get('latitude')).values

        for k in keys:
            val = data[k]
            data_list.append(val.values if hasattr(val, 'values') else np.asarray(val))

        if not title:
            title = [str(k) for k in keys]

    elif data is not None and isinstance(data, (xr.Dataset, xr.DataArray)):
        if lon is None:
            lon = data.coords.get('lon', data.coords.get('longitude')).values
        if lat is None:
            lat = data.coords.get('lat', data.coords.get('latitude')).values

        if c_array is None:
            if isinstance(data, xr.Dataset):
                data_list = [data[v].values for v in data.data_vars]
                if not title:
                    title = list(data.data_vars)
            else:
                data_list = [data.values]
        elif isinstance(c_array, str):
            data_list = [data[c_array].values]
        else:
            data_list = [data[v].values if hasattr(data[v], 'values') else data[v] for v in c_array]

    else:
        # Raw numpy path
        if isinstance(c_array, list):
            data_list = [np.asarray(a) for a in c_array]
        else:
            data_list = [np.asarray(c_array)]

    # Sanity for lon/lat
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # --- 2) Color scaling (same approach as your flex) ---
    all_data_stack = np.concatenate([np.asarray(a).ravel() for a in data_list]) if len(data_list) else np.array([])
    valid_data = all_data_stack[np.isfinite(all_data_stack)]

    d_min = valid_data.min() if valid_data.size > 0 else 0.0
    d_max = valid_data.max() if valid_data.size > 0 else 1.0

    if isinstance(vmin, (list, tuple)) and len(vmin) == 2:
        plot_vmin, plot_vmax = np.nanquantile(valid_data, vmin[0]), np.nanquantile(valid_data, vmin[1])
    else:
        plot_vmin = vmin if vmin is not None else d_min
        plot_vmax = vmax if vmax is not None else d_max

    if mycolor is None:
        mycolor = ["#DBD6D6", "#9D579A", "#009BD5", "#5EAD35", "#FABB00", "#CE1F5E"]
        num_levels = num_levels or 12
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", mycolor, N=num_levels)
        boundaries = np.linspace(plot_vmin, plot_vmax, num_levels + 1)
        norm = mcolors.BoundaryNorm(boundaries, ncolors=num_levels, clip=True)
    else:
        if mycolor == 'diff':
            cmap = plt.get_cmap('RdBu_r')
            limit = max(abs(plot_vmin), abs(plot_vmax))
            plot_vmin, plot_vmax = -limit, limit
        elif mycolor ==  'alpha_fade_around_zero':
            cmap, norm = make_alpha_fade_cmap(
                base_cmap=alpha_base_cmap,
                vmin=alpha_vmin, vmax=alpha_vmax, center=alpha_center,
                min_alpha=alpha_min, max_alpha=alpha_max, gamma=alpha_gamma, N=256
            )

        else:
            cmap = plt.get_cmap(mycolor if isinstance(mycolor, str) else 'viridis')

        if num_levels:
            boundaries = np.linspace(plot_vmin, plot_vmax, num_levels + 1)
            norm = mcolors.BoundaryNorm(boundaries, cmap.N, extend='both')
        else:
            norm = mcolors.Normalize(vmin=plot_vmin, vmax=plot_vmax)
            boundaries = None

    # --- 3) Tiler and projection for OSM ---
    tiler = None
    if use_osm:
        if osm_source == 'OSM':
            tiler = cimgt.OSM()
        elif osm_source == 'Stamen-terrain':
            tiler = cimgt.Stamen('terrain')
        elif osm_source == 'Stamen-toner':
            tiler = cimgt.Stamen('toner')
        elif osm_source == 'Stamen-watercolor':
            tiler = cimgt.Stamen('watercolor')
        else:
            tiler = cimgt.OSM()
        proj = tiler.crs  # Web Mercator
    else:
        proj = ccrs.Robinson() if projection_type == "curved" else ccrs.PlateCarree()

    # --- 4) Figure & grid (GeoAxes) ---
    num_panels = len(data_list)
    titles = [title] * num_panels if isinstance(title, str) else title
    ncols = min(3, num_panels)
    nrows = int(np.ceil(num_panels / ncols))

    fig = plt.figure(figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.5,
        cbar_mode=cbar_mode if num_panels > 1 else "single",
        cbar_location="right" if cbar_orientation == "vertical" else "bottom",
        cbar_size=cbar_size, cbar_pad=cbar_pad, share_all=True,
        axes_class=(GeoAxes, dict(projection=proj))
    )

    # --- 5) City extent logic ---
    def _city_extent(center_lon, center_lat, radius_km):
        dlat = radius_km / 111.0
        dlon = radius_km / (111.320 * np.cos(np.deg2rad(center_lat)) + 1e-12)
        return [center_lon - dlon, center_lon + dlon, center_lat - dlat, center_lat + dlat]

    if isinstance(extent, list) and len(extent) == 4:
        pass  # user-provided geographic extent
    elif extent == 'global':
        pass
    elif city_center is not None:
        extent = _city_extent(city_center[0], city_center[1], city_radius_km)
    else:
        # default to data bounds if available
        extent = [
            float(np.nanmin(lon)), float(np.nanmax(lon)),
            float(np.nanmin(lat)), float(np.nanmax(lat))
        ]

    # --- 6) Plotting loop ---
    for i, data_val in enumerate(data_list):
        ax = grid[i]

        # Basemap tiles (under everything)
        if use_osm and tiler is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            # Some Cartopy versions return None; pass alpha via kwargs instead
            ax.add_image(tiler, osm_zoom, alpha=osm_alpha)

        else:
            # Optional shaded land when NOT using OSM
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=land_alpha)
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=ocean_alpha)
            ax.add_feature(cfeature.COASTLINE, linewidth=edge_lw, alpha=coast_alpha)
            if province_border:
                ax.add_feature(cfeature.BORDERS, linewidth=edge_lw, alpha=borders_alpha)
                prov_lines = cfeature.NaturalEarthFeature(
                    category='cultural',
                    name='admin_1_states_provinces_lines',
                    scale='10m',
                    facecolor='none'
                )
                ax.add_feature(prov_lines, edgecolor='black', linewidth=edge_lw, alpha=borders_alpha)
            if extent == 'global':
                ax.set_global()
            else:
                ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Optionally draw features over OSM (usually not needed for city zooms)
        if use_osm and draw_features_over_osm:
            ax.add_feature(cfeature.BORDERS, linewidth=edge_lw, alpha=borders_alpha, zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=edge_lw, alpha=coast_alpha, zorder=2)

        # Gridlines if requested
        if gridline:
            ax.gridlines(draw_labels=False, linewidth=0.2, alpha=0.4, linestyle='--')

        # Data layer (on top of tiles)
        mesh = ax.pcolormesh(
            lon, lat, data_val,
            transform=ccrs.PlateCarree(),  # your grid is lon/lat degrees
            alpha=color_alpha,  
            cmap=cmap, norm=norm, shading='auto', zorder=3
        )

        # Title and axis labels
        if titles:
            ax.set_title(titles[i], fontsize=title_fontsize)
        if labels:
            if isinstance(labels, (tuple, list)) and len(labels) == 2:
                ax.set_xlabel(labels[0], fontsize=label_fontsize)
                ax.set_ylabel(labels[1], fontsize=label_fontsize)
            elif isinstance(labels, str):
                ax.set_xlabel(labels, fontsize=label_fontsize)
                ax.set_ylabel(labels, fontsize=label_fontsize)

        # Colorbar control
        if cbar_mode == "each" or (cbar_mode == "single" and i == num_panels - 1):
            cb_ax = grid.cbar_axes[i]
            ext_type = cbar_extend if cbar_extend != 'auto' else 'both'
            cb = plt.colorbar(mesh, cax=cb_ax, orientation=cbar_orientation, extend=ext_type)
            if cbar_title:
                cb.set_label(cbar_title, fontsize=label_fontsize, labelpad=6)
            if cbar_tick_format:
                formatter = mticker.FormatStrFormatter(f'%{cbar_tick_format}')
                if cbar_orientation == 'horizontal':
                    cb.ax.xaxis.set_major_formatter(formatter)
                else:
                    cb.ax.yaxis.set_major_formatter(formatter)
            if (num_levels is not None) and (cbar_tick_mode == "bounds"):
                cb.set_ticks(np.linspace(plot_vmin, plot_vmax, num_levels + 1))

        # Attribution (recommended by providers)
        if use_osm and add_osm_attribution and i == 0:
            ax.text(
                0.99, 0.01,
                "© OpenStreetMap contributors",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7, color='dimgray', bbox=dict(facecolor='white', alpha=0.6, lw=0)
            )

    if savefile:
        plt.savefig(savefile, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
        return None
    return fig
def plot_timeseries_with_components(

    df_raw: pd.DataFrame,
    res: pd.DataFrame,
    points: list[dict],
    wl: int,
    split_s: str = "om",
    agg: str = "median",
    err_floor: float = 0.02,
    obs_range: bool = True,
    show_group_bands: bool = True,
    show_total_band: bool = True,
    smooth_window: int | None = None,  # kept for compatibility; ignored
    prior_color: str = "tab:blue",
    obs_color: str = "tab:red",
    band_rest_color: str = "tab:purple",
    band_split_color: str = "tab:green",
    band_total_color: str = "0.6", 
    show_sensitivity_line: bool = False,
    sensitivity_pct: float = 0.15,   # 15% by default
    sensitivity_color: str = "0.3",
    sensitivity_linestyle: str = "--",
    sensitivity_linewidth: float = 1.6,
):
    """
    Plot observation, prior (simulation), and after-H (x ± 1σ) envelopes for each group
    at a selected wavelength for the nearest grid cell to each (lon,lat) point.

    Additions:
      - Sensitivity lines (optional): μ ± p * split(λ), where split(λ) comes from
        the monthly center (mean/median) of oopp_{split_s}aod_{wl}nm in df_raw.
        Controlled by `show_sensitivity_line` and `sensitivity_pct`.

    Existing behavior:
      - Red observation error bars: min–max range centered at monthly center (mean/median).
      - Optional group bands: μ ± σ_rest(λ) and μ ± σ_split(λ).
      - Optional total band (components only): μ ± sqrt(σ_rest^2 + σ_split^2).  (R shown separately)
      - R (instrument) bar: ±σ_R (black) centered at obs center, where σ_R = max(range, err_floor).

    Inputs
    ------
    df_raw : DataFrame
        Columns: lat, lon, date, aot_{wl}nm, oopp_aod_{wl}nm, oopp_{split_s}aod_{wl}nm
    res : DataFrame
        From fit_varcomp_per_grid_month: ['year','month','lat_bin','lon_bin','s2_rest', f's2_{split_s}', ...]
    points : list of dict
        [{'name': str, 'lon': float, 'lat': float}, ...]
    wl : int
        Wavelength to plot (e.g., 550).
    split_s : str
        Split species key, e.g., "om" -> columns "oopp_omaod_{wl}nm".
    agg : {'mean','median'}
        Aggregation for monthly summaries from df_raw.
    err_floor : float
        Floor for σ_R (computed from monthly range of obs).
    obs_range : bool
        If True, draw the red observation range bars (min–max).
    show_group_bands, show_total_band : bool
        Toggle shaded bands for group and combined components (excludes R).
    *color/line args* : styling
    """


    # ---- Infer bin sizes from res ----
    def _infer_step(vals):
        u = np.unique(np.asarray(vals))
        if len(u) <= 1:
            return 1.0
        d = np.diff(np.sort(u))
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.median(d)) if len(d) else 1.0

    lat_step = _infer_step(res["lat_bin"].values)
    lon_step = _infer_step(res["lon_bin"].values)

    def _bin_xy(x, step):
        return np.round(x / step) * step

    # Prepare raw data
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["lat_bin"] = _bin_xy(df["lat"].values.astype(float), lat_step)
    df["lon_bin"] = _bin_xy(df["lon"].values.astype(float), lon_step)

    # Column names at selected wl
    obs_col   = f"aot_{wl}nm"
    tot_col   = f"oopp_aod_{wl}nm"
    split_col = f"oopp_{split_s}aod_{wl}nm"

    # Aggregator
    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'")
    agg_fun_name = agg
    agg_fun = np.nanmean if agg == "mean" else np.nanmedian

    # Nearest grid cell centers
    cells = res[["lat_bin", "lon_bin"]].drop_duplicates().reset_index(drop=True).copy()
    cells["lat_bin"] = cells["lat_bin"].astype(float)
    cells["lon_bin"] = cells["lon_bin"].astype(float)

    def _nearest_cell(lon, lat):
        d2 = (cells["lat_bin"] - lat) ** 2 + (cells["lon_bin"] - lon) ** 2
        j = int(np.argmin(d2.values))
        return float(cells.loc[j, "lat_bin"]), float(cells.loc[j, "lon_bin"])

    # Figure
    n = len(points)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, p in enumerate(points):
        latb, lonb = _nearest_cell(p["lon"], p["lat"])

        g = df[(df["lat_bin"] == latb) & (df["lon_bin"] == lonb)].copy()

        # Ensure required columns exist
        for c in (obs_col, tot_col, split_col):
            if c not in g.columns:
                g[c] = np.nan

        # Observation range per month + σ_R (and variance)
        range_df = (
            g.groupby(["year", "month"], as_index=False)[obs_col]
             .agg(obs_min="min", obs_max="max")
        )
        # σ_R is the measurement std used in R; here defined as max(range, floor)
        range_df["sigma_R"] = np.maximum(range_df["obs_max"] - range_df["obs_min"], err_floor)
        range_df["var_R"]   = range_df["sigma_R"] ** 2

        # Monthly centers (obs/prior/split)
        agg_df = (
            g.groupby(["year", "month"], as_index=False)
             .agg(obs=(obs_col, agg_fun_name), prior=(tot_col, agg_fun_name), split=(split_col, agg_fun_name))
        )
        agg_df["h_split"] = agg_df["split"].astype(float)
        agg_df["h_rest"]  = (agg_df["prior"] - agg_df["split"]).astype(float)

        # Variance components from res
        res_cell = res[(res["lat_bin"] == latb) & (res["lon_bin"] == lonb)][
            ["year", "month", "s2_rest", f"s2_{split_s}"]
        ].copy()

        # Merge and build time axis
        dfm = (
            agg_df.merge(res_cell, on=["year", "month"], how="left")
                  .merge(range_df[["year", "month", "obs_min", "obs_max", "sigma_R", "var_R"]],
                         on=["year", "month"], how="left")
        )
        dfm["date"] = pd.to_datetime(dict(year=dfm["year"], month=dfm["month"], day=1))

        # 1-σ components at λ (for shaded bands)
        s2_rest  = np.clip(dfm["s2_rest"].to_numpy(float), 0, None)
        s2_split = np.clip(dfm[f"s2_{split_s}"].to_numpy(float), 0, None)
        h_rest   = np.abs(dfm["h_rest"].to_numpy(float))
        h_split  = np.abs(dfm["h_split"].to_numpy(float))
        sigma_rest  = np.sqrt(s2_rest)  * h_rest
        sigma_split = np.sqrt(s2_split) * h_split
        sigma_total = np.sqrt(sigma_rest**2 + sigma_split**2)  # (R shown separately)

        # Series
        t       = dfm["date"]
        obs     = dfm["obs"].to_numpy(float)
        prior   = dfm["prior"].to_numpy(float)
        split_c = dfm["split"].to_numpy(float)  # monthly center of split(λ)
        obs_min = dfm["obs_min"].to_numpy(float)
        obs_max = dfm["obs_max"].to_numpy(float)
        sigma_R = np.clip(dfm["sigma_R"].to_numpy(float), 0, np.inf)

        # ---- Plot ----
        ax = axes[i]

        # Shaded bands (toggle)
        if show_total_band:
            ax.fill_between(t, prior - sigma_total, prior + sigma_total,
                            color=band_total_color, alpha=0.25, label=f"μ ± $σ_{{rest+{split_s}}}$")
        if show_group_bands:
            ax.fill_between(t, prior - sigma_rest,  prior + sigma_rest,
                            color=band_rest_color,  alpha=0.22, label="μ ± $σ_{rest}$")
            ax.fill_between(t, prior - sigma_split, prior + sigma_split,
                            color=band_split_color, alpha=0.22, label=f"μ ± $σ_{{{split_s}}}$")

        # Prior line (μ = Hx^b)
        ax.plot(t, prior, color=prior_color, lw=2.0, label="prior (μ)")

        # --- NEW: Sensitivity lines μ ± p * split(λ) ---
        if show_sensitivity_line:
            delta = float(sensitivity_pct) * split_c  # p * h_split(λ)
            ax.plot(t, prior + delta, color=sensitivity_color, ls=sensitivity_linestyle,
                    lw=sensitivity_linewidth, label=f"μ + {sensitivity_pct:.0%}·{split_s}")
            ax.plot(t, prior - delta, color=sensitivity_color, ls=sensitivity_linestyle,
                    lw=sensitivity_linewidth, label=f"μ − {sensitivity_pct:.0%}·{split_s}")

        # Observations (points)
        ax.scatter(t, obs, s=18, color=obs_color, label=f"obs ({agg})")

        # Observation range bar (min–max) centered at obs
        if obs_range:
            yerr_obs = np.vstack([
                np.clip(obs - obs_min, 0, np.inf),
                np.clip(obs_max - obs, 0, np.inf)
            ])
            ax.errorbar(t, obs, yerr=yerr_obs, fmt="none",
                        ecolor=obs_color, elinewidth=1.2, capsize=2, alpha=0.95,
                        label="obs range (min–max)")

        # R bar (±σ_R) centered at obs (optional style; keeping per your last version)
        ax.errorbar(t, obs, yerr=sigma_R, fmt="none",
                    ecolor="k", elinewidth=1.0, capsize=0, alpha=0.8,
                    label=r"$\mathbf{R}$ ($\sigma_R$)")

        ax.set_title(f"{p.get('name','Point')}  (lon={p['lon']:.2f}, lat={p['lat']:.2f})  @ {wl} nm")
        ax.set_ylabel("AOD")
        ax.grid(alpha=0.3)

        # Legend (deduplicate)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="best", ncols=2, frameon=False, fontsize=9)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()


###################### REGION MASKS AND AGGREGATION ######################

# Default region boxes (longitude in 0--360) and surface-type defaults.
_REGION_DEFINITIONS = {
    'global':     {'lon_range': (0, 360),   'lat_range': (-90, 90),  'surface_type': 'all'},
    'africa':     {'lon_range': (15, 37),   'lat_range': (-15, 0),   'surface_type': 'land'},
    'amazon':     {'lon_range': (287, 317), 'lat_range': (-17, -3),  'surface_type': 'land'},
    'outflow_af': {'lon_range': (350, 8),   'lat_range': (-15, 3),   'surface_type': 'ocean'},
    'tropical':   {'lon_range': (0, 360),   'lat_range': (-23.5, 23.5), 'surface_type': 'all'},
}

_UNSET = object()


def _default_land_mask_path():
    """Return the project CAM5.3-Oslo land-fraction file path if it exists."""
    path = Path(__file__).parent.parent / (
        'Data/AEROCOM_III/CAM5.3-Oslo_AP3-CTRL2016-PD/'
        'aerocom3_CAM5.3-Oslo_AP3-CTRL2016-PD_landf_Surface_2010_monthly.nc'
    )
    return path if path.exists() else None


def shift360(data):
    """Shift the longitude from -180,180 to 0,360 and sort."""
    data_shifted = data.copy()
    data_shifted['lon'] = (data_shifted['lon'] + 360) % 360
    data_shifted = data_shifted.sortby('lon')
    return data_shifted


def create_mask_weigth(data, lon_initial, lon_final, lat_initial, lat_final):
    """Create a fractional-area lat/lon box mask (0--360 longitude)."""
    data_ones = xr.ones_like(data)
    dlat = np.abs(np.diff(data.lat.data)).mean()
    dlon = np.abs(np.diff(data.lon.data)).mean()

    if lon_initial > lon_final:
        mask_lon = data_ones.where((data.lon >= lon_initial) | (data.lon <= lon_final), 0)
    else:
        mask_lon = data_ones.where((data.lon >= lon_initial) & (data.lon <= lon_final), 0)

    mask_lat = mask_lon.where((data.lat >= lat_initial) & (data.lat <= lat_final), 0)

    # corner weights
    mask1 = mask_lat.where(~((np.abs(lat_initial - mask_lat.lat) < dlat / 2) &
                             (np.abs(lon_initial - mask_lat.lon) < dlon / 2)),
                           ((mask_lat.lat + dlat / 2 - lat_initial) / dlat) *
                           (dlon / 2 + mask_lat.lon - lon_initial) / dlon)

    mask2 = mask1.where(~((np.abs(lat_initial - mask1.lat) < dlat / 2) &
                          (np.abs(lon_final - mask1.lon) < dlon / 2)),
                        ((mask1.lat + dlat / 2 - lat_initial) / dlat) *
                        ((lon_final + dlon / 2 - mask1.lon) / dlon))

    mask3 = mask2.where(~((np.abs(lat_final - mask2.lat) < dlat / 2) &
                          (np.abs(lon_initial - mask2.lon) < dlon / 2)),
                        ((-mask2.lat + dlat / 2 + lat_final) / dlat) *
                        ((dlon / 2 + mask2.lon - lon_initial) / dlon))

    mask4 = mask3.where(~((np.abs(lat_final - mask3.lat) < dlat / 2) &
                          (np.abs(lon_final - mask3.lon) < dlon / 2)),
                        ((-mask3.lat + dlat / 2 + lat_final) / dlat) *
                        ((lon_final + dlon / 2 - mask3.lon) / dlon))

    mask_low = mask4.where(~((np.abs(lat_initial - mask4.lat) < dlat / 2) &
                             (lon_initial <= mask4.lon - dlon / 2) &
                             (lon_final >= mask4.lon + dlon / 2)),
                           ((mask4.lat + dlat / 2 - lat_initial) / dlat))

    mask_up = mask_low.where(~((np.abs(lat_final - mask_low.lat) < dlat / 2) &
                               (lon_initial <= mask_low.lon - dlon / 2) &
                               (lon_final >= mask_low.lon + dlon / 2)),
                             ((-mask_low.lat + dlat / 2 + lat_final) / dlat))

    mask_left = mask_up.where(~((lat_initial <= mask_up.lat - dlat / 2) &
                                (lat_final >= mask_up.lat + dlat / 2) &
                                (np.abs(lon_initial - mask_up.lon) < dlon / 2)),
                              ((dlon / 2 + mask_up.lon - lon_initial) / dlon))

    mask_right = mask_left.where(~((lat_initial <= mask_left.lat - dlat / 2) &
                                   (lat_final >= mask_left.lat + dlat / 2) &
                                   (np.abs(lon_final - mask_left.lon) < dlon / 2)),
                                 ((lon_final + dlon / 2 - mask_left.lon) / dlon))

    return mask_right


def create_mask(data, lon_initial, lon_final, lat_initial, lat_final):
    """Create a binary 0/1 lat/lon box mask (cell-center based)."""
    data_ones = xr.ones_like(data)

    if lon_initial > lon_final:
        mask_lon = data_ones.where((data.lon >= lon_initial) | (data.lon <= lon_final), 0)
    else:
        mask_lon = data_ones.where((data.lon >= lon_initial) & (data.lon <= lon_final), 0)

    mask_lat = mask_lon.where((data.lat >= lat_initial) & (data.lat <= lat_final), 0)
    return mask_lat


def _normalize_lon_range(lon_range):
    """Accept (lon_min, lon_max) in 0--360; None means full longitude."""
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
    raise TypeError("time_slice must be None, a slice, or a (start, end) pair of time labels")


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
    # Also drop any scalar time coordinate that may have been preserved by isel
    for coord in list(template.coords):
        if coord not in ('lat', 'lon'):
            template = template.drop_vars(coord, errors='ignore')
    for dim in list(template.dims):
        if dim not in ('lat', 'lon'):
            template = template.isel({dim: 0}, drop=True)
    return template


def _grids_match(m, t):
    """Return True if two (lat, lon) DataArrays share the same grid."""
    return (
        len(m.lat) == len(t.lat)
        and len(m.lon) == len(t.lon)
        and np.array_equal(m.lat.values, t.lat.values)
        and np.array_equal(m.lon.values, t.lon.values)
    )


def land_mask_from_cartopy(template, resolution='110m'):
    """
    Build a 0/1 land mask from the Cartopy Natural Earth land feature.

    Returns 1 for land and 0 for ocean on the template lat/lon grid.  The
    template may use either 0--360 or -180--180 longitude; the polygons are
    tested in the -180--180 frame and mapped back to the original grid.
    """
    import cartopy.feature as cfeature
    from shapely.vectorized import contains
    from shapely.ops import unary_union

    lat = np.asarray(template.lat.values)
    lon = np.asarray(template.lon.values)

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


def _land_sea_mask(template, surface_type, land_mask=None, land_mask_path=None):
    """
    Return a 0/1 mask on the template grid for land or sea.

    surface_type: 'all', 'land', 'sea', or 'ocean' (ocean is treated as sea)
    land_mask: xr.DataArray with 1=land, 0=sea (or fractional)
    land_mask_path: netCDF path; looks for lsm, land, land_mask, or landf variables
    """
    surface_type = 'all' if surface_type is None else surface_type

    if surface_type == 'ocean':
        surface_type = 'sea'

    if surface_type not in ('all', 'land', 'sea', 'ocean'):
        raise ValueError("surface_type must be 'all', 'land', 'sea', or 'ocean'")

    if surface_type == 'all':
        return xr.ones_like(template)

    if land_mask is None and land_mask_path is None:
        default_path = _default_land_mask_path()
        if default_path is not None:
            land_mask_path = str(default_path)

    if land_mask is not None:
        lsm = land_mask
    elif land_mask_path is not None:
        try:
            with xr.open_dataset(land_mask_path) as ds:
                for var in ('lsm', 'land', 'land_mask', 'LAND_MASK', 'landf'):
                    if var in ds:
                        lsm = ds[var].load()
                        break
                else:
                    raise KeyError(
                        f"No land-mask variable found in {land_mask_path}. "
                        "Expected one of: lsm, land, land_mask, LAND_MASK, landf"
                    )
        except Exception as exc:
            # Fall back to cartopy/regionmask if the provided/default file fails
            try:
                import regionmask
                land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
                lsm = land.mask(template.lon, template.lat) == 0
            except Exception:
                try:
                    lsm = land_mask_from_cartopy(template)
                except Exception as exc2:
                    raise ValueError(
                        "Land/sea masking requires land_mask, land_mask_path, "
                        "the 'regionmask' package, or cartopy+shapely."
                    ) from exc2
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

    # If the land mask has a time dimension, take the first time step BEFORE
    # any interpolation so that the time coordinate cannot conflict with the
    # spatial template grid.
    if 'time' in lsm.dims:
        lsm = lsm.isel(time=0, drop=True)
    for coord in list(lsm.coords):
        if coord not in ('lat', 'lon'):
            lsm = lsm.drop_vars(coord, errors='ignore')

    if 'lat' in lsm.dims and 'lon' in lsm.dims:
        if not np.array_equal(lsm.lat.values, template.lat.values) or not np.array_equal(lsm.lon.values, template.lon.values):
            lsm = lsm.interp(lat=template.lat, lon=template.lon, method='nearest')

    lsm = (lsm >= 0.5).astype(float)

    if surface_type == 'land':
        return lsm
    return 1.0 - lsm


def _spatial_weights(da, mask, edge_weighted=False):
    """
    Area weights for lat/lon aggregation.

    When the mask grid does not match the model's grid, the mask is rebuilt
    on the model's grid using the stored lon_range/lat_range attrs, or
    interpolated if those attrs are absent.
    """
    dlat, dlon = _grid_cell_sizes(da)
    data_template = _spatial_template(da)

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


def create_region_mask(
    template,
    region=None,
    surface_type=_UNSET,
    lon_range=None,
    lat_range=None,
    name=None,
    surface=None,
    land_mask=None,
    land_mask_path=None,
    mask_registry=None,
):
    """
    Build a named regional mask on the same lat/lon grid as `template`.

    New interface (recommended)
    ---------------------------
    Use the `region` and `surface_type` arguments for the four study regions:

    >>> create_region_mask(pr, region='africa')               # land by default
    >>> create_region_mask(pr, region='outflow_af', surface_type='ocean')
    >>> create_region_mask(pr, region='global', surface_type='all')

    Parameters
    ----------
    template : xr.DataArray or xr.Dataset
        Reference field (only lat/lon coordinates are used). Use any loaded
        AeroCom variable, e.g. model_data['AOD550'].isel(time=0).
    region : str, optional
        One of 'global', 'africa', 'amazon', 'outflow_af', or 'tropical'.
        When passed as the second positional argument it is auto-detected.
    surface_type : {'all', 'land', 'ocean'}, optional
        'land' / 'ocean' apply a land-sea mask. Region defaults:
        global='all', africa='land', amazon='land', outflow_af='ocean'.
    lon_range, lat_range : tuple, optional
        Override the region defaults. Enables custom regions.
    name : str, optional
        Name stored in mask.attrs. Defaults to `region`.
    surface : str, optional
        Backward-compatible alias for `surface_type`.
    land_mask : xr.DataArray, optional
        1=land, 0=sea on the same or an interpolatable grid.
    land_mask_path : str or Path, optional
        Path to a netCDF land-sea mask file. If None and surface_type is
        'land'/'ocean', the project CAM5.3-Oslo land-fraction file is tried,
        falling back to cartopy/regionmask if it is unavailable.
    mask_registry : dict, optional
        If given, stores the mask as mask_registry[name] = mask.

    Returns
    -------
    xr.DataArray
        Mask with values 0/1 (lat, lon). Metadata is stored in .attrs for use
        with `regional_aggregate(..., edge_weighted=True)`.
    """
    if isinstance(template, xr.Dataset):
        template = template[list(template.data_vars)[0]]

    # Backward compatibility: if the second positional argument is not a
    # recognized region, treat it as the old `name` argument.
    if region is not None and region not in _REGION_DEFINITIONS:
        if name is None:
            name = region
        region = None

    # Resolve region defaults
    if region is not None:
        cfg = _REGION_DEFINITIONS[region]
        lon_range = cfg['lon_range'] if lon_range is None else lon_range
        lat_range = cfg['lat_range'] if lat_range is None else lat_range
        if surface_type is _UNSET:
            surface_type = cfg.get('surface_type', 'all')
        if name is None:
            name = region
    else:
        if lon_range is None:
            lon_range = (0.0, 360.0)
        if lat_range is None:
            lat_range = (-90.0, 90.0)
        if name is None:
            raise ValueError("name is required when region is not specified")
        if surface_type is _UNSET:
            surface_type = 'all'

    # Backward-compatible alias
    if surface is not None:
        surface_type = surface

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
        surface_type,
        land_mask=land_mask,
        land_mask_path=land_mask_path,
    )
    mask = (box_mask * surface_mask).astype(float)
    mask.name = name
    mask.attrs.update({
        'name': name,
        'lon_range': lon_range if lon_range is not None else (0.0, 360.0),
        'lat_range': lat_range if lat_range is not None else (-90.0, 90.0),
        'surface': surface_type,
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
    skipna=False,
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
    skipna : bool
        If True, NaN values are omitted from the aggregation (both spatial and temporal).
        This is useful when fields such as lifetime have been masked for outliers.

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
        if skipna:
            norm = weights.where(da_work.notnull()).sum(dim=['lat', 'lon'])
            spatial_result = weighted_field.sum(dim=['lat', 'lon'], skipna=True) / norm.where(norm > 0)
        else:
            norm = weights.sum(dim=['lat', 'lon'])
            spatial_result = weighted_field.sum(dim=['lat', 'lon']) / norm
    else:
        spatial_result = weighted_field.sum(dim=['lat', 'lon'], skipna=skipna)

    if return_time_series or 'time' not in spatial_result.dims:
        return spatial_result

    if temporal == 'mean':
        return float(spatial_result.mean('time', skipna=skipna).values)
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
