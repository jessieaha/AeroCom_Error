"""Shared 1° lat/lon grid for ICON binning and the observation maps."""

from __future__ import annotations

import numpy as np

LAT = np.arange(-89.5, 90.0, 1.0)
LON = np.arange(0.5, 360.0, 1.0)


def bin_indices(lon, lat):
    """Cell indices on the 1° grid. Longitude may be −180..180 or 0..360."""
    lon = np.mod(np.asarray(lon, dtype=float), 360.0)
    lat = np.asarray(lat, dtype=float)
    ix = np.floor(lon).astype(np.int32) % LON.size
    iy = np.clip(np.floor(lat + 90.0).astype(np.int32), 0, LAT.size - 1)
    return iy, ix
