"""
kkpy.util
========================

Utility functions for my research

.. currentmodule:: util

wind

.. autosummary::
    kkpy.util.wind2uv
    kkpy.util.uv2wind
    kkpy.util.ms2knot
    kkpy.util.knot2ms
    kkpy.util.cross_section
    kkpy.util.proj_dfs

"""
import numpy as np

def wind2uv(wd=None, ws=None, radians=False, knots=False):
    """
    Convert wind direction and speed to u and v wind components.
    
    Examples
    ---------
    >>> u, v = kkpy.util.wind2uv(wd, ws)
    
    Parameters
    ----------
    wd : array_like
        Array containing wind direction in **degree**. It should be **meteorological** direction, not mathematical.
    ws : array_like
        Array containing wind speed in **m/s**.
    radians : bool, optional
        If this is set to True, the unit of *wd* is **radian**. The default is False (i.e. **degree**).
    knots : bool, optional
        If this is set to True, the unit of *ws* is **knots**. The default is False (i.e. **m/s**).
        
    Returns
    ---------
    u : ndarray
        Return u component of wind in **m/s**.
    v : ndarray
        Return v component of wind in **m/s**.
    """
    if not radians:
        wd = np.radians(wd)
    
    if knots:
        ws = ms2knot(ws)
    
    u = -ws * np.sin(wd)
    v = -ws * np.cos(wd)
    
    return u, v

def uv2wind(u, v):
    """
    Convert u and v wind components to wind direction and speed.
    
    Examples
    ---------
    >>> wd, ws = kkpy.util.uv2wind(u, v)
    
    Parameters
    ----------
    u : array_like
        Array containing u component of wind in **m/s**.
    v : array_like
        Array containing v component of wind in **m/s**.
        
    Returns
    ---------
    wd : ndarray
        Return wind direction in **degree**.
    ws : ndarray
        Return wind speed in **m/s**.
    """
        
    ws = np.hypot(u, v)
    wd = 270 - np.rad2deg(np.arctan2(v, u))
    wd = wd % 360
    
    return wd, ws



def ms2knot(ws_ms):
    """
    Convert unit of wind speed from meter per seconds to knots.
    
    Examples
    ---------
    >>> ws_knot = kkpy.util.ms2knot(ws_ms)
    
    Parameters
    ----------
    ws_ms : array_like
        Array containing wind speed in **m/s**.
        
    Returns
    ---------
    ws_knot : ndarray
        Return wind speed in **knots**.
    """
    
    ws_knot = ws_ms * 1. / 0.5144444
    
    return ws_knot

def knot2ms(ws_knot):
    """
    Convert unit of wind speed from knots to meter per seconds.
    
    Examples
    ---------
    >>> ws_ms = kkpy.util.knot2ms(ws_knot)
    
    Parameters
    ----------
    ws_knot : array_like
        Array containing wind speed in **knots**.
        
    Returns
    ---------
    ws_ms : ndarray
        Return wind speed in **m/s**.
    """
    
    ws_ms = ws_knot * 0.5144444
    
    return ws_ms

def cross_section_2d(dict_start, dict_end, lon2d, lat2d, value2d, avg_halfwidth=0, along='longitude'):
    """
    Get mean values along the transect of two points (lon/lat).
    
    Examples
    ---------
    >>> dict_start = {'lon':128.2, 'lat':37.24}
    >>> dict_end = {'lon':129.35, 'lat':38.2}
    >>> xaxis_cross, value_cross, index_cross = kkpy.util.cross_section_2d(dict_start, dict_end, lon2d, lat2d, value2d)
    >>> plt.plot(xaxis_cross, value_cross)
    >>> plt.show()
    
    Parameters
    ----------
    dict_start : dict
        Dictionary with lon and lat keys of starting points.
    dict_end : dict
        Dictionary with lon and lat keys of ending points.
    lon2d : 2D array
        Numpy 2D array containing longitude of each grid point. The shape of lon2d, lat2d, and value2d should be same.
    lat2d : 2D array
        Numpy 2D array containing latitude of each grid point. The shape of lon2d, lat2d, and value2d should be same.
    value2d : 2D array
        Numpy 2D array containing value of each grid point. The shape of lon2d, lat2d, and value2d should be same.
    avg_halfwidth : int
        The half width used in average over perpendicular direction along the transect. Default is zero (i.e. no average).
    along : str
        Return xaxis_cross as longitude if it is set to 'longitude', otherwise return it as latitude if 'latitude'.
    
    Returns
    ---------
    xaxis_cross : 1D array
        Return xaxis of cross-section in longitude or latitude unit. The unit is determined by along keyword.
    value_cross : 1D array
        Return averaged value along the cross-section.
    index_cross : dict
        Return a dictionary contatining the indicies of starting point and ending point of the transect.
    """
    
    if np.abs(lon2d[1,0] - lon2d[0,0]) > np.abs(lon2d[0,1] - lon2d[0,0]):
        lon_is_first_order = True
    else:
        lon_is_first_order = False
    
    if 'LONGITUDE' in along.upper():
        along_lon = True
    elif 'LATITUDE' in along.upper():
        along_lon = False
    else:
        sys.exit(f'{__name__}: along keyword should be LONGITUDE or LATITUDE')
        
    
    idx_start = np.unravel_index(np.argmin((lon2d - dict_start['lon'])**2 + (lat2d - dict_start['lat'])**2), lon2d.shape)
    idx_end = np.unravel_index(np.argmin((lon2d - dict_end['lon'])**2 + (lat2d - dict_end['lat'])**2), lon2d.shape)
    
    cnt_cross_idx = np.max([np.abs(idx_end[0]-idx_start[0]), np.abs(idx_end[1]-idx_start[1])])
    if lon_is_first_order:
        i1d_lon = np.int_(np.round(np.linspace(idx_start[0], idx_end[0], cnt_cross_idx)))
        i1d_lat = np.int_(np.round(np.linspace(idx_start[1], idx_end[1], cnt_cross_idx)))
    else:
        i1d_lat = np.int_(np.round(np.linspace(idx_start[0], idx_end[0], cnt_cross_idx)))
        i1d_lon = np.int_(np.round(np.linspace(idx_start[1], idx_end[1], cnt_cross_idx)))
        
    # Get averaged cross-section
    value_cross = np.empty((cnt_cross_idx))
    for ii, _i1d_lat in enumerate(i1d_lat):
        # perpendicular to cross line
        value_cross[ii] = np.nanmean(value2d[i1d_lon[ii]-avg_halfwidth:i1d_lon[ii]+avg_halfwidth+1, \
                                             i1d_lat[ii]+avg_halfwidth+1:i1d_lat[ii]-avg_halfwidth:-1] \
                                            [np.arange(avg_halfwidth*2+1),np.arange(avg_halfwidth*2+1)])
    
    if along_lon:
        xaxis_cross = np.linspace(dict_start['lon'],
                                  dict_end['lon'],
                                  cnt_cross_idx)
    else:
        xaxis_cross = np.linspace(dict_start['lat'],
                                  dict_end['lat'],
                                  cnt_cross_idx)
    
    idx_lon = {'start':i1d_lon[0],
               'end':i1d_lon[-1],
               'lower_start':i1d_lon[0]+avg_halfwidth+1,
               'lower_end':i1d_lon[-1]+avg_halfwidth+1,
               'upper_start':i1d_lon[0]-avg_halfwidth,
               'upper_end':i1d_lon[-1]-avg_halfwidth}
    idx_lat = {'start':i1d_lat[0],
               'end':i1d_lat[-1],
               'lower_start':i1d_lat[0]-avg_halfwidth,
               'lower_end':i1d_lat[-1]-avg_halfwidth,
               'upper_start':i1d_lat[0]+avg_halfwidth+1,
               'upper_end':i1d_lat[-1]+avg_halfwidth+1}
    index_cross = {'lon':idx_lon,
                   'lat':idx_lat}
    
    return xaxis_cross, value_cross, index_cross

def proj_dfs():
    """
    Return a lambert conformal conic projection of DFS (Digital Forecasting System) used in KMA.
    
    Examples
    ---------
    >>> ax = plt.subplot(proj=kkpy.util.proj_dfs())
    >>> ax.scatter([126], [38], transform=ccrs.PlateCarree())
    >>> plt.show()
    
    Returns
    ---------
    proj : cartopy projection
        Return a map projection of DFS.
    """
    import cartopy.crs as ccrs
    
    globe = ccrs.Globe(ellipse=None,
                       semimajor_axis=6371008.77,
                       semiminor_axis=6371008.77)
    
    proj = ccrs.LambertConformal(central_longitude=126,
                                 central_latitude=38,
                                 standard_parallels=(30,60),
                                 false_easting=400000,
                                 false_northing=789000,
                                 globe=globe)
    return proj
