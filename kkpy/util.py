"""
kkpy.util
========================

Utility functions for my research

.. currentmodule:: util

Winds
-------
.. autosummary::
    kkpy.util.wind2uv
    kkpy.util.uv2wind
    kkpy.util.ms2knot
    kkpy.util.knot2ms

Maps
-------
.. autosummary::
    kkpy.util.proj_dfs
    kkpy.util.dist_bearing

Spatial calculations
----------------------
.. autosummary::
    kkpy.util.cross_section_2d
    kkpy.util.cross_section_3d

Miscellaneous
---------------
.. autosummary::
    kkpy.util.std2d
    kkpy.util.nanstd2d
    kkpy.util.nanconvolve2d
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
    Get mean values along the transect of two points (lon/lat) for 2D array.
    
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
    import bottleneck as bn
    
    # check if longitude is the first dimension
    lon_is_first_order = _get_lon_is_first_order(lon2d)
    
    # check if data is stored from south to north
    lat_south_to_north = _get_lat_south_to_north(lat2d, lon_is_first_order)
    
    # check if output will be along longitude (E-W direction)
    along_lon = _get_along_lon(along, __name__)
    
    # find index of end point for cross section
    i1d_x, i1d_y, cnt_cross_idx = _get_i1d_for_cross(lon2d, lat2d, dict_start, dict_end, lon_is_first_order)
    
    # Get averaged value along cross section
    value_cross = np.empty((cnt_cross_idx))
    for ii, _i1d_y in enumerate(i1d_y):
        slicex = slice(i1d_x[ii]-avg_halfwidth, i1d_x[ii]+avg_halfwidth+1)
        if lat_south_to_north:
            slicey = slice(i1d_y[ii]+avg_halfwidth+1, i1d_y[ii]-avg_halfwidth, -1)
        else:
            slicey = slice(i1d_y[ii]-avg_halfwidth, i1d_y[ii]+avg_halfwidth+1)
        
        try:
            # perpendicular to cross line
            target = value2d[slicex,slicey]
            target = target.diagonal()
            value_cross[ii] = bn.nanmean(target)
        except RuntimeWarning:
            value_cross[ii] = np.nan
    
    # lon or lat values along cross section
    xaxis_cross = _get_xaxis_cross(dict_start, dict_end, cnt_cross_idx, along_lon)
    
    # dictionary for return value
    index_cross = _get_index_cross(i1d_x, i1d_y, avg_halfwidth)
    
    return xaxis_cross, value_cross, index_cross

def cross_section_3d(dict_start, dict_end, lon2d, lat2d, value3d, avg_halfwidth=0, along='longitude'):
    """
    Get mean values along the transect of two points (lon/lat) for 3D array.
    
    Examples
    ---------
    >>> dict_start = {'lon':128.2, 'lat':37.24}
    >>> dict_end = {'lon':129.35, 'lat':38.2}
    >>> zaxis = np.arange(value3d.shape[2])
    >>> xaxis_cross, value_cross, index_cross = kkpy.util.cross_section_3d(dict_start, dict_end, lon2d, lat2d, value3d)
    >>> plt.pcolormesh(xaxis_cross, zaxis, value_cross.T)
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
    value3d : 3D array
        Numpy 3D array containing value of each grid point. The size of first two dimension should be same with lon2d and lat2d. The third dimension is assumed to have a height axis.
    avg_halfwidth : int
        The half width used in average over perpendicular direction along the transect. Default is zero (i.e. no average).
    along : str
        Return xaxis_cross as longitude if it is set to 'longitude', otherwise return it as latitude if 'latitude'.
    
    Returns
    ---------
    xaxis_cross : 1D array
        Return xaxis of cross-section in longitude or latitude unit. The unit is determined by along keyword.
    value_cross : 2D array
        Return averaged value along the cross-section. The shape is (xaxis_cross, value3d.shape[2]).
    index_cross : dict
        Return a dictionary contatining the indicies of starting point and ending point of the transect.
    """
    import bottleneck as bn
    
    # check if longitude is the first dimension
    lon_is_first_order = _get_lon_is_first_order(lon2d)
    
    # check if data is stored from south to north
    lat_south_to_north = _get_lat_south_to_north(lat2d, lon_is_first_order)
    
    # check if output will be along longitude (E-W direction)
    along_lon = _get_along_lon(along, __name__)
    
    # find index of end point for cross section
    i1d_x, i1d_y, cnt_cross_idx = _get_i1d_for_cross(lon2d, lat2d, dict_start, dict_end, lon_is_first_order)
        
    # Get averaged value along cross section
    value_cross = np.empty((cnt_cross_idx, value3d.shape[2]))
    for ii, _i1d_y in enumerate(i1d_y):
        slicex = slice(i1d_x[ii]-avg_halfwidth, i1d_x[ii]+avg_halfwidth+1)
        if lat_south_to_north:
            slicey = slice(i1d_y[ii]+avg_halfwidth+1, i1d_y[ii]-avg_halfwidth, -1)
        else:
            slicey = slice(i1d_y[ii]-avg_halfwidth, i1d_y[ii]+avg_halfwidth+1)
        
        try:
            # perpendicular to cross line
            target = value3d[slicex,slicey,:]
            target = target.diagonal(axis1=0, axis2=1)
            value_cross[ii,:] = bn.nanmean(target, axis=1)
        except RuntimeWarning:
            value_cross[ii,:] = np.full(value3d.shape[2], np.nan)
    
    # lon or lat values along cross section
    xaxis_cross = _get_xaxis_cross(dict_start, dict_end, cnt_cross_idx, along_lon)
    
    # dictionary for return value
    index_cross = _get_index_cross(i1d_x, i1d_y, avg_halfwidth)
    
    return xaxis_cross, value_cross, index_cross

def _get_lon_is_first_order(lon2d):
    if (lon2d[1,0] - lon2d[0,0])**2 > (lon2d[0,1] - lon2d[0,0])**2:
        lon_is_first_order = True # lon2d[lon,lat], lat2d[lon,lat], value2d[lon,lat]
    else:
        lon_is_first_order = False # lon2d[lat,lon], lat2d[lat,lon], value2d[lat,lon]
    
    return lon_is_first_order

def _get_lat_south_to_north(lat2d, lon_is_first_order):
    if lon_is_first_order:
        if lat2d[0,1] - lat2d[0,0] > 0:
            lat_south_to_north = True # data is stored from south to north
        else:
            lat_south_to_north = False # data is stored from north to south
    else:
        if lat2d[1,0] - lat2d[0,0] > 0:
            lat_south_to_north = True # data is stored from south to north
        else:
            lat_south_to_north = False # data is stored from north to south
    
    return lat_south_to_north

def _get_along_lon(along, parent_funcname):
    if 'LONGITUDE' in along.upper():
        along_lon = True
    elif 'LATITUDE' in along.upper():
        along_lon = False
    else:
        sys.exit(f'{parent_funcname}: along keyword should be LONGITUDE or LATITUDE')
    
    return along_lon

def _get_i1d_for_cross(lon2d, lat2d, dict_start, dict_end, lon_is_first_order):
    import bottleneck as bn
    diff_start = np.asfortranarray((lon2d - dict_start['lon'])**2 + (lat2d - dict_start['lat'])**2)
    diff_end = np.asfortranarray((lon2d - dict_end['lon'])**2 + (lat2d - dict_end['lat'])**2)
    idx_start = np.unravel_index(bn.nanargmin(diff_start), lon2d.shape)
    idx_end = np.unravel_index(bn.nanargmin(diff_end), lon2d.shape)
    
    # find indices along cross section
    cnt_cross_idx = np.max([np.abs(idx_end[0]-idx_start[0]), np.abs(idx_end[1]-idx_start[1])])
    if lon_is_first_order:
        i1d_x = np.int_(np.round(np.linspace(idx_start[0], idx_end[0], cnt_cross_idx)))
        i1d_y = np.int_(np.round(np.linspace(idx_start[1], idx_end[1], cnt_cross_idx)))
    else:
        i1d_y = np.int_(np.round(np.linspace(idx_start[0], idx_end[0], cnt_cross_idx)))
        i1d_x = np.int_(np.round(np.linspace(idx_start[1], idx_end[1], cnt_cross_idx)))
    
    return i1d_x, i1d_y, cnt_cross_idx

def _get_xaxis_cross(dict_start, dict_end, cnt_cross_idx, along_lon):
    if along_lon:
        xaxis_cross = np.linspace(dict_start['lon'],
                                  dict_end['lon'],
                                  cnt_cross_idx)
    else:
        xaxis_cross = np.linspace(dict_start['lat'],
                                  dict_end['lat'],
                                  cnt_cross_idx)
    return xaxis_cross

def _get_index_cross(i1d_x, i1d_y, avg_halfwidth):
    idx_lon = {'start':i1d_x[0],
               'end':i1d_x[-1],
               'lower_start':i1d_x[0]+avg_halfwidth+1,
               'lower_end':i1d_x[-1]+avg_halfwidth+1,
               'upper_start':i1d_x[0]-avg_halfwidth,
               'upper_end':i1d_x[-1]-avg_halfwidth}
    idx_lat = {'start':i1d_y[0],
               'end':i1d_y[-1],
               'lower_start':i1d_y[0]-avg_halfwidth,
               'lower_end':i1d_y[-1]-avg_halfwidth,
               'upper_start':i1d_y[0]+avg_halfwidth+1,
               'upper_end':i1d_y[-1]+avg_halfwidth+1}
    index_cross = {'lon':idx_lon,
                   'lat':idx_lat}
    
    return index_cross

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

def dist_bearing(lonlat0, lonlat1, radians=False):
    """
    Get distance [km] and bearing [deg] between two lon/lat points.
    
    Examples
    ---------
    >>> dist_km, bearing_deg = kkpy.util.dist_bearing([127.5,36], [130,37])
    
    Parameters
    ----------
    lonlat0 : 1D Array
        Array containing longitude and latitude of the first point. Longitude (latitude) should be at the first (second) element.
    lonlat1 : 1D Array
        Array containing longitude and latitude of the second point. Longitude (latitude) should be at the first (second) element.
    radians : bool, optional
        If this is set to True, the unit of *bearing* is **radian**. The default is False (i.e. **degree**).
        
    Returns
    ---------
    distance : float
        Return distance between two points in **km**.
    bearing : ndarray
        Return bearing of two points in **degree**. If radians is True, the unit is **radians**.
    """
    from haversine import haversine
    from math import sin, cos, atan2
    lon0 = lonlat0[0]
    lat0 = lonlat0[1]
    lon1 = lonlat1[0]
    lat1 = lonlat1[1]
    
    dist = haversine((lat0,lon0),(lat1,lon1)) # km
    
    rlat0, rlon0, rlat1, rlon1 = np.radians((lat0, lon0, lat1, lon1))
    coslt0,  sinlt0  = cos(rlat0), sin(rlat0)
    coslt1,  sinlt1  = cos(rlat1), sin(rlat1)
    cosl0l1, sinl0l1 = cos(rlon1-rlon0), sin(rlon1-rlon0)
    cosc = sinlt0*sinlt1 + coslt0*coslt1*cosl0l1
    sinc = np.sqrt(1.0 - cosc**2)
    cosaz = (coslt0*sinlt1 - sinlt0*coslt1*cosl0l1) / sinc
    sinaz = sinl0l1*coslt1/sinc
    
    bearing = np.arctan(sinaz/cosaz)
    if not radians:
        bearing = np.degrees(bearing)
    
    return dist, bearing

def nanconvolve2d(slab, kernel, max_missing=0.99):
    """
    Get 2D convolution with missings ignored.
    
    Examples
    ---------
    Moving 2D standard deviation
    
    >>> import astropy.convolution
    >>> kernel = np.array(astropy.convolution.Box2DKernel(5))
    >>> c1 = kkpy.util.nanconvolve2d(arr2d, kernel)
    >>> c2 = kkpy.util.nanconvolve2d(arr2d*arr2d, kernel)
    >>> stddev2d = np.sqrt(c2 - c1*c1)
    
    Moving 2D average
    
    >>> import astropy.convolution
    >>> kernel = np.array(astropy.convolution.Box2DKernel(5))
    >>> avg2d = kkpy.util.nanconvolve2d(arr2d, kernel)
    
    Parameters
    ----------
    slab : 2D Array
        Input array to convolve. Can have numpy.nan or masked values.
    kernel : 1D Array
        Convolution kernel, must have sizes as odd numbers.
    max_missing : float, optional
        Float in (0,1), max percentage of missing in each convolution window is tolerated before a missing is placed in the result.
        
    Returns
    ---------
    result : 2D Array
        Return convolution result. Missings are represented as numpy.nans if they are in slab, or masked if they are masked in slab.
    
    Notes
    ---------
    This code is from Stack Overflow answer (https://stackoverflow.com/a/40416633/12272819), written by Jason (https://stackoverflow.com/users/2005415/jason).
    This is licensed under the Creative Commons Attribution-ShareAlike 3.0 license (CC BY-SA 3.0).
    Modified by Kwonil Kim in November 2020: modify docstring format, remove verbose argument, modify default value of max_missing, change numpy to np
    """
    from scipy.ndimage import convolve as sciconvolve

    assert np.ndim(slab)==2, "<slab> needs to be 2D."
    assert np.ndim(kernel)==2, "<kernel> needs to be 2D."
    assert kernel.shape[0]%2==1 and kernel.shape[1]%2==1, "<kernel> shape needs to be an odd number."
    assert max_missing > 0 and max_missing < 1, "<max_missing> needs to be a float in (0,1)."

    #--------------Get mask for missings--------------
    if not hasattr(slab,'mask') and np.any(np.isnan(slab))==False:
        has_missing=False
        slab2=slab.copy()

    elif not hasattr(slab,'mask') and np.any(np.isnan(slab)):
        has_missing=True
        slabmask=np.where(np.isnan(slab),1,0)
        slab2=slab.copy()
        missing_as='nan'

    elif (slab.mask.size==1 and slab.mask==False) or np.any(slab.mask)==False:
        has_missing=False
        slab2=slab.copy()

    elif not (slab.mask.size==1 and slab.mask==False) and np.any(slab.mask):
        has_missing=True
        slabmask=np.where(slab.mask,1,0)
        slab2=np.where(slabmask==1,np.nan,slab)
        missing_as='mask'

    else:
        has_missing=False
        slab2=slab.copy()

    #--------------------No missing--------------------
    if not has_missing:
        result=sciconvolve(slab2,kernel,mode='constant',cval=0.)
    else:
        H,W=slab.shape
        hh=int((kernel.shape[0]-1)/2)  # half height
        hw=int((kernel.shape[1]-1)/2)  # half width
        min_valid=(1-max_missing)*kernel.shape[0]*kernel.shape[1]

        # dont forget to flip the kernel
        kernel_flip=kernel[::-1,::-1]

        result=sciconvolve(slab2,kernel,mode='constant',cval=0.)
        slab2=np.where(slabmask==1,0,slab2)

        #------------------Get nan holes------------------
        miss_idx=zip(*np.where(slabmask==1))

        if missing_as=='mask':
            mask=np.zeros([H,W])

        for yii,xii in miss_idx:

            #-------Recompute at each new nan in result-------
            hole_ys=range(max(0,yii-hh),min(H,yii+hh+1))
            hole_xs=range(max(0,xii-hw),min(W,xii+hw+1))

            for hi in hole_ys:
                for hj in hole_xs:
                    hi1=max(0,hi-hh)
                    hi2=min(H,hi+hh+1)
                    hj1=max(0,hj-hw)
                    hj2=min(W,hj+hw+1)

                    slab_window=slab2[hi1:hi2,hj1:hj2]
                    mask_window=slabmask[hi1:hi2,hj1:hj2]
                    kernel_ij=kernel_flip[max(0,hh-hi):min(hh*2+1,hh+H-hi), 
                                     max(0,hw-hj):min(hw*2+1,hw+W-hj)]
                    kernel_ij=np.where(mask_window==1,0,kernel_ij)

                    #----Fill with missing if not enough valid data----
                    ksum=np.sum(kernel_ij)
                    if ksum<min_valid:
                        if missing_as=='nan':
                            result[hi,hj]=np.nan
                        elif missing_as=='mask':
                            result[hi,hj]=0.
                            mask[hi,hj]=True
                    else:
                        result[hi,hj]=np.sum(slab_window*kernel_ij)

        if missing_as=='mask':
            result=np.ma.array(result)
            result.mask=mask

    return result

def nanstd2d(X, window_size):
    """
    Get 2D standard deviation of 2D array efficiently with missings ignored.
    
    Examples
    ---------
    >>> std2d = kkpy.util.nanstd2d(arr2d, 3)
    
    Parameters
    ----------
    X : 2D Array
        Array containing the data.
    window_size : float
        Window size of x and y. Window sizes of x and y should be same.
        
    Returns
    ---------
    std2d : 2D Array
        Return 2D standard deviation.
    """
    import astropy.convolution
    kernel = np.array(astropy.convolution.Box2DKernel(window_size))
    c1 = nanconvolve2d(X, kernel)
    c2 = nanconvolve2d(X*X, kernel)
    return np.sqrt(c2 - c1*c1)

def std2d(X, window_size):
    """
    Get 2D standard deviation of 2D array efficiently.
    
    Examples
    ---------
    >>> std2d = kkpy.util.std2d(arr2d, 3)
    
    Parameters
    ----------
    X : 2D Array
        Array containing the data.
    window_size : float or 1D array
        Window size. If array of two elements, window sizes of x and y will be window_size[0] and window_size[1], respectively.
        
    Returns
    ---------
    std2d : 2D Array
        Return 2D standard deviation.
    
    Notes
    ---------
    This code is from https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html, written by Nick Cortale.
    Modified by Kwonil Kim in November 2020: add docstring, modify function name
    """
    from scipy.ndimage.filters import uniform_filter

    r,c = X.shape
    X+=np.random.rand(r,c)*1e-6
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    return np.sqrt(c2 - c1*c1)
