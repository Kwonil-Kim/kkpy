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

Radars
--------
.. autosummary::
    kkpy.util.dbzmean

Maps
-------
.. autosummary::
    kkpy.util.proj_dfs
    kkpy.util.dist_bearing

Spatial calculations
----------------------
.. autosummary::
    kkpy.util.nn_idx_2d
    kkpy.util.cross_section_2d

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

def nn_idx_2d(xtarget, ytarget, x2d, y2d):
    """
    Get nearest indices of the x and y values of target if 2D array is given.
    
    Examples
    ---------
    >>> x2d, y2d = np.meshgrid(np.arange(128,130,0.1), np.arange(36,38,0.1))
    >>> idx2d = kkpy.util.nn_idx_2d(128.35, 37.65, x2d, y2d)
    >>> print(x2d[idx2d], y2d[idx2d])
    
    Parameters
    ----------
    xtarget : float
        Target value of x.
    ytarget : float
        Target value of y.
    x2d : 2D array
        Numpy 2D array containing x value of each grid point. The shape of **x2d** and **y2d** should be same.
    y2d : 2D array
        Numpy 2D array containing y value of each grid point. The shape of **x2d** and **y2d** should be same.
    
    Returns
    ---------
    idx2d : 1D array
        Return an array of nearest x and y indicies.
    """
    import bottleneck as bn
    diff = (x2d - xtarget)**2 + (y2d - ytarget)**2
    idx2d = np.float_(np.unravel_index(bn.nanargmin(diff), x2d.shape))
    return np.int_(idx2d)

def cross_section_2d(xy0, xy1, x2d, y2d, value2d, linewidth=1, along='x', reduce_func=np.mean):
    """
    Get values along the transect of two points (lon/lat) for 2D array.
    
    Examples
    ---------
    >>> dem, lon, lat, proj = kkpy.io.read_dem(area='pyeongchang')
    >>> start = [127.52, 37.3]
    >>> end = [129, 36.52]
    >>> xaxis, vcross = kkpy.util.cross_section_2d(start, end, lon, lat, dem)
    >>> plt.plot(xaxis, vcross)
    >>> plt.show()
    
    Parameters
    ----------
    xy0 : array_like
        Array of x and y values of starting point.
    xy1 : array_like
        Array of x and y values of ending point.
    x2d : 2D array
        Numpy 2D array containing x value of each grid point. The shape of **x2d**, **y2d**, and **value2d** should be same.
    y2d : 2D array
        Numpy 2D array containing y value of each grid point. The shape of **x2d**, **y2d**, and **value2d** should be same.
    value2d : 2D array
        Numpy 2D array containing data value of each grid point. The shape of **x2d**, **y2d**, and **value2d** should be same.
    linewidth : int, optional
        The linewidth used in average over perpendicular direction along the transect. Default is 1 (i.e. no average).
    along : str, optional
        Set 'x' to return **xaxis** in x axis, otherwise return it in y axis. Default is 'x'.
    reduce_func : callable, optional
        Function used to calculate the aggregation of pixel values perpendicular to the profile_line direction when linewidth > 1. See skimage.measure.profile_line for detail.
    
    Returns
    ---------
    xaxis : 1D array
        Return xaxis of cross-section in longitude or latitude unit. The unit is determined by along keyword.
    vcross : 1D array
        Return averaged value along the cross-section.
    """
    import skimage.measure
    minx, maxx = np.nanmin(x2d), np.nanmax(x2d)
    miny, maxy = np.nanmin(y2d), np.nanmax(y2d)
    
    if not minx < xy0[0] < maxx:
        raise ValueError(f'xy0[0]({xy0[0]}) is out of range ({minx} ~ {maxx})')
    if not miny < xy0[1] < maxy:
        raise ValueError(f'xy0[1]({xy0[1]}) is out of range ({miny} ~ {maxy})')
    if not minx < xy1[0] < maxx:
        raise ValueError(f'xy1[0]({xy1[0]}) is out of range ({minx} ~ {maxx})')
    if not miny < xy1[1] < maxy:
        raise ValueError(f'xy1[1]({xy1[1]}) is out of range ({miny} ~ {maxy})')
    assert along in ['x', 'y'], "Check along keyword (should be 'x' or 'y')"
    
    # find indices
    istart = nn_idx_2d(xy0[0], xy0[1], x2d, y2d)
    iend = nn_idx_2d(xy1[0], xy1[1], x2d, y2d)
    
    # get xaxis
    ixlen, iylen = iend - istart
    ixsize = np.int(np.ceil(np.hypot(ixlen, iylen) + 1))
    if along in 'x':
        xaxis = np.linspace(xy0[0], xy1[0], ixsize)
    else:
        xaxis = np.linspace(xy0[1], xy1[1], ixsize)
        
    
    # get cross-section
    vcross = skimage.measure.profile_line(
        value2d,
        istart, iend,
        linewidth=linewidth,
        reduce_func=reduce_func,
        order=0,
        mode='constant',
        cval=np.nan
    )
    
    return xaxis, vcross

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

def dbzmean(dbz_arr, outside_radar=-9999., noprecip=-9998., qced=-9997.):
    """
    Get linear-scale average of reflectivity (dBZ).
    
    Examples
    ---------
    >>> dbz_avg = kkpy.util.dbzmean(dbz_arr)
    
    Parameters
    ----------
    X : array_like
        Array containing the reflectivity in dBZ.
        
    Returns
    ---------
    dbz_avg : array_like
        Return averaged reflectivity in dBZ.
    
    """

    dbz_arr[dbz_arr == outside_radar] = np.nan
    dbz_arr[dbz_arr == qced] = np.nan
    
    dbz_lin = 10.**(dbz_arr/10.)
    dbz_lin[dbz_arr == noprecip] = 0.
    
    return 10*np.log10(np.nanmean(dbz_lin))