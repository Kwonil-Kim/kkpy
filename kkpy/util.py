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

Microphysics
---------------
.. autosummary::
    kkpy.util.vel_atlas
    kkpy.util.calc_dsdmoments

Maps
-------
.. autosummary::
    kkpy.util.proj_dfs
    kkpy.util.proj_icepop
    kkpy.util.dist_bearing

Spatial calculations
----------------------
.. autosummary::
    kkpy.util.nn_idx_2d
    kkpy.util.cross_section_2d
    kkpy.util.to_lower_resolution
    
ICE-POP 2018
--------------
.. autosummary::
    kkpy.util.icepop_events
    kkpy.util.icepop_sites
    kkpy.util.icepop_extent

Miscellaneous
---------------
.. autosummary::
    kkpy.util.std2d
    kkpy.util.nanstd2d
    kkpy.util.nanconvolve2d
    kkpy.util.derivative
    kkpy.util.stats
    kkpy.util.summary
    kkpy.util.get_intersect
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
    radians : boolean, optional
        If this is set to True, the unit of *wd* is **radian**. The default is False (i.e. **degree**).
    knots : boolean, optional
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
        ws = knot2ms(ws)
    
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
    ixsize = int(np.ceil(np.hypot(ixlen, iylen) + 1))
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
    >>> import cartopy.crs as ccrs
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

def proj_icepop():
    """
    Return a lambert conformal conic projection for ICE-POP 2018 domain.
    
    Examples
    ---------
    >>> import cartopy.crs as ccrs
    >>> ax = plt.subplot(proj=kkpy.util.proj_icepop())
    >>> ax.scatter([129], [37.7], transform=ccrs.PlateCarree())
    >>> ax.set_extent(kkpy.util.icepop_extent(), crs=ccrs.PlateCarree())
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
    
    proj = ccrs.LambertConformal(central_longitude=128.75,
                                 central_latitude=37.65,
                                 standard_parallels=(30,60),
                                 false_easting=75000,
                                 false_northing=75000,
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
    radians : boolean, optional
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

def dbzmean(dbz_arr, outside_radar=-9999., noprecip=-9998., qced=-9997., axis=None):
    """
    Get linear-scale average of reflectivity (dBZ).
    
    Examples
    ---------
    >>> dbz_avg = kkpy.util.dbzmean(dbz_arr)
    
    Parameters
    ----------
    X : array_like
        Array containing the reflectivity in dBZ.
    outside_radar : float, optional
        Value indicating the grid point is located outside radar observable range. This will not be included in the count during the average.
    noprecip : float, optional
        Value indicating the grid point is clear echo. This will be considered as 0.0 mm6 m-3.
    qced : float, optional
        Value indicating the grid point is masked by QC process. This will not be included in the count during the average.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed.
        
        .. versionadded:: 0.2.0
        
    Returns
    ---------
    dbz_avg : array_like
        Return averaged reflectivity in dBZ.
    
    """

    dbz_arr[dbz_arr == outside_radar] = np.nan
    dbz_arr[dbz_arr == qced] = np.nan
    
    dbz_lin = 10.**(dbz_arr/10.)
    dbz_lin[dbz_arr == noprecip] = 0.
    
    return 10*np.log10(np.nanmean(dbz_lin, axis=axis))


def icepop_events(KST=False):
    """
    Get datetimes for nine major evnets in ICE-POP 2018.
    
    Examples
    ---------
    >>> dts = kkpy.util.icepop_events()
    >>> for eventno in np.arange(9)+1:
    >>>     dt = dts[eventno]
    >>>     print(f'eventno = {eventno}, start = {dt[0]}, end = {dt[1]}')
    
    Parameters
    ----------
    KST : boolean, optional
        True if return the datetime in KST, rather than UTC.
        
    Returns
    ---------
    dict_datetimes : dictionary
        Return a dictionary of array containing start and finish datetime for each event.
    
    """
    import datetime
    
    if not KST:
        events = {
            1: [datetime.datetime(2017,11,24,20), datetime.datetime(2017,11,25,19)],
            2: [datetime.datetime(2017,12,23,20), datetime.datetime(2017,12,25, 3)],
            3: [datetime.datetime(2018, 1,22, 3), datetime.datetime(2018, 1,23, 0)],
            4: [datetime.datetime(2018, 2,27,23), datetime.datetime(2018, 3, 1, 3)],
            5: [datetime.datetime(2018, 3, 4, 8), datetime.datetime(2018, 3, 5,10)],
            6: [datetime.datetime(2018, 3, 7, 5), datetime.datetime(2018, 3, 9, 3)],
            7: [datetime.datetime(2018, 3,14,19), datetime.datetime(2018, 3,16,12)],
            8: [datetime.datetime(2018, 3,18,10), datetime.datetime(2018, 3,19, 9)],
            9: [datetime.datetime(2018, 3,19,21), datetime.datetime(2018, 3,21,14)],
        }
    else:
        events = {
            1: [datetime.datetime(2017,11,25, 5), datetime.datetime(2017,11,26, 4)],
            2: [datetime.datetime(2017,12,24, 5), datetime.datetime(2017,12,25,12)],
            3: [datetime.datetime(2018, 1,22,12), datetime.datetime(2018, 1,23, 9)],
            4: [datetime.datetime(2018, 2,28, 8), datetime.datetime(2018, 3, 1,12)],
            5: [datetime.datetime(2018, 3, 4,17), datetime.datetime(2018, 3, 5,19)],
            6: [datetime.datetime(2018, 3, 7,14), datetime.datetime(2018, 3, 9,12)],
            7: [datetime.datetime(2018, 3,15, 4), datetime.datetime(2018, 3,16,21)],
            8: [datetime.datetime(2018, 3,18,19), datetime.datetime(2018, 3,19,18)],
            9: [datetime.datetime(2018, 3,20, 6), datetime.datetime(2018, 3,21,23)],
        }
    
    return events

def icepop_sites():
    """
    Get metadata of ICE-POP 2018 sites.
    
    Examples
    ---------
    >>> # Plotting in Lon/Lat coordinates
    >>> for lon, lat, hgt, site in kkpy.util.icepop_sites():
    >>>     plt.plot(lon, lat, marker='o', color='red', markersize=3, alpha=0.5, transform=ccrs.PlateCarree())
    >>>     plt.text(lon+0.01, lat, site, verticalalignment='center', fontsize=10, transform=ccrs.PlateCarree())
    
    >>> # Plotting in Lon/Hgt coordinates
    >>> for lon, lat, hgt, site in kkpy.util.icepop_sites():
    >>>     plt.plot(lon, hgt/1e3, marker='o', color='red', markersize=3, alpha=0.5)
    >>>     plt.text(lon+0.01, hgt/1e3, site, verticalalignment='center', fontsize=10)
    
    Returns
    ---------
    tuple_sites : tuple
        Return a tuple containing longitude, latitude, height, and sitename for each ICE-POP 2018 supersite.
    
    """
    
    sites = (
        (128.866858, 37.770897,  36, 'GWU'),
        (128.805847, 37.738157, 175, 'BKC'),
        (128.758636, 37.686953, 855, 'CPO'),
        (128.718825, 37.677331, 773, 'DGW'),
        (128.699611, 37.665208, 789, 'MHS'),
        (128.670494, 37.643342, 772, 'YPO'),
        (128.564700, 38.250900,  18, 'SCW'),
        (128.629700, 38.087200,   4, 'YYO'),
        (128.540700, 38.007400, 146, 'YDO'),
        (128.821100, 37.898300,  10, 'JMO'),
        (129.028900, 37.613500,  58, 'OGO'),
        (129.124300, 37.507100,  40, 'DHW'),
        (128.377600, 37.562000, 532, 'MOO'),
        (128.394600, 37.377900, 303, 'PCO')
    )
    
    return sites

def icepop_extent():
    """
    Get extent of ICE-POP 2018 domain.
    
    Examples
    ---------
    >>> ax.set_extent(kkpy.util.icepop_extent(), crs=ccrs.PlateCarree())
    
    Returns
    ---------
    list_extent : list
        Return a list containing lon0, lon1, lat0, and lat1 of default ICE-POP 2018 domain.
    
    """
    return [128, 129.5, 37, 38.3]

def vel_atlas(D):
    """
    Get Atlas et al. (1973) fall velocity.
    
    Examples
    ---------
    >>> V_arr = kkpy.util.vel_atlas(D_arr)
    
    Parameters
    ----------
    D : array_like
        Array containing the diameter in mm.
        
    Returns
    ---------
    V : array_like
        Return a fall velocity of raindrop for a given diameter.
    """
    return 9.65 - 10.3 * np.exp(-0.6 * D)

def calc_dsdmoments(ND, D, dD, VD=None, Vatlas=False):
    """
    Get DSD moments from drop size distribution.
    
    Examples
    ---------
    >>> tdvd = kkpy.util.calc_dsdmoments(ND_arr, D_arr, dD_arr, VD=VD_arr)
    
    >>> poss = kkpy.util.calc_dsdmoments(ND_arr, D_arr, dD_arr, Vatlas=True)
    
    Parameters
    ----------
    ND : array_like
        Array containing the number concentration in mm-1 m-3. If 2D array, the diameter should be at the first axis (i.e. axis = 0).
    D : array_like
        Array containing the median of each diameter channel in mm.
    dD : array_like
        Array containing the spread of each diameter channel in mm.
    VD : array_like, optional
        Array containing the mean velocity of each diameter channel in m s-1. Vatlas should be False if VD contains data.
    Vatlas : boolean, optional
        True if VD is replaced by the velocity-diameter relationship of Atlas et al. (1973).
        
    Returns
    ---------
    dict_moments : dictionary
        Return a dictionary containing the DSD moments: z[mm6 m-3], Z[dBZ], and R[mm hr-1].
    """
    
    if Vatlas:
        VD = vel_atlas(D)
    
    dict_ = {}
    if ND.ndim == 2:
        dict_['z'] = np.sum(ND*(D**6)*dD, axis=1) # mm6 m-3
        dict_['R'] = np.pi*6*1e-4*np.sum(ND*VD*(D**3)*dD, axis=1) # mm hr-1
    else:
        dict_['z'] = np.sum(ND*(D**6)*dD) # mm6 m-3
        dict_['R'] = np.pi*6*1e-4*np.sum(ND*VD*(D**3)*dD) # mm hr-1
    
    dict_['Z'] = 10*np.log10(dict_['z']) # dBZ
    
    return dict_

def stats(x, y,
          fmtbias='.3f', fmtrmse='.3f',
          fmtstd='.3f', fmtcorr='.3f'):
    """
    Get performance matrix BIAS, RMSE, STD, and CORR values and its f-string.
    
    Examples
    ---------
    >>> stats, str_stat = kkpy.util.stats(xarr, yarr)
    >>> print(stats)
    >>> plt.text(3, 5, str_stat)
    
    >>> stats, str_stat = kkpy.util.stats(xarr, yarr, fmtbias='.1f', fmtcorr='.2f')
    
    Parameters
    ----------
    x : array_like
        Array containing multiple variables and observations.
    y : array_like
        Array containing multiple variables and observations. The shape should be same as **x**.
    fmtbias : str, optional
        String format for BIAS. Default is '.3f'.
    fmtrmse : str, optional
        String format for RMSE. Default is '.3f'.
    fmtstd : str, optional
        String format for STD. Default is '.3f'.
    fmtcorr : str, optional
        String format for CORR. Default is '.3f'.

    Returns
    ---------
    dict_moments : dictionary
        Return a dictionary containing the DSD moments: z[mm6 m-3], Z[dBZ], and R[mm hr-1].
    """
    import pandas as pd
    
    bias = np.nanmean(y - x)
    rmse = np.nanmean((y - x) ** 2)**0.5
    std = np.nanstd(y - x)
    corr = pd.Series(x).corr(pd.Series(y))
    
    stats = [bias, rmse, std, corr]
    str_stat = f'BIAS={bias:.3f}\nRMSE={rmse:.3f}\nSTD={std:.3f}\nCORR={corr:.3f}'
    
    return stats, str_stat

def summary(arr, cnt_print=10):
    """
    Get summary of the array.
    
    Parameters
    ----------
    arr : array_like
        Array containing multiple variables and observations.
    cnt_print : int, optional
        The number of elements to print.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    print(pd.Series(arr).describe())
    cnt = arr.size
    count_nan = np.count_nonzero(np.isnan(arr))
    print(f'\nNans   {count_nan} ({count_nan/arr.size:.1f} %)\n')
    
    if cnt > cnt_print:
        print(f'First {cnt_print} values:\n {arr[:cnt_print]}\n')
        print(f'Last {cnt_print} values:\n {arr[-cnt_print:]}\n')
        
        arr = arr[~np.isnan(arr)]
        arr_sorted = np.sort(arr)
        print(f'Lowest {cnt_print} values:\n {arr_sorted[:cnt_print]}\n')
        print(f'Highest {cnt_print} values:\n {arr_sorted[-cnt_print:]}\n')
        arr_unique = np.unique(arr)
        print(f'Lowest {cnt_print} unique values:\n {arr_unique[:cnt_print]}\n')
        print(f'Highest {cnt_print} unique values:\n {arr_unique[-cnt_print:]}\n')
        
        hist = plt.hist(arr)
        print('Histograms')
        for y, x in zip(hist[0], hist[1]):
            print(f'{x:.3f} -- {y:.0f}')
        
    return

def derivative(var, cnt_x, pixelsize=1, axis=0):
    """
    Get derivative of 1D or 2D array.
    
    Examples
    ---------
    >>> # Derivative of 1D array with a window size of 5.
    >>> # Note that the horizontal resolution of xarr is 0.1.
    >>> xarr = np.arange(100)*0.1
    >>> yarr = np.exp(xarr)
    >>> dydx = kkpy.util.derivative(yarr, 5, pixelsize=0.1)
    
    >>> # This example gives the derivative of arr2d along x axis with a window size of 20.
    >>> # Assume that you have an arr2d[iy,ix] with horizontal resolution of 3 km.
    >>> dzdx = kkpy.util.derivative(arr2d, 11, pixelsize=3000, axis=1)
    
    Parameters
    ----------
    var : array_like
        Array containing multiple variables and observations.
    cnt_x : int
        The number of the pixel for derivative. This should be odd number. The derivative will be (Y[i+N/2]-Y[i-N/2]) / (N-1).
    pixelsize : float, optional
        The size of one pixel of x axis. If default (1), the derivative means the rate of change on an interval of **one pixel** with resolution of 1.
    axis : int, optional
        Axis along which the derivatives are computed. Default is 0.
        
    Returns
    ---------
    derivative : array_like
        Return a derivative for a given array.
    """
    import astropy.convolution

    if cnt_x % 2 != 1:
        raise ValueError('cnt_x must be odd.')
    
    if var.ndim > 2:
        raise ValueError('dimension size of var should be <= 2')

    kernel = np.zeros(cnt_x)
    kernel[0] = 1
    kernel[-1] = -1
    
    if var.ndim == 2:
        if axis == 0:
            kernel = kernel[:,np.newaxis]
        if axis == 1:
            kernel = kernel[np.newaxis,:]
    
    result = astropy.convolution.convolve(
            var,
            kernel,
            boundary='extend',
            normalize_kernel=False,
            nan_treatment='fill',
            fill_value=np.nan,
            preserve_nan=True
        ) / (cnt_x-1)
    
    if var.ndim == 1:
        for j in np.arange(int(cnt_x/2)+1):
            result[j] = (var[j+int(cnt_x/2)]-var[0]) / (j+int(cnt_x/2))
        for j in np.arange(var.size-1-int(cnt_x/2), var.size):
            result[j] = (var[-1]-var[j-int(cnt_x/2)]) / (var.size-j+int(cnt_x/2)-1)
    
    if var.ndim == 2:
        if axis == 0:
            for j in np.arange(int(cnt_x/2)+1):
                result[j,:] = (var[j+int(cnt_x/2),:]-var[0,:]) / (j+int(cnt_x/2))
            for j in np.arange(var.shape[0]-1-int(cnt_x/2), var.shape[0]):
                result[j,:] = (var[-1,:]-var[j-int(cnt_x/2),:]) / (var.shape[0]-j+int(cnt_x/2)-1)
        if axis == 1:
            for j in np.arange(int(cnt_x/2)+1):
                result[:,j] = (var[:,j+int(cnt_x/2)]-var[:,0]) / (j+int(cnt_x/2))
            for j in np.arange(var.shape[1]-1-int(cnt_x/2), var.shape[1]):
                result[:,j] = (var[:,-1]-var[:,j-int(cnt_x/2)]) / (var.shape[1]-j+int(cnt_x/2)-1)
    
    derivative = result / pixelsize
    
    return derivative

def to_lower_resolution(arr_in, ratio_x, ratio_y, dBZ=False):
    """
    Get 2D array with lower resolution.
    
    Examples
    ---------
    >>> arr1 = np.random.rand(250,150) # shape: (250,150)
    >>> arr2 = to_lower_resolution(arr1,10,10) # shape: (25,15)
    >>> arr3 = to_lower_resolution(arr1,5,10) # shape: (50,15)
    
    Parameters
    ----------
    arr_in : 2D array
        Array you want to lower the resolution.
    ratio_x : int
        The reduction ratio over x axis. The size of **arr_in** xaxis should be able to be divided by **ratio_x**.
    ratio_y : int
        The reduction ratio over y axis. The size of **arr_in** yaxis should be able to be divided by **ratio_y**.
    dBZ : boolean, optional
        True if return a linear-scale average of reflectivity (dBZ).
        
    Returns
    ---------
    arr_out : 2D array
        Return an array with lower resolution.
    
    Notes
    ---------
    This code is from Stack Overflow answer (https://stackoverflow.com/a/14916963/12272819), written by Jaime (https://stackoverflow.com/users/110026/jaime).
    This is licensed under the Creative Commons Attribution-ShareAlike 3.0 license (CC BY-SA 3.0).
    """
    
    if arr_in.ndim != 2:
        raise ValueError("arr_in must be a 2D array")
    if ratio_x <= 1 or ratio_y <= 1:
        raise ValueError("ratio_x and ratio_y must be greater than 1")
    if not isinstance(ratio_x, int) or not isinstance(ratio_y, int):
        raise ValueError("ratio_x and ratio_y must be an integer")
    
    nx_in, ny_in = arr_in.shape
    nx_out, ny_out = nx_in/ratio_x, ny_in/ratio_y
    
    if not nx_out.is_integer():
        raise ValueError(f"check ratio_x: nx_out={nx_out}")
    if not ny_out.is_integer():
        raise ValueError(f"check ratio_y: ny_out={ny_out}")
    
    arr_out = arr_in.reshape(
        [int(nx_out), ratio_x, int(ny_out), ratio_y])
    if not dBZ:
        arr_out = arr_out.mean(axis=3).mean(axis=1)
    else:
        arr_out = dbzmean(arr_out, axis=3)
        arr_out = dbzmean(arr_out, axis=1)
    
    return arr_out

def get_intersect(list_array):
    """
    Get intersection of multiple 1D arrays.
    
    Examples
    ---------
    >>> arr1 = np.array([0,1,2,4,5,6])
    >>> arr2 = np.array([0,1,3,4,6])
    >>> arr_inter = kkpy.util.get_intersect([arr1, arr2]) # [0,1,4,6]
    
    Parameters
    ----------
    list_array : array_like
        List of 1D numpy arrays.
        
    Returns
    ---------
    intersection : array_like
        Return an array with lower resolution.
    
    Notes
    ---------
    This code is based on Stack Exchange answer (https://codereview.stackexchange.com/a/145210), written by Peilonrayz (https://codereview.stackexchange.com/users/42401/peilonrayz).
    This is licensed under the Creative Commons Attribution-ShareAlike 3.0 license (CC BY-SA 3.0).
    """
    from functools import reduce
    intersection = reduce(np.intersect1d, list_array)
    return intersection
