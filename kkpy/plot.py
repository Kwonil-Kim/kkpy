"""
kkpy.plot
========================

Functions to read and write files

.. currentmodule:: plot

.. autosummary::
    kkpy.plot.koreamap
    kkpy.plot.icepop_sites
    kkpy.plot.cartopy_grid
    kkpy.plot.tickint

"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def koreamap(ax=None, edgecolor='k', linewidth1=1, linewidth2=0.3, city=True):
    """
    Draw boundaries of province and city of South Korea.
    
    Examples
    ---------
    >>> import cartopy.crs as ccrs
    >>> proj = kkpy.util.proj_dfs()
    >>> fig = plt.figure(figsize=(5,5), dpi=300)
    >>> ax = plt.subplot(projection=proj)
    >>> ax.set_extent([124, 130, 33, 43], crs=ccrs.PlateCarree())
    >>> kkpy.plot.koreamap(ax=ax)
    >>> plt.show()
    
    Parameters
    ----------
    ax : axes
        Axes class of matplotlib.
    proj : projection object
        Map projection of matplotlib axes.
    edgecolor : str, optional
        Edgecolor of the plot. Default is 'k' (black).
    linewidth1 : float, optional
        Linewidth of province. Default is 1.
    linewidth1 : float, optional
        Linewidth of city. Default is 1.
    city : boolean, optional
        Draw city if True. Draw province only if False.
    """
    import geopandas as gpd
    from pathlib import Path
    
    shpdir = f'{Path(__file__).parent}/SHP'

    kor1 = gpd.read_file(f'{shpdir}/gadm36_KOR_1.shp')
    kor2 = gpd.read_file(f'{shpdir}/gadm36_KOR_2.shp')
    
    kor1.to_crs(ax.projection.proj4_init).plot(color=(1,1,1,0), edgecolor=edgecolor, ax=ax, zorder=2, linewidth=linewidth1)
    if city:
        kor2.to_crs(ax.projection.proj4_init).plot(color=(1,1,1,0), edgecolor=edgecolor, ax=ax, zorder=2, linewidth=linewidth2)
    
    return

def icepop_sites(ax=None,
                 xz=False, yz=False,
                 marker='o', color='red',
                 markersize=3, alpha=0.5,
                 fontsize=10, textmargin=0.01,
                 textcolor='k',
                 verticalalignment='center',
                 transform=ccrs.PlateCarree(),
                 zunit='km',
                 include_site=['GWU', 'BKC', 'CPO', 'DGW', 'MHS',
                               'YPO', 'SCW', 'YYO', 'YDO', 'JMO',
                               'OGO', 'DHW', 'MOO', 'PCO'],
                 include_text=['GWU', 'BKC', 'CPO', 'DGW', 'MHS',
                               'YPO', 'SCW', 'YYO', 'YDO', 'JMO',
                               'OGO', 'DHW', 'MOO', 'PCO'],
                 exclude_site=[],
                 exclude_text=[]):
    """
    Draw supersites of ICE-POP 2018.
    
    Examples
    ---------
    >>> import cartopy.crs as ccrs
    >>> fig = plt.figure(figsize=(5,5), dpi=300)
    >>> ax = plt.subplot(projection=kkpy.util.proj_dfs())
    >>> ax.set_extent(kkpy.util.icepop_extent(), crs=ccrs.PlateCarree())
    >>> kkpy.plot.icepop_sites(ax=ax)
    >>> kkpy.plot.koreamap(ax=ax)
    >>> plt.show()
    
    Parameters
    ----------
    ax : axes
        Axes class of matplotlib.
    xz : boolean, optional
        True if plot in longitude-height coordinate.
    yz : boolean, optional
        True if plot in latitude-height coordinate.
    marker : str, optional
        Matplotlib marker. Default is 'o'.
    color : str, optional
        Matplotlib color for marker. Default is 'red'.
    markersize : float, optional
        Matplotlib markersize. Default is 3.
    alpha : float, optional
        Matplotlib alpha for marker. Default is 0.5.
    fontsize : float, optional
        Matplotlib fontsize for ax.text. Default is 10.
    textmargin : float, optional
        Longitude or latitude margin in decimal degree. Default is 0.1 degree.
    textcolor : str, optional
        Matplotlib color for text. Default is 'k'.
    verticalalignment : str, optional
        Matplotlib verticalalignment for ax.text. Default is 'center'.
    transform : cartopy crs, optional
        Cartopy crs. Default is ccrs.PlateCarree().
    zunit : str, optional
        Unit of vertical coordinate if xz or yz is True. Default is 'km'. Possible choices are 'km' or 'm'.
    """
    from . import util
    
    if not xz and not yz:
        for lon, lat, hgt, site in util.icepop_sites():
            if site in include_site and not site in exclude_site:
                ax.plot(lon, lat, marker=marker, color=color, markersize=markersize, alpha=alpha, transform=transform)
                if site in include_text and not site in exclude_text:
                    ax.text(lon+textmargin, lat, site, verticalalignment=verticalalignment, fontsize=fontsize, color=textcolor, transform=transform)
    
    if xz:
        for lon, lat, hgt, site in util.icepop_sites():
            if zunit in 'km':
                hgt = hgt/1e3
            ax.plot(lon, hgt, marker=marker, color=color, markersize=markersize, alpha=alpha)
            ax.text(lon+textmargin, hgt, site, verticalalignment=verticalalignment, fontsize=fontsize, color=textcolor)
    
    if yz:
        for lon, lat, hgt, site in util.icepop_sites():
            if zunit in 'km':
                hgt = hgt/1e3
            ax.plot(lat, hgt, marker=marker, color=color, markersize=markersize, alpha=alpha)
            ax.text(lat+textmargin, hgt, site, verticalalignment=verticalalignment, fontsize=fontsize, color=textcolor)
    
    return

def cartopy_grid(ax=None,
                 draw_labels=True,
                 dms=False,
                 x_inline=False, y_inline=False,
                 alpha=0.5, linestyle='dashed', 
                 ticks_lon=np.arange(128,129.5,0.2),
                 ticks_lat=np.arange(37,38.5,0.2),
                 **kwargs):
    """
    Draw gridlines in cartopy map.
    
    Examples
    ---------
    >>> import cartopy.crs as ccrs
    >>> fig = plt.figure(figsize=(5,5), dpi=300)
    >>> ax = plt.subplot(projection=kkpy.util.proj_dfs())
    >>> dem, lon, lat, _ = kkpy.io.read_dem(area='pyeongchang')
    >>> pm = ax.pcolormesh(lon, lat, dem, transform=ccrs.PlateCarree(), vmin=0, cmap=plt.cm.terrain)
    >>> ax.set_extent(kkpy.util.icepop_extent(), crs=ccrs.PlateCarree())
    >>> kkpy.plot.cartopy_grid(ax=ax)
    >>> kkpy.plot.icepop_sites(ax=ax)
    >>> kkpy.plot.koreamap(ax=ax)
    >>> plt.show()
        
    Parameters
    ----------
    ax : axes
        Axes class of matplotlib.
    draw_labels : boolean, optional
        Matplotlib draw_labels. Default is True.
    dms : boolean, optional
        Matplotlib dms. Default is False.
    x_inline : boolean, optional
        Matplotlib x_inline. Default is False.
    y_inline : boolean, optional
        Matplotlib y_inline. Default is False.
    alpha : float, optional
        Matplotlib alpha. Default is 0.5.
    linestyle : str, optional
        Matplotlib linestyle. Default is 'dashed'.
    ticks_lon : array_like, optional
        Grid tick locations of longitude in decimal degree. Default is np.arange(128,129.5,0.2).
    ticks_lon : array_like, optional
        Grid tick locations of latitude in decimal degree. Default is np.arange(37,38.5,0.2).
    \**kwargs : \**kwargs, optional
        \**kwargs for `ax.gridlines`
    """
    import matplotlib.ticker as mticker
    
    gl = ax.gridlines(draw_labels=draw_labels, dms=dms, x_inline=x_inline, y_inline=y_inline, alpha=alpha, linestyle=linestyle, **kwargs)
    gl.xlocator = mticker.FixedLocator(ticks_lon)
    gl.ylocator = mticker.FixedLocator(ticks_lat)
    gl.rotate_labels = False
    gl.top_labels = gl.right_labels = False
    
    return

def tickint(ax=None, major=None, minor=None, which='both'):
    """
    Set interval of major or minor ticks of axis intuitively.
    
    Examples
    ---------
    >>> ax = plt.subplot()
    >>> ax.plot(np.arange(30))
    >>> kkpy.plot.tickint(ax=ax, major=10, minor=5)
    >>> plt.show()

    >>> # xaxis only
    >>> ax = plt.subplot()
    >>> ax.plot(np.arange(30))
    >>> kkpy.plot.tickint(ax=ax, major=10, minor=5, which='xaxis')
    >>> plt.show()

    >>> # yaxis only
    >>> ax = plt.subplot()
    >>> ax.plot(np.arange(30))
    >>> kkpy.plot.tickint(ax=ax, major=10, which='yaxis')
    >>> plt.show()

    >>> # set two axes with different tick options
    >>> ax = plt.subplot()
    >>> ax.plot(np.arange(30))
    >>> kkpy.plot.tickint(ax=ax, major=10, minor=5, which='xaxis')
    >>> kkpy.plot.tickint(ax=ax, major=5, minor=1, which='yaxis')
    >>> plt.show()

    Parameters
    ----------
    ax : axes
        Axes class of matplotlib.
    major : float, optional
        Major tick interval.
    minor : float, optional
        Major tick interval.
    which : str, optional
        The axis to apply the changes on. Possible options are 'both', 'xaxis', and 'yaxis'. Default is 'both'.
    """
    
    xaxis = yaxis = False
    if which in 'both':
        xaxis = True
        yaxis = True
    elif which in 'xaxis':
        xaxis = True
    elif which in 'yaxis':
        yaxis = True
    else:
        raise ValueError('Possible which options: both, xaxis, and yaxis')
    
    if xaxis:
        if not major is None:
            ax.xaxis.set_major_locator(plt.MultipleLocator(major))
        if not minor is None:
            ax.xaxis.set_minor_locator(plt.MultipleLocator(minor))
    if yaxis:
        if not major is None:
            ax.yaxis.set_major_locator(plt.MultipleLocator(major))
        if not minor is None:
            ax.yaxis.set_minor_locator(plt.MultipleLocator(minor))
    
    return

def scatter(x, y, color='red', )


ax = plt.subplot(gs[i_v])
    ax.scatter(xvalue, yvalue, color='r', s=0.5)
    ax.set_xlabel('10-min RR (PAR02) [mm h$^{-1}$]')
    if i_v == 0:
        ax.set_ylabel('10-min RR (PAR03) [mm h$^{-1}$]')
    ax.set_xlim([0,55])
    ax.set_ylim([0,55])
    ax.plot([0,55], [0,55], color='k', linestyle='dashed', alpha=0.5, linewidth=0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.set_aspect('equal')
    if i_v == 0:
        plt.title('Before the time adjustment')
    if i_v == 1:
        plt.title('After the time adjustment')
    
    bias = np.nanmean(yvalue - xvalue)
    rmse = ((yvalue - xvalue) ** 2).mean() ** 0.5
    std = (yvalue - xvalue).std()
    corr = xvalue.corr(yvalue)
    plt.text(27, 5, f'BIAS={bias:.3f}\nRMSE={rmse:.3f}\nSTD={std:.3f}\nCORR={corr:.3f}')
    
    plt.grid()