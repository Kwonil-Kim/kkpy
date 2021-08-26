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
    kkpy.plot.scatter
    kkpy.plot.density2d

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
    include_site : list, optional
        List of sites you want to include in site points.
    include_text : list, optional
        List of sites you want to include in site annotations.
    exclude_site : list, optional
        List of sites you want to exclude in site points.
    exclude_text : list, optional
        List of sites you want to exclude in site annotations.
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
            if zunit in ['km']:
                hgt = hgt/1e3
            elif zunit in ['m']:
                pass
            else:
                raise ValueError("Possible zunit options: 'km' or 'm'")
            if site in include_site and not site in exclude_site:
                ax.plot(lon, hgt, marker=marker, color=color, markersize=markersize, alpha=alpha)
                if site in include_text and not site in exclude_text:
                    ax.text(lon+textmargin, hgt, site, verticalalignment=verticalalignment, fontsize=fontsize, color=textcolor)
    
    if yz:
        for lon, lat, hgt, site in util.icepop_sites():
            if zunit in ['km']:
                hgt = hgt/1e3
            elif zunit in ['m']:
                pass
            else:
                raise ValueError("Possible zunit options: 'km' or 'm'")
            if site in include_site and not site in exclude_site:
                ax.plot(lat, hgt, marker=marker, color=color, markersize=markersize, alpha=alpha)
                if site in include_text and not site in exclude_text:
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

def scatter(x, y,
            ax=None,
            color='r', s=0.5,
            xlabel=None, ylabel=None,
            xlim=None, ylim=None,
            alpha=None,
            identityline=True,
            identityline_color='k',
            identityline_linestyle='dashed',
            identityline_alpha=0.5,
            identityline_linewidth=0.5,
            xmajortickint=None,
            xminortickint=None,
            ymajortickint=None,
            yminortickint=None,
            aspect_equal=False,
            title=None,
            score=True,
            score_loc='lower right',
            score_fontsize=10,
            fmtbias='.3f',
            fmtrmse='.3f',
            fmtstd='.3f',
            fmtcorr='.3f',
            grid=True,
            grid_color='#b0b0b0',
            grid_linestyle='-',
            grid_alpha=None,
            grid_linewidth=0.8,
            grid_zorder=2.0,
            grid_which='both'):
    """
    Draw scatter plot.
    
    Examples
    ---------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> fig = plt.figure(figsize=(4,4), dpi=300)
    >>> ax = plt.subplot()
    >>> scores = kkpy.plot.scatter(x, y, ax=ax)
    >>> print(scores)
    >>> plt.show()

    >>> # without score, without identityline
    >>> kkpy.plot.scatter(x, y, ax=ax, score=False, identityline=False)
    
    >>> # location of score text #1 (text)
    >>> scores = kkpy.plot.scatter(x, y, ax=ax, score_loc='lower right')

    >>> # location of score text #2 (list of xpos, ypos)
    >>> scores = kkpy.plot.scatter(x, y, ax=ax, score_loc=[0.7, 0.05]) # near 'lower right'

    >>> # more complicated options
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)*3
    >>> fig = plt.figure(figsize=(4,4), dpi=300)
    >>> ax = plt.subplot()
    >>> scores = kkpy.plot.scatter(
            x, y, ax=ax, aspect_equal=True, score_loc='upper left',
            xlabel='X', ylabel='Y', xlim=[0,1], ylim=[0,3],
            title='TITLE', xmajortickint=0.2, xminortickint=0.1,
            ymajortickint=0.1, score_fontsize=12, fmtstd='.2f',
            alpha=0.5, s=2, color='g', identityline_color='b',
            identityline_linewidth=2, identityline_linestyle='solid'
        )
    >>> print(scores)
    >>> plt.show()
            
    Parameters
    ----------
    x : array_like
        Array containing multiple variables and observations.
    y : array_like
        Array containing multiple variables and observations. The shape should be same as **x**.
    ax : axes
        Axes class of matplotlib.
    color : str, optional
        Matplotlib color for scatter. Default is 'r'.
    s : float, optional
        Matplotlib size for scatter. Default is 0.5.
    xlabel : str, optional
        The label text of x axis.
    ylabel : str, optional
        The label text of y axis.
    xlim : list, optional
        The limits of x axis.
    ylim : list, optional
        The limits of y axis.
    alpha : float, optional
        Matplotlib alpha for scatter.
    identityline : boolean, optional
        True if draw identityline (one-to-one line). Default is True.
    identityline_color : str, optional
        Matplotlib color for identity line. Default is 'k'.
    identityline_linestyle : str, optional
        Matplotlib linestyle for identity line. Default is 'dashed'.
    identityline_alpha : float, optional
        Matplotlib alpha for identity line. Default is 0.5.
    identityline_linewidth : float, optional
        Matplotlib linewidth for identity line. Default is 0.5.
    xmajortickint : float, optional
        Major tick interval of x axis.
    xminortickint : float, optional
        Minor tick interval of x axis.
    ymajortickint : float, optional
        Major tick interval of y axis.
    yminortickint : float, optional
        Minor tick interval of y axis.
    aspect_equal : boolean, optional
        True if ax.set_aspect('equal'). Default is False.
    title : str, optional
        Title of the axis.
    score : boolean, optional
        True if annotate and return the evaluation score (bias, rmse, std, and corr)
    score_loc : str, optional
        Location of the evaluation score text in the plot. Possible options are list([xpos,ypos]), 'upper left', 'upper right', 'lower left', and 'lower right'. The xpos and ypos should be in the 'axes fraction' coordinate. Default is 'lower right'.
    score_fontsize : float, optional
        The fontsize of evaluation score text in the plot. Default is 10.
    fmtbias : str, optional
        String format for BIAS. Default is '.3f'.
    fmtrmse : str, optional
        String format for RMSE. Default is '.3f'.
    fmtstd : str, optional
        String format for STD. Default is '.3f'.
    fmtcorr : str, optional
        String format for CORR. Default is '.3f'.
    grid : boolean, optioinal
        True if draw grid.
        .. versionadded:: 0.3.4
    grid_color : str, optioinal
        Matplotlib color for grid.
        .. versionadded:: 0.3.4
    grid_linestyle : str, optioinal
        Matplotlib linestyle for grid.
        .. versionadded:: 0.3.4
    grid_alpha : str, optioinal
        Matplotlib alpha for grid.
        .. versionadded:: 0.3.4
    grid_linewidth : str, optioinal
        Matplotlib linewidth for grid.
        .. versionadded:: 0.3.4
    grid_zorder : str, optioinal
        Matplotlib zorder for grid.
        .. versionadded:: 0.3.4
    grid_which : str, optioinal
        The axis to draw the grid on. Possible options are 'both', 'xaxis', and 'yaxis'. Default is 'both'.
        .. versionadded:: 0.3.4
    
    Returns
    ---------
    score : list
        Return a score **if score is True**, otherwise no return.
    """
    from . import util
    
    # scatter
    ax.scatter(x, y, color=color, s=s, alpha=alpha)
    
    # xlabel, ylabel
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    # xlim, ylim
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # tick intervals
    if xmajortickint is not None:
        tickint(ax=ax, major=xmajortickint, which='xaxis')
    if xminortickint is not None:
        tickint(ax=ax, minor=xminortickint, which='xaxis')
    if ymajortickint is not None:
        tickint(ax=ax, major=ymajortickint, which='yaxis')
    if yminortickint is not None:
        tickint(ax=ax, minor=yminortickint, which='yaxis')
    
    # identityline
    if identityline:
        xlim = ax.get_xlim()
        ylim = ax.get_xlim()
        ax.plot(xlim, ylim,
                color=identityline_color,
                linestyle=identityline_linestyle,
                alpha=identityline_alpha,
                linewidth=identityline_linewidth)
    
    # aspect_equal
    if aspect_equal:
        ax.set_aspect('equal')
    
    # title
    if title is not None:
        ax.set_title(title)
    
    # score
    if score:
        scores, str_score = util.stats(x, y,
                                       fmtbias=fmtbias,
                                       fmtrmse=fmtrmse,
                                       fmtstd=fmtstd,
                                       fmtcorr=fmtcorr)
        if isinstance(score_loc, list):
            ax.annotate(str_score, xy=(score_loc[0], score_loc[1]), xycoords='axes fraction')
        elif isinstance(score_loc, str):
            if score_loc in 'upper left':
                ax.annotate(str_score, xy=(0.05, 0.75), xycoords='axes fraction', fontsize=score_fontsize)
            elif score_loc in 'upper right':
                ax.annotate(str_score, xy=(0.62, 0.75), xycoords='axes fraction', fontsize=score_fontsize)
            elif score_loc in 'lower left':
                ax.annotate(str_score, xy=(0.05, 0.05), xycoords='axes fraction', fontsize=score_fontsize)
            elif score_loc in 'lower right':
                ax.annotate(str_score, xy=(0.62, 0.05), xycoords='axes fraction', fontsize=score_fontsize)
            else:
                raise ValueError("Invalid score_loc! Possible options are list([xpos,ypos]), 'upper left', 'upper right', 'lower left', and 'lower right'")
        else:
            raise ValueError("Invalid score_loc! Possible options are list([xpos,ypos]), 'upper left', 'upper right', 'lower left', and 'lower right'")
    
    # grid
    if grid:
        ax.grid(color=grid_color,
                linestyle=grid_linestyle,
                alpha=grid_alpha,
                linewidth=grid_linewidth,
                zorder=grid_zorder,
                which=grid_which)
    
    # return
    if score:
        return scores
    else:
        return

def density2d(x, y,
            ax=None,
            xlabel=None, ylabel=None,
            xlim=None, ylim=None,
            identityline=True,
            identityline_color='k',
            identityline_linestyle='dashed',
            identityline_alpha=0.5,
            identityline_linewidth=0.5,
            xmajortickint=None,
            xminortickint=None,
            ymajortickint=None,
            yminortickint=None,
            aspect_equal=False,
            title=None,
            score=True,
            score_loc='lower right',
            score_fontsize=10,
            fmtbias='.3f',
            fmtrmse='.3f',
            fmtstd='.3f',
            fmtcorr='.3f',
            grid=True,
            grid_color='#b0b0b0',
            grid_linestyle='-',
            grid_alpha=None,
            grid_linewidth=0.8,
            grid_zorder=2.0,
            grid_which='both',
            bins=100,
            cmap=None):
    """
    Draw 2D density plot.
    
    Examples
    ---------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> fig = plt.figure(figsize=(4,4), dpi=300)
    >>> ax = plt.subplot()
    >>> scores = kkpy.plot.density2d(x, y, ax=ax)
    >>> print(scores)
    >>> plt.show()

    >>> # without score, without identityline
    >>> kkpy.plot.density2d(x, y, ax=ax, score=False, identityline=False)
    
    >>> # location of score text #1 (text)
    >>> scores = kkpy.plot.density2d(x, y, ax=ax, score_loc='lower right')

    >>> # location of score text #2 (list of xpos, ypos)
    >>> scores = kkpy.plot.density2d(x, y, ax=ax, score_loc=[0.7, 0.05]) # near 'lower right'

    >>> # more complicated options
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)*3
    >>> fig = plt.figure(figsize=(4,4), dpi=300)
    >>> ax = plt.subplot()
    >>> scores = kkpy.plot.density2d(
            x, y, ax=ax, score_loc='upper left',
            xlabel='X', ylabel='Y', xlim=[0,2], ylim=[0,3],
            title='TITLE', xmajortickint=0.2, xminortickint=0.1,
            ymajortickint=0.1, score_fontsize=12, fmtstd='.2f',
            identityline_color='b', identityline_linewidth=2,
            identityline_linestyle='solid'
        )
    >>> print(scores)
    >>> plt.show()
            
    Parameters
    ----------
    x : array_like
        Array containing multiple variables and observations.
    y : array_like
        Array containing multiple variables and observations. The shape should be same as **x**.
    ax : axes
        Axes class of matplotlib.
    xlabel : str, optional
        The label text of x axis.
    ylabel : str, optional
        The label text of y axis.
    xlim : list, optional
        The limits of x axis.
    ylim : list, optional
        The limits of y axis.
    alpha : float, optional
        Matplotlib alpha for scatter.
    identityline : boolean, optional
        True if draw identityline (one-to-one line). Default is True.
    identityline_color : str, optional
        Matplotlib color for identity line. Default is 'k'.
    identityline_linestyle : str, optional
        Matplotlib linestyle for identity line. Default is 'dashed'.
    identityline_alpha : float, optional
        Matplotlib alpha for identity line. Default is 0.5.
    identityline_linewidth : float, optional
        Matplotlib linewidth for identity line. Default is 0.5.
    xmajortickint : float, optional
        Major tick interval of x axis.
    xminortickint : float, optional
        Minor tick interval of x axis.
    ymajortickint : float, optional
        Major tick interval of y axis.
    yminortickint : float, optional
        Minor tick interval of y axis.
    aspect_equal : boolean, optional
        True if ax.set_aspect('equal'). Default is False.
    title : str, optional
        Title of the axis.
    score : boolean, optional
        True if annotate and return the evaluation score (bias, rmse, std, and corr)
    score_loc : str, optional
        Location of the evaluation score text in the plot. Possible options are list([xpos,ypos]), 'upper left', 'upper right', 'lower left', and 'lower right'. The xpos and ypos should be in the 'axes fraction' coordinate. Default is 'lower right'.
    score_fontsize : float, optional
        The fontsize of evaluation score text in the plot. Default is 10.
    fmtbias : str, optional
        String format for BIAS. Default is '.3f'.
    fmtrmse : str, optional
        String format for RMSE. Default is '.3f'.
    fmtstd : str, optional
        String format for STD. Default is '.3f'.
    fmtcorr : str, optional
        String format for CORR. Default is '.3f'.
    grid : boolean, optioinal
        True if draw grid.
    grid_color : str, optioinal
        Matplotlib color for grid.
    grid_linestyle : str, optioinal
        Matplotlib linestyle for grid.
    grid_alpha : float, optioinal
        Matplotlib alpha for grid.
    grid_linewidth : float, optioinal
        Matplotlib linewidth for grid.
    grid_zorder : float, optioinal
        Matplotlib zorder for grid.
    grid_which : str, optioinal
        The axis to draw the grid on. Possible options are 'both', 'xaxis', and 'yaxis'. Default is 'both'.
    bins : int or array_like
        Identical to bins of Matplotlib hist2d. Default is 100.
    cmap : obj
        Matplotlib cmap.

    Returns
    ---------
    score : list
        Return a score **if score is True**, otherwise no return.
    """
    from . import util
    import scipy.stats

    # xlim, ylim
    if xlim is not None:
        if ylim is not None:
            range=[xlim, ylim]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            raise ValueError('Both xlim and ylim should be defined if one of them is given')
    else:
        range=None
    
    hist2d, xedge, yedge, _ = scipy.stats.binned_statistic_2d(
        x, y, np.zeros(x.size),
        statistic='count',
        bins=bins,
        range=range)
    
    hist2d[hist2d == 0] = np.nan
    
    pm = ax.pcolormesh(xedge, yedge, hist2d.T, cmap=cmap, shading='flat')
    plt.colorbar(pm, ax=ax)
    
    # xlabel, ylabel
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    # tick intervals
    if xmajortickint is not None:
        tickint(ax=ax, major=xmajortickint, which='xaxis')
    if xminortickint is not None:
        tickint(ax=ax, minor=xminortickint, which='xaxis')
    if ymajortickint is not None:
        tickint(ax=ax, major=ymajortickint, which='yaxis')
    if yminortickint is not None:
        tickint(ax=ax, minor=yminortickint, which='yaxis')
    
    # identityline
    if identityline:
        xlim = ax.get_xlim()
        ylim = ax.get_xlim()
        ax.plot(xlim, ylim,
                color=identityline_color,
                linestyle=identityline_linestyle,
                alpha=identityline_alpha,
                linewidth=identityline_linewidth)
    
    # aspect_equal
    if aspect_equal:
        ax.set_aspect('equal')
    
    # title
    if title is not None:
        ax.set_title(title)
    
    # score
    if score:
        scores, str_score = util.stats(x, y,
                                       fmtbias=fmtbias,
                                       fmtrmse=fmtrmse,
                                       fmtstd=fmtstd,
                                       fmtcorr=fmtcorr)
        if isinstance(score_loc, list):
            ax.annotate(str_score, xy=(score_loc[0], score_loc[1]), xycoords='axes fraction')
        elif isinstance(score_loc, str):
            if score_loc in 'upper left':
                ax.annotate(str_score, xy=(0.05, 0.75), xycoords='axes fraction', fontsize=score_fontsize)
            elif score_loc in 'upper right':
                ax.annotate(str_score, xy=(0.62, 0.75), xycoords='axes fraction', fontsize=score_fontsize)
            elif score_loc in 'lower left':
                ax.annotate(str_score, xy=(0.05, 0.05), xycoords='axes fraction', fontsize=score_fontsize)
            elif score_loc in 'lower right':
                ax.annotate(str_score, xy=(0.62, 0.05), xycoords='axes fraction', fontsize=score_fontsize)
            else:
                raise ValueError("Invalid score_loc! Possible options are list([xpos,ypos]), 'upper left', 'upper right', 'lower left', and 'lower right'")
        else:
            raise ValueError("Invalid score_loc! Possible options are list([xpos,ypos]), 'upper left', 'upper right', 'lower left', and 'lower right'")
    
    # grid
    if grid:
        ax.grid(color=grid_color,
                linestyle=grid_linestyle,
                alpha=grid_alpha,
                linewidth=grid_linewidth,
                zorder=grid_zorder,
                which=grid_which)
    
    # return
    if score:
        return scores
    else:
        return
