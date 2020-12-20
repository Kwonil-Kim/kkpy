"""
kkpy.plot
========================

Functions to read and write files

.. currentmodule:: plot

.. autosummary::
    kkpy.plot.koreamap

"""

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
