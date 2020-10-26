"""
kkpy.cm
========================

Colormaps for my research

.. currentmodule:: cm

wind

.. autosummary::
    kkpy.cm.refl
    kkpy.cm.doppler
    kkpy.cm.precip

"""
import numpy as np
import matplotlib.colors as col

def refl(levels=None, snow=True):
    """
    KNU reflectivity colors.
    
    Examples
    ---------
    >>> cmap = kkpy.cm.refl()
    >>> pm = ax.pcolormesh(lon2d, lat2d, ref2d.T, cmap=cmap['cmap'], norm=cmap['norm'])
    >>> plt.colorbar(pm, cmap=cmap['cmap'], ticks=cmap['tick'])
    
    Parameters
    ----------
    levels : array_like
        Array containing user-defined color tick levels. It should have lower and upper bounds.
    snow : boolean
        Return broader tick levels suitable for snow. Default is False.
        
    Returns
    ---------
    dict_cmap : dictionary
        'cmap': matplotlib colormap
        'norm': color tick levels for plot
        'ticks': color tick levels for colorbar
    """
    
    if levels is None:
        level = z_level_rain()
        if snow:
            level = z_level_snow()
    else:
        level = levels
    
    dict_cmap = {}
    dict_cmap['cmap'] = ref_color()
    dict_cmap['norm'] = col.BoundaryNorm(level, ncolors=ref_color().N)
    dict_cmap['ticks'] = level[1:-1]
    
    return dict_cmap
        

def doppler(levels=None):
    """
    KNU Doppler velocity colors.
    
    Examples
    ---------
    >>> cmap = kkpy.cm.doppler()
    >>> pm = ax.pcolormesh(lon2d, lat2d, VD2d.T, cmap=cmap['cmap'], norm=cmap['norm'])
    >>> plt.colorbar(pm, cmap=cmap['cmap'], ticks=cmap['tick'])
    
    Parameters
    ----------
    levels : array_like
        Array containing user-defined color tick levels. It should have lower and upper bounds.
        
    Returns
    ---------
    dict_cmap : dictionary
        'cmap': matplotlib colormap
        'norm': color tick levels for plot
        'ticks': color tick levels for colorbar
    """
    
    if levels is None:
        level = vel_level()
    else:
        level = levels
    
    dict_cmap = {}
    dict_cmap['cmap'] = vel_color()
    dict_cmap['norm'] = col.BoundaryNorm(level, ncolors=ref_color().N)
    dict_cmap['ticks'] = level[1:-1]
    
    return dict_cmap
        
def precip(levels=None):
    """
    KNU Precipitation colors.
    
    Examples
    ---------
    >>> cmap = kkpy.cm.precip()
    >>> pm = ax.pcolormesh(lon2d, lat2d, prec2d.T, cmap=cmap['cmap'], norm=cmap['norm'])
    >>> plt.colorbar(pm, cmap=cmap['cmap'], ticks=cmap['tick'])
    
    Parameters
    ----------
    levels : array_like
        Array containing user-defined color tick levels. It should have lower and upper bounds.
        
    Returns
    ---------
    dict_cmap : dictionary
        'cmap': matplotlib colormap
        'norm': color tick levels for plot
        'ticks': color tick levels for colorbar
    """
    
    if levels is None:
        level = rain_level()
    else:
        level = levels
    
    dict_cmap = {}
    dict_cmap['cmap'] = ref_color()
    dict_cmap['norm'] = col.BoundaryNorm(level, ncolors=ref_color().N)
    dict_cmap['ticks'] = rain_tick()
    
    return dict_cmap
        


def ref_color():
    colors = ['#ffffff',
     '#c8c8c8', '#00c8ff', '#009bf5', '#0000f5',
     '#00f500', '#00be00', '#008c00', '#005a00',
     '#FFFF00', '#E6B400', '#ff9600', '#ff0000',
     '#b40000', '#E6468C', '#7828a0',
     '#000000']
    cmap_colors = col.ListedColormap(colors)
    plt.cm.register_cmap(name='knuref', cmap=cmap_colors)
    return cmap_colors

def rain_color():
    colors = ['#e0e0e0',
     '#87d9ff', '#3ec1ff', '#07abff', '#008dde', '#0077b3',
     '#69fc69', '#1ef269', '#00d500', '#00a400', '#008000',
     '#fff26f', '#ffe256', '#ffd039', '#ffbc1e', '#ffaa09',
     '#ff9c00', '#ff8b1b', '#ff8051', '#ff6e6e', '#f65e67',
     '#e84a56', '#d9343e', '#c91f25', '#bc0d0f', '#b30000',
     '#c55aff', '#c234ff', '#ad07ff', '#9200e4', '#7f00bf',
     '#333333']
    cmap_colors = col.ListedColormap(colors)
    plt.cm.register_cmap(name='knurain', cmap=cmap_colors)
    return cmap_colors

def vel_color():
    colors =['#d2d2d2', '#00c8ff', '#009bf5', '#0000f5', '#00f500', '#00be00', '#008c00', '#004f00', '#646464',
     '#ffff00', '#e6b400', '#ff9600', '#ff0000', '#b40000', '#e6468c', '#7828a0', '#000000']
    cmap_colors = col.ListedColormap(colors)
    plt.cm.register_cmap(name='knuvel', cmap=cmap_colors)
    return cmap_colors
    

# LEVELS
def z_level_rain():
    return [-100.0, -32.0, -10.0, 0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 500.0]

def rain_level(scale=1):
    return [x*scale for x in [-10000, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 1000]]

def vel_level():
    return [-100, -64, -42, -32, -24, -16, -8, -2, -0.5, 0.5, 2, 8, 16, 24, 32, 42, 64, 100]

# TICKS
def rain_tick(scale=1, coarse=False):
    level = rain_level(scale=scale)
    ticklabel = list(map(str, level))
    for i, _ticklabel in enumerate(ticklabel):
        if coarse:
            mask = [0,2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,20,22,23,24,25,27,28,29,30,32]
        else:
            mask = [0,32]
        if i in mask:
            ticklabel[i] = ' '
    
    return ticklabel