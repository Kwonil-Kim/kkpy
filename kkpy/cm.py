"""
kkpy.cm
========================

Colormaps for my research

.. currentmodule:: cm

wind

.. autosummary::
    kkpy.cm.refl
    kkpy.cm.doppler
    kkpy.cm.zdr
    kkpy.cm.kdp
    kkpy.cm.rhohv
    kkpy.cm.precip
    kkpy.cm.precip_kma
    kkpy.cm.precip_kma_aws
    kkpy.cm.wind

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import copy

def refl(levels=None, snow=False):
    """
    KNU reflectivity colors.
    
    Examples
    ---------
    >>> cmap = kkpy.cm.refl()
    >>> pm = ax.pcolormesh(lon2d, lat2d, ref2d.T, cmap=cmap['cmap'], norm=cmap['norm'])
    >>> plt.colorbar(pm, ticks=cmap['ticks'])
    
    Parameters
    ----------
    levels : array_like, optional
        Array containing user-defined color tick levels. It should have lower and upper bounds.
    snow : boolean, optional
        Return broader tick levels suitable for snow. Default is False.
        
    Returns
    ---------
    dict_cmap : dictionary
        'cmap': matplotlib colormap
        'norm': color tick levels for plot
        'ticks': color tick levels for colorbar
    """
    
    if levels is None:
        level = _z_level_rain()
        if snow:
            level = _z_level_snow()
    else:
        level = levels
        
    if snow:
        cmap = _ref_color_snow()
    else:
        cmap = _ref_color()
    
    dict_cmap = {}
    dict_cmap['cmap'] = cmap
    dict_cmap['norm'] = col.BoundaryNorm(level, ncolors=_ref_color().N)
    dict_cmap['ticks'] = level[1:-1]
    
    return dict_cmap
        

def doppler(levels=None):
    """
    KNU Doppler velocity colors.
    
    Examples
    ---------
    >>> cmap = kkpy.cm.doppler()
    >>> pm = ax.pcolormesh(lon2d, lat2d, VD2d.T, cmap=cmap['cmap'], norm=cmap['norm'])
    >>> plt.colorbar(pm, ticks=cmap['ticks'])
    
    Parameters
    ----------
    levels : array_like, optional
        Array containing user-defined color tick levels. It should have lower and upper bounds.
        
    Returns
    ---------
    dict_cmap : dictionary
        'cmap': matplotlib colormap
        'norm': color tick levels for plot
        'ticks': color tick levels for colorbar
    """
    
    if levels is None:
        level = _vel_level()
    else:
        level = levels
    
    dict_cmap = {}
    dict_cmap['cmap'] = _vel_color()
    dict_cmap['norm'] = col.BoundaryNorm(level, ncolors=_vel_color().N)
    dict_cmap['ticks'] = level[1:-1]
    
    return dict_cmap
        
def zdr(levels=None):
    """
    KNU differential reflectivity colors.
    
    Examples
    ---------
    >>> cmap = kkpy.cm.zdr()
    >>> pm = ax.pcolormesh(lon2d, lat2d, zdr2d.T, cmap=cmap['cmap'], norm=cmap['norm'])
    >>> plt.colorbar(pm, ticks=cmap['ticks'])
    
    Parameters
    ----------
    levels : array_like, optional
        Array containing user-defined color tick levels. It should have lower and upper bounds.
        
    Returns
    ---------
    dict_cmap : dictionary
        'cmap': matplotlib colormap
        'norm': color tick levels for plot
        'ticks': color tick levels for colorbar
    """
    
    if levels is None:
        level = _zdr_level()
    else:
        level = levels
        
    dict_cmap = {}
    dict_cmap['cmap'] = _ref_color()
    dict_cmap['norm'] = col.BoundaryNorm(level, ncolors=_ref_color().N)
    dict_cmap['ticks'] = level[1:-1]
    
    return dict_cmap
        
def kdp(levels=None):
    """
    KNU specific differential phase colors.
    
    Examples
    ---------
    >>> cmap = kkpy.cm.kdp()
    >>> pm = ax.pcolormesh(lon2d, lat2d, kdp2d.T, cmap=cmap['cmap'], norm=cmap['norm'])
    >>> plt.colorbar(pm, ticks=cmap['ticks'])
    
    Parameters
    ----------
    levels : array_like, optional
        Array containing user-defined color tick levels. It should have lower and upper bounds.
        
    Returns
    ---------
    dict_cmap : dictionary
        'cmap': matplotlib colormap
        'norm': color tick levels for plot
        'ticks': color tick levels for colorbar
    """
    
    if levels is None:
        level = _kdp_level()
    else:
        level = levels
        
    dict_cmap = {}
    dict_cmap['cmap'] = _ref_color()
    dict_cmap['norm'] = col.BoundaryNorm(level, ncolors=_ref_color().N)
    dict_cmap['ticks'] = level[1:-1]
    
    return dict_cmap

def rhohv(levels=None):
    """
    KNU cross correlation coefficient colors.
    
    Examples
    ---------
    >>> cmap = kkpy.cm.rhohv()
    >>> pm = ax.pcolormesh(lon2d, lat2d, rhohv2d.T, cmap=cmap['cmap'], norm=cmap['norm'])
    >>> plt.colorbar(pm, ticks=cmap['ticks'])
    
    Parameters
    ----------
    levels : array_like, optional
        Array containing user-defined color tick levels. It should have lower and upper bounds.
        
    Returns
    ---------
    dict_cmap : dictionary
        'cmap': matplotlib colormap
        'norm': color tick levels for plot
        'ticks': color tick levels for colorbar
    """
    
    if levels is None:
        level = _rhv_level()
    else:
        level = levels
        
    dict_cmap = {}
    dict_cmap['cmap'] = _ref_color()
    dict_cmap['norm'] = col.BoundaryNorm(level, ncolors=_ref_color().N)
    dict_cmap['ticks'] = level[1:-1]
    
    return dict_cmap

def precip(levels=None, coarse_ticks=False):
    """
    KNU Precipitation colors.
    
    Examples
    ---------
    >>> cmap = kkpy.cm.precip()
    >>> pm = ax.pcolormesh(lon2d, lat2d, prec2d.T, cmap=cmap['cmap'], norm=cmap['norm'])
    >>> plt.colorbar(pm, ticks=cmap['ticks'])
    
    Parameters
    ----------
    levels : array_like, optional
        Array containing user-defined color tick levels. It should have lower and upper bounds.
    coarse_ticks : boolean, optional
        True if colorbar levels are too dense.
        
    Returns
    ---------
    dict_cmap : dictionary
        'cmap': matplotlib colormap
        'norm': color tick levels for plot
        'ticks': color tick levels for colorbar
    """
    
    if levels is None:
        level = _rain_level()
    else:
        level = levels
    
    dict_cmap = {}
    dict_cmap['cmap'] = _rain_color()
    dict_cmap['norm'] = col.BoundaryNorm(level, ncolors=_rain_color().N)
    dict_cmap['ticks'] = level[1:-1]
    if coarse_ticks:
        dict_cmap['ticks'] = [0, 0.4, 1, 2, 5, 7, 10, 14, 20, 30, 50, 70, 100]
    
    return dict_cmap

def precip_kma(levels=None, coarse_ticks=False):
    """
    KMA Radar Precipitation colors.
    
    Examples
    ---------
    >>> cmap = kkpy.cm.precip_kma()
    >>> pm = ax.pcolormesh(lon2d, lat2d, prec2d.T, cmap=cmap['cmap'], norm=cmap['norm'])
    >>> plt.colorbar(pm, ticks=cmap['ticks'])
    
    Parameters
    ----------
    levels : array_like, optional
        Array containing user-defined color tick levels. It should have lower and upper bounds.
    coarse_ticks : boolean, optional
        True if colorbar levels are too dense.
        
    Returns
    ---------
    dict_cmap : dictionary
        'cmap': matplotlib colormap
        'norm': color tick levels for plot
        'ticks': color tick levels for colorbar
    """
    
    if levels is None:
        level = _rain_level_kma()
    else:
        level = levels
    
    dict_cmap = {}
    dict_cmap['cmap'] = _rain_color_kma()
    dict_cmap['norm'] = col.BoundaryNorm(level, ncolors=_rain_color_kma().N)
    dict_cmap['ticks'] = level[1:-1]
    if coarse_ticks:
        dict_cmap['ticks'] = [0, 1, 3, 5, 7, 10, 20, 30, 50, 70, 150]
    
    return dict_cmap

def precip_kma_aws(levels=None, coarse_ticks=False):
    """
    KMA AWS Precipitation colors.
    
    Examples
    ---------
    >>> cmap = kkpy.cm.precip_kma_aws()
    >>> pm = ax.pcolormesh(lon2d, lat2d, prec2d.T, cmap=cmap['cmap'], norm=cmap['norm'])
    >>> plt.colorbar(pm, ticks=cmap['ticks'])
    
    Parameters
    ----------
    levels : array_like, optional
        Array containing user-defined color tick levels. It should have lower and upper bounds.
    coarse_ticks : boolean, optional
        True if colorbar levels are too dense.
        
    Returns
    ---------
    dict_cmap : dictionary
        'cmap': matplotlib colormap
        'norm': color tick levels for plot
        'ticks': color tick levels for colorbar
    """
    
    if levels is None:
        level = _rain_level_kma_aws()
    else:
        level = levels
    
    dict_cmap = {}
    dict_cmap['cmap'] = _rain_color_kma_aws()
    dict_cmap['norm'] = col.BoundaryNorm(level, ncolors=_rain_color_kma_aws().N)
    dict_cmap['ticks'] = level[1:-1]
    if coarse_ticks:
        dict_cmap['ticks'] = [0, 0.4, 1, 2, 5, 7, 10, 14, 20, 30, 50, 70, 100]
    
    return dict_cmap

def wind(levels=None):
    """
    Wind direction colors (cyclic).
    
    Examples
    ---------
    >>> cmap = kkpy.cm.wind()
    >>> pm = ax.pcolormesh(lon2d, lat2d, winddir.T, cmap=cmap['cmap'], norm=cmap['norm'])
    >>> cb = plt.colorbar(pm, ticks=cmap['ticks'])
    >>> cb.set_ticklabels(['N', 'E', 'S', 'W', 'N'])
    
    Parameters
    ----------
    levels : array_like, optional
        Array containing user-defined color tick levels. It should have lower and upper bounds.
        
    Returns
    ---------
    dict_cmap : dictionary
        'cmap': matplotlib colormap
        'norm': color tick levels for plot
        'ticks': color tick levels for colorbar
    """
    
    if levels is None:
        level = [0,90,180,270,360]
    else:
        level = levels
    
    cmap = _wind_color()
    
    dict_cmap = {}
    dict_cmap['cmap'] = cmap
    dict_cmap['norm'] = col.BoundaryNorm(np.linspace(0,360,256), ncolors=256)
    dict_cmap['ticks'] = level
    
    return dict_cmap

def _ref_color():
    colors = ['#ffffff',
     '#c8c8c8', '#00c8ff', '#009bf5', '#0000f5',
     '#00f500', '#00be00', '#008c00', '#005a00',
     '#FFFF00', '#E6B400', '#ff9600', '#ff0000',
     '#b40000', '#E6468C', '#7828a0',
     '#000000']
    cmap_colors = col.ListedColormap(colors)
    plt.cm.register_cmap(name='knuref', cmap=cmap_colors)
    return cmap_colors

def _ref_color_snow():
    colors = ['#ffffff',
     '#c8c8c8', '#00c8ff', '#009bf5', '#0000f5',
     '#00f500', '#00be00', '#008c00', '#005a00',
     '#FFFF00', '#E6B400', '#ff9600', '#ff0000',
     '#b40000', '#7828a0',
     '#000000']
    cmap_colors = col.ListedColormap(colors)
    plt.cm.register_cmap(name='knuref', cmap=cmap_colors)
    return cmap_colors

def _rain_color():
    colors = ['#ffffff', '#e0e0e0',
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

def _rain_color_kma():
    colors = ['#ffffff', '#eeeeee',
     '#00c8ff', '#009bf5', '#004af5', '#00ff00', '#00be00',
     '#008c00', '#005a00', '#ffff00', '#ffdc1f', '#f9cd00',
     '#e0b900', '#ccaa00', '#ff6600', '#ff3200', '#d20000',
     '#b40000', '#e0a9ff', '#cc6aff', '#b329ff', '#9300e4',
     '#b3b4de', '#4c4eb1', '#000390', '#333333']
    cmap_colors = col.ListedColormap(colors)
    plt.cm.register_cmap(name='kmarain', cmap=cmap_colors)
    return cmap_colors

def _rain_color_kma_aws():
    colors = ['#ffffff', '#eeeeee',
     '#ffea6e', '#ffdc1f', '#f9cd00', '#e0b900', '#ccaa00',
     '#69fc69', '#1ef41e', '#00d500', '#00a400', '#008000',
     '#87d9ff', '#3ec1ff', '#07abff', '#008dde', '#0077b3',
     '#b3b4de', '#8081c7', '#4c4eb1', '#1f21ad', '#000390',
     '#da87ff', '#ce3eff', '#ad07ff', '#9200e4', '#7f00bf',
     '#fa8585', '#f63e3e', '#ee0b0b', '#d50000', '#bf0000',
     '#333333']
    cmap_colors = col.ListedColormap(colors)
    plt.cm.register_cmap(name='kmaawsrain', cmap=cmap_colors)
    return cmap_colors

def _vel_color():
    colors =['#d2d2d2', '#00c8ff', '#009bf5', '#0000f5',
     '#00f500', '#00be00', '#008c00', '#004f00', '#646464',
     '#ffff00', '#e6b400', '#ff9600', '#ff0000', '#b40000',
     '#e6468c', '#7828a0', '#000000']
    cmap_colors = col.ListedColormap(colors)
    plt.cm.register_cmap(name='knuvel', cmap=cmap_colors)
    return cmap_colors
    
def _vel_color_vpr_snow():
    colors =['#e6e6e6', '#005a00', '#007300', '#008c00', '#00be02', '#00f500', '#0000f5', '#0051f8', '#009bf5',
     '#00b2fa', '#00c8ff', '#646464', '#ffff00', '#ff9600', '#ff0000', '#b50000', '#7828a0']
    cmap_colors = col.ListedColormap(colors)
    plt.cm.register_cmap(name='knuvel', cmap=cmap_colors)
    return cmap_colors

def _wind_color():
    colors = ['#8000c3','#7e00c4','#7c00c5','#7a00c6','#7800c7','#7600c8','#7400c8','#7200c9','#7000ca','#6e00cb',
              '#6c00cc','#6a00cd','#6800ce','#6600cf','#6400d0','#6200d1','#6000d2','#5e00d3','#5c00d4','#5a00d5',
              '#5800d6','#5600d7','#5400d8','#5200d9','#5000da','#4e00db','#4c00dc','#4a00dd','#4800de','#4600de',
              '#4400df','#4200e0','#4000e1','#3e00e2','#3c00e3','#3a00e4','#3800e5','#3600e6','#3400e7','#3200e8',
              '#3000e9','#2e00ea','#2c00eb','#2a00ec','#2800ed','#2600ee','#2400ef','#2200f0','#2000f1','#1e00f2',
              '#1c00f3','#1a00f4','#1800f5','#1600f5','#1400f6','#1200f7','#1000f8','#0e00f9','#0c00fa','#0a00fb',
              '#0800fc','#0600fd','#0400fe','#0200ff','#0101ff','#0505fb','#0909f7','#0d0df3','#1111ef','#1515eb',
              '#1919e7','#1d1de3','#2121df','#2525db','#2929d7','#2d2dd3','#3131cf','#3535cb','#3939c7','#3d3dc3',
              '#4141bf','#4545bb','#4949b7','#4d4db3','#5151af','#5555ab','#5959a7','#5d5da2','#61619e','#65659a',
              '#696996','#6d6d92','#71718e','#75758a','#797986','#7d7d82','#81817e','#85857a','#898976','#8d8d72',
              '#91916e','#95956a','#999966','#9d9d62','#a1a15e','#a5a55a','#aaaa56','#aeae52','#b2b24e','#b6b64a',
              '#baba46','#bebe42','#c2c23e','#c6c63a','#caca36','#cece32','#d2d22e','#d6d62a','#dada26','#dede22',
              '#e2e21e','#e6e61a','#eaea16','#eeee12','#f2f20e','#f6f60a','#fafa06','#fefe02','#fffe00','#fffa00',
              '#fff600','#fff200','#ffee00','#ffea00','#ffe600','#ffe200','#ffde00','#ffda00','#ffd600','#ffd200',
              '#ffce00','#ffca00','#ffc600','#ffc200','#ffbe00','#ffba00','#ffb600','#ffb200','#ffae00','#ffaa00',
              '#ffa600','#ffa100','#ff9d00','#ff9900','#ff9500','#ff9100','#ff8d00','#ff8900','#ff8500','#ff8100',
              '#ff7d00','#ff7900','#ff7500','#ff7100','#ff6d00','#ff6900','#ff6500','#ff6100','#ff5d00','#ff5900',
              '#ff5500','#ff5100','#ff4d00','#ff4900','#ff4500','#ff4100','#ff3d00','#ff3900','#ff3500','#ff3100',
              '#ff2d00','#ff2900','#ff2500','#ff2100','#ff1d00','#ff1900','#ff1500','#ff1100','#ff0d00','#ff0900',
              '#ff0500','#ff0100','#fe0002','#fc0005','#fa0008','#f8000b','#f6000e','#f40011','#f20015','#f00018',
              '#ee001b','#ec001e','#ea0021','#e80024','#e60027','#e4002a','#e2002d','#e00030','#de0033','#dc0036',
              '#da0039','#d8003c','#d6003f','#d40042','#d20045','#d00048','#ce004b','#cc004e','#ca0051','#c80054',
              '#c60057','#c4005b','#c2005e','#c00061','#be0064','#bc0067','#ba006a','#b8006d','#b60070','#b40073',
              '#b20076','#b00079','#ae007c','#ac007f','#aa0082','#a80085','#a60088','#a4008b','#a2008e','#a00091',
              '#9e0094','#9c0097','#9a009a','#98009d','#9600a0','#9400a4','#9200a7','#9000aa','#8e00ad','#8c00b0',
              '#8a00b3','#8800b6','#8600b9','#8400bc','#8200bf','#8000c2']
    cmap_colors = col.ListedColormap(colors)
    plt.cm.register_cmap(name='kkpywind', cmap=cmap_colors)
    return cmap_colors

# LEVELS
def _z_level_rain():
    return [-100.0, -32.0, -10.0, 0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 500.0]

def _zdr_level():
    return [-100.0,  -5.0,  -3.0, -2.0, -1.0,  0.0,  0.2,  0.4,  0.6,  1.0,  1.4,  1.8,  2.2,  2.6,  3.0,  4.0,  5.0, 500.0]

def _kdp_level():
    return [-100.0,  -2.0,  -1.5, -1.0, -0.5,  0.0,  0.2,  0.5,  1.0,  1.5,  2.0,  2.5,  3.0,  3.5,  4.0,  4.5,  5.0, 500.0]

def _rhv_level():
    return [-100.0,   0.0,   0.5,  0.6,  0.7,  0.8, 0.82, 0.84,  0.88, 0.9, 0.92,  0.94, 0.96, 0.97, 0.98, 0.99, 1.0, 500.0]

def _vel_level():
    return [-100, -64, -42, -32, -24, -16, -8, -2, -0.5, 0.5, 2, 8, 16, 24, 32, 42, 64, 100]

def _rain_level(scale=1):
    return [x*scale for x in [-100., 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 1000]]

def _rain_level_kma():
    return [-100, 0, 0.1, 0.2, 0.5, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 90, 110, 150, 1000]

def _rain_level_kma_aws():
    return [-100, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 1000]

def _z_level_snow():
    return [-100.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 500.0]

def _vel_level_vpr_snow():
    return [-100, -10, -5, -3, -2, -1.6, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.5, 1.0, 3.0, 5.0, 100]

