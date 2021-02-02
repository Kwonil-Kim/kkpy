"""
Plotting DEM
======================

Read/plot DEM and korea map with DFS map projection
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import kkpy

# read DEM
dem, londem, latdem, projdem = kkpy.io.read_dem(area='korea')

# load DFS projection
proj = kkpy.util.proj_dfs()

# initialize figure and axes
fig = plt.figure(figsize=(5,5), dpi=300)
ax = plt.subplot(projection=proj)

# set boundary of map
ax.set_extent([124, 130, 33, 43], crs=ccrs.PlateCarree())

# plot dem
cmap = plt.cm.terrain
cmap.set_under('white')
plt.pcolormesh(londem, latdem, dem, transform=ccrs.PlateCarree(), cmap=cmap)
plt.colorbar()

# set grid
gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl.xlocator = mticker.FixedLocator(np.arange(124,130.1,1))
gl.ylocator = mticker.FixedLocator(np.arange(33,43.1,1))
gl.rotate_labels = False
gl.top_labels = gl.right_labels = False

#kkpy.plot.koreamap(ax=ax)
plt.show()
