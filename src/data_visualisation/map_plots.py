"""
map_plots
--------------------------------------------------------------------------------
GENERATE MAP PLOTS FOR WORKSHOP
# David T. Milodowski, 25/03/2019
"""

"""
import libraries needed
"""
import numpy as np
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package
import cartopy.crs as ccrs

sns.set()

"""
plot_xarray
-----------
a very simple script to produce a nice simple map from an xarray object
"""
def plot_xarray(xarr, figure_name = None,figsize_x=8,figsize_y=6,vmin=None,
                vmax=None,cmap='viridis',add_colorbar=False,cbar_kwargs={},
                show=True,title="",subplot_kw={}):
    if vmin is None:
        vmin = np.nanmin(xarr)
    if vmax is None:
        vmax =np.nanmax(xarr)

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(figsize_x,figsize_y),
                   subplot_kw=subplot_kw)
    if add_colorbar:
        extend = 'neither'
        if vmin > np.nanmin(xarr.values):
            if vmax < np.nanmax(xarr.values):
                extend = 'both'
            else:
                extend = 'min'
        else:
            if vmax < np.nanmax(xarr.values):
                extend = 'max'
        xarr.plot.imshow(ax=axis, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=add_colorbar,
                    extend=extend, cbar_kwargs=cbar_kwargs,transform = subplot_kw['projection'])
    else:
        xarr.plot.imshow(ax=axis, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=add_colorbar,transform = subplot_kw['projection'])
    axis.set_aspect("equal")
    axis.set_title(title,fontsize=16)
    if figure_name is not None:
        fig.savefig(figure_name)
    if show:
        fig.show()
    return fig,axis

# same as before, but places plot on an existing axis, rather than creating a new one
def plot_xarray_to_axis(xarr, ax, vmin=None, vmax=None, cmap='viridis',
                add_colorbar=False, cbar_kwargs={}, title="", subplot_kw={}):
    if vmin is None:
        vmin = np.nanmin(xarr)
    if vmax is None:
        vmax =np.nanmax(xarr)

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(figsize_x,figsize_y),
                   subplot_kw=subplot_kw)
    if add_colorbar:
        extend = 'neither'
        if vmin > np.nanmin(xarr.values):
            if vmax < np.nanmax(xarr.values):
                extend = 'both'
            else:
                extend = 'min'
        else:
            if vmax < np.nanmax(xarr.values):
                extend = 'max'
        xarr.plot.imshow(ax=axis, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=add_colorbar,
                    extend=extend, cbar_kwargs=cbar_kwargs,transform = subplot_kw['projection'])
    else:
        xarr.plot.imshow(ax=axis, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=add_colorbar,transform = subplot_kw['projection'])
    axis.set_aspect("equal")
    axis.set_title(title,fontsize=16)
    return 0
