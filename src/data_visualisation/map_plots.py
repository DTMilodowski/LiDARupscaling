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
sns.set()

"""
plot_xarray
-----------
a very simple script to produce a nice simple map from an xarray object
"""
def plot_xarray(xarr, figure_name = None,figsize_x=8,figsize_y=6,vmin=None,
                vmax=None,cmap='viridis',add_colorbar=False,cbar_kwargs={},
                show=True,title=""):
    if vmin is None:
        vmin = np.nanmin(xarr)
    if vmax is None:
        vmax =np.nanmax(xarr)
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(figsize_x,figsize_y))
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
        xarr.plot(ax=axis, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=add_colorbar,
                    extend=extend, cbar_kwargs=cbar_kwargs)
    else:
        xarr.plot(ax=axis, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=add_colorbar)
    axis.set_aspect("equal")
    axis.set_title(title,fontsize=16)
    if figure_name is not None:
        fig.savefig(figure_name)
    if show:
        fig.show()
    return fig,axis

"""
plot_AGBobs_and_AGBpot
-------------------------
plot observed and predicted AGB
"""
def plot_AGBobs_and_AGBpot(agb,agbpot,cmap='viridis',vmin=None,vmax=None,show=True):

    # Deal with colour limits for color ramp
    if vmin is None:
        vmin = np.nanmin(agb.values)
    if vmax is None:
        vmax =np.nanmax(agb.values)
    extend = 'neither'
    if vmin > np.nanmin(agb.values):
        if vmax < np.nanmax(agb.values):
            extend = 'both'
        else:
            extend = 'min'
    else:
        if vmax < np.nanmax(agb.values):
            extend = 'max'

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    agb.plot(ax=axes[0], vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=True,
                        extend=extend, cbar_kwargs={'label': 'AGB / Mg ha$^{-1}$',
                        'orientation':'horizontal'})
    agbpot.plot(ax=axes[1], vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=True,
                        extend=extend, cbar_kwargs={'label': 'AGB$_{pot}$ / Mg ha$^{-1}$',
                        'orientation':'horizontal'})
    for ax in axes:
        ax.set_aspect("equal")
    axes[0].set_title("Observed AGB")
    axes[1].set_title("Modelled potential AGB")
    if show:
        fig.show()
    return fig,axes

"""
plot_AGBpot_scenario
-------------------------
plot observed and predicted AGB
"""
def plot_AGBpot_scenario(agbpot,agbpot_scenario,agbpot_difference,scenario_name,show=True):

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,6))
    agbpot.plot(ax=axes[0], vmin=0, vmax=400, cmap='viridis', add_colorbar=True,
                        extend='max', cbar_kwargs={'label': 'AGB$_{pot}$  / Mg ha$^{-1}$',
                        'orientation':'horizontal'})
    agbpot_scenario.plot(ax=axes[1], vmin=0, vmax=400, cmap='viridis', add_colorbar=True,
                        extend='max', cbar_kwargs={'label': 'AGB$_{pot}$  / Mg ha$^{-1}$',
                        'orientation':'horizontal'})
    agbpot_difference.plot(ax=axes[2], vmin=-300, vmax=300, cmap='bwr_r', add_colorbar=True,
                        extend='both', cbar_kwargs={'label': 'difference in AGB$_{pot}$ / Mg ha$^{-1}$',
                        'orientation':'horizontal'})
    for ax in axes:
        ax.set_aspect("equal")
    axes[0].set_title("Potential AGB")
    axes[1].set_title("Potential AGB in 2070 under %s" % scenario_name)
    axes[2].set_title("Difference in potential AGB under %s" % scenario_name)
    fig.tight_layout()
    if show:
        fig.show()
    return fig,axes
