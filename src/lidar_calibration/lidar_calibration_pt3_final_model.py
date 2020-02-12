"""
lidar_calibration_pt3_final_model.py
================================================================================
Collate the field and LiDAR TCH data and visualise the relationship between
canopy height and AGB for the plots. Produces a six panel figure, the TCH-AGB
relationship and five plots from the dataset.
"""
import numpy as np
import xarray as xr
import pandas as pd

import seaborn as sns
sns.set()
from matplotlib import pyplot as plt

from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table
from sklearn.model_selection import LeaveOneOut

import stats_tools as st

import sys
import os
import copy as cp
import gc

sys.path.append('../data_io/')
import data_io as io

# CHANGE THIS STUFF
# Some file names - shapefiles and rasters should be in a projected coordinate system (i.e. UTM)
inventory_file = '../../data/lidar_calibration/Kiuic_400_live_trees.shp'
pt1_outfile = '../../data/lidar_calibration/sample_test.npz' # this will be a file to hold the compiled plot data
pt2_outfile = '../../saved_models/lidar_calibration/lidar_calibration_pt2_results.npz'
pt3_outfile_prefix = '../../data/lidar_calibration/kiuic_lidar_agb'
path2fig = '../../figures/'
gap_ht = 8.66
CI=0.95 # desired confidence interval

# Load required information from pt1
pt1_output = np.load(pt1_outfile)['arr_0'][()]
chm_results = pt1_output['chm']
inventory_AGB = pt1_output['inventory']

# COLLATE DATA INTO ARRAYS
ID=[]; AGB = []; TCH = []; COVER = []; NODATA = []; TCH_SD = []
for plot in chm_results.keys():
    id = chm_results[plot]['id']
    if chm_results[id]['status']=='PASS':
        if chm_results[id]['weighted_average']>0:
            if np.sum(np.isnan(chm_results[id]['raster_values'][0])*chm_results[id]['weights'])<0.05*np.sum(chm_results[id]['weights']):
                weights = chm_results[id]['weights'].values/np.sum(np.isfinite(chm_results[id]['raster_values'][0].values)*chm_results[id]['weights'].values)
                mean_tch = chm_results[id]['weighted_average'][0]
                ID.append(id)
                AGB.append(inventory_AGB[id])
                TCH.append(mean_tch)
                COVER.append(np.sum((chm_results[id]['raster_values'][0].values>=gap_ht)*weights))

# Convert to np arrays for conditional indexing convenience
AGB = np.asarray(AGB); TCH = np.asarray(TCH); COVER = np.asarray(COVER); ID = np.asarray(ID)

# calculate residual gap fraction based on COVER-AGB relationship
mask = np.isfinite(np.log(COVER/(1-COVER)))
ro1,ro0,r,p,_=stats.linregress(np.log(TCH[mask]),np.log(COVER[mask]/(1-COVER[mask])))

# Load results from model fitting
pt2_output = np.load(pt2_outfile)['arr_0'][()]
model = pt2_output['model']
cal_data = pt2_output['cal_data']
log_fit = pt2_output['log_fit'] # True if log transform used to fit the model

# Refit model
ols = smf.ols(model,data=cal_data)
results = ols.fit()

# Load in the raster data
chm20 = xr.open_rasterio('chm20.tif')[0]
chm20.values = chm20.values/100.
mask = chm20.values>=0
chm20.values[~mask]=np.nan

cover20 = xr.open_rasterio('cover20.tif')[0]
cover20.values[~mask]=np.nan

cover20_res = chm20.copy(deep=True)
cover20_res.values = cover20.values - (1/(1 + np.exp(-ro0) * cover20.values**(-ro1)))

cover_res_grid =  1/(1 + np.exp(-ro0) * cover20.values**(-ro1))
"""
THIS WILL NEED UPDATING DEPENDING ON VARIABLES USED IN THE MODEL
"""
domain_data = pd.DataFrame({'TCH':chm20.values[mask],'COVER':cover20.values[mask],
                        'COVER_res':cover_res_grid[mask],'lnTCH':np.log(chm20.values[mask]),
                        'lnCOVER_res':np.log(1+cover_res_grid[mask])})
agb_mean = chm20.copy(deep=True)
agb95_l = chm20.copy(deep=True)
agb95_u = chm20.copy(deep=True)

predictions = results.get_prediction(domain_data)
sf=predictions.summary_frame(alpha=1-CI)

if log_fit:
    agb_mean.values[mask] = CF*np.exp(sf['mean'])
    agb95_l.values[mask] = CF*np.exp(sf['obs_ci_lower'])
    agb95_u.values[mask] = CF*np.exp(sf['obs_ci_upper'])
else:
    agb_mean.values[mask] = sf['mean']
    agb95_l.values[mask] = sf['obs_ci_lower']
    agb95_u.values[mask] = sf['obs_ci_upper']

agb_mean.values[~mask]=np.nan
agb95_u.values[~mask]=np.nan
agb95_l.values[~mask]=np.nan

# produce figure with the three maps
vmin=0;vmax=np.nanmax(agb95_u.values)
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=[9,6],sharex=True,sharey=True,
                        gridspec_kw={'bottom':0.3},)
agb95_l.sel(x=slice(229000,239000),y=slice(2230310,2216000)).plot(ax=axes[0],
            vmin=vmin, vmax=vmax, cmap='viridis',add_colorbar=False)
im = agb_mean.sel(x=slice(229000,239000),y=slice(2230310,2216000)).plot(ax=axes[1],
            vmin=vmin, vmax=vmax, cmap='viridis',add_colorbar=False)
agb95_u.sel(x=slice(229000,239000),y=slice(2230310,2216000)).plot(ax=axes[2],
            vmin=vmin, vmax=vmax, cmap='viridis',add_colorbar=False)

# configure the axes
titles=['95% PI lower', 'mean', '95% PI upper']
for ii,ax in enumerate(axes):
    ax.set_aspect('equal')
    ax.set_ylabel('')
    ax.set_xlabel('')
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment('right')
    ax.grid(True)
    ax.set_title(titles[ii])

# add a colorbar
cax = fig.add_axes([0.33,0.15,0.33,0.025])
plt.colorbar(im,orientation='horizontal',label='AGB / Mg ha$^{-1}$',cax=cax,
                extend='min')

fig.savefig('%slidar_AGB_models_and_CI95.png' % path2fig)
fig.show()

# write arrays to file
io.write_xarray_to_GeoTiff(agb_mean,'%s_mean' % pt3_outfile_prefix)
io.write_xarray_to_GeoTiff(agb95_l,'%s_95l' % pt3_outfile_prefix)
io.write_xarray_to_GeoTiff(agb95_u,'%s_95u' % pt3_outfile_prefix)
