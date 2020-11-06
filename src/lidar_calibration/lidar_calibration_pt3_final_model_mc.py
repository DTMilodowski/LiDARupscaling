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
version = '034'
inventory_file = '../../data/lidar_calibration/Kiuic_400_live_trees.shp'
raster_file = '../../data/LiDAR_data/GliHT_TCH_1m_100.tif'
pt1_outfile = '../../data/lidar_calibration/kiuic_plot_lidar_sample_%s.npz' % version
pt2_outfile = '../../saved_models/lidar_calibration/lidar_calibration_pt2_results_%s.npz' % version
pt3_outfile_prefix = '../../data/lidar_calibration/kiuic_lidar_agb'
path2fig = '../../figures/'
CI=0.95 # desired confidence interval

# Load required information from pt2
pt2_output = np.load(pt2_outfile,allow_pickle=True)['arr_0'][()]
mc_results = pt2_output['mc_results']
cal_data = pt2_output['cal_data']

log_fit=True
if np.isnan(mc_results['CF'][0]):
    log_fit=False

os.system('gdalwarp -overwrite -tr 20 20 -r average %s chm20.tif' % raster_file)
chm20 = xr.open_rasterio('chm20.tif')[0].astype('float')
chm20.values/=100
mask = chm20.values>=0
chm20.values[~mask]=np.nan

"""
THIS WILL NEED UPDATING DEPENDING ON VARIABLES USED IN THE MODEL
"""
Niter = 100
N = mask.sum()
domain_data = pd.DataFrame({'lnTCH':np.log(chm20.values[mask]),'Collection':0,'plot':0})
agb_median = chm20.copy(deep=True)
agb95_l = chm20.copy(deep=True)
agb95_u = chm20.copy(deep=True)

agb_mc = np.zeros((Niter,N))
for ii in range(0,Niter):
    model = mc_results['fitted_models'][ii]
    agb_mc[ii]=mc_results['CF'][ii]*np.exp(model.predict(domain_data))

agb_median.values[~mask]=np.nan
agb95_u.values[~mask]=np.nan
agb95_l.values[~mask]=np.nan

agb_median.values[mask]=np.percentile(agb_mc,50,axis=0)
agb95_u.values[mask]=np.percentile(agb_mc,97.5,axis=0)
agb95_l.values[mask]=np.percentile(agb_mc,2.5,axis=0)

# produce figure with the three maps
vmin=0;vmax=np.nanpercentile(agb95_u.values,99)
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=[8,5],sharex=True,sharey=True,
                        gridspec_kw={'bottom':0.3},)
agb95_l.sel(x=slice(229000,239000),y=slice(2230310,2216000)).plot(ax=axes[0],
            vmin=vmin, vmax=vmax, cmap='viridis',add_colorbar=False)
im = agb_median.sel(x=slice(229000,239000),y=slice(2230310,2216000)).plot(ax=axes[1],
            vmin=vmin, vmax=vmax, cmap='viridis',add_colorbar=False)
agb95_u.sel(x=slice(229000,239000),y=slice(2230310,2216000)).plot(ax=axes[2],
            vmin=vmin, vmax=vmax, cmap='viridis',add_colorbar=False)

# configure the axes
titles=['2.5% CI', 'median', '97.5% CI']
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
                extend='max')

fig.savefig('%slidar_AGB_models_and_CI95_%s.png' % (path2fig,version))
fig.show()

# write arrays to file
io.write_xarray_to_GeoTiff(agb_median,'%s_%s_median' % (pt3_outfile_prefix,version))
io.write_xarray_to_GeoTiff(agb95_l,'%s_%s_95l' % (pt3_outfile_prefix,version))
io.write_xarray_to_GeoTiff(agb95_u,'%s_%s_95u' % (pt3_outfile_prefix,version))

"""
---------------------------------------------------------------------------------
REGRID TO DESIRED RESOLUTION (1 ha)
--------------------------------------------------------------------------------
"""

outres = 100. # 100 m (i.e. 1 ha)
agb20_mc = chm20.copy(deep=True)
for ii in range(0,Niter):
    model = mc_results['fitted_models'][ii]
    agb20_mc.values[~mask] = np.nan
    agb20_mc.values[mask] = agb_mc[ii]
    io.write_xarray_to_GeoTiff(agb20_mc,'%s_%s_20m_mc_%s' % (pt3_outfile_prefix, version, str(ii+1).zfill(3)))
"""
    os.system("gdalwarp -overwrite -dstnodata -9999 -tr 100 -100 -r average \
            -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
            %s_20m_mc_%s.tif %s_100m_mc_%s.tif" % (pt3_outfile_prefix, str(ii+1).zfill(3),
            pt3_outfile_prefix, str(ii+1).zfill(3)))

# equivalent for 0.25 ha
outres = 50. # 100 m (i.e. 1 ha)
agb20_mc = chm20.copy(deep=True)
for ii in range(0,Niter):
    model = mc_results['fitted_models'][ii]
    agb20_mc.values[~mask] = np.nan
    agb20_mc.values[mask] = agb_mc[ii]
    io.write_xarray_to_GeoTiff(agb20_mc,'%s_%s_20m_mc_%s' % (pt3_outfile_prefix, version, str(ii+1).zfill(3)))
    os.system("gdalwarp -overwrite -dstnodata -9999 -tr 50 -50 -r average \
            -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
            %s_20m_mc_%s.tif %s_50m_mc_%s.tif" % (pt3_outfile_prefix, str(ii+1).zfill(3),
            pt3_outfile_prefix, str(ii+1).zfill(3)))
"""
