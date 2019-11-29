"""
lidar_agb_map_confidence_intervals.py
--------------------------------------------------------------------------------
ESTIMATE CONFIDENCE INTERVALS FOR AGB MAP (PREDICTION INTERVALS)
This script calculates specified prediction intervals and confidence intervals
based on a multivariate linear regression, and produces corresponding error maps
(expressed as a fraction of the best estimate)

Input files are:
- the calibration data - a csv file with the following columns included:
    - plot_ID
    - plot_AGB
    - lidar_metric_1
    - lidar_metric_2
    - lidar_metric_3
  note, that the header names given here can be different, but will need to be
  updated in the script below to make sure that they are found correctly
- three rasters with the lidar metrics used. Specify the filenames in the
  "Project Info" section

Other things to specify:
- the confidence level required, specified via the alpha option
    - alpha = 0.05 -> 95% confidence range

11/11/2019 - D. T. Milodowski
--------------------------------------------------------------------------------
"""

"""
# Import the necessary packages
"""
import numpy as np                  # standard package for scientific computing
import xarray as xr                 # xarray geospatial package
import cartopy.crs as ccrs          # map projections
import pandas as pd                 # useful for dataframes and integration with stats
from statsmodels.formula.api import ols # ordinary least squares regression
from scipy import stats             # statistics module
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package
import os

# Import custom libaries
import sys
sys.path.append('./data_io/')
sys.path.append('./data_visualisation/')
sys.path.append('./random_forest/')

import data_io as io
import map_plots as mplt
import random_forest_functions as rff

"""
Project Info
"""
site_id = 'kiuic'
crs = ccrs.UTM('16N')
path2fig= '../../figures/'
path2output = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/lidar/error_map/'

path2field = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/field_inventory/' # specify location of field data
calibration_data_file = 'agb_carto_gliht_tnc_RF_filtered.csv' # the calibration data

# specify the headers to use (case sensitive!)
plot_ID = 'plot_ID'
plot_AGB = 'plot_AGB'
metric_1 = 'ARA4' # lidar variable #1
metric_2 = 'ElevP20' # lidar variable #2
metric_3 = 'PFRAM' # lidar variable #3

# raster data to be loaded in
path2raster = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/lidar/variables/' # specify path of raster data
lidar_raster_1 = 'ARA4_%s.tif' % site_id # raster with lidar variable #1
lidar_raster_2 = 'ElevP20_%s.tif' % site_id # raster with lidar variable #2
lidar_raster_3 = 'PFRAM_%s.tif' % site_id # raster with lidar variable #3

# specify desired confidence level
confidence_level = 0.95
alpha = 1-confidence_level

if(os.path.isdir(path2output)==False):
    os.mkdir(path2output)

"""
#===============================================================================
PART A: CALIBRATION OF LIDAR MAP
Load calibration data
Fit multivariate linear regression
Calculate prediction interval
#-------------------------------------------------------------------------------
"""
# Open csv file
df = pd.read_csv('%s%s' % (path2field,calibration_data_file))

# multivariate linear regression
formula = "%s ~ %s + %s + %s" % (plot_AGB,metric_1,metric_2,metric_3)
model = ols(formula,df).fit()
print(model.summary())

# calculate mean square error, MSE, for model fit
yhat  = model.predict(df)
MSE = np.sum((y.ravel()-yhat)**2)/(n-p)

# calculate the desired interval for each observation
n_obs = len(df)
serr = np.zeros(n_obs)
x0 = np.ones(n_obs)
x1 = df[metric_1].values
x2 = df[metric_2].values
x3 = df[metric_3].values
X = np.vstack((x0,x1,x2,x3)).T
for ii,Xh in enumerate(X):
    Xh = Xh.reshape(-1,1) # convert 1D array slice to 2D column vector (i.e. one column)
    serr[ii] = np.sqrt(MSE*(Xh.T)@(np.linalg.inv(X.T@X))@(Xh))

df['serr'] = serr.copy()
df['CI_%.2f' % confidence_level] = stats.t.ppf(1-alpha/2,n_obs-p)*serr
df['PI_%.2f' % confidence_level] = stats.t.ppf(1-alpha/2,n_obs-p)*np.sqrt(MSE+serr**2)

"""
#===============================================================================
PART B: PRODUCE MAPS
Apply estimates across pixels with LiDAR data
#-------------------------------------------------------------------------------
"""
# load in raster data
raster_1 = xr.open_rasterio('%s%s' % (path2raster,lidar_raster_1))[0]
raster_2 = xr.open_rasterio('%s%s' % (path2raster,lidar_raster_2))[0]
raster_3 = xr.open_rasterio('%s%s' % (path2raster,lidar_raster_3))[0]
for raster in [raster_1,raster_2,raster_3]:
    raster.values[raster.values<-3e38] = np.nan
mask = np.all((np.isfinite(raster_1.values),np.isfinite(raster_2.values),np.isfinite(raster_3.values)),axis=0)

n = mask.sum()
serr_ = np.zeros(n)
x0_ = np.ones(n)
x1_ = raster_1.values[mask]
x2_ = raster_2.values[mask]
x3_ = raster_3.values[mask]
X_ = np.vstack((x0_,x1_,x2_,x3_)).T
for ii,Xh in enumerate(X_):
    Xh = Xh.reshape(-1,1) # convert 1D array slice to 2D column vector (i.e. one column)
    serr_[ii] = np.sqrt(MSE*(Xh.T)@(np.linalg.inv(X.T@X))@(Xh))

# agb estimate
df_ = pandas.DataFrame({metric_1 : x1_, metric_2 : x2_, metric_3 : x3_})
agb_array = io.copy_xarray_template(raster_1)
agb_array[~mask]=np.nan
agb_array.values[mask] = model.predict(df_)

# confidence interval
CI_array = io.copy_xarray_template(raster_1)
CI_array[~mask]=np.nan
CI_array.values[mask] = stats.t.ppf(1-alpha/2,n_obs-p)*serr_
CI_array.values /= agb_array.values # normalise by estimated agb

# prediction interval
PI_array = io.copy_xarray_template(raster_1)
PI_array[~mask]=np.nan
PI_array.values[mask] = stats.t.ppf(1-alpha/2,n_obs-p)*np.sqrt(MSE+serr_**2)
PI_array.values /= agb_array.values # normalise by estimated agb

"""
#===============================================================================
PART C: PLOT LIDAR AGB AND UNCERTAINTY ESTIMATE AND WRITE TO GEOTIFF
#-------------------------------------------------------------------------------
"""
figure_name = '%s%s_lidar_agb_prediction_interval.png' % (path2fig,site_id,version)
figsize = (8,6)
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, subplot_kw = {'projection':crs})
mplt.plot_xarray_to_axis(agb_array, axes[0], vmin=0, vmax=250,
                    add_colorbar=True, cbar_kwargs={'label': 'AGB / Mg ha$^{-1}$',
                    'orientation':'horizontal'}, subplot_kw = {'projection':crs})

mplt.plot_xarray_to_axis(PI_array, axes[1], vmin=0, vmax=1,
                    add_colorbar=True, cbar_kwargs={'label': 'Prediction interval / %',
                    'orientation':'horizontal'}, subplot_kw = {'projection':crs})

# write geotiff
outfile_prefix = '%s%s_lidar_agb' % (path2fig,site_id,version)
io.write_xarray_to_GeoTiff(agb_array,outfile_prefix)
outfile_prefix = '%s%s_lidar_agb_confidence_interval' % (path2output,site_id)
io.write_xarray_to_GeoTiff(CI_array,outfile_prefix)
outfile_prefix = '%s%s_lidar_agb_prediction_interval' % (path2output,site_id)
io.write_xarray_to_GeoTiff(PI_array,outfile_prefix)
