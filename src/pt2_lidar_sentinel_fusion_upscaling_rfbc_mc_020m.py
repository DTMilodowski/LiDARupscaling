"""
pt2_lidar_sentinel_upscaling_rfbc.py
--------------------------------------------------------------------------------
UPSCALING AGB ESTIMATES USING FITTED RF REGRESSION MODEL INCLUDING BIAS
CORRECTION
This script loads the predictor (sentinel bands and derivatives) and target
(lidar estimated AGB) variables, and uses a fitted random forest regression
model (from pt1) to map AGB across the Sentinel 2 scene. This is a continuation
of the Monte Carlo upscaling process, with 100 LiDAR AGB models used to
calibrate 100 upscaled maps, providing a means to estimate confidence intervals.

Upscaled maps are plotted as a figure and also written to a geotiff

This code built using the open source programming language python, and utilises
the geospatial library xarray (http://xarray.pydata.org/en/stable/) and the
machine learning library scikit-learn
(https://scikit-learn.org/stable/index.html).

12/08/2020 - D. T. Milodowski
--------------------------------------------------------------------------------
"""

"""
# Import the necessary packages
"""
import numpy as np                  # standard package for scientific computing
import xarray as xr                 # xarray geospatial package
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package
import os
import glob
from scipy import ndimage as image

# Import some parts of the scikit-learn library
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import pickle

# Import cartographic projection library
import cartopy.crs as ccrs

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
version = '034'
crs = ccrs.UTM('16N')
path2alg = '../saved_models/'
path2fig= '../figures/'
path2data = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/'
path2lidar = '/exports/csce/datastore/geos/users/dmilodow/FOREST2020/LiDARupscaling/data/lidar_calibration/agb_mc_20m/'
path2output = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/output/'
if(os.path.isdir(path2output)==False):
    os.mkdir(path2output)

"""
#===============================================================================
PART A: SETTING UP
Load predictor and target variables
Load fitted rf models
#-------------------------------------------------------------------------------
"""
# AGB files
agb_list= glob.glob('%s*%s*mc_*.tif' % (path2lidar,version))
target = xr.open_rasterio(agb_list[0]).values[0]
target[target<0]=np.nan

# Load predictors & target
data_layers,data_mask,labels = io.load_predictors(layers=['sentinel2','alos'])
n_predictors = data_layers.shape[0]
print(labels)

# load forest mask
forest_mask_file = "/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/forest_mask/%s_forest_mask_20m.tif" % site_id
forest = xr.open_rasterio(forest_mask_file).values[0]
forest_mask=forest==1
forest_mask = forest_mask*data_mask

# Keep only areas for which we have biomass estimates
training_mask = np.isfinite(target)
training_mask = image.binary_erosion(training_mask,iterations=1)
training_mask = training_mask*forest_mask

# Apply masks to the predictor dataset to be ingested into sklearn routines
predictors = io.apply_mask_to_raster_stack(data_layers,forest_mask)
X = io.apply_mask_to_raster_stack(data_layers,training_mask)
"""
# PCA analysis to reduce dimensionality of predictor variables
pca = make_pipeline(StandardScaler(),PCA(n_components=0.999))
pca.fit(predictors)
X = pca.transform(predictors[mask,:])
"""

# load the trials data
trials = pickle.load(open('%s%s_%s_rfbc_sentinel_lidar_agb_trials.p' % (path2alg,site_id,version), "rb"))

# double check the number of accepted parameter sets
best_score = np.inf
for ii,tt in enumerate(trials.trials):
    if tt['result']['status']=='ok':
        if tt['result']['loss'] < best_score:
            best_score = tt['result']['loss']
            best_params=tt['misc']['vals']

"""
#===============================================================================
PART B: MONTECARLO MODEL FITTING
Fit RF model for 100 AGB maps
Save RFs for future reference
#-------------------------------------------------------------------------------
"""
N_iter = len(agb_list)
for ii, agb_file in enumerate(agb_list):
    print('Iteration %i of %i' % (ii+1,N_iter))
    target = xr.open_rasterio(agb_file).values[0]
    y = target[training_mask]

    rf = RandomForestRegressor(bootstrap=True,
                criterion='mse',           # criteria used to choose split point at each node
                max_depth= int(best_params['max_depth'][0]),            # ***maximum number of branching levels within each tree
                max_features=int(best_params['max_features'][0]),       # ***the maximum number of variables used in a given tree
                max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
                min_impurity_split=None,   # threshold impurity within an internal node before it will be split
                min_samples_leaf=int(best_params['min_samples_leaf'][0]),       # ***The minimum number of samples required to be at a leaf node
                min_samples_split=int(best_params['min_samples_split'][0]),       # ***The minimum number of samples required to split an internal node
                n_estimators=int(best_params['n_estimators'][0]), # ***Number of trees in the random forest
                n_jobs=20,                 # The number of jobs to run in parallel for both fit and predict
                oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
                random_state=29+ii,         # seed used by the random number generator
                )

    rf1,rf2 = rff.rfbc_fit(rf,X,y)
    # Save random forest model for future use
    rf_dict = {}
    rf_dict['rf1']=rf1
    rf_dict['rf2']=rf2
    joblib.dump(rf_dict,'%s%s_%s_optimised_rfbc_sentinel_alos_lidar_%s.pkl' % (path2alg,site_id,version,str(ii+1).zfill(3)))
X=None

"""
#===============================================================================
PART C: MONTECARLO UPSCALING
Fit RF model for 100 AGB maps
Save RFs for future reference
#-------------------------------------------------------------------------------
"""
# We'll load in an existing dataset to get the georeferencing information
template = io.load_geotiff(agb_list[0],option=1)
rows,cols=template.shape
agb_stack = np.zeros((N_iter,rows,cols))
#pca_predictors = pca.transform(predictors)
#predictors=None
for ii, agb_file in enumerate(agb_list):
    print('Iteration %i of %i' % (ii+1,N_iter))
    rf_dict = joblib.load('%s%s_%s_optimised_rfbc_sentinel_alos_lidar_%s.pkl' % (path2alg,site_id,version,str(ii+1).zfill(3)))
    agb_mod = rff.rfbc_predict(rf_dict['rf1'],rf_dict['rf2'],predictors)

    #let's copy to a new xarray for AGBpot
    agb = io.copy_xarray_template(template)
    agb.values[forest_mask] = agb_mod.copy()
    agb.values[agb.values==-9999]=np.nan
    agb.values[agb.values<0]=0

    outfile_prefix = '%s%s_%s_rfbc_agb_upscaled_%s' % (path2output,site_id,version,str(ii+1).zfill(3))
    io.write_xarray_to_GeoTiff(agb,outfile_prefix)

    agb_stack[ii] = agb.values

# summary arrays
agb_med = io.copy_xarray_template(template)
agb_med.values = np.median(agb_stack,axis=0)
agb_med.values[agb_med.values==-9999]=np.nan
outfile_prefix = '%s%s_%s_rfbc_agb_upscaled_median' % (path2output,site_id,version)
io.write_xarray_to_GeoTiff(agb_med,outfile_prefix)

agb_upper = io.copy_xarray_template(template)
agb_upper.values = np.percentile(agb_stack,97.5,axis=0)
agb_upper.values[agb_upper.values==-9999]=np.nan
outfile_prefix = '%s%s_%s_rfbc_agb_upscaled_upper' % (path2output,site_id,version)
io.write_xarray_to_GeoTiff(agb_upper,outfile_prefix)

agb_lower = io.copy_xarray_template(template)
agb_lower.values = np.percentile(agb_stack,2.5,axis=0)
agb_lower.values[agb_lower.values==-9999]=np.nan
outfile_prefix = '%s%s_%s_rfbc_agb_upscaled_lower' % (path2output,site_id,version)
io.write_xarray_to_GeoTiff(agb_lower,outfile_prefix)

"""
#===============================================================================
PART D: PLOT UPSCALED AGB ESTIMATE AND WRITE TO GEOTIFF
#-------------------------------------------------------------------------------
"""
agb_med.values[agb_med.values==-9999]=np.nan
agb_lower.values[agb_lower.values==-9999]=np.nan
agb_upper.values[agb_upper.values==-9999]=np.nan
plt.rcParams["axes.axisbelow"] = False
figure_name = '%s%s_%s_agb_upscaled.png' % (path2fig,site_id,version)
fig1,axes = plt.subplots(nrows=1,ncols=2,figsize=(9,6),sharex='all',sharey='all')
mplt.plot_xarray_to_axis(agb_med,axes[0],
                    vmin=0,vmax=200,add_colorbar=True,
                    cbar_kwargs={'label': 'AGB / Mg ha$^{-1}$',
                    'orientation':'horizontal'})
mplt.plot_xarray_to_axis(agb_upper-agb_lower,axes[1],
                    vmin=0,vmax=60,add_colorbar=True,
                    cbar_kwargs={'label': 'Uncertainty in AGB / Mg ha$^{-1}$',
                    'orientation':'horizontal'})

for ax in axes:
    ax.grid(True,which='both')
    ax.set_xlabel('Easting / m')
    ax.set_xticks(np.arange(200000., 263197., 20000))
    ax.set_xticks(np.arange(200000., 263197., 10000), minor=True)
    ax.set_yticks(np.arange(2180000., 2230300., 20000))
    ax.set_yticks(np.arange(2170000., 2230300., 10000), minor=True)

axes[0].set_ylabel('Northing / m')
axes[1].set_ylabel('')
axes[0].ticklabel_format(useOffset=False,style='plain')
fig1.tight_layout()
fig1.show()
fig1.savefig(figure_name)
