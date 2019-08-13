"""
pt2_lidar_sentinel_upscaling_rfbc.py
--------------------------------------------------------------------------------
UPSCALING AGB ESTIMATES USING FITTED RF REGRESSION MODEL INCLUDING BIAS
CORRECTION
This script loads the predictor (sentinel bands and derivatives) and target
(lidar estimated AGB) variables, and uses a fitted random forest regression
model (from pt1) to map AGB across the Sentinel 2 scene.

This code built using the open source programming language python, and utilises
the geospatial library xarray (http://xarray.pydata.org/en/stable/) and the
machine learning library scikit-learn
(https://scikit-learn.org/stable/index.html).

13/08/2019 - D. T. Milodowski
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

# Import some parts of the scikit-learn library
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

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
version = '010'
crs = ccrs.UTM('16N')
path2alg = '../saved_models/'
path2fig= '../figures/'
path2data = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/'
path2output = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/output/'
if(os.path.isdir(path2output)==False):
    os.mkdir(path2output)

"""
#===============================================================================
PART A: UPSCALING
Load predictor and target variables
Load fitted rf models
Apply fitted model across Sentinel scene
#-------------------------------------------------------------------------------
"""
# Load predictors & target
predictors,target,landmask,labels=io.load_predictors()

# Load rf models
rf_dict = joblib.load('%s%s_%s_rfbc_sentinel_lidar_agb_bayes_opt.pkl' % (path2alg,site_id,version))
rf1=rf_dict['rf1']
rf2=rf_dict['rf2']
# Now the model has been fitted, we will predict the potential AGB across the
# full dataset
agb_mod = rff.rfbc_predict(rf1,rf2,predictors)

# Now lets plot this onto a map
# We'll load in an existing dataset to get the georeferencing information
template_file = '%s/lidar/processed/%s_AGB_07-31-19_regridded.tif' % (path2data,site_id)
template = io.load_geotiff(template_file,option=1)

#let's copy to a new xarray for AGBpot
agb = io.copy_xarray_template(template)
agb.values[landmask] = agb_mod.copy()
agb.values[agb.values==-9999]=np.nan
agb.values[agb.values<0]=0

"""
#===============================================================================
PART B: PLOT UPSCALED AGB ESTIMATE AND WRITE TO GEOTIFF
#-------------------------------------------------------------------------------
"""
figure_name = '%s%s_%s_agb_upscaled.png' % (path2fig,site_id,version)
fig1,axis = mplt.plot_xarray(agb, figure_name = figure_name,figsize_x=6,figsize_y=8,
                    vmin=0,vmax=250,add_colorbar=True,
                    cbar_kwargs={'label': 'AGB$_{def}$ / Mg ha$^{-1}$',
                    'orientation':'horizontal'}, subplot_kw = {'projection':crs})

outfile_prefix = '%s%s_%s_rf_agb_upscaled' % (path2output,site_id,version)
io.write_xarray_to_GeoTiff(agb,outfile_prefix)
