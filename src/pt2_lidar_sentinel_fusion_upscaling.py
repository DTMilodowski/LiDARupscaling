"""
pt2_lidar_sentinel_upscaling.py
--------------------------------------------------------------------------------
UPSCALING AGB ESTIMATES USING FITTED RF REGRESSION MODEL
This script loads the predictor (sentinel bands and derivatives) and target
(lidar estimated AGB) variables, and uses a fitted random forest regression
model (from pt1) to map AGB across the Sentinel 2 scene.

This code built using the open source programming language python, and utilises
the geospatial library xarray (http://xarray.pydata.org/en/stable/) and the
machine learning library scikit-learn
(https://scikit-learn.org/stable/index.html).

24/04/2019 - D. T. Milodowski
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

import data_io as io
import map_plots as mplt

"""
Project Info
"""
site_id = 'kiuic'
version = '002'
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
Load fitted rf model
Apply fitted model across Sentinel scene
#-------------------------------------------------------------------------------
"""
# Load predictors & target
predictors,target,landmask,labels=io.load_predictors()

# Load rf model
rf = joblib.load('%s%s_%s_rf_sentinel_lidar_agb.pkl' % (path2alg,site_id,version))

# Now the model has been fitted, we will predict the potential AGB across the
# full dataset
agb_mod = rf.predict(predictors)

# Now lets plot this onto a map
# We'll load in an existing dataset to get the georeferencing information
template_file = '%s/lidar/processed/%s_lidar_agb_regridded.tif' % (path2data,site_id)
template = io.load_geotiff(template_file,option=1)

#let's copy to a new xarray for AGBpot
agb = io.copy_xarray_template(template)
agb.values[landmask] = agb_mod.copy()
agb.values[agb.values==-9999]=np.nan

"""
#===============================================================================
PART B: PLOT UPSCALED AGB ESTIMATE AND WRITE TO GEOTIFF
#-------------------------------------------------------------------------------
"""
figure_name = '%s%s_%s_agb_upscaled.png' % (path2fig,site_id,version)
fig1,axis = mplt.plot_xarray(agb, figure_name = figure_name,vmin=0,vmax=250,
                    add_colorbar=True,
                    cbar_kwargs={'label': 'AGB$_{def}$ / Mg ha$^{-1}$',
                    'orientation':'horizontal'}, subplot_kw = {'projection':crs})

outfile_prefix = '%s%s_%s_rf_agb_upscaled' % (path2output,site_id,version)
io.write_xarray_to_GeoTiff(agb,outfile_prefix)
