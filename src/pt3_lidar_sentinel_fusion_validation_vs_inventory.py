"""
pt3_lidar_sentinel_fusion_validation_vs_inventory.py
--------------------------------------------------------------------------------
VALIDATION OF UPSCALED MAPS VS INVENTORY DATA
This script loads a map of lidar estimated AGB, the upscaled AGB map, and the
field inventory data. It then produces three plots:
i) LiDAR AGB vs. inventory used for training LIDAR map
ii) Upscaled AGB vs. inventory used for training LiDAR map
iii) Upscaled AGB vs. inventory not used for training LiDAR map

The final plot is an independent validation of the upscaled product. The first
two allow an assessment of the relative performance of the upscaled map relative
to the original LiDAR-field calibrated relationship.

This code built using the open source programming language python, and utilises
the geospatial library xarray (http://xarray.pydata.org/en/stable/) and the
machine learning library scikit-learn
(https://scikit-learn.org/stable/index.html).

22/08/2019 - D. T. Milodowski
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
PART A: LOAD DATA
Load the LiDAR AGB map, upscaled AGB map and inventory data
Split inventory data into calibration vs. validation
#-------------------------------------------------------------------------------
"""
# LiDAR AGB
lidar_file = '%s/lidar/processed/%s_AGB_07-31-19_regridded.tif' % (path2data,site_id)
lidar = io.load_geotiff(template_file,option=1)
lidar.values[lidar.values==-9999]=np.nan
lidar.values[lidar.values<0]=0

# Upscaled AGB
upscaled_file = '%s%s_%s_rfbc_agb_upscaled.tif' % (path2output,site_id,version)
upscaled = io.load_geotiff(template_file,option=1)
upscaled.values[upscaled.values==-9999]=np.nan
upscaled.values[upscaled.values<0]=0

# inventory
inventory_file = ''
inventory = ''


"""
#===============================================================================
PART B: COMPARE INVENTORY vs LiDAR/UPSCALED AGB ESTIMATES
#-------------------------------------------------------------------------------
"""
