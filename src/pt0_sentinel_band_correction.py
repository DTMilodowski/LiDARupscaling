"""
pt1_lidar_sentinel_fusion_train_rf.py
--------------------------------------------------------------------------------
FITTING RANDOM FOREST MODEL TO LINK SENTINEL LAYERS TO LIDAR ESTIMATED AGB
This script loads the predictor (sentinel bands and derivatives) and target
(lidar estimated AGB) variables, calibrates and validates a random forest
regression model, and fits a final model using te full training set.

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
import os

# Import some parts of the scikit-learn library
from sklearn.linear_model import LinearRegression

# Import custom libaries
import sys
sys.path.append('./data_io/')

import data_io as io

"""
Project Info
"""
site_id = 'kiuic'
path2fig= '../figures/'
if(os.path.isdir(path2fig)==False):
    os.mkdir(path2fig)
path2clipped = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/sentinel/band_correction/overlap/'
path2full = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/sentinel/band_correction/full_extent/'
path2processed = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/sentinel/band_correction/processed/'

"""
#===============================================================================
For each band
- load clipped scenes where overlapping
- linear regression
- load full scene for tile 2
- apply linear model to tile 2 to match tile 1
#-------------------------------------------------------------------------------
"""
for bb in range(0,4):
    band = bb+1
    clip1=xr.open_rasterio('%s%s1_overlap_b%i.tif' % (path2clipped,site_id,band))
    clip2=xr.open_rasterio('%s%s2_overlap_b%i.tif' % (path2clipped,site_id,band))
    X = clip2.values.reshape(clip2.values.size,1)
    y = clip1.values.reshape(clip1.values.size)
    mask = np.all((np.isfinite(clip1.values.ravel()),np.isfinite(clip2.values.ravel())),axis=0)
    X=X[mask]
    y=y[mask]
    lm = LinearRegression()
    lm.fit(X,y)
    print("calibration score: %.02f" % lm.score(X,y))

    full2=xr.open_rasterio('%s%s2_band%i.tif' % (path2full,site_id,band))[0]
    mask = np.isfinite(full2.values)
    X = full2.values[mask].reshape(mask.sum(),1)

    full2_new = io.copy_xarray_template(full2)
    full2_new.values[mask]=lm.predict(X)

    outfile_prefix = ('%s%s2_band%i' % (path2processed,site_id,band))
    io.write_xarray_to_GeoTiff(full2_new,outfile_prefix)
    os.system('cp %s%s1_band%i.tif %s%s1_band%i.tif' % (path2full,site_id,band,path2processed,site_id,band))
