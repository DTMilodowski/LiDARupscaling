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
path2final = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/sentinel/processed/'

# Extent
N = 2231870.281
S = 2170879.873
E =  262713.933
W = 201723.525
xres = 9.998427514012533
yres = -9.998427514012489

"""
#===============================================================================
For each band
- load clipped scenes where overlapping
- linear regression
- load full scene for tile 2
- apply linear model to tile 2 to match tile 1
- merge tiles
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
    full2_new.values[mask][full2_new.values[mask]<0]=0

    outfile_prefix = ('%s%s2_band%i_corrected' % (path2processed,site_id,band))
    io.write_xarray_to_GeoTiff(full2_new,outfile_prefix)

    # use gdal to merge and warp to extent
    os.system("gdal_merge.py -a_nodata -9999 -ot float32 -o %s%s_b%s_temp.tif %s%s2_band%i_corrected.tif %s%s1_band%i.tif" % (path2final, site_id, band, path2processed, site_id, band, path2full, site_id, band))
    os.system("gdalwarp -overwrite -te %f %f %f %f -tr %f %f -r bilinear %s%s_b%i_temp.tif %s%s_b%i_merge.tif" % (W,S,E,N,xres,yres,path2final, site_id, band,path2final, site_id, band))
    os.system("rm %s%s_b%i_temp.tif" % (path2final, site_id, band))
    os.system("chmod +777 %s%s_b%i_merge.tif" % (path2final, site_id, band))
