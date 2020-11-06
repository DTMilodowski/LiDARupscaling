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
import os

"""
Project Info
"""
site_id = 'kiuic'
path2fig= '../figures/'
if(os.path.isdir(path2fig)==False):
    os.mkdir(path2fig)
path2data = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/'
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
- regrid to spatial extent and resolution required
#-------------------------------------------------------------------------------
"""
os.system("gdalwarp -overwrite -dstnodata -9999 -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs'  -te %f %f %f %f -tr %f %f -r bilinear %s%s_HH_HV_Enlee_clip.tif %s%s_HH_HV_Enlee_regridded.tif" % (W,S,E,N,xres,yres,path2data, site_id, path2data, site_id))
os.system("gdal_translate -b 1 %s%s_HH_HV_Enlee_regridded.tif %s%s_ALOS_HH_Enlee.tif" % (path2data, site_id, path2final, site_id))
os.system("gdal_translate -b 2 %s%s_HH_HV_Enlee_regridded.tif %s%s_ALOS_HV_Enlee.tif" % (path2data, site_id, path2final, site_id))
