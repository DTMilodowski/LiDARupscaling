"""
CALIBRATE_LIDAR_AGB.PY
--------------------------------------------------------------------------------
Calibrate LiDAR AGB maps based on field inventory data
D.T. Milodowski
"""
import numpy as np
import xarray as xr
import geospatial_tools as gst
from shapely.geometry import Point
import copy as cp

# CHANGE THIS STUFF
# Some file names - shapefiles and rasters should be in a projected coordinate system (i.e. UTM)
plot_file = 'FILENAME.csv' # this should be a set of point coordinates associated with one plot
raster_file = '../../data/LiDAR_data/gliht_elevmax_dtm.tif'  # this should be the raster dataset that you want to interrogate
dem_file = '../../data/LiDAR_data/gliht_dtm.tif'
outfile = 'sample_test.csv' # this will be a file to hold the output data
plot_area = ???
gap_ht = 2 # height at which to define canopy gaps

# LOAD FILES, CONVERT TO FLOAT AND SPECIFY NODATA REGIONS
dem = xr.open_rasterio(dem_file)[0]
chm = xr.open_rasterio(raster_file)[0]
chm.values = chm.values.astype(float)
chm.values[chm.values<0] = np.nan
chm.values[dem.values==0] = np.nan

# Gaps < 2m
gf = cp.deepcopy(chm)
gf.values[gf.values<=gap_ht]=1
gf.values[gf.values>gap_ht]=0

# A DICTIONARY TO CONTAIN THE RESULTS
chm_results = {}
gap_results = {}

# LOAD THE PLOT DATA FROM FILE
plots = np.genfromtxt(plot_file,dtype=[]) # NEED TO CHECK PLOT COORDINATE FILES
ids = ???
n_plots = len(ids)

# calculate plot radius
radius = np.sqrt(plot_area/np.pi)

# SAMPLE RASTER FOR EACH PLOT NEIGHBOURHOOD
for pp in range(0,len(ids)):
    plot_centre = Point(plots['X'][pp],plots['Y'][pp]]))
    chm_results[ids[pp]] = gst.sample_raster_by_point_neighbourhood(chm,plot_centre,radius,x_dim='x',y_dim='y',label = ids[pp])
    gap_results[ids[pp]] = gst.sample_raster_by_point_neighbourhood(gap_ht,plot_centre,radius,x_dim='x',y_dim='y',label = ids[pp])

# CALIBRATION STATISTICS
