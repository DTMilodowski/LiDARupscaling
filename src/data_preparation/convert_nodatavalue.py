"""
#-----------------------------
# apply_nodata_buffer.py
#=============================
# D. T. Milodowski, 18/04/2019
#-----------------------------
# This function applies a nodata buffer around the data within a tiff file,
# and writes to a new geotiff
#-----------------------------
"""

"""
Import libraries
"""
import numpy as np
import xarray as xr
from scipy import ndimage as nd
import glob as glob
import sys
sys.path.append('./data_io')
import data_io as io

"""
List some file paths
"""
path2files = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/uoe_lidar_sentinel/agb_Lidar_maps/artodata/'
path2output = '/home/dmilodow/DataStore_GCEL/YucatanBiomass/uoe_lidar_sentinel/agb_Lidar_maps/artodata_nodata_corr/'

"""
Loop through the data files, and apply buffer in each case
"""
ulim=500
files = glob.glob('%s*.tif' % path2files)
count=0
for f in files:
    count+=1
    print('\r%i/%i' % (count,len(files)))
    ds = xr.open_rasterio(f)
    mask = ds.values[0]<0
    ds_new = ds.copy()
    ds_new.values[0][mask] = -9999
    ds_new.values[ds.values>ulim] = -9999
    io.write_xarray_to_GeoTiff(ds_new.sel(band=1),'%s%s' % (path2output,f.split('/')[-1].split('.')[0]))
