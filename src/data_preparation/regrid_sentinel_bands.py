"""
regrid_sentinel_bands.py
--------------------------------------------------------------------------------
Regridding sentinel bands to 10m resolution for the region of interest, using
nearest neighbour interpolation to preserve raster band values
--------------------------------------------------------------------------------
"""

"""
# Import the necessary packages
"""
import os
import glob

"""
Project Info
"""
site_id = 'kiuic'
path2sentinel = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/sentinel/processed/'
path2final = '../../data/processed_pre_texture/sentinel_10m/'

os.mkdir(path2final)

# Extent
N = 2230310
S = 2171030
E = 263207
W = 197967
xres=10
yres=-10
xres20=20
yres20=-20

"""
#===============================================================================
SENTINEL - use nearest neighbour @10m resolution
- regrid to spatial extent and resolution required
#-------------------------------------------------------------------------------
"""
"""
sentinel_layers = glob.glob('%s*_merge.tif' % path2sentinel)+glob.glob('%s*_ndvi.tif' % path2sentinel)
for ii,layer in enumerate(sentinel_layers):
    variable= layer.split('/')[-1][:-4]
    os.system("gdalwarp -overwrite -dstnodata -9999 \
                -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
                -te %f %f %f %f -tr %f %f -r near \
                %s %s/%s_10m.tif" % (W,S,E,N,xres,yres,layer, path2final, variable))
"""
"""
#===============================================================================
FOREST MASK - use mode
- regrid to spatial extent and resolution required
#-------------------------------------------------------------------------------
"""
os.system("gdalwarp -overwrite -dstnodata -9999 -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
                -te %f %f %f %f -tr %f %f -r mode -ot Float32\
                /exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/forest_mask/kiuic_10_regridded.tif \
                /exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/forest_mask/kiuic_forest_mask_20m.tif" % (W,S,E,N,xres20,yres20))

"""
#===============================================================================
FOREST MANAGEMENT CLASSES
#-------------------------------------------------------------------------------
"""
os.system("gdal_rasterize -a_nodata -9999 -a OBJECTID_1 -of ENVI\
            -a_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
            -te %f %f %f %f -tr %f %f  \
            /home/dmilodow/DataStore_DTM/FOREST2020/LiDARupscaling/data/State_PA_UTM.shp \
            /home/dmilodow/DataStore_DTM/FOREST2020/LiDARupscaling/data/State_PA_UTM.tif" % (W,S,E,N,xres20,yres20))
os.system("gdal_rasterize -a_nodata -9999 -a cat_id -of ENVI\
            -a_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
            -te %f %f %f %f -tr %f %f  \
            /home/dmilodow/DataStore_DTM/FOREST2020/LiDARupscaling/data/National_PA_UTM.shp \
            /home/dmilodow/DataStore_DTM/FOREST2020/LiDARupscaling/data/National_PA_UTM.tif" % (W,S,E,N,xres20,yres20))
