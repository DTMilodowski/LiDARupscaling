"""
regrid_satellite_data_1ha.py
--------------------------------------------------------------------------------
Regridding datasets to match the 1ha reference grid
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
path2fig= '../figures/'
if(os.path.isdir(path2fig)==False):
    os.mkdir(path2fig)
path2sentinel = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/sentinel/processed/'
path2alos = '/home/dmilodow/DataStore_DTM/FOREST2020/LiDARupscaling/data/alos_data/'
path2topo = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/topo/'
path2final = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/satellite_1ha/'

os.mkdir(path2final)

# Extent
"""
N = 2231870.281
S = 2170879.873
E =  262713.933
W = 201723.525
xres = 9.998427514012533
yres = -9.998427514012489
"""
N = 2230310
S = 2171030
E = 263207
W = 197967
xres=100
yres=-100


"""
#===============================================================================
ALOS - use average
- regrid to spatial extent and resolution required
#-------------------------------------------------------------------------------
"""
alos_layers = glob.glob('%s*tif' % path2alos)
for ii,layer in enumerate(alos_layers):
    variable= layer.split('/')[-1][:-4]
    os.system("gdalwarp -overwrite -dstnodata -9999 -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
                -te %f %f %f %f -tr %f %f -r average \
                %s %s%s_alos_%s_1ha.tif" % (W,S,E,N,xres,yres,layer, path2final, site_id, variable))

"""
#===============================================================================
SENTINEL - use average
- regrid to spatial extent and resolution required
#-------------------------------------------------------------------------------
"""
sentinel_layers = glob.glob('%s*tif' % path2sentinel)
for ii,layer in enumerate(sentinel_layers):
    variable= layer.split('/')[-1][:-4]
    os.system("gdalwarp -overwrite -dstnodata -9999 -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
                -te %f %f %f %f -tr %f %f -r average \
                %s %s%s_1ha.tif" % (W,S,E,N,xres,yres,layer, path2final, variable))

"""
#===============================================================================
TOPOGRAPHY - use average, except aspect - use
- regrid to spatial extent and resolution required
#-------------------------------------------------------------------------------
os.system("gdalwarp -overwrite -r cubicspline -dstnodata -9999 -of ENVI\
            -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
            /exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/topo/kiuic_dem.tif \
            /exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/topo/temp/kiuic_dem.bil")
"""

topo_layers = glob.glob('%s*tif' % path2topo)
for ii,layer in enumerate(topo_layers):
    variable= layer.split('/')[-1][:-4]
    os.system("gdalwarp -overwrite -dstnodata -9999 -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
                -te %f %f %f %f -tr %f %f -r average \
                %s %s%s_1ha.tif" % (W,S,E,N,xres,yres,layer, path2final, variable))
