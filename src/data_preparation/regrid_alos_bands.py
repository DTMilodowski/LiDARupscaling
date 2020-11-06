"""
regrid_alos_bands.py
--------------------------------------------------------------------------------
Regridding sentinel bands for the region of interest

Two resolutions used:
    (1) 20m, for integration into finest lengthscale analysis. Uses the average
        resampling approach
    (2) 25m, for integration into the 50m and 100m lengthscale analysis. Uses
        the nearest neighbour interpolation
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
path2alos = '/home/dmilodow/DataStore_DTM/FOREST2020/LiDARupscaling/data/alos_data/'
path2final1 =  '../../data/processed_pre_texture/alos_20m/'
path2final2 =  '../../data/processed_pre_texture/alos_25m/'

os.mkdir(path2final1)
os.mkdir(path2final2)

# Extent
N = 2230310
S = 2171030
E = 263207
W = 197967
xres=25
yres=-25
xres20=20
yres20=-20

"""
#===============================================================================
ALOS - use average for 20m and nearest neighbour for 25m
- regrid to spatial extent and resolution required
#-------------------------------------------------------------------------------
"""
os.system('gdal_calc.py --calc="A/B" --outfile=%s/kiuic_HH_HV_ratio_Enlee.tif\
            -A %skiuic_HH_Enlee.tif \
            -B %skiuic_HV_Enlee.tif' % (path2alos,path2alos,path2alos))

alos_layers = glob.glob('%s*tif' % path2alos)

for ii,layer in enumerate(alos_layers):
    variable= layer.split('/')[-1][:-4]
    if variable == 'kiuic_HH_Enlee':
        os.system("gdalwarp -overwrite -dstnodata -9999 \
                -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
                -te %f %f %f %f -tr %f %f -r average \
                %s %s/%s_alos_hh_enlee_20m.tif" % (W,S,E,N,xres20,yres20,layer, path2final1, site_id))
    elif variable == 'kiuic_HV_Enlee':
        os.system("gdalwarp -overwrite -dstnodata -9999 \
                -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
                -te %f %f %f %f -tr %f %f -r average \
                %s %s/%s_alos_hv_enlee_20m.tif" % (W,S,E,N,xres20,yres20,layer, path2final1, site_id))
    else:
        os.system("gdalwarp -overwrite -dstnodata -9999 \
                -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
                -te %f %f %f %f -tr %f %f -r average \
                %s %s/%s_alos_hh_hv_ratio_enlee_20m.tif" % (W,S,E,N,xres20,yres20,layer, path2final1, site_id))

for ii,layer in enumerate(alos_layers):
    variable= layer.split('/')[-1][:-4]
    if variable == 'kiuic_HH_Enlee':
        os.system("gdalwarp -overwrite -dstnodata -9999\
                -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
                -te %f %f %f %f -tr %f %f -r near \
                %s %s/%s_alos_hh_enlee_25m.tif" % (W,S,E,N,xres,yres,layer, path2final2, site_id))
    elif variable == 'kiuic_HV_Enlee':
        os.system("gdalwarp -overwrite -dstnodata -9999 \
                -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
                -te %f %f %f %f -tr %f %f -r near \
                %s %s/%s_alos_hv_enlee_25m.tif" % (W,S,E,N,xres,yres,layer, path2final2, site_id))
    else:
        os.system("gdalwarp -overwrite -dstnodata -9999 \
                -t_srs '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs' \
                -te %f %f %f %f -tr %f %f -r near \
                %s %s/%s_alos_hh_hv_ratio_enlee_25m.tif" % (W,S,E,N,xres,yres,layer, path2final2, site_id))
