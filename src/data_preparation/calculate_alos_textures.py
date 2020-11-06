"""
calculate_alos_textures.py
--------------------------------------------------------------------------------
Calculate textures for the alos bands at different resolutions:
- mean
- variance
- gclm metrics
--------------------------------------------------------------------------------
"""
import os
import sys
import glob

import numpy as np
import xarray as xr
import skimage.feature.texture as tex

sys.path.append('../data_io/')
import data_io as io

pi = np.pi

path2alos = '../../data/processed_pre_texture/alos_25m/'
path2textures = '../../data/processed_textures/'

try:
    os.mkdir(path2textures)
    os.mkdir('%s/alos_050m/' % path2textures)
    os.mkdir('%s/alos_100m/' % path2textures)
except:
    print('directory %s exists already' % path2textures)

"""
Specify the extent and range of grid resolution interested in
"""
N = 2230310
S = 2171030
E = 263207
W = 197967

xres=25
yres=-25

target_xres = np.array([50, 100])
target_yres = np.array([-50, -100])

stride_x = (target_xres/xres).astype('int')
stride_y = (target_yres/yres).astype('int')

"""
find all the alos to be considered
"""
alos_layers = glob.glob('%s/*tif' % path2alos)
textures = ['mean','variance','contrast','dissimilarity','homogeneity','asm','correlation']

# loop through alos layers and textures to calculate
for sfile in alos_layers:
    layer = xr.open_rasterio(sfile).sel(band=1)
    rows,cols = layer.shape
    layer.values[layer.values==layer.nodatavals[0]]=np.nan
    layer_min = np.nanmin(layer)
    layer_max = np.nanmax(layer)
    for rr, resolution in enumerate(target_xres):
        print('Processing %s for  %.0f metre grid' % (sfile,resolution))
        target_x = np.arange(0,cols,stride_x[rr])*xres + W + target_xres[rr]/2
        target_y = np.arange(0,rows,stride_y[rr])*yres + E + target_yres[rr]/2

        target_rows = np.arange(0,rows,stride_y[rr]).size
        target_cols = np.arange(0,cols,stride_x[rr]).size

        mean = xr.DataArray(data=np.zeros((target_rows,target_cols))*np.nan,
                            coords={'x':target_x, 'y':target_y}, dims=['y', 'x'])
        variance = xr.DataArray(data=np.zeros((target_rows,target_cols))*np.nan,
                        coords={'x':target_x, 'y':target_y}, dims=['y', 'x'])
        contrast = xr.DataArray(data=np.zeros((target_rows,target_cols))*np.nan,
                        coords={'x':target_x, 'y':target_y}, dims=['y', 'x'])
        dissimilarity = xr.DataArray(data=np.zeros((target_rows,target_cols))*np.nan,
                        coords={'x':target_x, 'y':target_y}, dims=['y', 'x'])
        homogeneity = xr.DataArray(data=np.zeros((target_rows,target_cols))*np.nan,
                        coords={'x':target_x, 'y':target_y}, dims=['y', 'x'])
        correlation = xr.DataArray(data=np.zeros((target_rows,target_cols))*np.nan,
                        coords={'x':target_x, 'y':target_y}, dims=['y', 'x'])
        asm = xr.DataArray(data=np.zeros((target_rows,target_cols))*np.nan,
                        coords={'x':target_x, 'y':target_y}, dims=['y', 'x'])

        for ii, i_row in enumerate(np.arange(0,rows,stride_y[rr])):
            if i_row%50 == 0:
                print("%.1f percent" % (float(i_row)/float(rows)*100),end='\r')
            for jj, i_col in enumerate(np.arange(0,cols,stride_x[rr])):
                # clip the raster subset for further calculations
                subset = layer.values[i_row:i_row+stride_y[rr],i_col:i_col+stride_x[rr]]
                if np.isnan(subset).sum()==0:
                    mean.values[ii,jj] = np.mean(subset)
                    variance.values[ii,jj] = np.var(subset)

                    subset_scaled = 255*(subset-layer_min)/(layer_max-layer_min)
                    glcm = tex.greycomatrix(subset_scaled.astype('int'), [1],
                                [0, pi/4, pi/2, pi*3/4], levels=256)
                    contrast.values[ii,jj]=tex.greycoprops(glcm, 'contrast')[0].mean()
                    dissimilarity.values[ii,jj]=tex.greycoprops(glcm, 'dissimilarity')[0].mean()
                    homogeneity.values[ii,jj]=tex.greycoprops(glcm, 'homogeneity')[0].mean()
                    correlation.values[ii,jj]=tex.greycoprops(glcm, 'correlation')[0].mean()
                    asm.values[ii,jj]=tex.greycoprops(glcm, 'ASM')[0].mean()

        # write array to new geotiff
        prefix = sfile.split('/')[-1][:-8]
        res_str = str(resolution).zfill(3)
        io.write_xarray_to_GeoTiff(mean,'%s/alos_%sm/%s_%sm_mean.tif' %
                                    (path2textures,res_str,prefix,res_str))
        io.write_xarray_to_GeoTiff(variance,'%s/alos_%sm/%s_%sm_variance.tif' %
                                    (path2textures,res_str,prefix,res_str))
        io.write_xarray_to_GeoTiff(contrast,'%s/alos_%sm/%s_%sm_contrast.tif' %
                                    (path2textures,res_str,prefix,res_str))
        io.write_xarray_to_GeoTiff(dissimilarity,'%s/alos_%sm/%s_%sm_dissimilarity.tif' %
                                    (path2textures,res_str,prefix,res_str))
        io.write_xarray_to_GeoTiff(homogeneity,'%s/alos_%sm/%s_%sm_homogeneity.tif' %
                                    (path2textures,res_str,prefix,res_str))
        io.write_xarray_to_GeoTiff(correlation,'%s/alos_%sm/%s_%sm_correlation.tif' %
                                    (path2textures,res_str,prefix,res_str))
        io.write_xarray_to_GeoTiff(asm,'%s/alos_%sm/%s_%sm_ASM.tif' %
                                    (path2textures,res_str,prefix,res_str))
