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

"""
useful funtion to copy xarray template
"""
def copy_xarray_template(xarr):
    xarr_new = xarr.copy()
    xarr_new.values = np.zeros(xarr.values.shape)*np.nan
    return xarr_new

def create_geoTrans(array,x_name='x',y_name='y'):
    lat = array.coords[y_name].values
    lon = array.coords[x_name].values
    dlat = lat[1]-lat[0]
    dlon = lon[1]-lon[0]
    geoTrans = [0,dlon,0,0,0,dlat]
    geoTrans[0] = np.min(lon)
    if geoTrans[5]>0:
        geoTrans[3]=np.min(lat)
    else:
        geoTrans[3]=np.max(lat)
    return geoTrans

def check_array_orientation(array,geoTrans,north_up=True):
    if north_up:
        # for north_up array, need the n-s resolution (element 5) to be negative
        if geoTrans[5]>0:
            geoTrans[5]*=-1
            geoTrans[3] = geoTrans[3]-(array.shape[0]+1.)*geoTrans[5]
        # Get array dimensions and flip so that it plots in the correct orientation on GIS platforms
        if len(array.shape) < 2:
            print('array has less than two dimensions! Unable to write to raster')
            sys.exit(1)
        elif len(array.shape) == 2:
            array = np.flipud(array)
        elif len(array.shape) == 3:
            (NRows,NCols,NBands) = array.shape
            for i in range(0,NBands):
                array[:,:,i] = np.flipud(array[:,:,i])
        else:
            print('array has too many dimensions! Unable to write to raster')
            sys.exit(1)

    else:
        # for north_up array, need the n-s resolution (element 5) to be positive
        if geoTrans[5]<0:
            geoTrans[5]*=-1
            geoTrans[3] = geoTrans[3]-(array.shape[0]+1.)*geoTrans[5]
        # Get array dimensions and flip so that it plots in the correct orientation on GIS platforms
        if len(array.shape) < 2:
            print('array has less than two dimensions! Unable to write to raster')
            sys.exit(1)
        elif len(array.shape) == 2:
            array = np.flipud(array)
        elif len(array.shape) == 3:
            (NRows,NCols,NBands) = array.shape
            for i in range(0,NBands):
                array[:,:,i] = np.flipud(array[:,:,i])
        else:
            print ('array has too many dimensions! Unable to write to raster')
            sys.exit(1)

    # Get array dimensions and flip so that it plots in the correct orientation on GIS platforms
    if len(array.shape) < 2:
        print ('array has less than two dimensions! Unable to write to raster')
        sys.exit(1)
    elif len(array.shape) == 2:
        array = np.flipud(array)
    elif len(array.shape) == 3:
        (NRows,NCols,NBands) = array.shape
        for i in range(0,NBands):
            array[:,:,i] = np.flipud(array[:,:,i])
    else:
        print ('array has too many dimensions! Unable to write to raster')
        sys.exit(1)

    return array,geoTrans

def write_xarray_to_GeoTiff(array, OUTFILE_prefix,north_up=True):
    NBands = 1
    NRows,NCols = array.values.shape

    # create geotrans object
    geoTrans = create_geoTrans(array)
    EPSG_CODE = array.attrs['crs'].split(':')[-1]

    # check orientation
    array.values,geoTrans = check_array_orientation(array.values,geoTrans,north_up=north_up)

    # set nodatavalue
    array.values[np.isnan(array.values)] = -9999

    # Write GeoTiff
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    # set all the relevant geospatial information
    dataset = driver.Create( OUTFILE_prefix+'.tif', NCols, NRows, NBands, gdal.GDT_Float32 )
    dataset.SetGeoTransform( geoTrans )
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS( 'EPSG:'+EPSG_CODE )
    dataset.SetProjection( srs.ExportToWkt() )
    # write array
    dataset.GetRasterBand(1).SetNoDataValue( -9999 )
    dataset.GetRasterBand(1).WriteArray( array.values )
    dataset = None
    return 0



"""
List some file paths
"""
path2files = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/uoe_lidar_sentinel/agb_Lidar_maps/gliht/'
path2output = '/home/dmilodow/DataStore_GCEL/YucatanBiomass/uoe_lidar_sentinel/agb_Lidar_maps/buffered_gliht/'

"""
Loop through the data files, and apply buffer in each case
"""
buffer_width = 9
ulim=500
files = glob.glob('%s*.tif' % path2files)
count=0
for f in files:
    count+=1
    print('\r%i/%i' % (count,len(files)))
    ds = xr.open_rasterio(f)
    mask = nd.maximum_filter(ds.values[0]<0,buffer_width,mode='constant',cval=0)
    ds_new = ds.copy()
    ds_new.values[0][mask] = -9999
    ds_new.values[ds.values>ulim] = -9999
    write_xarray_to_GeoTiff(ds_new.sel(band=1),'%s%s' % (path2output,f.split('/')[-1]))
