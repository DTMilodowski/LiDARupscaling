"""
27/02/2019 - DTM
Paring back scripts to the data io routines required in the potential biomass
estimation.

30/11/2018 - DTM
Rewritten some of the functions specific to Forests2020 potential biomass work
- no LUH data
- restricted set of soilgrids parameters

12/11/2018 - JFE
This file contains the definition of some useful functions
for the pantrop-AGB-LUH work
"""
import numpy as np
import glob
import xarray as xr #xarray to read all types of formats
import rasterio
from osgeo import gdal
import osr

"""
load_geotiff
A very simple function that reads a geotiff and returns it as an xarray. Nodata
values are converted to the numpy nodata value.
The input arguments are:
- filename (this should include the full path to the file)
Optional arguments are:
- band (default = 1)
- x_name (default = 'longitude')
- y_name (default = 'latitude')
- nodata_option (default = 0).
            0: use value in dataset metadata. This is usually fine, except if
                there is an issue with the precision, and is applied in all
                cases.
            1: arbitrary cutoff for nodata to account for precision problems
                with float32. Other similar options could be added if necessary.
            2: set all negative values as nodata


"""
def load_geotiff(filename, band = 1,x_name='longitude',y_name='latitude',option=0):
    xarr = xr.open_rasterio(filename).sel(band=band)
    if(option==0):
        xarr.values[xarr.values==xarr.nodatavals[0]]=np.nan
    if(option==1):
        xarr.values[xarr.values<-3*10**38]=np.nan
    if(option==2):
        xarr.values[xarr.values<0]=np.nan
    return xarr #return the xarray object

"""
copy_xarray_template
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
    geoTrans[0] = np.min(lon)-dlon/2.
    if geoTrans[5]>0:
        geoTrans[3]=np.min(lat)-dlat/2.
    else:
        geoTrans[3]=np.max(lat)-dlat/2.
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
    #EPSG_CODE = array.attrs['crs'].split(':')[-1]

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
    #srs = osr.SpatialReference()
    #srs.SetWellKnownGeogCS( 'EPSG:'+EPSG_CODE )
    #dataset.SetProjection( srs.ExportToWkt() )
    # write array
    dataset.GetRasterBand(1).SetNoDataValue( -9999 )
    dataset.GetRasterBand(1).WriteArray( array.values )
    dataset = None
    return 0


"""
load_predictors
This function loads all of the datasets containing the explanatory variables,
and any nodata values are removed.

The function takes two input arguments:
    site_id
This is the prefix that should start all filenames to be included in the
analysis
    path2data
This is the file path to the data directory for the workshop, and can be
expressed either as a full path or relative path. It is needed so that the code
knows exactly where things are stored - note the correct directory structure is
required!!!

The function returns four objects:
    1) predictors: a large 2D numpy array where the rows correspond with
       pixels and the columns correspond with each successive data set.
    2) target: a large 1D numpy array with the AGB estimate for each land pixel.
       Note that the pixel order matches the pixel order in the predictors
       array.
    3) sentinel nodata mask
    4) labels for predictor variables
"""
def load_predictors(site_id = 'kiuic', path2data = "../data/",layers = ['sentinel2','alos'], resolution='20'):

    path2sentinel = '%s/processed_textures/sentinel_%sm/' % (path2data,resolution.zfill(3))
    if resolution=='20m':
        path2alos = '%s/processed_pre_texture/alos_%sm/' % (path2data,resolution)
    else:
        path2alos = '%s/processed_textures/alos_%sm/' % (path2data,resolution.zfill(3))

    #  load the LiDAR data to check we only keep pixels with AGB estimates
    files = glob.glob('%s%s*.tif' % (path2sentinel,site_id))
    template = xr.open_rasterio(files[0]).values[0]
    rows,cols=template.shape

    data_layers = np.zeros((0,rows,cols))
    mask=np.ones((rows,cols),dtype='bool')
    nodata=[]
    labels = []

    # Load the sentinel data
    if 'sentinel2' in layers:
        sentinel_files = sorted(glob.glob('%s%s*tif' % (path2sentinel,site_id)))
        for ff in sentinel_files:
            nodata.append(rasterio.open(ff).nodatavals[0])
            labels.append(ff.split('/')[-1].split('.')[0])

        sentinel = np.zeros((len(sentinel_files),rows,cols))
        for ii,ff in enumerate(sentinel_files):
            #print(ff)
            sentinel[ii] = xr.open_rasterio(ff).values[0]
            mask = mask & (sentinel[ii]!=nodata[ii])
            mask = mask & (sentinel[ii]>-3*10**38)

        data_layers = np.concatenate((data_layers,sentinel),axis=0)
        print('Loaded Sentinel-2 data')

    # load the alos data
    if 'alos' in layers:
        alos_files = sorted(glob.glob('%s%s*tif' % (path2alos,site_id)))
        for ff in alos_files:
            nodata.append(rasterio.open(ff).nodatavals[0])
            labels.append(ff.split('/')[-1].split('.')[0])

        alos = np.zeros((len(alos_files),rows,cols))
        for ii,ff in enumerate(alos_files):
            #print(ff)
            alos[ii] = xr.open_rasterio(ff).values[0]
            mask = mask & (alos[ii]!=nodata[ii])
            mask = mask & (alos[ii]>-3*10**38)

        data_layers = np.concatenate((data_layers,alos),axis=0)
        print('Loaded ALOS data')

    return(data_layers,mask,labels)

"""
apply mask to raster stack
-----------------------------------------
apply 2D mask to stack of rasters (3rd dimension = different variables) to be
fed into machine learning applications
"""
def apply_mask_to_raster_stack(predictors,mask):
    n_vars = predictors.shape[0]
    X = np.zeros((mask.sum(),n_vars))
    for ii in range(0,n_vars):
        X[:,ii] = predictors[ii][mask]

    return X
