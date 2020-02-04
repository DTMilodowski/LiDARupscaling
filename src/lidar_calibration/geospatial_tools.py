"""
# GEOSPATIAL_TOOLS.PY
--------------------------------------------------------------------------------
A set of geospatial tools, e.g. spatial intersections of pixels with other
objects, for integration into raster query functions
-----------------
D.T. Milodowski
"""
import numpy as np
import xarray as xr
from shapely.geometry import shape, Polygon, Point

"""
sample_raster_by_polygon
------------------------
Sample a raster based on a polygon, returning the information needed to
calculate the weighted average of the pixels intersecting the polygon, weighted
by the area of intersection
Inputs are:
- raster (xarray)
- polygon (shapely Polygon)
Returns:
- dictionary containing the x and y coordinates of the pixels (for checking),
  the raster values at these pixels and the corresponding weights, and an
  identifying label
"""
def sample_raster_by_polygon(raster,polygon,x_dim='x',y_dim='y',label = None):

    # generate 'unique' ID label if none specified
    if label is None:
        label = str('ID%i' % np.random.random()*10**9)

    results={}
    results['id'] = label

    # Get dimension info etc
    X = raster[x_dim].values; Y = raster[y_dim].values
    dX = X[1]-X[0]; dY = Y[1]-Y[0]
    rad = np.sqrt(2.*max((dX/2.)**2,(dY/2.)**2))

    # Check number of bands is consistent with expectations for subsequent processing
    if len(raster.values.shape)==2:
        bands = 1
        raster=raster.expand_dims('band')
    elif len(raster.values.shape)==3:
        bands = raster.shape[0]

    # Generate mask around AOI to make subsequent code more efficient
    Xmin,Ymin,Xmax,Ymax = polygon.bounds
    if(np.all((Xmin-rad>=X.min(),Xmax+rad<=X.max(),Ymin-rad>=Y.min(),Ymax+rad<=Y.max()))):

        if dY<0:
            raster_sub = raster.sel(x=slice(Xmin-rad,Xmax+rad),y=slice(Ymax+rad,Ymin-rad))
        else:
            raster_sub = raster.sel(x=slice(Xmin-rad,Xmax+rad),y=slice(Ymin-rad,Ymax+rad))

        X1 = raster_sub[x_dim].values-dX/2.; X2 = raster_sub[x_dim].values+dX/2.
        Y1 = raster_sub[y_dim].values-dY/2.; Y2 = raster_sub[y_dim].values+dY/2.

        # now find all pixels from subset that at least partially fall within the polygon boundary
        #in_neighbourhood = np.zeros((rows_sub,cols_sub))
        in_neighbourhood = raster_sub[0].copy(deep=True)
        in_neighbourhood.values=in_neighbourhood.values.astype('float')
        in_neighbourhood.values*=np.nan
        for rr in range(0,raster_sub[y_dim].values.size):
            for cc in range(0,raster_sub[x_dim].values.size):
                # create a pixel polygon
                pixel = Polygon(np.asarray([(X1[cc],Y1[rr]),(X2[cc],Y1[rr]),(X2[cc],Y2[rr]),(X1[cc],Y2[rr])]))
                # calculate the intersection fraction
                in_neighbourhood.values[rr,cc] = pixel.intersection(polygon).area/pixel.area

        # calculate the weighted average
        weighted_average = np.zeros(bands)*np.nan
        for bb in range(0,bands):
            weighted_average[bb] = np.nansum(raster_sub.values[bb]*in_neighbourhood.values)/np.nansum(in_neighbourhood.values)

        # Store results for this sample
        results['weights']=in_neighbourhood.copy(deep=True)
        results['raster_values']=raster_sub.copy(deep=True)
        results['weighted_average'] = weighted_average.copy()
        results['status'] = 'PASS'
    else:
        results['status'] = 'FAIL'
    return results


"""
sample_raster_by_point_neighbourhood
------------------------
Sample a raster based on a circular neighbourhood, returning the information
needed to calculate the weighted average of the pixels intersecting the polygon,
weighted by the area of intersection
Inputs are:
- raster (xarray)
- point (shapely Point)
- radius (float)
Returns:
- dictionary containing the x and y coordinates of the pixels (for checking),
  the raster values at these pixels and the corresponding weights, and an
  identifying label
"""
def sample_raster_by_point_neighbourhood(raster,point,radius,x_dim='x',y_dim='y',label = None):

    # generate 'unique' ID label if none specified
    if label is None:
        label = str('ID%i' % np.random.random()*10**9)

    # a dictionary to hold the results
    results={}
    results['id'] = label

    # Get dimension info etc
    X = raster[x_dim].values; Y = raster[y_dim].values
    dX = X[1]-X[0]; dY = Y[1]-Y[0]
    rad = np.sqrt(2.*max((dX/2.)**2,(dY/2.)**2))

    # Check number of bands is consistent with expectations for subsequent processing
    if len(raster.values.shape)==2:
        bands = 1
        raster=raster.expand_dims('band')
    elif len(raster.values.shape)==3:
        bands = raster.shape[0]

    # Generate mask around AOI to make subsequent code more efficient
    neighbourhood = point.buffer(radius)
    Xmin,Ymin,Xmax,Ymax = neighbourhood.bounds
    test_distance_inner = radius*np.sqrt(0.5)
    if(np.all((Xmin-rad>=X.min(),Xmax+rad<=X.max(),Ymin-rad>=Y.min(),Ymax+rad<=Y.max()))):
        if dY<0:
            raster_sub = raster.sel(x=slice(Xmin-rad,Xmax+rad),y=slice(Ymax+rad,Ymin-rad))
        else:
            raster_sub = raster.sel(x=slice(Xmin-rad,Xmax+rad),y=slice(Ymin-rad,Ymax+rad))

        X1 = raster_sub[x_dim].values-dX/2.; X2 = raster_sub[x_dim].values+dX/2.
        Y1 = raster_sub[y_dim].values-dY/2.; Y2 = raster_sub[y_dim].values+dY/2.

        # Isolate pixels fully inside plot boundary
        x_mask = np.all([np.max([X1,X2],axis=0)<=point.x+test_distance_inner,
                        np.min([X1,X2],axis=0)>=point.x-test_distance_inner],axis=0)
        y_mask = np.all([np.max([Y1,Y2],axis=0)<=point.y+test_distance_inner,
                        np.min([Y1,Y2],axis=0)>=point.y-test_distance_inner],axis=0)
        mask_inner = np.ix_(y_mask,x_mask)

        # now find all pixels from subset that at least partially fall within the polygon boundary
        in_neighbourhood = raster_sub[0].copy(deep=True)
        in_neighbourhood.values=in_neighbourhood.values.astype('float')
        in_neighbourhood.values*=np.nan
        in_neighbourhood.values[mask_inner]=1

        for rr in range(0,raster_sub[y_dim].values.size):
            for cc in range(0,raster_sub[x_dim].values.size):
                if np.isnan(in_neighbourhood.values[rr,cc]):
                    # create a pixel polygon
                    pixel = Polygon(np.asarray([(X1[cc],Y1[rr]),(X2[cc],Y1[rr]),(X2[cc],Y2[rr]),(X1[cc],Y2[rr])]))
                    # calculate the intersection fraction
                    in_neighbourhood.values[rr,cc] = pixel.intersection(neighbourhood).area/pixel.area

        # calculate the weighted average
        weighted_average = np.zeros(bands)*np.nan
        for bb in range(0,bands):
            weighted_average[bb] = np.nansum(raster_sub.values[bb]*in_neighbourhood.values)/np.nansum(in_neighbourhood.values)

        # Store results for this sample
        results['weights']=in_neighbourhood.copy(deep=True)
        results['raster_values']=raster_sub.copy(deep=True)
        results['weighted_average'] = weighted_average.copy()
        results['status'] = 'PASS'
    else:
        results['status'] = 'FAIL'
    return results
