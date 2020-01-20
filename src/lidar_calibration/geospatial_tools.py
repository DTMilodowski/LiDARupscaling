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

    results={}
    results['id'] = label

    if label is None:
        label = str('ID%i' % np.random.random()*10**9)

    X = raster[x_dim].values
    Y = raster[y_dim].values
    dX = X[1]-X[0]
    dY = Y[1]-Y[0]
    rad = np.sqrt(2.*max((dX/2.)**2,(dY/2.)**2))

    if len(raster.values.shape)==2:
        bands = 1
    elif len(raster.values.shape)==3:
        bands = raster.shape[0]

    # Generate mask around AOI to make subsequent code more efficient
    Xmin = polygon.bounds[0]; Xmax = polygon.bounds[2]
    Ymin = polygon.bounds[1]; Ymax = polygon.bounds[3]

    if(np.all((Xmin-rad>=X.min(),Xmax+rad<=X.max(),Ymin-rad>=Y.min(),Ymax+rad<=Y.max()))):

        x_mask = np.all((X>=Xmin-rad,X<=Xmax+rad),axis=0)
        y_mask = np.all((Y>=Ymin-rad,Y<=Ymax+rad),axis=0)
        mask = np.ix_(y_mask,x_mask)

        X_sub = X[x_mask]
        Y_sub = Y[y_mask]
        X1 = X_sub-dX/2.
        X2 = X_sub+dX/2.
        Y1 = Y_sub-dY/2.
        Y2 = Y_sub+dY/2.

        cols_sub = x_mask.sum()
        rows_sub = y_mask.sum()
        raster_sub = np.zeros((bands,rows_sub,cols_sub))
        for bb in range(0,bands):
            raster_sub[bb,:,:] = raster[bb,:,:][mask]

        # now find all pixels from subset that at least partially fall within the polygon boundary
        in_polygon = np.zeros((rows_sub,cols_sub))
        for rr in range(0, rows_sub):
            for cc in range(0, cols_sub):
                # create a pixel polygon
                pixel = Polygon(np.asarray([(X1[cc],Y1[rr]),(X2[cc],Y1[rr]),(X2[cc],Y2[rr]),(X1[cc],Y2[rr])]))
                # calculate the intersection fraction
                in_polygon[rr,cc] = pixel.intersection(polygon).area/pixel.area

        # calculate the weighted average
        weighted_average = np.zeros(bands)*np.nan
        for bb in range(0,bands):
            weighted_average[bb] = np.nansum(raster_sub[bb]*in_plot)/np.nansum(in_neighbourhood)

        results={}
        results['id'] = label
        results['weights']=in_plot.copy()
        results['raster_values']=raster_sub.copy()
        results['x']=X_sub.copy()
        results['y']=Y_sub.copy()
        results['weighted_average'] = weighted_average
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

    neighbourhood = point.buffer(radius)

    results={}
    results['id'] = label

    if label is None:
        label = str('ID%i' % np.random.random()*10**9)

    X = raster[x_dim].values
    Y = raster[y_dim].values
    dX = X[1]-X[0]
    dY = Y[1]-Y[0]
    rad = np.sqrt(2.*max((dX/2.)**2,(dY/2.)**2))

    if len(raster.values.shape)==2:
        bands = 1
        raster=raster.expand_dims('band')
    elif len(raster.values.shape)==3:
        bands = raster.shape[0]

    # Generate mask around AOI to make subsequent code more efficient
    Xmin = neighbourhood.bounds[0]; Xmax = neighbourhood.bounds[2]
    Ymin = neighbourhood.bounds[1]; Ymax = neighbourhood.bounds[3]

    if(np.all((Xmin-rad>=X.min(),Xmax+rad<=X.max(),Ymin-rad>=Y.min(),Ymax+rad<=Y.max()))):

        x_mask = np.all((X>=Xmin-rad,X<=Xmax+rad),axis=0)
        y_mask = np.all((Y>=Ymin-rad,Y<=Ymax+rad),axis=0)
        mask = np.ix_(y_mask,x_mask)

        X_sub = X[x_mask]
        Y_sub = Y[y_mask]
        X1 = X_sub-dX/2.
        X2 = X_sub+dX/2.
        Y1 = Y_sub-dY/2.
        Y2 = Y_sub+dY/2.

        cols_sub = x_mask.sum()
        rows_sub = y_mask.sum()
        raster_sub = np.zeros((bands,rows_sub,cols_sub))

        for bb in range(0,bands):
            raster_sub[bb] = raster.values[bb,:,:][mask]

        # now find all pixels from subset that at least partially fall within the polygon boundary
        in_neighbourhood = np.zeros((rows_sub,cols_sub))
        for rr in range(0, rows_sub):
            for cc in range(0, cols_sub):
                # create a pixel polygon
                pixel = Polygon(np.asarray([(X1[cc],Y1[rr]),(X2[cc],Y1[rr]),(X2[cc],Y2[rr]),(X1[cc],Y2[rr])]))
                # calculate the intersection fraction
                in_neighbourhood[rr,cc] = pixel.intersection(neighbourhood).area/pixel.area

        # calculate the weighted average
        weighted_average = np.zeros(bands)*np.nan
        for bb in range(0,bands):
            weighted_average[bb] = np.nansum(raster_sub[bb]*in_neighbourhood)/np.nansum(in_neighbourhood)

        results['weights']=in_neighbourhood.copy()
        results['raster_values']=raster_sub.copy()
        results['x']=X_sub.copy()
        results['y']=Y_sub.copy()
        results['weighted_average'] = weighted_average
        results['status'] = 'PASS'
    else:
        results['status'] = 'FAIL'
    return results
