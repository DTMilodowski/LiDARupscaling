"""
utility
--------------------------------------------------------------------------------
Miscellaneous useful functions
David T. Milodowski, 29/08/2019
"""
import numpy as np
import xarray as xr
from shapely.geometry.point import Point
from shapely.geometry import Polygon

"""
sample_raster_by_point_neighbourhood_weighted_average
"""
def sample_raster_by_point_neighbourhood(plot,raster,radius):

    X_raster = raster.coords['x'].values
    Y_raster = raster.coords['y'].values
    dX = X_raster[1]-X_raster[0]
    dY = Y_raster[1]-Y_raster[0]
    rad = np.sqrt(2.*max((dX/2.)**2,(dY/2.)**2))

    # mask raster around region of interest
    Xmin = plot['geometry']['coordinates'][0]-radius
    Ymin = plot['geometry']['coordinates'][1]-radius
    Xmax = plot['geometry']['coordinates'][0]+radius
    Ymax = plot['geometry']['coordinates'][1]+radius

    x_mask = np.all((X_raster>=Xmin-rad,X_raster<=Xmax+rad),axis=0)
    y_mask = np.all((Y_raster>=Ymin-rad,Y_raster<=Ymax+rad),axis=0)
    mask = np.ix_(y_mask,x_mask)

    # Get subset indices for masked array
    rows_sub = y_mask.sum()
    cols_sub = x_mask.sum()
    X_sub = X_raster[x_mask]
    Y_sub = Y_raster[y_mask]
    X1 = X_sub-dX/2.
    X2 = X_sub+dX/2.
    Y1 = Y_sub-dY/2.
    Y2 = Y_sub+dY/2.

    # subset the rasters for sampling
    raster_sub = raster.values[mask]

    # Create a Shapely Point object for the plot centre and buffer to 1 ha area
    plot_boundary = Point(plot['geometry']['coordinates'][0],plot['geometry']['coordinates'][1]).buffer(radius)

    # now find all pixels from subset that at least partially fall within the plot radius
    in_plot = np.zeros((rows_sub,cols_sub))

    # for each pixel, check intersection area
    for rr in range(0, rows_sub):
        for cc in range(0, cols_sub):
            # create a pixel polygon
            pixel = Polygon(np.asarray([(X1[cc],Y1[rr]),(X2[cc],Y1[rr]),(X2[cc],Y2[rr]),(X1[cc],Y2[rr])]))
            # calculate the intersection fraction
            in_plot[rr,cc] = pixel.intersection(plot_boundary).area/pixel.area

    # Now calculate average weighted by fraction of pixel area within the plot
    weighted_mean = np.sum(raster_sub*in_plot)/np.sum(in_plot)
    return weighted_mean
