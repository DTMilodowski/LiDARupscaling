"""
compare_inventory_against_gridded_data.py
--------------------------------------------------------------------------------
COMPARE GRIDDED DATA VS INVENTORY DATA
This script loads a map and the field inventory data.
It then produces a plot comparing the two, to facilitate basic data exploration

This code built using the open source programming language python, and utilises
the geospatial library xarray (http://xarray.pydata.org/en/stable/) and the
machine learning library scikit-learn
(https://scikit-learn.org/stable/index.html).

22/08/2019 - D. T. Milodowski
--------------------------------------------------------------------------------
"""

"""
# Import the necessary packages
"""
import numpy as np                  # standard package for scientific computing
import xarray as xr                 # xarray geospatial package
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package
sns.set_style('darkgrid')
import pandas as pd
from scipy import stats
import shapely
from shapely.geometry.point import Point
from shapely.geometry import Polygon
import fiona

from sklearn.metrics import mean_squared_error

# Import custom libaries
import sys
sys.path.append('./data_io/')
import data_io as io

"""
Project Info
"""
path2data = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/'

"""
#===============================================================================
PART A: LOAD DATA
Load the raster data and inventory data
#-------------------------------------------------------------------------------
"""
raster_file = '%s/sentinel/processed/kiuic_b3_texture_mean.tif' % (path2data)
raster = io.load_geotiff(raster_file,option=1)
raster.values[raster.values==-9999]=np.nan

# inventory
inventory = fiona.open('%s/field_inventory/PUNTOS.shp' % path2data)

# Get the coordinate information for the raster datasets
X_raster = raster.coords['x'].values; Y_raster = raster.coords['y'].values
dX = X_raster[1]-X_raster[0]; dY = Y_raster[1] - Y_raster[0]
rad = np.sqrt(2.*max((dX/2.)**2,(dY/2.)**2))

# split inventory based on inside vs. outside lidar extent
raster_data = []; field_data = []; loc_x = []; loc_y = []
radius_1ha = np.sqrt(10.**4/np.pi)

# Loop through the plots
for ii,plot in enumerate(inventory):
    # Generate mask around plot to make subsequent code more efficient
    Xmin = plot['geometry']['coordinates'][0]-radius_1ha
    Ymin = plot['geometry']['coordinates'][1]-radius_1ha
    Xmax = plot['geometry']['coordinates'][0]+radius_1ha
    Ymax = plot['geometry']['coordinates'][1]+radius_1ha

    x_mask = np.all((X_raster>=Xmin-rad,X_raster<=Xmax+rad),axis=0)
    y_mask = np.all((Y_raster>=Ymin-rad,Y_raster<=Ymax+rad),axis=0)
    mask = np.ix_(y_mask,x_mask)

    # Get subset indices for masked array
    rows_sub = y_mask.sum(); cols_sub = x_mask.sum()
    X_sub = X_raster[x_mask]; Y_sub = Y_raster[y_mask]
    X1 = X_sub-dX/2.; X2 = X_sub+dX/2.; Y1 = Y_sub-dY/2.; Y2 = Y_sub+dY/2.
    # subset the rasters for sampling
    raster_sub = raster.values[mask]

    # Create a Shapely Point object for the plot centre and buffer to 1 ha area
    #plot_1ha = Point(plot['x'],plot['y']).buffer(radius_1ha)
    plot_1ha = Point(plot['geometry']['coordinates'][0],plot['geometry']['coordinates'][1]).buffer(radius_1ha)

    # now find all pixels from subset that at least partially fall within the plot radius
    in_plot = np.zeros((rows_sub,cols_sub))

    # for each pixel, check intersection area
    for rr in range(0, rows_sub):
        for cc in range(0, cols_sub):
            # create a pixel polygon
            pixel = Polygon(np.asarray([(X1[cc],Y1[rr]),(X2[cc],Y1[rr]),(X2[cc],Y2[rr]),(X1[cc],Y2[rr])]))
            # calculate the intersection fraction
            in_plot[rr,cc] = pixel.intersection(plot_1ha).area/pixel.area

    # Now calculate average AGB in 1 ha plot weighted by fraction of pixel area
    # within the plot
    wt_mean = np.sum(raster_sub*in_plot)/np.sum(in_plot)
    field_data.append(plot['properties']['AGB'])
    raster_data.append(wt_mean)
    loc_x.append(plot['geometry']['coordinates'][0])
    loc_y.append(plot['geometry']['coordinates'][1])

"""
#===============================================================================
PART B: COMPARE INVENTORY vs LiDAR/UPSCALED AGB ESTIMATES
#-------------------------------------------------------------------------------
"""
df = pd.DataFrame({'raster':raster_data,'plot':field_data,'x':loc_x,'y':loc_y})
# Now plot up summaries according to the subset in question
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[5,5])
sns.regplot('raster','plot',data=df,ax=ax,scatter_kws={'s':2})

temp1,temp2,r,temp3,temp4 = stats.linregress(df.dropna()['raster'],df.dropna()['plot'])
R2 = r**2
RMSE = np.sqrt(mean_squared_error(df.dropna()['raster'],df.dropna()['plot']))

ax.annotate('R$^2$=%.3f\nRMSE=%.1f' % (R2,RMSE), xy=(0.95,0.05), xycoords='axes fraction',
                        backgroundcolor='none',horizontalalignment='right',
                        verticalalignment='bottom', fontsize=10)
ax.set_xlabel('raster_value',fontsize=10)
ax.set_ylabel('AGB$_{inventory}$ / Mg ha$^{-1}$',fontsize=10)

fig.tight_layout()
fig.show()
