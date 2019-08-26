"""
pt3_lidar_sentinel_fusion_validation_vs_inventory.py
--------------------------------------------------------------------------------
VALIDATION OF UPSCALED MAPS VS INVENTORY DATA
This script loads a map of lidar estimated AGB, the upscaled AGB map, and the
field inventory data. It then produces three plots:
i) LiDAR AGB vs. inventory used for training LIDAR map
ii) Upscaled AGB vs. inventory used for training LiDAR map
iii) Upscaled AGB vs. inventory not used for training LiDAR map

The final plot is an independent validation of the upscaled product. The first
two allow an assessment of the relative performance of the upscaled map relative
to the original LiDAR-field calibrated relationship.

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
import os
import pandas as pd
from scipy import stats
import shapely
from shapely.geometry.point import Point
from shapely.geometry import Polygon


from sklearn.metrics import r2_score, mean_squared_error

# Import custom libaries
import sys
sys.path.append('./data_io/')
import data_io as io

"""
Project Info
"""
site_id = 'kiuic'
version = '010'
crs = ccrs.UTM('16N')
path2alg = '../saved_models/'
path2fig= '../figures/'
path2data = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/'
path2output = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/output/'
if(os.path.isdir(path2output)==False):
    os.mkdir(path2output)

"""
#===============================================================================
PART A: LOAD DATA
Load the LiDAR AGB map, upscaled AGB map and inventory data
Split inventory data into calibration vs. validation
#-------------------------------------------------------------------------------
"""
# LiDAR AGB
lidar_file = '%s/lidar/processed/%s_AGB_07-31-19_regridded.tif' % (path2data,site_id)
lidar = io.load_geotiff(lidar_file,option=1)
lidar.values[lidar.values==-9999]=np.nan
lidar.values[lidar.values<0]=0

# Upscaled AGB
upscaled_file = '%s%s_%s_rf_agb_upscaled.tif' % (path2output,site_id,version)
upscaled = io.load_geotiff(upscaled_file,option=1)
upscaled.values[upscaled.values==-9999]=np.nan
upscaled.values[upscaled.values<0]=0

# inventory
inventory_file = '%s/field_inventory/%s_field_inventory.csv' % (path2data,site_id)
inventory = np.genfromtxt(inventory_file,delimiter=',',names=True)

# Get the coordinate information for the raster datasets
X_raster = lidar.coords['x'].values
Y_raster = lidar.coords['y'].values
dX = X_raster[1]-X_raster[0]
dY = Y_raster[1] - Y_raster[0]
rad = np.sqrt(2.*max((dX/2.)**2,(dY/2.)**2))

# split inventory based on inside vs. outside lidar extent
lidar_agb_field = []
lidar_agb_lidar = []
lidar_agb_upscaled = []
other_agb_field = []
other_agb_upscaled = []
radius_1ha = np.sqrt(10.**4/np.pi)

# Loop through the plots
for ii,plot in enumerate(inventory):

    # Generate mask around plot to make subsequent code more efficient
    Xmin = plot['x']-radius_1ha
    Ymin = plot['y']-radius_1ha
    Xmax = plot['x']+radius_1ha
    Ymax = plot['y']+radius_1ha

    x_mask = np.all((X_raster>=Xmin-rad,X_raster<=Xmax+rad),axis=0)
    y_mask = np.all((Y_raster>=Ymin-rad,Y_raster<=Ymax+rad),axis=0)
    mask = np.ix_(y_mask,x_mask)

    # Get subset indices for masked array
    rows_sub = y_mask.sum()
    cols_sub = x_mask.sum()
    raster_sub = np.zeros((rows_sub,cols_sub))
    X_sub = X_raster[x_mask]
    Y_sub = Y_raster[y_mask]
    X1 = X_sub-dX/2.
    X2 = X_sub+dX/2.
    Y1 = Y_sub-dY/2.
    Y2 = Y_sub+dY/2.

    # subset the rasters for sampling
    lidar_sub = lidar.values[mask]
    upscaled_sub = upscaled.values[mask]

    # Create a Shapely Point object for the plot centre and buffer to 1 ha area
    plot_1ha = Point(plot['x'],plot['y']).buffer(radius_1ha)

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
    lidar_agb = np.sum(lidar_sub*in_plot)/np.sum(in_plot)
    upscaled_agb = np.sum(upscaled_sub*in_plot)/np.sum(in_plot)

    if np.isfinite(lidar_agb):
        lidar_agb_field.append(plot['AGB'])
        lidar_agb_lidar.append(lidar_agb)
        lidar_agb_upscaled.append(upscaled_agb)
    else:
        other_agb_field.append(plot['AGB'])
        other_agb_upscaled.append(upscaled_agb)

    """
    # this code snippet locates nearest pixel. Not used
    nearest_x = np.argsort((lidar.coords['x'].values-plot['x'])**2)[0]
    nearest_y = np.argsort((lidar.coords['y'].values-plot['y'])**2)[0]
    lidar_agb = lidar.values[nearest_y,nearest_x]
    if np.isfinite(lidar_agb):
        lidar_agb_field.append(plot['AGB'])
        lidar_agb_lidar.append(lidar_agb)
        lidar_agb_upscaled.append(upscaled.values[nearest_y,nearest_x])
    else:
        other_agb_field.append(plot['AGB'])
        other_agb_upscaled.append(upscaled.values[nearest_y,nearest_x])
    """

"""
#===============================================================================
PART B: COMPARE INVENTORY vs LiDAR/UPSCALED AGB ESTIMATES
#-------------------------------------------------------------------------------
"""
lidar_df = pd.DataFrame({'lidar':lidar_agb_lidar,'upscaled':lidar_agb_upscaled,
                    'plot':lidar_agb_field})
other_df = pd.DataFrame({'upscaled':other_agb_upscaled,'plot':other_agb_field})

# Now plot up summaries according to the subset in question
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=[10,3.4],sharex='all',sharey='all')

sns.regplot('lidar','plot',data=lidar_df,ax=axes[0],scatter_kws={'s':2})
sns.regplot('upscaled','plot',data=lidar_df,ax=axes[1],scatter_kws={'s':2})
sns.regplot('upscaled','plot',data=other_df,ax=axes[2],scatter_kws={'s':2})

x_labs = ['AGB$_{lidar}$ / Mg ha$^{-1}$',
        'AGB$_{upscaled}$ / Mg ha$^{-1}$',
        'AGB$_{upscaled}$ / Mg ha$^{-1}$']
annotations = ['lidar vs.\ninventory','upscaled vs.\ninventory (lidar)',
                'upscaled vs.\ninventory (outside)']

temp1,temp2,r_a,temp3,temp4 = stats.linregress(lidar_df.dropna()['lidar'],lidar_df.dropna()['plot'])
temp1,temp2,r_b,temp3,temp4 = stats.linregress(lidar_df.dropna()['upscaled'],lidar_df.dropna()['plot'])
temp1,temp2,r_c,temp3,temp4 = stats.linregress(other_df.dropna()['upscaled'],other_df.dropna()['plot'])

R2 = [r_a**2,r_b**2,r_c**2]

RMSE = [np.sqrt(mean_squared_error(lidar_df.dropna()['lidar'],lidar_df.dropna()['plot'])),
    np.sqrt(mean_squared_error(lidar_df.dropna()['upscaled'],lidar_df.dropna()['plot'])),
    np.sqrt(mean_squared_error(other_df.dropna()['upscaled'],other_df.dropna()['plot']))]

for ii,ax in enumerate(axes):
    ax.annotate(annotations[ii], xy=(0.05,0.95), xycoords='axes fraction',
                        backgroundcolor='none',horizontalalignment='left',
                        verticalalignment='top', fontsize=10)
    ax.annotate('R$^2$=%.3f\nRMSE=%.1f' % (R2[ii],RMSE[ii]), xy=(0.95,0.05), xycoords='axes fraction',
                        backgroundcolor='none',horizontalalignment='right',
                        verticalalignment='bottom', fontsize=10)
    ax.set_xlabel(x_labs[ii],fontsize=10)
    ax.set_ylabel('AGB$_{inventory}$ / Mg ha$^{-1}$',fontsize=10)
    ax.set_aspect('equal')

fig.tight_layout()
fig.savefig('%s%s_%s_inventory_comparison.png' % (path2fig,site_id,version))
fig.show()
