"""
CALIBRATE_LIDAR_AGB.PY
--------------------------------------------------------------------------------
Calibrate LiDAR AGB maps based on field inventory data
D.T. Milodowski
"""
import numpy as np
import xarray as xr
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table

import fiona
from shapely.geometry import Point
import geospatial_tools as gst

import copy as cp
import gc

# CHANGE THIS STUFF
# Some file names - shapefiles and rasters should be in a projected coordinate system (i.e. UTM)
#inventory_file = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/field_inventory/PUNTOS.shp'
inventory_file = '../../data/field_calibration/Kiuic_AGB_2018_LiDAR.shp'
#raster_file = '../../data/LiDAR_data/gliht_elevmax_dtm.tif'  # this should be the raster dataset that you want to interrogate
raster_file = '../../data/LiDAR_data/GliHT_TCH_1m_100.tif'
dem_file = '../../data/LiDAR_data/gliht_dtm.tif'
outfile = 'sample_test.csv' # this will be a file to hold the output data
plot_area = 10.**4 # 1 ha
gap_ht = 5 # height at which to define canopy gaps

# LOAD FILES, CONVERT TO FLOAT AND SPECIFY NODATA REGIONS
dem = xr.open_rasterio(dem_file)[0].sel(x=slice(229000,239000),y=slice(2230310,2214000))
chm = xr.open_rasterio(raster_file)[0].sel(x=slice(229000,239000),y=slice(2230310,2214000))
chm.values[dem.values==0]=-255
chm.values = chm.values.astype(np.float16)
chm.values[chm.values<0] = np.nan
chm.values/=100.
# Gap fraction
cover = cp.deepcopy(chm)
cover.values[chm.values<=gap_ht]=0
cover.values[chm.values>gap_ht]=1

# inventory data
inventory = fiona.open(inventory_file)

# A DICTIONARY TO CONTAIN THE RESULTS
chm_results = {}
cover_results = {}
#canopy_perc = {}
dem_results = {}
inventory_AGB = {}
# calculate plot radius
radius = np.sqrt(plot_area/np.pi)

# SAMPLE RASTER FOR EACH PLOT NEIGHBOURHOOD
for pp,plot in enumerate(inventory):
    id = plot['id']
    inventory_AGB[id]=plot['properties']['agb']
    #canopy_perc[id]=plot['properties']['Return 1 c']/plot['properties']['Total firs']
    plot_centre = Point(plot['geometry']['coordinates'][0],plot['geometry']['coordinates'][1])
    chm_results[id] = gst.sample_raster_by_point_neighbourhood(chm,plot_centre,radius,x_dim='x',y_dim='y',label = id)
    dem_results[id] = gst.sample_raster_by_point_neighbourhood(dem,plot_centre,radius,x_dim='x',y_dim='y',label = id)
    cover_results[id] = gst.sample_raster_by_point_neighbourhood(cover,plot_centre,radius,x_dim='x',y_dim='y',label = id)

# clear memory
dem.close(); dem=None
chm.close(); chm=None
cover=None
gc.collect()

# CALIBRATION STATISTICS
ID=[]
AGB = []
TCH = []
COVER = []
#CANOPY_PERC = []
for plot in inventory:
    id = plot['id']
    if chm_results[id]['status']=='PASS':
        if chm_results[id]['weighted_average']>0:
            ID.append(id)
            AGB.append(inventory_AGB[id])
            TCH.append(chm_results[id]['weighted_average'][0])
            COVER.append(cover_results[id]['weighted_average'][0])
            #CANOPY_PERC.append(canopy_perc[id])

AGB = np.asarray(AGB)
TCH = np.asarray(TCH)
COVER = np.asarray(COVER)
#CANOPY_PERC = np.asarray(CANOPY_PERC)
ID = np.asarray(ID)

fig,axes = plt.subplots(nrows=1,ncols=4,figsize=[12,4])
axes[0].plot(TCH,AGB,'.')
axes[0].set_title('gliht LiDAR vs. inventory AGB')
axes[0].set_xlabel('mean TCH / m')
axes[0].set_ylabel('field AGB / Mg ha$^{-1}$')

im1 = axes[1].imshow(chm_results['20']['raster_values'][0],vmin=0,vmax=15)
plt.colorbar(im1,orientation='horizontal',ax=axes[1])
im2 = axes[2].imshow(chm_results['27']['raster_values'][0],vmin=0,vmax=15)
plt.colorbar(im2,orientation='horizontal',ax=axes[2])
im2 = axes[3].imshow(chm_results['3']['raster_values'][0],vmin=0,vmax=15)
plt.colorbar(im2,orientation='horizontal',ax=axes[3])
axes[1].set_title('plot 20; AGB = %.2f Mg/ha' % inventory_AGB['20'])
axes[2].set_title('plot 27; AGB = %.2f Mg/ha' % inventory_AGB['27'])
axes[3].set_title('plot 3; AGB = %.2f Mg/ha' % inventory_AGB['3'])

fig.show()
fig.savefig('example_of_plot_query.png')

# Calculate estimated GF from GF-AGB relationship
ro1,ro0,r,p,_=stats.linregress(np.log(TCH),np.log(COVER/(1-COVER)))
COVER_ = 1/(1 - np.exp(ro0) * TCH**ro1)
COVER_residual = COVER-COVER_

# fit multivariate model
cal_data = pd.DataFrame({'AGB':AGB,'TCH':TCH,'COVER_res':COVER_residual,
                        'lnAGB':np.log(AGB),'lnTCH':np.log(TCH),
                        'lnCOVER_res':np.log(COVER_residual)})
ols = smf.ols('lnAGB ~ lnTCH ',data=cal_data)
results = ols.fit()
st, fit_data, ss2 = summary_table(results)
fittedvalues = fit_data[:,2]
predict_mean_se  = fit_data[:,3]
predict_mean_ci_low, predict_mean_ci_upp = fit_data[:,4:6].T
predict_ci_low, predict_ci_upp = fit_data[:,6:8].T

print(results.summary())

"""
Plot calibrated model against inventory
"""
sns.set()
cal_df = pd.DataFrame({'mod':np.exp(fittedvalues),'obs':AGB,
                        'predict_ci_low':predict_ci_low,
                        'predict_ci_upp':predict_ci_upp})
# Now plot up summaries according to the subset in question
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[4,4])

ax.plot([0,cal_df['obs'].max()],[0,cal_df['obs'].max()],':',color='0.7')
#ax.errorbar(cal_df['obs'],cal_df['mod'],
            #yerr=(cal_df['mod']-cal_df['predict_ci_low'],cal_df['predict_ci_upp']-cal_df['mod']),
            #color='0.25',linewidth=0.5,linestyle='none')
ax.plot(cal_df['mod'],cal_df['obs'],'.',color='black')

RMSE = np.sqrt(np.mean((cal_df['mod']-cal_df['obs'])**2))

ax.annotate('R$^2$=%.2f\nRMSE=%.1f' % (results.rsquared_adj,RMSE), xy=(0.95,0.05),
        xycoords='axes fraction', backgroundcolor='none',ha='right',va='bottom',
        fontsize=10)

ax.set_xlabel('AGB$_{LiDAR}$ / Mg ha$^{-1}$',fontsize=10)
ax.set_ylabel('AGB$_{inventory}$ / Mg ha$^{-1}$',fontsize=10)
ax.set_aspect('equal')

#ax.set_ylim(bottom=0)
#ax.set_xlim(left=0)

fig.tight_layout()
#fig.savefig('%s%s_%s_inventory_comparison.png' % (path2fig,site_id,version))
fig.show()
