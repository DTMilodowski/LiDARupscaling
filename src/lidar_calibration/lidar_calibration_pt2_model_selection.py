"""
lidar_calibration_pt2_model_selection.py
================================================================================
Collate the field and LiDAR TCH data and visualise the relationship between
canopy height and AGB for the plots. Produces a six panel figure, the TCH-AGB
relationship and five plots from the dataset.
"""
import numpy as np
import xarray as xr
import pandas as pd

import seaborn as sns
sns.set()
from matplotlib import pyplot as plt

from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table
from sklearn.model_selection import LeaveOneOut

import fiona
import stats_tools as st

import sys
import os
import copy as cp
import gc

sys.path.append('../data_io/')
import data_io as io

# CHANGE THIS STUFF
# Some file names - shapefiles and rasters should be in a projected coordinate system (i.e. UTM)
inventory_file = '../../data/lidar_calibration/Kiuic_400_live_trees.shp'
pt1_outfile = '../../data/lidar_calibration/sample_test.npz' # this will be a file to hold the compiled plot data
pt2_outfile = '../../saved_models/lidar_calibration/lidar_calibration_pt2_results.npz'
path2fig = '../../figures/'
plot_area = 400. # 1 ha
gap_ht = 8.66
radius = np.sqrt(plot_area/np.pi)

# LOAD FILES, CONVERT TO FLOAT AND SPECIFY NODATA REGIONS
pt1_output = np.load(pt1_outfile)['arr_0'][()]
chm_results = pt1_output['chm']
dem_results = pt1_output['dem']
inventory_AGB = pt1_output['inventory']

# COLLATE DATA INTO ARRAYS
ID=[]; AGB = []; TCH = []; COVER = []; NODATA = []; TCH_SD = []
for plot in chm_results.keys():
    id = chm_results[plot]['id']
    if chm_results[id]['status']=='PASS':
        if chm_results[id]['weighted_average']>0:
            if np.sum(np.isnan(chm_results[id]['raster_values'][0])*chm_results[id]['weights'])<0.05*np.sum(chm_results[id]['weights']):
                weights = chm_results[id]['weights'].values/np.sum(np.isfinite(chm_results[id]['raster_values'][0].values)*chm_results[id]['weights'].values)
                mean_tch = chm_results[id]['weighted_average'][0]
                ID.append(id)
                AGB.append(inventory_AGB[id])
                TCH.append(mean_tch)
                TCH_SD.append(np.sqrt(np.nansum(weights*(chm_results[id]['raster_values'][0].values-mean_tch)**2)))
                COVER.append(np.sum((chm_results[id]['raster_values'][0].values>=gap_ht)*weights))
                NODATA.append(np.mean(~np.isfinite(chm_results[id]['raster_values'][0].values)))

# Convert to np arrays for conditional indexing convenience
AGB = np.asarray(AGB); TCH = np.asarray(TCH); COVER = np.asarray(COVER);
TCH_SD = np.asarray(TCH_SD); ID = np.asarray(ID)

TCH_SDrel = TCH_SD/TCH

host_plot = []
for id in ID:
    host_plot.append(id[:-2])

"""
MAY WANT TO INCREASE THE NUMBER OF ADDITIONAL METRICS
# calculate the additional metrics as reqiured here
"""
# calculate residual gap fraction based on COVER-AGB relationship
mask = np.isfinite(np.log(COVER/(1-COVER)))
ro1,ro0,r,p,_=stats.linregress(np.log(TCH[mask]),np.log(COVER[mask]/(1-COVER[mask])))
COVER_ = 1/(1 + np.exp(-ro0) * TCH**(-ro1))
COVER_residual = COVER-COVER_
"""
alternative that uses nonlinear regression, rather than log-log OLS, to fit
the logistic model
"""
"""
def fun5(params,x,y):
    return np.mean((1/(1 + np.exp(-params[0]) * x**-params[1]) - y)**2)
opt2 = minimize(fun5,params0,method='L-BFGS-B',args=(TCH,COVER))
minimize(fun5,np.array([-5.80746225,  4.03222333]),method='L-BFGS-B',args=(TCH,COVER))

COVER_ = 1/(1 + np.exp(-opt2.x[0]) * TCH**(-opt2.x[1]))
COVER_residual = COVER-COVER_
"""

"""
UPDATE THIS PART TO MAKE SURE THAT ALL VARIABLES THAT YOU WANT ARE STORED IN THE
DATAFRAME
"""
cal_data = pd.DataFrame({'AGB':AGB,'TCH':TCH,'COVER_res':COVER_residual,
                        'lnAGB':np.log(AGB),'lnTCH':np.log(TCH),
                        'lnCOVER':np.log(COVER),'lnCOVER_res':np.log(1+COVER_residual),
                        'TCH_SD':TCH_SD,'TCH_SDrel':TCH_SDrel,
                        'lnTCH_SD':np.log(TCH_SD),'lnTCH_SDrel':np.log(TCH_SDrel),
                        'ID':ID,'plot':host_plot})

#cal_data=cal_data[cal_data.AGB<290]

"""
UPDATE THIS PART TO PROVIDE A LIST OF MODELS TO USE
"""
models_to_test = ['lnAGB ~ lnTCH * lnCOVER_res',
                    'lnAGB ~ lnTCH + lnCOVER_res',
                    'lnAGB ~ lnTCH',
                    'AGB ~ TCH * COVER_res',
                    'AGB ~ TCH + COVER_res',
                    'AGB ~ TCH',
                    'AGB ~ TCH * TCH_SD',
                    'AGB ~ TCH + TCH_SD',
                    'lnAGB ~ lnTCH * lnTCH_SD',
                    'lnAGB ~ lnTCH + lnTCH_SD',
                    'AGB ~ TCH * TCH_SDrel',
                    'AGB ~ TCH + TCH_SDrel',
                    'lnAGB ~ lnTCH * lnTCH_SDrel',
                    'lnAGB ~ lnTCH + lnTCH_SDrel'
                    ]
# specify whether fit is undertaken in log-transformed space
log_log = [True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            True,
            True]

# note to use plot averages, use the suffix:
#                       .groupby('plot').mean())
best_rmse=np.inf
best_model = '_'
n_obs = cal_data.shape[0]
best_AGB_loo = np.zeros(n_obs)
log_fit = False
# loop through models to test
for ii,model in enumerate(models_to_test):
    # leave one out cross-validation
    AGB_mod_loo= np.zeros(n_obs)
    count = 0
    loo = LeaveOneOut()
    loo.get_n_splits(cal_data)

    for train_index,test_index in loo.split(cal_data):
        mod_iter = smf.ols(model,data=cal_data.iloc[train_index]).fit()
        AGB_mod_loo[count] = mod_iter.predict(cal_data.iloc[test_index])
        count+=1

    if log_log[ii]:
        # Baskerville (1972) correction factor for bias in untransformed mean
        CF = st.calculate_baskervilleCF(cal_data['lnAGB'],AGB_mod_loo)
        AGB_mod_loo = CF*np.exp(AGB_mod_loo)

    _,_,r,_,_=stats.linregress(cal_data.AGB,AGB_mod_loo)
    r2_score = r**2
    rmse_score = np.sqrt(np.mean((cal_data.AGB-AGB_mod_loo)**2))

    print('%s\tr2 = %.3f; rmse = %.2f' % (model,r2_score,rmse_score))
    if rmse_score<best_rmse:
        best_rmse=rmse_score
        best_model = model
        best_r2 = r2_score
        best_AGB_loo = AGB_mod_loo.copy()
        log_fit = log_log[ii]

# fit best
# print summary to screen
ols = smf.ols(best_model,data=cal_data)
results = ols.fit()
st_, fit_data, ss2 = summary_table(results)
fittedvalues = fit_data[:,2]
predict_mean_se  = fit_data[:,3]
predict_mean_ci_low, predict_mean_ci_upp = fit_data[:,4:6].T
predict_ci_low, predict_ci_upp = fit_data[:,6:8].T

print(results.summary())

if log_fit:
    CF = st.calculate_baskervilleCF(cal_data['lnAGB'],fittedvalues)
    fittedvalues = CF*fittedvalues

"""
Plot calibrated model against inventory
"""
cal_df = pd.DataFrame({'mod':best_AGB_loo,'obs':cal_data['AGB'],
                        'predict_ci_low':predict_ci_low,
                        'predict_ci_upp':predict_ci_upp,
                        'ID':cal_data['ID']})

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[5,5])
ax.plot([0,cal_df['obs'].max()],[0,cal_df['obs'].max()],':',color='red')
ax.plot(cal_df['obs'],cal_df['mod'],'.',color='black')
ax.set_title('LiDAR AGB vs. inventory AGB')
ax.set_ylabel('LiDAR AGB / Mg ha$^{-1}$')
ax.set_xlabel('field AGB / Mg ha$^{-1}$')
ax.set_aspect('equal')
ax.annotate('RMSE = %.2f Mg ha$^{-1}$\n$r^2$ = %.3f' % (best_rmse,best_r2),xy=(0.95,0.05),
                xycoords='axes fraction',ha='right',va='bottom')

fig.show()
fig.tight_layout()
fig.savefig('%sfield_AGB_vs_LiDAR_AGB_LOO.png' % path2fig)

"""
Plot model for range of values observed across the domain
THIS WILL NEED UPDATING DEPENDING ON THE VARIABLES USED IN THE MODEL
"""
# first create grids at 20 m resolution
cover = xr.open_rasterio(raster_file)[0]
cover.values[cover.values<gap_ht*100]=0
cover.values[cover.values>=gap_ht*100]=1
io.write_xarray_to_GeoTiff(cover,'cover_temp')
cover.close()
os.system('gdalwarp -overwrite -tr 20 20 -r average %s chm20.tif' % raster_file)
os.system('gdalwarp -overwrite -tr 20 20 -r average cover_temp.tif cover20.tif')

chm20 = xr.open_rasterio('chm20.tif')[0].sel(x=slice(229000,239000),y=slice(2230310,2214000)).astype('float')
chm20.values/=100
mask = chm20.values>=0
chm20.values[~mask]=np.nan

cover20 = xr.open_rasterio('cover20.tif')[0].sel(x=slice(229000,239000),y=slice(2230310,2214000))
cover20.values[~mask]=np.nan

cover20_res = chm20.copy(deep=True)
cover20_res.values = cover20.values - (1/(1 + np.exp(-ro0) * cover20.values**(-ro1)))

cover_res_grid =  1/(1 + np.exp(-ro0) * cover20.values**(-ro1))
"""
THIS WILL NEED UPDATING DEPENDING ON VARIABLES USED IN THE MODEL
"""
domain_data = pd.DataFrame({'TCH':chm20.values[mask],'COVER':cover20.values[mask],
                        'COVER_res':cover_res_grid[mask],'lnTCH':np.log(chm20.values[mask]),
                        'lnCOVER_res':np.log(1+cover_res_grid[mask])})
agb_model = chm20.copy(deep=True)
if log_fit:
    agb_model.values[mask] = CF*np.exp(results.predict(domain_data))
else:
    agb_model.values[mask] = results.predict(domain_data)


fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[6,4])
scatter = ax.scatter(domain_data.TCH,domain_data.COVER,c=agb_model.values[mask],
                    marker='.',vmin=0)
ax.set_xlabel('TCH / m')
ax.set_ylabel('Canopy cover fraction at 8.66 m')
plt.colorbar(scatter,label='Simulated AGB / Mg ha$^{-1}$')
fig.tight_layout()
fig.show()
fig.savefig('%smodel_AGB_across_parameter_space.png' % path2fig)

"""
SAVE DETAILS OF MODEL FOR FUTURE USE
"""
pt2_output = {'model':best_model,'cal_data':cal_data,'log_fit':log_fit}
np.savez(pt2_outfile,pt2_output)
