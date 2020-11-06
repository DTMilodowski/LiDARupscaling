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
import statsmodels as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table
from sklearn.model_selection import LeaveOneOut

import stats_tools as st

import sys
import os
import copy as cp
import gc

sys.path.append('../data_io/')
import data_io as io

# CHANGE THIS STUFF
# Some file names - shapefiles and rasters should be in a projected coordinate system (i.e. UTM)
#inventory_file = '../../data/lidar_calibration/Kiuic_400_live_trees.shp'
version = '035'
inventory_file = '../../data/lidar_calibration/Kiuic_400_live_biomass_unc.shp'
pt1_outfile = '../../data/lidar_calibration/kiuic_plot_lidar_sample_%s.npz' % version # this will be a file to hold the compiled plot data
pt2_outfile = '../../saved_models/lidar_calibration/lidar_calibration_pt2_results_%s.npz' % version
path2fig = '../../figures/'
plot_area = 400. # 1 ha
radius = np.sqrt(plot_area/np.pi)

# LOAD FILES, CONVERT TO FLOAT AND SPECIFY NODATA REGIONS
pt1_output = np.load(pt1_outfile,allow_pickle=True)['arr_0'][()]
chm_results = pt1_output['chm']
dem_results = pt1_output['dem']
inventory_AGB = pt1_output['inventory']

# COLLATE DATA INTO ARRAYS
ID=[]; AGB = []; AGB_UNC = []; TCH = []; COVER2 = []; NODATA = []; TCH_SD = [];
TCH_all = []; QUANTILES = []; QUANTILES_all = []; COVER_all = [];
COVER10=[]; COVERmean=[]; H95=[]
CI95_l=[];CI95_u=[];CI50_l=[];CI50_u=[]
COLLECTION = []
for plot in chm_results.keys():
    id = chm_results[plot]['id']
    if id not in ['55410.1','55410.2','55410.3','55410.4','9.8','56635.2','56635.3']:
        if chm_results[id]['status']=='PASS':
            if chm_results[id]['weighted_average']>0:
                if np.sum(np.isnan(chm_results[id]['raster_values'][0])*chm_results[id]['weights'])<0.05*np.sum(chm_results[id]['weights']):
                    weights = chm_results[id]['weights'].values/np.sum(np.isfinite(chm_results[id]['raster_values'][0].values)*chm_results[id]['weights'].values)
                    mean_tch = chm_results[id]['weighted_average'][0]
                    ID.append(id)
                    AGB.append(inventory_AGB[id]['AGB'])
                    AGB_UNC.append(inventory_AGB[id]['uncertainty'])
                    #TCH.append(mean_tch)
                    TCH.append(np.median(chm_results[id]['tch_mc']))
                    TCH_all.append(chm_results[id]['tch_mc'].copy())
                    QUANTILES_all.append(chm_results[id]['quantiles_mc'].copy())
                    QUANTILES.append(chm_results[id]['quantiles'].copy())
                    TCH_SD.append(np.sqrt(np.nansum(weights*(chm_results[id]['raster_values'][0].values-mean_tch)**2)))

                    pts_sub = chm_results[id]['point_cloud'].copy()
                    point_heights = chm_results[id]['point_heights'].copy()
                    point_weights = 1/pts_sub[:,7]

                    COVER2.append(chm_results[id]['cover_fraction_from_pointcloud'])
                    COVER10.append(np.sum(point_weights[point_heights>=10]/np.sum(point_weights)))
                    COVERmean.append(np.sum(point_weights[point_heights>=mean_tch]/np.sum(point_weights)))

                    COVER_all.append(chm_results[id]['cover_fraction_from_pointcloud_mc'].copy())
                    NODATA.append(np.mean(~np.isfinite(chm_results[id]['raster_values'][0].values)))

                    H95.append(np.percentile(chm_results[id]['tch_mc'],97.5)-np.percentile(chm_results[id]['tch_mc'],2.5))

                    CI95_l.append(np.percentile(chm_results[id]['tch_mc'],2.5))
                    CI95_u.append(np.percentile(chm_results[id]['tch_mc'],97.5))
                    CI50_l.append(np.percentile(chm_results[id]['tch_mc'],25))
                    CI50_u.append(np.percentile(chm_results[id]['tch_mc'],75))


                    if len(id)>5:
                        COLLECTION.append(1)
                    else:
                        COLLECTION.append(0)


# Convert to np arrays for conditional indexing convenience
AGB = np.asarray(AGB); AGB_UNC = np.asarray(AGB_UNC); TCH = np.asarray(TCH);
TCH_SD = np.asarray(TCH_SD); ID = np.asarray(ID); TCH_all = np.asarray(TCH_all)
QUANTILES_all = np.asarray(QUANTILES_all); QUANTILES = np.asarray(QUANTILES)
COVER2 = np.asarray(COVER2); COVER_all = np.asarray(COVER_all); COVER10 = np.asarray(COVER10);
COVERmean = np.asarray(COVERmean); H95=np.asarray(H95); COLLECTION = np.asarray(COLLECTION)
TCH_SDrel = TCH_SD/TCH
CI95_l=np.array(CI95_l);CI95_u=np.array(CI95_u)
CI50_l=np.array(CI50_l);CI50_u=np.array(CI50_u)

host_plot = []
for id in ID:
    host_plot.append(id[:-2])
host_plot = np.asarray(host_plot)

"""
MAY WANT TO INCREASE THE NUMBER OF ADDITIONAL METRICS
# calculate the additional metrics as reqiured here
"""
# calculate residual gap fraction based on COVER-AGB relationship
mask = np.isfinite(np.log(COVER2/(1-COVER2)))
ro1,ro0,r,p,_=stats.linregress(np.log(TCH[mask]),np.log(COVER2[mask]/(1-COVER2[mask])))
COVER_residual2 = COVER2- (1/(1 + np.exp(-ro0) * TCH**(-ro1)))

mask = np.isfinite(np.log(COVER10/(1-COVER10)))
ro1,ro0,r,p,_=stats.linregress(np.log(TCH[mask]),np.log(COVER10[mask]/(1-COVER10[mask])))
COVER_ = 1/(1 + np.exp(-ro0) * TCH**(-ro1))
COVER_residual10 = COVER10- COVER_

"""
UPDATE THIS PART TO PROVIDE A LIST OF MODELS TO USE
"""
#models_to_test = ['lnAGB ~ lnTCH + Collection']
models_to_test = ['lnAGB ~ lnTCH']#,
#'lnAGB ~ lnTCH + Collection + COVER_res10',
#'AGB ~ TCH + Collection',
#'AGB ~ TCH*Collection',
#'AGB ~ TCH + COVERmean + Collection',
#'AGB ~ TCH + COVER_res2 + Collection',
#'AGB ~ TCH + COVER_res10 + Collection']

# specify whether fit is undertaken in log-transformed space
log_log = [True,
            True,
            False,
            False,
            False,
            False,
            False]

test_qqplot = [True,
                True,
                True,
                True,
                True,
                True,
                True]

"""
NO MC VERSION
"""
n_models=len(models_to_test)
"""
X = np.hstack((QUANTILES,TCH.reshape(-1,1)))
X= np.hstack((X,COVERmean.reshape(-1,1)))
Xpca = pca.transform(X)
"""
cal_data = pd.DataFrame({'AGB':AGB,
                        'TCH':TCH,
                        'COVERmean':COVERmean,
                        'COVER2':COVER2,
                        'COVER10':COVER10,
                        'COVER_res2':COVER_residual2,
                        'COVER_res10':COVER_residual10,
                        'TCH_squared':TCH**2,
                        'lnAGB':np.log(AGB),
                        'lnTCH':np.log(TCH),
                        'plot':host_plot,
                        'Collection':COLLECTION})

"""
# Model selection - run LOO cross validation to find best model from predefined
# candidates. Include a residual check if desired (set test_qqplot=True)
"""
results={}
n_obs = cal_data.shape[0]
for mm,model in enumerate(models_to_test):
    results[model]={}
    # note to use plot averages, use the suffix:
    #                       .groupby('plot').mean())
    # leave one out cross-validation
    AGB_mod_loo= np.zeros(n_obs)
    count = 0
    loo = LeaveOneOut()
    loo.get_n_splits(cal_data)
    for train_index,test_index in loo.split(cal_data):
        # fit a linear mixed effects model that incorporates the plot cluster as
        # a random effect
        mod_iter = smf.mixedlm(model,data=cal_data.iloc[train_index],
                                    groups='plot').fit()
        AGB_mod_loo[count] = mod_iter.predict(cal_data.iloc[test_index])
        count+=1

    if log_log[mm]:
        # Baskerville (1972) correction factor for bias in untransformed mean
        CF = st.calculate_baskervilleCF(cal_data['lnAGB'],AGB_mod_loo)
        AGB_mod_loo = CF*np.exp(AGB_mod_loo)

    _,_,r,_,_=stats.linregress(cal_data.AGB,AGB_mod_loo)
    results[model]['r2_score'] = r**2
    results[model]['rmse_score'] = np.sqrt(np.mean((cal_data.AGB-AGB_mod_loo)**2))

    # fit full model
    mod_full = smf.mixedlm(model,data=cal_data,groups='plot').fit()
    results[model]['residual'] = mod_full.resid
    print(model,'%.3f' % np.mean(results[model]['r2_score']))

    # plot qq plot if desired
    if test_qqplot[mm]:
        fig1,ax1 = plt.subplots(nrows=1,ncols=1,figsize=[4,4])
        st.qq_plot(mod_full.resid[COLLECTION==0],fig=fig1,ax=ax1,ci=.95,ylabel='residual',show=True)

# final candidate
#model = 'lnAGB ~ lnTCH + Collection'
model = 'lnAGB ~ lnTCH'
log_log=True

# leave one out cross-validation
n_iter = TCH_all.shape[1]
AGB_mod_loo= np.zeros((n_obs,n_iter))
loo = LeaveOneOut()
np.random.seed(29)
for ii in range(0,n_iter):
    AGB_iter=AGB+np.random.randn(AGB.size)*AGB_UNC
    AGB_iter[AGB_iter<0]=AGB[AGB_iter<0]

    cal_data_mc = pd.DataFrame({'AGB':AGB_iter,
                                'TCH':TCH_all[:,ii],
                                'lnAGB':np.log(AGB),
                                'lnTCH':np.log(TCH_all[:,ii]),
                                'plot':host_plot})

    loo.get_n_splits(cal_data_mc)
    for train_index,test_index in loo.split(cal_data_mc):
        # fit a linear mixed effects model that incorporates the plot cluster as
        # a random effect
        mod_iter = smf.mixedlm(model,data=cal_data_mc.iloc[train_index],
                                    groups='plot').fit()
        AGB_mod_loo[test_index,ii] = mod_iter.predict(cal_data_mc.iloc[test_index])

    if log_log:
        # Baskerville (1972) correction factor for bias in untransformed mean
        CF = st.calculate_baskervilleCF(cal_data_mc['lnAGB'],AGB_mod_loo[:,ii])
        AGB_mod_loo[:,ii] = CF*np.exp(AGB_mod_loo[:,ii])

_,_,r,_,_=stats.linregress(cal_data.AGB,np.median(AGB_mod_loo,axis=1))
r2_score = r**2
rmse_score = np.sqrt(np.mean((cal_data.AGB-np.median(AGB_mod_loo,axis=1))**2))

# fit full model
mod_full = smf.mixedlm(model,data=cal_data,groups='plot').fit()
if log_log:
    # Baskerville (1972) correction factor for bias in untransformed mean
    CF = st.calculate_baskervilleCF(cal_data['lnAGB'],mod_full.fittedvalues)

cal_data['AGBmod'] = pd.Series(CF*np.exp(mod_full.fittedvalues), index=cal_data.index)
cal_data.sort_values(by=['TCH'],inplace=True)

test_data = cal_data.copy()
test_data['plot'][:]=0
test_data['AGBmod'] = pd.Series(CF*np.exp(mod_full.predict(test_data)), index=test_data.index)
"""
# plot final model and observed vs LOO prediction
fig2, axes = plt.subplots(nrows=1,ncols=2,figsize = [8,4])
axes[0].plot(cal_data.TCH[cal_data.Collection==0],cal_data.AGB[cal_data.Collection==0],'.',color='red')
axes[0].plot(cal_data.TCH[cal_data.Collection==1],cal_data.AGB[cal_data.Collection==1],'.',color='blue')
axes[0].plot(cal_data.TCH[cal_data.Collection==0],test_data.AGBmod[cal_data.Collection==0],'-',color='red')
axes[0].plot(cal_data.TCH[cal_data.Collection==1],test_data.AGBmod[cal_data.Collection==1],'-',color='blue')
axes[0].errorbar(TCH,AGB,xerr=(TCH-CI95_l,CI95_u-TCH),
                    marker='',linestyle='',color='0.67',linewidth=0.5)
axes[0].errorbar(TCH,AGB,xerr=(TCH-CI50_l,CI50_u-TCH),yerr=AGB_UNC,
                    marker='',linestyle='',color='0.33',linewidth=0.75)

axes[0].set_title('TCH vs. field AGB')
axes[0].set_ylabel('Modelled AGB (LOO) / Mg ha$^{-1}$')
axes[0].set_xlabel('TCH / m')

axes[1].plot([0,np.max(AGB)],[0,np.max(AGB)],'--',color='0.5')
axes[1].plot(AGB[COLLECTION==0],AGB_mod_loo[COLLECTION==0],'.',color='red')
axes[1].plot(AGB[COLLECTION==1],AGB_mod_loo[COLLECTION==1],'.',color='blue')
axes[1].errorbar(AGB,AGB_mod_loo,xerr=(AGB_UNC),
                    marker='',linestyle='',color='0.33',linewidth=0.75)

axes[1].set_title('field AGB vs. LOO prediction')
axes[1].set_xlabel('field AGB / Mg ha$^{-1}$')
axes[1].set_ylabel('Modelled AGB (LOO) / Mg ha$^{-1}$')
axes[1].annotate('RMSE = %.2f Mg ha$^{-1}$\n$r^2$ = %.3f' % (rmse_score,r2_score),
                    xy=(0.95,0.05),xycoords='axes fraction',ha='right',va='bottom')

fig2.tight_layout()
fig2.savefig('montecarlo_fitted_model_and_LOO_%s.png' % version)
fig2.show()
"""
"""
equivalent, but only with the LOO validation
"""
fig2, ax = plt.subplots(nrows=1,ncols=1,figsize = [4,4])
xerr = (np.median(AGB_mod_loo,axis=1)-np.percentile(AGB_mod_loo,2.5,axis=1),
        np.percentile(AGB_mod_loo,97.5,axis=1)-np.median(AGB_mod_loo,axis=1))
ax.plot([0,np.max(AGB)],[0,np.max(AGB)],'--',color='0.5',lw=0.8)
ax.plot(np.median(AGB_mod_loo[COLLECTION==0],axis=1),AGB[COLLECTION==0],'.',color='black',ms=5)
ax.plot(np.median(AGB_mod_loo[COLLECTION==1],axis=1),AGB[COLLECTION==1],'.',mec='black',mfc='white',ms=5)
ax.errorbar(np.median(AGB_mod_loo,axis=1),AGB,yerr=(AGB_UNC),xerr=xerr,
                    marker='',linestyle='',color='0.33',linewidth=0.75)

ax.set_title('field AGB vs. LOO prediction')
ax.set_ylabel('field AGB / Mg ha$^{-1}$')
ax.set_xlabel('Modelled AGB (LOO) / Mg ha$^{-1}$')
ax.annotate('RMSE = %.2f Mg ha$^{-1}$\n$r^2$ = %.3f' % (rmse_score,r2_score),
                    xy=(0.95,0.05),xycoords='axes fraction',ha='right',va='bottom')

fig2.tight_layout()
fig2.savefig('lidar_AGB_LOO_cross_validation_%s.png' % version)
fig2.show()


print( np.sum( np.all((AGB+2*AGB_UNC>=np.percentile(AGB_mod_loo,2.5,axis=1),
                        AGB-2*AGB_UNC<=np.percentile(AGB_mod_loo,97.5,axis=1)),axis=0) )/AGB.size )


"""
Plot fitted model onto TCH-AGB relationship with subplot sample
"""

# Plot the figure with montecarlo postion errors
plots_to_plot = ['24.1','15.4','5.2','6.1','3.1']
import matplotlib.patches as mpatches
fig,fig_axes = plt.subplots(nrows=2,ncols=3,figsize=[12,7])
axes=fig_axes.flatten()
axes[0].plot(cal_data.TCH,test_data.AGBmod,'-',color='black',lw=0.8)
#axes[0].plot(cal_data.TCH[cal_data.Collection==1],test_data.AGBmod[cal_data.Collection==1],'--',color='black',lw=0.8)
axes[0].plot(TCH[COLLECTION==0],AGB[COLLECTION==0],'.',color='black',ms=5)
axes[0].plot(TCH[COLLECTION==1],AGB[COLLECTION==1],'.',mec='black',mfc='white',ms=5)
axes[0].errorbar(TCH,AGB,xerr=(TCH-CI95_l,CI95_u-TCH),
                    marker='',linestyle='',color='0.67',linewidth=0.5)
axes[0].errorbar(TCH,AGB,xerr=(TCH-CI50_l,CI50_u-TCH),yerr=AGB_UNC,
                    marker='',linestyle='',color='0.33',linewidth=0.75)
axes[0].set_title('TCH vs. inventory AGB')
axes[0].set_xlabel('mean TCH / m')
axes[0].set_ylabel('field AGB / Mg ha$^{-1}$')
for ii,plot_id in enumerate(plots_to_plot):
    plot_boundary = mpatches.Circle((0,0),radius,ec='white',fill=False)
    chm_results[plot_id]['chm_raster'].plot(ax=axes[ii+1], vmin=0, vmax=21,
                extend='max', cbar_kwargs={'label':'height / m'})
    axes[ii+1].add_artist(plot_boundary)
    axes[ii+1].set_title('plot %s; AGB = %.1f Mg/ha' % (plot_id,AGB[ID==plot_id]))
    axes[0].plot(TCH[ID==plot_id],AGB[ID==plot_id],'.',color='#2db27d')#'#bc1655')
    axes[0].annotate(' %s' % plot_id,xy=(TCH[ID==plot_id],AGB[ID==plot_id]),color='#2db27d')
    if ii>=2:
        axes[ii+1].set_xlabel('distance / m')
    else:
        axes[ii+1].set_xlabel('')
    axes[ii+1].set_ylabel('')
fig.tight_layout()
fig.savefig('%slidar_TCH_vs_field_AGB_400m_after_mc_%s.png' % (path2fig,version))
fig.show()


"""
MONTE CARLO UNCERTAINTY PROPAGATION
"""

mc_results={}
mc_iterations = 100
nan_array = np.zeros(mc_iterations)*np.nan
np.random.seed(29)
mc_results['model']=model
mc_results['r2_scores']=nan_array.copy()
mc_results['rmse_scores']=nan_array.copy()
mc_results['params']=np.zeros((mc_iterations,3))*np.nan
mc_results['CF']=nan_array.copy()
mc_results['fitted_models']=[]
for ii in range(0,mc_iterations):
    AGB_iter=AGB+np.random.randn(AGB.size)*AGB_UNC
    AGB_iter[AGB_iter<0]=AGB[AGB_iter<0]
    TCH_iter=TCH_all[:,ii]
    cal_data = pd.DataFrame({'AGB':AGB_iter,
                            'TCH':TCH_iter,
                            'lnAGB':np.log(AGB_iter),
                            'lnTCH':np.log(TCH_iter),
                            'Collection':COLLECTION,
                            'plot':host_plot})

    mod_iter = smf.mixedlm(model,data=cal_data,groups=cal_data['plot']).fit()

    if log_log:
        # Baskerville (1972) correction factor for bias in untransformed mean
        CF = st.calculate_baskervilleCF(cal_data['lnAGB'],mod_iter.fittedvalues)
        mc_results['CF'][ii]=CF

    AGBmod_iter = CF*mod_iter.fittedvalues
    _,_,r,_,_=stats.linregress(cal_data.AGB,AGBmod_iter)

    mc_results['r2_scores'][ii] = r**2
    mc_results['rmse_scores'][ii] = np.sqrt(np.mean((cal_data.AGB-AGBmod_iter)**2))
    mc_results['params'][ii]=mod_iter.params
    mc_results['fitted_models'].append(mod_iter)
print(model,'%.3f' % np.median(mc_results['r2_scores']))

"""
plot histogram of montecarlo parameters
"""
fig3,axes=plt.subplots(nrows=1,ncols=3,figsize=[8,3])
#axes_ = axes.flatten()
sns.distplot(mc_results['params'][:,0],bins=20,ax=axes[0])
sns.distplot(mc_results['params'][:,1],bins=20,ax=axes[1])
sns.distplot(mc_results['params'][:,2],bins=20,ax=axes[2])

axes[0].annotate('$ln(AGB) = a + b*ln(TCH)$\n$+ R.E.(cluster)$',fontsize=8,
                    xy=(0.95,0.95),xycoords='axes fraction',ha='right',va='top')
axes[0].set_xlabel('a')
axes[1].set_xlabel('b')
#axes_[2].set_xlabel('c')
axes[2].set_xlabel('var(cluster)')
fig3.tight_layout()
fig3.savefig('montecarlo_fit_parameters_%s.png' % version)
fig3.show()

"""
SAVE DETAILS OF MODEL FOR FUTURE USE
"""
pt2_output = {'cal_data':cal_data,'mc_results':mc_results}
np.savez(pt2_outfile,pt2_output)
