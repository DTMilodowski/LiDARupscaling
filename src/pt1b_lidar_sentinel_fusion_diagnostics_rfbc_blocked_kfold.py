"""
pt1_lidar_sentinel_fusion_train_rf_random_bayesian.py
--------------------------------------------------------------------------------
FITTING RANDOM FOREST MODEL TO LINK SENTINEL LAYERS TO LIDAR ESTIMATED AGB
This script loads the predictor (sentinel bands and derivatives) and target
(lidar estimated AGB) variables, calibrates and validates a random forest
regression model, and fits a final model using te full training set.

The random forest algroithm is optimised using a two-step process: 1st a
randomized search is undertaken to locate a reasonable starting point; then a
bayesian optimiser (TPE) is used to refine the parameterisation.

Feature importance and cross validation analysis, implemented with
buffered-blocked k-fold strategy

This code built using the open source programming language python, and utilises
the geospatial library xarray (http://xarray.pydata.org/en/stable/) and the
machine learning library scikit-learn
(https://scikit-learn.org/stable/index.html).

24/04/2019 - D. T. Milodowski
--------------------------------------------------------------------------------
"""

"""
# Import the necessary packages
"""
import numpy as np                  # standard package for scientific computing
import xarray as xr                 # xarray geospatial package
import pandas as pd                 # data frames
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package
import os
import fiona
from scipy import stats
from scipy import ndimage as image

# Import some parts of the scikit-learn library
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import pickle

# Import custom libaries

import sys
sys.path.append('./random_forest/')
sys.path.append('./data_io/')
sys.path.append('./data_visualisation/')

import data_io as io
import general_plots as gplt
import random_forest_functions as rff
import cal_val as cv
import utility

"""
Project Info
"""
site_id = 'kiuic'
version_trials='034'
version = '035'
resolution = '20'
path2alg = '../saved_models/'
if(os.path.isdir(path2alg)==False):
    os.mkdir(path2alg)
path2fig= '../figures/'
if(os.path.isdir(path2fig)==False):
    os.mkdir(path2fig)

"""
#===============================================================================
PART A: LOAD IN DATA AND SUBSET THE TRAINING DATA
Load data
Filter areas where we have LiDAR estimates
Subsample if desired/required
#-------------------------------------------------------------------------------
"""
print('Loading data')

# load a template raster
lidar_agb_file = '../data/lidar_calibration/%sm/kiuic_lidar_agb_%s_median.tif' % (resolution.zfill(3),version_trials)
lidar = io.load_geotiff(lidar_agb_file,option=1)
target=lidar.values.copy()
target[target<0] = np.nan

# Load predictors & target
data_layers,data_mask,labels = io.load_predictors(layers=['sentinel2','alos'],
                                                    resolution=resolution)
#layers_to_remove = ['ASM','homogeneity','correlation']
#layers_to_remove = ['ASM','homogeneity','correlation','contrast','dissimilarity']
layers_to_remove = []
n_predictors = data_layers.shape[0]
layer_mask = np.ones(n_predictors,dtype='bool')
labels_update=[]
for ii,lab in enumerate(labels):
    for layer in layers_to_remove:
        if layer in lab:
            print('remove', lab)
            layer_mask[ii] = False
    if layer_mask[ii]:
        labels_update.append(lab)
data_layers = data_layers[layer_mask]
labels = labels_update
n_predictors = data_layers.shape[0]
print(labels)
if resolution=='100':
    data_layers=data_layers[:,:,:-1]
    data_mask=data_mask[:,:-1]
# load forest mask
forest_mask_file = "../data/forest_mask/%s_forest_mask_%sm.tif" % (site_id,resolution.zfill(3))
forest = xr.open_rasterio(forest_mask_file).values[0]
forest_mask=forest==1
forest_mask = forest_mask*data_mask

# Keep only areas for which we have biomass estimates
training_mask = np.isfinite(target)
training_mask = image.binary_erosion(training_mask,iterations=1)
training_mask = training_mask*forest_mask

# Apply masks to the predictor dataset to be ingested into sklearn routines
predictors = io.apply_mask_to_raster_stack(data_layers,forest_mask)
X = predictors[training_mask[forest_mask],:]
y = target[training_mask]

# load in the optimisation results
# load the trials data
trials = pickle.load(open('%s%s_%s_rfbc_sentinel_lidar_agb_trials.p' % (path2alg,site_id,version_trials), "rb"))

# double check the number of accepted parameter sets
best_score = np.inf
for ii,tt in enumerate(trials.trials):
    if tt['result']['status']=='ok':
        if tt['result']['loss'] < best_score:
            best_score = tt['result']['loss']
            best_params=tt['misc']['vals']

rf = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth= int(best_params['max_depth'][0]),            # ***maximum number of branching levels within each tree
            max_features=int(best_params['max_features'][0]),       # ***the maximum number of variables used in a given tree
            max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
            min_impurity_split=None,   # threshold impurity within an internal node before it will be split
            min_samples_leaf=int(best_params['min_samples_leaf'][0]),       # ***The minimum number of samples required to be at a leaf node
            min_samples_split=int(best_params['min_samples_split'][0]),       # ***The minimum number of samples required to split an internal node
            n_estimators=int(best_params['n_estimators'][0]), # ***Number of trees in the random forest
            n_jobs=20,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            random_state=2940,         # seed used by the random number generator
            )

"""
#===============================================================================
PART B: VALIDATION
- Observed vs. modelled biomass using the blocked buffered strategy to avoid
  bias due to spatial autocorrelation
#-------------------------------------------------------------------------------
"""
# Set up k-fold cross validation
block_res = 1000
buffer_width = 250
k=8
np.random.seed(1000)
cal_blocks,val_blocks = cv.get_k_fold_cal_val_blocked(lidar,block_res,buffer_width,training_mask=training_mask,k=k)
# Take best hyperparameter set and apply cal-val on full training set
print('Applying buffered k-fold cross validation')
rfbc_k = {}
y_obs = np.zeros(y.size)
y_mod = np.zeros(y.size)
index = 0
for ii in range(0,k):
    n = val_blocks['iter%i' % (ii+1)].sum()
    print(n)
    rfbc_k['rfbc%i' % (ii+1)] = {}
    rfbc_k['rfbc%i' % (ii+1)]['rf1'],rfbc_k['rfbc%i' % (ii+1)]['rf2'] = rff.rfbc_fit(rf,X[cal_blocks['iter%i' % (ii+1)]],y[cal_blocks['iter%i' % (ii+1)]])
    y_mod[index:index+n] =  rff.rfbc_predict(rfbc_k['rfbc%i' % (ii+1)]['rf1'],rfbc_k['rfbc%i' % (ii+1)]['rf2'],X[val_blocks['iter%i' % (ii+1)]])
    y_obs[index:index+n] = y[val_blocks['iter%i' % (ii+1)]]
    index+=n
temp1,temp2,r,temp3,temp4 = stats.linregress(y_obs,y_mod)
r2 = r**2
rmse = np.sqrt(np.mean((y_mod-y_obs)**2))
rel_rmse = rmse/np.mean(y_obs)
print("Validation\n\tR^2 = %.02f" % r2)
print("\tRMSE = %.02f" % rmse)
print("\trelative RMSE = %.02f" % rel_rmse)
annotation = 'R$^2$ = %.2f\nRMSE = %.1f Mg ha$^{-1}$\nrelative RMSE = %.1f%s' % (r2,rmse,rel_rmse*100,'%')
fig1, axes1 = gplt.plot_validation(y_obs,y_mod,annotation=annotation)
fig1.savefig('%s%s_%s_%sm_validation_blocked_kfold.png' % (path2fig,site_id,version,resolution.zfill(3)))

"""
#===============================================================================
PART C: FEATURE IMPORTANCE
- Feature importance calculated based on fraction of explained variance (i.e.
  fractional drop in R^2) on random permutation of each predictor variable.
- Five iterations, with mean and standard deviation reported and plotted.
#-------------------------------------------------------------------------------
"""
n_iter=3
n = y.size
score_drops = []
var_labels=[]

# First define the score random_forest_functions as fractional decrease in
# variance explained
def r2_score(X,y):
    y_obs = np.zeros(y.size)
    y_mod = np.zeros(y.size)
    index = 0
    for ii in range(0,k):
        n = val_blocks['iter%i' % (ii+1)].sum()
        y_mod[index:index+n] = rff.rfbc_predict(rfbc_k['rfbc%i' % (ii+1)]['rf1'],rfbc_k['rfbc%i' % (ii+1)]['rf2'],X[val_blocks['iter%i' % (ii+1)]])
        y_obs[index:index+n] = y[val_blocks['iter%i' % (ii+1)]]
    return r**2

# Additional importance estimates that holistically consider the impact of
# permuting all the layers from a given sensor
alos_mask = np.zeros(n_predictors,dtype='bool')
sentinel_mask = np.zeros(n_predictors,dtype='bool')
sentinel_labs = ['b1','b2','b3','b4','ndvi']
#texture_labs = ['value','contrast','correlation','dissimilarity','entropy','homogeneity','mean','second_moment','variance']
#texture_labs_alt = ['value','cont','corr','diss','ent','hom','mean','s_m_','var']
#texture_labs_display = ['value','contrast','correlation','dissimilarity','entropy','homogeneity','mean','second moment','variance']

texture_labs = ['mean','variance','contrast','correlation','dissimilarity','homogeneity','ASM']
texture_labs_alt = ['enlee_20m','var','contr','corr','diss','hom','asm']
texture_labs_display = ['mean','variance','contrast','correlation','dissimilarity','homogeneity','ASM']
for ii,lab in enumerate(labels):
    if 'alos' in lab:
        alos_mask[ii] = True
    for ll in sentinel_labs:
        if ll in lab:
            sentinel_mask[ii] = True

print('\tpermutation importance...')
base_score = r2_score(X,y)
for ii in range(n_iter):
    print('\t\t...interation %i or %i, base_score = %.2f' % (ii+1,n_iter,base_score))
    X_sent = X.copy()
    X_alos = X.copy()
    permutation_index = np.random.choice(np.arange(n,dtype='int'),size=n,replace='False')
    X_permute = X[permutation_index]
    # first shuffle sentinel
    X_sent[:,sentinel_mask]=X_permute[:,sentinel_mask]
    # next shuffle alos
    X_alos[:,alos_mask]=X_permute[:,alos_mask]

    # importance calculation
    score_drops.append(base_score-r2_score(X_sent,y))
    var_labels.append('all sentinel')
    score_drops.append(base_score-r2_score(X_alos,y))
    var_labels.append('all alos')
    """
    # individual sentinel bands
    for bb,band in enumerate(sentinel_labs):
        X_shuffle = X.copy()
        for ll,lab in enumerate(labels):
            if band in lab:
                X_shuffle[:,ll]=X_permute[:,ll]
        score_drops.append(base_score-r2_score(X_shuffle,y))
        var_labels.append('all %s' % band)
    """
    # textures
    for tt,texture in enumerate(texture_labs):
        X_shuffle = X.copy()
        for ll,lab in enumerate(labels):
            #if texture == 'value':
            #    if len(lab) <20:
            #        X_shuffle[:,ll]=X_permute[:,ll]
            #el
            if texture in lab:
                X_shuffle[:,ll]=X_permute[:,ll]
            elif texture_labs_alt[tt] in lab:
                X_shuffle[:,ll]=X_permute[:,ll]
        score_drops.append(base_score-r2_score(X_shuffle,y))
        var_labels.append(texture_labs_display[tt])

    """
    # now loop through the individual variables
    for ll,lab in enumerate(labels):
        X_shuffle = X.copy()
        X_shuffle[:,ll]=X_permute[:,ll]
        score_drops.append(base_score-r2_score(X_shuffle,y))
        var_labels.append(lab)
    """
# Plot importances
imp_df = pd.DataFrame(data = {'variable': var_labels,
                            'permutation_importance': score_drops})

fig2,axes = gplt.plot_permutation_importances(imp_df,show=True,figsize=[6,5],emphasis=['all sentinel','all alos'])
fig2.savefig('%s%s_%s_%sm_permutation_importances_by_texture.png' % (path2fig,site_id,version,resolution.zfill(3)))

variable_mask = np.zeros(len(imp_df),dtype='bool')
for ii,var in enumerate(imp_df['variable']):
    if "all" in var:
        variable_mask[ii]=True

fig3,axes = gplt.plot_permutation_importances(imp_df[variable_mask],show=True,emphasis=['all sentinel','all alos'],figsize=[6,5])
fig3.savefig('%s%s_%s_%sm_permutation_importances_summary.png' % (path2fig,site_id,version,resolution.zfill(3)))


"""
Aggregation to 1ha
"""

lidar_cal = np.load('../../saved_models/lidar_calibration/lidar_calibration_pt2_results_%s.npz' % version, allow_pickle=True)['arr_0'][()]['loo_results']

model_20m = lidar.copy(deep=True)
model_20m.values*=np.nan
placement = model_20m.values[training_mask]
index = 0
for ii in range(0,k):
    n = val_blocks['iter%i' % (ii+1)].sum()
    placement[val_blocks['iter%i' % (ii+1)]] = rff.rfbc_predict(rfbc_k['rfbc%i' % (ii+1)]['rf1'],rfbc_k['rfbc%i' % (ii+1)]['rf2'],X[val_blocks['iter%i' % (ii+1)]])
model_20m.values[training_mask]=placement.copy()


# plot up the residuals
obs_20m=lidar.copy(deep=True)
obs_20m.values[~training_mask]=np.nan
residuals = obs_20m-model_20m
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(12,5),sharex='all',sharey='all')
obs_20m[0:700,1600:2100].plot.imshow(ax=axes[0],vmin=0,vmax=200,
                cbar_kwargs={'label':'lidar / Mg/ha','orientation':'horizontal'})
model_20m[0:700,1600:2100].plot.imshow(ax=axes[1],vmin=0,vmax=200,
                cbar_kwargs={'label':'satellite / Mg/ha','orientation':'horizontal'})
residuals[0:700,1600:2100].plot.imshow(ax=axes[2],vmin=-150,vmax=150,cmap='bwr',
                cbar_kwargs={'label':'residual / Mg/ha','orientation':'horizontal'})
axes[0].scatter(x=lidar_cal['x'],y=lidar_cal['y'],c=lidar_cal['residual'],cmap='bwr',vmin=-150,vmax=150)
fig.show()

#plt.colorbar(im1,ax=axes[0],label='lidar / Mg/ha',orientation='horizontal')
im2 = axes[1].imshow(model_20m[0:700,1600:2100],vmin=0,vmax=200)
plt.colorbar(im2,ax=axes[1],label='satellite / Mg/ha',orientation='horizontal')
im3 = axes[2].imshow(residuals[0:700,1600:2100],vmin=-150,vmax=150,cmap='bwr')
plt.colorbar(im3,label='residual / Mg/ha',orientation='horizontal',ax=axes[2])
plt.show()


if resolution in ['50','050']:
    block_width=2
else:
    block_width=5
rows_1ha = model_20m.shape[0]//block_width
cols_1ha = model_20m.shape[1]//block_width
model_1ha = np.zeros((rows_1ha,cols_1ha))
obs_1ha = np.zeros((rows_1ha,cols_1ha))

for rr,row in enumerate(np.arange(0,model_20m.shape[0]-block_width,block_width)):
    for cc, col in enumerate(np.arange(0,model_20m.shape[1]-block_width,block_width)):
        model_1ha[rr,cc]=np.mean(model_20m[row:row+block_width,col:col+block_width])
        obs_1ha[rr,cc]=np.mean(obs_20m[row:row+block_width,col:col+block_width])

y_obs_ha = obs_1ha[np.isfinite(obs_1ha)]
y_mod_ha = model_1ha[np.isfinite(model_1ha)]
temp1,temp2,r,temp3,temp4 = stats.linregress(y_obs_ha,y_mod_ha)
r2 = r**2
rmse = np.sqrt(np.mean((y_mod-y_obs)**2))
rel_rmse = rmse/np.mean(y_obs)
print("Validation\n\tR^2 = %.02f" % r2)
print("\tRMSE = %.02f" % rmse)
print("\trelative RMSE = %.02f" % rel_rmse)
annotation = 'R$^2$ = %.2f\nRMSE = %.1f Mg ha$^{-1}$\nrelative RMSE = %.1f%s' % (r2,rmse,rel_rmse*100,'%')
x_label = 'AGB$_{satellite}$ / Mg ha$^{-1}$'
y_label = 'AGB$_{LiDAR}$ / Mg ha$^{-1}$'
title =  'Upscaling resolution: %sm' % resolution
fig1, axes1 = gplt.plot_validation(y_obs,y_mod,annotation=annotation,title=title,x_label=x_label,y_label=y_label)
fig1.savefig('%s%s_%s_%sm_res_1ha_validation_blocked_kfold.png' % (path2fig,site_id,version,resolution.zfill(3)))
