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
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
# immport hyperopt library
from hyperopt import tpe, rand, Trials, fmin, hp, STATUS_OK, space_eval, STATUS_FAIL
from hyperopt.pyll.base import scope
from functools import partial
from eli5.permutation_importance import get_score_importances

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
version = '034'
path2data = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/'
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
lidar_agb_file = '/exports/csce/datastore/geos/users/dmilodow/FOREST2020/LiDARupscaling/data/lidar_calibration/kiuic_lidar_agb_%s_median.tif' % version
lidar = io.load_geotiff(lidar_agb_file,option=1)
target=lidar.values.copy()
target[target<0] = np.nan

# Load predictors & target
data_layers,data_mask,labels = io.load_predictors(layers=['sentinel2','alos'])
n_predictors = data_layers.shape[0]
print(labels)

# load forest mask
forest_mask_file = "/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/forest_mask/%s_forest_mask_20m.tif" % site_id
forest = xr.open_rasterio(forest_mask_file).values[0]
forest_mask=forest==1
forest_mask = forest_mask*data_mask

# Keep only areas for which we have biomass estimates
training_mask = np.isfinite(target)
training_mask = image.binary_erosion(training_mask,iterations=1)
training_mask = training_mask*forest_mask

# Apply masks to the predictor dataset to be ingested into sklearn routines
predictors = io.apply_mask_to_raster_stack(data_layers,forest_mask)

# PCA analysis to reduce dimensionality of predictor variables
"""
pca = make_pipeline(StandardScaler(),PCA(n_components=0.999))
pca.fit(predictors)
X = pca.transform(predictors[training_mask[forest_mask],:])
"""
X = predictors[training_mask[forest_mask],:]
y = target[training_mask]

"""
#===============================================================================
PART B: CAL-VAL USING RANDOMISED SEARCH THEN BAYESIAN HYPERPARAMETER OPTIMISATION
Cal-val
Cal-val figures
#-------------------------------------------------------------------------------
"""
# Set up k-fold cross validation
block_res = 1000
buffer_width = 100
k=5
cal_blocks,val_blocks = cv.get_k_fold_cal_val_blocked(lidar,block_res,buffer_width,training_mask=training_mask,k=k)

print('Hyperparameter optimisation')
rf = RandomForestRegressor(criterion="mse",bootstrap=True,oob_score=True,n_jobs=-1)
param_space = { "max_depth":scope.int(hp.quniform("max_depth",20,200,1)),              # ***maximum number of branching levels within each tree
                "max_features":scope.int(hp.uniform("max_features",1,X.shape[1])),      # ***the maximum number of variables used in a given tree
                "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf",1,100,1)),    # ***The minimum number of samples required to be at a leaf node
                "min_samples_split": scope.int(hp.quniform("min_samples_split",2,250,1)),  # ***The minimum number of samples required to split an internal node
                "n_estimators":scope.int(hp.quniform("n_estimators",80,400,1)),          # ***Number of trees in the random forest
                "n_jobs":hp.choice("n_jobs",[20,20]),
                "oob_score":hp.choice("oob_score",[True,True])
                }

# set seed
np.random.seed(2909)

# define a function to quantify the objective function
def f(params):
    global best_mse
    global fail_count
    # check the hyperparameter set is sensible
    # - check 1: min_samples_split > min_samples_leaf
    if params['min_samples_split']<params['min_samples_leaf']:
        fail_count+=1
        return {'loss': None, 'status': STATUS_FAIL}

    params['random_state']=int(np.random.random()*10**6)
    rf = RandomForestRegressor(**params)
    r2_scores = np.zeros((k,2))
    MSE_scores = np.zeros((k,2))
    grad_scores = np.zeros((k,2))
    for kk in range(k):
        train_mask = cal_blocks!=kk
        test_mask = val_blocks==kk

        rf1,rf2 = rff.rfbc_fit(rf,X[train_mask],y[train_mask])
        y_rf = rff.rfbc_predict(rf1,rf2,X)

        m,temp1,r,temp2,temp3 = stats.linregress(y[test_mask],y_rf[test_mask])
        r2_scores[kk,0] = r**2
        MSE_scores[kk,0] = np.mean( (y[test_mask]-y_rf[test_mask]) **2 )
        grad_scores[kk,0] = m

        m,temp1,r,temp2,temp3 = stats.linregress(y[train_mask],y_rf[train_mask])
        r2_scores[kk,1] = r**2
        MSE_scores[kk,1] = np.mean( (y[train_mask]-y_rf[train_mask]) **2 )
        grad_scores[kk,1] = m

    r2_score=r2_scores[:,0].mean()
    mse=MSE_scores[:,0].mean()
    # - if error reduced, then update best model accordingly
    if mse < best_mse:
        best_mse = mse
        print('new best r^2: %.5f; best RMSE: %.5f' % (r2_score, np.sqrt(mse)))
        print(params)
    return {'loss' : mse, 'status' : STATUS_OK,
            'mse_test' : MSE_scores[:,0].mean(),
            'mse_train' : MSE_scores[:,1].mean(),
            'gradient_test' : grad_scores[:,0].mean(),
            'gradient_train' : grad_scores[:,1].mean(),
            'r2_test' : r2_scores[:,0].mean(),
            'r2_train' : r2_scores[:,1].mean()}

trials=Trials()
# Set algoritm parameters
# - TPE
# - randomised search used to initialise (n_startup_jobs iterations)
# - percentage of hyperparameter combos identified as "good" (gamma)
# - number of sampled candidates to calculate expected improvement (n_EI_candidates)
max_evals_target = 250
spin_up_target = 100
best_mse = np.inf
fail_count=0

# Start with randomised search - setting this explicitly to account for some
# iterations not being accepted
print("Starting randomised search (spin up)")
best = fmin(f, param_space, algo=rand.suggest, max_evals=spin_up_target, trials=trials)
spin_up = spin_up_target+fail_count
while (len(trials.trials)-fail_count)<spin_up_target:
    print('\tTarget: %i; iterations: %i; failures: %i' % (spin_up_target,len(trials.trials),fail_count))
    spin_up+=1
    best = fmin(f, param_space, algo=rand.suggest, max_evals=spin_up, trials=trials)

print('randomised search complete; saving trials to file for future reference')
pickle.dump(trials, open('%s%s_%s_rf_sentinel_lidar_agb_trials_rfbc.p' % (path2alg,site_id,version), "wb"))

# Now do the TPE search
print("Starting TPE search")
max_evals = max_evals_target+fail_count
algorithm = partial(tpe.suggest, n_startup_jobs=spin_up, gamma=0.15, n_EI_candidates=80)
best = fmin(f, param_space, algo=algorithm, max_evals=max_evals, trials=trials)

# Not every hyperparameter set will be accepted, so need to conitnue searching
# until the required number of evaluations is met
max_evals = max_evals_target+fail_count
while (len(trials.trials)-fail_count)<max_evals_target:
    print('\tTarget: %i; iterations: %i; failures: %i' % (max_evals_target,len(trials.trials),fail_count))
    max_evals+=1
    best = fmin(f, param_space, algo=algorithm, max_evals=max_evals, trials=trials)

print('first phase of TPE search complete; saving trials to file for future reference')
pickle.dump(trials, open('%s%s_%s_rfbc_sentinel_lidar_agb_trials.p' % (path2alg,site_id,version), "wb"))

# Now repeat TPE search for another 200 iterations with a refined search window
max_evals_target+=200
max_evals = max_evals_target+fail_count
algorithm = partial(tpe.suggest, n_startup_jobs=spin_up, gamma=0.25, n_EI_candidates=32)
while (len(trials.trials)-fail_count)<max_evals_target:
    print('\tTarget: %i; iterations: %i; failures: %i' % (max_evals_target,len(trials.trials),fail_count))
    max_evals+=1
    best = fmin(f, param_space, algo=algorithm, max_evals=max_evals, trials=trials)

print('\n\n%i iterations, from which %i failed' % (max_evals,fail_count))
print('best:')
print(best)

# save trials for future reference
print('TPE search complete; saving trials to file for future reference')
pickle.dump(trials, open('%s%s_%s_rfbc_sentinel_lidar_agb_trials.p' % (path2alg,site_id,version), "wb"))
# open with:
# trials = pickle.load(open('%s%s_%s_rfbc_sentinel_lidar_agb_trials.p' % (path2alg,site_id,version), "rb"))

# plot summary of optimisation runs
print('Basic plots summarising optimisation results')
parameters = ['n_estimators','max_depth', 'max_features', 'min_samples_leaf', 'min_samples_split']

# double check the number of accepted parameter sets
success_count = 0
fail_count = 0
for tt in trials.trials:
    if tt['result']['status']=='ok':
        success_count+=1
    else:
        fail_count+=1


trace = {}
trace['scores'] = np.zeros(success_count)
trace['iteration'] = np.arange(success_count)+1
for pp in parameters:
    trace[pp] = np.zeros(success_count)
ii=0
for tt in trials.trials:
    if tt['result']['status']=='ok':
        trace['scores'][ii] = tt['result']['loss']
        for pp in parameters:
            trace[pp][ii] = tt['misc']['vals'][pp][0]
        ii+=1

df = pd.DataFrame(data=trace)
fig2,axes = gplt.plot_hyperparameter_search_scores(df,parameters)
fig2.savefig('%s%s_%s_hyperpar_search_score_rfbc.png' % (path2fig,site_id,version))

# Plot traces to see progression of hyperparameter selection
fig3,axes = gplt.plot_hyperparameter_search_trace(df,parameters)
fig3.savefig('%s%s_%s_hyperpar_search_trace_rfbc.png' % (path2fig,site_id,version))


# Take best hyperparameter set and apply cal-val on full training set
print('Applying cal-val to full training set and withheld validation set')
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
            n_estimators=int(best_params['n_estimators'][0]),          # ***Number of trees in the random forest
            n_jobs=-1,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            random_state=29,         # seed used by the random number generator
            )

rf1,rf2 = rff.rfbc_fit(rf,X,y)
y_rfbc = rff.rfbc_predict(rf1,rf2,X)
# Save random forest model for future use
rf_dict = {}
rf_dict['rf1']=rf1
rf_dict['rf2']=rf2
joblib.dump(rf_dict,'%s%s_%s_rfbc_sentinel_lidar_agb_bayes_opt.pkl' % (path2alg,site_id,version))


"""
#===============================================================================
PART C: FEATURE IMPORTANCE
- Feature importance calculated based on fraction of explained variance (i.e.
  fractional drop in R^2) on random permutation of each predictor variable.
- Five iterations, with mean and standard deviation reported and plotted.
#-------------------------------------------------------------------------------
"""
# First define the score random_forest_functions as fractional decrease in
# variance explained
def r2_score(X,y):
    y_rfbc1 = rff.rfbc_predict(rfbc1['rf1'],rfbc1['rf2'],X[val_blocks==0])
    y_rfbc2 = rff.rfbc_predict(rfbc2['rf1'],rfbc2['rf2'],X[val_blocks==1])
    y_rfbc3 = rff.rfbc_predict(rfbc3['rf1'],rfbc3['rf2'],X[val_blocks==2])
    y_rfbc4 = rff.rfbc_predict(rfbc4['rf1'],rfbc4['rf2'],X[val_blocks==3])
    y_rfbc5 = rff.rfbc_predict(rfbc5['rf1'],rfbc5['rf2'],X[val_blocks==4])
    y_obs = np.hstack((y[val_blocks==0],y[val_blocks==1],y[val_blocks==2],y[val_blocks==3],y[val_blocks==4]))
    y_mod = np.hstack((y_rfbc1,y_rfbc2,y_rfbc3,y_rfbc4,y_rfbc5))
    temp1,temp2,r,temp3,temp4 = stats.linregress(y_obs,y_mod)
    return r**2

rfbc1 = {}; rfbc2 = {}; rfbc3 = {}; rfbc4 = {}; rfbc5 = {}
rfbc1['rf1'],rfbc1['rf2'] = rff.rfbc_fit(rf,X[cal_blocks!=0],y[cal_blocks!=0])
rfbc2['rf1'],rfbc2['rf2'] = rff.rfbc_fit(rf,X[cal_blocks!=1],y[cal_blocks!=1])
rfbc3['rf1'],rfbc3['rf2'] = rff.rfbc_fit(rf,X[cal_blocks!=2],y[cal_blocks!=2])
rfbc4['rf1'],rfbc4['rf2'] = rff.rfbc_fit(rf,X[cal_blocks!=3],y[cal_blocks!=3])
rfbc5['rf1'],rfbc5['rf2'] = rff.rfbc_fit(rf,X[cal_blocks!=4],y[cal_blocks!=4])

n_iter=5
base_score,score_drops = get_score_importances(r2_score,X,y,n_iter=n_iter)

# Additional importance estimates that holistically consider the impact of
# permuting all the layers from a given sensor
alos_mask = np.zeros(n_predictors,dtype='bool')
sentinel_mask = np.zeros(n_predictors,dtype='bool')
sentinel_labs = ['b1','b2','b3','b4','ndvi']
texture_labs = ['value','contrast','correlation','dissimilarity','entropy','homogeneity','mean','second_moment','variance']
texture_labs_alt = ['value','cont','corr','diss','ent','hom','mean','s_m_','var']
texture_labs_display = ['value','contrast','correlation','dissimilarity','entropy','homogeneity','mean','second moment','variance']

for ii,lab in enumerate(labels):
    if 'alos' in lab:
        alos_mask[ii] = True
    for ll in sentinel_labs:
        if ll in lab:
            sentinel_mask[ii] = True

#sentinel_drops = np.zeros(n_iter)
#alos_drops = np.zeros(n_iter)
n = y.size
score_drops = []
var_labels=[]
base_score = r2_score(X,y)
print('\tpermutation importance...')
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
            if texture == 'value':
                if len(lab) <20:
                    X_shuffle[:,ll]=X_permute[:,ll]
            elif texture in lab:
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

fig5,axes = gplt.plot_permutation_importances(imp_df,show=True,figsize=[6,5],emphasis=['all sentinel','all alos'])
fig5.savefig('%s%s_%s_permutation_importances_by_texture.png' % (path2fig,site_id,version))

variable_mask = np.zeros(len(imp_df),dtype='bool')
for ii,var in enumerate(imp_df['variable']):
    if "all" in var:
        variable_mask[ii]=True

fig5,axes = gplt.plot_permutation_importances(imp_df[variable_mask],show=True,emphasis=['all sentinel','all alos'],figsize=[6,5])
fig5.savefig('%s%s_%s_permutation_importances_summary.png' % (path2fig,site_id,version))


"""
#===============================================================================
PART D: VALIDATION
- Observed vs. modelled biomass using the blocked buffered strategy to avoid
  bias due to spatial autocorrelation
#-------------------------------------------------------------------------------
"""
y_rfbc1 = rff.rfbc_predict(rfbc1['rf1'],rfbc1['rf2'],X[val_blocks==0])
y_rfbc2 = rff.rfbc_predict(rfbc2['rf1'],rfbc2['rf2'],X[val_blocks==1])
y_rfbc3 = rff.rfbc_predict(rfbc3['rf1'],rfbc3['rf2'],X[val_blocks==2])
y_rfbc4 = rff.rfbc_predict(rfbc4['rf1'],rfbc4['rf2'],X[val_blocks==3])
y_rfbc5 = rff.rfbc_predict(rfbc5['rf1'],rfbc5['rf2'],X[val_blocks==4])
y_obs = np.hstack((y[val_blocks==0],y[val_blocks==1],y[val_blocks==2],y[val_blocks==3],y[val_blocks==4]))
y_mod = np.hstack((y_rfbc1,y_rfbc2,y_rfbc3,y_rfbc4,y_rfbc5))
temp1,temp2,r,temp3,temp4 = stats.linregress(y_obs,y_mod)
r2 = r**2
rmse = np.sqrt(np.mean((y_mod-y_obs)**2))
rel_rmse = rmse/np.mean(y_obs)
print("Validation\n\tR^2 = %.02f" % r2)
print("\tRMSE = %.02f" % rmse)
print("\trelative RMSE = %.02f" % rel_rmse)
annotation = 'R$^2$ = %.2f\nRMSE = %.1f\nrelative RMSE = %.1f%s' % (r2,rmse,rel_rmse*100,'%')
fig6, axes6 = gplt.plot_validation(y_obs,y_mod,annotation=annotation)
fig6.savefig('%s%s_%s_validation_blocked_kfold.png' % (path2fig,site_id,version))
