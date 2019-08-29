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

# Import some parts of the scikit-learn library
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.externals import joblib

import eli5
from eli5.sklearn import PermutationImportance

from hyperopt import tpe, rand, Trials, fmin, hp, STATUS_OK, space_eval, STATUS_FAIL
from hyperopt.pyll.base import scope
from functools import partial

import pickle

# Import custom libaries

import sys
sys.path.append('./random_forest/')
sys.path.append('./data_io/')
sys.path.append('./data_visualisation/')

import data_io as io
import general_plots as gplt
import random_forest_functions as rff

"""
Project Info
"""
site_id = 'kiuic'
version = '010'
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
# Load predictors & target
predictors,target,landmask,labels=io.load_predictors()
n_predictors = predictors.shape[1]
print(labels)
"""
# Custom mask
target[0:800,2600:2728] = np.nan
target[4000:,:2000] = np.nan
target[2186:] = np.nan
"""
# Keep only areas for which we have biomass estimates
mask = np.isfinite(target[landmask])
X = predictors[mask,:]
y = target[landmask][mask]

"""
#===============================================================================
PART B: CAL-VAL USING RANDOMISED SEARCH THEN BAYESIAN HYPERPARAMETER OPTIMISATION
Cal-val
Cal-val figures
#-------------------------------------------------------------------------------
"""
print('Hyperparameter optimisation')
#split train and test subset, specifying random seed for reproducability
# due to processing limitations, we use only 200000 in the initial
# hyperparameter optimisation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,test_size=0.25,random_state=23)

rf = RandomForestRegressor(criterion="mse",bootstrap=True,oob_score=True,n_jobs=-1)
param_space = { "max_depth":scope.int(hp.quniform("max_depth",20,400,1)),              # ***maximum number of branching levels within each tree
                "max_features":scope.int(hp.quniform("max_features",int(n_predictors/4),n_predictors,1)),      # ***the maximum number of variables used in a given tree
                "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf",2,20,1)),    # ***The minimum number of samples required to be at a leaf node
                "min_samples_split": scope.int(hp.quniform("min_samples_split",3,60,1)),  # ***The minimum number of samples required to split an internal node
                "n_estimators":scope.int(hp.quniform("n_estimators",80,200,1)),          # ***Number of trees in the random forest
                "min_impurity_decrease":hp.uniform("min_impurity_decrease",0.0,0.02),
                "n_jobs":hp.choice("n_jobs",[20,20]),
                "oob_score":hp.choice("oob_score",[True,True])
                }

# define a function to quantify the objective function
def f(params):
    global best_score
    global seed
    global fail_count
    # check the hyperparameter set is sensible
    # - check 1: min_samples_split > min_samples_leaf
    if params['min_samples_split']<params['min_samples_leaf']:
        fail_count+=1
        #print("INVALID HYPERPARAMETER SELECTION",params)
        return {'loss': None, 'status': STATUS_FAIL}

    # template rf model for rfbc definition
    rf = RandomForestRegressor(**params)
    rf.fit(X_train,y_train)
    score = rf.oob_score_
    # - if error reduced, then update best model accordingly
    if score > best_score:
        best_score = score
        print('new best r^2: ', -best_score, params)
    seed+=1
    return {'loss': -score, 'status': STATUS_OK}

trials=Trials()
# Set algoritm parameters
# - TPE
# - randomised search used to initialise (n_startup_jobs iterations)
# - percentage of hyperparameter combos identified as "good" (gamma)
# - number of sampled candidates to calculate expected improvement (n_EI_candidates)
max_evals_target = 300
spin_up_target = 50
best_score = -np.inf
seed=0
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

# Now do the TPE search
print("Starting TPE search")
max_evals = max_evals_target+fail_count
algorithm = partial(tpe.suggest, n_startup_jobs=spin_up, gamma=0.15, n_EI_candidates=100)
best = fmin(f, param_space, algo=algorithm, max_evals=max_evals, trials=trials)
# Not every hyperparameter set will be accepted, so need to conitnue searching
# until the required number of evaluations is met
max_evals = max_evals_target+fail_count
while (len(trials.trials)-fail_count)<max_evals_target:
    print('\tTarget: %i; iterations: %i; failures: %i' % (max_evals_target,len(trials.trials),fail_count))
    max_evals+=1
    best = fmin(f, param_space, algo=algorithm, max_evals=max_evals, trials=trials)

print('\n\n%i iterations, from which %i failed' % (max_evals,fail_count))
print('best:')
print(best)

# save trials for future reference
print('saving trials to file for future reference')
pickle.dump(trials, open('%s%s_%s_rf_sentinel_lidar_agb_trials_rfbc.p' % (path2alg,site_id,version), "wb"))
# open with:
# trials = pickle.load(open('%s%s_%s_rf_sentinel_lidar_agb_trials_rfbc.p' % (path2alg,site_id,version), "rb"))

# plot summary of optimisation runs
print('Basic plots summarising optimisation results')
parameters = ['n_estimators','max_depth', 'max_features', 'min_impurity_decrease','min_samples_leaf', 'min_samples_split']


trace = {}
trace['scores'] = np.zeros(max_evals_target)
trace['iteration'] = np.arange(max_evals_target)+1
for pp in parameters:
    trace[pp] = np.zeros(max_evals_target)
ii=0
for tt in trials.trials:
    if tt['result']['status']=='ok':
        trace['scores'][ii] = -tt['result']['loss']
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
idx = np.argsort(trace['scores'])[-1]
best_params = trials.trials[idx]['misc']['vals']
rf = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth= int(best_params['max_depth'][0]),            # ***maximum number of branching levels within each tree
            max_features=int(best_params['max_features'][0]),       # ***the maximum number of variables used in a given tree
            max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
            min_impurity_decrease=best_params['min_impurity_decrease'][0], # the miminum drop in the impurity of the clusters to justify splitting further
            min_impurity_split=None,   # threshold impurity within an internal node before it will be split
            min_samples_leaf=int(best_params['min_samples_leaf'][0]),       # ***The minimum number of samples required to be at a leaf node
            min_samples_split=int(best_params['min_samples_split'][0]),       # ***The minimum number of samples required to split an internal node
            n_estimators=500,# int(best_params['n_estimators'][0])         # ***Number of trees in the random forest
            n_jobs=-1,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True            # use out-of-bag samples to estimate the R^2 on unseen data
            )

# fit the calibration sample
rf1,rf2 = rff.rfbc_fit(rf,X_train,y_train)
y_train_rfbc = rff.rfbc_predict(rf1,rf2,X_train)
cal_score = r2_score(y_train_rfbc,y_train) # calculate coefficeint of determination R^2 of the calibration
print("Calibration R^2 = %.02f" % cal_score)

# fit the validation sample
y_test_rfbc = rff.rfbc_predict(rf1,rf2,X_test)
val_score = r2_score(y_test_rfbc,y_test)
print("Validation R^2 = %.02f" % val_score)

# Plot cal-val
fig1,axes = gplt.plot_cal_val_agb(y_train,y_train_rfbc,y_test,y_test_rfbc)
fig1.savefig('%s%s_%s_cal_val_rfbc.png' % (path2fig,site_id,version))

# Save random forest model for future use
rf_dict = {}
rf_dict['rf1']=rf1
rf_dict['rf2']=rf2
joblib.dump(rf_dict,'%s%s_%s_rfbc_sentinel_lidar_agb_bayes_opt.pkl' % (path2alg,site_id,version))

"""
# Importances
perm_rf1 = PermutationImportance(rf).fit(X_test, y_test)
imp_df = pd.DataFrame(data = {'variable': labels,
                              'permutation_importance': perm_rf1.feature_importances_,
                              'gini_importance': rf.feature_importances_})
fig5,axes = gplt.plot_importances(imp_df)
fig5.savefig('%s%s_%s_importances.png' % (path2fig,site_id,version))

perm_rf2 = PermutationImportance(rf2).fit(X_test, residuals_test)
imp_residual_df = pd.DataFrame(data = {'variable': labels,
                              'permutation_importance': perm_rf2.feature_importances_,
                              'gini_importance': rf2.feature_importances_})
fig5,axes = gplt.plot_importances(imp_residual_df)
fig5.savefig('%s%s_%s_residual_importances.png' % (path2fig,site_id,version))
"""
