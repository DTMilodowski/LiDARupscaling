"""
pt1_lidar_sentinel_fusion_train_rf_random_bayesian_lno_buffered.py
--------------------------------------------------------------------------------
FITTING RANDOM FOREST MODEL TO LINK SENTINEL LAYERS TO LIDAR ESTIMATED AGB
This script loads the predictor (sentinel bands and derivatives) and target
(lidar estimated AGB) variables, calibrates and validates a random forest
regression model, and fits a final model using te full training set.

The random forest algroithm is optimised using a two-step process: 1st a
randomized search is undertaken to locate a reasonable starting point; then a
bayesian optimiser (TPE) is used to refine the parameterisation.

To deal with spatial autocorrelation effects, this script tries an alternative
approach to the cal-val process during the optimisation. Specifically, having
determined the lengthscale of spatial correlation (initially based on spatial
autocorrelation of the target variable) a random set of N points is saved for
validation purposes. The training data is then filtered to remove all locations
within a buffer of the validation points. This buffer is a circular
neighbourhood with a radius equal to the effective autocorrelation length,
defined as the range over which the semivariance reaches 95% of the sill value
of the weibull function fitted to the semivariagram.

This code built using the open source programming language python, and utilises
the geospatial library xarray (http://xarray.pydata.org/en/stable/), the machine
learning library scikit-learn (https://scikit-learn.org/stable/index.html), and
the bayesian optimisation library hyperopt.

12/09/2019 - D. T. Milodowski
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
from scipy import signal
from scipy import spatial

# Import some parts of the scikit-learn library
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.externals import joblib

from hyperopt import tpe, rand, Trials, fmin, hp, STATUS_OK, space_eval, STATUS_FAIL
from hyperopt.pyll.base import scope
from functools import partial

import pickle

# Import custom libaries

import sys
sys.path.append('./random_forest/')
sys.path.append('./data_io/')
sys.path.append('./data_visualisation/')
sys.path.append('./data_exploration/')
import data_io as io
import general_plots as gplt
import random_forest_functions as rff
import semivariagram as sv
import utility

"""
Project Info
"""
site_id = 'kiuic'
version = '014'
path2alg = '../saved_models/'
path2data = "/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/"
if(os.path.isdir(path2alg)==False):
    os.mkdir(path2alg)
path2fig= '../figures/'
if(os.path.isdir(path2fig)==False):
    os.mkdir(path2fig)


N=10
k=1
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
predictors,target,datamask,labels=io.load_predictors()
n_predictors = predictors.shape[1]
print(labels)
# Keep only areas for which we have biomass estimates
mask = np.isfinite(target[datamask])
spatialmask=np.isfinite(target)*datamask

X = predictors[mask,:]
y = target[datamask][mask]

"""
#===============================================================================
PART B: DETERMINE AUTOCORRELATION LENGTHSCALE
Semivariagram
Determine effective scale -> sample grid post spacing
Create sample grid
#-------------------------------------------------------------------------------
"""
raster_file = '%s/lidar/processed/kiuic_AGB_07-31-19_regridded.tif' % (path2data)
raster = io.load_geotiff(raster_file,option=1)
raster.values[raster.values==-9999]=np.nan

N_sample = 8000
bandwidth = 0.05
llim=0
ulim=20
p0=[200,0.8,1700]
semivar = sv.empirical_semivariagram_from_xarray(raster,N_sample,llim,ulim,bandwidth)
df = pd.DataFrame({'lag':semivar[0],'semivariance':semivar[1],
                    'fit':sv.fit_weibull_distribution_from_cdf(semivar[0],semivar[1],norm=False,p0=p0)})
effective_scale = sv.get_effective_scale(df['lag'],df['fit'],threshold=.95)
# Now plot up summaries according to the subset in question
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[5,3])
sns.scatterplot('lag','semivariance',data=df,ax=ax)
ax.plot(df['lag'],df['fit'],'-',color='red',data=df)
ax.annotate('Effective scale = %.1f m' % effective_scale,
                        xy=(0.95,0.05), xycoords='axes fraction',
                        backgroundcolor='none',horizontalalignment='right',
                        verticalalignment='bottom', fontsize=10)
ax.set_xlabel('lag / m',fontsize=10)
ax.set_ylabel('semivariance / (Mg ha$^{-1}$)$^2$',fontsize=10)

fig.tight_layout()
fig.show()

"""
#===============================================================================
PART C: CREATE VALIDATION DATASET
Randomly sample N points from the available training data as validation points
Create mask in which these points are buffered by a radius defined by the
    effective scale of autocorrelation
Build the calibration and validation datasets based on the above
Repeat for k iterations

The reason for sampling N points, rather than hte buffered LOO approach outlined
by Roberts et al (Ecography, 2017), is to reduce the computational load of the
hyperoptimisation. Ideally once a hyperparameter set has been selected, the
LOO approach can be employed to generate error estimates based on a larger
number of iterations.
#-------------------------------------------------------------------------------
"""
# Build circular neighbourhood for convolution
rows,cols=raster.values.shape
dx = raster.coords['x'][1]-raster.coords['x'][0]
dy = raster.coords['y'][1]-raster.coords['y'][0]
buffer_radius = int(np.ceil(effective_scale)/np.abs(np.max([dx,dy])))

buffer = np.zeros((2*buffer_radius+1,2*buffer_radius+1))
col_idx,row_idx = np.meshgrid(np.arange(2*buffer_radius+1)-buffer_radius,
                                np.arange(2*buffer_radius+1)-buffer_radius)
dist = np.sqrt(col_idx**2+row_idx**2)
buffer[dist<=buffer_radius]=1
col_idx=None;row_idx=None;dist=None

# Get validation sample points
row_data,col_data = np.where(spatialmask)

# get cal and val sets for each iteration
X_train={};y_train={}
X_test={};y_test={}
for iter in range(0,k):
    key='iter%i' % (k+1)
    sample_rows = np.random.choice(row_data,N,replace=False)
    sample_cols = np.random.choice(col_data,N,replace=False)

    tree = spatial.cKDTree(np.asarray([sample_rows,sample_cols]).transpose())


    # Now generate masks to define calibration and validation datasets
    validation_mask = np.zeros((rows,cols))
    validation_mask[sample_rows,sample_cols]=1
    calibration_mask = datamask.copy()

    for ii in range(0,row_data.size):
        row=row_data[ii]
        col=col_data[ii]

    # 2D convolution against buffer with reflected boundary
    calibration_mask=signal.convolve2d(calibration_mask,buffer,mode='same',boundary='symm')
    calibration_mask=calibration_mask<1
    calibration_mask*=landmask
    # Retrieve cal-val sets
    #cal_mask = calibration_mask[landmask][mask]
    X_train[key] = predictors[mask,:][calibration_mask][mask]
    y_train[key] = target[validation_mask]

    X_test[key] = predictors[mask,:][calibration_mask][mask]
    y_test[key] = target[validation_mask]

pts,max_pts_per_tree = 10**6):
    npts = pts.shape[0]
    ntrees = int(np.ceil(npts/float(max_pts_per_tree)))
    trees = []
    starting_ids = []

    for tt in range(0,ntrees):
        i0=tt*max_pts_per_tree
        i1 = (tt+1)*max_pts_per_tree
        if i1 < pts.shape[0]:
            trees.append(spatial.cKDTree(pts[i0:i1,0:2],leafsize=32,balanced_tree=True))
        else:
            trees.append(spatial.cKDTree(pts[i0:,0:2],leafsize=32,balanced_tree=True))
        starting_ids.append(i0)

        starting_ids = np.asarray(starting_ids,dtype='int')

centre_x = np.mean(subplots[keys[ss]][pp][0:4,0])
                centre_y = np.mean(subplots[keys[ss]][pp][0:4,1])
                radius = np.sqrt(sample_res[ss]**2/2.)
                ids = trees[0].query_ball_point([centre_x,centre_y], radius)

"""
#===============================================================================
PART B: CAL-VAL USING RANDOMISED SEARCH THEN BAYESIAN HYPERPARAMETER OPTIMISATION
Cal-val
Cal-val figures
#-------------------------------------------------------------------------------
"""
print('Hyperparameter optimisation')
rf = RandomForestRegressor(criterion="mse",bootstrap=True,oob_score=True,n_jobs=-1)
"""
param_space = { "max_depth":scope.int(hp.quniform("max_depth",20,400,1)),              # ***maximum number of branching levels within each tree
                "max_features":scope.int(hp.quniform("max_features",int(n_predictors/4),n_predictors,1)),      # ***the maximum number of variables used in a given tree
                "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf",2,20,1)),    # ***The minimum number of samples required to be at a leaf node
                "min_samples_split": scope.int(hp.quniform("min_samples_split",3,60,1)),  # ***The minimum number of samples required to split an internal node
                "n_estimators":scope.int(hp.quniform("n_estimators",60,100,1)),          # ***Number of trees in the random forest
                "min_impurity_decrease":hp.uniform("min_impurity_decrease",0.0,0.02),
                "n_jobs":hp.choice("n_jobs",[20,20]),
                "oob_score":hp.choice("oob_score",[True,True])
                }
"""
param_space = { "max_depth":scope.int(hp.quniform("max_depth",20,400,1)),              # ***maximum number of branching levels within each tree
                "max_features":scope.int(hp.quniform("max_features",int(n_predictors/4),n_predictors,1)),      # ***the maximum number of variables used in a given tree
                "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf",2,50,1)),    # ***The minimum number of samples required to be at a leaf node
                "min_samples_split": scope.int(hp.quniform("min_samples_split",3,200,1)),  # ***The minimum number of samples required to split an internal node
                "n_estimators":scope.int(hp.quniform("n_estimators",60,100,1)),          # ***Number of trees in the random forest
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

    # k iterations of leave-N-out
    iter_scores = np.zeros(k,N)
    for ii,key in enumerate(iteration_keys):
        # Fit random forest model to calibration set
        rf = RandomForestRegressor(**params)
        rf.fit(X_train[key],y_train[key])
        # predict for validation set
        y_rf = rf.fit(X_test[key])
        # calculate RMSE
        iter_scores[ii] = np.sqrt(np.mean((y_test[key]-y_rf)**2))

    score = np.mean(iter_scores)
    # - if error reduced, then update best model accordingly
    if score < best_score:
        best_score = score
        print('new best RMSE: ', best_score, params)
    return {'loss': -score, 'status': STATUS_OK}

trials=Trials()
# Set algoritm parameters
# - TPE
# - randomised search used to initialise (n_startup_jobs iterations)
# - percentage of hyperparameter combos identified as "good" (gamma)
# - number of sampled candidates to calculate expected improvement (n_EI_candidates)
max_evals_target = 100
spin_up_target = 30
best_score = np.inf
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
parameters = ['n_estimators','max_depth', 'max_features', ,'min_samples_leaf', 'min_samples_split']


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
idx = np.argsort(trace['scores'])[0]
best_params = trials.trials[idx]['misc']['vals']
rf = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth= int(best_params['max_depth'][0]),            # ***maximum number of branching levels within each tree
            max_features=int(best_params['max_features'][0]),       # ***the maximum number of variables used in a given tree
            max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
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
