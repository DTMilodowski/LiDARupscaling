"""
pt1_lidar_sentinel_fusion_train_rf.py
--------------------------------------------------------------------------------
FITTING RANDOM FOREST MODEL TO LINK SENTINEL LAYERS TO LIDAR ESTIMATED AGB
This script loads the predictor (sentinel bands and derivatives) and target
(lidar estimated AGB) variables, calibrates and validates a random forest
regression model, and fits a final model using te full training set.

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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.externals import joblib

import eli5
from eli5.sklearn import PermutationImportance

# Import custom libaries

import sys
sys.path.append('./random_forest/')
sys.path.append('./data_io/')
sys.path.append('./data_visualisation/')

import data_io as io
import general_plots as gplt

"""
Project Info
"""
site_id = 'kiuic'
version = '002'
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

# Custom mask
target[0:800,2600:2728] = np.nan
target[4000:,:2000] = np.nan
# Keep only areas for which we have biomass estimates
mask = np.isfinite(target[landmask])
X = predictors[mask,:]
y = target[landmask][mask]

"""
#===============================================================================
PART B: CAL-VAL USING GRID SEARCH RF HYPERPARAMETER OPTIMISATION
Cal-val
Cal-val figures
Importances via permutation importance
#-------------------------------------------------------------------------------
"""
print('Calibration/validation')
#split train and test subset, specifying random seed for reproducability
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,
                                                test_size=0.25, random_state=29)

#define the parameters for the gridsearch
param_grid = {  "bootstrap"=True,
                "max_depth"=None,            # ***maximum number of branching levels within each tree
                "max_features"=np.linspace(.35,.7,8),       # ***the maximum number of variables used in a given tree
                "min_samples_leaf"=np.linspace(1,10,5,dtype='i')},       # ***The minimum number of samples required to be at a leaf node
                "min_samples_split"=20,       # ***The minimum number of samples required to split an internal node
                "n_estimators"=100,          # ***Number of trees in the random forest
                }
#create the random forest object with predefined parameters
rf = RandomForestRegressor(n_jobs=20,random_state=26,
                            n_estimators = 1000,bootstrap=True)

#perform a grid search on hyper parameters using training subset of data
rf_grid = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,
                            verbose = 3,scoring = 'neg_mean_squared_error', n_jobs=1)

# Do the grid search
rf_grid.fit(X_train,y_train)

# create a pandas dataframe storing parameters and results of the cv
cv_res = pd.DataFrame(rf_grid.cv_results_['params'])
params = cv_res.columns #save parameter names for later

#get the scores as RMSE
cv_res['mean_train_score'] = .5*(-rf_grid.cv_results_['mean_train_score'])**.5
cv_res['mean_test_score'] = .5*(-rf_grid.cv_results_['mean_test_score'])**.5
cv_res['ratio_score'] = cv_res['mean_test_score'] / cv_res['mean_train_score']

# plot grid hyperparameter performance comparison
sns.pairplot(data=cv_res,hue='bootstrap')
plt.savefig('%s%s_%s_RFgrid_pairplot.png' % (path2fig,site_id,version))

# fit the calibration sample
rf_grid.best_estimator_.fit(X_train,y_train)
y_train_rf = rf_grid.best_estimator_.predict(X_train)
cal_score = rf_grid.best_estimator_.score(X_train,y_train) # calculate coefficeint of determination R^2 of the calibration
print("Calibration R^2 = %.02f" % cal_score)

# fit the validation sample
y_test_rf = rf_grid.best_estimator_.predict(X_test)
val_score = rf_grid.best_estimator_.score(X_test,y_test)
print("Validation R^2 = %.02f" % val_score)

# Plot cal-val
fig1,axes = gplt.plot_cal_val_agb(y_train,y_train_rf,y_test,y_test_rf)
fig1.savefig('%s%s_%s_cal_val.png' % (path2fig,site_id,version))

# Importances
perm = PermutationImportance(rf_grid.best_estimator_).fit(X_test, y_test)
imp_df = pd.DataFrame(data = {'variable': labels,
                              'permutation_importance': perm.feature_importances_,
                              'gini_importance': rf_grid.best_estimator_.feature_importances_})
fig2,axes = gplt.plot_importances(imp_df)
fig2.savefig('%s%s_%s_importances.png' % (path2fig,site_id,version))

"""
#===============================================================================
PART C: FINAL FIT
Fit model with full training set
Save model
#-------------------------------------------------------------------------------
"""
print('Final calibration')
rf=rf_grid.best_estimator_
rf.fit(X,y)
cal_score = rf.score(X,y) # calculate coefficeint of determination R^2 of the calibration
print("Calibration R^2 = %.02f" % cal_score)

# Save random forest model for future use
joblib.dump(rf_grid,'%s%s_%s_rf_sentinel_lidar_agb_grid.pkl' % (path2alg,site_id,version))
joblib.dump(rf_grid,'%s%s_%s_rf_sentinel_lidar_agb_best.pkl' % (path2alg,site_id,version))
