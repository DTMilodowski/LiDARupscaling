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
#sns.set()                           # set some nice default plotting options

# Import some parts of the scikit-learn library
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import eli5
from eli5.sklearn import PermutationImportance

# Import custom libaries

import sys
sys.path.append('./random_forest/')
sys.path.append('./data_io/')
sys.path.append('./data_visualisation/')

import data_io as io

"""
Project Info
"""
site_id = 'kiuic'
version = '001'
path2alg = '../saved_models/'
path2predictors = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/sentinel/processed/'
path2target = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/lidar/processed/'

"""
#===============================================================================
PART A: LOAD IN DATA AND SUBSET THE TRAINING DATA
Load data
Filter areas where we have LiDAR estimates
Subsample if desired/required
#-------------------------------------------------------------------------------
"""
# Load predictors & target
predictors,target,landmask,labels=io.load_predictors()

# Keep only areas for which we have biomass estimates
mask = np.isfinite(target)
X = predictors[mask,:]
y = target[mask]

"""
#===============================================================================
PART B: CAL-VAL
Cal-val
Cal-val figures
Importances via permutation importance
#-------------------------------------------------------------------------------
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,
                                                    test_size=0.25)
rf = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth=None,            # ***maximum number of branching levels within each tree
            max_features='auto',       # ***the maximum number of variables used in a given tree
            max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
            min_impurity_decrease=0.0, # the miminum drop in the impurity of the clusters to justify splitting further
            min_impurity_split=None,   # threshold impurity within an internal node before it will be split
            min_samples_leaf=5,       # ***The minimum number of samples required to be at a leaf node
            min_samples_split=20,       # ***The minimum number of samples required to split an internal node
            min_weight_fraction_leaf=0.0,
            n_estimators=100,          # ***Number of trees in the random forest
            n_jobs=-1,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            random_state=None,         # seed used by the random number generator
            verbose=0)

# fit the calibration sample
rf.fit(X_train,y_train)
y_train_rf = rf.predict(X_train)
cal_score = rf.score(X_train,y_train) # calculate coefficeint of determination R^2 of the calibration
print("Calibration R^2 = %.02f" % cal_score)

# fit the validation sample
y_test_rf = rf.predict(X_test)
val_score = rf.score(X_test,y_test)
print("Validation R^2 = %.02f" % val_score)

# Plot cal-val
fig1,axes = gplt.plot_cal_val_agb(y_train,y_train_rf,y_test,y_test_rf)

# Importances
perm = PermutationImportance(rf).fit(X_test, y_test)
imp_df = pd.DataFrame(data = {'variable': labels,
                              'permutation_importance': perm.feature_importances_,
                              'gini_importance': rf.feature_importances_})
gplt.plot_importances(imp_df)

"""
#===============================================================================
PART C: FINAL FIT
Fit model with full training set
Save model
#-------------------------------------------------------------------------------
"""
rf.fit(X,y)
cal_score = rf.score(X,y) # calculate coefficeint of determination R^2 of the calibration
print("Calibration R^2 = %.02f" % cal_score)

# Save random forest model for future use
joblib.dump(rf,'%s%s_%s_rf_sentinel_lidar_agb.pkl' % (path2alg,site_id,version))
