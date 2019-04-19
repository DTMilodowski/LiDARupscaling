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
#===============================================================================
PART A: LOAD IN DATA AND SUBSET THE TRAINING DATA
Fit random forest regression models to estimate potential biomass stocks.
Calibrate and validate potential biomass models
#-------------------------------------------------------------------------------
"""
predictors,AGB,landmask,labels=io.load_predictors()
