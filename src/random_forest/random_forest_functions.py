"""
# Random forest functions
#-------------------------------------------------------------------------------
# This set of functions contains augmented random forest regression routines,
# for example bias correction.
#-------------------------------------------------------------------------------
# David T. Milodowski, August 6 2019
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy
"""
rfbc_fit
-----------
This function fits the bias correction rf model following by Xu et al., Carbon
Balance and Management, 2016, which uses a modification of the bootstrap method
developed by Hooker et al, Statistics and Computing, 2018

Input arguments are a RandomForestRegressor object (rf), the predictors matrix
(X), and the target vector (y).
Returns: two fitted random forest models that are required for the bias
correction
"""
def rfbc_fit(rf,X,y):
    rf1 = deepcopy(rf)
    rf2 = deepcopy(rf)
    # fit first random forest model to the observations, y
    rf1.fit(X,y)
    # Retrieve out of bag prediction
    y_oob = rf1.oob_prediction_
    # New target variable = y_oob - residual
    # Note that this is more biased than the RF estimate
    y_new = 2*y_oob-y
    # Fit second random forest regression to predict y_new
    rf2.fit(X,y_new)

    return rf1,rf2

"""
rfbc_predict
-----------
This function uses the fitted, bias-corrected random forest regression model
to make a prediction based on the given predictor matrix X
Input arguments are the fitted RandomForestRegressor objects from rfbc_fit (rf1,
rf2), and the predictors matrix (X).
Returns: prediction from bias corrected random forest regression model
"""
def rfbc_predict(rf1,rf2,X):
    y_hat = 2*rf1.predict(X)-rf2.predict(X)
    return y_hat
