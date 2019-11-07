"""
mlr_ci_pi
-----------------------------------------------------------------------
multiple linear regression with confidence and prediction intervals
Model described by equation:
y = X.b + err
where y is the target (dependent variable)
X is the predictors (independent variables)
b is the coefficients
err is the error term (due residual variance not accounted for in model)
"""
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
"""
PART 1: set up the test data
"""

# independent variables
n=30 # number of samples used to train model
p = 4 # number of parameters in model (in this case three + intercept)
alpha = 0.05 # confidence level = 1-alpha/2 (0.05 -> 95% confidence level)

x0 = np.ones(n) # constant
x1 = np.random.normal(0,1,n)
x2 = np.random.random(n)
x3 = np.random.random(n)

# as a matrix:
X = np.vstack((x0,x1,x2,x3)).T

# coefficients of MLR
b0 = 30; b1 = 5; b2 = 6; b3 = 0.1
b = np.array((b0,b1,b2,b3)).reshape(4,1)

#error term
err = 10*np.random.random(n).reshape(n,1)

# dependent variable
y = X@b +err  # @ gives the dot product of the two matrices

"""
PART 2: fit regression model and calculate mean standard error
"""
data_train = pd.DataFrame({'y':y.ravel(),'x1':x1,'x2':x2,'x3':x3})
# multivariate linear regression
model = ols("y ~ x1 + x2 + x3",data_train).fit()
print(model.summary())

# calculate mean square error, MSE, for model fit
yhat  = model.predict(data_train)
MSE = np.sum((y.ravel()-yhat)**2)/(n-p)

"""
PART 3: Confidence and prediction intervals
"""
# create new test locations
n_ = 10
x0_ = np.ones(n_)
x1_ = np.linspace(-3,3,n_)
x2_ = np.random.random(n_)
x3_ = np.random.random(n_)
X_ = np.vstack((x0_,x1_,x2_,x3_)).T

# estimate standard error for prediction at each location in parameter space
serr = np.zeros(n_)
for ii,Xh in enumerate(X_):
    Xh = Xh.reshape(-1,1) #c onvert 1D array slice to 2D column vector (i.e. one column)
    serr[ii] = np.sqrt(MSE*(Xh.T)@(np.linalg.inv(X.T@X))@(Xh))

# make some predictions at different points in the parameter space
data_predict = pd.DataFrame({'x1':x1_,'x2':x2_,'x3':x3_})
y_predict = model.predict(data_predict)

# formula for the confidence interval at each prediction point
CI = stats.t.ppf(1-alpha/2,n-p)*serr
# formula for the prediction interval at each prediction point
PI = stats.t.ppf(1-alpha/2,n-p)*np.sqrt(MSE+serr**2)

plt.errorbar(x1_,y_predict,yerr=PI,linestyle='none',marker='.',c='red',alpha=0.5) # Prediction interval
plt.errorbar(x1_,y_predict,yerr=CI,linestyle='none',marker='.',c='blue',alpha=0.5) # Confidence interval
plt.plot(x1,y,'+',c='k') # original observations
plt.show()
