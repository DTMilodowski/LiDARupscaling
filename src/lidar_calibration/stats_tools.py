"""
stats_tools.py
============================
useful stats functions
D.T.Milodowski
"""
import numpy as np
# Calculate correction factor due to bias in mean fitted in log-transformed
# parameter space (Baskerville 1972)
def calculate_baskervilleCF(log_y,log_yhat):
    MSE = np.mean((log_y-log_yhat)**2)
    CF = np.exp(MSE/2) # Correction factor due to fitting regression in log-space (Baskerville, 1972)
    return CF


# calculate weighted quantiles
def weighted_quantiles(data,weights,quantiles=[.01,.025,.25,.5,.75,.975,.99]):
    quantiles = np.array(quantiles)
    data = np.array(data)
    weights = np.array(weights)

    N=data.size
    mask = np.all((np.isfinite(data),weights>0),axis=0)
    data_sub = data[mask]
    weights_sub = weights[mask]

    idx = np.argsort(data_sub)
    data_sub = data_sub[idx]
    weights_sub = weights_sub[idx]

    weighted_quantiles = (np.cumsum(weights_sub) - 0.5*weights_sub)/np.sum(weights_sub)
    #print(quantiles.shape,weighted_quantiles.shape,data.shape)
    quantile_vals = np.interp(quantiles,weighted_quantiles,data_sub)

    return quantile_vals
