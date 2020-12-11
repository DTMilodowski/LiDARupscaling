"""
stats_tools.py
============================
useful stats functions
D.T.Milodowski
"""
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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

# Generate a qq plot woth confidence interval for normal distribution based on
# sample size (default is 95%)
def qq_plot(variable,fig=None,ax=None,ci=.95,ylabel = "",show=True,savename = None):

    N = variable.size
    v = np.zeros(N,dtype=np.float64)
    v[-1] = 0.5**(1./N)
    v[0] = 1 - v[-1]
    i = np.arange(2,N)
    v[1:-1]=(i-0.3175)/(N+0.365)
    x_qq = stats.norm.ppf(v)

    ci_low_quantile = (1. - ci)/2.
    ci_upp_quantile = 1 - ci_low_quantile

    if ax is None:
        fig,ax=plt.subplots(nrows=1,ncols=1)

    ci_band = np.zeros((1000,N))

    rmse=np.sqrt(np.mean(variable**2))
    for jj in range(1000):
        ci_band[jj]=np.random.normal(scale=rmse,size=N)
        ci_band[jj].sort()

    ci_upp = np.percentile(ci_band,ci_upp_quantile*100,axis=0)
    ci_low = np.percentile(ci_band,ci_low_quantile*100,axis=0)
    ax.fill_between(x_qq,ci_low,ci_upp,color='0.5',alpha=0.5)

    ax.set_ylabel(ylabel)
    ax.set_xlabel('standard normal variate')

    sm.qqplot(variable, marker='.',ax=ax,line='s')

    if show:
        fig.show()

    if savename is not None:
        fig.savefig(savename)

# moving average filter
def moving_bin_residual(x_mod,x_obs,bin_halfwidth=10,post_spacing=1.):
    res = x_obs-x_mod
    x = np.arange(x_mod.min(),x_mod.max(),post_spacing)
    res_ = np.zeros(x.shape)*np.nan
    res_sd = np.zeros(x.shape)*np.nan
    for ii,xi in enumerate(x):
        if xi-x.min() < bin_halfwidth + 1:
            mask = x_mod<(x.min()+2*bin_halfwidth+1)
        elif x.max()-xi < bin_halfwidth + 1:
            mask = x_mod>=(x.max()-2*bin_halfwidth+1)
        else:
            mask = (x_mod>=xi-bin_halfwidth) * (x_mod<=xi+bin_halfwidth)
        res_[ii] = np.median(res[mask])
        res_sd[ii] = np.std(res[mask])

    return x,res_,res_sd
