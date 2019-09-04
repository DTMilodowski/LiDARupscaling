"""
semivariagram.py
--------------------------------------------------------------------------------
Efficient calculation of emipirical semivariagram and model fitting
Models currently available are:
- weibull

--------------------------------------------------------------------------------
D. T. Milodowski, 30/08/2019
--------------------------------------------------------------------------------
"""
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

"""
BASIC FUNCTIONS
- Construct empirical semivariagram
"""
# Construct empirical semivariagram
# Input arguments:
# - P: a Nx3 array, with columns for x,y,data
# - hs: a 1D array with the bin centres
# - bw: the bandwidth of the bins
def empirical_semivariagram(P,hs,bw):

    N = P.shape[0]

    # get point-point distances and squared difference. Use dot-product for
    # efficiency
    lower_triangle=np.tril(np.ones((N,N)))

    A = np.ones((N,2))*-1
    A[:,0]=P[:,0]
    B = np.ones((2,N))
    B[1,:]=P[:,0]
    separation_x = np.dot(A,B)

    A = np.ones((N,2))*-1
    A[:,0]=P[:,1]
    B = np.ones((2,N))
    B[1,:]=P[:,1]
    separation_y = np.dot(A,B)
    separation = np.sqrt(separation_x**2+separation_y**2)[lower_triangle==0]

    A = np.ones((N,2))*-1
    A[:,0]=P[:,2]
    B = np.ones((2,N))
    B[1,:]=P[:,2]
    difference = np.dot(A,B)
    squared_difference = difference[lower_triangle==0]**2

    # now distribute into bins
    semivariance = np.zeros(hs.size)*np.nan
    for ii, h in enumerate(hs):
        mask = np.all((separation>=h-bw/2.,separation<=h+bw/2.),axis=0)
        semivariance[ii] = np.sum(squared_difference[mask])/(2.*mask.sum())

    sv = [ [ hs[ii], semivariance[ii] ] for ii in range( len( hs ) ) if semivariance[ii] > 0 ]
    return np.array( sv ).T


"""
THEORETICAL MODELS
Various models for fitting to semivariagrams to be included here
- Weibull
- Retrieve effective scale from semivariagram based on model fit
"""
# takes in a cdf and fits a weibull distribution. Note that if it is a true
# probability density distribution, then use weibull_cdf_norm, which tops out
# at one
def weibull_cdf_norm(x,scale,shape):
    return 1-np.exp(-((x/scale)**shape))
def weibull_cdf(x,scale,shape,sill):
    return sill*(1-np.exp(-((x/scale)**shape)))
# fit weibull distribution to a specified cumulative density function
def fit_weibull_distribution_from_cdf(x,cdf,norm=True,p0=[]):
    mask = np.isfinite(cdf)
    x=x[mask];cdf=cdf[mask]
    if norm:
        if len(p0)<1:
            popt,pcov = curve_fit(weibull_cdf_norm,x,cdf)
        else:
            popt,pcov = curve_fit(weibull_cdf_norm,x,cdf,p0=p0)
        weib = weibull_cdf(x,popt[0],popt[1])
    else:
        if len(p0)<1:
            popt,pcov = curve_fit(weibull_cdf,x,cdf)
        else:
            popt,pcov = curve_fit(weibull_cdf,x,cdf,p0=p0)
        weib = weibull_cdf(x,popt[0],popt[1],popt[2])
    return weib

# get the effective scale of the spatial autocorrelation
def get_effective_scale(x,model,threshold=0.95):
    limit=threshold*np.max(model)
    x1 = float(np.max(x[model<limit]))
    y1 = float(model[x==x1])
    x2 = float(np.min(x[model>=limit]))
    y2 = float(model[x==x2])
    scale = x1+(x2-x1)*(limit-y1)/(y2-y1)
    return scale

"""
WRAPPER FUNCTIONS
Automatically create empirical semivariagrams from specific data types
"""
# Semivariagram from xarray raster
def empirical_semivariagram_from_xarray(raster,N_sample,llim,ulim,bandwidth,
                        x_name ='x',y_name='y'):
    xx,yy=np.meshgrid(raster.coords['x'].values,raster.coords['y'].values)
    mask = np.isfinite(raster.values)
    if N_sample>mask.sum():
        N_sample=mask.sum()
    P = np.zeros((N_sample,3))
    sample_idx = np.random.choice(np.arange(mask.sum()),N_sample,replace=False)
    P[:,0]=xx[mask][sample_idx]
    P[:,1]=yy[mask][sample_idx]
    P[:,2]=raster.values[mask][sample_idx]

    lags = np.arange(llim,ulim,bandwidth)
    semivar = empirical_semivariagram(P,lags,bandwidth)
    return semivar
