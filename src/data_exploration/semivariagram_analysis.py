"""
semivariagram_analysis.py
--------------------------------------------------------------------------------
SEMI-VARIAGAM ANALYSIS
This script loads a map constructs a semivariagram based on a random sample of a
specified size.

29/08/2019 - D. T. Milodowski
--------------------------------------------------------------------------------
"""

"""
# Import the necessary packages
"""
import numpy as np                  # standard package for scientific computing
import xarray as xr                 # xarray geospatial package
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package
sns.set_style('darkgrid')
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Import custom libaries
import sys
sys.path.append('./data_io/')
import data_io as io

"""
SEMIVARIAGRAM FUNCTIONS
"""
def SVh( P, h, bw ):
    '''
    Experimental semivariogram for a single lag
    '''
    pd = squareform( pdist( P[:,:2] ) )
    N = pd.shape[0]
    Z = list()
    for i in range(N):
        for j in range(i+1,N):
            if( pd[i,j] >= h-bw )and( pd[i,j] <= h+bw ):
                Z.append( ( P[i,2] - P[j,2] )**2.0 )
    return np.sum( Z ) / ( 2.0 * len( Z ) )

def SV( P, hs, bw ):
    '''
    Experimental variogram for a collection of lags
    '''
    sv = list()
    for h in hs:
        sv.append( SVh( P, h, bw ) )
    sv = [ [ hs[i], sv[i] ] for i in range( len( hs ) ) if sv[i] > 0 ]
    return np.array( sv ).T



"""
Project Info
"""
path2data = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/'

"""
#===============================================================================
PART A: LOAD DATA
Load the raster data and inventory data
#-------------------------------------------------------------------------------
"""
raster_file = '%s/lidar/processed/kiuic_AGB_07-31-19_regridded.tif' % (path2data)
raster = io.load_geotiff(raster_file,option=1)
raster.values[raster.values==-9999]=np.nan

# restrict area to main LiDAR patch
raster.values[2186:] = np.nan



"""
#===============================================================================
PART B: CREATE SEMIVARIAGRAM
Load the raster data and inventory data
#-------------------------------------------------------------------------------
"""
# Random sample points
N_sample = 5000
xx,yy=np.meshgrid(raster.coords['x'].values,raster.coords['y'].values)
mask = np.isfinite(raster.values)
P = np.zeros((N_sample,3))
sample_idx = np.random.choice(np.arange(mask.sum()),N_sample,replace=False)
P[:,0]=xx[mask][sample_idx]
P[:,1]=yy[mask][sample_idx]
P[:,2]=raster.values[mask][sample_idx]

bandwidth = 20
lags = np.arange(0,2000,bandwidth)
semivariagram = SV(P,lags,bandwidth)

"""
#===============================================================================
PART C: PLOT SEMIVARIAGRAM
#-------------------------------------------------------------------------------
"""
df = pd.DataFrame({'lag':semivariagram[0],'semivariance':semivariagram[1]})
# Now plot up summaries according to the subset in question
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[5,3])
sns.scatterplot('lag','semivariance',data=df,ax=ax,s=2)
ax.set_xlabel('lag / m',fontsize=10)
ax.set_ylabel('semivariance / (Mg ha$^{-1}$)$^2$',fontsize=10)

fig.tight_layout()
fig.show()
