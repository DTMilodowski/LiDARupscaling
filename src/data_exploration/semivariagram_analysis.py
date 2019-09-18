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

# Import custom libaries
import sys
sys.path.append('./data_io/')
sys.path.append('./data_exploration')
import data_io as io
import semivariagram as sv

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
raster_file = '%s/lidar/agb_maps/kiuic_AGB_07-31-19_regridded.tif' % (path2data)
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
N_sample = 8000
bandwidth = 20
llim=10
ulim=2030
semivar = sv.empirical_semivariagram_from_xarray(raster,N_sample,llim,ulim,bandwidth)

"""
#===============================================================================
PART C: PLOT SEMIVARIAGRAM
#-------------------------------------------------------------------------------
"""
df = pd.DataFrame({'lag':semivar[0],'semivariance':semivar[1],
                    'fit':sv.fit_weibull_distribution_from_cdf(semivar[0],semivar[1],norm=False)})
# Now plot up summaries according to the subset in question
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[5,3])
sns.scatterplot('lag','semivariance',data=df,ax=ax)
ax.plot(df['lag'],df['fit'],'-',color='red',data=df)
ax.annotate('Effective scale = %.1f m' % sv.get_effective_scale(df['lag'],df['fit'],threshold=.95),
                        xy=(0.95,0.05), xycoords='axes fraction',
                        backgroundcolor='none',horizontalalignment='right',
                        verticalalignment='bottom', fontsize=10)
ax.set_xlabel('lag / m',fontsize=10)
ax.set_ylabel('semivariance / (Mg ha$^{-1}$)$^2$',fontsize=10)

fig.tight_layout()
fig.show()
