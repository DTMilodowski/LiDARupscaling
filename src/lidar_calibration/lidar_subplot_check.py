import numpy as np
import xarray as xr
import pandas as pd

import seaborn as sns
sns.set()
from matplotlib import pyplot as plt

from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table
from sklearn.model_selection import LeaveOneOut

import stats_tools as st

import sys
import os
import copy as cp
import gc

sys.path.append('../data_io/')
import data_io as io

# CHANGE THIS STUFF
# Some file names - shapefiles and rasters should be in a projected coordinate system (i.e. UTM)
#inventory_file = '../../data/lidar_calibration/Kiuic_400_live_trees.shp'
inventory_file = '../../data/lidar_calibration/Kiuic_400_live_biomass_unc.shp'
pt1_outfile = '../../data/lidar_calibration/sample_test.npz' # this will be a file to hold the compiled plot data
pt2_outfile = '../../saved_models/lidar_calibration/lidar_calibration_pt2_results.npz'
path2fig = '../../figures/'
plot_area = 400. # 1 ha
gap_ht = 8.66
radius = np.sqrt(plot_area/np.pi)

# LOAD FILES, CONVERT TO FLOAT AND SPECIFY NODATA REGIONS
pt1_output = np.load(pt1_outfile)['arr_0'][()]
chm_results = pt1_output['chm']
inventory_AGB = pt1_output['inventory']
ii=0
count=0
figure_number=1
fig,fig_axes = plt.subplots(nrows=5,ncols=5,figsize=[15,15])
axes=fig_axes.flatten()
while ii< len(chm_results.keys()):
    id = list(chm_results.keys())[ii]
    if chm_results[id]['status']=='PASS':
        plot_boundary = mpatches.Circle((0,0),radius,ec='white',fill=False)#,fc=None)
        chm_results[id]['chm_raster'].plot(ax=axes[count], vmin=0, vmax=21,
                    extend='max', cbar_kwargs={'label':'height / m'})
        axes[count].add_artist(plot_boundary)
        axes[count].set_title('plot %s; AGB = %.1f Mg/ha' % (id,inventory_AGB[id]['AGB']))
        axes[count].set_xlabel('')
        axes[count].set_ylabel('')
        count+=1
    if count+1>=25:
        fig.tight_layout()
        fig.savefig('plot_check_%i' % figure_number)
        fig.show()
        plt.cla()
        fig,fig_axes = plt.subplots(nrows=5,ncols=5,figsize=[15,15])
        axes=fig_axes.flatten()
        figure_number+=1
        count=0
    ii+=1

fig.tight_layout()
fig.savefig('plot_check_%i' % figure_number)
fig.show()
plt.cla()
