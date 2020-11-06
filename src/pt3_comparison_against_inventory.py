"""
pt3_comparison_against_inventory.py
--------------------------------------------------------------------------------
COMPARISON OF UPSCALED PRODUCT (AND ALSO LIDAR PRODUCT) WITH INVENTORY DATA
ACROSS THE REGION OF INTEREST
The analysis undertaken is as follows:

- comparison of upscaled AGB estimates against inventory estimates (inside &
  outside training area). Figure: scatter plot of the upscaled AGB (+confidence
  intervals) against inventory estimate. Inventory plots aggregated to the plot
  cluster level (4 x subplots)

- AGB distributions of inventory (inside vs. outside training area), lidar AGB,
  and upscaled AGB estimate (median, inside vs. outside training area). Inside
  training area indicated by darker fill to distribution, with a lighter fill
  employed for outside training area

This code built using the open source programming language python, and utilises
the geospatial library xarray (http://xarray.pydata.org/en/stable/)

12/08/2020 - D. T. Milodowski
--------------------------------------------------------------------------------
"""

"""
# Import the necessary packages
"""
import sys
import fiona                        # shapefile handling
import numpy as np                  # standard package for scientific computing
import pandas as pd                 # dataframes
import xarray as xr                 # xarray geospatial package
import seaborn as sns               # another useful plotting package
import cartopy.crs as ccrs          # cartographic projection library
import matplotlib.pyplot as plt     # plotting package

from shapely.geometry import Point, Polygon # shapefile manipulation

sys.path.append('./lidar_calibration')
sys.path.append('./data_visualisation/')
import stats_tools as st            # some useful stats routines
import general_plots as gplt        # general plotting library
import geospatial_tools as gst      # some useful geospatial routines

"""
Project Info
"""
site_id = 'kiuic'
version = '034'
path2data = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/data/'
path2lidar = '/exports/csce/datastore/geos/users/dmilodow/FOREST2020/LiDARupscaling/data/lidar_calibration/'
path2upscaled = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/output/'
path2inventory = '../data/lidar_calibration/'
path2fig= '../figures/'

lidar_cal = '../saved_models/lidar_calibration/lidar_calibration_pt2_results_%s.npz' % version

# from ICM plots - mean and standard deviation for total AGB in small stems
# (2.5cm <= DBH < 7.5cm)
small_stem_agb = 24.39092
small_stem_std = 13.54176

"""
#===============================================================================
PART A: LOAD IN DATA AND SPLIT THE TRAINING DATA FROM THE REMAINING DATA
#-------------------------------------------------------------------------------
"""
print('Loading data')
# agb model (to correct NFI inventory, so that they can be compared with the ICM
# plots and upscaled maps)
lidar_cal_output = np.load(lidar_cal,allow_pickle=True)['arr_0'][()]
mc_results = lidar_cal_output['mc_results']
correction_factor = np.median(mc_results['params'][:,2])

# inventory data
dX=20;dY=-20
inventory_file = '%s/Kiuic_400_live_biomass_unc.shp' % path2inventory
plot_area = 400. # 1 ha
radius = np.sqrt(plot_area/np.pi)
buffer = 10*radius*dX
inventory = fiona.open(inventory_file)
plot_info = {}
for plot in inventory:
    pp='%.0f' % (plot['properties']['plot'])
    if pp not in plot_info.keys():
        plot_info[pp]={}
        plot_info[pp]['Xmin']=plot['properties']['x']
        plot_info[pp]['Xmax']=plot['properties']['x']
        plot_info[pp]['Ymin']=plot['properties']['y']
        plot_info[pp]['Ymax']=plot['properties']['y']
        plot_info[pp]['N']=1
    else:
        plot_info[pp]['Xmin']=np.min([plot_info[pp]['Xmin'],plot['properties']['x']])
        plot_info[pp]['Xmax']=np.max([plot_info[pp]['Xmax'],plot['properties']['x']])
        plot_info[pp]['Ymin']=np.min([plot_info[pp]['Ymin'],plot['properties']['y']])
        plot_info[pp]['Ymax']=np.max([plot_info[pp]['Ymax'],plot['properties']['y']])
        plot_info[pp]['N']+=1
    plot_info[pp]['buffer'] = Polygon([(plot_info[pp]['Xmin'],plot_info[pp]['Ymin']),
                                    (plot_info[pp]['Xmin'],plot_info[pp]['Ymax']),
                                    (plot_info[pp]['Xmax'],plot_info[pp]['Ymax']),
                                    (plot_info[pp]['Xmax'],plot_info[pp]['Ymin'])]).buffer(buffer)

# lidar data @20m resolution
lidar_agb_file = '%s/kiuic_lidar_agb_median.tif' % path2lidar
lidar = xr.open_rasterio(lidar_agb_file)[0]
lidar.values[lidar.values==-9999]=np.nan

lidar_ll_file = '%s/kiuic_lidar_agb_95l.tif' % path2lidar
lidar_lower = xr.open_rasterio(lidar_ll_file)[0]
lidar_lower.values[lidar_lower.values==-9999]=np.nan

lidar_ul_file = '%s/kiuic_lidar_agb_95u.tif' % path2lidar
lidar_upper = xr.open_rasterio(lidar_ul_file)[0]
lidar_upper.values[lidar_upper.values==-9999]=np.nan

# upscaled data
med_agb_file = '%s/kiuic_%s_rfbc_agb_upscaled_median.tif' % (path2upscaled,version)
agb_med = xr.open_rasterio(med_agb_file)[0]
agb_med.values[agb_med.values==-9999]=np.nan

lower_agb_file = '%s/kiuic_%s_rfbc_agb_upscaled_lower.tif' % (path2upscaled,version)
agb_lower = xr.open_rasterio(lower_agb_file)[0]
agb_lower.values[agb_lower.values==-9999]=np.nan

upper_agb_file = '%s/kiuic_%s_rfbc_agb_upscaled_upper.tif' % (path2upscaled,version)
agb_upper = xr.open_rasterio(upper_agb_file)[0]
agb_upper.values[agb_upper.values==-9999]=np.nan
"""
#===============================================================================
PART B: PLOT COMPARISON OF UPSCALED MAP AGAINST AGGREGATED PLOT CLUSTERS
#------------------------------
"""
plot_list = []; subplot_list = []; invCOLLECTION = []
invAGB = []; invERR = []
lidarAGB = []; lidarULIM = []; lidarLLIM=[]
satAGB = []; satULIM = []; satLLIM = []
status = []; inside_lidar = []
n_iter = 100 #iterations of montecarlo positional error
positional_error=5.
# SAMPLE RASTER FOR EACH PLOT NEIGHBOURHOOD
plot_clusters = plot_info.keys()
for pp, plot in enumerate(plot_clusters):
    print('\t\tprocessing %s' % plot,end='\r')
    # Generate mask around AOI to make subsequent code more efficient
    Xmin,Ymin,Xmax,Ymax = plot_info[plot]['buffer'].bounds
    Xmin-=100;Xmax+=100;Ymin-=100;Ymax+=100
    polygon_around_plot = np.array([[Xmin,Ymin],[Xmax,Ymin],[Xmax,Ymax],[Xmin,Ymax]])

    # load raster subsets
    if dY<0:
        lidar_sub = lidar.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
        lidar_upper_sub = lidar_upper.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
        lidar_lower_sub = lidar_lower.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
        agb_med_sub = agb_med.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
        agb_low_sub = agb_lower.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
        agb_upp_sub = agb_upper.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
    else:
        lidar_sub = lidar.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))
        lidar_upper_sub = lidar_upper.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))
        lidar_lower_sub = lidar_lower.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))
        agb_med_sub = agb_med.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))
        agb_low_sub = agb_lower.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))
        agb_upp_sub = agb_upper.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))

    # if plot within raster extent, get plot-level data
    if np.all((len(agb_med_sub.x)>0,len(agb_med_sub.y)>0)):
        if agb_med_sub.values.size>0:
            if np.sum(np.isfinite(agb_med_sub.values))>0:
                for subplot in inventory:
                    if ('%.0f' % subplot['properties']['plot'])==plot:
                        id = '%.0f.%.0f' % (subplot['properties']['plot'],subplot['properties']['subplot'])
                        print('\t\tprocessing %s (cluster %i/%i)' % (id,pp+1,len(plot_clusters)),end='\r')
                        plot_list.append(plot)
                        subplot_list.append(id)
                        # inventory
                        if len(id)>5:
                            invCOLLECTION.append(1)
                            invAGB.append(subplot['properties']['agb']+small_stem_agb)
                            invERR.append(np.sqrt(subplot['properties']['unc']**2+small_stem_std**2))

                        else:
                            invCOLLECTION.append(0)
                            invAGB.append(subplot['properties']['agb'])
                            invERR.append(subplot['properties']['unc'])

                        plot_centre = Point(subplot['geometry']['coordinates'][0],subplot['geometry']['coordinates'][1])

                        # lidar
                        query = gst.sample_raster_by_point_neighbourhood(lidar_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)
                        if query['status']=='PASS':
                            if np.mean(np.isfinite(query['raster_values'][0].values))>=.9:
                                status.append('PASS')
                                inside_lidar.append(True)
                                med_mc = np.zeros(n_iter)*np.nan
                                LLIM_mc = np.zeros(n_iter)*np.nan
                                ULIM_mc = np.zeros(n_iter)*np.nan
                                for ii in range(0,n_iter):
                                    xerr = np.random.randn()*positional_error
                                    yerr = np.random.randn()*positional_error
                                    plot_centre = Point(subplot['geometry']['coordinates'][0]+xerr,subplot['geometry']['coordinates'][1]+yerr)

                                    med_mc[ii] = gst.sample_raster_by_point_neighbourhood(lidar_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)['weighted_average'][0]
                                    LLIM_mc[ii] = gst.sample_raster_by_point_neighbourhood(lidar_lower_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)['weighted_average'][0]
                                    ULIM_mc[ii] = gst.sample_raster_by_point_neighbourhood(lidar_upper_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)['weighted_average'][0]

                                lidarAGB.append(np.nanmedian(med_mc))
                                lidarULIM.append(np.nanpercentile(ULIM_mc,97.5))
                                lidarLLIM.append(np.nanpercentile(LLIM_mc,2.5))
                            else:
                                inside_lidar.append(False)
                                lidarAGB.append(np.nan)
                                lidarLLIM.append(np.nan)
                                lidarULIM.append(np.nan)
                        else:
                            inside_lidar.append(False)
                            lidarAGB.append(np.nan)
                            lidarLLIM.append(np.nan)
                            lidarULIM.append(np.nan)

                        # satellite
                        query = gst.sample_raster_by_point_neighbourhood(agb_med_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)
                        if query['status']=='PASS':
                            if np.mean(np.isfinite(query['raster_values'][0].values))>=.9:
                                status.append('PASS')
                                med_mc = np.zeros(n_iter)*np.nan
                                LLIM_mc = np.zeros(n_iter)*np.nan
                                ULIM_mc = np.zeros(n_iter)*np.nan
                                for ii in range(0,n_iter):
                                    xerr = np.random.randn()*positional_error
                                    yerr = np.random.randn()*positional_error
                                    plot_centre = Point(subplot['geometry']['coordinates'][0]+xerr,subplot['geometry']['coordinates'][1]+yerr)

                                    med_mc[ii] = gst.sample_raster_by_point_neighbourhood(agb_med_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)['weighted_average'][0]
                                    LLIM_mc[ii] = gst.sample_raster_by_point_neighbourhood(agb_low_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)['weighted_average'][0]
                                    ULIM_mc[ii] = gst.sample_raster_by_point_neighbourhood(agb_upp_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)['weighted_average'][0]

                                satAGB.append(np.nanmedian(med_mc))
                                satULIM.append(np.nanpercentile(ULIM_mc,97.5))
                                satLLIM.append(np.nanpercentile(LLIM_mc,2.5))

                            else:
                                status.append('FAIL')
                                satAGB.append(np.nan)
                                satLLIM.append(np.nan)
                                satULIM.append(np.nan)

                        else:
                            status.append('FAIL')
                            satAGB.append(np.nan)
                            satLLIM.append(np.nan)
                            satULIM.append(np.nan)

# load relevant data into a pandas dataframe
subplot_df = pd.DataFrame({'plot':plot_list, 'subplot':subplot_list,
                        'inside_lidar':inside_lidar,'collection':invCOLLECTION,
                        'invAGB':invAGB,'invERR':invERR,
                        'lidarAGB':lidarAGB, 'lidarLLIM':lidarLLIM, 'lidarULIM':lidarULIM,
                        'satAGB':satAGB, 'satLLIM':satLLIM, 'satULIM':satULIM})

data_mask = np.isfinite(subplot_df.satAGB)
#####
# remove plots with errosr in field data
# 55410, 56635
###
remove_subplots = ['55410.1','55410.2','55410.3','55410.4','56635.2','56635.3']
for subplot in remove_subplots:
     data_mask[subplot_df['subplot']==subplot]= False
subplot_df = subplot_df[data_mask]
host_plots,subplots_count=np.unique(subplot_df['plot'],return_counts=True)

plot_df = subplot_df.groupby('plot',as_index=False).mean()[subplots_count==4]

# aggregate to plot level
invERR_aggregated = []
for plot in plot_df['plot']:
    mask = np.asarray(plot_list)==plot
    invERR_aggregated.append(np.mean(np.sqrt(np.sum((np.asarray(invERR)[mask])**2))))
plot_df['invERR']=np.asarray(invERR_aggregated)#[data_mask][subplots_count==4]

sns.set()
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=[8,8],sharex=True,sharey=True)
x_range = np.array([0,np.max(plot_df.satULIM)])

axes[0].set_title('Inside lidar survey'); axes[1].set_title('Outside lidar survey')

inside_mask = plot_df['inside_lidar']==1
nfi_mask = plot_df['collection']==1
icm_mask = plot_df['collection']==0
axes[0].plot(x_range,x_range,'--',color='black')
axes[0].plot(plot_df.satAGB[inside_mask*nfi_mask],plot_df.invAGB[inside_mask*nfi_mask],'.',mfc='white',mec='black')
axes[0].plot(plot_df.satAGB[inside_mask*icm_mask],plot_df.invAGB[inside_mask*icm_mask],'.',color='black')
axes[0].errorbar(plot_df.satAGB[inside_mask],plot_df.invAGB[inside_mask],
                    xerr=(plot_df.satAGB[inside_mask]-plot_df.satLLIM[inside_mask],
                        plot_df.satULIM[inside_mask]-plot_df.satAGB[inside_mask]),
                        yerr=plot_df.invERR[inside_mask],
                    marker='',linestyle='',color='0.33',linewidth=0.5)
outside_mask = plot_df['inside_lidar']<1

annotations =[]
temp1,temp2,r,temp3,temp4 = stats.linregress(plot_df.satAGB[inside_mask],plot_df.invAGB[inside_mask])
r2 = r**2
rmse = np.sqrt(np.mean((plot_df.satAGB[inside_mask]-plot_df.invAGB[inside_mask])**2))
rel_rmse = rmse/np.mean(plot_df.invAGB[inside_mask])
print("Validation\n\tR^2 = %.02f" % r2)
print("\tRMSE = %.02f" % rmse)
print("\trelative RMSE = %.02f" % rel_rmse)
annotations.append('R$^2$ = %.2f\nRMSE = %.1f\nrelative RMSE = %.1f%s' % (r2,rmse,rel_rmse*100,'%'))

axes[1].plot(x_range,x_range,'--',color='black')
axes[1].plot(plot_df.satAGB[outside_mask*nfi_mask],plot_df.invAGB[outside_mask*nfi_mask],'.',mfc='white',mec='black')
axes[1].plot(plot_df.satAGB[outside_mask*icm_mask],plot_df.invAGB[outside_mask*icm_mask],'.',color='black')
axes[1].errorbar(plot_df.satAGB[outside_mask],plot_df.invAGB[outside_mask],
                    xerr=(plot_df.satAGB[outside_mask]-plot_df.satLLIM[outside_mask],
                        plot_df.satULIM[outside_mask]-plot_df.satAGB[outside_mask]),
                    yerr=plot_df.invERR[outside_mask],
                    marker='',linestyle='',color='0.33',linewidth=0.5)

temp1,temp2,r,temp3,temp4 = stats.linregress(plot_df.satAGB[outside_mask],plot_df.invAGB[outside_mask])
r2 = r**2
rmse = np.sqrt(np.mean((plot_df.satAGB[outside_mask]-plot_df.invAGB[outside_mask])**2))
rel_rmse = rmse/np.mean(plot_df.invAGB[outside_mask])
print("Validation\n\tR^2 = %.02f" % r2)
print("\tRMSE = %.02f" % rmse)
print("\trelative RMSE = %.02f" % rel_rmse)
annotations.append('R$^2$ = %.2f\nRMSE = %.1f\nrelative RMSE = %.1f%s' % (r2,rmse,rel_rmse*100,'%'))

for ii,ax in enumerate(axes):
    ax.set_aspect('equal')
    ax.set_xlabel('upscaled AGB$_{upscaled}$ / Mg ha$^{-1}$')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.annotate(annotations[ii],xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none', ha='left', va='top')


axes[0].set_ylabel('field AGB$_{inventory}$ / Mg ha$^{-1}$')

fig.tight_layout()
fig.savefig('%s%s_%s_upscaled_vs_inventory.png' % (path2fig,site_id,version))
fig.show()


"""
#===============================================================================
PART C: PLOT DISTRIBUTIONS OF AGB
#-------------------------------------------------------------------------------
"""
inside_mask = np.isfinite(lidar.values)*np.isfinite(agb_med.values)
outside_mask = np.isfinite(agb_med.values)*(inside_mask==False)
lidar_agb = lidar.values[inside_mask]
upscaled_agb_inside = agb_med.values[inside_mask]
upscaled_agb_outside = agb_med.values[outside_mask]
n_in = inside_mask.sum()
n_out = outside_mask.sum()

n_inv_in = subplot_df.inside_lidar.values.sum()
n_inv_out = len(subplot_df)-n_inv_in
inside_mask_inv = subplot_df.inside_lidar.values
outside_mask_inv = subplot_df.inside_lidar.values==False
"""
df = pd.DataFrame({'AGB':np.concatenate((subplot_df.invAGB[inside_mask_inv].values,subplot_df.invAGB[outside_mask_inv].values,lidar_agb,upscaled_agb_inside,upscaled_agb_outside)),
                    'label':np.concatenate((np.tile('inventory (inside lidar area)',n_inv_in),
                                            np.tile('inventory (outside lidar area)',n_inv_out),
                                            np.tile('lidar',n_in),
                                            np.tile('upscaled (inside lidar area)',n_in),
                                            np.tile('upscaled (outside lidar area)',n_out)))})
"""
df = pd.DataFrame({'AGB':np.concatenate((subplot_df.invAGB.values,lidar_agb,upscaled_agb_inside,upscaled_agb_outside)),
                    'label':np.concatenate((np.tile('inventory',n_inv_in+n_inv_out),
                                            np.tile('lidar',n_in),
                                            np.tile('upscaled (inside lidar area)',n_in),
                                            np.tile('upscaled (outside lidar area)',n_out)))})


# plot distribtions
sns.set(rc={"axes.facecolor": (0, 0, 0, 0)})
g = sns.FacetGrid(df, row="label", aspect=6, height=1)
g.map(sns.kdeplot, "AGB", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2, color='0.5')
g.map(sns.kdeplot, "AGB", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

for ii,ax in enumerate(g.axes.ravel()):
    ax.text(.9, .2, g.row_names[ii], fontweight="bold", ha="right", va="center", transform=ax.transAxes)
    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.5)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(left=True)
    ax.grid(False)
plt.savefig('%s%s_%s_agb_distributions.png' % (path2fig,site_id,version))
plt.show()
