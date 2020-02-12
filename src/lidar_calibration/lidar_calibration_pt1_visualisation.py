"""
lidar_calibration_pt1_visualisation.py
================================================================================
Collate the field and LiDAR TCH data and visualise the relationship between
canopy height and AGB for the plots. Produces a six panel figure, the TCH-AGB
relationship and five plots from the dataset.
D.T.Milodowski
"""
import numpy as np
import xarray as xr

import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import fiona
from shapely.geometry import Point, Polygon
import geospatial_tools as gst

import gc

# CHANGE THIS STUFF
# Some file names - shapefiles and rasters should be in a projected coordinate system (i.e. UTM)
inventory_file = '../../data/lidar_calibration/Kiuic_400_live_trees.shp'
raster_file = '../../data/LiDAR_data/GliHT_TCH_1m_100.tif'
dem_file = '../../data/LiDAR_data/gliht_dtm.tif'
outfile = '../../data/lidar_calibration/sample_test.npz' # this will be a file to hold the compiled plot data
path2fig = '../../figures/'
plot_area = 400. # 1 ha
radius = np.sqrt(plot_area/np.pi)
gap_ht = 8.66 # height at which to define canopy gaps
plots_to_plot = ['24.1','15.4','5.2','6.1','3.1']

# LOAD FILES, CONVERT TO FLOAT AND SPECIFY NODATA REGIONS
dem = xr.open_rasterio(dem_file)[0].sel(x=slice(229000,239000),y=slice(2230310,2214000))
chm = xr.open_rasterio(raster_file)[0].sel(x=slice(229000,239000),y=slice(2230310,2214000))
chm.values[dem.values==0]=-255
chm.values = chm.values.astype(np.float16)
chm.values[chm.values<0] = np.nan
chm.values/=100. # rescale chm values

# Coordinates etc
X = chm['x'].values; Y = chm['y'].values
dX = X[1]-X[0]; dY = Y[1]-Y[0]
buffer = radius+np.sqrt(2.*max((dX)**2,(dY)**2))

# inventory data
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

# A DICTIONARY TO CONTAIN THE RESULTS
chm_results = {}
dem_results = {}
inventory_AGB = {}

# SAMPLE RASTER FOR EACH PLOT NEIGHBOURHOOD
plot_clusters = plot_info.keys()
for pp, plot in enumerate(plot_clusters):
    # Generate mask around AOI to make subsequent code more efficient
    Xmin,Ymin,Xmax,Ymax = plot_info[plot]['buffer'].bounds
    if dY<0:
        chm_sub = chm.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
        dem_sub = dem.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
    else:
        chm_sub = chm.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))
        dem_sub = dem.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))

    for subplot in inventory:
        if ('%.0f' % subplot['properties']['plot'])==plot:
            id = '%.0f.%.0f' % (subplot['properties']['plot'],subplot['properties']['subplot'])
            print('\t\tprocessing %s (cluster %i/%i)' % (id,pp+1,len(plot_clusters)),end='\r')
            if chm_sub.values.size>0:
                inventory_AGB[id]=subplot['properties']['agb']
                plot_centre = Point(subplot['geometry']['coordinates'][0],subplot['geometry']['coordinates'][1])
                chm_results[id] = gst.sample_raster_by_point_neighbourhood(chm_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)
                dem_results[id] = gst.sample_raster_by_point_neighbourhood(dem_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)

for pp, plot in enumerate(plot_clusters):
    # Generate mask around AOI to make subsequent code more efficient
    Xmin,Ymin,Xmax,Ymax = plot_info[plot]['buffer'].bounds
    if dY<0:
        chm_sub = chm.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
        dem_sub = dem.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
    else:
        chm_sub = chm.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))
        dem_sub = dem.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))
    for subplot in inventory:
        if ('%.0f' % subplot['properties']['plot'])==plot:
            id = '%.0f.%.0f' % (subplot['properties']['plot'],subplot['properties']['subplot'])
            plot_centre = Point(subplot['geometry']['coordinates'][0],subplot['geometry']['coordinates'][1])
            if chm_sub.values.size>0:
                if chm_results[id]['status']=='PASS':
                    # create second subset of the subplot for visualisation
                    Xmin2,Ymin2,Xmax2,Ymax2 = plot_centre.buffer(5*radius).bounds
                    if dY<0:
                        chm_results[id]['chm_raster'] = chm.sel(x=slice(Xmin2,Xmax2),y=slice(Ymax2,Ymin2)).copy(deep=True)
                        dem_results[id]['dem_raster'] = dem.sel(x=slice(Xmin2,Xmax2),y=slice(Ymax2,Ymin2)).copy(deep=True)
                    else:
                        dem_results[id]['dem_raster'] = dem.sel(x=slice(Xmin2,Xmax2),y=slice(Ymin2,Ymax2)).copy(deep=True)
                        chm_results[id]['chm_raster'] = chm.sel(x=slice(Xmin2,Xmax2),y=slice(Ymin2,Ymax2)).copy(deep=True)
                    # rescale coordinates to plot centre
                    chm_results[id]['chm_raster'].x.values=chm_results[id]['chm_raster'].x.values - plot_centre.coords[0][0]
                    chm_results[id]['chm_raster'].y.values=chm_results[id]['chm_raster'].y.values - plot_centre.coords[0][1]
                    dem_results[id]['dem_raster'].x.values=dem_results[id]['dem_raster'].x.values - plot_centre.coords[0][0]
                    dem_results[id]['dem_raster'].y.values=dem_results[id]['dem_raster'].y.values - plot_centre.coords[0][1]

# clear memory
dem.close(); dem=None
gc.collect()

# CALIBRATION STATISTICS
ID=[]; AGB = []; TCH = []; COVER = []; NODATA = []
for plot in chm_results.keys():
    id = chm_results[plot]['id']
    if chm_results[id]['status']=='PASS':
        if chm_results[id]['weighted_average']>0:
            if np.sum(np.isnan(chm_results[id]['raster_values'][0])*chm_results[id]['weights'])<0.05*np.sum(chm_results[id]['weights']):
                weights = chm_results[id]['weights']/np.sum(np.isfinite(chm_results[id]['raster_values'][0].values)*chm_results[id]['weights'])
                ID.append(id)
                AGB.append(inventory_AGB[id])
                TCH.append(chm_results[id]['weighted_average'][0])
                COVER.append(np.sum((chm_results[id]['raster_values'][0].values>=gap_ht)*weights))
                NODATA.append(np.mean(~np.isfinite(chm_results[id]['raster_values'][0].values)))

# Convert to np arrays for conditional indexing convenience
AGB = np.asarray(AGB); TCH = np.asarray(TCH); COVER = np.asarray(COVER); ID = np.asarray(ID)

"""
Explore positional uncertainty and it's impact on TCH estimation
"""
chm_results_mc = {}
positional_error = 5.
n_iter = 100
# SAMPLE RASTER FOR EACH PLOT NEIGHBOURHOOD
plot_clusters = plot_info.keys()
for pp, plot in enumerate(plot_clusters):
    # Generate mask around AOI to make subsequent code more efficient
    Xmin,Ymin,Xmax,Ymax = plot_info[plot]['buffer'].bounds
    Xmin-=50;Xmax+=50;Ymin-=50;Ymax+=50
    if dY<0:
        chm_sub = chm.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
    else:
        chm_sub = chm.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))

    for subplot in inventory:
        if ('%.0f' % subplot['properties']['plot'])==plot:
            id = '%.0f.%.0f' % (subplot['properties']['plot'],subplot['properties']['subplot'])
            print('\t\tprocessing %s (cluster %i/%i)' % (id,pp+1,len(plot_clusters)),end='\r')
            if chm_sub.values.size>0:
                tch_mc = np.zeros(n_iter)*np.nan
                for ii in range(0,n_iter):
                    xerr = np.random.randn()*positional_error
                    yerr = np.random.randn()*positional_error
                    plot_centre = Point(subplot['geometry']['coordinates'][0]+xerr,subplot['geometry']['coordinates'][1]+yerr)
                    results_iter = gst.sample_raster_by_point_neighbourhood(chm_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)
                    if results_iter['status']=='PASS':
                        tch_mc[ii] = results_iter['weighted_average'][0]

                    chm_results[id]['tch_mc_position']=tch_mc.copy()

                chm_results_mc[id] = {'mean':np.mean(tch_mc),'sd':np.std(tch_mc),
                                        'med':np.median(tch_mc),'2.5perc':np.percentile(tch_mc,2.5),
                                        '97.5perc':np.percentile(tch_mc,97.5),
                                        '25perc':np.percentile(tch_mc,25),
                                        '50perc':np.percentile(tch_mc,50),
                                        '75perc':np.percentile(tch_mc,75),
                                        '95perc':np.percentile(tch_mc,95)}

# CALIBRATION STATISTICS
mean=[]; sd = []; CI95_l = []; CI95_u = []; CI50_l = []; CI50_u = []
for plot in chm_results.keys():
    id = chm_results[plot]['id']
    if chm_results[id]['status']=='PASS':
        if chm_results[id]['weighted_average']>0:
            if np.sum(np.isnan(chm_results[id]['raster_values'][0])*chm_results[id]['weights'])<0.05*np.sum(chm_results[id]['weights']):
                mean.append(chm_results_mc[id]['mean'])
                sd.append(chm_results_mc[id]['sd'])
                CI95_l.append(chm_results_mc[id]['2.5perc'])
                CI95_u.append(chm_results_mc[id]['97.5perc'])
                CI50_l.append(chm_results_mc[id]['25perc'])
                CI50_u.append(chm_results_mc[id]['75perc'])
SD=np.array(sd);MEAN=np.array(mean);CI95_l=np.array(CI95_l);CI95_u=np.array(CI95_u)
CI50_l=np.array(CI50_l);CI50_u=np.array(CI50_u)

# save for future use
collated_results = {'chm':chm_results,'dem':dem_results,'inventory':inventory_AGB}
np.savez(outfile,collated_results)

"""
PLOTTING
"""
# Plot the figure (original)
fig,fig_axes = plt.subplots(nrows=2,ncols=3,figsize=[12,7])
axes=fig_axes.flatten()
axes[0].plot(TCH,AGB,'.',color='black')
axes[0].set_title('TCH vs. inventory AGB')
axes[0].set_xlabel('mean TCH / m')
axes[0].set_ylabel('field AGB / Mg ha$^{-1}$')
for ii,plot_id in enumerate(plots_to_plot):
    plot_boundary = mpatches.Circle((0,0),radius,ec='white',fill=False)#,fc=None)
    #im = axes[ii+1].imshow(chm_results[plot_id]['chm_raster'][0],vmin=0,vmax=23)
    chm_results[plot_id]['chm_raster'].plot(ax=axes[ii+1], vmin=0, vmax=21,
                extend='max', cbar_kwargs={'label':'height / m'})
    axes[ii+1].add_artist(plot_boundary)
    axes[ii+1].set_title('plot %s; AGB = %.1f Mg/ha' % (plot_id,AGB[ID==plot_id]))
    axes[0].plot(TCH[ID==plot_id],AGB[ID==plot_id],'.',color='#2db27d')#'#bc1655')
    axes[0].annotate(' %s' % plot_id,xy=(TCH[ID==plot_id],AGB[ID==plot_id]),color='#2db27d')
    if ii>=2:
        axes[ii+1].set_xlabel('distance / m')
    else:
        axes[ii+1].set_xlabel('')
    axes[ii+1].set_ylabel('')
fig.tight_layout()
fig.savefig('%slidar_TCH_vs_field_AGB_400m.png' % path2fig)
fig.show()

# Plot the figure with montecarlo postion errors
fig,fig_axes = plt.subplots(nrows=2,ncols=3,figsize=[12,7])
axes=fig_axes.flatten()
axes[0].plot(MEAN,AGB,'.',color='black')
#axes[0].errorbar(MEAN,AGB,xerr=SD,
#                    marker='',linestyle='',color='0.5',linewidth=0.5)
axes[0].errorbar(MEAN,AGB,xerr=(MEAN-CI95_l,CI95_u-MEAN),
                    marker='',linestyle='',color='0.67',linewidth=0.5)
axes[0].errorbar(MEAN,AGB,xerr=(MEAN-CI50_l,CI50_u-MEAN),
                    marker='',linestyle='',color='0.33',linewidth=0.75)
axes[0].set_title('TCH vs. inventory AGB')
axes[0].set_xlabel('mean TCH / m')
axes[0].set_ylabel('field AGB / Mg ha$^{-1}$')
for ii,plot_id in enumerate(plots_to_plot):
    plot_boundary = mpatches.Circle((0,0),radius,ec='white',fill=False)#,fc=None)
    #im = axes[ii+1].imshow(chm_results[plot_id]['chm_raster'][0],vmin=0,vmax=23)
    chm_results[plot_id]['chm_raster'].plot(ax=axes[ii+1], vmin=0, vmax=21,
                extend='max', cbar_kwargs={'label':'height / m'})
    axes[ii+1].add_artist(plot_boundary)
    axes[ii+1].set_title('plot %s; AGB = %.1f Mg/ha' % (plot_id,AGB[ID==plot_id]))
    axes[0].plot(MEAN[ID==plot_id],AGB[ID==plot_id],'.',color='#2db27d')#'#bc1655')
    axes[0].annotate(' %s' % plot_id,xy=(TCH[ID==plot_id],AGB[ID==plot_id]),color='#2db27d')
    if ii>=2:
        axes[ii+1].set_xlabel('distance / m')
    else:
        axes[ii+1].set_xlabel('')
    axes[ii+1].set_ylabel('')
fig.tight_layout()
fig.savefig('%slidar_TCH_vs_field_AGB_400m_after_mc.png' % path2fig)
fig.show()
