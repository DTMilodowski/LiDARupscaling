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
import stats_tools as st

import sys
sys.path.append('../data_io')
import LiDAR_io as lidar
import LiDAR_tools as lidar_tools

import gc

# CHANGE THIS STUFF
# Some file names - shapefiles and rasters should be in a projected coordinate system (i.e. UTM)
version = '035'
las_file_list = './las_list.txt'
inventory_file = '../../data/lidar_calibration/Kiuic_400_live_biomass_unc.shp'
raster_file = '../../data/LiDAR_data/GliHT_TCH_1m_100.tif'
dem_file = '../../data/LiDAR_data/DTM/Kiuic/kiuic_scale100_dtm.tif'#'../../data/LiDAR_data/gliht_dtm.tif'
outfile = '../../data/lidar_calibration/kiuic_plot_lidar_sample_%s.npz' % version # this will be a file to hold the compiled plot data
path2fig = '../../figures/'
plot_area = 400. # 1 ha
radius = np.sqrt(plot_area/np.pi)
gap_ht = 2 # height at which to define canopy gaps
plots_to_plot = ['24.1','15.4','5.2','6.1','3.1']
quantiles = [.025,.1,.25,.5,.75,.9,.975]

# from ICM plots - mean and standard deviation for total AGB in small stems
# (2.5cm <= DBH < 7.5cm)
small_stem_agb = 24.39092
small_stem_std = 13.54176

# LOAD FILES, CONVERT TO FLOAT AND SPECIFY NODATA REGIONS
dem = xr.open_rasterio(dem_file)[0]
chm = xr.open_rasterio(raster_file)[0]

# Coordinates etc
X = chm['x'].values; Y = chm['y'].values
dX = X[1]-X[0]; dY = Y[1]-Y[0]
buffer = 10*radius+np.sqrt(2.*max((dX)**2,(dY)**2))

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
    print('\t\tprocessing %s' % plot,end='\r')
    # Generate mask around AOI to make subsequent code more efficient
    Xmin,Ymin,Xmax,Ymax = plot_info[plot]['buffer'].bounds
    polygon_around_plot = np.array([[Xmin,Ymin],[Xmax,Ymin],[Xmax,Ymax],[Xmin,Ymax]])

    # load raster subsets
    if dY<0:
        chm_sub = chm.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
        dem_sub = dem.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
    else:
        chm_sub = chm.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))
        dem_sub = dem.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))

    # if plot within raster extent, get plot-level data
    if np.all((len(chm_sub.x)>0,len(chm_sub.y)>0)):
        if chm_sub.values.size>0:
            chm_sub.values[dem_sub.values<=0]=-255
            chm_sub.values = chm_sub.values.astype(np.float16)
            chm_sub.values[chm_sub.values<0] = np.nan
            chm_sub.values/=100. # rescale chm values

            dem_sub.values[dem_sub.values<=0]=-9999
            dem_sub.values = dem_sub.values.astype(np.float16)
            dem_sub.values[dem_sub.values<0] = np.nan
            dem_sub.values/=100. # rescale chm values

            if np.sum(np.isfinite(chm_sub.values))>0:
                """
                # load point cloud data
                lidar_pts, starting_ids, trees = lidar.load_lidar_data_by_polygon(las_file_list,polygon_around_plot)

                # subtract ground elevation from point cloud
                XX,YY = np.meshgrid(dem_sub['x'].values,dem_sub['y'].values)
                XX=XX.ravel()
                YY=YY.ravel()
                ZZ=dem_sub.values.ravel()
                temp_ids, dem_trees = lidar.create_KDTree(np.array([XX,YY]).T)
                """
                for subplot in inventory:
                    if ('%.0f' % subplot['properties']['plot'])==plot:
                        id = '%.0f.%.0f' % (subplot['properties']['plot'],subplot['properties']['subplot'])
                        print('\t\tprocessing %s (cluster %i/%i)' % (id,pp+1,len(plot_clusters)),end='\r')
                        #if chm_sub.values.size>0:
                        # for NFI, add in expected AGB component from small stems
                        if len(id)>5:
                            inventory_AGB[id]={'AGB':subplot['properties']['agb']+small_stem_agb,
                                                'uncertainty':np.sqrt(subplot['properties']['unc']**2+small_stem_std**2),
                                                'x':subplot['geometry']['coordinates'][0],
                                                'y':subplot['geometry']['coordinates'][1]}
                        else:
                            inventory_AGB[id]={'AGB':subplot['properties']['agb'],
                                                'uncertainty':subplot['properties']['unc'],
                                                'x':subplot['geometry']['coordinates'][0],
                                                'y':subplot['geometry']['coordinates'][1]}

                        plot_centre = Point(subplot['geometry']['coordinates'][0],subplot['geometry']['coordinates'][1])
                        chm_results[id] = gst.sample_raster_by_point_neighbourhood(chm_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)
                        dem_results[id] = gst.sample_raster_by_point_neighbourhood(dem_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)

                        if np.mean(np.isfinite(chm_results[id]['raster_values'][0].values))>=.9:
                            chm_results[id]['quantiles'] = st.weighted_quantiles(chm_results[id]['raster_values'][0].values,chm_results[id]['weights'],quantiles)
                            """
                            # now get canopy cover directly from LiDAR point cloud
                            pts_sub = lidar_tools.filter_lidar_data_by_neighbourhood(lidar_pts,[subplot['geometry']['coordinates'][0],subplot['geometry']['coordinates'][1]],radius)
                            point_heights = np.zeros(pts_sub.shape[0])

                            for idx in range(0,point_heights.size):
                                dist,pixel_id = dem_trees[0].query(pts_sub[idx,:2],k=1)
                                point_heights[idx] =pts_sub[idx,2]-ZZ[pixel_id]

                            point_weights = 1/pts_sub[:,7]
                            canopy_mask = point_heights>=gap_ht
                            chm_results[id]['cover_fraction_from_pointcloud'] = np.sum(point_weights[canopy_mask]/np.sum(point_weights))
                            chm_results[id]['point_cloud'] = pts_sub.copy()
                            chm_results[id]['point_heights'] = point_heights.copy()
                            """
                        else:
                            chm_results[id]['status']='FAIL'

                        if chm_results[id]['status']=='PASS':
                            # create second subset of the subplot for visualisation
                            Xmin2,Ymin2,Xmax2,Ymax2 = plot_centre.buffer(5*radius).bounds
                            if dY<0:
                                chm_results[id]['chm_raster'] = chm_sub.sel(x=slice(Xmin2,Xmax2),y=slice(Ymax2,Ymin2)).copy(deep=True)
                                dem_results[id]['dem_raster'] = dem_sub.sel(x=slice(Xmin2,Xmax2),y=slice(Ymax2,Ymin2)).copy(deep=True)
                            else:
                                dem_results[id]['dem_raster'] = dem_sub.sel(x=slice(Xmin2,Xmax2),y=slice(Ymin2,Ymax2)).copy(deep=True)
                                chm_results[id]['chm_raster'] = chm_sub.sel(x=slice(Xmin2,Xmax2),y=slice(Ymin2,Ymax2)).copy(deep=True)
                            # rescale coordinates to plot centre
                            #chm_results[id]['chm_raster'].x.values=chm_results[id]['chm_raster'].x.values - plot_centre.coords[0][0]
                            #chm_results[id]['chm_raster'].y.values=chm_results[id]['chm_raster'].y.values - plot_centre.coords[0][1]
                            #dem_results[id]['dem_raster'].x.values=dem_results[id]['dem_raster'].x.values - plot_centre.coords[0][0]
                            #dem_results[id]['dem_raster'].y.values=dem_results[id]['dem_raster'].y.values - plot_centre.coords[0][1]
                            chm_results[id]['chm_raster'].coords['x']=chm_results[id]['chm_raster'].coords['x'] - plot_centre.x
                            chm_results[id]['chm_raster'].coords['y']=chm_results[id]['chm_raster'].coords['y'] - plot_centre.y
                            dem_results[id]['dem_raster'].coords['x']=dem_results[id]['dem_raster'].coords['x'] - plot_centre.x
                            dem_results[id]['dem_raster'].coords['y']=dem_results[id]['dem_raster'].coords['y'] - plot_centre.y



# CALIBRATION STATISTICS
ID=[]; AGB = []; TCH = []; COVER = []; NODATA = []; AGBunc = []
for plot in chm_results.keys():
    id = chm_results[plot]['id']
    if chm_results[id]['status']=='PASS':
        if chm_results[id]['weighted_average']>0:
            if np.sum(np.isnan(chm_results[id]['raster_values'][0])*chm_results[id]['weights'])<0.05*np.sum(chm_results[id]['weights']):
                weights = chm_results[id]['weights']/np.sum(np.isfinite(chm_results[id]['raster_values'][0].values)*chm_results[id]['weights'])
                ID.append(id)
                AGB.append(inventory_AGB[id]['AGB'])
                AGBunc.append(inventory_AGB[id]['uncertainty'])
                TCH.append(chm_results[id]['weighted_average'][0])
                COVER.append(np.sum((chm_results[id]['raster_values'][0].values>=gap_ht)*weights))
                NODATA.append(np.mean(~np.isfinite(chm_results[id]['raster_values'][0].values)))

# Convert to np arrays for conditional indexing convenience
AGB = np.asarray(AGB); AGBunc = np.asarray(AGBunc); TCH = np.asarray(TCH); COVER = np.asarray(COVER); ID = np.asarray(ID)

"""
Explore positional uncertainty and it's impact on TCH estimation
"""
chm_results_mc = {}
positional_error = 5.
n_iter = 100
# SAMPLE RASTER FOR EACH PLOT NEIGHBOURHOOD
plot_clusters = plot_info.keys()
for pp, plot in enumerate(plot_clusters):
    print('\t\tprocessing %s' % plot,end='\r')
    # Generate mask around AOI to make subsequent code more efficient
    Xmin,Ymin,Xmax,Ymax = plot_info[plot]['buffer'].bounds
    Xmin-=100;Xmax+=100;Ymin-=100;Ymax+=100
    if dY<0:
        chm_sub = chm.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
        dem_sub = dem.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))
    else:
        chm_sub = chm.sel(x=slice(Xmin,Xmax),y=slice(Ymin,Ymax))
        dem_sub = dem.sel(x=slice(Xmin,Xmax),y=slice(Ymax,Ymin))

    if np.all((len(chm_sub.x)>0,len(chm_sub.y)>0)):
        if chm_sub.values.size>0:
            chm_sub.values[dem_sub.values==0]=-255
            chm_sub.values = chm_sub.values.astype(np.float16)
            chm_sub.values[chm_sub.values<0] = np.nan
            chm_sub.values/=100. # rescale chm values

            dem_sub.values[dem_sub.values<=0]=-9999
            dem_sub.values = dem_sub.values.astype(np.float16)
            dem_sub.values[dem_sub.values<0] = np.nan
            dem_sub.values/=100. # rescale chm values

            if np.sum(np.isfinite(chm_sub.values))>0:
                # load point cloud data
                polygon_around_plot = np.array([[Xmin,Ymin],[Xmax,Ymin],[Xmax,Ymax],[Xmin,Ymax]])
                lidar_pts, starting_ids, trees = lidar.load_lidar_data_by_polygon(las_file_list,polygon_around_plot)

                # subtract ground elevation from point cloud
                XX,YY = np.meshgrid(dem_sub['x'].values,dem_sub['y'].values)
                XX=XX.ravel()
                YY=YY.ravel()
                ZZ=dem_sub.values.ravel()
                temp_ids, dem_trees = lidar.create_KDTree(np.array([XX,YY]).T)

                for subplot in inventory:
                    if ('%.0f' % subplot['properties']['plot'])==plot:
                        id = '%.0f.%.0f' % (subplot['properties']['plot'],subplot['properties']['subplot'])
                        print('\t\tprocessing %s (cluster %i/%i)' % (id,pp+1,len(plot_clusters)),end='\r')
                        #if chm_sub.values.size>0:
                        if chm_results[id]['status']=='PASS':
                            tch_mc = np.zeros(n_iter)*np.nan
                            cover_mc = np.zeros(n_iter)*np.nan
                            cover_fraction_from_pointcloud_mc = np.zeros(n_iter)*np.nan
                            quantiles_mc = np.zeros((n_iter,len(quantiles)))*np.nan
                            for ii in range(0,n_iter):
                                xerr = np.random.randn()*positional_error
                                yerr = np.random.randn()*positional_error
                                plot_centre = Point(subplot['geometry']['coordinates'][0]+xerr,subplot['geometry']['coordinates'][1]+yerr)
                                results_iter = gst.sample_raster_by_point_neighbourhood(chm_sub,plot_centre,radius,x_dim='x',y_dim='y',label = id)
                                if results_iter['status']=='PASS':
                                    tch_mc[ii] = results_iter['weighted_average'][0]
                                    weights = results_iter['weights']/np.sum(np.isfinite(results_iter['raster_values'][0].values)*results_iter['weights'])
                                    cover_mc[ii]= np.sum((results_iter['raster_values'][0].values>=gap_ht)*weights)
                                    quantiles_mc[ii] = st.weighted_quantiles(results_iter['raster_values'][0].values,results_iter['weights'],quantiles)

                                    # now get canopy cover directly from LiDAR point cloud
                                    pts_sub = lidar_tools.filter_lidar_data_by_neighbourhood(lidar_pts,[subplot['geometry']['coordinates'][0]+xerr,subplot['geometry']['coordinates'][1]+yerr],radius)

                                    point_heights = np.zeros(pts_sub.shape[0])
                                    for idx in range(0,point_heights.size):
                                        dist,pixel_id = dem_trees[0].query(pts_sub[idx,:2],k=1)
                                        point_heights[idx] =pts_sub[idx,2]-ZZ[pixel_id]

                                    point_weights = 1/pts_sub[:,7]
                                    canopy_mask = point_heights>=gap_ht
                                    cover_fraction_from_pointcloud_mc[ii] = np.sum(point_weights[canopy_mask]/np.sum(point_weights))

                            chm_results[id]['tch_mc']=tch_mc.copy()
                            chm_results[id]['tch_mc_mean']=np.mean(tch_mc)
                            chm_results[id]['cover_mc']=cover_mc.copy()
                            chm_results[id]['cover_mc_mean']=np.mean(cover_mc)
                            chm_results[id]['quantiles_mc']=quantiles_mc.copy()
                            chm_results[id]['quantiles_mc_mean']=np.mean(quantiles_mc)
                            chm_results[id]['quantiles_ref']=quantiles
                            chm_results[id]['cover_fraction_from_pointcloud_mc']=cover_fraction_from_pointcloud_mc.copy()
                            chm_results[id]['cover_fraction_from_pointcloud_mc_mean']=np.mean(cover_fraction_from_pointcloud_mc)

                            chm_results_mc[id]={}
                            chm_results_mc[id]['TCH'] = {'mean':np.mean(tch_mc),
                                                    'sd':np.std(tch_mc),
                                                    'med':np.median(tch_mc),
                                                    '2.5perc':np.percentile(tch_mc,2.5),
                                                    '97.5perc':np.percentile(tch_mc,97.5),
                                                    '25perc':np.percentile(tch_mc,25),
                                                    '50perc':np.percentile(tch_mc,50),
                                                    '75perc':np.percentile(tch_mc,75),
                                                    'all':tch_mc.copy()}

                            chm_results_mc[id]['Cover'] = {'mean':np.mean(cover_fraction_from_pointcloud_mc),
                                                    'sd':np.std(cover_fraction_from_pointcloud_mc),
                                                    'med':np.median(cover_fraction_from_pointcloud_mc),
                                                    '2.5perc':np.percentile(cover_fraction_from_pointcloud_mc,2.5),
                                                    '97.5perc':np.percentile(cover_fraction_from_pointcloud_mc,97.5),
                                                    '25perc':np.percentile(cover_fraction_from_pointcloud_mc,25),
                                                    '50perc':np.percentile(cover_fraction_from_pointcloud_mc,50),
                                                    '75perc':np.percentile(cover_fraction_from_pointcloud_mc,75),
                                                    'all':tch_mc.copy()}

# clear memory
dem.close(); dem=None
gc.collect()

# save for future use
collated_results = {'chm':chm_results,'dem':dem_results,'inventory':inventory_AGB}
np.savez(outfile,collated_results)

# CALIBRATION STATISTICS
mean=[]; sd = []; CI95_l = []; CI95_u = []; CI50_l = []; CI50_u = []
for plot in chm_results.keys():
    id = chm_results[plot]['id']
    if chm_results[id]['status']=='PASS':
        if chm_results[id]['weighted_average']>0:
            if np.sum(np.isnan(chm_results[id]['raster_values'][0])*chm_results[id]['weights'])<0.05*np.sum(chm_results[id]['weights']):
                mean.append(chm_results_mc[id]['TCH']['mean'])
                sd.append(chm_results_mc[id]['TCH']['sd'])
                CI95_l.append(chm_results_mc[id]['TCH']['2.5perc'])
                CI95_u.append(chm_results_mc[id]['TCH']['97.5perc'])
                CI50_l.append(chm_results_mc[id]['TCH']['25perc'])
                CI50_u.append(chm_results_mc[id]['TCH']['75perc'])
SD=np.array(sd);MEAN=np.array(mean);CI95_l=np.array(CI95_l);CI95_u=np.array(CI95_u)
CI50_l=np.array(CI50_l);CI50_u=np.array(CI50_u)


"""
PLOTTING
"""
# Plot the figure with montecarlo postion errors
fig,fig_axes = plt.subplots(nrows=2,ncols=3,figsize=[12,7])
axes=fig_axes.flatten()
axes[0].plot(MEAN,AGB,'.',color='black')
#axes[0].errorbar(MEAN,AGB,xerr=SD,
#                    marker='',linestyle='',color='0.5',linewidth=0.5)
axes[0].errorbar(MEAN,AGB,xerr=(MEAN-CI95_l,CI95_u-MEAN),
                    marker='',linestyle='',color='0.67',linewidth=0.5)
axes[0].errorbar(MEAN,AGB,xerr=(MEAN-CI50_l,CI50_u-MEAN),yerr=AGBunc,
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
fig.savefig('%slidar_TCH_vs_field_AGB_400m_after_mc_%s.png' % (path2fig,version))
fig.show()


# Plot the figure with montecarlo postion errors
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[6,6])
ax.plot(MEAN,AGB,'.',color='black')
ax.errorbar(MEAN,AGB,xerr=(MEAN-CI95_l,CI95_u-MEAN),
                    marker='',linestyle='',color='0.67',linewidth=0.5)
ax.errorbar(MEAN,AGB,xerr=(MEAN-CI50_l,CI50_u-MEAN),yerr=(AGBunc),
                    marker='',linestyle='',color='0.33',linewidth=0.75)
ax.set_title('TCH vs. inventory AGB')
ax.set_xlabel('mean TCH / m')
ax.set_ylabel('field AGB / Mg ha$^{-1}$')
fig.tight_layout()
fig.savefig('%slidar_TCH_vs_field_AGB_400m_after_mc_single_panel_%s.png' % (path2fig,version))
fig.show()
