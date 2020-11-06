"""
pt4_protected_areas.py
--------------------------------------------------------------------------------
COMPARISON OF AGB DISTRIBUTIONS FOR PROTECTED AREAS, PRODUCTION FOREST AND
RESTORATION AREAS
The analysis undertaken is as follows:

- comparison of upscaled AGB distributions for three subregions:
    (i)     protected areas (both national and state level (i.e. Reserva Estatal
            Biocultural del Puuc)
    (ii)    production forest
    (iii)   areas of disturbed forest allocated for restoration


This code built using the open source programming language python, and utilises
the geospatial library xarray (http://xarray.pydata.org/en/stable/)

01/09/2020 - D. T. Milodowski
--------------------------------------------------------------------------------
"""
"""
# Import the necessary packages
"""
import numpy as np                  # standard package for scientific computing
import pandas as pd                 # dataframes
import xarray as xr                 # xarray geospatial package
import seaborn as sns               # another useful plotting package
import matplotlib.pyplot as plt     # plotting package

"""
Project Info
"""
site_id = 'kiuic'
version = '034'
path2data = '/exports/csce/datastore/geos/users/dmilodow/FOREST2020/LiDARupscaling/data/'
path2upscaled = '/exports/csce/datastore/geos/groups/gcel/YucatanBiomass/output/'
path2fig= '../figures/'

"""
#===============================================================================
PART A: LOAD IN DATA AND SPLIT THE TRAINING DATA FROM THE REMAINING DATA
#-------------------------------------------------------------------------------
"""
print('Loading data')
# lidar data @20m resolution
national_PA_file = '%s/National_PA_UTM.tif' % path2data
national_PA = xr.open_rasterio(national_PA_file).values[0]
state_PA_file = '%s/State_PA_UTM.tif' % path2data
state_PA = xr.open_rasterio(state_PA_file).values[0]
kaxil_kiuic_file =  '%s/Reserva_Biocultural_Kaxil_Kiuic.tif' % path2data
kaxil_PA = xr.open_rasterio(kaxil_kiuic_file).values[0]

# upscaled data
med_agb_file = '%s/kiuic_%s_rfbc_agb_upscaled_median.tif' % (path2upscaled,version)
agb_med = xr.open_rasterio(med_agb_file)[0]
agb_med.values[agb_med.values==-9999]=np.nan
mask= np.isfinite(agb_med.values)

"""
#===============================================================================
PART B: PLOT DISTRIBUTIONS OF AGB ACCORDING TO FOREST MANAGEMENT CATEGORY
#-------------------------------------------------------------------------------
"""
print('plotting AGB distributions')
kaxil_mask = (kaxil_PA==1) * mask
del_puuc_mask = (state_PA==1) * (kaxil_PA!=1) * mask
kaax_mask = (national_PA==1) * (state_PA!=1) * (state_PA!=2) * (kaxil_PA!=1) * mask
production_mask = (national_PA==2) * (state_PA!=1) * (state_PA!=2) * (kaxil_PA!=1) * mask
restoration_mask = (national_PA==3) * (state_PA!=1) * (state_PA!=2) * (kaxil_PA!=1) * mask

agb_kaxil = agb_med.values[kaxil_mask]
agb_del_puuc = agb_med.values[del_puuc_mask]
agb_kaax = agb_med.values[kaax_mask]
agb_restoration = agb_med.values[restoration_mask]
agb_production = agb_med.values[production_mask]

n_kaxil = agb_kaxil.size
n_puuc = agb_del_puuc.size
n_kaax = agb_kaax.size
n_restoration = agb_restoration.size
n_production = agb_production.size

df = pd.DataFrame({'AGB':np.concatenate((agb_kaxil,agb_del_puuc,agb_kaax,agb_production,agb_restoration)),
                    'label':np.concatenate((np.tile('Kaxil Kiuic',n_kaxil),
                                            np.tile('del Puuc',n_puuc),
                                            np.tile("Bala'an Kaax",n_kaax),
                                            np.tile('production',n_production),
                                            np.tile('restoration',n_restoration)))})

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
plt.savefig('%s%s_%s_agb_distributions_by_forest_management_class.png' % (path2fig,site_id,version))
plt.show()
