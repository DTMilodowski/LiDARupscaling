"""
#-----------------------------
# apply_nodata_buffer.py
#=============================
# D. T. Milodowski, 18/04/2019
#-----------------------------
# This function applies a nodata buffer around the data within a tiff file,
# and writes to a new geotiff
#-----------------------------
"""

"""
Import libraries
"""
import numpy as np
import xarray as xr
from scipy import ndimage as nd

"""
List some file paths
"""
