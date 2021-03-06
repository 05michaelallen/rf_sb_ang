#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 07:58:15 2021

@author: mallen
"""

import os
import rasterio as rio
from rasterio import mask 
import numpy as np
import fiona
import matplotlib.pyplot as plt

os.chdir("/Volumes/ellwood/rf_sb_ang/code")

# =============================================================================
# 
# =============================================================================
# output false color and rgb?
flag_QUICKLOOK = True
flag_DIAG_PLOT = True

# set bands for fc/rgb quicklooks
r = 66
g = 46
b = 28
nir = 98
swir1 = 262

# import study area buffer shpfile
# note: Santa Barbara is right at the edge of utm zone 10/11, important to make sure crs matches. in this case it does.
with fiona.open("../data/shp/ang_ucsb_utm11n_wgs84.shp") as shapefile:
    shp = [feature["geometry"] for feature in shapefile]

# list lines
# note: these are specific to the 2014 AVNG data
# mosaicked lines. can mosaic using rio.merge.merge, 
# this takes a long time (particularly w/ many bands) so I have pre-mosaicked these in a separate script
lines = ['20140603_sbdr_masked_mosaic'] 
# pick the line
linei = lines[0]
 
# open connection
t1 = rio.open("../data/AVng" + linei)

# mask using shapefile
t1c, t1c_trans = mask.mask(t1, shp, crop = True)

# copy metadata and update
t1c_meta = t1.meta.copy()
t1c_meta.update({'driver': "GTiff",
                 'nodata': -999,
                 'width': t1c.shape[2],
                 'height': t1c.shape[1],
                 'transform': t1c_trans})

# output
with rio.open("../data/AVng" + linei + "_clip.tif", 'w', **t1c_meta) as out:
    out.write(t1c)

# output quicklooks
if flag_QUICKLOOK:
    # output easy rgb
    t1c_rgb = t1c[[r,g,b],:,:]
    t1c_fc = t1c[[swir1,nir,r],:,:]
    # meta
    t1c_ql_meta = t1c_meta.copy()
    t1c_ql_meta.update({'count': 3})
    # output
    with rio.open("../data/AVng" + linei + "clip_falsecolor.tif", 'w', **t1c_ql_meta) as out:
        out.write(t1c_fc)
    with rio.open("../data/AVng" + linei + "clip_rgb.tif", 'w', **t1c_ql_meta) as out:
        out.write(t1c_rgb)
        
# plot single band for diagnostics
if flag_DIAG_PLOT:
    t1c[t1c < 0] = np.nan
    p = plt.imshow(t1c[nir,:,:])
    plt.colorbar(p)
