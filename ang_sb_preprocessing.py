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
# import study area buffer shpfile
# note: Santa Barbara is right at the edge of utm zone 10/11, important to make sure crs matches. in this case it does.
with fiona.open("../data/shp/ang_ucsb_utm11n_wgs84.shp") as shapefile:
    shp = [feature["geometry"] for feature in shapefile]
    
# output false color and rgb?
fcrgb_flag = True
# set bands for fc/rgb output
r = 66
g = 46
b = 28
nir = 98
swir1 = 262
    
# list lines
# 2014
#lines = ["20140603t195821", "20140603t201129", "20140603t202947"] # individual lines
lines = ['20140603_sbdr_masked_mosaic'] # mosaicked lines. can mosaic using rio.merge.merge, this takes a long time (particularly w/ many bands) so I have pre-mosaicked these in a separate script
# pick the line
linei = lines[0]
 
# import 
# 2014
#t1 = rio.open("../data/AVng" + linei + "_corr_v1_rot")
t1 = rio.open("../data/AVng" + linei)

# mask using shapefile
t1c, t1c_trans = rio.mask.mask(t1, shp, crop = True)

# plot single band 
#t1c[t1c < -999] = np.nan
#plt.imshow(t1c[100,:,:])

# copy metadata and update
t1c_meta = t1.meta.copy()
t1c_meta.update({'driver': "GTiff",
                 'width': t1c.shape[2],
                 'height': t1c.shape[1],
                 'transform': t1c_trans})

# output
# 2014
with rio.open("../data/AVng" + linei + "_clip.tif", 'w', **t1c_meta) as out:
    out.write(t1c)

if fcrgb_flag == 1:
    # output easy rgb
    t1c_rgb = t1c[[r,g,b],:,:]
    t1c_fc = t1c[[r,nir,swir1],:,:]
    # meta
    t1c_ql_meta = t1c_meta.copy()
    t1c_meta.update({'count': 3})
    # output
    with rio.open("../data/AVng" + linei + "clip_falsecolor.tif", 'w', **t1c_meta) as out:
        out.write(t1c_fc)
    with rio.open("../data/AVng" + linei + "clip_rgb.tif", 'w', **t1c_meta) as out:
        out.write(t1c_rgb)