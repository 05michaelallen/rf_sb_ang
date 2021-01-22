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


os.chdir("/Volumes/ellwood/ang_sb_portfolio/code")
# =============================================================================
# 
# =============================================================================
with fiona.open("../data/shp/ang_sb_shp_utm11.shp") as shapefile:
    shp = [feature["geometry"] for feature in shapefile]
    
# output fc and rgb?
fcrgb_flag = 0
    
# list lines
# 2014
#lines = ["20140603t195821", "20140603t201129", "20140603t202947"]
lines = ['20140603_sbdr_masked_mosaic']
# 2019
#lines = ["20190629t203832", "20190629t210339"]
# pick the line
linei = lines[0]

# import 
# 2014
#t1 = rio.open("../data/AVng" + linei + "_corr_v1_rot")
t1 = rio.open("../data/AVng" + linei)
# 2019          
#t1 = rio.open("../data/ang" + linei + "_rfl_v2u1/ang" + linei + "_corr_v2u1_img")
t1c, t1c_trans = rio.mask.mask(t1, shp, crop = True)

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
# 2019
#with rio.open("../data/ang" + linei + "_rfl_v2u1/ang" + linei + "_corr_v2u1_img_clip.tif", 'w', **t1c_meta) as out:
#    out.write(t1c)

if fcrgb_flag == 1:
    # output easy rgb
    t1c_rgb = t1c[[57,41,22],:,:]
    t1c_fc = t1c[[62,92,256],:,:]
    # meta
    t1c_ql_meta = t1c_meta.copy()
    t1c_meta.update({'count': 3})
    # output
    with rio.open("../data/ang" + linei + "_rfl_v2u1/ang" + linei + "_corr_v2u1_img_clip_falsecolor.tif", 'w', **t1c_meta) as out:
        out.write(t1c_fc)
        
    with rio.open("../data/ang" + linei + "_rfl_v2u1/ang" + linei + "_corr_v2u1_img_clip_rgb.tif", 'w', **t1c_meta) as out:
        out.write(t1c_rgb)