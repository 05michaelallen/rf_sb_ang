#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:39:00 2021

@author: mallen
"""

import os
import rasterio as rio
from rasterio import mask 
from rasterio import features
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime


os.chdir("/Volumes/ellwood/rf_sb_ang/code")
flag_DIAG_PLOT = True
# =============================================================================
# 
# =============================================================================
# training data was generated manually in QGIS - I use polygons (approx. 10x10 each)
# classes have approximately equal traning data, but there is some underrepresentation 
# for rare classes (e.g. astroturf)

# select image
lines = ['20140603_sbdr_masked_mosaic']
linei = lines[0]

# import class key
ckey = pd.read_csv("../data/meta/classids_2014_v1.csv")

# import shapefile
pts = gpd.read_file("../data/train/2014_train_poly_v1.shp")

# because we're rasterizing the training data (timesink now, but much faster later on)
# we need to match classnames to integer class labels
# rename classid as classname - this is need
pts['classname'] = pts['classid']

# replace class name with id
for c in range(len(ckey)):
    for i in range(len(pts)):
        if pts['classid'][i] == ckey.iloc[c, 1]: 
            pts['classid'][i] = ckey.iloc[c, 0]

# create a generator of geom, value pairs to use in rasterizing
shapes = ((geom, value) for geom, value in zip(pts.geometry, pts.classid))

# import image
img = rio.open("../data/AVng" + linei + "_clip.tif")

# burn traning polygons into a new raster 
pts_rast = features.rasterize(shapes, 
                              out_shape = img.shape, 
                              transform = img.transform)
# set nodata
# note: must convert to float (instead of uint)
pts_rast = pts_rast.astype('float64')
pts_rast[pts_rast < 1] = np.nan

# plot 
if flag_DIAG_PLOT:
    p = plt.imshow(pts_rast)
    plt.colorbar(p)
    np.unique(pts_rast, return_counts = True)

# output
# record date/time
t = datetime.now()
t = t.strftime("%d_%m_%Y_%H_%M_%S")
meta = img.meta.copy()
meta.update({'count': 1,
             'dtype': 'float64',
             'nodata': -999,
             'width': pts_rast.shape[1],
             'height': pts_rast.shape[0]})
with rio.open("../data/AVng" + linei + "_clip_train_" + t + ".tif", 'w+', **meta) as dst:
    dst.write(pts_rast[None,:,:])