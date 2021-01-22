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


os.chdir("/Volumes/ellwood/ang_sb_portfolio/code")
# =============================================================================
# 
# =============================================================================
# select image
lines = ['20140603_sbdr_masked_mosaic']
linei = lines[0]

# import class key
ckey = pd.read_csv("../data/classids_2014_v1.csv")

# import shapefile
pts = gpd.read_file("../data/train/2014_train_poly_v1.shp")
# redo class to classid
pts['classname'] = pts['classid']

# replace class name with id
for c in range(len(ckey)):
    for i in range(len(pts)):
        if pts['classid'][i] == ckey.iloc[c, 1]: 
            pts['classid'][i] = ckey.iloc[c, 0]

# reate a generator of geom, value pairs to use in rasterizing
shapes = ((geom, value) for geom, value in zip(pts.geometry, pts.classid))

# import image
img = rio.open("../data/AVng" + linei + "_clip.tif")

# burn features into a raster of the same size
pts_rast = features.rasterize(shapes, out_shape = img.shape, transform = img.transform)

# plot 
p = plt.imshow(pts_rast)
plt.colorbar(p)
np.unique(pts_rast, return_counts = True)

# output
meta = img.meta.copy()
meta.update({'count': 1,
             'dtype': 'uint8',
             'nodata': 0,
             'width': pts_rast.shape[1],
             'height': pts_rast.shape[0]})
with rio.open("../data/AVng" + linei + "train_v1.tif", 'w+', **meta) as dst:
    dst.write(pts_rast[None,:,:])