#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:57:04 2021

@author: mallen
"""

import os
import rasterio as rio
from rasterio import mask 
from rasterio import features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

os.chdir("/Volumes/ellwood/rf_sb_ang/code")

# =============================================================================
# 
# =============================================================================
# select image
lines = ['20140603_sbdr_masked_mosaic']
linei = lines[0]

# imgs
img = rio.open("../data/AVng" + linei + "_clip.tif")
tv = rio.open("../data/AVng" + linei + "train_v1.tif")
classes = ['Water', 
           'Sand', 
           'Bare Soil',
           'NPV',
           'Road',
           'Roof',
           'Red Tile Roof',
           'Astroturf',
           'Submurged Vegetation',
           'Tree',
           'Turfgrass']

# read them and reshape
# convert to pandas df b/c easier filtering
# cols are wl, rows are pixels
imgr = pd.DataFrame(img.read().reshape([img.count, img.height * img.width]).T)
tvr = pd.Series(tv.read().reshape([img.height * img.width]).T, name = "class")

# get metadata
# list from header file
bbl = pd.read_csv("../data/meta/bbl2014.csv")
bblidx = np.where(bbl == 0)[0]
imgrd = imgr.drop(bblidx, axis = 1)

# merge with t/v
itv = pd.concat([imgrd, tvr], axis = 1)
# filter
itv_filt = itv[itv['class'] > 0]
# split t/v
it, iv = train_test_split(itv_filt)
train = it.iloc[:,:it.shape[1]-1]

# =============================================================================
# 
# =============================================================================
rf = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 1, oob_score = True, random_state = 1)
rf.fit(it.iloc[:,:it.shape[1]-1], it['class'])

pred = rf.predict(itv.iloc[:,:itv.shape[1]-1])

# feature importanced
fscores = rf.feature_importances_
oob = rf.oob_score_

# reshape
pred_img = pred.reshape([img.height, img.width])

# plot
fig, ax = plt.subplots(1, 1)

# set up colormap
colors = ['dodgerblue', 
          'tan', 
          'saddlebrown', 
          'red', 
          '0.8', 
          '0.4', 
          'indianred', 
          'cyan', 
          'teal', 
          'forestgreen',
          'limegreen']
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors, 11)

p = plt.imshow(np.rot90(pred_img, 0)-0.5, vmin = 0, vmax = 11, cmap = cmap)
cbar = plt.colorbar(p, 
                    ax = ax, 
                    ticks = np.arange(0.5, 11.5, 1), 
                    shrink = 0.5)
cbar.ax.set_yticklabels(classes)
plt.savefig("../plots/test.png", dpi = 1000)

# output
meta = img.meta.copy()
meta.update({'count': 1,
             'dtype': 'uint8',
             'nodata': 0,
             'width': img.shape[1],
             'height': img.shape[0]})
with rio.open("../data/AVng" + linei + "_clip_predict_v1.tif", 'w+', **meta) as dst:
    dst.write(pred_img[None,:,:])
    
# =============================================================================
# 
# =============================================================================
# indices of validation pixels
valid = iv['class']
pred_valid = pred[np.isin(pred.index]