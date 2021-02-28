#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:33:08 2021

@author: vegveg
"""

import os
import rasterio as rio
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

### set wd
wd = "/home/vegveg/rf_sb_ang/code/"
os.chdir(wd)

# from other scripts
from import_reshape import import_reshape

# =============================================================================
# main script
# =============================================================================
# import and reshape image
imagefn = "../data/AVng20140603_sbdr_masked_mosaic_reclass_rmbadbands_sbmetro.tif"
img, meta = import_reshape(wd, imagefn)

### processing 
# apply nodata
img[img == meta['nodata']] = np.nan
# reclassify erroneous values
img[img > 1] = 1

# convert to df
# drop masked pixels
imgdf = pd.DataFrame(img)
imgdf_clean = imgdf.dropna()
# get indices for output reshaping
img_ids = imgdf_clean.index

# training sample
# sample % of non-NA data 
imgdf_train = imgdf_clean.sample(frac = 0.01)

# n_clusters
n_clusters = np.arange(3, 12, 1)
### loop over the cluster sizes we want to test
for n in n_clusters:
    # create model and fit based on n-clusters
    kmodel = KMeans(n_clusters = n)
    kmodel.fit(imgdf_train)
    # predict over full numpy array
    pred = kmodel.predict(imgdf_clean)
    
    ### post-processing
    # merge back with the original dataset labels
    pred_df = pd.DataFrame(pred, index = img_ids, columns = ['pred'])
    # fill in un-labeled pixels (so we can create an image)
    pred_df_img = pd.concat([imgdf.iloc[:,0], pred_df], axis = 1).drop(0, axis = 1)
    # reshape output 
    pred_df_imgr = np.reshape(np.array(pred_df_img), [meta['height'], meta['width']])
    
    ### output file
    # check if output dir exists
    if not os.path.exists("../data/kmeans/"):
        os.makedirs("../data/kmeans")
    meta_out = meta.copy()
    meta_out.update({'count': 1})
    with rio.open("../data/kmeans/" + str(n) + "_cluster_kmean.tif", 'w', **meta_out) as dst:
        dst.write(pred_df_imgr[None,:,:].astype(np.float32))