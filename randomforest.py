#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:07:16 2021

@author: vegveg
"""

import os
import pandas as pd
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# functions
# =============================================================================


# =============================================================================
# import and pre-process data
# =============================================================================
# set up filepaths
wd = "/home/vegveg/rf_sb_ang/code/"
img = "../data/AVng20140603_sbdr_masked_mosaic_reclass_rmbadbands_clip"
bblfn = "../data/meta/bbl2014_conservative.csv"
ttvfn = "../data/train/try2_03242021"
classkeysfn = "../data/train/try2_03242021_classkeys.txt"
outfn = "../data/rf/try2_03242021"
boundaryfn = "../data/shp/david_ch2_sbmetroclips/sbmetro_extent_dlm_ch2/santa_barbara_ca_urbanized_area_utmz11_avngsub3_wgs84.shp"

# set wd
os.chdir(wd)
# from other scripts 
from pre_processing_scripts import import_reshape, rm_bad_bands, reclassify_NAs, clip
from model_prep_scripts import rasterize_ttv

### pre-processing (note: only need to run these if needed)
#reclassify_NAs(wd, img, 0, -999)
#rm_bad_bands(wd, img, bblfn, 'status', 0)
#clip(wd, img, boundaryfn)
#rasterize_ttv(wd, img, ttvfn + ".gpkg", classkeysfn)


# import and reshape the raster and training data
r, rmet, rdes = import_reshape(wd, img)
t, tmet, tdes = import_reshape(wd, ttvfn)

# clean up ang data
r[r < -0.1] = np.nan
r[(r >= -0.1) & (r < 0)] = 0

# import class keys
classkeys = pd.read_csv(classkeysfn)
# import bbl for wl labels
bbl = pd.read_csv(bblfn)
bbl = bbl[bbl['status'] == 1]

# convert to df and merge bands and labels
# note this is memory intensive, swap to numpy instead
r = pd.DataFrame(r, columns = rdes)
t = pd.DataFrame(t, columns = ['labels'])
rt = pd.concat([r, t], axis = 1)
del r, t

# grab labeled pixels
rt_labeled = rt[rt['labels'] > 0]

# =============================================================================
# train/test, tune hyperparameters
# =============================================================================
# split into train/test
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(rt_labeled.iloc[:,:-1], rt_labeled['labels'])

# set up cv-folds

# set hyperparameters
rf = RandomForestClassifier(n_estimators = 500,
                            n_jobs = -1,
                            verbose = 1, 
                            oob_score = True)



# =============================================================================
# run full model with good parameters
# =============================================================================
# train model on full training set
rf.fit(X_train, y_train)

# test parameters
rf.oob_score_
fi = rf.feature_importances_
# plot fi as a function of wl
plt.scatter(bbl['wl'], fi)
plt.scatter(bbl['wl'], rt_labeled.iloc[100,:-1])


# =============================================================================
# use test set to evaluate generalizability/fit
# =============================================================================
rf.fit(X_test, y_test)
rf.oob_score_


# =============================================================================
# predict on entire image, post-processing, output
# =============================================================================
### first process the image 
# take off labels, drop masked pixels
rt_final = rt.iloc[:,:-1].dropna()

### set up, train, and predict using selected hyperparameters
rf_final = RandomForestClassifier(n_estimators = 500,
                                  n_jobs = -1,
                                  verbose = 1, 
                                  oob_score = True)

rf_final.fit(rt_labeled.iloc[:,:-1], rt_labeled['labels'])
predicted = rf_final.predict(rt_final)
rf_final.oob_score_
### process output
# convert to df with indices from the nadrop dataset
predicted = pd.DataFrame(predicted, 
                         index = rt_final.index, 
                         columns = ['class'])
# merge with shape of input image (with masked pixels), drop helper column
predicted = pd.concat([predicted, rt.iloc[:,0]], axis = 1)['class']
# reshape into an image
predicted_image = np.reshape(np.array(predicted), 
                             (rmet['height'], rmet['width']))

### ouput
with rio.open(wd + outfn, 'w', **tmet) as dst:
    dst.write(predicted_image[None,:,:])

