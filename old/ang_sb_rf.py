#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:57:04 2021

@author: mallen
"""

import os
from glob import glob
from datetime import datetime
import rasterio as rio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

os.chdir("/media/vegveg/bedlam/rf_sb_ang/code")
flag_CLASSPLOT = True
flag_OUTPUTRASTER = True

# =============================================================================
# import and merge, train/test split
# =============================================================================
# select aviris image
lines = ['20140603_sbdr_masked_mosaic']
linei = lines[0]
# open connection
img = rio.open("../data/AVng" + linei + "_clip.tif")

# list, select, and open rasterized training data
trainlist = glob("../data/AVng20140603_sbdr_masked_mosaic_clip_train_*")
trainfn = trainlist[0]
train = rio.open(trainfn)
# list classes
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

# read them into numpy arrays and reshape
# convert to pandas df b/c easier filtering
# cols are wl, rows are pixels
imgr = pd.DataFrame(img.read().reshape([img.count, img.height * img.width]).T)
tvr = pd.Series(train.read().reshape([img.height * img.width]).T, name = "class")

# import bad bands list (note: this is from the JPL generated header file)
bbl = pd.read_csv("../data/meta/bbl2014.csv")
bblidx = np.where(bbl == 0)[0]
# drop bad bands
imgrd = imgr.drop(bblidx, axis = 1)

# merge image with train/test labels
tv = pd.concat([imgrd, tvr], axis = 1)

# split off instances tagged as training/test data
tv_filt = tv[tv['class'] > 0]

# split train/test
X_train, X_test, y_train, y_test = train_test_split(tv_filt.iloc[:,:-1], tv_filt['class'])

# =============================================================================
# model setup, train, predict
# =============================================================================
# set model hyperparameters
rf = RandomForestClassifier(n_estimators = 200,
                            n_jobs = -1, 
                            verbose = 1, 
                            oob_score = True,
                            random_state = 1)
# train model 
rf.fit(X_train, y_train)

# =============================================================================
# assess performance
# =============================================================================
# in model performance
fi = rf.feature_importances_
plt.bar(bbl[bbl['status'] == 1]['wl'], fi, width = 5) # plot w/wavelengths
oob = rf.oob_score_

# cross valudation
cv = cross_val_score(rf, X_train, y_train, cv = 3)
print('three fold cross validation accuracy:' + str(cv))
# generally have decent accuracy, but not super useful because some of the
# classes are pretty skewed

# confusion matrix
y_train_pred = cross_val_predict(rf, X_train, y_train, cv = 5)
cm = confusion_matrix(y_train, y_train_pred)
recall = recall_score(y_train, y_train_pred, average = 'weighted')
precision = precision_score(y_train, y_train_pred, average = 'weighted')
f1 = f1_score(y_train, y_train_pred, average = 'weighted')

# independent test 
y_test_pred = rf.predict(X_test)
cm_test = confusion_matrix(y_test, y_test_pred)

# =============================================================================
# output products
# =============================================================================
# classify whole image
pred = rf.predict(imgrd)
# reshape
pred_img = pred.reshape([img.height, img.width])

# plot class output (note: matplotlib does a poor job of rendering these)
if flag_CLASSPLOT: 
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
if flag_OUTPUTRASTER:
    # record date/time
    t = datetime.now()
    t = t.strftime("%d_%m_%Y_%H_%M_%S")
    meta = img.meta.copy()
    meta.update({'count': 1,
                 'dtype': 'float64',
                 'nodata': 0,
                 'width': img.shape[1],
                 'height': img.shape[0]})
    with rio.open("../data/AVng" + linei + "_clip_predict_" + t + ".tif", 'w+', **meta) as dst:
        dst.write(pred_img[None,:,:])
        
