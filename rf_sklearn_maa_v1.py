# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

### load packages and set wd
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.colors

os.chdir("/Users/mallen/Documents/q9/277/colleen_rf/code")
seed = 277


# =============================================================================
# load data, clean and split into train and valid
# =============================================================================
# load data as a csv
feat = pd.read_csv("../data/dat.csv")

### create set for traning the model 
# remove geotags
# remove bad bands
# apply the >50% GV mask
# bbl is from enbo
featr = feat.iloc[:,3:]
featrbbl = featr.iloc[:, list(range(4, 59))+
                 list(range(62, 79))+
                 list(range(80, 104))+
                 list(range(119, 150))+
                 list(range(173, 218))+
                 list(range(224, 227))]
featrbblv = featrbbl[featrbbl['mask'] == 1]
# save index for later
featrbblv_index = featrbblv.iloc[:,0:172].dropna().index


# split into train and valid, take only reflectance bands
# get row labels for train
train = featrbblv[featrbblv['training'] > -1].iloc[:, 
                                                   list(range(0, 172))+
                                                   list(range(173, 174))].dropna()
train_labels = np.array(train['training'])

# do the same for validation
valid = featrbblv[featrbblv['validation'] > -1].iloc[:, 
                                                     list(range(0, 172))+
                                                     list(range(174, 175))].dropna()
valid_labels = np.array(valid['validation'])


# =============================================================================
# train and predict model
# =============================================================================
# set up model parameters
model = RandomForestClassifier(n_estimators = 5000,
                               random_state = seed, 
                               max_features = 'sqrt',
                               n_jobs = -1, 
                               verbose = 1)

# fit on training data
model.fit(np.array(train.iloc[:,0:172]), train_labels)

# predict from model fit
pred = model.predict(np.array(featrbblv.iloc[:,0:172].dropna()))


# =============================================================================
# merge back with original dataset and plot classification
# =============================================================================
# move to df, with indices from filtered df
preddf = pd.DataFrame(pred, index = featrbblv_index, columns = ["predicted"])
# output to csv
preddf.to_csv("../data/dat_predicted.csv")

# concatenate with feat
featpred = pd.concat([feat, preddf], axis = 1)

# back to numpy array for conversion to raster
featpred_arr = np.array(featpred.predicted)

# reshape to dimensions
featpred_arr_reshape = np.reshape(featpred_arr, (2143, 1214))

### plot using imshow
# set up colormap
colors = ['b', 
          'c', 
          'r', 
          'm', 
          'g', 
          'y', 
          'dodgerblue', 
          'indianred', 
          '0.5', 
          '0',
          'saddlebrown']
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors, 11)

fig, ax = plt.subplots(1, 1, figsize = [5, 5])

img = ax.imshow(featpred_arr_reshape - 0.5, # this centers each class in the colorbar
                vmin = 0,
                vmax = 11,
                cmap = cmap)
cbar = plt.colorbar(img, 
                    ax = ax, 
                    ticks = np.arange(0.5, 11.5, 1), 
                    shrink = 0.5)
cbar.ax.set_yticklabels(['alfalfa', 
                         'almond', 
                         'corn', # NOTE: CORN DOES NOT EXIST IN THE TRAIN/VAL DATA
                         'cotton', 
                         'deciduous', 
                         'grain', 
                         'subtropical',
                         'tomato',
                         'trunk',
                         'uncultivated',
                         'vine'])
plt.tight_layout()
plt.savefig("../plots/rf_sklearn_maa_outputclass_v2.png", dpi = 600)


# =============================================================================
# accuracy assessment
# =============================================================================
# filter predicted results for validation pixels
validtest = featpred[["validation", "predicted"]]
# drop non validation and nan predictions (bad input data)
validtestf = validtest[validtest['validation'] > -1].dropna()

### generate confusion matrix
cm = confusion_matrix(np.array(validtestf['validation']), 
                      np.array(validtestf['predicted']))

### calculate accuracies
cmpx = np.sum(cm, axis = 0)

# overall accuracy
oacc = 0
for i in range(0, 10):
    oacc = oacc + cm[i, i]
oacc = oacc/cmpx.sum()

# producers accuracy 
pacc = []
for i in range(0, 10):
    pacc.append(cm[i, i] / cmpx[i])
pacc = np.asarray(pacc)

# users accuracy 
uacc = []
for i in range(0, 10):
    uacc.append(cm[i, i] / np.sum(cm[i,:]))
uacc = np.append(uacc, oacc)
uacc = np.asarray([uacc])

# stack all of the accuracies together to fill out the confusion matrix
cm_acc = np.vstack([cm, pacc])
cm_acc = np.hstack([cm_acc, np.transpose(uacc)])

# list of classes in cm
classes = np.array((['alfalfa', 
                     'almond', 
                     #'corn', # NOTE: CORN DOES NOT EXIST IN THE TRAIN/VAL DATA
                     'cotton', 
                     'deciduous', 
                     'grain', 
                     'subtropical',
                     'tomato',
                     'trunk',
                     'uncultivated',
                     'vine',
                     "producers acc"]))

# to df
cm_accdf = pd.DataFrame(cm_acc, columns = classes)

# save to csv
cm_accdf.to_csv("../data/rf_sklearn_maa_confusionmatrix_v2.csv")
