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
import rasterstats as rs
import geopandas as gpd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from datetime import datetime


# =============================================================================
# USER DEFINED PARAMETERS
# =============================================================================
# set up filepaths
wd = "/home/vegveg/rf_sb_ang/code/code/"
imgfn = "../data/AVng20140603_sbdr_masked_mosaic"
bblfn = "../data/meta/bbl2014_conservative.csv"
trainingfn = "../data/train/try2_03242021"
classkeysfn = "../data/train/try2_03242021_classkeys.txt"
outfn = "../data/rf/try2_03242021"
boundaryfn = "../data/shp/david_ch2_sbmetroclips/sbmetro_extent_dlm_ch2/santa_barbara_ca_urbanized_area_utmz11_avngsub3_wgs84.shp"

### flags for different running modes (e.g., preprocessing, testing/production)
FLAG_preprocessing = False
FLAG_trainingspectralplots = False
FLAG_hyperparametertuning = True
FLAG_plotbandimportance = True

# type of training vectors, can be "point" or "polygon". 
# "polygon" generates an additional polygon-scale confusion matrix
training_type = 'polygon'

### set hyperparameters or gridsearch candidates
param_grid = [{'n_estimators': [50, 100, 200], 
               'max_features': [None, 'auto', 'log2']}]
# **or** manually define RF hyperparameters
params = {'n_estimators': 500, "max_features": 'auto'}


# =============================================================================
# ENTER MAIN: PRE-PROCESSING
# =============================================================================
print(datetime.now())
print("Enter preprocessing.")

# set wd and import helper functions from other scripts 
os.chdir(wd)
from pre_processing_scripts import import_reshape, rm_bad_bands, reclassify_NAs, clip_scene
from model_prep_scripts import rasterize_training, classwise_plots, check_bandimportance

### run scene preprocessing helper functions
if FLAG_preprocessing:
    reclassify_NAs(wd, imgfn, 0)
    rm_bad_bands(wd, imgfn, bblfn, 'status', 0)
    clip_scene(wd, imgfn, boundaryfn)
else:
    print("FLAG_preprocessing == False, running on preprocessed data.")

### run training/test polygon/point vector preprocessing
# this splits traning data at the plot level to avoid contamination 
# from the same polygons in training and test.
shp = gpd.read_file(trainingfn + ".gpkg") # load data
classes = np.unique(shp['class']).tolist() # list classes
training = []
testing = []
# manual random 80/20 split object-wise
for c in classes:
    shpc = shp[shp['class'] == c]
    trainingc = shpc.sample(frac = 0.8)
    testingc = shpc.loc[~shpc.index.isin(trainingc.index)]
    # append to new split lists
    training.append(trainingc)
    testing.append(testingc)
    del shpc, trainingc, testingc # cleanup
    
# concat back into a dataframe and output split train/test vectors
training = pd.concat(training).to_file(trainingfn + "_training.gpkg", driver = "GPKG")
testing = pd.concat(testing).to_file(trainingfn + "_testing.gpkg", driver = "GPKG")

# import and reshape the pre-processed scene and rasterized training data
r, rmet, rdes = import_reshape(wd, imgfn)

# rasterize training and test using reflectance image to burn
rasterize_training(wd, imgfn, trainingfn + "_training.gpkg", classkeysfn)
rasterize_training(wd, imgfn, trainingfn + "_testing.gpkg", classkeysfn)

# import and reshape train/test
t, tmet, tdes = import_reshape(wd, trainingfn + "_training")
v, vmet, vdes = import_reshape(wd, trainingfn + "_testing")

# merge band features and labels
r = pd.DataFrame(r, columns = rdes)
t = pd.DataFrame(t, columns = ['traininglabels'])
v = pd.DataFrame(v, columns = ['testlabels'])
rtv = pd.concat([r, t, v], axis = 1)
del r, t, v

# generate model inputs
train = rtv[rtv['traininglabels'] > 0]
X_train = train.iloc[:,:-2]
y_train = train['traininglabels']
test = rtv[rtv['testlabels'] > 0]
X_test = test.iloc[:,:-2]
y_test = test['testlabels']
del train, test

### plot classwise spectra
if FLAG_trainingspectralplots:
    classkeys = pd.read_csv(classkeysfn)
    for c in classkeys['class']:
        classwise_plots(wd, imgfn, trainingfn + "_training", classkeysfn, c, bblfn)

print(datetime.now())
print("Images and labels imported, training/test split done, moving to hyperparameter turning.")

# =============================================================================
# TUNE MODEL HYPERPARAMETERS, GENERATE EVALUATION METRICS
# =============================================================================
### tune/set hyperparameters
if FLAG_hyperparametertuning: 
    # loop through list of dicts w/hyperparameters to find optimal params
    # generally for rf we add trees until we get convergence unless features are 
    # messy or we detect overfitting
    gs = GridSearchCV(RandomForestClassifier(), param_grid, cv = 4)
    gs.fit(X_train, y_train)
    print("Optimal hyperparameters from gridsearch: " + str(gs.best_params_)) 
    # set up model w/ best params identified in gridsearch
    rf_tuned = RandomForestClassifier(**gs.best_params_, n_jobs = -1, oob_score = True)
else:
    print("Running with pre-set hyperparameters: " + str(params))
    rf_tuned = RandomForestClassifier(**params, n_jobs = -1, oob_score = True)
    
### test feature/internal/external validity
# train model on full training set
rf_tuned.fit(X_train, y_train)

# calculate oob and cross validation scores on train
print("Training OOB score: ", str(rf_tuned.oob_score_))
kcval = cross_val_score(rf_tuned, X_train, y_train, cv = 4)
print("k-fold Cross Val scores:", str(kcval))

# test feature validity for aerosol/water influence
if FLAG_plotbandimportance:
    check_bandimportance(wd, rf_tuned, bblfn, 'status')

# pixel-wise evaluation using test set
y_pred = rf_tuned.predict(X_test)
# generate pixel-wise confusion matrix 
cm_pixel_test = confusion_matrix(y_test, y_pred)
    
print(datetime.now())
print("Continuing to final prediction. Make sure model params and internal/external validity look reasonable.")


# =============================================================================
# PREDICT ON SCENE USING TUNED MODEL, POLYGON-SCALE ASSESSMENT, POST-PROCESSING
# =============================================================================
### process output
# take full set, drop labels and masked pixels
rt_final = rtv.iloc[:,:-2].dropna()
predicted = rf_tuned.predict(rt_final)

# convert to df with indices from the nadrop dataset
predicted_df = pd.DataFrame(predicted, index = rt_final.index, columns = ['class'])

# merge with shape of input image (with masked pixels), drop helper column
predicted_df = pd.concat([predicted_df, rtv.iloc[:,0]], axis = 1)['class']

# reshape into an image using height and width from metadata
predicted_image = np.reshape(np.array(predicted_df), (rmet['height'], rmet['width']))

# polygon (e.g., field, neighborhood) scale assessment
if training_type == "polygon":
    # load test polygons
    testpoly = gpd.read_file(trainingfn + "_testing.gpkg")
    # compute zonal majorioty 
    y_pred_poly = rs.zonal_stats(testpoly, 
                                 predicted_image,
                                 affine = rmet['transform'],
                                 stats = ["majority"])
    y_pred_poly = pd.DataFrame(y_pred_poly)

    # replace class name with id
    classkeys = pd.read_csv(classkeysfn)
    for c in range(len(classkeys)):
        for i in range(len(testpoly)):
            if testpoly['class'][i] == classkeys.iloc[c, 1]: 
                testpoly['class'][i] = classkeys.iloc[c, 0]

    # merge into confusion matrix
    cm_poly_test = confusion_matrix(pd.DataFrame(testpoly['class']).astype(float), y_pred_poly)
elif training_type == "point":
    pass
else:
    raise ValueError("Training_type is invalid, must be 'polygon' or 'point'.")  

# ouput predicted image
with rio.open(wd + outfn, 'w', **tmet) as dst:
    dst.write(predicted_image[None,:,:])