#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:30:01 2021

@author: vegveg
"""

import os
import pandas as pd
import numpy as np
import rasterio as rio
from rasterio import features
import geopandas as gpd
import matplotlib.pyplot as plt

# =============================================================================
# functions
# =============================================================================
### model prep ###
def rasterize_training(wd, imgfn, trainingfn, classkeysfn):
    """Rasterizes training/test/valid data

    Parameters
    ----------
    wd : str
        Working directory
    imgfn : str
        Relative path to image
    trainingfn : str
        Relative path to training/test/valid vector file (geopackage, .gpkg)
    classkeysfn : str
        Relative path to classkeys comma separated file.

    Returns
    -------
    Rasterized training/test/validation data w/meta matching raw raster
    """
    os.chdir(wd)
    # open raster
    r = rio.open(imgfn)
    # import shapefile
    shp = gpd.read_file(trainingfn)
    
    # import classkeys
    classkeys = pd.read_csv(classkeysfn)
    
    # duplicate classnames column
    shp['classname'] = shp['class']
    
    # replace class name with id    
    for c in range(len(classkeys)):
        for i in range(len(shp)):
            if shp['class'][i] == classkeys.iloc[c, 1]: 
                shp['class'][i] = classkeys.iloc[c, 0]

    # create a generator of geom, value pairs to use in rasterizing
    shapes = ((geom, value) for geom, value in zip(shp['geometry'], shp['class']))
    
    # burn traning polygons into a new raster 
    pts_rast = features.rasterize(shapes, out_shape = r.shape, transform = r.transform)
    # set nodata
    pts_rast = pts_rast.astype('float32')
    pts_rast[pts_rast < 1] = np.nan
    # modify metadata
    meta = r.meta.copy()
    meta.update({'count': 1,
                 'dtype': 'float32',
                 'nodata': -999,
                 'width': pts_rast.shape[1],
                 'height': pts_rast.shape[0]})
    # output raster
    with rio.open(trainingfn[:-5], 'w+', **meta) as dst:
        dst.write(pts_rast[None,:,:])
        

def classwise_plots(wd, img, ttvimgfn, classkeysfn, classname, bblfn, band_status_column_name = 'status', bad_band_value = 0):
    """Plots all candidate spectra from the named class

    Parameters
    ----------
    wd : str
        working directory \n
    img : str
        relative path to image \n
    ttvimgfn : str
        relative path to image \n
    classkeysfn : str
        relative path to classkeys comma separated file. \n
    classname : str
        class to plot \n
    bblfn : srt
        relative path to bad bands list (.csv) \n
    band_status_column_name : str
        which column has the bb information (usually 1 and 0) \n
    bad_band_value : int
        what is the value that indicates a band is bad? (usually 0)
    
    Returns
    -------
    Plot of candidate spectra.

    """
    # set wd
    os.chdir(wd)
    
    # open raster
    r = rio.open(img).read()
    
    # import training labels
    t = rio.open(ttvimgfn).read()
    
    # import classkeys
    classkeys = pd.read_csv(classkeysfn)
    
    # convert class name to key
    try:
        classid = int(classkeys[classkeys['class'] == classname]['classid'])
    except:
        raise ValueError("classname not found in class keys")
    
    # import bblfn for wls
    wls = pd.read_csv(bblfn)
    wls = wls[wls[band_status_column_name] != bad_band_value]['wl']
    
    ### merge the rast and traning labels
    t = pd.DataFrame(t.reshape([t.shape[0], t.shape[1] * t.shape[2]])).T
    t.columns = ['labels']
    r = pd.DataFrame(r.reshape([r.shape[0], r.shape[1] * r.shape[2]])).T
    rt = pd.concat([t, r], axis = 1)
    del r, t
    
    # filter for classname
    rt_labeled = rt[rt['labels'] > 0]
    rt_labeled = rt_labeled[rt_labeled['labels'] == classid].T
    rt_labeled = rt_labeled.drop('labels')
    
    # plot each
    fig, ax = plt.subplots(1, 1)
    for c in range(rt_labeled.shape[1]):
        ax.plot(wls, rt_labeled.iloc[:,c])
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("Reflectance")
    ax.grid()
    ax.text(0.98, 0.98, classname, va = "top", ha = "right", fontsize = 14, transform = ax.transAxes)
    ax.tick_params(top = True, right = True)
    plt.show()
    
def check_bandimportance(wd, model, bblfn, band_status_column_name, bad_band_value = 0):
    '''Plots band importance. Note: HIGHLY recommended for hyperspectral or 
    aerial imagery to check for aerosol/atm influence. Spikes in FI are generally
    from spurious variance in atmospherically influenced bands.
    
    Parameters
    ----------
    wd : str
        Working directory
    model : RandomForestClassifier object
        Trained RF model
    bblfn : str
        Relative path to bad bands list (.csv)
    band_status_column_name : str
        Which column has the bad band information
    bad_band_value : int
        What is the value that indicates a band is bad? The default is 0.

    Returns
    -------
    Plots band-by-band Gini FI.
    
    '''
    os.chdir(wd)
    # import bbl for wavelength labels
    bbl = pd.read_csv(bblfn)
    bbl = bbl[bbl[band_status_column_name] != bad_band_value]
    # plot wavelength vs. FI 
    plt.scatter(bbl.iloc[:,0], model.feature_importances_, edgecolors = 'k')
    plt.xlabel("Wavelength")
    plt.ylabel("Feature Importance")
    plt.show()
