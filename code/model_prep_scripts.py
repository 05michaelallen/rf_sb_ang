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
def rasterize_ttv(wd, img, ttvfn, classkeysfn):
    """Rasterizes traning/test/valid data using the raw raster

    Parameters
    ----------
    wd : str
        working directory
    ttvfn : str
        relative path to ttv geopackage
    classkeysfn : str
        relative path to classkeys comma separated file.

    Returns
    -------
    Rasterized training/test/validation data w/meta matching raw raster
    """
    # set wd
    os.chdir(wd)
    
    # open raster
    r = rio.open(img)
    
    # import shapefile
    shp = gpd.read_file(ttvfn)
    
    # import classkeys
    classkeys = pd.read_csv(classkeysfn)
    
    # because we're rasterizing the training data (timesink now, but much faster later on)
    # we need to match classnames to integer class labels
    # duplicate classids
    shp['classname'] = shp['class']
    
    # replace class name with id    
    for c in range(len(classkeys)):
        for i in range(len(shp)):
            if shp['class'][i] == classkeys.iloc[c, 1]: 
                shp['class'][i] = classkeys.iloc[c, 0]

    # create a generator of geom, value pairs to use in rasterizing
    shapes = ((geom, value) for geom, value in zip(shp['geometry'], shp['class']))
    
    # burn traning polygons into a new raster 
    pts_rast = features.rasterize(shapes, 
                                  out_shape = r.shape, 
                                  transform = r.transform)
    # set nodata
    # note: must convert to float (instead of uint)
    pts_rast = pts_rast.astype('float32')
    pts_rast[pts_rast < 1] = np.nan
    meta = r.meta.copy()
    meta.update({'count': 1,
                 'dtype': 'float32',
                 'nodata': -999,
                 'width': pts_rast.shape[1],
                 'height': pts_rast.shape[0]})

    with rio.open(ttvfn[:-5], 'w+', **meta) as dst:
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
    
    
