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

# =============================================================================
# functions
# =============================================================================
def rasterize_ttv(wd, img, ttvfn, classkeysfn):
    """Rasterizes traning/test/valid data using the raw raster

    Parameters
    ----------
    wd : str
        working directory \n
    ttvfn : str
        relative path to image \n
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
        
        