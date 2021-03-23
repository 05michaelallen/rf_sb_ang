#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:21:55 2021

@author: vegveg
"""

import rasterio as rio

# =============================================================================
# functions
# =============================================================================
def import_reshape(wd, img):  
    """imports and reshapes the multiband raster
    
    inputs: 
        wd: working directory
        img: relative path to image 
        
    output: numpy array with rows = pixels, cols = bands/features
    """
    # open connection, grab metadata
    rast = rio.open(wd + img)
    meta = rast.meta.copy()
    
    # read and reshape
    rastr = rast.read()
    rastrr = rastr.reshape([rast.count, rast.height * rast.width]).T 
    
    # return metadata (for output) and reshaped raster
    return rastrr, meta

# =============================================================================
# manual run
# =============================================================================
# set params
#wd = '/home/vegveg/rf_sb_ang'
#img = '/data/AVng20140603_sbdr_masked_mosaic_rmbadbands_sbmetro.tif'

# run it
#rast, meta = import_reshape(wd, img)
