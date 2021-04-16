#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 13:59:11 2021

@author: vegveg
"""

import rasterio as rio
import os

# =============================================================================
# functions
# =============================================================================
def reclassify_NAs(wd, img, old_na, new_na):
    """reclassifies NA values in a raster
    
    inputs:
        wd: working directory
        img: relative path to image
        old_na: old na value
        new_na: new/target na value
    
    result: outputs an ENVI raster with appropriate metadata
    """
    os.chdir(wd)
    r = rio.open(wd + img)
    meta = r.meta.copy()
    rr = r.read()
    # get descriptions
    r_descr = list(r.descriptions)
    
    # reclassify
    rr[rr == old_na] = new_na
    
    # change metadata
    meta.update({'nodata': new_na})
    # output
    with rio.open(wd + img + "_reclass", 'w', **meta) as dst:
        for b in range(r.count):
            dst.set_band_description(b+1, r_descr[b])
        dst.write(rr)

# =============================================================================
# manual run
# =============================================================================
# set wd and parameters
#wd = "/home/vegveg/rf_sb_ang"
#img = "/data/AVng20140603_sbdr_masked_mosaic"
#old_na = 0
#new_na = -999

#reclassify_NAs(wd, img, old_na, new_na)