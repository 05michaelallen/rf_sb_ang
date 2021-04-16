#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 13:48:52 2021

@author: vegveg
"""

import rasterio as rio
import pandas as pd

# =============================================================================
# functions
# =============================================================================
def rm_bad_bands(wd, img, bad_bands_fn, band_status_column_name, bad_band_value):
    """reclassifies NA values in a raster
    
    inputs:
        wd: working directory
        img: relative path to image
        bad_bands_fn: relative path to bad bands list (.csv)
        band_status_column_name: which column has the bb information (usually 1 and 0)
        bad_band_value: what is the value that indicates a band is bad? (usually 0)
    
    result: outputs a raster with the bad bands clipped. retains the metadata.
    """
    
    # import raster, grab metadata
    rast = rio.open(wd + img)
    rast_descr = list(rast.descriptions)
    rast_rd = rast.read()
    meta = rast.meta.copy()
    
    # import bad bands list
    bbands = pd.read_csv(wd + bad_bands_fn)
    # filter based on bad band value
    bbands = bbands[bbands[band_status_column_name] != bad_band_value]
    bbands_idx = bbands.index.values.tolist()
    try:
        band_labs = [rast_descr[i] for i in bbands_idx]
    except:
        print("band descriptions not found")
    
    # remove bad bands from raster
    rast_rd_bbl = rast_rd[bbands_idx]
    del rast_rd
    
    # update metadata
    meta.update({'count': len(bbands_idx)})
    
    
    # output file
    with rio.open(wd + img + "_rmbadbands", 'w', **meta) as dst:
        for b in range(len(bbands_idx)):
            dst.set_band_description(b+1, band_labs[b])
        dst.write(rast_rd_bbl)

# =============================================================================
# manual run
# =============================================================================
# set params
#directory = '/home/vegveg/rf_sb_ang'
#img = '/data/AVng20140603_sbdr_masked_mosaic_reclass'
#bad_bands_fn = '/data/meta/bbl2014.csv'
#band_status_column_name = 'status'
#bad_band_value = 0

# run it
#rm_bad_bands(directory, img, bad_bands_fn, band_status_column_name, bad_band_value)