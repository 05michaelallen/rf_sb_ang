#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 13:48:52 2021

@author: vegveg
"""

import rasterio as rio
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

# =============================================================================
# functions
# =============================================================================
def rm_bad_bands(directory, img, bad_bands_fn, band_status_column_name, bad_band_value):
    # import raster, grab metadata
    rast = rio.open(directory + img)
    rast_descr = list(rast.descriptions)
    rast_rd = rast.read()
    meta = rast.meta.copy()
    
    # import bad bands list
    bbands = pd.read_csv(directory + bad_bands_fn)
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
    with rio.open(directory + img + "_rmbadbands", 'w', **meta) as dst:
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