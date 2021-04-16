#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 11:38:00 2021

@author: vegveg
"""

import rasterio as rio
import rasterio.mask
import geopandas as gpd
import os

# =============================================================================
# functions
# =============================================================================
def clip(wd, img, shpfn):
    """Clips a multiband raster to a shapefile. Retains metadata. 

    Parameters
    ----------
    wd : str
        working directory \n
    img : str
        relative path to image \n
    shpfn : str
        relative path to shapefile

    Returns
    -------
    Generates a clipped raster in ENVI format with correct metadata

    """
    
    # set wd and open image connection
    os.chdir(wd)
    r = rio.open(wd + img)
    meta = r.meta.copy() # grab metadata
    
    # get band descriptions
    r_descr = list(r.descriptions)
    
    # open shapefile
    shp = gpd.read_file(wd + shpfn)
    
    # clip 
    rc, rc_trans = rio.mask.mask(r, shp[['geometry']].values.flatten(), nodata = meta['nodata'], crop = True)
    
    # change metadata
    meta.update({'height': rc.shape[1], 
                 'width': rc.shape[2],
                 'transform': rc_trans})
    # output
    with rio.open(wd + img + "_clip", 'w', **meta) as dst:
        for b in range(r.count):
            dst.set_band_description(b+1, r_descr[b])
        dst.write(rc)

# =============================================================================
# manual run
# =============================================================================
# set wd and parameters
wd = "/home/vegveg/rf_sb_ang"
img = "/data/AVng20140603_sbdr_masked_mosaic_reclass_rmbadbands"
shpfn = "/data/shp/david_ch2_sbmetroclips/sbmetro_extent_dlm_ch2/santa_barbara_ca_urbanized_area_utmz11_avngsub3_wgs84.shp"