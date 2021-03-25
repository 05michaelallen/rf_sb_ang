#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 11:55:00 2021

@author: vegveg
"""

import rasterio as rio
import rasterio.mask
import pandas as pd
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
    Writes a clipped raster in ENVI format with correct metadata. Suffix = _clip

    """
    
    # set wd and open image connection
    os.chdir(wd)
    r = rio.open(img)
    meta = r.meta.copy() # grab metadata
    
    # get band descriptions
    r_descr = list(r.descriptions)
    
    # open shapefile
    shp = gpd.read_file(shpfn)
    
    # clip 
    rc, rc_trans = rio.mask.mask(r, shp[['geometry']].values.flatten(), nodata = meta['nodata'], crop = True)
    
    # change metadata
    meta.update({'height': rc.shape[1], 
                 'width': rc.shape[2],
                 'transform': rc_trans})
    # output
    with rio.open(img + "_clip", 'w', **meta) as dst:
        for b in range(r.count):
            dst.set_band_description(b+1, r_descr[b])
        dst.write(rc)
      


def rm_bad_bands(wd, img, bblfn, band_status_column_name, bad_band_value):
    """Reclassifies NA values in a raster
    
    Parameters
    ----------
    wd : str
        working directory
    img : str
        relative path to image
    bblfn : str
        relative path to bad bands list (.csv)
    band_status_column_name : str
        which column has the bb information (usually 1 and 0)
    bad_band_value : int
        what is the value that indicates a band is bad? (usually 0)
    
    Returns
    -------
    Writes an ENVI raster with the bad bands clipped. retains the metadata.
    
    """
    
    # import raster, grab metadata
    os.chdir(wd)
    rast = rio.open(img)
    rast_descr = list(rast.descriptions)
    rast_rd = rast.read()
    meta = rast.meta.copy()
    
    # import bad bands list
    bbands = pd.read_csv(bblfn)
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
    with rio.open(img + "_rmbadbands", 'w', **meta) as dst:
        for b in range(len(bbands_idx)):
            dst.set_band_description(b+1, band_labs[b])
        dst.write(rast_rd_bbl)



def reclassify_NAs(wd, img, old_na, new_na):
    """Reclassifies NA values in a raster
    
    Parameters
    ----------
    wd : str
        working directory
    img : str
        relative path to image
    old_na : int
        old na value
    new_na : int
        new/target na value
    
    Returns
    -------
    Outputs an ENVI raster with appropriate metadata.
    
    """
    os.chdir(wd)
    r = rio.open(img)
    meta = r.meta.copy()
    rr = r.read()
    # get descriptions
    r_descr = list(r.descriptions)
    
    # reclassify
    rr[rr == old_na] = new_na
    
    # change metadata
    meta.update({'nodata': new_na})
    # output
    with rio.open(img + "_reclass", 'w', **meta) as dst:
        for b in range(r.count):
            dst.set_band_description(b+1, r_descr[b])
        dst.write(rr)



def import_reshape(wd, img):  
    """Imports and reshapes a multiband raster
    
    Parameters
    ----------
    wd : str
        working directory
    img: str
        relative path to image 
        
    Returns
    ------- 
    Numpy array with rows = pixels, cols = bands/features.
    Metadata
    Band descriptions
    
    """
    # open connection, grab metadata
    os.chdir(wd)
    rast = rio.open(img)
    meta = rast.meta.copy()
    desc = list(rast.descriptions)
    
    # read and reshape
    rastr = rast.read()
    rastrr = rastr.reshape([rast.count, rast.height * rast.width]).T 
    
    # return metadata (for output) and reshaped raster
    return rastrr, meta, desc
