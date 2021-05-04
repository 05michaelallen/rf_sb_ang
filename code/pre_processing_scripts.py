#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 11:55:00 2021

@author: vegveg
"""

import rasterio as rio
import rasterio.mask as mask
import pandas as pd
import geopandas as gpd
import os

# =============================================================================
# functions
# =============================================================================
### preprocessing helper functions ###
def reclassify_NAs(wd, imgfn, old_na, new_na = -999):
    """Reclassifies NA values in a scene.
    
    Parameters
    ----------
    wd : str
        Working directory
    imgfn : str
        Relative path to image
    old_na : int
        Old na value
    new_na : int
        New/target na value. The default is -999.
    
    Returns
    -------
    Writes a filtered ENVI raster with appropriate metadata.
    
    """
    
    # set wd and open image connection
    os.chdir(wd)
    r = rio.open(imgfn)
    meta = r.meta.copy()
    rr = r.read()
    # get descriptions
    r_descr = list(r.descriptions)
    
    # reclassify
    rr[rr == old_na] = new_na
    
    # change metadata
    meta.update({'nodata': new_na})
    # output
    with rio.open(imgfn, 'w', **meta) as dst:
        for b in range(r.count):
            dst.set_band_description(b+1, r_descr[b])
        dst.write(rr)


def rm_bad_bands(wd, imgfn, bblfn, band_status_column_name, bad_band_value = 0):
    """Removes bad bands from a hyperspectral scene. 
    
    Parameters
    ----------
    wd : str
        Working directory
    imgfn : str
        Relative path to image
    bblfn : str
        Relative path to bad bands list (.csv)
    band_status_column_name : str
        Which column has the bad band information
    bad_band_value : int
        What is the value that indicates a band is bad? The default is 0.
    
    Returns
    -------
    Writes a raster with bad bands removed. Retains image metadata.
    
    """
    
    # set wd and open image connection
    os.chdir(wd)
    rast = rio.open(imgfn)
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
        print("Band descriptions not found")
    
    # remove bad bands from raster
    rast_rd_bbl = rast_rd[bbands_idx]
    del rast_rd
    
    # update metadata
    meta.update({'count': len(bbands_idx)})
    
    # output file
    with rio.open(imgfn, 'w', **meta) as dst:
        for b in range(len(bbands_idx)):
            dst.set_band_description(b+1, band_labs[b])
        dst.write(rast_rd_bbl)


def clip_scene(wd, imgfn, shpfn):
    """Clips a scene to a shapefile. Retains band-by-band metadata. 

    Parameters
    ----------
    wd : str
        Working directory
    imgfn : str
        Relative path to image
    shpfn : str
        Relative path to shapefile

    Returns
    -------
    Writes a clipped raster with correct metadata. 

    """
    
    # set wd and open image connection
    os.chdir(wd)
    r = rio.open(imgfn)
    meta = r.meta.copy() # grab metadata
    
    # get band descriptions
    r_descr = list(r.descriptions)
    
    # open shapefile
    shp = gpd.read_file(shpfn)
    
    # clip 
    rc, rc_trans = mask.mask(r, shp[['geometry']].values.flatten(), 
                             nodata = meta['nodata'], crop = True)
    
    # change metadata
    meta.update({'height': rc.shape[1], 
                 'width': rc.shape[2],
                 'transform': rc_trans})
    # output
    with rio.open(imgfn, 'w', **meta) as dst:
        for b in range(r.count):
            dst.set_band_description(b+1, r_descr[b])
        dst.write(rc)

def import_reshape(wd, imgfn):  
    """Imports and reshapes a scene
    
    Parameters
    ----------
    wd : str
        Working directory
    imgfn : str
        Relative path to image 
        
    Returns
    ------- 
    Numpy array with rows = pixels, cols = bands/features.
    Metadata
    Band descriptions (i.e., band labels)
    
    """
    # open connection, grab metadata
    os.chdir(wd)
    try:
        rast = rio.open(imgfn)
        meta = rast.meta.copy()
        desc = list(rast.descriptions)
        
        # read and reshape
        rastr = rast.read()
        rastrr = rastr.reshape([rast.count, rast.height * rast.width]).T 

        # return metadata (for output) and reshaped raster
        return rastrr, meta, desc
    except:
        raise ValueError("Pre-processed image not found. Need to run pre-proccessing functions?")