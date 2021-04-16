===============================================================================
AVIRIS-NG Data Product Distribution Document 
===============================================================================

Sarah R. Lundeen, Sarah.R.Lundeen@jpl.nasa.gov
Robert O. Green, Robert.O.Green@jpl.nasa.gov
Michael Eastwood, meastwood@jpl.nasa.gov 
David R. Thompson, David.R.Thompson@jpl.nasa.gov 

This document describes the AVIRIS Next Generation (AVIRIS-NG) L1 and L2 data 
products.  AVIRIS-NG is an imaging spectrometer that measures reflected 
radiance at 5nm intervals in the Visible/Short-Wave Infrared (VSWIR) spectral 
range from 380-2500nm.  

Additional information may be found on http://avirisng.jpl.nasa.gov

-------------------------------------------------------------------------------
OVERVIEW
-------------------------------------------------------------------------------

Each flightline uses a specific base filename prefix: angYYYYMMDDtHHNNSS

YYYY:  The year of the airborne flight run.
MM:    The month of the airborne flight run (i.e. 05 represents May).
DD:    The day of the airborne flight run (22 is the 22nd day of the month)
HH:    UTC hour at the start of acquisition
NN:    UTC minute at the start of acquisition
SS:    UTC second at the start of acquisition

The AVIRIS-NG  data products for a particular airborne flight run are organized 
in the following directory structure, labeled with a processing version marker 
VVV:

/YYYYMMDDtHHNNSS_VVV
|   angYYYYMMDDtHHNNSS_h2o_VVV_img
|   angYYYYMMDDtHHNNSS_h2o_VVV_img.hdr
|   angYYYYMMDDtHHNNSS_rdn_VVV_img
|   angYYYYMMDDtHHNNSS_rdn_VVV_img.hdr
|   angYYYYMMDDtHHNNSS_corr_VVV_img
|   angYYYYMMDDtHHNNSS_corr_VVV_img.hdr
|   angYYYYMMDDtHHNNSS_rdn_VVV_glt
|   angYYYYMMDDtHHNNSS_rdn_VVV_glt.hdr
|   angYYYYMMDDtHHNNSS_rdn_VVV_igm
|   angYYYYMMDDtHHNNSS_rdn_VVV_igm.hdr
|   angYYYYMMDDtHHNNSS_rdn_VVV_loc
|   angYYYYMMDDtHHNNSS_rdn_VVV_loc.hdr
|   angYYYYMMDDtHHNNSS_rdn_VVV_obs
|   angYYYYMMDDtHHNNSS_rdn_VVV_obs.hdr
|   angYYYYMMDDtHHNNSS_rdn_VVV_obs_ort
|   angYYYYMMDDtHHNNSS_rdn_VVV_obs_ort.hdr

The file product codes signify:

   *rdn_obs_ort    	parameters relating to the geometry of observation and
                        illumination rendered using the *glt lookup table
   *rdn_VVV_obs_ort.hdr obs_ort image header file
   *rdn_obs	    	parameters relating to the geometry of observation and
                        illumination in the raw spatial format
   *rdn_VVV_obs.hdr 	obs image header file
   *rdn_VVV_glt         geometric look-up table
   *rdn_VVV_glt.hdr   	GLT image header file
   *rdn_VVV_igm        	input geometry file
   *rdn_VVV_igm.hdr    	IGM image header file
   *rdn_VVV_loc     	pixel location data file
   *rdn_VVV_loc.hdr    	LOC image header file   
   *rdn_VVV_img         orthocorrected, scaled radiance image file
   *rdn_VVV_img.hdr     RDN image header file
   *corr_VVV_img        orthocorrected, scaled reflectance image file
   *corr_VVV_img.hdr    CORR image header file
   *h2o_VVV_img         orthocorrected water absorption data
   *h2o_VVV.hdr         H2O header file


------------------------------------------------------------------------------
FILE DESCRIPTIONS
------------------------------------------------------------------------------

*rdn_VVV_glt   GEOMETRIC LOOKUP TABLE (GLT)

Contents:  Orthocorrected product with a fixed pixel size projected into a 
           rotated UTM system that contains the information about which 
           original pixel occupies which output pixel in the final product. 
           Additionally, each pixel is sign-coded to indicate if it is real 
           (indicated by a positive value) or a nearest-neighbor infill 
           (indicated by negative values).
           
           The GLT file contains two parameters:
              1) sample number
              2) original line number
              
File type: BINARY 32-bit signed long integer.

Format:    Band interleaved by pixel

------------------------------------------------------------------------------

*rdn_VVV_glt.hdr  HEADER FILE FOR GEOMETRIC LOOKUP TABLE (GLT.HDR) DATA
                 
Contents:  Format of each *rdn_VVV_glt file.  This file contains the number of 
           lines, samples, channel, integer format, pixel size, scene 
           elevation, UTM zone number and rotation angle information, etc.
           
File type: ASCII

-------------------------------------------------------------------------------

*rdn_VVV_igm  INPUT GEOMETRY FILE (IGM)

Contents:  UTM ground locations (x,y,elevation) in meters for each pixel in the corresponding 	
		   unorthocorrected radiance image. The IGM file data 
           contain three parameters:
           1) Easting (meters)
           2) Northing (meters)
           3) Estimated ground elevation at each pixel center 
              (reported in meters)
                
           No map correction or resampling is applied to the radiance image 
           cube; the IGM file only reports the surface location of the 
           unadjusted pixel centers.

File type: BINARY 64-bit double precision, floating point IEEE.

Units:     Band 1 - meters
           Band 2 - meters
           Band 3 - meters

Format:    Band interleaved by pixel 

----------------------------------------------------------------------------

*rdn_VVV_igm.hdr  HEADER FILE FOR INPUT GEOMETRY FILE (IGM) DATA

Contents:  Format of each *rdn_VVV_igm file.  This file contains the number of 
           lines, samples, channel, integer format, etc.

File type: ASCII

----------------------------------------------------------------------------

*rdn_VVV_img CALIBRATED PRISM RADIANCE (IMAGE) DATA

Contents:  PRISM calibrated radiance 

File type: BINARY 32-bit big-endian floating point Intel.

Units:     microwatts per centimeter_squared per nanometer per steradian

Format:    Band interleaved by line 

--------------------------------------------------------------------------------

*rdn_VVV_img.hdr HEADER FILE FOR CALIBRATED PRISM RADIANCE (IMAGE) DATA

Contents:  Format of each PRISM calibrated radiance scene.  This file contains  
           the number of lines, samples, channel, etc. It also records the 
           spectral calibration (wavelength and full-width at half-maximum 
           value) for every channel in the radiance data.

File type: ASCII

----------------------------------------------------------------------------

*rdn_VVV_loc   PIXEL LOCATION DATA (LOC)

Contents:  Pixel locations (WGS-84 lat/lon) for each science pixel in the corresponding 
           unorthocorrected radiance image. The LOC file data 
           contain three parameters:
           1) WGS-84 longitude (decimal degrees)
           2) WGS-84 latitude (decimal degrees)
           3) Estimated ground elevation at each pixel center 
              (reported in meters)

File type: BINARY 64-bit double-precision, floating point.

Units:     Band 1: decimal degrees
           Band 2: decimal degrees
           Band 3: meters

Format:    Band interleaved by line 

--------------------------------------------------------------------------------

*rdn_VVV_loc.hdr 	HEADER FILE FOR PIXEL LOCATION DATA (LOC)

Contents:  Format of each PRISM *loc file.  This file contains the 
           number of lines, samples, channel, etc. 

File type: ASCII

----------------------------------------------------------------------------

*rdn_VVV_obs   OBSERVATION PARAMTER FILE (OBS)

Contents:  Observation parameter files in the raw spatial format; matches the corresponding 
           unorthocorrected radiance image. The OBS file data 
           contain eleven parameters:
           1) path length (sensor-to-ground in meters)
           2) to-sensor-azimuth (0 to 360 degrees clockwise from N)
           3) to-sensor-zenith (0 to 90 degrees from zenith)
           4) to-sun-azimuth (0 to 360 degrees clockwise from N)
           5) to-sun-zenith (0 to 90 degrees from zenith)
           6) solar phase (degrees between to-sensor and to-sun vectors in principal plane)
           7) slope (local surface slope as derived from DEM in degrees)
           8) aspect (local surface aspect 0 to 360 degrees clockwise from N)
           9) cosine i (apparent local illumination factor based on DEM slope and aspect 
           	and to sun vector, -1 to 1)
          10) UTC time (decimal hours for mid-line pixels)
          11) Earth-sun distance (AU)

File type: BINARY 64-bit double-precision, floating point.

Format:    Band interleaved by pixel 

--------------------------------------------------------------------------------

*rdn_VVV_obs.hdr 	OBSERVATION PARAMETER FILE (OBS)

Contents:  Format of each PRISM *obs file.  This file contains the 
           number of lines, samples, channel, etc. 

File type: ASCII

----------------------------------------------------------------------------

*rdn_VVV_obs_ort   ORTHOCORRECTED OBSERVATION PARAMTER FILE (OBS ORT)

Contents:  Observation parameter file that has been rendered using the GLT lookup table and
	   matches the orthocorrected imagery. The OBS ORT file data 
           contain eleven parameters:
           1) path length (sensor-to-ground in meters)
           2) to-sensor-azimuth (0 to 360 degrees clockwise from N)
           3) to-sensor-zenith (0 to 90 degrees from zenith)
           4) to-sun-azimuth (0 to 360 degrees clockwise from N)
           5) to-sun-zenith (0 to 90 degrees from zenith)
           6) solar phase (degrees between to-sensor and to-sun vectors in principal plane)
           7) slope (local surface slope as derived from DEM in degrees)
           8) aspect (local surface aspect 0 to 360 degrees clockwise from N)
           9) cosine i (apparent local illumination factor based on DEM slope and aspect 
           	and to sun vector, -1 to 1)
          10) UTC time (decimal hours for mid-line pixels)
          11) Earth-sun distance (AU)

File type: BINARY 64-bit double-precision, floating point.

Format:    Band interleaved by pixel 

--------------------------------------------------------------------------------

*rdn_VVV_obs_ort.hdr 	ORTHOCORRECTED OBSERVATION PARAMETER FILE (OBS ORT)

Contents:  Format of each PRISM *obs_ort file.  This file contains the 
           number of lines, samples, channel, etc. 

File type: ASCII

----------------------------------------------------------------------------

*corr_VVV_img     CALIBRATED AVIRIS-NG REFLECTANCE (IMAGE) DATA

Contents:  AVIRIS-NG calibrated reflectance
           
File type: BINARY 32-bit little-endian floating point IEEE.
           
Units:     Apparent surface reflectance (Gao et al., 1993)
           
Format:    Band interleaved by line 

--------------------------------------------------------------------------------

*corr_VVV_img.hdr   HEADER FILE FOR CALIBRATED AVIRIS REFLECTANCE (IMAGE) DATA

Contents:  Format of each AVIRIS calibrated radiance scene.  This file contains
           the number of lines, samples, channel, etc. It also records the 
           spectral calibration (wavelength and full-width at half-maximum 
           values) for every channel in the radiance data. The "Smoothing 
           factors" field contains a list of multiplicative coefficients 
           applied to smooth the resulting reflectance spectrum.  These 
           coefficients were derived from Calibration measurements using 
           spectrally-invariant surface targets. To remove this correction 
           simply divide the apparent reflectance by these values.
           
File type: ASCII

----------------------------------------------------------------------------

*h2o_VVV   WATER ABSORPTION PATH (IMAGE) DATA

Contents:  Retrieved column water vapor and optical absorption paths for liquid
           H2O and ice

File type: BINARY 32-bit little-endian floating point IEEE.

Units:     Band 1: Retrieved column H2O vapor in cm
           Band 2: Total liquid H2O absorption path in cm
           Band 3: Total ice absorption path in cm

Format:    Band interleaved by line 

--------------------------------------------------------------------------------

*h2o_VVV.hdr HEADER FILE FOR WATER ABSORPTION PATH (IMAGE) DATA

Contents:  Format of each AVIRIS-NG H2O cene.  This file contains the 
           number of lines, samples, channel, etc. 

File type: ASCII


--------------------------------------------------------------------------------
REFERENCES
--------------------------------------------------------------------------------

Gao, B.C., K. H. Heidebrecht, and A. F. H. Goetz, Derivation of scaled surface 
    reflectances from AVIRIS data, Remote Sens. Env., 44, 165-178, 1993


--------------------------------------------------------------------------------
ACKNOWLEDGEMENTS
--------------------------------------------------------------------------------

This research was performed at the Jet Propulsion Laboratory, California 
Institute of Technology, under contract with the National Aeronautics and 
Space Administration (NASA).  We are grateful for the help and assistance of 
colleagues including Bo-Cai Gao (NRL), Ian McCubbin (JPL), Dar Roberts (UCSB),
Mark Helmlinger (JPL), Scott Nolte (JPL), Ernie Diaz (JPL), Daniel Nunes (JPL),
Yasha Mouradi (JPL), and the rest of the AVIRIS-NG team. Copyright 2014 
California Institute of Technology.  All Rights Reserved.  U.S. Government 
Support Acknowledged.

--------------------------------------------------------------------------------
MODIFICATIONS
--------------------------------------------------------------------------------
10 Sept. 2014 (D. R. Thompson) - Initial document
25 July 2016 (S. R. Lundeen) - Revision to orthorectified data products
13 Oct. 2016 (S. R. Lundeen) - Added obs_ort file description
