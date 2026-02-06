# HELMIMOD - A predictive machine-learning model of Summer near-ground temperatures in Helsinki parks and urban forests.

A short summary of the model, performance metrics and limitations.

# HPC
We used high-performance computing (HPC) to (pre-)process airborne laser scanning tiles and train/test the model. HPC tasks were performed on CSC's Puhti supercomputer. Unfortunately, Puhti reached the end of his lifecycle in early to mid 2026 which may affect reproducibility in some parts. However, the general SLURM/Lustre logic stays the same regardless. Generally, all HPC analysis involving ALS tiles are embarassingly parallel (or almost) and were ran using single-core job arrays.

# STATIC VARIABLES
## AIRBORNE LASER SCANNING DERIVED VARIABLES
### Downloading ALS files
### Preparing ALS files
We remove files without ground points (relies on Helsinki ground classification) 1281 -> 1200 tiles.
The reason for that is that they cannot be used for DTM generation and will cause issues and overhead on the supercomputer. Thus filter_tiles.R script creates a new index that only retains tiles with valid ground points and creates a output table for debugging with information.

Homogenize files and assign correct CRS to header using las2las. 
las2las64 \ -i "$FILE" \ -o "$OUT_FILE" \ -set_point_type 6 \ -set_point_size 41 \ -epsg 3879 \ -olaz 
This is done to prevent CRS mismatches due to wrongly assigned headers (we know native CRS is EPSG:3879) and warning messages due to mismatches in point size (.laz version context)

Splitting the file list into 4 concurrent blocks because the array size on Puhti's small partition is capped to a maximum of 1000 total tasks. In embarassingly parallel lazy tasks using file lists as indexer such as the one used here, this prevents referential indexing via the task number (e.g. an array with tasks 900-1200 would not be allowed, despite being less a 1000 tasks.)
### Stage 1: DTM, DSM, CHM, and normalized point clouds
### Stage 2: Canopy metrics
### Stage 3: SVF
SVF is a separate stage simply it because it relies on GRASS GIS r.skyview, instead of R. Here, a single merged CHM is expected as input.

## OTHER STATIC VARIABLES
Contains script to generate additional static predictors relevant to the model, mainly related to topography, water presence, and build-up matter. 

# DYNAMIC VARIABLES
## CLF CONVERSION
Scripts used to convert different native logger formats into a common logger format (CLF). Scripts available for SurveyTag and TOMST (TMS4, Thermologgers). CLF looks as follows: 

| datetime                  	| t1   	| t2   	| t3   	| SMC 	|
|---------------------------	|------	|------	|------	|-----	|
| 2024-03-01 00:00:00+00:00 	| 19.5 	| 21.1 	| 22.1 	| 500 	|
| 2024-03-01 00:15:00+00:00 	| 19.6 	| 21.2 	| 22.2 	| 505 	|
| 2024-03-01 00:30:00+00:00 	| 19.2 	| 20.8 	| 21.6 	| 508 	|

datetime is the timestamp in datetime format and UTC timezone. Conversion to UTC is necessary because TOMST loggers record in UTC. Note that is extremely important to keep track of your logger's local timezones, in order to (re)convert timezones.

t1 is soil temperature sensor (if present) in degrees Celsius.
t2 is surface temperature sensor reading (if present) in degrees Celsius
t3 is air temperature sensor reading in degrees Celsius.
SMC is soil moisture count if available. 

## QA

Scripts to conduct various pre-processing and quality assessment operations. 

### date_filter
Filters out data prior to the start of measurements. TOMST loggers always record and cannot be turned off, so they will produce heaps of unrelated signals. The date filter assumes a period of relative temperature stability as the loggers are stored in air conditioning (~21 deg room temperature). This means the date filter won't work correctly if the time series contains real signal from an earlier field campaign. 

Date filter is the script in which I figured out the logic. Loop script does the same in batch.

### big_QA 
Performs various preprocessing operations in batch and spits out a report that allows you to identify which files need special attention. QA is quite conservative, because I have not tested function behavior extensively on the available data. 

### Sledgehammer 
A script for manual quality improvements that still remain after the big_QA, basically the final cleanup. Has some functions to remove and interpolate data and other data curation functions, including a function to remove faulty soil temperature readings and flag out-of-soil (OOS) periods in the data.

Note that there is no functions to fill large gaps in time series. 

### Duplicate detector
Deprecated script to detect timestamp duplicates (faulty measurements). Used for diagnostics, but vibecoded and not refactored.  

### Outlier detection
Deprecated first attempt at writing an outlier detection. Works, but the fine tuning takes too much time and in the end I had to do so much manual data curation anyways, it was easier to just manually remove extreme outliers.  


# Acknowledgements
This publication builds on many excellent open-source software, packages and libraries. Many thanks to the authors and maintainers who make this work possible. ❤️
See [CREDITS.md](CREDITS.md) for details.