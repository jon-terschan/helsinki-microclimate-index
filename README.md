# HELMI - Helsinki Microclimate Index: A predictive machine-learning model of Summer near-ground temperatures in Helsinki parks and urban forests.

A short summary of the model, performance metrics and limitations. The model's target domain is urban parks and forests. 

## Performance
Performance overview.

## Limitations
Limitations overview.

## Improvements
Proposed changes to improve the accuracy of future versions:
* Change model to XGBoost, probably marginal increases but still.
* Incorporate detailed cloud information (e.g., hourly METEOSAT cloud masks) into the model.
* Add more ERA5-Land fields (radiation flow, wind) as dynamic predictors.  

## Acknowledgements
This publication builds on many excellent open-source software, packages and libraries. Many thanks to the authors and maintainers who make this work possible. ❤️
See [CREDITS.md](CREDITS.md) for details.

For acknowledgements related to the research in which Helmi was published, please check out the corresponding publication.

## A note on HPC
High-performance computing (HPC) was used to (pre-)process airborne laser scanning (ALS) tiles and for model training, testing, and predictions. ALS data sets are often composed of hundreds of tiles, making them suitable for (almost) embarassingly parallel HPC processing. We used CSC's Puhti supercomputer and want to acknowledge the computational resources contributed by CSC here. Unfortunately, Puhti will be decommissioned in Spring 2026. Although this may affect the reproducibility of the provided scripts, it should be (relatively) straightforward to adapt the steps to a different HPC system, as the general SLURM/Lustre-related logic will be similar.

# Additional documentation

More in-depth (technical) documentation of the content of this repository can be found below.
___

# STATIC VARIABLES
## AIRBORNE LASER SCANNING DERIVED VARIABLES 
### Download
Airborne lidar coverage of all of Helsinki is provided by the city administration. Unfortunately, the city's GUI for downloading open data ([kartta.hel.fi](https://kartta.hel.fi/)) does not allow bulk downloads. Therefore, we retrieved the storage location using the developer view when creating a file request and used it to create a bulk downloader `hel_lidar_tile_downloader.R`. It utilizes a tile index file `hel_lidar_tiles.txt` to request all data in a (hopefully) non-offensive way. The full download takes about 4-6 hours. The file index provided here was created by creating a faux index exploiting the coordinate-based naming convention of the tiles (brute force) and filling the gaps manually.

### Preparation
We used `filter_tiles.R` to create a list (index) of all point clouds containing ground points. Tiles without ground points are not useful to us, because ground points are necessary to triangulate a DTM. Our script relies on the City's existing classification of pointg - we did not conduct our own ground classification (e.g., using cloth simulation functions or other methods). Removing point clouds without ground points reduced the amount of tiles from 1281 to 1200. In contrast, `prep_tiles.R` creates a ground-naive index text file (and spatial index as `.gpkg`). 

We used the tiles-with-ground index to homogenize the input and assign the correct CRS (EPSG:3879) to the ```.laz``` header using LAStools `las2las` [(see Documentation)](https://downloads.rapidlasso.de/html/las2las_README.html).
`las2las64 \ -i "$FILE" \ -o "$OUT_FILE" \ -set_point_type 6 \ -set_point_size 41 \ -epsg 3879 \ -olaz`

Homogenization prevents future coordinate reference system (CRS) mismatches and warning messages due to `.las/.laz` version conflicts (e.g., connected to the point size stored in the files). The native CRS of the point clouds is EPSG:3879 (see City of Helsinki documentation), the assignment here is thus to prevent warnings. 

Finally, we split the tile index into four concurrent sub-indices (blocks), because the array size on Puhti's small partition is capped to a maximum of 1000 tasks and the number of jobs that are allowed to queue is limited. The reason for that is purely technical: In array-like jobs that make use of deterministic file lists to index the input (as the one we used here), referential indexing via task number becomes impossible once the task maximum is exceeded. For instance, an array set to fulfill the tasks 900-1200 is not allowed on Puhti, despite the number of tasks being far below the threshold. To circumvent this, we split the index into four sub-indices and always create the same array (`1-300%40`).

We later learned that the maximum number of jobs that can be queued in Puhti are 400, so retroactively it would make more sense to split the tile index into three sub-indices (each containing 400 files), instead of four. 

### Stage 1: DTM, DSM, CHM, and normalized point clouds
The first processing step was to derive digital topographic and surface models (DTM and DSM) and canopy height models (CHM), as well as height-normalized point clouds. DTMs are a prerequisite for height normalization, and CHMs are required to discretize canopy metrics. These preprocessing steps are well established and documented elsewhere. Here, we used the lasR and lidR R packages to implement them. LasR is a fast laser scanning pipelining tool that operates as a C++ API within R. It has some limited support for task-internal parallelism (OpenMP), we do not make use of here. 

From lidR, we mainly used the `las.catalog` engine to handle file reading and writing ([see Documentation](https://cran.r-project.org/web/packages/lidR/vignettes/lidR-LAScatalog-engine.html)).

![A conceptual flowchart of stage 1](https://github.com/jon-terschan/scripts/blob/main/figures/stage1_concept.png)

For each tile, neighbor tiles intersecting with a certain buffer distance are identified. For that reason, we need the spatial tile index `tile_index.gpkg` that was generated earlier. The pipeline then runs over the tile's full neighborhood (core tile + neighbor tile) and, in the end, copies the core tile results from a temporary location to a permanent output folder. After the task, the temporary locations can be deleted, as the output within it will be redundant and of inferior quality to the core tiles (which are processed with the full neighborhood). 

Our approach creates an excessive amount of redundant calculations and outputs: For a single DTM, up to nine files will be processed sequentially. But, considering it will always be necessary to process a neighborhood to avoid edge artifacts, it is the most memory efficient, HPC-friendly solution we found within the constraints of lasR/lidR after extensive testing.

### Stage 2: Canopy metrics
The next step was to estimate canopy metrics from the canopy height models. The methodology here is described in detail in the publication, but generally, these are all simple calculations. Neighborhoods are not needed here, so everything is embarassingly parallel in the truest sense. Each task will calculate metrics on a per-pixel basis and then rasterize the output into the correct output folder. 

### Stage 3: SVF
Calculating the skyview factor is a separate stage because it relies on GRASS GIS [r.skyview](https://grass.osgeo.org/grass-stable/manuals/addons/r.skyview.html), instead of R. Here, a single merged CHM for the whole AOI is expected as input. Reasons for that are to avoid both edge artifacts and more tiling/neighborhood operations. 

In terms of settings, we adhered to [Dirksen et al. (2019)](https://www.sciencedirect.com/science/article/pii/S2212095519300604), who recommended estimating SVF on a 1 m resolution with a radius of 100 meters and 16 search directions. 

Dirksen, M., Ronda, R. J., Theeuwes, N. E., & Pagani, G. A. (2019). Sky view factor calculations and its application in urban heat island studies. Urban climate, 30, 100498.

## OTHER STATIC VARIABLES
We created many scripts generating additional static predictors for the model related to topography, water presence and built-up matter. Some examples include:

* Elevation, slope, slopeaspect (Eastness/Southness), and ruggedness
* Water presence, distance to oceans/inland water bodies
* Building presence, height, and distance from buildings
* Presence of other impervious surfaces (concrete roads and sealed surfaces)
* Rocky outcrop presence 

The topographic features derived from the 2021 [City of Helsinki digital elevation model](https://hri.fi/data/en_GB/dataset/helsingin-korkeusmalli). Water, building, and other rasters are derived from the 2024 [Helsinki region land cover data set](https://www.hsy.fi/en/environmental-information/open-data/avoin-data---sivut/helsinki-region-land-cover-dataset/). These are simple data preparation steps, that do not warrant long documentation. Most involve rasterization to the same 1 m grid template and then upscaling to the 10 m prediction grid.  

Building height is a separate script, because it requires the merged CHM to be masked by a building mask.
# DYNAMIC VARIABLES
## REMODELING DATA
We downloaded and tested both ERA5-Land and CERRA-Land data as reference data sets of ambient climate. CERRA has a better spatial but worse time resolution than ERA-5 Land, so it likely captures the ocean-land climatic gradient better. 

## MICROCLIMATE MEASUREMENTS
### CLF CONVERSION
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

### QA

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
