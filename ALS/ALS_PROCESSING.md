
# AIRBORNE LASER SCANNING DERIVED VARIABLES

High-performance computing (HPC) was used to (pre-)process airborne laser scanning (ALS) tiles. ALS data sets are often composed of hundreds of tiles, making them suitable for (almost) embarassingly parallel HPC processing. Unfortunately, Puhti will be decommissioned in Spring 2026. Although this may affect the reproducibility of the provided scripts, it should be (relatively) straightforward to adapt the steps to a different HPC system, as the general SLURM/Lustre-related logic will be similar.

## Download

Airborne lidar coverage of all of Helsinki is provided by the city administration. Unfortunately, the city's GUI for downloading open data ([kartta.hel.fi](https://kartta.hel.fi/)) does not allow bulk downloads. Therefore, we retrieved the storage location using the developer view when creating a file request and used it to create a bulk downloader `hel_lidar_tile_downloader.R`. It utilizes a tile index file `hel_lidar_tiles.txt` to request all data in a (hopefully) non-offensive way. The full download takes about 4-6 hours. The file index provided here was created by creating a faux index exploiting the coordinate-based naming convention of the tiles (brute force) and filling the gaps manually.

## Preparation

We used `filter_tiles.R` to create a list (index) of all point clouds containing ground points. Tiles without ground points are not useful to us, because ground points are necessary to triangulate a DTM. Our script relies on the City's existing classification of pointg - we did not conduct our own ground classification (e.g., using cloth simulation functions or other methods). Removing point clouds without ground points reduced the amount of tiles from 1281 to 1200. In contrast, `prep_tiles.R` creates a ground-naive index text file (and spatial index as `.gpkg`). 

We used the tiles-with-ground index to homogenize the input and assign the correct CRS (EPSG:3879) to the ```.laz``` header using LAStools `las2las` [(see Documentation)](https://downloads.rapidlasso.de/html/las2las_README.html).
`las2las64 \ -i "$FILE" \ -o "$OUT_FILE" \ -set_point_type 6 \ -set_point_size 41 \ -epsg 3879 \ -olaz`

Homogenization prevents future coordinate reference system (CRS) mismatches and warning messages due to `.las/.laz` version conflicts (e.g., connected to the point size stored in the files). The native CRS of the point clouds is EPSG:3879 (see City of Helsinki documentation), the assignment here is thus to prevent warnings. 

Finally, we split the tile index into four concurrent sub-indices (blocks), because the array size on Puhti's small partition is capped to a maximum of 1000 tasks and the number of jobs that are allowed to queue is limited. The reason for that is purely technical: In array-like jobs that make use of deterministic file lists to index the input (as the one we used here), referential indexing via task number becomes impossible once the task maximum is exceeded. For instance, an array set to fulfill the tasks 900-1200 is not allowed on Puhti, despite the number of tasks being far below the threshold. To circumvent this, we split the index into four sub-indices and always create the same array (`1-300%40`).

We later learned that the maximum number of jobs that can be queued in Puhti are 400, so retroactively it would make more sense to split the tile index into three sub-indices (each containing 400 files), instead of four.

## Stage 1: DTM, DSM, CHM, and normalized point clouds
The first processing step was to derive digital topographic and surface models (DTM and DSM) and canopy height models (CHM), as well as height-normalized point clouds. DTMs are a prerequisite for height normalization, and CHMs are required to discretize canopy metrics. These preprocessing steps are well established and documented elsewhere. Here, we used the lasR and lidR R packages to implement them. LasR is a fast laser scanning pipelining tool that operates as a C++ API within R. It has some limited support for task-internal parallelism (OpenMP), we do not make use of here.

From lidR, we mainly used the `las.catalog` engine to handle file reading and writing ([see Documentation](https://cran.r-project.org/web/packages/lidR/vignettes/lidR-LAScatalog-engine.html)).

![A conceptual flowchart of stage 1](https://github.com/jon-terschan/scripts/blob/main/figures/stage1_concept.png)

For each tile, neighbor tiles intersecting with a certain buffer distance are identified. For that reason, we need the spatial tile index `tile_index.gpkg` that was generated earlier. The pipeline then runs over the tile's full neighborhood (core tile + neighbor tile) and, in the end, copies the core tile results from a temporary location to a permanent output folder. After the task, the temporary locations can be deleted, as the output within it will be redundant and of inferior quality to the core tiles (which are processed with the full neighborhood).

Our approach creates an excessive amount of redundant calculations and outputs: For a single DTM, up to nine files will be processed sequentially. But, considering it will always be necessary to process a neighborhood to avoid edge artifacts, it is the most memory efficient, HPC-friendly solution we found within the constraints of lasR/lidR after extensive testing.

## Stage 2: Canopy metrics
The next step was to estimate canopy metrics from the canopy height models. The methodology here is described in detail in the publication, but generally, these are all simple calculations. Neighborhoods are not needed here, so everything is embarassingly parallel in the truest sense. Each task will calculate metrics on a per-pixel basis and then rasterize the output into the correct output folder.

## Stage 3: SVF
Calculating the skyview factor is a separate stage because it relies on GRASS GIS [r.skyview](https://grass.osgeo.org/grass-stable/manuals/addons/r.skyview.html), instead of R. In our implementation, a single merged raster is expected as input because we wanted to avoid both edge artifacts and introducing additional tiling/neighborhood logic to the scripts.

As input raster, we used our canopy height model because topographical sky occlusion in Helsinki urban parks and forests on 10 m resolution is negligible. In an area of interest characterized by steep topographical inclines and many height differences and such, we recommend calculating SVF using the digital surface model instead.

In terms of calculation settings, we followed the recommendation of [Dirksen et al. (2019)](https://www.sciencedirect.com/science/article/pii/S2212095519300604) by estimating SVF on a 1 m resolution with a radius of 100 meters and 16 search directions.

## References
Dirksen, M., Ronda, R. J., Theeuwes, N. E., & Pagani, G. A. (2019). Sky view factor calculations and its application in urban heat island studies. Urban climate, 30, 100498.