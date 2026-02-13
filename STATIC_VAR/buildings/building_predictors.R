# Compute building fractions and distance to buildings from the building footprints
# Inputs: building footprint vector
# Outputs: building frac (10m), building frac mean (50m), distance to buildings (1km max), all at 10 m resolution.
# -----------------------------------------------------------------------------------------------------------
# 

# --- header ---
library(terra)

bldg <- vect("C:/Users/terschan/Downloads/building_metrics/bldgs_helsinki.gpkg")
dtm_all <- rast("C:/Users/terschan/Downloads/topo_metrics/topometrics/DTM_10m_Helsinki.tif")
out_dir <- "C:/Users/terschan/Downloads/building_metrics/"

# --- processing ---
#  CRS consistency
bldg <- project(bldg, crs(dtm_all))

# create empty template grid 
template_1m <- rast(
  ext(dtm_all),
  resolution = 1,
  crs = crs(dtm_all)
)

# use template to rasterize buildings
bldg_1m <- rasterize(
  bldg,
  template_1m,
  field = 1,
  background = 0
)

# aggregate building presence to 10m (mean)
bldg_frac_10m <- aggregate(
  bldg_1m,
  fact = 10,
  fun = mean,
  na.rm = TRUE
)

# aggregate building presence to 50m
bldg_frac_mean_50m <- focal(
  bldg_frac_10m,
  w = 5,
  fun = mean,
  na.rm = TRUE
)

# create binary building presence mask and calculate distance
bldg_bin <- bldg_frac_10m > 0
bldg_src <- ifel(bldg_bin, 1, NA)
dist_building <- distance(bldg_src)
dist_building <- clamp(dist_building, 0, 1000)

#--- output ---
writeRaster(
  bldg_frac_10m,
  file.path(out_dir, "BLDG_FRAC_10m.tif"),
  overwrite = TRUE,
  datatype = "FLT4S",
  gdal = "COMPRESS=LZW"
)

writeRaster(
  bldg_frac_mean_50m,
  file.path(out_dir, "BLDG_FRAC_MEAN_50m.tif"),
  overwrite = TRUE,
  datatype = "FLT4S",
  gdal = "COMPRESS=LZW"
)

writeRaster(
  dist_building,
  file.path(out_dir, "BLDG_DIST.tif"),
  overwrite = TRUE,
  datatype = "FLT4S",
  gdal = "COMPRESS=LZW"
)

