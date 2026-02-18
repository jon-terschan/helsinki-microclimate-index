# Compute building topgraphic metrics (slope, eastness, southness, ruggedness) and water metrics (distance to inland water, ocean and inland water/ocean fraction)
# Inputs: DTM tiles (from ALS), water body polygons (from LULC)
# Outputs: building frac (10m), building frac mean (50m), distance to buildings (1km max), all at 10 m resolution.
# -----------------------------------------------------------------------------------------------------------
# 

# --- header ---
library(terra)

target_crs <- "EPSG:3879"

# inputs
in_dir  <- "C:/Users/terschan/Downloads/topo_metrics/DTM"
out_dir <- "C:/Users/terschan/Downloads/topo_metrics/"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
ocean_poly <- vect("C:/Users/terschan/Downloads/topo_metrics/lc_sea_hel.gpkg") # ocean polygon
river_poly <- vect("C:/Users/terschan/Downloads/topo_metrics/lc_water_hel.gpkg") # inland water polygon

# outputs
dtm_file <- file.path(out_dir, "topometrics/DTM_10m_Helsinki.tif") 
# outputs: water metrics
ocean_frac_10m_file <- file.path(out_dir, "watermask/OCEAN_FRAC_10m_Helsinki.tif")
water_frac_10m_file <- file.path(out_dir, "watermask/WATER_FRAC_10m_Helsinki.tif")
dist_ocean_file <- file.path(out_dir, "watermask/OCEAN_DIST_10m_Helsinki.tif")
dist_river_file <- file.path(out_dir, "watermask/RIVER_DIST_10m_Helsinki.tif")
# outputs: topometrics
slope_file <- file.path(out_dir, "topometrics/SLOPE_10m_Helsinki.tif")
eastness_file <- file.path(out_dir, "topometrics/EASTNESS_10m_Helsinki.tif")
southness_file <- file.path(out_dir, "topometrics/SOUTHNESS_10m_Helsinki.tif")
rugged_file <- file.path(out_dir, "topometrics/RUGGEDNESS_10m_Helsinki.tif")

# --- processing ---
# --- 1: create merged DTM from tiles (native distribution) ---
xyz_files <- list.files(in_dir, pattern = "\\.xyz$", full.names = TRUE)
dtm_list <- lapply(xyz_files, function(f) {
  cat("  reading:", basename(f), "\n")
  xyz <- read.table(f)
  colnames(xyz) <- c("x", "y", "z")
  r <- rast(xyz, type = "xyz")
  crs(r) <- target_crs
  r
})

dtm_all <- do.call(merge, dtm_list) # merge tiles

# write output
writeRaster(
  dtm_all,
  dtm_file,
  overwrite = TRUE,
  datatype = "FLT4S",
  gdal = c("COMPRESS=LZW")
) 

# --- 2: water metrics ---
# crate a grid template based on the merged DTM
template_1m <- rast(
  ext(dtm_all),
  resolution = 1,
  crs = crs(dtm_all)
)

# rasterize the polygon
ocean_1m <- rasterize(ocean_poly, template_1m, field = 1, background = 0)
# aggregate to ocean fraction
ocean_frac_10m <- aggregate(
  ocean_1m,
  fact = 10,
  fun = mean,
  na.rm = TRUE
)

# rasterize
water_1m  <- rasterize(river_poly, template_1m, field = 1, background = 0)
# aggregate to inland water fracton
water_frac_10m <- aggregate(
  water_1m,
  fact = 10,
  fun = mean,
  na.rm = TRUE
)

# create water presence binary and calculate distance
river_bin  <- water_frac_10m  > 0
ocean_bin  <- ocean_frac_10m  > 0
river_src <- ifel(river_bin, 1, NA)
ocean_src <- ifel(ocean_bin, 1, NA)
dist_river <- distance(river_src) # euc dist
dist_ocean <- distance(ocean_src) # euclidian dist
dist_river <- clamp(dist_river, 0, 500) # cap effect size
dist_ocean <- clamp(dist_ocean, 0, 3000) # cap effect distance

# write outputs
writeRaster(ocean_frac_10m, ocean_frac_10m_file, overwrite = TRUE, datatype = "FLT4S")
writeRaster(water_frac_10m, water_frac_10m_file, overwrite = TRUE, datatype = "FLT4S")
writeRaster(dist_river, dist_river_file, overwrite = TRUE, datatype = "FLT4S")
writeRaster(dist_ocean, dist_ocean_file, overwrite = TRUE, datatype = "FLT4S")

# --- 3: topographic metrics ----
slope  <- terrain(dtm_all, "slope", unit = "degrees") # calc slope
aspect <- terrain(dtm_all, "aspect", unit = "degrees")

# convert aspect to east/southness
eastness  <- sin(aspect * pi / 180)
southness <- -cos(aspect * pi / 180)

# ruggedness (3x3 tiles sd, probably same as slope)
rugged <- focal(dtm_all, w = 3, fun = sd, na.rm = TRUE)

# masking, superfluous because DTM covers more
# than helsinki, whereas the mask does not 
#slope     <- mask(slope, ocean_mask, inverse = TRUE)
#eastness  <- mask(eastness, ocean_mask, inverse = TRUE)
#southness <- mask(southness, ocean_mask, inverse = TRUE)
#rugged    <- mask(rugged, ocean_mask, inverse = TRUE)

writeRaster(slope, slope_file, overwrite = TRUE, datatype = "FLT4S", gdal = "COMPRESS=LZW")
writeRaster(eastness, eastness_file, overwrite = TRUE, datatype = "FLT4S", gdal = "COMPRESS=LZW")
writeRaster(southness, southness_file, overwrite = TRUE, datatype = "FLT4S", gdal = "COMPRESS=LZW")
writeRaster(rugged, rugged_file, overwrite = TRUE, datatype = "FLT4S", gdal = "COMPRESS=LZW")


## TPI ##
# --- TPI (Topographic Position Index) ---
out_dir <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/"

dtm <- file.path(out_dir, "/DTM_10m_Helsinki.tif") 
dtm <- rast(dtm)
master <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/MASTER_TEMPLATE_10m.tif")
ext(master)
compareGeom(master, dtm)
# project/align exactly to master geometry
dtm_master <- project(
  dtm_all,
  master,
  method = "bilinear"
)

# 5x5 window (~50 m at 10 m resolution)
w_tpi <- matrix(1, 5, 5)

# mean elevation of neighborhood
mean_neigh <- focal(
  dtm,
  w = w_tpi,
  fun = mean,
  na.rm = TRUE,
  pad = TRUE
)

# TPI = cell elevation - neighborhood mean
tpi <- dtm - mean_neigh

# export
tpi_file <- file.path(out_dir, "/TPI_50m_10m_Helsinki.tif")
writeRaster(tpi, tpi_file, overwrite = TRUE, datatype = "FLT4S", gdal = "COMPRESS=ZSTD")
global(tpi, c("min", "max", "mean", "sd"), na.rm=TRUE)
