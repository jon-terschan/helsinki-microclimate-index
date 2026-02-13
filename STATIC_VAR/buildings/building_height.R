# Compute building height from canopy height models and the building footprint.
# Inputs: canopy height model (CHM) raster, building footprint vector
# Outputs: rasters of building height (max, 95th percentile) at x m resolution, masked to building footprints
# -----------------------------------------------------------------------------------------------------------
# script assumes that CHM is normalized to height above ground

# ---- header ---
library(terra)

target_crs <- "EPSG:3879" # target crs

chm_file  <- "C:/Users/terschan/Downloads/building_metrics/chmfill/CHM_merged.tif"
bldg_file <- "C:/Users/terschan/Downloads/building_metrics/bldgs_helsinki.gpkg"
out_dir   <- "C:/Users/terschan/Downloads/building_metrics"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE) 

# ---- load inputs ---
chm <- rast(chm_file) # chm as raster
bldg <- vect(bldg_file) # building footprints as vector
# enforce CRS
bldg <- project(bldg, target_crs)
if (is.na(crs(chm)) || crs(chm) != target_crs) {
  crs(chm) <- target_crs
}

# ---- processing ---
bldg <- crop(bldg, ext(chm) + 5) # crop building to CHM extent
bldg_buf <- buffer(bldg, width = 1) # 1m buffer to ensure overhangs and misalignments are captured

# rasterize building mask at CHM res
bldg_mask <- rasterize(
  bldg_buf,
  chm,
  field = 1,
  background = NA
)

chm_bldg <- mask(chm, bldg_mask) # mask CHM to footprints

fact_10m <- round(10 / res(chm)[1])  # aggregate to 10m should be 20 for 0.5 m

# max height
bldg_max_10m <- aggregate(
  chm_bldg,
  fact = fact_10m,
  fun = max,
  na.rm = TRUE
)
# 95th percentile
bldg_h95_10m <- aggregate(
  chm_bldg,
  fact = fact_10m,
  fun = function(x) {
    if (all(is.na(x))) return(NA)
    quantile(x, probs = 0.95, na.rm = TRUE)
  }
)

# ---- write outputs ----
writeRaster(
  bldg_h95_10m,
  file.path(out_dir, "BLDG_HEIGHT_95Q_10m_Helsinki.tif"),
  overwrite = TRUE,
  datatype = "FLT4S",
  gdal = c("COMPRESS=LZW", "TILED=YES")
)
writeRaster(
  bldg_max_10m,
  file.path(out_dir, "BLDG_HEIGHT_MAX_10m_Helsinki.tif"),
  overwrite = TRUE,
  datatype = "FLT4S",
  gdal = c("COMPRESS=LZW", "TILED=YES")
)