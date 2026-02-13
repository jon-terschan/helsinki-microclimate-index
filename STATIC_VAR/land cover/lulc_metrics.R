# Compute other LULC surface metrics (non-building impervious surfaces, exposed bedrocks)
# Inputs: DTM, rocky outcrop polygon, impervious surface polygon
# Outputs: Impervious fraction 10 m and 50 m neighborhood, rock fraction at 10 m, both in 10 m resolution
# -----------------------------------------------------------------------------------------------------------
# 
# --- header ---
library(terra)
# inputs
impervious_poly <- vect("C:/Users/terschan/Downloads/topo_metrics/lc_impervious_surfaces.gpkg")
rock_poly       <- vect("C:/Users/terschan/Downloads/topo_metrics/lc_rock_hel.gpkg")
dtm_all <- rast("C:/Users/terschan/Downloads/topo_metrics/topometrics/DTM_10m_Helsinki.tif") # only needed for template and CRS

# outputs
out_dir <- "C:/Users/terschan/Downloads/topo_metrics/"
imperv_frac_10m_file <- file.path(out_dir, "landcover/IMPERV_FRAC_10m_Helsinki.tif")
imperv_frac_50m_file <- file.path(out_dir, "landcover/IMPERV_FRAC_50m_Helsinki.tif")
rock_frac_10m_file   <- file.path(out_dir, "landcover/ROCK_FRAC_10m_Helsinki.tif")

dir.create(file.path(out_dir, "landcover"), recursive = TRUE, showWarnings = FALSE)

# --- processing ---
# create grid template
template_1m <- rast(
  ext(dtm_all),
  resolution = 1,
  crs = crs(dtm_all)
)

# rasterize to grid template
imperv_1m <- rasterize(
  impervious_poly,
  template_1m,
  field = 1,
  background = 0
)

# aggregate to fractions
imperv_frac_10m <- aggregate(
  imperv_1m,
  fact = 10,
  fun = mean,
  na.rm = TRUE
)

imperv_frac_50m <- aggregate(
  imperv_1m,
  fact = 50,
  fun = mean,
  na.rm = TRUE
)

# rasterize bedrock
rock_1m <- rasterize(
  rock_poly,
  template_1m,
  field = 1,
  background = 0
)

rock_frac_10m <- aggregate(
  rock_1m,
  fact = 10,
  fun = mean,
  na.rm = TRUE
)

# --- write outputs ---
writeRaster(imperv_frac_10m, imperv_frac_10m_file, overwrite = TRUE, datatype = "FLT4S", gdal = "COMPRESS=LZW")
writeRaster(imperv_frac_50m, imperv_frac_50m_file, overwrite = TRUE, datatype = "FLT4S", gdal = "COMPRESS=LZW")
writeRaster(rock_frac_10m, rock_frac_10m_file, overwrite = TRUE, datatype = "FLT4S", gdal = "COMPRESS=LZW")