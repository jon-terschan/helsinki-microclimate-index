# Compute other LULC surface metrics (non-building impervious surfaces, exposed bedrocks)
# Inputs: DTM, rocky outcrop polygon, impervious surface polygon
# Outputs: Impervious fraction 10 m and 50 m neighborhood, rock fraction at 10 m, both in 10 m resolution
# -----------------------------------------------------------------------------------------------------------
# 
# REFACTOR LATER TO FIX PATHS TO NEW DIR STRUCTURE
# --- header ---
library(terra)
# inputs
impervious_poly <- vect("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/LULC/lc_impervious_surfaces.gpkg")
rock_poly       <- vect("C:/Users/terschan/Downloads/topo_metrics/lc_rock_hel.gpkg")
dtm_all <- rast("C:/Users/terschan/Downloads/topo_metrics/topometrics/DTM_10m_Helsinki.tif") # only needed for template and CRS

# outputs
out_dir <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/"
imperv_frac_10m_file <- file.path(out_dir, "IMPERV_FRAC_10m_Helsinki.tif")
imperv_frac_50m_file <- file.path(out_dir, "IMPERV_FRAC_50m_Helsinki.tif")
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

w <- focalMat(imperv_frac_10m, 50, type = "circle")

imperv_frac_50m <- focal(
  imperv_frac_10m,
  w = w,
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
writeRaster(imperv_frac_50m, imperv_frac_50m_file, overwrite = TRUE, datatype = "FLT4S", gdal = "COMPRESS=ZSTD")
writeRaster(rock_frac_10m, rock_frac_10m_file, overwrite = TRUE, datatype = "FLT4S", gdal = "COMPRESS=LZW")




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



# --- trees and non woody natural land cover (low veg + bare soil)---

trees_poly <- vect("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/LULC/lc_trees_all.gpkg")
nwn_poly <- vect("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/LULC/lc_nwn.gpkg")
master_temp <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/MASTER_TEMPLATE_10m.tif")

template_1m <- rast(
  ext(master_temp),
  resolution = 1,
  crs = crs(master_temp)
)

trees_poly <- aggregate(trees_poly)
trees_1m <- rasterize(
  trees_poly,
  template_1m,
  field = 1,
  background = 0
)

tree_frac_10m <- aggregate(
  trees_1m,
  fact = 10,
  fun = mean,
  na.rm = TRUE
)

nwn_poly <- aggregate(nwn_poly)
nwn_1m <- rasterize(
  nwn_poly,
  template_1m,
  field = 1,
  background = 0
)

nwn_frac_10m <- aggregate(
  nwn_1m,
  fact = 10,
  fun = mean,
  na.rm = TRUE
)

writeRaster(
  tree_frac_10m,
  "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/TREE_FRAC_10m.tif",
  overwrite = TRUE,
  datatype = "FLT4S",
  gdal = "COMPRESS=LZW"
)
writeRaster(
  nwn_frac_10m,
  "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/NWN_FRAC_10m.tif",
  overwrite = TRUE,
  datatype = "FLT4S",
  gdal = "COMPRESS=LSTD"
)
