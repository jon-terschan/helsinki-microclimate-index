# Stack static predictor rasters into a single multiband predictor stack.
# Inputs: individual 10 m predictor rasters and a master template.
# Outputs: compressed 10 m multiband predictor stack for model training and prediction.
# -----------------------------------------------------------------------------------------------------------

# ---- header ---
library(terra)

# ---- input paths ---
pred_dir <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/"
template <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/MASTER_TEMPLATE_10m.tif")
aoi_poly <- vect("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/aoi_outer_buffer.gpkg")

# ---- processing ----
# list predictor rasters
files <- list.files(pred_dir, pattern = "\\.tif$", full.names = TRUE)

# separate canopy metrics (existing multiband) from single band rasters
cm_file <- files[grepl("CM_loc", files)]
other_files <- files[!grepl("CM_loc", files)]

# load them
cm <- rast(cm_file)           # multiband canopy metrics
others <- rast(other_files)   # stack of single-band rasters

# give clean names, bit old fashioned but i like it this way
names(others) <- c(
  "bldg_dis",
  "bldg_fr_10",
  "bldg_fr_50",
  "chm_max_10m",
  "elev_10",
  "eastness",
  "imp_fr_10",
  "imp_fr_50",
  "nwn_fr_10",
  "oce_dis",
  "oce_fr_10",
  "rock_fr_10",
  "ruggedness",
  "slope",
  "southness",
  "tpi_50",
  "tree_fr_10",
  "water_dis",
  "water_fr_10"
)

# names(others) # check that its correct

# change canopy metrics names
names(cm) <- c(
  "CC",
  "PAI",
  "CCl",
  "uCC",
  "n_als"
)

# stack rasters
stack_all <- c(others, cm)

# export as compressed multiband raster
writeRaster(
  stack_all,
  file.path("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/full_stack/pred_stack_10m.tif"),
  overwrite = TRUE,
  datatype = "FLT4S",
  gdal = "COMPRESS=ZSTD"
)

#diagnostics in case the stacking fails, compare resolution, extent and such
#files <- list.files(pred_dir, pattern="\\.tif$", full.names=TRUE)
#files
#rasters <- lapply(files, rast)

#meta <- do.call(rbind, lapply(files, function(f) {
#  r <- rast(f)
#  e <- ext(r)
#  
#  data.frame(
#    filename = basename(f),
#    res_x = res(r)[1],
#    res_y = res(r)[2],
#    xmin  = e[1],
#    xmax  = e[2],
#    ymin  = e[3],
#    ymax  = e[4],
#    ncol  = ncol(r),
#    nrow  = nrow(r),
#    stringsAsFactors = FALSE
#  )
#}))

#meta
