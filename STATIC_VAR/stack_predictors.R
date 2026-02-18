library(terra)
pred_dir <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/"
template <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/MASTER_TEMPLATE_10m.tif")
aoi_poly <- vect("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/aoi_outer_buffer.gpkg")

files <- list.files(pred_dir, pattern = "\\.tif$", full.names = TRUE)
files

# Separate CM_loc (multiband) from others
cm_file <- files[grepl("CM_loc", files)]
other_files <- files[!grepl("CM_loc", files)]

# Load
cm <- rast(cm_file)           # multiband
others <- rast(other_files)   # stack of single-band rasters
names(others) <- tools::file_path_sans_ext(basename(other_files))

# For CM multiband
names(cm) <- paste0("CM_loc_band", 1:nlyr(cm))

stack_all <- c(others, cm)
stack_masked <- mask(stack_all, aoi_poly)

# NA mask still missing
writeRaster(
  stack_clean,
  file.path("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/full_stack/pred_stack_10m.tif"),
  overwrite = TRUE,
  datatype = "FLT4S",
  gdal = "COMPRESS=ZSTD"
)



#diagnostics, comment out later
files <- list.files(pred_dir, pattern="\\.tif$", full.names=TRUE)
files
rasters <- lapply(files, rast)

meta <- do.call(rbind, lapply(files, function(f) {
  r <- rast(f)
  e <- ext(r)
  
  data.frame(
    filename = basename(f),
    res_x = res(r)[1],
    res_y = res(r)[2],
    xmin  = e[1],
    xmax  = e[2],
    ymin  = e[3],
    ymax  = e[4],
    ncol  = ncol(r),
    nrow  = nrow(r),
    stringsAsFactors = FALSE
  )
}))

meta
