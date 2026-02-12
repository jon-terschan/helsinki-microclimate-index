
# Merge CHM tiles (or any other raster tiles) into a large raster
# Inputs: folder of CHM tiles (or any other raster tiles)
# Outputs: a merged raster 
# -----------------------------------------------------------------------------------------------------------
# careful: NO CRS checking, so unified CRS and resolution/alignment is assumed for the input.

# --- merge alternative ---
library(terra) 
terraOptions(memfrac = 0.8) # we aint have all day boy

# ---- paths ----
in_dir   <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chmfill"
out_dir <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chm_full"
tag <- "CHM_05m_HEL"   # tag for whatever is the input/output name

out_file <- file.path(out_dir, paste0("merged_", tag, ".tif"))
out_file

# ---- list & merge ----
raster_files <- list.files(in_dir, pattern="\\.tif$", full.names=TRUE)
merged_raster <- do.call(terra::merge, lapply(raster_files, terra::rast))
merged_raster <- do.call(
  terra::merge,
  c(lapply(raster_files, terra::rast), list(filename = out_file))
)

merged_raster <- do.call(
  terra::merge,
  c(
    lapply(raster_files, terra::rast),
    list(
      filename = out_file,
      wopt = list(
        gdal = c(
          "TILED=YES",
          "COMPRESS=LZW",
          "PREDICTOR=2",
          "BIGTIFF=YES"
        )
      )
    )
  )
)

# ---- write output ----
writeRaster(
  merged_raster,
  out_file,
  overwrite = TRUE,
  gdal = c(
    "TILED=YES",
    "COMPRESS=LZW",
    "PREDICTOR=2",  
    "BIGTIFF=YES"
  )
)

#---alternative using vrt-----
raster_files <- list.files(in_dir, pattern="\\.tif$", full.names=TRUE)

vrt_file <- file.path(out_dir, paste0(tag, ".vrt"))

vrt <- vrt(
  raster_files,
  filename = vrt_file,
  overwrite = TRUE
)

writeRaster(
  rast(vrt_file),
  out_file,
  overwrite = TRUE,
  gdal = c(
    "TILED=YES",
    "COMPRESS=LZW",
    "PREDICTOR=2",
    "BIGTIFF=YES"
  )
)
