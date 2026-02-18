
# Merge CHM tiles (or any other raster tiles) into a large raster
# Inputs: folder of CHM tiles (or any other raster tiles)
# Outputs: a merged raster 
# -----------------------------------------------------------------------------------------------------------
# careful: NO CRS checking, so unified CRS and resolution/alignment is assumed for the input.

library(terra) # for spatial operations
terraOptions(memfrac = 0.8) # we aint have all day boy

### paths
in_dir   <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/loc_canopy_metrics"
out_dir <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/loc_canopy_metrics"
tag <- "CM_loc_HEL"   # tag for whatever is the input/output name

out_file <- file.path(out_dir, paste0("merged_", tag, ".tif"))
out_file

master_template_path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/MASTER_TEMPLATE_10m.tif"
master_template <- rast(master_template_path)

# ---- list & merge ----
raster_files <- list.files(in_dir, pattern="\\.tif$", full.names=TRUE)
merged_raster <- do.call(terra::merge, lapply(raster_files, terra::rast))

# ---- write output ----
writeRaster(
  merged_raster,
  out_file,
  overwrite = TRUE,
  gdal = c(
    "TILED=YES",
    "COMPRESS=LSTD",
    "PREDICTOR=2",  
    "BIGTIFF=YES"
  )
)


### compare geometry and such with master template
compareGeom(master_template, merged_raster)
all(res(master_template) == res(merged_raster)) &&
all(origin(master_template) == origin(merged_raster)) &&
crs(master_template) == crs(merged_raster)
# adjust merged raster to master template (assuming they have same grid origin, res and only differ in extent.)
merged_raster <- extend(merged_raster, master_template)
merged_raster <- crop(merged_raster, master_template)


### ALT1: this is all in one operation, much better for i/o 
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

### ALT2: using virtual rasters, dont really nknow if this is any faster
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
