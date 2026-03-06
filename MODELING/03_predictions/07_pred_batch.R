#################################
### HELMI PREDICTION WRAPPER ####
#################################
# the purpose of this script is to batch predict using the helmi model
# implementation is mostly ranger and parallel 
# this uses manual 2-core multithreading based on r parallel package and manual spatial tiling
# so it can run on the Puhti large partition (enforced multithreading) 
# for some reason simply reducing terra/ranger predict implementation 
# took much much much longer, around 2 hours per prediction step (1h), i suspect it scales
# poorly for large models or large prediction rasters, so everything here is manual

# packages
library(terra)
library(ranger)
library(ncdf4)
library(lubridate)

# memory logger script for debugging 
# this is now completely superfluous
# and was only to test how much overhead we
# are need to plan for with the raster preprocessing
log_mem <- function(stage) {
  gc_out <- gc()

  n_mb <- gc_out["Ncells","used"] * 56 / 1024^2
  v_mb <- gc_out["Vcells","used"] * 8  / 1024^2
  r_mem_mb <- n_mb + v_mb

  rss_kb <- as.numeric(system("grep VmRSS /proc/self/status | awk '{print $2}'", intern = TRUE))
  rss_mb <- rss_kb / 1024

  cat(sprintf(
    "\n[%s] %s\n  R heap: %.1f MB\n  RSS: %.1f MB\n\n",
    format(Sys.time(), "%H:%M:%S"),
    stage,
    r_mem_mb,
    rss_mb
  ))

  flush.console()
}

# disable all other multithreading options to make sure they dont mess with each other
options(ranger.num.threads = 1) # ranger internal multithreading command 
# open MP variables, just to make sure
Sys.setenv(
  OMP_NUM_THREADS = 1,
  OPENBLAS_NUM_THREADS = 1,
  MKL_NUM_THREADS = 1
)

# SLURM IDENTIFIERS
array_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))
job_id   <- Sys.getenv("SLURM_JOB_ID")

# set temp directory
tmp_dir <- file.path(
  "/scratch/project_2001208/Jonathan/tmp",
  paste0("job_", job_id, "_", array_id)
)
dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)

# terra options
terraOptions(
  memfrac = 0.1,     # 10% of available RAM
  memmax  = 3,       # GB hard cap
  todisk  = TRUE,
  progress = 1,
  tempdir = tmp_dir
)

# ---- prediction schedule
schedule <- readRDS("/scratch/project_2001208/Jonathan/model/data/processed/ML/prediction_schedule.rds")
target_time <- schedule[array_id]

# ---- STATIC RASTERS
model <- readRDS("/scratch/project_2001208/Jonathan/model/models/rf/helmi_2000_v1.4_test.rds")
static_stack <- rast("/scratch/project_2001208/Jonathan/model/data/processed/rasters_static/pred_stack_10m.tif")
pred_mask <- rast("/scratch/project_2001208/Jonathan/model/data/processed/rasters_static/prediction_mask.tif")
feature_order <- readLines("/scratch/project_2001208/Jonathan/model/data/processed/ML/feature_order.txt")

log_mem("Loaded static stack")

# -----ERA5
# open era 5 netcdf and extract everything, then close it
nc_path <- "/scratch/project_2001208/Jonathan/model/data/processed/rasters_dynamic/ERA5l_SUMMER_24_25_HEL.netcdf"
nc <- nc_open(nc_path)

time_raw  <- ncvar_get(nc, "valid_time")
era_times <- as.POSIXct(time_raw, origin = "1970-01-01", tz = "UTC")
nc_index  <- match(target_time, era_times)
if (is.na(nc_index)) { nc_close(nc); stop("Timestamp not found in NetCDF") }

extract_var <- function(varname, ts_index) {
  ncvar_get(nc, varname, start = c(1,1,ts_index), count = c(-1,-1,1))
}

t2m  <- extract_var("t2m",    nc_index)
ssrd <- extract_var("ssrd",   nc_index)
u10  <- extract_var("u10",    nc_index)
v10  <- extract_var("v10",    nc_index)
tp   <- extract_var("tp",     nc_index)
wind <- extract_var("wind_s", nc_index)
lon  <- ncvar_get(nc, "longitude")
lat  <- ncvar_get(nc, "latitude")

nc_close(nc)  # close as soon as all data is extracted
log_mem("Extracted ERA slice")

# ---- Output path
base_dir <- "/scratch/project_2001208/Jonathan/model/predictions/10m"
day_path <- file.path(base_dir, format(target_time, "%Y%m%d"))
dir.create(day_path, recursive = TRUE, showWarnings = FALSE)

output_path <- file.path(
  day_path,
  paste0("pred_", format(target_time, "%Y%m%d_%H%M"), ".tif")
)

# ---- Build ERA rasters 
dx <- abs(lon[2] - lon[1])
dy <- abs(lat[2] - lat[1])

xmin <- min(lon) - dx/2
xmax <- max(lon) + dx/2
ymin <- min(lat) - dy/2
ymax <- max(lat) + dy/2

make_raster <- function(slice) {
  slice2 <- aperm(slice, c(2,1))
  r <- rast(
    nrows = length(lat),
    ncols = length(lon),
    xmin = xmin, xmax = xmax,
    ymin = ymin, ymax = ymax,
    crs  = "EPSG:4326"
  )
  values(r) <- as.vector(t(slice2[nrow(slice2):1, ]))
  r
}

era_stack <- c(
  make_raster(t2m),
  make_raster(ssrd),
  make_raster(u10),
  make_raster(v10),
  make_raster(tp),
  make_raster(wind)
)
log_mem("Built ERA stack")
names(era_stack) <- c("t2m","ssrd","u10","v10","tp","wind_s")

# ---- Crop ERA to prediction mask extent (in WGS84) 
mask_ext_static  <- ext(pred_mask)
mask_poly_static <- as.polygons(mask_ext_static, crs = crs(pred_mask))
mask_poly_wgs84  <- project(mask_poly_static, "EPSG:4326")

# ---- Crop ERA with buffer of 1 ERA5 cell
# this is done to avoid edge data loss 
era_dx <- res(era_stack)[1]
era_dy <- res(era_stack)[2]

mask_ext_wgs84 <- ext(mask_poly_wgs84)
buffered_ext <- ext(
  mask_ext_wgs84$xmin - era_dx,
  mask_ext_wgs84$xmax + era_dx,
  mask_ext_wgs84$ymin - era_dy,
  mask_ext_wgs84$ymax + era_dy
)

era_cropped <- crop(era_stack, buffered_ext)

# ---- ERA reprojection
# disk backed to reduce memory load
tmp_file <- file.path(tmp_dir, "era_proj.tif")

era_proj <- project(
  era_cropped,
  static_stack,
  method   = "bilinear",
  filename = tmp_file,
  overwrite = TRUE,
  wopt = list(datatype = "FLT4S")
)
log_mem("After projection")

rm(era_stack, era_cropped, t2m, ssrd, u10, v10, tp, wind)
gc()

# ---- Cyclical encoding ----
template <- static_stack[[1]]

hour <- hour(target_time)
doy  <- yday(target_time)

hour_sin_val <- sin(2*pi*hour/24)
hour_cos_val <- cos(2*pi*hour/24)
doy_sin_val  <- sin(2*pi*doy/365)
doy_cos_val  <- cos(2*pi*doy/365)

r_hour_sin <- template * 0 + hour_sin_val
r_hour_cos <- template * 0 + hour_cos_val
r_doy_sin  <- template * 0 + doy_sin_val
r_doy_cos  <- template * 0 + doy_cos_val

names(r_hour_sin) <- "hour_sin"
names(r_hour_cos) <- "hour_cos"
names(r_doy_sin)  <- "doy_sin"
names(r_doy_cos)  <- "doy_cos"

# ---- Combine predictors ----
full_stack <- c(
  static_stack,
  era_proj,
  r_hour_sin,
  r_hour_cos,
  r_doy_sin,
  r_doy_cos
)
log_mem("After stacking predictors")
full_stack <- full_stack[[feature_order]]

# ---- Apply prediction mask ----
full_stack <- mask(full_stack, pred_mask)
log_mem("Applied prediction mask")
gc()

# ---- Write full_stack to disk so parallel workers can re-open it safely ----
# terra SpatRaster external pointers are not valid across spawned processes;
# workers must re-open from a file path instead.
full_stack_path <- file.path(tmp_dir, "full_stack.tif")
writeRaster(full_stack, full_stack_path,
  overwrite = TRUE,
  datatype  = "FLT4S",
  gdal      = c("COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6")
)
log_mem("Written full_stack to disk")

# ---- Tiling configuration ----
library(parallel)

nx <- 4
ny <- 4

ncols_total <- ncol(full_stack)
nrows_total <- nrow(full_stack)

col_breaks <- round(seq(1, ncols_total + 1, length.out = nx + 1))
row_breaks <- round(seq(1, nrows_total + 1, length.out = ny + 1))

tiles <- vector("list", nx * ny)
k <- 1
for (iy in 1:ny) {
  for (ix in 1:nx) {
    tiles[[k]] <- list(
      row_from = row_breaks[iy],
      row_to   = row_breaks[iy + 1] - 1,
      col_from = col_breaks[ix],
      col_to   = col_breaks[ix + 1] - 1,
      path     = file.path(tmp_dir, paste0("tile_", k, ".tif"))
    )
    k <- k + 1
  }
}

# ---- Tile prediction function ----
# Re-opens raster from disk inside the worker — avoids invalid external pointer error
predict_tile <- function(tile, stack_path, mdl) {
  library(terra)
  library(ranger)
  stack <- rast(stack_path)
  tile_ext <- ext(
    xFromCol(stack, tile$col_from) - res(stack)[1] / 2,
    xFromCol(stack, tile$col_to)   + res(stack)[1] / 2,
    yFromRow(stack, tile$row_to)   - res(stack)[2] / 2,
    yFromRow(stack, tile$row_from) + res(stack)[2] / 2
  )
  full_tile <- crop(stack, tile_ext)
  predict(full_tile, mdl,
    filename  = tile$path,
    overwrite = TRUE,
    wopt = list(datatype = "FLT4S", gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6"))
  )
  gc()
  return(tile$path)
}

# ---- Run parallel prediction — 2 workers, one tile at a time each ----
cl <- makeCluster(2)
clusterExport(cl, c("full_stack_path", "model"), envir = environment())

tile_files <- parLapply(cl, tiles, predict_tile,
                        stack_path = full_stack_path, mdl = model)
stopCluster(cl)

tile_files <- unlist(tile_files)
log_mem("All tiles predicted")

# ---- Mosaic with merge() — tolerant of floating point extent differences ----
tile_rasts <- lapply(tile_files, rast)
do.call(merge, c(tile_rasts, list(
  filename  = output_path,
  overwrite = TRUE,
  wopt = list(datatype = "FLT4S", gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6"))
)))
log_mem("After mosaic")

# ---- Re-apply mask to remove ranger predictions on NA/water cells ----
result <- rast(output_path)
result <- mask(result, pred_mask)
writeRaster(result, output_path,
  overwrite = TRUE,
  datatype  = "FLT4S",
  gdal      = c("COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6")
)

# ---- Cleanup ----
file.remove(tmp_file)
file.remove(full_stack_path)
file.remove(tile_files)