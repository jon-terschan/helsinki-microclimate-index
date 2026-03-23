#################################
### HELMI PREDICTION WRAPPER ####
#################################
# Batch prediction script for the HELMI urban temperature model.
# Runs as a SLURM array job on Puhti (small partition, 2 CPUs, 16GB RAM).
# Each array task predicts one hourly timestep.
#
# Parallelism: manual spatial tiling with R parallel (makeCluster, 2 socket workers).
# Workers load model and raster stack from disk — avoids socket serialization of large objects.
# Tiles are written as INT2S (x100 scaled) to reduce I/O, converted back to FLT4S at the end.

library(terra)
library(ranger)
library(ncdf4)
library(lubridate)
library(parallel)

log_step <- function(msg) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), msg))
  flush.console()
}

# ---- threading — disable all implicit multithreading
# parallelism is handled explicitly via makeCluster(2)
options(ranger.num.threads = 1)
Sys.setenv(
  OMP_NUM_THREADS      = 1,
  OPENBLAS_NUM_THREADS = 1,
  MKL_NUM_THREADS      = 1
)

# ---- SLURM identifiers
array_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))
job_id   <- Sys.getenv("SLURM_JOB_ID")
log_step(sprintf("Array task %d started (job %s)", array_id, job_id))

# ---- stagger startup to avoid Lustre I/O contention when multiple array jobs
# land on the same node simultaneously — spreads cluster init across 5 x 30s slots
stagger_secs <- (array_id %% 5) * 30
log_step(sprintf("Stagger sleep: %ds", stagger_secs))
Sys.sleep(stagger_secs)

# ---- per-job temp directory on scratch
tmp_dir <- file.path(
  "/scratch/project_2001208/Jonathan/tmp",
  paste0("job_", job_id, "_", array_id)
)
dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)
log_step(sprintf("Temp dir: %s", tmp_dir))

# ---- terra options
terraOptions(
  memfrac  = 0.1,  # 10% of available RAM
  memmax   = 3,    # hard cap in GB
  todisk   = TRUE,
  progress = 0,    # suppress progress bars in batch logs
  tempdir  = tmp_dir
)

# ---- prediction schedule
schedule    <- readRDS("/scratch/project_2001208/Jonathan/model/data/processed/ML/prediction_schedule.rds")
target_time <- schedule[array_id]
log_step(sprintf("Target time: %s", format(target_time)))

# ---- static inputs
# model is never loaded in the parent process — workers read it directly from disk
# this keeps ~2GB free during the entire preprocessing phase
model_path    <- "/scratch/project_2001208/Jonathan/model/models/rf/helmi_2000_v1.4_test.rds"
static_stack  <- rast("/scratch/project_2001208/Jonathan/model/data/processed/rasters_static/pred_stack_10m.tif")
pred_mask     <- rast("/scratch/project_2001208/Jonathan/model/data/processed/rasters_static/prediction_mask.tif")
feature_order <- readLines("/scratch/project_2001208/Jonathan/model/data/processed/ML/feature_order.txt")
log_step("Static inputs loaded")

# ---- ERA5 — open, extract all variables, close immediately
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

nc_close(nc)
log_step("ERA5 extracted and NetCDF closed")

# ---- output path
base_dir <- "/scratch/project_2001208/Jonathan/model/predictions/10m"
day_path <- file.path(base_dir, format(target_time, "%Y%m%d"))
dir.create(day_path, recursive = TRUE, showWarnings = FALSE)

output_path <- file.path(
  day_path,
  paste0("pred_", format(target_time, "%Y%m%d_%H%M"), ".tif")
)
log_step(sprintf("Output path: %s", output_path))

# ---- build ERA5 raster stack
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
    xmin  = xmin, xmax = xmax,
    ymin  = ymin, ymax = ymax,
    crs   = "EPSG:4326"
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
names(era_stack) <- c("t2m","ssrd","u10","v10","tp","wind_s")
log_step("ERA5 raster stack built")

# ---- crop ERA5 to mask extent with 1-cell buffer to avoid edge data loss
mask_ext_static  <- ext(pred_mask)
mask_poly_static <- as.polygons(mask_ext_static, crs = crs(pred_mask))
mask_poly_wgs84  <- project(mask_poly_static, "EPSG:4326")

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
log_step("ERA5 cropped to mask extent")

# ---- reproject ERA5 to static stack grid — disk backed
tmp_file <- file.path(tmp_dir, "era_proj.tif")

era_proj <- project(
  era_cropped,
  static_stack,
  method    = "bilinear",
  filename  = tmp_file,
  overwrite = TRUE,
  wopt      = list(datatype = "FLT4S")
)
log_step("ERA5 reprojected to static grid")

rm(era_stack, era_cropped, t2m, ssrd, u10, v10, tp, wind)
gc()

# ---- cyclical time encoding
template <- static_stack[[1]]

hour <- hour(target_time)
doy  <- yday(target_time)

r_hour_sin <- template * 0 + sin(2*pi*hour/24)
r_hour_cos <- template * 0 + cos(2*pi*hour/24)
r_doy_sin  <- template * 0 + sin(2*pi*doy/365)
r_doy_cos  <- template * 0 + cos(2*pi*doy/365)

names(r_hour_sin) <- "hour_sin"
names(r_hour_cos) <- "hour_cos"
names(r_doy_sin)  <- "doy_sin"
names(r_doy_cos)  <- "doy_cos"
log_step("Cyclical time encoding done")

# ---- assemble and mask full predictor stack
full_stack <- c(
  static_stack,
  era_proj,
  r_hour_sin,
  r_hour_cos,
  r_doy_sin,
  r_doy_cos
)

full_stack <- full_stack[[feature_order]]
full_stack <- mask(full_stack, pred_mask)
gc()
log_step("Full predictor stack assembled and masked")

# ---- write full_stack to disk for workers to re-open
# terra SpatRaster pointers are not valid across spawned processes
full_stack_path <- file.path(tmp_dir, "full_stack.tif")
writeRaster(full_stack, full_stack_path,
  overwrite = TRUE,
  datatype  = "FLT4S",
  gdal      = c("COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6")
)
log_step("Full stack written to disk")

# ---- tiling configuration — 4x4 = 16 tiles, 2 workers process in parallel
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
log_step(sprintf("Tiling configured: %dx%d = %d tiles", nx, ny, nx*ny))

# free plain R preprocessing objects before spawning workers
# SpatRaster objects are left for gc() — explicit rm() of terra objects
# can invalidate C++ external pointers and cause crashes
rm(r_hour_sin, r_hour_cos, r_doy_sin, r_doy_cos, template)
gc()
log_step("Parent memory freed before cluster start")

# ---- tile prediction function
# libraries and terraOptions already set via clusterEvalQ — nothing loaded here
# PID-based stagger on first tile only to spread concurrent model loads across time
# predict to float, scale x100, store as INT2S to reduce tile file size
# INT2S range covers -327.68 to +327.67 C — sufficient for all realistic temperatures
predict_tile <- function(tile, stack_path, model_path) {
  if (tile$path == first_tile_path) {
    stagger <- (as.integer(Sys.getpid()) %% 4) * 15
    Sys.sleep(stagger)
  }
  stack <- rast(stack_path)
  mdl   <- readRDS(model_path)
  tile_ext <- ext(
    xFromCol(stack, tile$col_from) - res(stack)[1] / 2,
    xFromCol(stack, tile$col_to)   + res(stack)[1] / 2,
    yFromRow(stack, tile$row_to)   - res(stack)[2] / 2,
    yFromRow(stack, tile$row_from) + res(stack)[2] / 2
  )
  full_tile <- crop(stack, tile_ext)
  tmp_path  <- paste0(tile$path, "_float.tif")
  predict(full_tile, mdl,
    filename  = tmp_path,
    overwrite = TRUE,
    wopt      = list(datatype = "FLT4S")
  )
  r <- rast(tmp_path)
  r <- round(r * 100)
  writeRaster(r, tile$path,
    overwrite = TRUE,
    datatype  = "INT2S",
    gdal      = c("COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6")
  )
  file.remove(tmp_path)
  gc()
  return(tile$path)
}

# ---- parallel prediction
# socket cluster — only small strings exported over the socket
# worker environment fully initialised via clusterEvalQ:
#   - library paths match parent session (required on Puhti)
#   - threading env vars prevent OpenMP expansion in fresh worker sessions
#   - terra and ranger loaded once per worker
#   - each worker gets its own terra tmpdir subdirectory to avoid cross-worker
#     temp file interference when both workers initialise terra simultaneously
worker_libs     <- .libPaths()
first_tile_path <- tiles[[1]]$path
worker_tmp_1    <- file.path(tmp_dir, "worker_1")
worker_tmp_2    <- file.path(tmp_dir, "worker_2")
dir.create(worker_tmp_1, showWarnings = FALSE)
dir.create(worker_tmp_2, showWarnings = FALSE)
worker_tmps     <- c(worker_tmp_1, worker_tmp_2)

log_step("Starting cluster")
cl <- makeCluster(2)
clusterExport(cl,
  c("full_stack_path", "model_path", "worker_libs", "first_tile_path", "worker_tmps"),
  envir = environment()
)
clusterEvalQ(cl, {
  .libPaths(worker_libs)
  Sys.setenv(OMP_NUM_THREADS = 1, OPENBLAS_NUM_THREADS = 1, MKL_NUM_THREADS = 1)
  options(ranger.num.threads = 1)
  library(terra)
  library(ranger)
  # assign each worker its own terra tmpdir based on its position in the cluster
  worker_id <- as.integer(Sys.getenv("R_PARALLEL_WORKER_ID", unset = "0")) + 1
  worker_id <- max(1L, min(worker_id, length(worker_tmps)))
  terraOptions(
    memfrac  = 0.05,  # tight cap — 2 workers + parent must share job memory limit
    memmax   = 2,     # hard cap per worker in GB
    todisk   = TRUE,
    progress = 0,
    tempdir  = worker_tmps[worker_id]
  )
})
log_step("Workers initialised")

tile_files <- parLapply(cl, tiles, predict_tile,
                        stack_path = full_stack_path, model_path = model_path)
stopCluster(cl)
log_step("All tiles predicted")

tile_files <- unlist(tile_files)

# ---- mosaic tiles — merge() tolerates floating point extent differences
log_step("Starting mosaic")
tile_rasts <- lapply(tile_files, rast)
do.call(merge, c(tile_rasts, list(
  filename  = output_path,
  overwrite = TRUE,
  wopt      = list(datatype = "INT2S", gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6"))
)))
log_step("Mosaic complete")

# ---- convert back to Celsius (XX.XX) and apply final mask
result <- rast(output_path)
result <- result / 100
result <- mask(result, pred_mask)
writeRaster(result, output_path,
  overwrite = TRUE,
  datatype  = "FLT4S",
  gdal      = c("COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6")
)
log_step("Final output written")

# ---- cleanup scratch
file.remove(tmp_file)
file.remove(full_stack_path)
file.remove(tile_files)
log_step("Cleanup done — job complete")