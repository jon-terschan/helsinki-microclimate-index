library(terra)
library(ranger)
library(ncdf4)
library(lubridate)

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
options(ranger.num.threads = 1)

Sys.setenv(
  OMP_NUM_THREADS = 1,
  OPENBLAS_NUM_THREADS = 1,
  MKL_NUM_THREADS = 1
)

log_mem("Start script")

# ---- SLURM identifiers (define ONCE) ----
array_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))
job_id   <- Sys.getenv("SLURM_JOB_ID")

tmp_dir <- file.path(
  "/scratch/project_2001208/Jonathan/tmp",
  paste0("job_", job_id, "_", array_id)
)

dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)

terraOptions(
  memfrac = 0.2,     # 5% of available RAM
  memmax  = 4,        # hard cap in GB
  todisk  = TRUE,
  progress = 1,
  tempdir = tmp_dir
)

# ---- Prediction schedule ----
schedule <- readRDS("/scratch/project_2001208/Jonathan/model/data/processed/ML/prediction_schedule.rds")
target_time <- schedule[array_id]

# ---- Load model and static predictors ----
model <- readRDS("/scratch/project_2001208/Jonathan/model/models/rf/helmi_2000_v1.4_test.rds")
static_stack <- rast("/scratch/project_2001208/Jonathan/model/data/processed/rasters_static/pred_stack_10m.tif")
pred_mask <- rast("/scratch/project_2001208/Jonathan/model/data/processed/rasters_static/prediction_mask.tif")
feature_order <- readLines("/scratch/project_2001208/Jonathan/model/data/processed/ML/feature_order.txt")
log_mem("Loaded static stack")
# ---- Open ERA NetCDF ----
nc_path <- "/scratch/project_2001208/Jonathan/model/data/processed/rasters_dynamic/ERA5l_SUMMER_24_25_HEL.netcdf"
nc <- nc_open(nc_path)

time_raw <- ncvar_get(nc, "valid_time")
era_times <- as.POSIXct(time_raw, origin="1970-01-01", tz="UTC")

nc_index <- match(target_time, era_times)
if (is.na(nc_index)) stop("Timestamp not found in NetCDF")

# ---- Output path ----
base_dir <- "/scratch/project_2001208/Jonathan/model/predictions/10m"
day_path <- file.path(base_dir, format(target_time, "%Y%m%d"))
dir.create(day_path, recursive = TRUE, showWarnings = FALSE)

output_path <- file.path(
  day_path,
  paste0("pred_", format(target_time, "%Y%m%d_%H%M"), ".tif")
)

# ---- Extract ERA slice ----
extract_var <- function(varname, ts_index) {
  ncvar_get(nc, varname, start = c(1,1,ts_index), count = c(-1,-1,1))
}

t2m  <- extract_var("t2m", nc_index)
ssrd <- extract_var("ssrd", nc_index)
u10  <- extract_var("u10", nc_index)
v10  <- extract_var("v10", nc_index)
tp   <- extract_var("tp", nc_index)
wind <- extract_var("wind_s", nc_index)

log_mem("Extracted ERA slice")
lon <- ncvar_get(nc, "longitude")
lat <- ncvar_get(nc, "latitude")

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

# ---- Disk-backed reprojection ----
tmp_file <- file.path(tmp_dir, "era_proj.tif")

era_proj <- project(
  era_stack,
  static_stack,
  method = "bilinear",
  filename = tmp_file,
  overwrite = TRUE,
  wopt = list(datatype = "FLT4S")
)
log_mem("After projection")
# free ERA in-memory objects
rm(era_stack, t2m, ssrd, u10, v10, tp, wind)
gc()

# ---- Cyclical encoding (disk-safe constant rasters) ----
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

# ---- Prediction ----
# ---- Spatial tiling prediction ----
# ---- Spatial tiling configuration ----
nx <- 4
ny <- 4

ncols_total <- ncol(static_stack)
nrows_total <- nrow(static_stack)

col_breaks <- round(seq(1, ncols_total + 1, length.out = nx + 1))
row_breaks <- round(seq(1, nrows_total + 1, length.out = ny + 1))

tiles <- vector("list", nx * ny)
k <- 1

for (ix in 1:nx) {
  for (iy in 1:ny) {
	tiles[[k]] <- ext(
  		static_stack,
  		row_breaks[iy],
  		row_breaks[iy + 1] - 1,
  		col_breaks[ix],
  		col_breaks[ix + 1] - 1
		)
    k <- k + 1
  }
}

tile_files <- character(length(tiles))

# predict loop
for (i in seq_along(tiles)) {

  cat("Processing tile", i, "of", length(tiles), "\n")

  tile_ext <- tiles[[i]]

  # Crop predictors to tile
  full_tile <- crop(full_stack, tile_ext)

  tile_path <- file.path(tmp_dir, paste0("tile_", i, ".tif"))
  tile_files[i] <- tile_path

  predict(
    full_tile,
    model,
    filename  = tile_path,
    overwrite = TRUE,
    wopt = list(
      datatype = "FLT4S",
      gdal = c("COMPRESS=LZW")
    )
  )

  rm(full_tile)
  gc()
}

log_mem("All tiles predicted")

# ---- Mosaic tiles ----
mosaic(
  rast(tile_files),
  filename = output_path,
  overwrite = TRUE,
  wopt = list(
    datatype = "FLT4S",
    gdal = c("COMPRESS=LZW")
  )
)

log_mem("After mosaic")
# ---- Cleanup ----

file.remove(tmp_file)
nc_close(nc)