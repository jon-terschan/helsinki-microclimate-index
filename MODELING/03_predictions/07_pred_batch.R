library(terra) # rasters
library(ranger) # pred
library(ncdf4) # era handling
library(lubridate) # timezone bs

# array index and lookup table
schedule <- readRDS("/scratch/project_2001208/Jonathan/model/data/processed/ML/prediction_schedule.rds")
array_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))
target_time <- schedule[array_id]

# load stuff
model <- readRDS("/scratch/project_2001208/Jonathan/model/models/rf/helmi_2000_v1.4_test.rds") # helmi
static_stack <- rast("/scratch/project_2001208/Jonathan/model/data/processed/rasters_static/pred_stack_10m.tif") # predstack
feature_order <- readLines("/scratch/project_2001208/Jonathan/model/data/processed/ML/feature_order.txt") # deterministic feature order

# open era netcdf
nc_path <- "/scratch/project_2001208/Jonathan/model/data/processed/rasters_dynamic/ERA5l_SUMMER_24_25_HEL.netcdf"
nc <- nc_open(nc_path)

# extract time from era
time_raw <- ncvar_get(nc, "valid_time")
era_times <- as.POSIXct(time_raw,
                        origin="1970-01-01",
                        tz="UTC")

nc_index <- match(target_time, era_times)
if (is.na(nc_index)) stop("Timestamp not found in NetCDF")

# output folder
base_dir <- "/scratch/project_2001208/Jonathan/model/predictions/10m"
day_folder <- format(target_time, "%Y%m%d")
day_path <- file.path(base_dir, day_folder)

# create day folder
if (!dir.exists(day_path)) {
  dir.create(day_path, recursive = TRUE)
}

# output for predictions
output_path <- file.path(
  day_path,
  paste0("pred_", format(target_time, "%Y%m%d_%H%M"), ".tif")
)

# extract era variables for one timestamp
extract_var <- function(varname, ts_index) {
  ncvar_get(nc, varname,
            start = c(1, 1, ts_index),
            count = c(-1, -1, 1))
}

t2m   <- extract_var("t2m", nc_index)
ssrd  <- extract_var("ssrd", nc_index)
u10   <- extract_var("u10", nc_index)
v10   <- extract_var("v10", nc_index)
tp    <- extract_var("tp", nc_index)
wind  <- extract_var("wind_s", nc_index)

# build terra raster template
lon <- ncvar_get(nc, "longitude")
lat <- ncvar_get(nc, "latitude")

dx <- abs(lon[2] - lon[1])
dy <- abs(lat[2] - lat[1])

xmin <- min(lon) - dx/2
xmax <- max(lon) + dx/2
ymin <- min(lat) - dy/2
ymax <- max(lat) + dy/2

make_raster <- function(slice) {
  slice2 <- aperm(slice, c(2,1))  # lat x lon (4 x 9)
  r <- rast(
    nrows = length(lat),
    ncols = length(lon),
    xmin = xmin,
    xmax = xmax,
    ymin = ymin,
    ymax = ymax,
    crs  = "EPSG:4326"
  )
  values(r) <- as.vector(t(slice2[nrow(slice2):1, ])) # correct coordinate permutation
  r
}

r_t2m  <- make_raster(t2m)
r_ssrd <- make_raster(ssrd)
r_u10  <- make_raster(u10)
r_v10  <- make_raster(v10)
r_tp   <- make_raster(tp)
r_ws   <- make_raster(wind)

era_stack <- c(r_t2m, r_ssrd, r_u10, r_v10, r_tp, r_ws)
names(era_stack) <- c("t2m","ssrd","u10","v10","tp","wind_s")

# reproject to pred stack
era_proj <- project(era_stack, static_stack, method = "bilinear")

# cyclical encoding of time stuff
hour <- hour(target_time)
doy  <- yday(target_time)
hour_sin <- sin(2 * pi * hour / 24)
hour_cos <- cos(2 * pi * hour / 24)
doy_sin  <- sin(2 * pi * doy / 365)
doy_cos  <- cos(2 * pi * doy / 365)
template <- static_stack[[1]]
r_hour_sin <- setValues(template, hour_sin)
r_hour_cos <- setValues(template, hour_cos)
r_doy_sin  <- setValues(template, doy_sin)
r_doy_cos  <- setValues(template, doy_cos)
names(r_hour_sin) <- "hour_sin"
names(r_hour_cos) <- "hour_cos"
names(r_doy_sin)  <- "doy_sin"
names(r_doy_cos)  <- "doy_cos"

# combine stacks
full_stack <- c(static_stack, era_proj,
                r_hour_sin, r_hour_cos,
                r_doy_sin,  r_doy_cos)
full_stack <- full_stack[[feature_order]]

# predict
pred <- predict(
  full_stack,
  model,
  filename = output_path,
  overwrite = TRUE
)

nc_close(nc)