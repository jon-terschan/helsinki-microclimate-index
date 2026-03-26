#################################
### HELMI PREDICTION WRAPPER ###
### SINGLE-THREADED VERSION   ###
#################################

library(terra)
library(ranger)
library(ncdf4)
library(lubridate)

log_step <- function(msg) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), msg))
  flush.console()
}

options(ranger.num.threads = 1)

Sys.setenv(
  OMP_NUM_THREADS      = 1,
  OPENBLAS_NUM_THREADS = 1,
  MKL_NUM_THREADS      = 1
)

array_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))
job_id   <- Sys.getenv("SLURM_JOB_ID")

log_step(sprintf("Array task %d started (job %s)", array_id, job_id))

set.seed(array_id + as.integer(Sys.getpid()))
Sys.sleep(runif(1,0,90))
Sys.sleep((array_id %% 5) * 30)

#########################################################
# Temporary directory
#########################################################

tmp_dir <- file.path(
  "/scratch/project_2001208/Jonathan/model/tmp",
  paste0("job_", job_id, "_", array_id)
)

dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)

terraOptions(
  memfrac  = 0.1,
  memmax   = 3,
  todisk   = TRUE,
  progress = 0,
  tempdir  = tmp_dir
)

job_success <- FALSE

on.exit({
  if (job_success) {
    log_step("Job completed successfully, removing tmp directory")
    unlink(tmp_dir, recursive = TRUE, force = TRUE)
  } else {
    log_step("Job failed, tmp directory retained")
  }
}, add = TRUE)

#########################################################
# Prediction schedule
#########################################################

schedule <- readRDS("/scratch/project_2001208/Jonathan/model/data/processed/ML/prediction_schedule_H3.rds")
target_time <- schedule[array_id]

log_step(sprintf("Target time: %s", target_time))

#########################################################
# Static inputs
#########################################################

model_path <- "/scratch/project_2001208/Jonathan/model/models/rf/helmi_2000_v1.4_test.rds"

static_stack <- rast("/scratch/project_2001208/Jonathan/model/data/processed/rasters_static/pred_stack_10m.tif")
pred_mask    <- rast("/scratch/project_2001208/Jonathan/model/data/processed/rasters_static/prediction_mask.tif")

feature_order <- readLines("/scratch/project_2001208/Jonathan/model/data/processed/ML/feature_order.txt")

rf_model <- readRDS(model_path)

log_step("Static inputs loaded")

#########################################################
# ERA5 NetCDF extraction
#########################################################

nc_path <- "/scratch/project_2001208/Jonathan/model/data/processed/rasters_dynamic/ERA5L_H3_pre.nc" 

nc <- nc_open(nc_path)

time_raw  <- ncvar_get(nc,"valid_time")
era_times <- as.POSIXct(time_raw, origin="1970-01-01", tz="UTC")

nc_index <- match(target_time,era_times)

if (is.na(nc_index)) {
  nc_close(nc)
  stop("Timestamp not found in NetCDF")
}

extract_var <- function(v){
  ncvar_get(nc,v,start=c(1,1,nc_index),count=c(-1,-1,1))
}

t2m  <- extract_var("t2m")
ssrd <- extract_var("ssrd")
u10  <- extract_var("u10")
v10  <- extract_var("v10")
tp   <- extract_var("tp")
wind <- extract_var("wind_s")

lon <- ncvar_get(nc,"longitude")
lat <- ncvar_get(nc,"latitude")

nc_close(nc)

log_step("ERA5 extracted")

#########################################################
# Convert ERA slices to rasters
#########################################################

dx <- abs(lon[2]-lon[1])
dy <- abs(lat[2]-lat[1])

xmin <- min(lon)-dx/2
xmax <- max(lon)+dx/2
ymin <- min(lat)-dy/2
ymax <- max(lat)+dy/2

make_raster <- function(slice){

  slice2 <- aperm(slice,c(2,1))

  r <- rast(
    nrows=length(lat),
    ncols=length(lon),
    xmin=xmin,xmax=xmax,
    ymin=ymin,ymax=ymax,
    crs="EPSG:4326"
  )

  values(r) <- as.vector(t(slice2[nrow(slice2):1,]))

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

log_step("ERA stack built")

#########################################################
# Crop + project ERA
#########################################################

mask_ext_static <- ext(pred_mask)

mask_poly_static <- as.polygons(mask_ext_static,crs=crs(pred_mask))
mask_poly_wgs84  <- project(mask_poly_static,"EPSG:4326")

era_dx <- res(era_stack)[1]
era_dy <- res(era_stack)[2]

mask_ext_wgs84 <- ext(mask_poly_wgs84)

buffered_ext <- ext(
  mask_ext_wgs84$xmin-2*era_dx,
  mask_ext_wgs84$xmax+2*era_dx,
  mask_ext_wgs84$ymin-2*era_dy,
  mask_ext_wgs84$ymax+2*era_dy
)

era_cropped <- crop(era_stack,buffered_ext)

tmp_file <- file.path(tmp_dir,"era_proj.tif")

era_proj <- project(
  era_cropped,
  static_stack,
  method="bilinear",
  filename=tmp_file,
  overwrite=TRUE
)

log_step("ERA reprojected")

#########################################################
# Time features
#########################################################

template <- static_stack[[1]]

hour <- hour(target_time)
doy  <- yday(target_time)

r_hour_sin <- template*0 + sin(2*pi*hour/24)
r_hour_cos <- template*0 + cos(2*pi*hour/24)
r_doy_sin  <- template*0 + sin(2*pi*doy/365)
r_doy_cos  <- template*0 + cos(2*pi*doy/365)

names(r_hour_sin) <- "hour_sin"
names(r_hour_cos) <- "hour_cos"
names(r_doy_sin)  <- "doy_sin"
names(r_doy_cos)  <- "doy_cos"

#########################################################
# Build full predictor stack (disk-backed)
#########################################################

full_stack <- c(
  static_stack,
  era_proj,
  r_hour_sin,
  r_hour_cos,
  r_doy_sin,
  r_doy_cos
)

full_stack <- full_stack[[feature_order]]
full_stack <- mask(full_stack,pred_mask)

log_step("Full predictor stack built")

full_stack_path <- file.path(tmp_dir,"full_stack.tif")

writeRaster(
  full_stack,
  full_stack_path,
  overwrite=TRUE,
  datatype="FLT4S",
  gdal=c("COMPRESS=NONE")
)

# reload as disk-backed raster
full_stack <- rast(full_stack_path)

#########################################################
# Tile definition
#########################################################

nx <- 4
ny <- 4

ncols_total <- ncol(full_stack)
nrows_total <- nrow(full_stack)

col_breaks <- round(seq(1,ncols_total+1,length.out=nx+1))
row_breaks <- round(seq(1,nrows_total+1,length.out=ny+1))

tiles <- list()
k <- 1

for (iy in 1:ny) {
for (ix in 1:nx) {

tiles[[k]] <- list(
row_from=row_breaks[iy],
row_to=row_breaks[iy+1]-1,
col_from=col_breaks[ix],
col_to=col_breaks[ix+1]-1,
path=file.path(tmp_dir,paste0("tile_",k,".tif"))
)

k <- k+1
}}

log_step("Tiling ready")

#########################################################
# Sequential prediction
#########################################################

predict_tile <- function(tile){

  tile_ext <- ext(
    xFromCol(full_stack,tile$col_from)-res(full_stack)[1]/2,
    xFromCol(full_stack,tile$col_to)+res(full_stack)[1]/2,
    yFromRow(full_stack,tile$row_to)-res(full_stack)[2]/2,
    yFromRow(full_stack,tile$row_from)+res(full_stack)[2]/2
  )

  full_tile <- crop(full_stack,tile_ext)

  p <- predict(full_tile,rf_model)

  values(p) <- round(values(p)*100)

  writeRaster(
    p,
    tile$path,
    overwrite=TRUE,
    datatype="INT2S",
    gdal=c("COMPRESS=DEFLATE","PREDICTOR=2","ZLEVEL=6")
  )

  tile$path
}

tile_files <- lapply(tiles,predict_tile)
tile_files <- unlist(tile_files)

log_step("Tiles predicted")

tile_rasts <- lapply(tile_files,rast)

#########################################################
# Output directory
#########################################################

prediction_root <- "/scratch/project_2001208/Jonathan/model/predictions"

day_folder <- file.path(
prediction_root,
format(target_time,"%Y%m%d")
)

if(!dir.exists(day_folder)){
dir.create(day_folder,recursive=TRUE)
}

output_path <- file.path(
day_folder,
paste0("pred_",format(target_time,"%Y%m%d_%H%M"),".tif")
)

#########################################################
# Mosaic tiles
#########################################################

do.call(
merge,
c(tile_rasts,list(filename=output_path,overwrite=TRUE))
)

log_step("Mosaic complete")

#########################################################
# Final scaling + masking
#########################################################

result <- rast(output_path)

result <- result/100
result <- mask(result,pred_mask)

writeRaster(
result,
output_path,
overwrite=TRUE,
datatype="FLT4S",
gdal=c("COMPRESS=DEFLATE","PREDICTOR=2","ZLEVEL=6")
)

log_step("Final output written")
job_success <- TRUE

#########################################################
# Cleanup
#########################################################

if (job_success) {
  log_step("Job completed successfully, removing tmp directory")
  unlink(tmp_dir, recursive = TRUE, force = TRUE)
} else {
  log_step("Job did not complete successfully, tmp directory retained for debugging")
}