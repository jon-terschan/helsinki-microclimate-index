#################################
### HELMI PREDICTION WRAPPER ###
#################################

library(terra)
library(ranger)
library(ncdf4)
library(lubridate)
library(parallel)

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

stagger_secs <- (array_id %% 5) * 30
Sys.sleep(stagger_secs)

tmp_dir <- file.path(
  "/scratch/project_2001208/Jonathan/tmp",
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

schedule <- readRDS("/scratch/project_2001208/Jonathan/model/data/processed/ML/prediction_schedule.rds")
target_time <- schedule[array_id]

log_step(sprintf("Target time: %s", target_time))

model_path <- "/scratch/project_2001208/Jonathan/model/models/rf/helmi_2000_v1.4_test.rds"

static_stack <- rast("/scratch/project_2001208/Jonathan/model/data/processed/rasters_static/pred_stack_10m.tif")

pred_mask <- rast("/scratch/project_2001208/Jonathan/model/data/processed/rasters_static/prediction_mask.tif")

feature_order <- readLines("/scratch/project_2001208/Jonathan/model/data/processed/ML/feature_order.txt")

log_step("Static inputs loaded")

nc_path <- "/scratch/project_2001208/Jonathan/model/data/processed/rasters_dynamic/ERA5l_SUMMER_24_25_HEL.netcdf"

nc <- nc_open(nc_path)

time_raw  <- ncvar_get(nc, "valid_time")
era_times <- as.POSIXct(time_raw, origin="1970-01-01", tz="UTC")

nc_index <- match(target_time, era_times)

if (is.na(nc_index)) {
  nc_close(nc)
  stop("Timestamp not found in NetCDF")
}

extract_var <- function(v) {
  ncvar_get(nc, v, start=c(1,1,nc_index), count=c(-1,-1,1))
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

mask_ext_static <- ext(pred_mask)

mask_poly_static <- as.polygons(mask_ext_static, crs=crs(pred_mask))
mask_poly_wgs84  <- project(mask_poly_static,"EPSG:4326")

era_dx <- res(era_stack)[1]
era_dy <- res(era_stack)[2]

mask_ext_wgs84 <- ext(mask_poly_wgs84)

buffered_ext <- ext(
  mask_ext_wgs84$xmin-era_dx,
  mask_ext_wgs84$xmax+era_dx,
  mask_ext_wgs84$ymin-era_dy,
  mask_ext_wgs84$ymax+era_dy
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
  gdal=c("COMPRESS=DEFLATE","PREDICTOR=2","ZLEVEL=6")
)

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
  }
}

log_step("Tiling ready")

predict_tile <- function(tile){
  cat("Worker processing:", tile$path, "\n")
  flush.console()

  if (!exists("pred_stack")) stop("Predictor stack not initialized")
  if (!exists("rf_model")) stop("Model not initialized")

  tryCatch({

    tile_ext <- ext(
      xFromCol(pred_stack, tile$col_from) - res(pred_stack)[1]/2,
      xFromCol(pred_stack, tile$col_to)   + res(pred_stack)[1]/2,
      yFromRow(pred_stack, tile$row_to)   - res(pred_stack)[2]/2,
      yFromRow(pred_stack, tile$row_from) + res(pred_stack)[2]/2
    )

    full_tile <- crop(pred_stack, tile_ext)

    p <- predict(full_tile, rf_model)

    values(p) <- round(values(p) * 100)

    writeRaster(
      p,
      tile$path,
      overwrite=TRUE,
      datatype="INT2S",
      gdal=c("COMPRESS=DEFLATE","PREDICTOR=2","ZLEVEL=6")
    )

    tile$path

  }, error=function(e){

    cat("Worker error:", conditionMessage(e), "\n")
    stop(e)

  })
}

worker_libs <- .libPaths()

worker_tmp_1 <- file.path(tmp_dir,"worker_1")
worker_tmp_2 <- file.path(tmp_dir,"worker_2")

dir.create(worker_tmp_1)
dir.create(worker_tmp_2)

worker_tmps <- c(worker_tmp_1,worker_tmp_2)

cl <- makeCluster(2,type="PSOCK")

clusterExport(
  cl,
  c("tiles","predict_tile","full_stack_path","model_path","worker_libs","worker_tmps"),
  envir=environment()
)

clusterApply(cl,1:2,function(id,libs,tmps,stack_path,model_path){

  .libPaths(libs)

  library(terra)
  library(ranger)

  terraOptions(
    memfrac=0.05,
    memmax=2,
    todisk=TRUE,
    progress=0,
    tempdir=tmps[id]
  )

  # copy stack locally to worker temp
  local_stack_path <- file.path(tmps[id],"stack_copy.tif")
  file.copy(stack_path, local_stack_path)

  pred_stack <<- rast(local_stack_path)

  rf_model <<- readRDS(model_path)

  NULL

},libs=worker_libs,
  tmps=worker_tmps,
  stack_path=full_stack_path,
  model_path=model_path)

log_step("Workers ready")

tile_files <- parLapply(cl,tiles,predict_tile)

stopCluster(cl)

tile_files <- unlist(tile_files)

log_step("Tiles predicted")

tile_rasts <- lapply(tile_files,rast)

output_path <- file.path(
  "/scratch/project_2001208/Jonathan/model/predictions",
  paste0("pred_",format(target_time,"%Y%m%d_%H%M"),".tif")
)

do.call(
  merge,
  c(tile_rasts,
    list(filename=output_path,overwrite=TRUE))
)

log_step("Mosaic complete")

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