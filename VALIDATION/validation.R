# data prep
library(sf)
library(data.table)
library(lubridate)

files <- list.files("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/botanical_garden/", full.names = TRUE)

get_id <- function(f) sub("data_([0-9]+)_.*", "\\1", basename(f))

dt_list <- lapply(files, function(f) {

  d <- fread(f, sep = ";", header = FALSE, colClasses = "character")
  
  setnames(d, c(
    "rec_id", "timestamp", "tz",
    "soil_temp", "surface_temp", "air_temp",
    paste0("extra_", 1:(ncol(d)-6))
  ))
  d[, tz := as.numeric(tz)]
  d[, air_temp := as.numeric(air_temp)]
  d <- d[, .(timestamp, air_temp)]
  d[, sensor_id := get_id(f)]

  d
})

dt <- rbindlist(dt_list)
dt <- unique(dt, by = c("sensor_id", "timestamp"))
dt[, time := ymd_hm(timestamp, tz = "UTC")]

dt <- dt[
  time >= as.POSIXct("2024-05-01 00:00:00", tz = "UTC") &
  time <= as.POSIXct("2024-09-30 23:59:59", tz = "UTC")
]

dt[, time_hour := floor_date(time, "hour")]

dt_hourly <- dt[, .(
  observed = mean(air_temp, na.rm = TRUE)
), by = .(sensor_id, time = time_hour)]


## step 2, sensor ID extraction

coords_sf <- st_read("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/VALIDATION/botanical_sensors.gpkg")

coords_sf <- st_transform(coords_sf, 3879)  # ensure city of helsinki crs

coords_sf$x <- st_coordinates(coords_sf)[,1]
coords_sf$y <- st_coordinates(coords_sf)[,2]

coords <- coords_sf[, c("Desc.", "x", "y")]

setnames(coords, "Desc.", "sensor_id")
head(coords$sensor_id)
str(coords)


# merge with hourly
dt_hourly <- merge(dt_hourly, coords, by = "sensor_id")
nrow(dt_hourly)
dt_hourly[is.na(x) | is.na(y)]
plot(dt_hourly$x, dt_hourly$y)
dt_hourly


dt_hourly[, geom := NULL]
dt_hourly <- dt_hourly[!is.nan(observed)]


# sensor level table
sensor_pts <- unique(dt_hourly[, .(sensor_id, x, y)])
sensor_pts


# static predictor extrac
library(terra)

r <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/full_stack/pred_stack_10m.tif")

pts <- vect(sensor_pts, geom = c("x", "y"), crs = "EPSG:3879")
vals <- terra::extract(r, pts)
vals <- vals[, -1]

sensor_static <- cbind(sensor_pts, vals)

dt_hourly <- merge(dt_hourly, sensor_static, by = c("sensor_id", "x", "y"))


# era5 extrac

# validation 