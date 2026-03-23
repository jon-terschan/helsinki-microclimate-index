#######################
# VALIDATION PIPELINE #
#######################
# top to bottom we take the kumpula botanical garden sensors
# and beat them into a shape and form that we can predict
# at-sensor location over them and retrieve validation metrics
# as usual, the most cancerous part here is the era5-extraction
# everything before is just data prep/conditioning

# data prep#####
################
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


## step 2, sensor ID extraction#
################################
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

dt_hourly

# ERA5 extraction
##################

library(ncdf4)
library(sf)
library(data.table)
library(lubridate)

# remove any prior ERA / derived columns
drop_cols <- grep("^(t2m|ssrd|u10|v10|tp|wind_s|hour|doy)(\\.|$)", names(dt_hourly), value = TRUE)
if (length(drop_cols) > 0) {
  dt_hourly[, (drop_cols) := NULL]
}

# open NetCDF
nc <- nc_open("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/era5/era5_combined/ERA5l_SUMMER_24_25_HEL.netcdf")

lon <- ncvar_get(nc, "longitude")
lat <- ncvar_get(nc, "latitude")

time_raw  <- ncvar_get(nc, "valid_time")
era_times <- as.POSIXct(time_raw, origin = "1970-01-01", tz = "UTC")

# exact time match; drop unmatched
unique_times <- sort(unique(dt_hourly$time))
time_idx <- match(unique_times, era_times)

keep_time <- !is.na(time_idx)
unique_times <- unique_times[keep_time]
time_idx     <- time_idx[keep_time]

dt_hourly <- dt_hourly[time %in% unique_times]

# build sensor coordinates in WGS84
pts_sf  <- st_as_sf(sensor_pts, coords = c("x", "y"), crs = 3879)
pts_wgs <- st_transform(pts_sf, 4326)

coords_wgs <- as.data.table(st_coordinates(pts_wgs))
coords_wgs[, sensor_id := sensor_pts$sensor_id]
setnames(coords_wgs, c("X", "Y"), c("lon_s", "lat_s"))

# explicit nearest-index helper
nearest_idx <- function(x, grid) {
  grid[which.min(abs(grid - x))]
}

nearest_pos <- function(x, grid) {
  which.min(abs(grid - x))
}

# compute ERA grid indices from SENSOR coordinates
coords_wgs[, lon_idx := vapply(lon_s, nearest_pos, integer(1), grid = lon)]
coords_wgs[, lat_idx := vapply(lat_s, nearest_pos, integer(1), grid = lat)]

# hard checks
stopifnot(all(coords_wgs$lon_idx >= 1L), all(coords_wgs$lon_idx <= length(lon)))
stopifnot(all(coords_wgs$lat_idx >= 1L), all(coords_wgs$lat_idx <= length(lat)))

# if this fails, print diagnostics immediately
print(range(coords_wgs$lon_idx))
print(range(coords_wgs$lat_idx))
print(length(lon))
print(length(lat))

vars <- c("t2m", "ssrd", "u10", "v10", "tp")

extract_one_time <- function(ti, tval) {

  slices <- lapply(vars, function(v) {
    s <- ncvar_get(
      nc, v,
      start = c(1, 1, ti),
      count = c(-1, -1, 1)
    )
    drop(s)   # critical: convert 9x4x1 -> 9x4
  })
  names(slices) <- vars

  out <- data.table(
    sensor_id = coords_wgs$sensor_id,
    time = tval
  )

  for (v in vars) {
    slice <- slices[[v]]
    out[[v]] <- slice[cbind(coords_wgs$lon_idx, coords_wgs$lat_idx)]
  }

  out
}

era_dt <- rbindlist(
  lapply(seq_along(time_idx), function(k) {
    extract_one_time(time_idx[k], unique_times[k])
  })
)

dt_hourly <- merge(dt_hourly, era_dt, by = c("sensor_id", "time"), all.x = TRUE)

dt_hourly[, wind_s := sqrt(u10^2 + v10^2)]
dt_hourly[, hour := hour(time)]
dt_hourly[, doy  := yday(time)]

dt_hourly[, hour_sin := sin(2*pi*hour/24)]
dt_hourly[, hour_cos := cos(2*pi*hour/24)]
dt_hourly[, doy_sin  := sin(2*pi*doy/365)]
dt_hourly[, doy_cos  := cos(2*pi*doy/365)]

summary(dt_hourly$t2m)
summary(dt_hourly$ssrd)
sum(is.na(dt_hourly$t2m))
sum(is.na(dt_hourly$ssrd))

dt_hourly

saveRDS(dt_hourly, "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/validation/dt_validation.rds")

#### validation ####
####################
library(ranger)
library(data.table)

dt_hourly <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/validation/dt_validation.rds")
model <- readRDS("E:/ALS/MODEL/helmi_2000_v1.4_test.rds")

features <- model$forest$independent.variable.names
missing <- setdiff(features, names(dt_hourly))
extra   <- setdiff(names(dt_hourly), features)

missing

X <- as.data.frame(dt_hourly)[, features, drop = FALSE]
colSums(is.na(X))

# predict
pred <- predict(model, data = X)$predictions
dt_hourly[, pred := pred]

# evaluate
dt_hourly[, .(
  RMSE = sqrt(mean((observed - pred)^2)),
  MAE  = mean(abs(observed - pred)),
  R2   = cor(observed, pred)^2
)]

# baseline
dt_hourly[, .(
  RMSE_model = sqrt(mean((observed - pred)^2)),
  RMSE_era5  = sqrt(mean((observed - t2m)^2))
)]


dt_hourly[, .(
  RMSE = sqrt(mean((observed - pred)^2)),
  MAE  = mean(abs(observed - pred)),
  bias = mean(pred - observed),
  sd_err = sd(observed - pred)
), by = hour][order(hour)]

dt_hourly[, err := observed - pred]

summary(dt_hourly$err)

dt_hourly[, time_local := with_tz(time, "Europe/Helsinki")]
dt_hourly[, hour_local := hour(time_local)]
dt_hourly[, .(
  obs = mean(observed),
  pred = mean(pred)
), by = hour_local][order(hour_local)] |>
  (\(d) plot(d$hour_local, d$obs, type="l"))()

saveRDS(dt_hourly, "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/validation/dt_validation_post-pre.rds")

######validation figures#######
###############################

library(data.table)
library(lubridate)
library(ggplot2)
library(patchwork)

# PREP
dt_time <- dt_hourly[, .(
  observed = mean(observed),
  pred     = mean(pred)
), by = time]

dt_time[, err := pred - observed]
dt_time[, `:=`(
  date = as.Date(with_tz(time, "Europe/Helsinki")),
  hour = hour(with_tz(time, "Europe/Helsinki"))
)]

# daily
date_err <- dt_time[, .(
  RMSE = sqrt(mean(err^2)),
  sd_err = sd(err)
), by = date]

date_bias <- dt_time[, .(
  bias = mean(err),
  sd_err = sd(err)
), by = date]

# hourly
hour_err <- dt_time[, .(
  RMSE = sqrt(mean(err^2)),
  sd_err = sd(err)
), by = hour]

hour_bias <- dt_time[, .(
  bias = mean(err),
  sd_err = sd(err)
), by = hour]

# STYLING
col_rmse  <- "#0072B2"
fill_rmse <- "#56B4E9"

col_bias  <- "#D55E00"
fill_bias <- "#F4A582"

ylim_rmse <- c(0, max(date_err$RMSE + date_err$sd_err))
ylim_bias <- range(
  c(date_bias$bias - date_bias$sd_err,
    date_bias$bias + date_bias$sd_err)
)

base_theme <- theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold")
  )

# PLOTS

# RMSE over time
p1 <- ggplot(date_err, aes(x = date)) +
  geom_ribbon(aes(ymin = RMSE - sd_err, ymax = RMSE + sd_err),
              fill = fill_rmse, alpha = 0.15) +
  geom_line(aes(y = RMSE), color = col_rmse, linewidth = 0.8) +
  geom_smooth(aes(y = RMSE), color = col_rmse, se = FALSE, linewidth = 1) +
  geom_hline(yintercept = 1, linetype = "dotted", color = "grey50") +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  coord_cartesian(ylim = ylim_rmse) +
  labs(
    title = "Temporal evolution of RMSE",
    y = "RMSE (°C)", x = NULL
  ) +
  base_theme

# RMSE over hour
p2 <- ggplot(hour_err, aes(x = hour)) +
  geom_ribbon(aes(ymin = RMSE - sd_err, ymax = RMSE + sd_err),
              fill = fill_rmse, alpha = 0.15) +
  geom_line(aes(y = RMSE), color = col_rmse, linewidth = 1) +
  geom_hline(yintercept = 1, linetype = "dotted", color = "grey50") +
  coord_cartesian(ylim = ylim_rmse) +
  labs(
    title = "Diurnal RMSE pattern",
    y = "RMSE (°C)", x = "Hour (local time)"
  ) +
  base_theme

# Bias over time
p3 <- ggplot(date_bias, aes(x = date, y = bias)) +
  geom_ribbon(aes(ymin = bias - sd_err, ymax = bias + sd_err),
              fill = fill_bias, alpha = 0.15) +
  geom_line(color = col_bias, linewidth = 0.8) +
  geom_smooth(color = col_bias, se = FALSE, linewidth = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  coord_cartesian(ylim = ylim_bias) +
  labs(
    title = "Temporal evolution of bias",
    y = "Bias (°C)", x = NULL
  ) +
  base_theme

# Bias over hour
p4 <- ggplot(hour_bias, aes(x = hour, y = bias)) +
  geom_ribbon(aes(ymin = bias - sd_err, ymax = bias + sd_err),
              fill = fill_bias, alpha = 0.15) +
  geom_line(color = col_bias, linewidth = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  coord_cartesian(ylim = ylim_bias) +
  labs(
    title = "Diurnal bias pattern",
    y = "Bias (°C)", x = "Hour (local time)"
  ) +
  base_theme

# combine
fig <- (p1 | p2) / (p3 | p4)

# save
ggsave(
  "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/figures/validation_summary.png",
  fig,
  width = 10,
  height = 8,
  dpi = 300
)


###GITHUB DARKMODE FIGURE###
############################

library(data.table)
library(lubridate)
library(ggplot2)
library(patchwork)

# PREP
dt_time <- dt_hourly[, .(
  observed = mean(observed),
  pred     = mean(pred)
), by = time]

dt_time[, err := pred - observed]
dt_time[, `:=`(
  date = as.Date(with_tz(time, "Europe/Helsinki")),
  hour = hour(with_tz(time, "Europe/Helsinki"))
)]

date_err <- dt_time[, .(
  RMSE = sqrt(mean(err^2)),
  sd_err = sd(err)
), by = date]

hour_bias <- dt_time[, .(
  bias = mean(err),
  sd_err = sd(err)
), by = hour]

# COLORS (dark mode)
col_rmse  <- "#58A6FF"
fill_rmse <- "#58A6FF"

col_bias  <- "#F78166"
fill_bias <- "#F78166"

bg <- "#0d1117"

ylim_rmse <- c(0, max(date_err$RMSE + date_err$sd_err))
ylim_bias <- range(
  c(hour_bias$bias - hour_bias$sd_err,
    hour_bias$bias + hour_bias$sd_err)
)

# THEME
theme_github_dark <- theme_dark(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = bg, color = NA),
    panel.background = element_rect(fill = bg, color = NA),
    panel.grid.major = element_line(color = "#30363d"),
    panel.grid.minor = element_blank(),
    text             = element_text(color = "white"),
    axis.text        = element_text(color = "white"),
    axis.title       = element_text(color = "white"),
    plot.title       = element_text(face = "bold", color = "white")
  )

# ensure English month labels
Sys.setlocale("LC_TIME", "C")

# PLOTS
# RMSE over time
p1 <- ggplot(date_err, aes(x = date)) +
  geom_ribbon(aes(ymin = RMSE - sd_err, ymax = RMSE + sd_err),
              fill = fill_rmse, alpha = 0.2) +
  geom_line(aes(y = RMSE), color = col_rmse, linewidth = 1) +
  geom_smooth(aes(y = RMSE), color = col_rmse, se = FALSE, linewidth = 1.2) +
  geom_hline(yintercept = 1, linetype = "dotted", color = "#8b949e") +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  coord_cartesian(ylim = ylim_rmse) +
  labs(title = "RMSE over time", y = "RMSE (°C)", x = NULL) +
  theme_github_dark

# Bias over hour
p2 <- ggplot(hour_bias, aes(x = hour, y = bias)) +
  geom_ribbon(aes(ymin = bias - sd_err, ymax = bias + sd_err),
              fill = fill_bias, alpha = 0.2) +
  geom_line(color = col_bias, linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#8b949e") +
  coord_cartesian(ylim = ylim_bias) +
  labs(title = "Diurnal bias pattern", y = "Bias (°C)", x = "Hour (local time)") +
  theme_github_dark

# combine (side-by-side for README)
fig <- (p1 | p2) &
  theme(
    plot.background = element_rect(fill = bg, color = NA),
    panel.background = element_rect(fill = bg, color = NA)
  )
# save
ggsave(
  "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/figures/validation_summary_darkmode.png",
  fig,
  width = 10,
  height = 4,
  dpi = 300,
  bg = bg
)