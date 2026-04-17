library(lubridate)
output_path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/03_predictions/deterministic/"

# PREDICTOR ORDER FOR HPC
model <- readRDS("E:/ALS/MODEL/helmi_2000_v1.4_test.rds")
feature_order <- model$forest$independent.variable.names

writeLines(feature_order, paste0(output_path, "feature_order.txt") )

# LIST OF PREDICTION TARGETS
start_time <- ymd_hms("2024-06-01 01:00:00", tz = "UTC")
end_time   <- ymd_hms("2024-06-01 23:00:00", tz = "UTC")

prediction_times <- seq(
  from = start_time,
  to   = end_time,
  by   = "1 hour"
)
saveRDS(prediction_times, paste0(output_path, "prediction_schedule.rds"))

#----
cat(paste0("Correct array size: ", length(prediction_times), "\nMake sure it matches the SLURM --array upper bound.\n"))


#### CREATE PREDICTION TARGETS FOR HEATWAVE EVENTS BASED ON HEATWAVE DATA.
library(ncdf4)
era_path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/era5/heatwaves_hourly/preprocessed/ERA5L_H2_corrected_pre.nc"
nc <- nc_open(era_path)

time_vals <- ncvar_get(nc, "valid_time")
time_units <- ncatt_get(nc, "valid_time", "units")$value

# ----------------------------------------
# CONVERT TO POSIXct
# ----------------------------------------
time_units <- ncatt_get(nc, "valid_time", "units")$value
origin <- sub(".*since ", "", time_units)

prediction_times <- as.POSIXct(time_vals,
                               origin = "1970-01-01",
                               tz = "UTC")
                            
prediction_times <- sort(unique(prediction_times))

# SAVE
saveRDS(prediction_times, paste0(output_path, "prediction_schedule_H2_corrected.rds"))

# SANITY CHECK
cat(paste0("Total prediction timesteps: ", length(prediction_times), "\n"))
cat(paste0("First timestep: ", prediction_times[1], "\n"))
cat(paste0("Last timestep:  ", tail(prediction_times, 1), "\n"))
cat("Make sure this matches SLURM --array upper bound.\n")

#### CREATE PREDICTION TARGETS FOR BASELINE SCENARIO (DUMMY)
# version 2, now for the multiple baseline hours
# ----------------------------
# PATHS
# ----------------------------
output_path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/03_predictions/deterministic/"
era_path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/era5/baseline/climatology/era5land_climatology_JULY_10_18.nc"

library(ncdf4)

# ----------------------------
# LOAD NETCDF
# ----------------------------
nc <- nc_open(era_path)

time_vals  <- ncvar_get(nc, "valid_time")
time_units <- ncatt_get(nc, "valid_time", "units")$value

# ----------------------------
# CONVERT TO POSIXct (same logic)
# ----------------------------
origin <- sub(".*since ", "", time_units)

prediction_times <- as.POSIXct(
  time_vals,
  origin = origin,
  tz = "UTC"
)

# ensure consistency with your pipeline
prediction_times <- sort(unique(prediction_times))

# ----------------------------
# SAVE
# ----------------------------
saveRDS(
  prediction_times,
  paste0(output_path, "prediction_schedule_climatology_JULY_10_18.rds")
)

# ----------------------------
# SANITY CHECK
# ----------------------------
cat(paste0("Total prediction timesteps: ", length(prediction_times), "\n"))
cat(paste0("First timestep: ", prediction_times[1], "\n"))
cat(paste0("Last timestep:  ", tail(prediction_times, 1), "\n"))


nc <- nc_open(era_path)

time_vals  <- ncvar_get(nc, "valid_time")
time_units <- ncatt_get(nc, "valid_time", "units")$value
origin <- sub(".*since ", "", time_units)

nc_time <- as.POSIXct(time_vals, origin = origin, tz = "UTC")

print(nc_time)
print(prediction_times)
str(nc_time)
str(prediction_times)
