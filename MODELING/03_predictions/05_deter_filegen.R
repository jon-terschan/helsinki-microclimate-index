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
era_path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/era5/baseline/climatology/era5land_climatology_JULY_10_18_P90.nc"

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
  paste0(output_path, "prediction_schedule_climatology_JULY_10_18_P90.rds")
)

# ----------------------------
# SANITY CHECK
# ----------------------------
cat(paste0("Total prediction timesteps: ", length(prediction_times), "\n"))
cat(paste0("First timestep: ", prediction_times[1], "\n"))
cat(paste0("Last timestep:  ", tail(prediction_times, 1), "\n"))

# TIMESTAMP DEBUGGING
nc <- nc_open(era_path)

time_vals  <- ncvar_get(nc, "valid_time")
time_units <- ncatt_get(nc, "valid_time", "units")$value
origin <- sub(".*since ", "", time_units)

nc_time <- as.POSIXct(time_vals, origin = origin, tz = "UTC")

print(nc_time)
print(prediction_times)
str(nc_time)
str(prediction_times)



# CREATE PREDICTION TARGETS FOR A LONG TIME SERIES
# ----------------------------
# SETTINGS
# ----------------------------
library(ncdf4)

era_path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/era5/era5_combined/ERA5l_SUMMER_24_25_HEL.netcdf"

output_path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/03_predictions/deterministic/schedules/"
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

months_keep <- 5:9
max_jobs <- 1000

# ----------------------------
# LOAD ERA5 TIME
# ----------------------------
nc <- nc_open(era_path)

time_vals  <- ncvar_get(nc, "valid_time")
time_units <- ncatt_get(nc, "valid_time", "units")$value

nc_close(nc)

# ----------------------------
# CONVERT TO POSIXct
# ----------------------------
origin <- sub(".*since ", "", time_units)

times <- as.POSIXct(
  time_vals,
  origin = origin,
  tz = "UTC"
)

times <- sort(unique(times))

# ----------------------------
# FILTER MONTHS (safety, even if already summer)
# ----------------------------
times <- times[as.integer(format(times, "%m")) %in% months_keep]

cat("Total timestamps after filtering:", length(times), "\n")

# ----------------------------
# (OPTIONAL) FILTER HOURS
# ----------------------------
# If you want full diurnal cycle → skip this
# Example: only 10–18 local (UTC+3)
# local_hour <- (as.integer(format(times, "%H")) + 3) %% 24
# times <- times[local_hour %in% 10:18]

# ----------------------------
# SPLIT INTO ≤1000 CHUNKS
# ----------------------------
n_chunks <- ceiling(length(times) / max_jobs)

cat("Number of chunks:", n_chunks, "\n")

split_idx <- split(
  seq_along(times),
  ceiling(seq_along(times) / max_jobs)
)

# ----------------------------
# SAVE CHUNKS
# ----------------------------
for (i in seq_along(split_idx)) {
  
  chunk_times <- times[split_idx[[i]]]
  
  out_file <- paste0(
    output_path,
    sprintf("prediction_schedule_%02d_%04d.rds", i, length(chunk_times))
  )
  
  saveRDS(chunk_times, out_file)
  
  cat("Saved:", out_file, "\n")
}

# ----------------------------
# SAVE INDEX FILE
# ----------------------------
saveRDS(
  list(
    total_steps = length(times),
    chunks = n_chunks,
    source_file = era_path
  ),
  paste0(output_path, "schedule_index.rds")
)