library(dplyr)
library(readr)

# ----------------------------
# INPUT / OUTPUT
# ----------------------------
path_in  <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/01_traindataprep/06_train_data.rds"
out_dir  <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/figures/r1_figure"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

local_tz <- "Europe/Helsinki"

# ----------------------------
# LOAD DATA
# ----------------------------
df <- readRDS(path_in)

if (!("time" %in% names(df))) stop("Column 'time' not found.")
if (!("temp" %in% names(df))) stop("Column 'temp' not found.")

# Try to detect a sensor/site ID column
sensor_candidates <- c("sensor_id", "site_id", "station_id", "plot_id", "id", "sensor")
sensor_col <- intersect(sensor_candidates, names(df))
sensor_col <- if (length(sensor_col) > 0) sensor_col[1] else NA_character_

if (is.na(sensor_col)) {
  message("No obvious sensor ID column found; n sites will be NA.")
} else {
  message("Using sensor ID column: ", sensor_col)
}

# ----------------------------
# KEEP ORIGINAL TIME IN UTC
# ----------------------------
df <- df %>%
  mutate(
    time_utc = as.POSIXct(time, tz = "UTC"),
    month_utc = format(time_utc, "%m")
  )

# July subset based on UTC timestamps
df_july <- df %>%
  filter(month_utc == "07") %>%
  mutate(
    # Ad hoc local-time variables for reporting only
    time_local = as.POSIXct(format(time_utc, tz = local_tz, usetz = TRUE), tz = local_tz),
    hour_local = as.integer(format(time_local, "%H")),
    date_local = as.Date(time_local)
  )

# ----------------------------
# SUMMARY VALUES FOR REPORTING
# ----------------------------
n_observations <- nrow(df_july)

n_sites <- if (is.na(sensor_col)) {
  NA_integer_
} else {
  n_distinct(df_july[[sensor_col]], na.rm = TRUE)
}

date_start_local <- format(min(df_july$time_local, na.rm = TRUE), tz = local_tz, usetz = TRUE)
date_end_local   <- format(max(df_july$time_local, na.rm = TRUE), tz = local_tz, usetz = TRUE)

temp_min <- min(df_july$temp, na.rm = TRUE)
temp_max <- max(df_july$temp, na.rm = TRUE)

# Hourly mean profile in LOCAL TIME for reporting
hourly_profile_local <- df_july %>%
  group_by(hour_local) %>%
  summarise(
    mean_temp = mean(temp, na.rm = TRUE),
    sd_temp   = sd(temp, na.rm = TRUE),
    n         = n(),
    .groups   = "drop"
  ) %>%
  arrange(hour_local)

peak_row <- hourly_profile_local[which.max(hourly_profile_local$mean_temp), ]
peak_hour_local <- peak_row$hour_local
peak_time_local <- sprintf("%02d:00 %s", peak_hour_local, "local")

# Daylight statistics in LOCAL TIME
daylight_local <- df_july %>%
  filter(hour_local >= 10, hour_local <= 18)

daylight_mean <- mean(daylight_local$temp, na.rm = TRUE)
daylight_sd   <- sd(daylight_local$temp, na.rm = TRUE)

# ----------------------------
# PRINT REPORT-READY OUTPUT
# ----------------------------
cat("\n--- JULY TRAINING DATA SUMMARY (LOCAL REPORTING TIME) ---\n")
cat("n observations:", n_observations, "\n")
cat("n sites:", n_sites, "\n")
cat("Date range:", date_start_local, "to", date_end_local, "\n")
cat("Hourly peak time:", peak_time_local, "\n")
cat("Temperature range:", round(temp_min, 2), "to", round(temp_max, 2), "\n")
cat("Daylight mean temperature (10–18 local):",
    round(daylight_mean, 2), "+/-", round(daylight_sd, 2), "\n")

# ----------------------------
# EXPORT FOR PYTHON
# ----------------------------
# Keep the raw time variable in UTC; local columns are only derived helpers.
# If you prefer absolutely no local columns in the exported data, drop them here.
write_csv(
  df_july %>% select(-time_local),
  file.path(out_dir, "train_data_july_utc.csv")
)

# Hourly profile already aggregated in local time for plotting/reporting
write_csv(hourly_profile_local, file.path(out_dir, "july_hourly_profile_local.csv"))

# One-line summary table
summary_df <- tibble::tibble(
  n_observations = n_observations,
  n_sites = n_sites,
  date_start_local = date_start_local,
  date_end_local = date_end_local,
  peak_time_local = peak_time_local,
  temp_min = round(temp_min, 3),
  temp_max = round(temp_max, 3),
  daylight_mean_local_10_18 = round(daylight_mean, 3),
  daylight_sd_local_10_18 = round(daylight_sd, 3)
)

write_csv(summary_df, file.path(out_dir, "july_summary_local.csv"))

cat("\nFiles written to:\n", out_dir, "\n")