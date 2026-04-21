library(dplyr)
library(sf)

# ----------------------------
# LOAD
# ----------------------------
path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/01_traindataprep/06_train_data.rds"
df <- readRDS(path)

# ----------------------------
# TIME HANDLING (NO CONVERSION)
# ----------------------------
df <- df %>%
  mutate(
    hour = as.integer(format(time, "%H")),
    month = format(time, "%m")
  )

# ----------------------------
# FILTER: JULY (NO HOUR FILTER YET)
# ----------------------------
df_july <- df %>%
  filter(month == "07")

cat("\n--- DATA SUMMARY (JULY) ---\n")
cat("Rows:", nrow(df_july), "\n")

# ----------------------------
# CHECK MISSING
# ----------------------------
na_summary <- sapply(df_july, function(x) sum(is.na(x)))
if (sum(na_summary) == 0) {
  cat("Missing values: NONE\n")
} else {
  print(sort(na_summary, decreasing = TRUE))
}

# ----------------------------
# 1. OVERALL STATS (JULY)
# ----------------------------
overall <- df_july %>%
  summarise(
    mean = mean(temp),
    sd   = sd(temp),
    min  = min(temp),
    max  = max(temp),
    p05  = quantile(temp, 0.05),
    p50  = quantile(temp, 0.50),
    p95  = quantile(temp, 0.95)
  ) %>%
  st_drop_geometry()

cat("\n--- JULY TEMPERATURE ---\n")
print(overall)

# ----------------------------
# 2. FULL DIURNAL (RAW HOURS)
# ----------------------------
diurnal <- df_july %>%
  group_by(hour) %>%
  summarise(
    mean = mean(temp),
    sd   = sd(temp),
    .groups = "drop"
  ) %>%
  arrange(hour)

peak_hour <- diurnal$hour[which.max(diurnal$mean)]
peak_temp <- max(diurnal$mean)

min_hour  <- diurnal$hour[which.min(diurnal$mean)]
min_temp  <- min(diurnal$mean)

amplitude <- peak_temp - min_temp

cat("\n--- DIURNAL (RAW TIME) ---\n")
cat("Min:", round(min_temp,2), "at", min_hour, "\n")
cat("Max:", round(peak_temp,2), "at", peak_hour, "\n")
cat("Amplitude:", round(amplitude,2), "\n")

# ----------------------------
# 3. ENVIRONMENTAL CLASSES
# ----------------------------
df_july <- df_july %>%
  mutate(
    class = case_when(
      tree_fr_10 > nwn_fr_10 & tree_fr_10 > imp_fr_10 ~ "tree",
      nwn_fr_10 > imp_fr_10 ~ "nwn",
      TRUE ~ "urban"
    )
  )

class_stats <- df_july %>%
  group_by(class) %>%
  summarise(
    mean = mean(temp),
    sd   = sd(temp),
    n    = n(),
    .groups = "drop"
  )

cat("\n--- CLASS STATS (ALL HOURS) ---\n")
print(class_stats)

tree_mean  <- class_stats$mean[class_stats$class == "tree"]
urban_mean <- class_stats$mean[class_stats$class == "urban"]
nwn_mean   <- class_stats$mean[class_stats$class == "nwn"]

cat("Tree - Urban:", round(tree_mean - urban_mean, 3), "\n")
cat("Tree - NWN:", round(tree_mean - nwn_mean, 3), "\n")

# ----------------------------
# 4. DAYTIME FILTER (RAW HOURS)
# ----------------------------
df_day <- df_july %>%
  filter(hour >= 10, hour <= 18)

cat("\n--- DAYTIME DATA ---\n")
cat("Rows:", nrow(df_day), "\n")

# ----------------------------
# 5. DAYTIME STATS
# ----------------------------
overall_day <- df_day %>%
  summarise(
    mean = mean(temp),
    sd   = sd(temp),
    p05  = quantile(temp, 0.05),
    p50  = quantile(temp, 0.50),
    p95  = quantile(temp, 0.95)
  ) %>%
  st_drop_geometry()

cat("\n--- DAYTIME TEMPERATURE ---\n")
print(overall_day)

# ----------------------------
# 6. DAYTIME DIURNAL
# ----------------------------
diurnal_day <- df_day %>%
  group_by(hour) %>%
  summarise(
    mean = mean(temp),
    sd   = sd(temp),
    .groups = "drop"
  ) %>%
  arrange(hour)

peak_hour_day <- diurnal_day$hour[which.max(diurnal_day$mean)]
peak_temp_day <- max(diurnal_day$mean)

cat("\n--- DAYTIME DIURNAL ---\n")
cat("Peak:", round(peak_temp_day,2), "at", peak_hour_day, "\n")

# ----------------------------
# 7. DAYTIME CLASS EFFECTS
# ----------------------------
class_stats_day <- df_day %>%
  group_by(class) %>%
  summarise(
    mean = mean(temp),
    sd   = sd(temp),
    n    = n(),
    .groups = "drop"
  )

cat("\n--- CLASS STATS (DAYTIME) ---\n")
print(class_stats_day)

tree_mean_d  <- class_stats_day$mean[class_stats_day$class == "tree"]
urban_mean_d <- class_stats_day$mean[class_stats_day$class == "urban"]
nwn_mean_d   <- class_stats_day$mean[class_stats_day$class == "nwn"]

cat("Tree - Urban:", round(tree_mean_d - urban_mean_d, 3), "\n")
cat("Tree - NWN:", round(tree_mean_d - nwn_mean_d, 3), "\n")


# ----------------------------
# 8. FINAL SUMMARY (REPORT-READY)
# ----------------------------

cat("\n============================\n")
cat("FINAL SUMMARY\n")
cat("============================\n")

# FULL DAY
cat("\n--- FULL JULY (ALL HOURS) ---\n")
cat("Mean:", round(overall$mean, 2), "\n")
cat("SD:", round(overall$sd, 2), "\n")
cat("Min:", round(min_temp, 2), "at", min_hour, "\n")
cat("Max:", round(peak_temp, 2), "at", peak_hour, "\n")
cat("Amplitude:", round(amplitude, 2), "\n")

cat("\nClass differences:\n")
cat("Tree - Urban:", round(tree_mean - urban_mean, 3), "\n")
cat("Tree - NWN:", round(tree_mean - nwn_mean, 3), "\n")

# DAYTIME
cat("\n--- DAYTIME (10–18) ---\n")
cat("Mean:", round(overall_day$mean, 2), "\n")
cat("SD:", round(overall_day$sd, 2), "\n")
cat("Peak:", round(peak_temp_day, 2), "at", peak_hour_day, "\n")

cat("\nClass differences (daytime):\n")
cat("Tree - Urban:", round(tree_mean_d - urban_mean_d, 3), "\n")
cat("Tree - NWN:", round(tree_mean_d - nwn_mean_d, 3), "\n")