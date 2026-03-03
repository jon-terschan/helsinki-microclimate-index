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
