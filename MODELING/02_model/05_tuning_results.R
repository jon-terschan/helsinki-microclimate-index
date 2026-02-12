# ==========================================
# 05_aggregate_tuning_results.R
# ==========================================
# Aggregates per-fold CV results across
# hyperparameter sets, only works if the results
# are downloaded and stored in the expected locations
library(dplyr)
library(purrr)

# input params
results_dir <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/tuning_results/"
out_file <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/tuning_results/tuning_summary.rds"
param_grid <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/tuning_grid_40.rds")

# list result files
files <- list.files(
  results_dir,
  pattern = "^results_param_[0-9]+\\.rds$",
  full.names = TRUE
)
# read all
all_results <- map_dfr(files, readRDS)

# aggregate all
summary_df <- all_results %>%
  group_by(param_id) %>%
  summarise(
    n_folds   = n(),
    mean_rmse = mean(rmse),
    sd_rmse   = sd(rmse),
    mean_mae  = mean(mae),
    sd_mae    = sd(mae),
    mean_bias = mean(bias),
    sd_bias   = sd(bias),
    mean_r2   = mean(r2),
    sd_r2     = sd(r2),
    .groups = "drop"
  ) %>%
  arrange(mean_rmse)

# join with hyperparameter grid to show best model choices
summary_df <- summary_df %>%
  left_join(param_grid, by = "param_id")

# export
saveRDS(summary_df, out_file)
print(head(summary_df, 10))
