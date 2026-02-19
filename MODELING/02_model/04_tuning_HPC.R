# ==========================================
# Runs full 5x5 CV for ONE hyperparameter set
# ==========================================
library(dplyr)
library(ranger)

set.seed(42)
# ------------------------------------------
# DEBUG SETTINGS
# ------------------------------------------
MAX_FOLDS <- Inf   # set to Inf for full CV

# ------------------------------------------
# 1) IDENTIFY PARAM ID TO SLURM
# ------------------------------------------
param_id <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))

if (is.na(param_id)) {
  param_id <- 1
  message("SLURM_ARRAY_TASK_ID not set, defaulting to param_id = 1")
}
cat("Running parameter set:", param_id, "\n")

# ------------------------------------------
# 2) LOAD LOCALLY PREPARED DATA AND ERROR METRICS
# ------------------------------------------
train_model <- readRDS("/scratch/project_2001208/Jonathan/model/data/processed/ML/fold_train.rds")
fold_defs   <- readRDS("/scratch/project_2001208/Jonathan/model/data/processed/ML/fold_defs.rds")
param_grid  <- readRDS("/scratch/project_2001208/Jonathan/model/data/processed/ML/tuning_grid_40.rds")

source("/scratch/project_2001208/Jonathan/model/scripts/03.4_model_spec.R")
source("/scratch/project_2001208/Jonathan/model/scripts/03.3_metrics.R")

params <- param_grid[param_id, ]
print(params)

# ------------------------------------------
# 3) RUN CV 
# ------------------------------------------
n_folds <- min(MAX_FOLDS, nrow(fold_defs))
results <- vector("list", n_folds)

for (i in seq_len(n_folds)) {

  sf <- fold_defs$spatial_fold[i]
  tf <- fold_defs$time_fold[i]

  cat("Fold", i, "| spatial:", sf, "| time:", tf, "\n")

  train_data <- train_model %>%
    filter(!(spatial_fold == sf & time_fold == tf))

  test_data <- train_model %>%
    filter(spatial_fold == sf & time_fold == tf)

  rf <- ranger(
    formula = formula_rf,
    data = train_data,
    num.trees = 1000,
    mtry = params$mtry,
    min.node.size = params$min.node.size,
    sample.fraction = params$sample.fraction,
    seed = 42,
    num.threads = as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK")),
    importance = "none"
  )

  preds <- predict(rf, data = test_data)$predictions

  results[[i]] <- data.frame(
    param_id = param_id,
    spatial_fold = sf,
    time_fold = tf,
    rmse = rmse(test_data$temp, preds),
    mae  = mae(test_data$temp, preds),
    bias = bias(test_data$temp, preds),
    r2   = r2(test_data$temp, preds)
  )
}

results_df <- bind_rows(results)

# ------------------------------------------
# 4) SAVE RESULTS
# ------------------------------------------
outfile <- paste0("/scratch/project_2001208/Jonathan/model/logs/model/out/tuning_results/results_param_", param_id, ".rds")
saveRDS(results_df, outfile)

cat("Saved:", outfile, "\n")
