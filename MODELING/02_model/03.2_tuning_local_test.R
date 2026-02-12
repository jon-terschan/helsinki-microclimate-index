# ===============================
# local test of the tuning loop 
# ==============================
# this just runs the tuning loop with 1 
# hyperparameter set and 2 folds to check code and outcome works as expected

library(dplyr)
library(ranger)

set.seed(42)

# -------------------------------
# PARAMS
# -------------------------------
train_model <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/fold_train.rds")
fold_defs       <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/fold_defs.rds")
param_grid  <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/tuning_grid_40.rds")

source("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/03.3_metrics.R")
source("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/03.4_model_specs.R")

params <- param_grid[1, ] # 1 set of hyperparameters from the grid
print(params)
folds <- fold_defs[1:2, ] # two folds only

# -------------------------------
# LOOP
# -------------------------------
results <- list()

for (i in seq_len(nrow(folds))) {

  sf <- folds$spatial_fold[i]
  tf <- folds$time_fold[i]

  cat("Running fold | spatial:", sf, "| time:", tf, "\n")

  train_data <- train_model %>%
    filter(!(spatial_fold == sf & time_fold == tf))

  test_data <- train_model %>%
    filter(spatial_fold == sf & time_fold == tf)

  rf <- ranger(
    formula = formula_rf,
    data = train_data,
    num.trees = 500,
    mtry = params$mtry,
    min.node.size = params$min.node.size,
    sample.fraction = params$sample.fraction,
    seed = 42,
    num.threads = 1,
    importance = "none"
  )

  preds <- predict(rf, data = test_data)$predictions

  results[[i]] <- data.frame(
    spatial_fold = sf,
    time_fold = tf,
    rmse = rmse(test_data$temp, preds),
    mae  = mae(test_data$temp, preds),
    bias = bias(test_data$temp, preds),
    r2   = r2(test_data$temp, preds)
  )
}

# -------------------------------
# CHECK OUTPUT
# -------------------------------
results_df <- bind_rows(results)
print(results_df)

results_df %>%
  summarise(
    mean_rmse = mean(rmse),
    mean_mae  = mean(mae),
    mean_bias = mean(bias),
    mean_r2   = mean(r2)
  )
