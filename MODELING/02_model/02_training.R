# -------------------------------
# FINAL MODEL FIT
# -------------------------------
# this should be run after tuning and summarizing the tuning results
# can be run locally, but the prerequisites must be downloaded from HPC first
library(ranger)
library(dplyr)

# -------------------------------
# PARAMS
# -------------------------------
# model specs EDIT before running
source("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/03.4_model_specs.R")
# train data CHECK before running
train <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/fold_train.rds")
# extract best params from aggregate table
tuning_summary <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/tuning_results/tuning_summary.rds")
best_params <- tuning_summary %>%
  arrange(mean_rmse) %>%
  slice(1)
best_params

# -------------------------------
# MODEL FIT
# -------------------------------
rf_final <- ranger(
  formula = formula_rf,
  data = train,
  num.trees = rf_fixed$num.trees,
  mtry = best_params$mtry,
  min.node.size = best_params$min.node.size,
  sample.fraction = best_params$sample.fraction,
  importance = rf_fixed$importance,
  #respect.unordered.factors = rf_fixed$respect.unordered.factors,
  seed = rf_fixed$seed
)

# save output
saveRDS(rf_final, "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/models/helmi_final.rds"
)

glimpse(rf_final)
