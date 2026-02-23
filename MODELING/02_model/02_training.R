# -------------------------------
# FINAL MODEL FIT
# -------------------------------
# this should be run after tuning and summarizing the tuning results
# can be run locally, but the prerequisites must be downloaded from HPC first
library(ranger)
library(dplyr)
set.seed(42)

# -------------------------------
# PARAMS
# -------------------------------
# model specs EDIT before running
source("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/03.4_model_specs.R")
# train data CHECK before running
train <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/fold_train.rds")
# extract best params from aggregate table
tuning_summary <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/02_tuningresults/tuning_summary_1.rds")
str(tuning_summary)

best_params <- tuning_summary %>%
  arrange(mean_rmse) %>%
  slice(1)
str(best_params)

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
  seed = rf_fixed$seed,
  oob.error = FALSE # not needed here
)

# save final model
saveRDS(rf_final, 
"//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/03_models/helmi_2000_v1.3_2.rds",
compress = "xz")

glimpse(rf_final)
