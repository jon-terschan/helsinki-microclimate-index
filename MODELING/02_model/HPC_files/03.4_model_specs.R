# ===============================
# 03.4_model_spec.R
# Fixed model definition
# ===============================

response <- "temp"

train <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/fold_train.rds")

# exclude the things that are no predictors here
predictors <- train %>%
  select(-sensor_id,
         -sensor_channel,
         -time,
         -temp,
         -spatial_fold,
         -time_fold,
         -x,
         -OOS,
         -y)

predictors <- c(names(predictors))
predictors

formula_rf <- as.formula(
  paste(response, "~", paste(predictors, collapse = " + "))
)

# -------------------------------
# RF fixed settings
# -------------------------------
rf_fixed <- list(
  num.trees = 2000,                 # higher for final model
  importance = "none",              # consistent with tuning
  #respect.unordered.factors = "order", # double check later
  seed = 42
)