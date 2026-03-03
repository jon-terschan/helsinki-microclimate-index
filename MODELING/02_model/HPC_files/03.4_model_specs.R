# ===============================
# 03.4_model_spec.R
# Fixed model definition
# ===============================
response <- "temp"
# local
train <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/fold_train.rds")
# HPC: train <- readRDS("/scratch/project_2001208/Jonathan/model/data/processed/ML/fold_train.rds")
glimpse(train)
# exclude non predictor columns
predictors <- train %>%
  select(-sensor_id,
         -sensor_channel,
         -time,
         -temp,
         -spatial_fold,
         -time_fold,
         -x,
         -OOS,
         -CCl, # for the time being, its currently a token value
         -y,
         -SMC
         #-t2m_lag1, # adds 0.02 RMSE
         #-t2m_lag3, # adds 0.02 RMSE
         #-t2m_lag6, # adds 0.02 RMSE
         #-t2m_lag24, # adds 0.02 RMSE
         #-ssrd_roll3, # adds 0.02 RMSE
         #-ssrd_roll6, # adds 0.02 RMSE
         #-bldg_fr_10 # no signal in the train data)
         ) 

predictors <- c(names(predictors))
predictors

formula_rf <- as.formula(
  paste(response, "~", paste(predictors, collapse = " + "))
)

# -------------------------------
# RF fixed settings
# -------------------------------
rf_fixed <- list(
  num.trees = 1500,                 # higher for final model
  importance = "none",              # consistent with tuning
  #respect.unordered.factors = "order", # double check later
  seed = 42
)
