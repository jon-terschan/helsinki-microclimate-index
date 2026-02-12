library(ranger)
library(fastshap)
library(dplyr)

source("03.4_model_spec.R")

# -------------------------------
# Load model + data
# -------------------------------
rf <- readRDS(
  "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/models/helmi_final.rds"
)

train <- readRDS(
  "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/data/fold_train.rds"
)

X <- train[, predictors]

# -------------------------------
# Prediction wrapper
# -------------------------------
pred_fun <- function(object, newdata) {
  predict(object, data = newdata)$predictions
}

# -------------------------------
# Subsample for SHAP
# -------------------------------
set.seed(42)
idx <- sample(seq_len(nrow(X)), 2000)
X_sub <- X[idx, ]

# -------------------------------
# SHAP
# -------------------------------
shap <- fastshap::explain(
  object       = rf,
  X            = X_sub,
  pred_wrapper = pred_fun,
  nsim         = 100
)

saveRDS(shap, "shap_values.rds")
