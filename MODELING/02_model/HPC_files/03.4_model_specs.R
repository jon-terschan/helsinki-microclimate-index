# ===============================
# 03.4_model_spec.R
# Fixed model definition
# ===============================

response <- "temp"

predictors <- c(
  "SMC", "bldg_dist", "bldg_frac_10m", "bldg_frac_mean_50m",
  "dtm", "eastness", "imperv_frac", "imperv_frac_50m",
  "ocean_dist", "ocean_frac", "rock_frac", "ruggedness",
  "slope", "southness", "water_dist", "water_frac",
  "t2m", "ssrd", "tp", "wind_s",
  "hour_sin", "hour_cos", "doy_sin", "doy_cos"
)

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