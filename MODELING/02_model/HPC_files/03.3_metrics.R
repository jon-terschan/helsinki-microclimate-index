# ===============================
# Performance metrics (locked)
# ===============================

rmse <- function(obs, pred) {
  sqrt(mean((pred - obs)^2, na.rm = TRUE))
}

mae <- function(obs, pred) {
  mean(abs(pred - obs), na.rm = TRUE)
}

bias <- function(obs, pred) {
  mean(pred - obs, na.rm = TRUE)
}

r2 <- function(obs, pred) {
  ss_res <- sum((obs - pred)^2, na.rm = TRUE)
  ss_tot <- sum((obs - mean(obs, na.rm = TRUE))^2, na.rm = TRUE)
  1 - ss_res / ss_tot
}

# Model selection metric: RMSE
# Reported metrics: RMSE, MAE, Bias
# R2 reported descriptively only