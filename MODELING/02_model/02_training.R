set.seed(42)
library(ranger)
library(fastshap)

# load training data
train <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/train_data/06_final_train.rds")

# define predictors
predictors <- setdiff(
  names(train),
  c("temp", "time", "fold", "id", "sensor_id", "sensor_channel")
)

# fit baseline RF
rf <- ranger(
  formula = temp ~ .,
  data = train[, c("temp", predictors)],
  num.trees = 800,
  mtry = floor(sqrt(length(predictors))),
  min.node.size = 10,
  importance = "permutation",
  respect.unordered.factors = "order",
  sample.fraction = 0.8,
  seed = 42
)

# ---- SHAP ----

pred_fun <- function(object, newdata) {
  predict(object, data = newdata)$predictions
}

# design matrix
X <- train[, predictors]

# subsample for SHAP
set.seed(42)
idx <- sample(seq_len(nrow(train)), 2000)
X_sub <- X[idx, ]

shap <- fastshap::explain(
  object       = rf,
  X            = X_sub,
  pred_wrapper = pred_fun,
  nsim         = 100
)

plot(
  X_sub$t2m,
  shap$t2m,
  xlab = "ERA5 t2m",
  ylab = "SHAP value (effect on predicted temp)",
  pch = 16, col = rgb(0,0,0,0.3)
)
abline(h = 0, col = "red")