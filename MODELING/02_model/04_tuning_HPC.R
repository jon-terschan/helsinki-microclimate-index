# Required packages
library(dplyr)
library(sf)         # optional but convenient for coords
library(caret)
library(ranger)
library(doParallel)

# ---------------------------
# Example data assumptions (EDIT as needed)
# data: a data.frame with columns:
#   lon, lat, time (numeric or year), response, and predictors...
# ---------------------------
# load your data here
data <- read.csv("my_data.csv")
# For the example I'll assume 'data' exists

# Ensure there's an index column
data$.row_id <- seq_len(nrow(data))

# 1) Create spatial blocks (simple: kmeans on coords)
k_spatial <- 3   # EDIT: number of spatial blocks (rows in your figure)
coords <- data %>% select(lon, lat)
set.seed(42)
spatial_clust <- kmeans(coords, centers = k_spatial)$cluster
data$spatial_block <- as.integer(spatial_clust)

# Alternative: use blockCV::spatialBlock for polygon/block-based folding if you prefer.
# library(blockCV)
# sb <- spatialBlock(speciesData = sf::st_as_sf(data, coords = c("lon","lat"), crs = 4326),
#                    theRange = 50000, # meters, adjust
#                    k = k_spatial, selection = "random")
# data$spatial_block <- sb$foldID

# 2) Create temporal bins
k_time <- 3  # EDIT: number of temporal folds (columns in your figure)
# If your time is year:
# data$time_bin <- as.integer(cut(data$year, breaks = k_time, labels = FALSE))
# If time is continuous, use ntile to make roughly equal counts:
library(dplyr)
data$time_bin <- as.integer(ntile(data$time, k_time))

# 3) Build spatio-temporal test folds (one combo per fold)
combos <- expand.grid(spatial_block = sort(unique(data$spatial_block)),
                      time_bin = sort(unique(data$time_bin)),
                      KEEP.OUT.ATTRS = FALSE,
                      stringsAsFactors = FALSE)

# Remove combos with zero observations
combo_has_obs <- apply(combos, 1, function(r) {
  any(data$spatial_block == r["spatial_block"] & data$time_bin == r["time_bin"])
})
combos <- combos[combo_has_obs, , drop = FALSE]

# Build caret-style index lists: training indices for each fold and test indices
index <- list()
indexOut <- list()
for(i in seq_len(nrow(combos))){
  sp <- combos$spatial_block[i]
  tm <- combos$time_bin[i]
  test_idx <- which(data$spatial_block == sp & data$time_bin == tm)
  if(length(test_idx) == 0) next
  train_idx <- setdiff(seq_len(nrow(data)), test_idx)
  # caret expects indices of the rows used for training in each resample
  index[[paste0("Fold", i)]] <- train_idx
  indexOut[[paste0("Fold", i)]] <- test_idx
}

# 4) Setup caret trainControl using manual indices
trControl <- trainControl(
  method = "cv",
  number = length(index),
  index = index,        # list of training indices
  indexOut = indexOut,  # list of test indices
  savePredictions = "final",
  allowParallel = TRUE
)

# 5) Define model/tuning grid (example for ranger)
# tuneGrid could include mtry, min.node.size, sample.fraction etc.
tuneGrid <- expand.grid(
  mtry = c(2, 4, 6),            # EDIT according to #predictors
  splitrule = c("variance"),    # for regression, or "gini"/"extratrees" for classification
  min.node.size = c(5, 10)
)

# 6) Parallel register (on a single node)
ncores <- parallel::detectCores() - 1
cl <- makeCluster(ncores)
registerDoParallel(cl)

set.seed(123)
caret_mod <- train(
  x = data %>% select(-.row_id, -response) %>% select(-spatial_block, -time_bin), # EDIT predictors selection
  y = data$response,   # EDIT response column
  method = "ranger",
  trControl = trControl,
  tuneGrid = tuneGrid,
  importance = 'permutation',
  num.trees = 500,
  metric = "RMSE"      # or "Accuracy" for classification
)

stopCluster(cl)
registerDoSEQ()

# Results
print(caret_mod)
