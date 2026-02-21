# -------------------------------
# LOCAL PREP FOR TUNING
# -------------------------------
# this mainly consists of creating the spatio temporal folds for tuning
# and the hyperparameter tuning grid which will serve as indexer
# dont run this without carefully reviewing the appripriateness of each step
# can be run locally as its just prep for HPC tuning 

library(sf)
library(dplyr)
library(blockCV)
library(units)
library(data.table)

set.seed(42)

# -------------------------------
# 0) Load data and extract coords
# -------------------------------
train <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/01_traindataprep/06_train_data.rds")

coords <- st_coordinates(train)
train$x <- coords[, 1]
train$y <- coords[, 2]

stopifnot(inherits(train, "sf"))
stopifnot(all(c("sensor_id", "time", "x", "y") %in% names(train)))

# -------------------------------
# 1) Sensor-level sf (one per station)
# -------------------------------
sensor_sf <- train %>%
  st_drop_geometry() %>%
  distinct(sensor_id, .keep_all = TRUE) %>%
  st_as_sf(coords = c("x", "y"),
           crs = st_crs(train),
           remove = FALSE)

n_sensors <- nrow(sensor_sf)
message("n_sensors = ", n_sensors)  # ~90

# -------------------------------
# 2) Choose spatial block size
# -------------------------------
# With ~90 sensors, NN-distance heuristic is more stable than variograms

dmat <- st_distance(sensor_sf)
diag(dmat) <- NA

nn <- apply(dmat, 1, function(r) min(as.numeric(r), na.rm = TRUE))
median_nn <- median(nn, na.rm = TRUE)

# multiplier controls block coarseness (2–4 typical)
suggested_size <- median_nn * 3

message(
  "Median NN distance (m): ", round(median_nn, 1),
  " → block size = ", round(suggested_size, 1), " m"
)

library(sf)

suggested_size <- as.numeric(suggested_size)

# -------------------------------
# 3) Spatial folds with blockCV
# -------------------------------
k_spatial <- 5  # sensible for ~90 sensors

cv_sp <- cv_spatial(
  x = sensor_sf,
  k = k_spatial,
  size = suggested_size,
  selection = "random",
  iteration = 100,
  progress = TRUE
)

# -------------------------------
# 4) SANITY PLOT 1: spatial folds
# -------------------------------
#png("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model//spatial_folds.png",
#    width = 2000, height = 1600, res = 200)

#plot(st_geometry(sensor_sf),
#    col = cv_sp$folds_ids,
#     pch = 19,
#     main = "Spatial CV folds (sensor-level)",
#     axes = TRUE)

#legend("topright",
#       legend = paste("Fold", sort(unique(cv_sp$folds_ids))),
#       col = sort(unique(cv_sp$folds_ids)),
#       pch = 19,
#       bty = "n")

#dev.off()

# -------------------------------
# 5) Inspect fold balance
# -------------------------------
print(table(cv_sp$folds_ids))

# -------------------------------
# 6) Map spatial folds back to rows
# -------------------------------
sensor_folds_df <- data.frame(
  sensor_id = sensor_sf$sensor_id,
  spatial_fold = cv_sp$folds_ids,
  stringsAsFactors = FALSE
)

train <- train %>%
  left_join(sensor_folds_df, by = "sensor_id")

stopifnot(!any(is.na(train$spatial_fold)))

# -------------------------------
# 7) Temporal folds (blocked)
# -------------------------------
k_time <- 5

train <- train %>%
  arrange(time) %>%
  mutate(time_fold = ntile(time, k_time))

# -------------------------------
# 8) Build spatio-temporal folds
# -------------------------------
min_test_size <- 50

folds <- list()
i <- 1

for (s in sort(unique(train$spatial_fold))) {
  for (t in sort(unique(train$time_fold))) {

    test_idx <- which(train$spatial_fold == s &
                      train$time_fold == t)

    if (length(test_idx) < min_test_size) next

    train_idx <- setdiff(seq_len(nrow(train)), test_idx)

    folds[[paste0("fold_", i)]] <- list(
      train = train_idx,
      test  = test_idx,
      spatial_fold = s,
      time_fold = t
    )

    i <- i + 1
  }
}

glimpse(train)
message("Total spatio-temporal folds: ", length(folds))
glimpse(folds)

# -------------------------------
# 9) SANITY PLOT 2: fold sizes
# -------------------------------
fold_sizes <- sapply(folds, function(f) length(f$test))

png("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/st_fold_sizes.png",
    width = 2000, height = 1200, res = 200)

hist(fold_sizes,
     breaks = 30,
     col = "grey80",
     main = "Spatio-temporal CV fold sizes",
     xlab = "Number of test samples")

dev.off()

summary(fold_sizes)

# -------------------------------
# 10) Save folded training data
# -------------------------------
# ADD ALS PREDICTORS LATER
# factor handling FOR LAND COVER ADD LATER

# drop geogmetry if still present and then
train_model <- train %>%
  st_drop_geometry() %>%
  drop_na() 

glimpse(train_model)
sum(is.na(train_model))

saveRDS(train_model,
        "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/fold_train.rds",
        compress = "xz")

# superfluous but i also export a lookup table of the folds    
fold_def <- train_model %>%
  distinct(spatial_fold, time_fold) %>%
  arrange(spatial_fold, time_fold)

saveRDS(fold_def, "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/fold_defs.rds")

# -------------------------------
# 11) HYPERPARAMETER GRID
# -------------------------------
# ADD ALS PREDICTORS LATER
library(data.table)
train <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/fold_train.rds")
glimpse(train)
# check the full number of predictors
# to test mtry > n predictors
predictors <- train %>%
  select(-sensor_id,
         -sensor_channel,
         -time,
         -temp,
         #-geom,
         -spatial_fold,
         -time_fold,
         -OOS,
         -x,
         -y,
         -t2m_lag1, # adds 0.02 RMSE
         -t2m_lag3, # adds 0.02 RMSE
         -t2m_lag6, # adds 0.02 RMSE
         -t2m_lag24, # adds 0.02 RMSE
         -ssrd_roll3, # adds 0.02 RMSE
         -ssrd_roll6, # adds 0.02 RMSE
         -bldg_fr_10 # no signal in the train data)
         )

pred <- c(names(predictors))
p <- length(predictors)
message("Number of predictors: ", p)

# establish grid
param_grid <- CJ(
  mtry = unique(round(c(sqrt(p), p/4, p/3, p/2, p * 0.75))),
  min.node.size = c(5, 10, 20, 40),
  sample.fraction = c(0.6, 0.8)
)
param_grid[, param_id := .I]
setcolorder(param_grid,
            c("param_id",
              "mtry",
              "min.node.size",
              "sample.fraction")
              )
print(param_grid)

# save
saveRDS(param_grid, "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/HPC_files/tuning_grid_40.rds", compress = "xz")
