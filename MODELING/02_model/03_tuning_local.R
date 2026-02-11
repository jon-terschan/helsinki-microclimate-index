# ===============================
# 03.1_tuning_local.R
# Spatio-temporal CV preparation
# ===============================

library(sf)
library(dplyr)
library(blockCV)
library(units)

set.seed(42)

# -------------------------------
# 0) Load data and extract coords
# -------------------------------
train <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/train_data/06_final_train.rds")

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
png("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model//spatial_folds.png",
    width = 2000, height = 1600, res = 200)

plot(st_geometry(sensor_sf),
     col = cv_sp$folds_ids,
     pch = 19,
     main = "Spatial CV folds (sensor-level)",
     axes = TRUE)

legend("topright",
       legend = paste("Fold", sort(unique(cv_sp$folds_ids))),
       col = sort(unique(cv_sp$folds_ids)),
       pch = 19,
       bty = "n")

dev.off()

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

message("Total spatio-temporal folds: ", length(folds))

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
# 10) Save frozen artifacts
# -------------------------------
saveRDS(folds, "tuning/st_folds.rds")

df_model <- st_drop_geometry(train)
saveRDS(df_model, "tuning/data_model.rds")


# --------------------------------
# GRID SUMMARY FOR VISUALIZATION
# --------------------------------
library(dplyr)
library(ggplot2)
library(lubridate)
library(sf)

plot_df <- train %>%
  st_drop_geometry() %>%
  mutate(
    month = month(time),
    spatial_fold = factor(spatial_fold),
    time_fold    = factor(time_fold)
  ) %>%
  filter(month %in% 5:9)

# Expand data: one copy per (spatial_fold, time_fold) panel
plot_df <- plot_df %>%
  tidyr::crossing(
    panel_spatial = levels(plot_df$spatial_fold),
    panel_time    = levels(plot_df$time_fold)
  ) %>%
  mutate(
    set = ifelse(
      spatial_fold == panel_spatial &
      time_fold    == panel_time,
      "Test", "Train"
    ),
    set = factor(set, levels = c("Train", "Test"))
  )

cv_grid_plot <- ggplot(
  plot_df,
  aes(x = month, y = temp, color = set)
) +
  geom_point(
    alpha = 0.6,
    size = 1.8
  ) +
  scale_color_manual(
    values = c(Train = "steelblue", Test = "red")
  ) +
  scale_x_continuous(
    breaks = 5:9,
    labels = c("May", "Jun", "Jul", "Aug", "Sep")
  ) +
  facet_grid(
    rows = vars(panel_spatial),
    cols = vars(panel_time)
  ) +
  labs(
    title = "Spatio-temporal cross-validation structure",
    subtitle = "Each panel holds out one spatial × temporal block",
    x = "Month",
    y = "Temperature",
    color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

cv_grid_plot

# -------------------------------
# 4) Save to file
# -------------------------------

ggsave(
  filename = "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/MODELING/02_model/st_cv_structure_5x5.png",
  plot = cv_grid_plot,
  width = 14,
  height = 12,
  dpi = 300
)


# placeholders for later
# predictors <- ...
# saveRDS(predictors, "tuning/predictors.rds")

# param_grid <- ...
# saveRDS(param_grid, "tuning/param_grid.rds")

saveRDS(list(metric = "RMSE"), "tuning/tuning_meta.rds")
