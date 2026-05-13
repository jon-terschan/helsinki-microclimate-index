# ============================================================
# Diagnostics: count unique sensor IDs in training data table
# ============================================================

library(dplyr)

# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

train_path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/01_traindataprep/06_train_data.rds"

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

train <- readRDS(train_path)

# ------------------------------------------------------------
# BASIC DIAGNOSTICS
# ------------------------------------------------------------

cat("\n--- Object class ---\n")
print(class(train))

cat("\n--- Dimensions ---\n")
print(dim(train))

cat("\n--- Column names ---\n")
print(names(train))

cat("\n--- Structure ---\n")
str(train)

cat("\n--- First rows ---\n")
print(head(train))

cat("\n--- Columns containing 'sensor' ---\n")
sensor_like_cols <- grep("sensor", names(train), ignore.case = TRUE, value = TRUE)
print(sensor_like_cols)

cat("\n--- Columns containing 'id' ---\n")
id_like_cols <- grep("id", names(train), ignore.case = TRUE, value = TRUE)
print(id_like_cols)


# ------------------------------------------------------------
# IDENTIFY SENSOR ID COLUMN
# ------------------------------------------------------------

preferred_sensor_col <- "sensor_id"

if (preferred_sensor_col %in% names(train)) {
  sensor_col <- preferred_sensor_col
} else if (length(sensor_like_cols) == 1) {
  sensor_col <- sensor_like_cols
} else {
  stop(
    paste0(
      "Could not uniquely identify sensor ID column.\n",
      "Expected column: sensor_id\n",
      "Sensor-like columns found: ",
      paste(sensor_like_cols, collapse = ", ")
    )
  )
}

cat("\n--- Sensor ID column used ---\n")
print(sensor_col)


# ------------------------------------------------------------
# SENSOR ID DIAGNOSTICS
# ------------------------------------------------------------

cat("\n--- Sensor ID type ---\n")
print(class(train[[sensor_col]]))

cat("\n--- Missing sensor IDs ---\n")
print(sum(is.na(train[[sensor_col]])))

cat("\n--- Number of unique sensor IDs ---\n")
n_unique_sensors <- n_distinct(train[[sensor_col]], na.rm = TRUE)
print(n_unique_sensors)

cat("\n--- First 20 unique sensor IDs ---\n")
print(head(sort(unique(train[[sensor_col]])), 20))

cat("\n--- Observations per sensor: summary ---\n")
sensor_counts <- train %>%
  st_drop_geometry_if_present() %>%
  count(.data[[sensor_col]], name = "n_obs", sort = TRUE)

print(summary(sensor_counts$n_obs))

cat("\n--- Observations per sensor: first 20 ---\n")
print(head(sensor_counts, 20))


# ------------------------------------------------------------
# HELPER: DROP GEOMETRY ONLY IF PRESENT
# ------------------------------------------------------------

st_drop_geometry_if_present <- function(x) {
  if (inherits(x, "sf")) {
    sf::st_drop_geometry(x)
  } else {
    x
  }
}