# ============================================================
# Diagnostic helper: 
# counts unique sensor IDs in training data table
# ============================================================
library(dplyr) # data manipulation

# ------------------------------------------------------------
# INPUT# ------------------------------------------------------------

train_path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/01_traindataprep/06_train_data.rds"
# Path to the saved training data (RDS) - we assume the train data table is up to date and saved here

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
train <- readRDS(train_path)
# Load the training dataset into memory as an R object.

# ------------------------------------------------------------
# BASIC DIAGNOSTICS
# ------------------------------------------------------------
cat("\n--- Object class ---\n")
 # Print the R class(es) of the loaded object (e.g. data.frame, tibble, sf)
print(class(train))
cat("\n--- Dimensions ---\n")
 # Show dimensions (rows, columns) for tabular objects
print(dim(train))
cat("\n--- Column names ---\n")
 # List all column names to inspect available fields
print(names(train))
cat("\n--- Structure ---\n")
 # Print internal structure including types of each column
str(train)
cat("\n--- First rows ---\n")
 # Show the first rows to get a quick sense of the data
print(head(train))
cat("\n--- Columns containing 'sensor' ---\n")
 # Find column names that contain the substring 'sensor' (case-insensitive)
sensor_like_cols <- grep("sensor", names(train), ignore.case = TRUE, value = TRUE)
print(sensor_like_cols)
cat("\n--- Columns containing 'id' ---\n")
 # Find column names that contain the substring 'id' to help identify ID fields
id_like_cols <- grep("id", names(train), ignore.case = TRUE, value = TRUE)
print(id_like_cols)

# ------------------------------------------------------------
# IDENTIFY SENSOR ID COLUMN
# ------------------------------------------------------------
preferred_sensor_col <- "sensor_id"
# Preferred/expected column name that holds the sensor identifier

# Determine which column to use as the sensor ID:
# 1) If the preferred name exists, use it.
# 2) If exactly one `sensor`-like column exists, use that.
# 3) Otherwise, stop with an informative error.
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
 # Report which column will be used for sensor ID analyses
print(sensor_col)


# ------------------------------------------------------------
# SENSOR ID DIAGNOSTICS
# ------------------------------------------------------------
cat("\n--- Sensor ID type ---\n")
 # Show the R class of the sensor ID column (e.g. integer, character, factor)
print(class(train[[sensor_col]]))

cat("\n--- Missing sensor IDs ---\n")
 # Count how many rows have missing/NA sensor IDs
print(sum(is.na(train[[sensor_col]])))

cat("\n--- Number of unique sensor IDs ---\n")
 # Count unique sensor identifiers, ignoring NA
n_unique_sensors <- n_distinct(train[[sensor_col]], na.rm = TRUE)
print(n_unique_sensors)

cat("\n--- First 20 unique sensor IDs ---\n")
 # Show a sample (first 20) of sorted unique sensor IDs for inspection
print(head(sort(unique(train[[sensor_col]])), 20))

cat("\n--- Observations per sensor: summary ---\n")
# Compute number of observations per sensor. If `train` is an sf object,
# drop spatial geometry before counting to avoid issues.
sensor_counts <- train %>%
  st_drop_geometry_if_present() %>%
  count(.data[[sensor_col]], name = "n_obs", sort = TRUE)

 # Print a summary (min, median, mean, max, etc.) of per-sensor counts
print(summary(sensor_counts$n_obs))

cat("\n--- Observations per sensor: first 20 ---\n")
 # Show the top 20 sensors by observation count (sorted because count used sort=TRUE)
print(head(sensor_counts, 20))

# ------------------------------------------------------------
# HELPER: DROP GEOMETRY ONLY IF PRESENT
# ------------------------------------------------------------
st_drop_geometry_if_present <- function(x) {
  # Helper that removes geometry if `x` is an sf (simple features) object.
  # This prevents sf geometry columns from interfering with dplyr operations.
  if (inherits(x, "sf")) {
    sf::st_drop_geometry(x)
  } else {
    x
  }
}