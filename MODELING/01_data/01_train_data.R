# ============================================================
# TRAINING DATA GENERATION AND FEATURE ASSEMBLY
# ============================================================
# This script:
# 1) Converts CLF sensor data to long format
# 2) aggregates temp observations to hourly resolution
# 3) filters to summer non-OOS air temp observations
# 4) attaches sensor geometries to the table
# 5) extracts static raster predictors at sensor locations
# 6) loads and joins era5 predictors
# 7) handles sine cosine encoding of day and exports final data
# script is not super modular but its a one time data processing step that
# needs to be changed later so i prioritized readibility
# ============================================================

# training data generation split
# output, training data table
source("scripts/00_config.R")

##############################################################
# ----- CONVERT CLF FORMAT TO LONG, AGGREGATE TO HOURLY  ----- 
##############################################################
# Each CLF file contains sub-hourly temperatures for one sensor. This section:
#  - Reads all sensor CSVs
#  - Converts channel columns (t1, t2, t3, ...) to long format
#  - Aggregates to hourly means
#  - Preserves OOS flags
library(readr)
library(dplyr)
library(tidyr)
library(lubridate)
library(sf)
library(arrow)

# folder with CLF files. There is no further quality control here, so this should all be done in an earlier time step
csv_dir <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/11.25/processed/4_finetuning/output"
files <- list.files(csv_dir, pattern = "\\.csv$", full.names = TRUE)

# empty list to collect results - one element per file
out <- vector("list", length(files))

for (i in seq_along(files)) {
  f <- files[i]
  # Extract numeric sensor ID from filename
  # Assumes filenames contain an underscore followed by digits
  sensor_id <- sub("^[^_]+_([0-9]+).*", "\\1", basename(f)) # absurd regex to extract the numeric part (sensor id)
  df <- read_csv(f, show_col_types = FALSE) # read csv
  # ensure OOS exists, might have forgotten it somewhere in asensor without problems
  if (!"OOS" %in% names(df)) {  
    df$OOS <- 0
  }

  df_hourly <- df %>%
    # Convert wide channel format (t1, t2, t3, …) to long
    pivot_longer(
      cols = starts_with("t"),
      names_to = "sensor_channel",
      values_to = "temp"
    ) %>%
    # parse time and attach sensor id
    mutate(
      time = as.POSIXct(datetime, tz = "UTC"),
      sensor_id = sensor_id
    ) %>%
    # remove original datetime, this is honestly superfluous but i dont want to refactor this code again
    select(-datetime) %>%
    # snap to hourly timestamps and aggregate data to hourly means
    mutate(
      time = lubridate::floor_date(time, "hour")
    ) %>%
    group_by(sensor_id, sensor_channel, time) %>%
    summarise(
      temp = mean(temp, na.rm = TRUE),
      SMC  = mean(SMC,  na.rm = TRUE),
      OOS  = as.integer(any(OOS == 1)),
      .groups = "drop"
    ) %>%
    # Replace NaN with NA
    mutate(
      temp = ifelse(is.nan(temp), NA_real_, temp)
    )
  out[[i]] <- df_hourly # store per file result
}

df_all_hourly <- bind_rows(out) # combine all sensors into one long hourly table
#glimpse(df_all_hourly) #check integrity

# save and load from disk
saveRDS(df_all_hourly, file = "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/train_data/01_df_all_hourly.rds")

##############################################################
# ----- FILTER TO SUMMER, IN-CANOPY TRAINING DATA ---------- 
##############################################################
#  - channel t3, which is air temperature in CLF
#  - non-OOS observations
#  - summer season (May 20 – Sep 10)
df_all_hourly <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/train_data/01_df_all_hourly.rds")

df_t3_summer <- df_all_hourly %>%
  filter(
    sensor_channel == "t3",
    OOS == 0,
    !is.na(temp),
    yday(time) >= 140,   # May 20, EDIT
    yday(time) <= 253    # Sep 10, EDIT
  )

# save to disk
saveRDS(df_t3_summer, file = "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/train_data/02_df_train_summer.rds")
# glimpse(df_t3_summer)

##############################################################
# ----- JOIN TRAINING DATA WITH SPATIAL INDEX -------------- 
##############################################################
# Sensor coordinates exist in two spatial indeces:
#  - helmostatus_11.25.gpkg (newest, preffered
#  - helmostatus_original.gpkg (original, fallback)
# We prefer geometries from the newer file (since they reflect changed locations)
# and fill missing sensors from the original file, to catch stolen ones etc.
library(sf)
library(dplyr)
# target CRS
target_crs <- 3879  # ETRS-TM35FIN

# ---- combine spatial index files ----
# read original and current tile index files
gpk1 <- st_read(
  "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/helmo_map/helmostatus_11.25.gpkg",
  quiet = TRUE
) %>%
  st_transform(target_crs) %>%
  transmute(
    sensor_id = as.character(SERIAL),
    geom
  )

gpk2 <- st_read(
  "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/helmo_map/helmostatus_original.gpkg",
  quiet = TRUE
) %>%
  st_transform(target_crs) %>%
  transmute(
    sensor_id = as.character(SERIAL),
    geom
  )

# combine indices with pref for the newer one (but fill missing sensors from gpkg2)
stations <- bind_rows(
  gpk1,
  gpk2 %>%
    filter(!sensor_id %in% gpk1$sensor_id)
)

# ---- join spatial index files with training data ----
# load training data file
df_t3_summer <-readRDS(file = "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/train_data/02_df_train_summer.rds")

# join train training data file with spatial index
df_train <- df_t3_summer %>%
  left_join(
    stations %>% select(sensor_id, geom),
    by = "sensor_id"
  )

# convert to sf object for spatial index and later spatial joins with rasters
df_train_sf <- st_as_sf(
  df_train,
  sf_column_name = "geom",
  crs = 3879
)

# write to disk
st_write(df_train_sf, "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/train_data/03_stations_training.gpkg", delete_dsn = TRUE)

##############################################################
# ----- LOAD AND ALIGN ALL STATIC RASTERS ------------------ 
##############################################################
# Static predictors are stored as separate raster files.
# This section:
#  - Loads all rasters 
#  - Aligns them to a common grid, ATTENTION: the common grid should already be enforced in more detail
# during raster creation, the resampling here is just a safety net
#  - Standardizes layer names
library(terra)
library(sf)
library(dplyr)
library(tools)

# folder with static predictors
raster_dir <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/rasters_static/"

# list raster files
rasters <- list.files(
  raster_dir,
  pattern = "\\.(tif|grd)$",
  full.names = TRUE
)

# build raster stack resampling template from DTM
template <- rast(rasters[grepl("DTM", rasters)])
template

# Load and align all rasters to the template
r_list <- lapply(rasters, function(f) {
  r <- rast(f)
  # resample if geom doesnt perfectly match template
  if (!compareGeom(r, template, stopOnError = FALSE)) {
    r <- resample(r, template, method = "bilinear")
  }
  
  r
})

# STACK EM
layer_names <- file_path_sans_ext(basename(rasters))

# read & name in one step 
r_stack <- rast(r_list)
names(r_stack) <- layer_names
n <- names(r_stack)
n <- gsub("_10m_Helsinki", "", n)
n <- gsub("_Helsinki", "", n)
names(r_stack) <- tolower(n)
names(r_stack)

# check alignment and CRS
compareGeom(r_stack, template)
crs(r_stack)

# load all sensor locations as tables
stations <- st_read(
  "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/train_data/03_stations_training.gpkg",
  quiet = TRUE
)

# check that they are same CRS
st_crs(stations)
crs(r_stack)

# transform to vector
stations_v <- vect(stations)
pred_vals <- terra::extract( # extract raster values at vector loc
  r_stack,
  stations_v
)

# bind back to training data table
stations_pred <- cbind(
  stations,
  pred_vals[, -1]  # drop terra ID column
)

glimpse(stations_pred)

# write intermediary result to disk
st_write(stations_pred, "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/train_data/04_training_static.gpkg", delete_dsn = TRUE)

# ---- quality checks ----
# check one or two predictors
summary(stations_pred$dtm) # if there is NAs here, one of the sensors is not in the raster, probably not in the gpkg
summary(stations_pred$slope)

# NA rates per predictor (should be 0)
na_frac <- sort(
  colMeans(is.na(st_drop_geometry(stations_pred))),
  decreasing = TRUE
)
na_frac

##############################################################
# ----- + ERA5 HANDLING OF DYNAMIC PREDICTORS  ----- 
##############################################################
# externalized to a separate pyscript because era5 handling in R is a nightmare
# we just load in the parquet file and cnvert it, double check 
# that everything aligns and then we join it
era5_phys <- read_parquet("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/train_data/05_era5_variables.parquet")
df <- as.data.frame(era5_phys)
stations_pred <- st_read("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/train_data/04_training_static.gpkg", quiet = TRUE)

str(era5_phys)
str(stations_pred)

# JOIN WUHUUU
train_joined <- stations_pred %>%
  left_join(
    era5_phys,
    by = c("sensor_id", "time")
  )
str(train_joined)
train_joined <- train_joined %>% st_drop_geometry()

# row count shold not change
nrow(stations_pred)
nrow(train_joined)

# check for missing ERA5 after join
train_joined %>%
  summarise(across(c(t2m, ssrd, u10, v10, tp, wind_s), ~mean(is.na(.))))

##############################################################
# ----- + FURTHER ENCODING AND EXPORT ----- 
##############################################################
train <- train_joined %>%
  mutate(
    hour = lubridate::hour(time),
    hour_sin = sin(2 * pi * hour / 24),
    hour_cos = cos(2 * pi * hour / 24),
    doy = lubridate::yday(time),
    doy_sin = sin(2 * pi * doy / 365),
    doy_cos = cos(2 * pi * doy / 365)
    ) %>%
  select(-hour, -doy)
str(train)
# str(train)
saveRDS(train, "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/train_data/06_final_train.rds")

