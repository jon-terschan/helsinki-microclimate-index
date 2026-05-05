# CONNECTIVITY / MICROREFUGIA ANALYSIS
# OG: Iris Starck
# adapted for Helsinki heatwave microrefugia analysis
#
# The core logic of the McGuire/Senior connectivity analysis is preserved,
# but the interpretation and design choices are adapted for an urban heatwave context:
# patching -> adjacency -> coolest reachable destination -> connectivity metric.
# In the original paper, patches are built from a temperature gradient, merged within a distance threshold,
# small patches are removed, and each patch is traced to the coolest reachable destination patch.
# Connectivity is current temperature minus destination future temperature. fileciteturn2file0
#
# For this study:
# B = baseline microclimate (normal conditions)
# C = heatwave microclimate (event conditions)
# A = background macroclimate only (diagnostic/context)
#
# Main design choice implemented here:
# 1) Run the analysis for each hour separately.
# 2) Then aggregate across hours to identify persistent refugia and compare event behavior.

library(terra)
library(sf)
library(igraph)
library(dplyr)
library(purrr)
library(lwgeom)
library(tidyverse)

rm(list = ls())
set.seed(1)

# =============================================================================
# 0. USER-DEFINED INPUTS
# =============================================================================

# --- FILE PATHS ---
base_dir <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA"
predictions_base <- file.path(base_dir, "predictions")
predictor_folder <- file.path(base_dir, "predictorstack")
#mask_folder <- file.path(base_dir, "masks")
#A_folder <- file.path(base_dir, "ERA5_background")
result_folder <- file.path(base_dir, "connectivity")

# B predictions live here:
#   .../predictions/baseline/15cm_July_allday/
B_folder <- file.path(predictions_base, "baseline", "15cm_July_allday")

# C predictions live under year-specific folders, with one date folder each:
#   .../predictions/2010/20100728/
#   .../predictions/2018/20180717/
#   .../predictions/2021/20210714/
C_year_folders <- file.path(predictions_base, c("2010", "2018", "2021"))
C_date_folders  <- c(
  file.path(predictions_base, "2010", "20100728"),
  file.path(predictions_base, "2018", "20180717"),
  file.path(predictions_base, "2021", "20210714")
)

# Spatial reference layer
# DECISION POINT: use a Helsinki boundary, study-area polygon, or buffered city boundary.
#study_area_shp <- file.path(mask_folder, "helsinki_boundary.shp")

# Fractional vegetation inputs (raster files; .tif)
# Tree fraction raster and NWN raster are both SpatRaster inputs with values in [0, 1].
# They are stored together in the predictor stack folder.
# DECISION POINT: combine them with pmax(), average them, or keep them separate for sensitivity testing.
tree_files <- list.files(
  predictor_folder,
  pattern = "^TREE_FRAC_.*\\.tif$|^TREE_FRAC.*\\.tif$|^TREE.*\\.tif$",
  full.names = TRUE
)

nwn_files <- list.files(
  predictor_folder,
  pattern = "^NWN_FRAC_.*\\.tif$|^NWN_FRAC.*\\.tif$|^NWN.*\\.tif$",
  full.names = TRUE
)
# Baseline microclimate (B): normal-condition temperatures.
# Example file naming convention from your screenshot:
# pred_20000715_0700.tif
# pred_20000715_0800.tif
# pred_20000715_0900.tif
# ...
B_files <- list.files(
  B_folder,
  pattern = "^pred_\\d{8}_\\d{4}\\.tif$",
  full.names = TRUE
)

# Heatwave microclimate (C): nested year/date folders.
# Example:
# 2010/20100728/pred_20100728_1200.tif
# 2018/20180717/pred_20180717_1200.tif
# 2021/20210714/pred_20210714_1200.tif
# DECISION POINT: if your event files are nested one level deeper or the date stamp differs, adjust the recursive search.
C_files <- list.files(
  C_year_folders,
  pattern = "^pred_\\d{8}_\\d{4}\\.tif$",
  full.names = TRUE,
  recursive = TRUE
)
# Optional background macroclimate (A)
# DECISION POINT: keep as diagnostic/background only, not the main connectivity surface.
# DECISION POINT: keep as diagnostic/background only, not the main connectivity surface.
#A_files <- list.files(A_folder, pattern = "\\.tif$", full.names = TRUE)

# --- ANALYSIS PARAMETERS ---
params <- list(
  crs_target = "EPSG:3879",      # DECISION POINT: confirm CRS suitable for Helsinki.
  res_target  = 10,               # Working resolution in meters; 10 m per pixel.
  veg_threshold = 0.5,            # Threshold for fractional vegetation presence after combining tree + NWN.
  temperature_bin = 0.5,          # DECISION POINT: keep 0.5, use finer bins, or use continuous temperatures.
  patch_gap_m = 100,             # DECISION POINT: original paper used 2000 m; judge whether this is appropriate for Helsinki.
  min_patch_area_m2 = 10000,   # DECISION POINT: original paper used 10 km2 (10000000 m2); judge whether this is appropriate for Helsinki.
  connectivity_mode = "thermal_gain"  # Keep thermal_gain; do not use safety_margin here.
)

# Hour and event handling.
# Filenames are in UTC.
# DECISION POINT: choose the UTC hours you want to compare.
# Example: if you want local Helsinki summer 12:00 / 15:00 / 17:00,
# those correspond to UTC 09:00 / 12:00 / 14:00.
# Keep all analysis in UTC, and apply the local-time offset only when reporting outputs.
extract_timecode <- function(x) {
  as.integer(sub(".*_(\\d{4})\\.tif$", "\\1", basename(x)))
}
available_B_times <- sort(unique(extract_timecode(B_files)))
available_C_times <- sort(unique(extract_timecode(C_files)))

# DECISION POINT: set this manually instead of using the full overlap.
# Use UTC values here, e.g. c(900, 1200, 1400) or c(1200, 1500, 1700).
analysis_hours <- c(900, 1200, 1400)
analysis_hours <- analysis_hours[analysis_hours %in% available_B_times & analysis_hours %in% available_C_times]
analysis_events <- c(2010, 2018, 2021)

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

read_first_raster <- function(x) {
  if (length(x) == 0) return(NULL)
  rast(x[1])
}

# Create patch polygons from a temperature raster.
# This mirrors the paper's temperature binning -> raster-to-polygon step. fileciteturn2file0
build_temperature_patches <- function(temp_rast, bin_size = 0.5, gap_m = 2000) {
  mm <- minmax(temp_rast)
  rmin <- floor(mm[1, 1] / bin_size) * bin_size
  rmax <- ceiling(mm[2, 1] / bin_size) * bin_size
  breaks <- seq(rmin, rmax, by = bin_size)

  if (length(breaks) < 2) stop("Temperature range too narrow for binning.")

  reclass_matrix <- cbind(breaks[-length(breaks)], breaks[-1], seq_len(length(breaks) - 1))
  temp_classes <- classify(temp_rast, reclass_matrix, include.lowest = TRUE)

  temp_polygons <- as.polygons(temp_classes, aggregate = TRUE)
  temp_polygons <- disagg(temp_polygons)
  temp_polygons[["ID"]] <- seq_len(nrow(temp_polygons))
  names(temp_polygons) <- c("Class", "ID")

  sf_polygons <- st_as_sf(temp_polygons)
  sf_use_s2(FALSE)

  sf_polygons$patch_id <- NA_integer_
  next_patch_id <- 1
  classes <- sort(unique(sf_polygons$Class))

  for (class in classes) {
    p <- sf_polygons[sf_polygons$Class == class, , drop = FALSE]
    nb <- st_is_within_distance(p, p, dist = gap_m)
    g <- graph_from_adj_list(nb, mode = "all")
    cl <- components(g)

    p$patch_id <- cl$membership + next_patch_id - 1
    sf_polygons$patch_id[sf_polygons$Class == class] <- p$patch_id
    next_patch_id <- max(sf_polygons$patch_id, na.rm = TRUE) + 1
  }

  polygons_dissolved <- sf_polygons |>
    group_by(patch_id) |>
    summarise(Class = mean(Class), .groups = "drop")

  polygons_dissolved$area_m2 <- as.numeric(st_area(polygons_dissolved))
  polygons_dissolved$area_km2 <- polygons_dissolved$area_m2 / 1e6

  list(
    temp_classes = temp_classes,
    polygons = polygons_dissolved,
    sf_polygons = sf_polygons
  )
}

# Combine tree and NWN fractions into a single vegetation raster.
# DECISION POINT: choose one of the options below.
# Option 1 (default here): vegetation_fraction = pmax(tree_fraction, nwn_fraction)
# Option 2: vegetation_fraction = (tree_fraction + nwn_fraction) / 2
# Option 3: keep tree and NWN separate and run parallel analyses.
combine_vegetation_mask <- function(tree_r, nwn_r, template) {
  tree_r <- project(tree_r, template, method = "bilinear")
  nwn_r  <- project(nwn_r, template, method = "bilinear")
  veg_fraction <- pmax(tree_r, nwn_r)
  veg_fraction
}

# Extract mean values for each patch from a raster stack.
extract_patch_means <- function(rasters, patches_sf) {
  terra::extract(rasters, vect(patches_sf), fun = mean, na.rm = TRUE)
}

# Build adjacency table of touching patches.
# DECISION POINT: touching only, or touching + within small gap? For urban heat refugia this matters.
build_adjacency_table <- function(patches_sf) {
  patches_sf$new_patch_id <- seq_len(nrow(patches_sf))
  neighbors_list <- st_relate(patches_sf, patches_sf, pattern = "F***1****")

  neighbor_table <- do.call(rbind, lapply(seq_along(neighbors_list), function(i) {
    if (length(neighbors_list[[i]]) == 0) return(NULL)
    data.frame(
      src = patches_sf$new_patch_id[i],
      nbr = patches_sf$new_patch_id[neighbors_list[[i]]]
    )
  }))

  neighbor_table
}

# Trace each patch to the coolest reachable destination patch.
# This is the key McGuire/Senior logic, adapted here so the destination is the coolest reachable refuge.
trace_destinations <- function(patches_df, neighbor_table, current_col = "baseline") {
  if (!current_col %in% names(patches_df)) stop("current_col not found in patches_df")

  neighbors <- cbind(neighbor_table, Origin = 0, Dest = 0)

  colnames(neighbors)[1] <- "new_patch_id"
  neighbors <- neighbors %>%
    inner_join(patches_df %>% select(new_patch_id, all_of(current_col)), by = "new_patch_id")
  colnames(neighbors)[1] <- "new_patch_id_src"

  colnames(neighbors)[2] <- "new_patch_id"
  neighbors <- neighbors %>%
    inner_join(patches_df %>% select(new_patch_id, all_of(current_col)), by = "new_patch_id")
  colnames(neighbors)[2] <- "new_patch_id_nbr"

  # Keep this robust to join changes.
  # In the original script, geometry columns were removed by positional index.
  # Here we remove only if needed.
  if (ncol(neighbors) >= 8) neighbors <- neighbors[, -c(6, 8)]

  colnames(neighbors)[colnames(neighbors) == paste0(current_col, ".x")] <- "Mean1"
  colnames(neighbors)[colnames(neighbors) == "new_patch_id_src"] <- "Cores1"
  colnames(neighbors)[colnames(neighbors) == paste0(current_col, ".y")] <- "Mean2"
  colnames(neighbors)[colnames(neighbors) == "new_patch_id_nbr"] <- "Cores2"

  neighbors$Origin <- NA_integer_
  neighbors$Dest   <- NA_integer_

  for (i in seq_len(nrow(neighbors))) {
    if (neighbors$Mean1[i] > neighbors$Mean2[i]) {
      neighbors$Origin[i] <- neighbors$Cores1[i]
      neighbors$Dest[i]   <- neighbors$Cores2[i]
    } else {
      neighbors$Origin[i] <- neighbors$Cores2[i]
      neighbors$Dest[i]   <- neighbors$Cores1[i]
    }
  }

  connections <- setNames(
    lapply(patches_df$new_patch_id, function(pid) {
      neighbors$Dest[neighbors$Origin == pid]
    }),
    patches_df$new_patch_id
  )

  uniquetemps <- sort(unique(patches_df[[current_col]]))
  patches_df$dest <- NA_integer_
  patches_df$dest_temp <- NA_real_
  patches_df$inter_patch <- NA_character_

  running <- patches_df[, c("new_patch_id", current_col)]
  names(running)[2] <- "current"

  for (i in seq_along(uniquetemps)) {
    this_temp <- uniquetemps[i]
    inds <- which(running$current == this_temp)

    for (j in seq_along(inds)) {
      this_pid <- running$new_patch_id[inds[j]]
      dest_pids <- connections[[as.character(this_pid)]]

      if (length(dest_pids) > 0) {
        dest_inds <- match(dest_pids, patches_df$new_patch_id)
        dest_inds <- dest_inds[!is.na(dest_inds)]

        if (length(dest_inds) == 0) {
          patches_df$dest[inds[j]] <- this_pid
          patches_df$dest_temp[inds[j]] <- this_temp
          next
        }

        t <- min(running$current[dest_inds], na.rm = TRUE)
        min_ind <- dest_inds[which.min(running$current[dest_inds])]

        patches_df$dest_temp[inds[j]] <- t
        patches_df$dest[inds[j]] <- running$new_patch_id[min_ind]

        running$current[inds[j]] <- t
        running$new_patch_id[inds[j]] <- running$new_patch_id[min_ind]

        inter_patch <- patches_df$inter_patch[min_ind]
        if (is.na(inter_patch)) {
          patches_df$inter_patch[inds[j]] <- patches_df$new_patch_id[min_ind]
        } else {
          patches_df$inter_patch[inds[j]] <- paste(
            patches_df$new_patch_id[min_ind],
            inter_patch[!is.na(inter_patch)],
            sep = ";"
          )
        }
      } else {
        patches_df$dest[inds[j]] <- this_pid
        patches_df$dest_temp[inds[j]] <- this_temp
      }
    }
  }

  patches_df
}

# Compute the refuge metric.
# DECISION POINT: keep thermal_gain only. Do not use safety_margin in this species-agnostic workflow.
compute_refuge_metric <- function(patches_df, event_temp_col) {
  if (!event_temp_col %in% names(patches_df)) stop("event_temp_col not found in patches_df")

  patches_df$dest_event <- vapply(seq_len(nrow(patches_df)), function(x) {
    dest <- patches_df$dest[x]
    patches_df[[event_temp_col]][patches_df$new_patch_id == dest]
  }, FUN.VALUE = numeric(1))

  patches_df$thermal_gain <- round(patches_df[[event_temp_col]] - patches_df$dest_event, 2)
  patches_df$thermal_anomaly <- round(patches_df[[event_temp_col]] - patches_df$baseline, 2)

  patches_df$refuge_class <- case_when(
    is.na(patches_df$thermal_gain) ~ NA_character_,
    patches_df$thermal_gain > 0 ~ "cooler_reachable_refuge",
    patches_df$thermal_gain == 0 ~ "neutral",
    TRUE ~ "no_cool_refuge"
  )

  patches_df
}

# =============================================================================
# 2. READ DATA
# =============================================================================

#study_area <- vect(study_area_shp)
tree_files <- sort(tree_files)
nwn_files  <- sort(nwn_files)
B_files    <- sort(B_files)
C_files    <- sort(C_files)

B_template <- read_first_raster(B_files)
if (is.null(B_template)) stop("No baseline rasters found in B_folder")

# =============================================================================
# 3. MERGE / PROJECT / HARMONIZE SPATIAL DATA
# =============================================================================
# Read single vegetation rasters
tree_r <- rast(tree_files[1])
nwn_r  <- rast(nwn_files[1])

B_template <- project(B_template, params$crs_target, method = "bilinear")

tree_proj <- project(tree_r, B_template, method = "bilinear")
nwn_proj  <- project(nwn_r, B_template, method = "bilinear")

veg_fraction <- max(tree_proj, nwn_proj)

# Optional A background layer
# DECISION POINT: use A for plots/QC only.
#A_stack <- NULL
#if (length(A_files) > 0) {
#  A_list <- lapply(A_files, rast)
#  A_stack <- rast(A_list)
#  A_stack <- project(A_stack, B_template, method = "bilinear")
#  A_stack <- mask(A_stack, study_area)
#}

# =============================================================================
# 4. BUILD VEGETATION MASK
# =============================================================================

# DECISION POINT: define what counts as habitat/green space.
# Here, vegetation is present where combined tree + NWN fraction exceeds the threshold.
veg_rec <- ifel(veg_fraction > params$veg_threshold, 1, NA)

# =============================================================================
# 5. BUILD PATCH NETWORK FROM BASELINE MICROCLIMATE (B)
# =============================================================================

# B is the core landscape used to define the baseline thermal network.
# A is background only.
# C will be run per event/hour.

B_list  <- lapply(B_files, rast)
B_stack <- rast(B_list)
B_stack <- project(B_stack, B_template, method = "bilinear")
B_stack <- mask(B_stack, veg_rec)

# DECISION POINT: optionally mask B_stack with elevation or other structural constraints.
# This should only be kept if you have a strong urban-ecological reason.
# B_stack <- mask(B_stack, elev)

# Name baseline layers explicitly using the HHMM code from the filenames.
# DECISION POINT: choose which baseline hour should define the patch geometry.
baseline_times <- extract_timecode(B_files)

if (length(baseline_times) == nlyr(B_stack)) {
  names(B_stack) <- paste0("baseline_", sprintf("%04d", baseline_times))
} else {
  names(B_stack) <- paste0("baseline_", sprintf("%04d", seq_len(nlyr(B_stack))))
}

# Use one of the analysis hours (UTC) to define topology
baseline_ref_time <- analysis_hours[1]

layer_name <- paste0("baseline_", sprintf("%04d", baseline_ref_time))

if (!(layer_name %in% names(B_stack))) {
  stop("Chosen baseline_ref_time not found in B_stack")
}

B_topology_layer <- B_stack[[layer_name]]
# DECISION POINT: choose the hour that best matches the main event hour, or use baseline_15 as the default.

patches_obj <- build_temperature_patches(
  temp_rast = B_topology_layer,
  bin_size = params$temperature_bin,
  gap_m = params$patch_gap_m
)

patches_sf <- patches_obj$polygons
patches_sf <- patches_sf %>%
filter(area_m2 >= params$min_patch_area_m2)

# DECISION POINT: the original 10 km2 threshold from the paper is not transferable by default.
# =============================================================================
# 6. EXTRACT BASELINE VALUES FOR EACH PATCH
# =============================================================================

baseline_values <- extract_patch_means(B_stack, patches_sf)
patches_df <- st_drop_geometry(patches_sf)

# Store baseline hourly values dynamically using the HHMM code from the filenames.
# baseline_values has an ID column in column 1, so the raster layers start at column 2.
for (i in seq_along(baseline_times)) {
  col_nm <- paste0("baseline_", sprintf("%04d", baseline_times[i]))
  patches_df[[col_nm]] <- round(baseline_values[, i + 1], 2)
}

# Choose one baseline column as the reference for destination tracing.
# DECISION POINT: use hour-matched baseline for each heatwave hour if possible.
ref_baseline_col <- paste0("baseline_", sprintf("%04d", baseline_ref_time))
if (!ref_baseline_col %in% names(patches_df)) {
  stop("Reference baseline column not found; check file naming and layer import.")
}
patches_df$baseline <- patches_df[[ref_baseline_col]]
# DECISION POINT: use hour-matched baseline for each heatwave hour if possible.
if (!"baseline_15" %in% names(patches_df)) {
  stop("Expected baseline_15 to exist; rename inputs or revise the script.")
}
patches_df$baseline <- patches_df$baseline_15

# =============================================================================
# 7. BUILD PATCH ADJACENCY GRAPH
# =============================================================================
neighbor_table <- build_adjacency_table(patches_sf)

# =============================================================================
# 8. TRACE DESTINATIONS IN BASELINE SPACE
# =============================================================================
patches_df <- trace_destinations(
  patches_df = patches_df,
  neighbor_table = neighbor_table,
  current_col = "baseline"
)

# =============================================================================
# 9. PROCESS HEATWAVE EVENTS / HOURS (C)
# =============================================================================

# Recommended analysis strategy:
# A) run each hour separately: 12, 15, 17
# B) within each event, aggregate across hours to find persistent refugia
# C) compare events (2010 vs 2018 vs 2021)
#
# DECISION POINT: if each heatwave event is a multiband raster, rename bands so the hours are explicit.
# If each hour is already a separate file, use those files directly.

# Helper: find a raster file by event/hour in a file list.
find_event_file <- function(files, year, hour) {
  y <- as.character(year)
  hhmm <- sprintf("%04d", as.integer(hour))
  hit <- files[grepl(y, basename(files)) & grepl(paste0("_", hhmm, "\.tif$"), basename(files))]
  if (length(hit) == 0) return(NA_character_)
  hit[1]
}

# Container for hourly outputs.
hourly_results <- list()

for (yy in analysis_events) {
  for (hh in analysis_hours) {
    f <- find_event_file(C_files, yy, hh)
    if (is.na(f)) next

    event_r <- rast(f)
    event_r <- project(event_r, B_template, method = "bilinear")
    event_r <- mask(event_r, veg_rec)

    # DECISION POINT: if event_r is multiband, select the layer that corresponds to hh.
    # For now, assume a single-layer raster per file.
    event_layer <- event_r

    event_vals <- terra::extract(event_layer, vect(patches_sf), fun = mean, na.rm = TRUE)
    if (ncol(event_vals) < 2) next

    patches_tmp <- patches_df
    patches_tmp$event <- round(event_vals[, 2], 2)
    patches_tmp$event_name <- paste0(yy, "_", sprintf("%04d", hh))
    patches_tmp$event_year <- yy
    patches_tmp$event_hour <- hh

    # Hour-matched baseline comparison if available.
    # DECISION POINT: if a matching baseline hour is missing, the script falls back to the reference baseline hour.
    baseline_col_this_hour <- paste0("baseline_", sprintf("%04d", hh))
    if (baseline_col_this_hour %in% names(patches_tmp)) {
      patches_tmp$baseline <- patches_tmp[[baseline_col_this_hour]]
    } else {
      patches_tmp$baseline <- patches_tmp[[ref_baseline_col]]
    }

    patches_tmp <- compute_refuge_metric(patches_tmp, event_temp_col = "event")

    hourly_results[[paste0(yy, "_", hh)]] <- patches_tmp
  }
}

# =============================================================================
# 10. AGGREGATE WITHIN EVENT ACROSS HOURS
# =============================================================================

# For each event, combine the hourly outputs to identify persistent refugia.
# DECISION POINT: choose aggregation rule:
#   - max thermal_gain = best cooling opportunity across hours
#   - mean thermal_gain = average cooling opportunity
#   - min thermal_gain = worst-case exposure
# Recommended starting point: compute all three and compare.

event_aggregates <- list()

for (yy in analysis_events) {
  event_keys <- names(hourly_results)[grepl(paste0("^", yy, "_"), names(hourly_results))]
  if (length(event_keys) == 0) next

  event_stack <- bind_rows(hourly_results[event_keys], .id = "hour_key")

  summary_tbl <- event_stack %>%
    group_by(new_patch_id) %>%
    summarise(
      n_hours = n(),
      mean_event = mean(event, na.rm = TRUE),
      max_event = max(event, na.rm = TRUE),
      min_event = min(event, na.rm = TRUE),
      mean_thermal_gain = mean(thermal_gain, na.rm = TRUE),
      max_thermal_gain = max(thermal_gain, na.rm = TRUE),
      min_thermal_gain = min(thermal_gain, na.rm = TRUE),
      persistent_refuge_hours = sum(refuge_class == "cooler_reachable_refuge", na.rm = TRUE),
      .groups = "drop"
    )

  summary_tbl$event_year <- yy
  event_aggregates[[as.character(yy)]] <- summary_tbl
}

# =============================================================================
# 11. COMPARISON ACROSS HOURS AND EVENTS
# =============================================================================

# Optional comparison object for downstream plotting/statistics.
all_hourly <- bind_rows(hourly_results, .id = "event_hour_key")
all_event_agg <- bind_rows(event_aggregates, .id = "event_year_key")

# Suggested comparative indicators:
# - which patches are refugia in all hours of a heatwave?
# - which patches are refugia only at 12:00 / 15:00 / 17:00?
# - how stable are destination patches across hours?
# - do heatwaves differ in the spatial persistence of refugia?

persistent_refugia <- all_hourly %>%
  group_by(new_patch_id) %>%
  summarise(
    n_total = n(),
    n_refuge = sum(refuge_class == "cooler_reachable_refuge", na.rm = TRUE),
    prop_refuge = n_refuge / n_total,
    mean_gain = mean(thermal_gain, na.rm = TRUE),
    .groups = "drop"
  )

# =============================================================================
# 12. OUTPUTS
# =============================================================================

# DECISION POINT: save as CSV, shapefile, geopackage, and/or raster.
# Recommended outputs:
# 1) baseline-traced patch table
# 2) hourly refuge tables
# 3) event-level aggregated tables
# 4) persistent refugia table

# write.csv(patches_df, file.path(result_folder, "patches_baseline_traced.csv"), row.names = FALSE)
# write.csv(all_hourly, file.path(result_folder, "all_hourly_refugia.csv"), row.names = FALSE)
# write.csv(all_event_agg, file.path(result_folder, "event_aggregates.csv"), row.names = FALSE)
# write.csv(persistent_refugia, file.path(result_folder, "persistent_refugia.csv"), row.names = FALSE)

# st_write(st_as_sf(patches_df), file.path(result_folder, "patches_baseline_traced.gpkg"), delete_dsn = TRUE)
# st_write(st_as_sf(all_hourly), file.path(result_folder, "hourly_refugia.gpkg"), delete_dsn = TRUE)

# =============================================================================
# 13. INTERPRETATION NOTES
# =============================================================================

# 1) A is background climate context only.
# 2) B defines the baseline thermal structure.
# 3) C defines event stress and refugia.
# 4) Hourly analyses are primary.
# 5) Event-level aggregation is secondary and used to identify persistence.
# 6) The original 10 km2 minimum patch size is a decision point, not a fixed rule for Helsinki.
# 7) The original 2 km patch-aggregation threshold is a decision point, not a fixed rule for Helsinki.
# 8) Bin size, gap distance, and minimum area should be sensitivity-tested.
# 9) Use thermal gain, not safety margin, because the analysis is species-agnostic.
# 10) Use hour-matched baseline vs heatwave if possible.
# 11) Treat destination patches as coolest reachable refugia during the event.
# 12) Tree fraction and NWN fraction are both raster vegetation inputs; the default here combines them with pmax(), but this should be tested against averaging or separate-mask approaches.
# 13) The working raster resolution is 10 m, so all thresholds should be interpreted at that scale.

# End of refactored script
