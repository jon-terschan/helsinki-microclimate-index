# calculate local (neighborhood naive) per pixel canopy metrics
# mainly canopy cover and PAI
# these are very simple 

library(lidR)
library(terra)

# -------------------------
# USER PATHS
# -------------------------
input_dir  <- "E:/ALS/stage1_output_12.2/norm"
output_dir <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/loc_canopy_metrics"
master_template_path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/MASTER_TEMPLATE_10m.tif"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# list input files
las_files <- list.files(
  input_dir,
  pattern = "\\.(las|laz)$",
  full.names = TRUE,
  ignore.case = TRUE
)

# -------------------------
# MASTER GRID TEMPLATE
# -------------------------
# load master grid template and extrect coords
master_template <- rast(master_template_path)
crs(master_template) <- "EPSG:3879"

res_master   <- res(master_template)[1]
start_master <- c(xmin(master_template), ymin(master_template))

metric_names <- c("CC","PAI","CLOS","UCC","N")

# -------------------------
# METRIC PARAMETERS
# -------------------------
canopy_height       <- 2
upper_canopy_height <- 10
z_min               <- 0.2
pai_k               <- 0.5
#min_points_cell     <- 3

# -------------------------
# METRIC FUNCTION
# -------------------------
# -------------------------
# METRIC FUNCTION
# -------------------------
cm_fun <- function(z) {

  N_all <- length(z)

  # NA for completely empty cell
  # should not exist in ALS data at 10m res, but defensive coding yadayada
  if (N_all == 0) {
    return(list(CC=NA_real_, PAI=NA_real_, CLOS=NA_real_, UCC=NA_real_, N=0))
  }

  # Remove very low noise points, these may be artifacts of triangulation
  # and subsequent normalization
  zf <- z[z > z_min]

  # if nothing exists above the ground noise threshold we consider it open ground
  if (length(zf) == 0) {
    return(list(CC=0, PAI=0, CLOS=0, UCC=0, N=N_all))
  }

  # -------------------------
  # gap fraction stabilisation
  # -------------------------
  # so there is many problems in estimating beer-lambert GF from 
  # ALS, one issue is that point returns are a poor indicator of
  # light permeability in certain cases, especially forest edges, where forests are usually
  # extremely permeable due to the frontier.
  # i tried forcing gf away from 0 and 1s using epsilon clamps, but it created
  # insane PAI values in forest ages, so I switched to apply a pseudo-count (Laplace smoothing).
  #
  # This avoids:
  #   gf = 0  (infinite PAI in dense canopy)
  #   gf = 1  (log(1) = 0 edge instability)
  #
  # Ecologically this is maybe a bit better at forest edges where discrete ALS
  # sampling may miss sub-canopy returns, nevertheless
  # i wouldnt call this PAI and more like "PAI"
  #
  # Formula:
  #   gf = (k + 1) / (n + 2)
  #
  # This guarantees:
  #   0 < gf < 1  always
  # -------------------------

  k_ground <- sum(zf < canopy_height)
  n_total  <- length(zf)

  gf  <- (k_ground + 1) / (n_total + 2)

  # canopy metrics
  CC  <- sum(zf > canopy_height) / n_total
  UCC <- sum(zf > upper_canopy_height) / n_total

  # "PAI"
  PAI <- -log(gf) / pai_k

  list(CC=CC, PAI=PAI, CLOS=CC, UCC=UCC, N=N_all)
}

# -------------------------
# BATCH SETTINGS
# -------------------------
batch_size <- 400

existing_batches <- list.files(
  output_dir,
  pattern = "^merged_batch_\\d+\\.tif$",
  full.names = FALSE
)

if (length(existing_batches) > 0) {
  batch_numbers <- as.numeric(
    gsub("merged_batch_|\\.tif", "", existing_batches)
  )
  last_completed_batch <- max(batch_numbers)
} else {
  last_completed_batch <- 0
}

cat("Last completed batch:", last_completed_batch, "\n")

start_tile <- last_completed_batch * batch_size + 1
cat("Resuming from tile:", start_tile, "\n")

batch_list <- list()
batch_id   <- last_completed_batch + 1

# -------------------------
# PROCESS LOOP
# -------------------------
for (i in seq(from = start_tile, to = length(las_files))) {

  cat(sprintf("Processing %d/%d: %s\n",
              i, length(las_files),
              basename(las_files[i])))

  las <- readLAS(las_files[i], select = "xyz")

  if (is.empty(las)) {
    cat("  -> empty tile, skipping\n")
    next
  }

  r_tile <- pixel_metrics(
    las,
    ~cm_fun(Z),
    res   = res_master,
    start = start_master
  )

  if (is.null(r_tile) || nlyr(r_tile) != 5) {
    cat("  -> unexpected layer structure, skipping\n")
    rm(las, r_tile)
    gc()
    next
  }

  names(r_tile) <- metric_names

  batch_list[[length(batch_list) + 1]] <- r_tile

  rm(las, r_tile)
  gc()

  # -------------------------
  # WRITE BATCH
  # -------------------------
  if (length(batch_list) == batch_size || i == length(las_files)) {

    out_file <- file.path(
      output_dir,
      sprintf("merged_batch_%03d.tif", batch_id)
    )

    if (!file.exists(out_file)) {

      cat(sprintf("Writing batch %03d\n", batch_id))

      batch_mosaic <- do.call(mosaic, batch_list)

      writeRaster(
        batch_mosaic,
        out_file,
        overwrite = FALSE,
        gdal = c("COMPRESS=DEFLATE", "TILED=YES")
      )

      rm(batch_mosaic)
      gc()

    } else {
      cat(sprintf("Batch %03d already exists — skipping write\n", batch_id))
    }

    batch_list <- list()
    batch_id   <- batch_id + 1
  }
}

cat("Finished.\n")
