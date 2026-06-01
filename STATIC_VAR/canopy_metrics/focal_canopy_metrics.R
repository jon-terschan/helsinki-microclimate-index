# calculate focal (neighborhood aware) canopy metrics 
# mainly mean scan angle corrected canopy cover and PAI 
# these should be somewhat more accurate, albeit still simplified 
# PAI=−k⋅cos(θ)​ln(GF)​
# GFcorr​=cos(θ)​GF​, this approximates oblique path length by mean scan angle,
# these are of course limited to the scan angle ranges, so they dont consider
# the full hemisphere at all. it is basically truncated to the maximum scan angle, which is around 30-40 degrees for our data.

# this wasnt used in the paper - it is here for reference and future use.
library(lidR)
library(terra)
library(future)

# user paths
input_dir  <- "E:/ALS/stage1_output_12.2/norm"
output_dir <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chm_full/canopy_metrics"
master_template_path <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/MASTER_TEMPLATE_10m.tif"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

output_file <- file.path(output_dir, "CM_scanangle_corrected_10m.tif")

# master grid geometry
master_template <- rast(master_template_path)
crs(master_template) <- "EPSG:3879"

res_master   <- res(master_template)[1]
start_master <- c(xmin(master_template), ymin(master_template))

# emetric calculation parameters
canopy_height       <- 2 # canopy height threshold for GF estimation
z_min               <- 0.2 # minimum height
pai_k               <- 0.5 # extinction coefficient, here simplified based on lit.
min_points_cell     <- 3 # minimum number of points in a cell to consider it valid for metric calculation

metric_names <- c("CLOS_scan", "PAI_scan")

# las catalog setup
ctg <- readLAScatalog(input_dir)

# Build .lax only if missing
if (!all(file.exists(paste0(ctg@data$filename, ".lax")))) {
  cat("Building .lax spatial indices...\n")
  lidR:::catalog_laxindex(ctg)
}
opt_chunk_size(ctg)   <- 800
opt_chunk_buffer(ctg) <- 30
opt_select(ctg)       <- "xyzsa"
opt_progress(ctg)     <- TRUE

# Optional parallelization
plan(multisession, workers = 4)
.options <- list(
  raster_alignment = list(
    res   = res_master,
    start = start_master
  )
)

# angle corrected closure + PAI
cm_fun <- function(Z, ScanAngle) {

  valid <- Z > z_min
  n_valid <- sum(valid)

  if (n_valid < min_points_cell)
    return(list(CLOS_scan=NA_real_, PAI_scan=NA_real_))

  Zf     <- Z[valid]
  theta  <- abs(ScanAngle[valid]) * pi/180

  # Gap fraction (vertical reference)
  gf_mean <- sum(Zf < canopy_height) / n_valid

  if (gf_mean <= 0 || gf_mean >= 1) {
    PAI <- NA_real_
  } else {
    PAI <- -log(gf_mean) / (pai_k * mean(cos(theta)))
  }

  # -------------------------
  # "hemispherical "closure" - naja scan-angle corrected closure
  # Compute directional gap fraction per point
  gap_indicator <- as.numeric(Zf < canopy_height)

  # Hemispherical weights
  w <- sin(theta) * cos(theta)

  # Avoid division by zero
  if (sum(w) == 0) {
    CLOS <- NA_real_
  } else {
    gf_hemi <- sum(w * gap_indicator) / sum(w)
    CLOS <- 1 - gf_hemi
  }

  list(
    CLOS_scan = CLOS,
    PAI_scan  = PAI
  )
}

# process catalog
out <- catalog_map(
  ctg,
  function(las) {

    if (is.empty(las)) return(NULL)

    r <- pixel_metrics(
      las,
      ~cm_fun(Z, ScanAngle),
      res = res_master
    )

    names(r) <- metric_names
    return(r)
  },
  .options = .options
)

# merge chunks and write
merged <- do.call(terra::merge, out)
writeRaster(
  merged,
  output_file,
  overwrite = TRUE,
  wopt = list(
    gdal = c("COMPRESS=DEFLATE", "TILED=YES")
  )
)
