# Fill holes in CHM merger
# Inputs: folder of CHM tiles (or any other raster tiles)
# Outputs: a merged raster 
# -----------------------------------------------------------------------------------------------------------
# careful: NO CRS checking, so unified CRS and resolution/alignment is assumed for the input.

library(terra)
terraOptions(memfrac = 0.8)

in_file  <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chm_full/merged_CHM_05m_Hel.tif"
out_file <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chm_full/merged_CHM_05m_Hel_filled.tif"

chm <- rast(in_file)

# first pass
chm_filled1 <- focal(
  chm,
  w = 3,
  fun = max,
  na.rm = TRUE,
  filename = tempfile(fileext = ".tif"),
  overwrite = TRUE
)
# replace only NA
chm_filled1 <- ifel(
  is.na(chm),
  chm_filled1,
  chm
)
# second pass
chm_filled2 <- focal(
  chm_filled1,
  w = 5,
  fun = max,
  na.rm = TRUE,
  filename = out_file,
  overwrite = TRUE,
  wopt = list(
    datatype = "FLT4S",
    gdal = c(
      "TILED=YES",
      "COMPRESS=ZSTD",
      "PREDICTOR=2",
      "BIGTIFF=YES"
    )
  )
)


chm <- rast(out_file)

ocean <- vect("C:/Users/terschan/Downloads/topo_metrics/lc_sea_hel.gpkg")
inland <- vect("C:/Users/terschan/Downloads/topo_metrics/lc_water_hel.gpkg")
water <- rbind(ocean, inland)
water <- project(water,crs(chm))
water_rast <- rasterize(water, chm, field=1)

plot(water_rast)
chm_filled <- chm 
chm_filled[!is.na(water_rast)] <-0
writeRaster(chm_filled, 
"//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chm_full/CHM_05m_Hel_fill_02.tif", 
overwrite = TRUE,
gdal = c(
      "TILED=YES",
      "COMPRESS=ZSTD",
      "PREDICTOR=2",
      "BIGTIFF=YES"
    ))



pred10 <- rast("C:/Users/terschan/Downloads/topo_metrics/topometrics/SLOPE_10m_Helsinki.tif")
chm_05 <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/MASTER_TEMPLATE_10m.tif")
res(pred10)
res(chm_05)

crs(pred10)
crs(chm_05)

ext(pred10)
ext(chm_05)
compareGeom(pred10, chm_05, stopOnError=FALSE)


library(terra)

# ---- paths (edit) ----
chm_05_file <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chm_full/CHM_05m_Hel_fill_02.tif"       # your raw CHM 0.5 m
master_temp  <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/MASTER_TEMPLATE_10m.tif"
  # your canonical DTM (10 m)
out_dir      <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chm_full/chm_resampled"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- load ----
chm_05 <- rast(chm_05_file)
dtm_10 <- rast(master_temp)  # this is your MASTER 10m

# ---- 1) create a 0.5 m template snapped to master_10m origin ----
master_ext <- ext(dtm_10)
master_crs <- crs(dtm_10)


# create 0.5 m template with exactly same origin/extent as master when aggregated by 20
tpl_0_5 <- rast(ext = master_ext, resolution = 0.5, crs = master_crs)

# ---- 2) align CHM to that 0.5m template (nearest neighbor to avoid smoothing) ----
# This will regrid CHM so its cells line up exactly with subsequent 20x aggregation.
chm_0_5_aligned_file <- file.path(out_dir, "CHM_0_5m_aligned.tif")
chm_0_5_aligned <- resample(chm_05, tpl_0_5, method = "near",
                            filename = chm_0_5_aligned_file, overwrite = TRUE,
                            gdal = c(
      "TILED=YES",
      "COMPRESS=ZSTD",
      "PREDICTOR=2",
      "BIGTIFF=YES"
    ))

ext(chm_0_5_aligned)
ext(dtm_10)          # should be 10 10
print(res(chm_10_max))      # should be 10 10
print(compareGeom(chm_0_5_aligned, dtm_10, stopOnError = FALSE))  # should be TRUE
print(compareGeom(dtm_10, dtm_10, stopOnError = FALSE)) 
# ---- optional: remove extreme spikes BEFORE aggregation (small focal median or percentile cap) ----

fact <- 20 # factor 20 because its 0.5 -> 10 m 

# (A) MAX
chm_10_max_file <- file.path(out_dir, "CHM_10m_MAX.tif")
chm_10_max <- aggregate(chm_0_5_aligned, fact = fact, fun = max, na.rm = TRUE,
                        filename = chm_10_max_file, overwrite = TRUE, 
                        gdal = c(
      "TILED=YES",
      "COMPRESS=ZSTD",
      "PREDICTOR=2",
      "BIGTIFF=YES"
    ))

# (B) P95 (95th percentile)
p95fun <- function(v) {
  if (all(is.na(v))) return(NA)
  as.numeric(quantile(v, probs = 0.95, na.rm = TRUE, type = 7))
}

chm_10_p95_file <- file.path(out_dir, "CHM_10m_P95.tif")
chm_10_p95 <- aggregate(chm_0_5_aligned, fact = fact, fun = p95fun, na.rm = TRUE,
                        filename = chm_10_p95_file, overwrite = TRUE, 
                        gdal = c(
      "TILED=YES",
      "COMPRESS=ZSTD",
      "PREDICTOR=2",
      "BIGTIFF=YES"
    ))

# ---- 4) verify geometry and coverage ----
print(res(dtm_10))          # should be 10 10
print(res(chm_10_max))      # should be 10 10
print(compareGeom(dtm_10, chm_10_max, stopOnError = FALSE))  # should be TRUE
print(compareGeom(dtm_10, dtm_10, stopOnError = FALSE)) 
# coverage fraction (proportion of master cells with CHM data)
frac_cov <- global(!is.na(chm_10_max), "mean", na.rm = TRUE)
print(frac_cov)
