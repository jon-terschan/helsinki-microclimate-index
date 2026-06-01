# Fill holes in CHM merger and resample the filled CHM to 10 m.
# Inputs: merged 0.5 m CHM raster and water body polygons.
# Outputs: filled 0.5 m CHM, aligned 0.5 m CHM, and aggregated 10 m CHM.
# -----------------------------------------------------------------------------------------------------------

# ---- header ---
library(terra)
terraOptions(memfrac = 0.8)

# ---- input paths ---

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

# mask inland water and ocean since the CHM is invalid for these areas
ocean <- vect("C:/Users/terschan/Downloads/topo_metrics/lc_sea_hel.gpkg")
inland <- vect("C:/Users/terschan/Downloads/topo_metrics/lc_water_hel.gpkg")
water <- rbind(ocean, inland)
water <- project(water,crs(chm))
water_rast <- rasterize(water, chm, field=1)

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

# step 2, calculate canopy max height (or i guess just general max height, since it includes buildings)
# ---- paths (edit) ----
chm_05_file <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chm_full/CHM_05m_Hel_fill_02.tif" # raw chm
master_temp  <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/MASTER_TEMPLATE_10m.tif" #  master grid template
out_dir      <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chm_full/chm_resampled"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- load ----
chm_05 <- rast(chm_05_file)
master <- rast(master_temp)  

# ---- 1) create a 0.5 m template snapped to master_10m origin ----
master_ext <- ext(master)
master_crs <- crs(master)

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



chm_0_5_aligned <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chm_full/chm_resampled/CHM_0_5m_aligned.tif"    
chm_0_5_aligned <- rast(chm_0_5_aligned)

ext(chm_0_5_aligned)
ext(master)   

   # should be 10 10
print(res(chm_10_max))      # should be 10 10
print(compareGeom(chm_10_max, master, stopOnError = FALSE))  # should be TRUE
print(compareGeom(master, master, stopOnError = FALSE)) 

fact <- 20 # factor 20 because its 0.5 -> 10 m 

# CANOPY MAX HEIGHT
chm_10_max_file <- file.path(out_dir, "CHM_10m_MED.tif")
chm_10_max <- aggregate(chm_0_5_aligned, fact = fact, fun = median, na.rm = TRUE,
                        filename = chm_10_max_file, overwrite = TRUE, 
                        gdal = c(
      "TILED=YES",
      "COMPRESS=ZSTD",
      "PREDICTOR=2",
      "BIGTIFF=YES"
    ))

# ---- 4) verify geometry and coverage ----
print(res(master))          # should be 10 10
print(res(chm_10_max))      # should be 10 10
print(compareGeom(master, chm_10_max, stopOnError = FALSE))  # should be TRUE