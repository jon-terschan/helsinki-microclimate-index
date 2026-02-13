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

