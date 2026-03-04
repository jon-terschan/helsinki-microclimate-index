library(terra)

stack <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/full_stack/pred_stack_10m.tif")
water <- stack[["water_fr_10"]]
ocean <- stack[["oce_fr_10"]]

aoi <- vect("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/v2021.gpkg")

aoi <- makeValid(aoi) # not sure if it does anything
aoi <- buffer(aoi, 0.0001)        # dumb solution but seems to fix small topology errors
aoi_outer <- aggregate(aoi)  # dissolve to outer bounds
aoi_outer <- project(aoi_outer, crs(stack))

water_cells <- ocean >= 0.9 | water >= 0.9 # water masking
land_mask   <- ifel(water_cells, NA, 1) 

# rasterize aoi mask
aoi_raster <- rasterize(
  aoi_outer,
  stack[[1]],
  field = 1,
  touches = FALSE
)

aoi_raster[aoi_raster != 1] <- NA

final_mask <- mask(land_mask, aoi_raster) # combine

writeRaster(
  final_mask,
  "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/prediction_mask.tif",
  overwrite = TRUE,
  wopt = list(gdal = "COMPRESS=LZW")
)

