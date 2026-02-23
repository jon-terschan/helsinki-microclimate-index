# pred single time point 
library(terra)

# quick prediction mask, basically if vegetation is present we will predict
tree_fr <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/TREE_FRAC_10m.tif")
nwn_fr  <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/TNWN_FRAC_10m.tif")
s <- c(tree_fr, nwn_fr)
pred_mask <- ifel(s[[1]] > 0 | s[[2]] > 0, 1, NA) # mask: 1 if either layer > 0, else NA
writeRaster(pred_mask, 
"//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/PRED_MASK.tif", overwrite = TRUE)
print("lol")
# load input 
static_stack <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/full_stack/pred_stack_10m.tif")
rf_final <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/03_models/helmi_2000_v1.3_2.rds")
 <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/11.25/ERA/combined/ERA_SUMMER_24_25_HEL.netcdf")
pred_mask <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/PRED_MASK.tif")

# clip raster stack to prediction mask
pred_stack <- mask(pred_stack, pred_mask)


time(era5)

target_time <- as.POSIXct("2024-07-15 16:00:00", tz = "UTC")
idx <- which(time(era5) == target_time)
era5_slice <- era5[[idx]]
era5_crop <- crop(era5_slice, static_stack)

era5_resampled <- resample(
  era5_crop,
  static_stack,
  method = "bilinear"
)

memory.limit() 
sessionInfo()
names(era5_resampled) <- "era5_temp"  # must match training name

pred_stack <- c(static_stack, era5_resampled)

pred_raster <- terra::predict(
  pred_stack,
  rf_final,
  na.rm = FALSE
)

writeRaster(pred_raster, "prediction_2024-07-15_16.tif", overwrite = TRUE)