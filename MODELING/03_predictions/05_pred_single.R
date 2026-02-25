# pred single time point 
library(terra)
library(dplyr)
library(ranger)

# quick prediction mask, basically if vegetation is present we will predict
tree_fr <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/TREE_FRAC_10m.tif")
nwn_fr  <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/NWN_FRAC_10m.tif")
s <- c(tree_fr, nwn_fr)
pred_mask <- ifel(s[[1]] > 0 | s[[2]] > 0, 1, NA) # mask: 1 if either layer > 0, else NA
writeRaster(pred_mask, 
"//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/PRED_MASK.tif", overwrite = TRUE)
print("lol")
# load input 
# -----------------------------
# 2) Load predictors
# -----------------------------
static_stack <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/full_stack/pred_stack_10m.tif")

era5 <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/04_predictions/era_files/era5_2024-07-15_16_10m.tif")

rf_final <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/03_models/helmi_2000_v1.4_noSMC.rds")

pred_mask <- rast(
"//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/PRED_MASK.tif")

# -----------------------------
# 3) Sanity checks (important)
# -----------------------------
stopifnot(
  compareGeom(static_stack, era5, stopOnError = FALSE)
)

# Ensure names match training exactly
names(era5)
names(static_stack)
pred_stack <- c(static_stack, era5)


# Apply vegetation mask
pred_stack <- mask(pred_stack, pred_mask)
pred_stack <- pred_stack[[names(pred_stack) != "n_als"]]

# token SMC for trest
#smc_r <- pred_stack[[1]]
#values(smc_r) <- 500
#names(smc_r) <- "SMC"
#pred_stack <- c(smc_r, pred_stack)
# -----------------------------
# 5) Predict
# -----------------------------

terraOptions(progress = 1)
terraOptions(memfrac = 0.2)
pred_raster <- terra::predict(
  pred_stack,
  rf_final,
  filename = "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/04_predictions/pred_test.tif",   # writes while computing
  overwrite = TRUE,
  na.rm = TRUE,
  type = "response"
)

rf_final$forest$independent.variable.names
names(pred_stack)
# Optional: mask output again (safer)
pred_raster <- mask(pred_raster, pred_mask)

# -----------------------------
# 6) Export
# -----------------------------
writeRaster(
  pred_raster,
  "prediction_2024-07-15_16.tif",
  overwrite = TRUE
)



bs <- blocks(pred_stack)

out <- rast(pred_stack)
writeStart(out, filename="//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/04_predictions/pred_test.tif", overwrite=TRUE)

for (i in 1:nrow(bs)) {
  v <- terra::values(pred_stack, row=bs$row[i], nrows=bs$nrows[i], dataframe=TRUE)
  p <- predict(rf_final, data=v)$predictions
  writeValues(out, p, bs$row[i])
}

writeStop(out)

ncell(pred_stack)
rf_final$num.trees


library(terra)

# create empty output raster (1 layer only)
out <- rast(pred_stack, nlyr=1)
names(out) <- "prediction"

# define blocks
bs <- blocks(pred_stack)

# start writing
out <- writeStart(
  out,
  filename = "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/04_predictions/pred_test.tif",
  overwrite = TRUE
)

for (i in 1:bs$n) {

  v <- terra::values(
    pred_stack,
    row   = bs$row[i],
    nrows = bs$nrows[i],
    dataframe = TRUE
  )

  # ensure correct column order
  v <- v[, rf_final$forest$independent.variable.names]

  p <- predict(rf_final, data = v)$predictions

  out <- writeValues(out, p, bs$row[i])
}

out <- writeStop(out)