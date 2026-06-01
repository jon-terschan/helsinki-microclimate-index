# Merge post-processed tiles into one final raster.
# Inputs: tiled raster files and a CSV index with inner extents.
# Outputs: merged SVF raster file.
# -----------------------------------------------------------------------------------------------------------

# ---- header ---
library(terra)

# ---- input paths ---
processed_dir <- "processed_tiles"
index_file <- "tiles/tile_inner_index.csv"
# CSV index file with inner extents for each tile

# Read the tile extent index
tile_index <- read.csv(index_file)

# List to collect trimmed tile rasters
tiles_trimmed <- list()

for (i in 1:nrow(tile_index)) {
  # Current tile identifier
  tile_id <- tile_index$tile_id[i]

  # Construct path to the processed tile file
  tile_file <- file.path(
    processed_dir,
    paste0("svf_tile_", sprintf("%02d", tile_id), ".tif")
  )

  # Load the tile raster
  r_tile <- rast(tile_file)

  # Define the inner extent for trimming
  ext_inner <- ext(
    tile_index$xmin[i],
    tile_index$xmax[i],
    tile_index$ymin[i],
    tile_index$ymax[i]
  )

  # Crop the tile to its inner extent
  r_trim <- crop(r_tile, ext_inner, snap="out")

  # Store the trimmed raster
  tiles_trimmed[[i]] <- r_trim

  cat("Trimmed tile", tile_id, "\n")
}

# Merge all trimmed tiles together without resampling
svf_merged <- do.call(mosaic, c(tiles_trimmed, fun="first"))

# Write the final merged SVF raster
writeRaster(
  svf_merged,
  "SVF_merged_final.tif",
  overwrite=TRUE,
  gdal=c("COMPRESS=DEFLATE")
)
