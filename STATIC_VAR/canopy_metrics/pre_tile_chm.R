# DEPRECATED because we didnt use skyview factor in this paper
# here i had a moment 
# where i decided to split the CHM into different tiles
# in order to run SVF calculation embarassingly parallel 
library(terra)

# -----------------------------
# USER SETTINGS
# -----------------------------
input_file <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chm_full/chm_resampled/CHM_0_5m_aligned.tif"       
master_temp  <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/MASTER_TEMPLATE_10m.tif"
output_dir  <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/chm_full/tiles"

dir.create(output_dir, showWarnings = FALSE)

n_tiles_per_side <- 5       # 5x5 grid
overlap_pixels   <- 200     # 200 pixels = 100 m at 0.5 m

# -----------------------------
# LOAD RASTER
# -----------------------------


r <- rast(input_file)
mast <-rast(master_temp)
ext(mast)
ext(r)
nrows_total <- nrow(r)
ncols_total <- ncol(r)

res_x <- res(r)[1]
res_y <- res(r)[2]

cat("Raster size:", nrows_total, "rows x", ncols_total, "cols\n")

library(terra)

# Basic raster info
nrows_total <- nrow(r)
ncols_total <- ncol(r)

base_rows <- floor(nrows_total / n_tiles_per_side)
base_cols <- floor(ncols_total / n_tiles_per_side)

tile_id <- 1
index_list <- list()

for (row_index in 0:(n_tiles_per_side - 1)) {
  for (col_index in 0:(n_tiles_per_side - 1)) {

    # ---------------------------------
    # INNER TILE LIMITS (no overlap)
    # ---------------------------------

    row_start <- row_index * base_rows + 1
    col_start <- col_index * base_cols + 1

    if (row_index == n_tiles_per_side - 1) {
      tile_rows <- nrows_total - row_index * base_rows
    } else {
      tile_rows <- base_rows
    }

    if (col_index == n_tiles_per_side - 1) {
      tile_cols <- ncols_total - col_index * base_cols
    } else {
      tile_cols <- base_cols
    }

    inner_row_start <- row_start
    inner_row_end   <- row_start + tile_rows - 1

    inner_col_start <- col_start
    inner_col_end   <- col_start + tile_cols - 1

    # ---------------------------------
    # BUFFERED LIMITS (with overlap)
    # ---------------------------------

    row_start_buf <- max(1, inner_row_start - overlap_pixels)
    col_start_buf <- max(1, inner_col_start - overlap_pixels)

    row_end_buf <- min(
      nrows_total,
      inner_row_end + overlap_pixels
    )

    col_end_buf <- min(
      ncols_total,
      inner_col_end + overlap_pixels
    )

    # ---------------------------------
    # Convert BUFFERED row/col → extent
    # ---------------------------------

    cell_min_buf <- cellFromRowCol(r, row_end_buf, col_start_buf)
    cell_max_buf <- cellFromRowCol(r, row_start_buf, col_end_buf)

    xy_min_buf <- xyFromCell(r, cell_min_buf)
    xy_max_buf <- xyFromCell(r, cell_max_buf)

    ext_buf <- ext(
      xy_min_buf[1],  # xmin
      xy_max_buf[1],  # xmax
      xy_min_buf[2],  # ymin
      xy_max_buf[2]   # ymax
    )

    tile <- crop(r, ext_buf, snap = "out")

    # ---------------------------------
    # WRITE TILE
    # ---------------------------------

    out_name <- file.path(
      output_dir,
      paste0("chm_tile_", sprintf("%02d", tile_id), ".tif")
    )

    writeRaster(
      tile,
      out_name,
      overwrite = TRUE,
      gdal = c("COMPRESS=DEFLATE")
    )

    cat("Wrote tile", tile_id, "\n")

    # ---------------------------------
    # Convert INNER row/col → extent
    # ---------------------------------

    cell_min_inner <- cellFromRowCol(r, inner_row_end, inner_col_start)
    cell_max_inner <- cellFromRowCol(r, inner_row_start, inner_col_end)

    xy_min_inner <- xyFromCell(r, cell_min_inner)
    xy_max_inner <- xyFromCell(r, cell_max_inner)

    index_list[[tile_id]] <- data.frame(
      tile_id = tile_id,
      xmin = xy_min_inner[1],
      xmax = xy_max_inner[1],
      ymin = xy_min_inner[2],
      ymax = xy_max_inner[2]
    )

    tile_id <- tile_id + 1
  }
}

# ---------------------------------
# SAVE INNER EXTENT INDEX
# ---------------------------------

tile_index <- do.call(rbind, index_list)

write.csv(
  tile_index,
  file.path(output_dir, "tile_inner_index.csv"),
  row.names = FALSE
)

cat("Saved tile_inner_index.csv\n")
