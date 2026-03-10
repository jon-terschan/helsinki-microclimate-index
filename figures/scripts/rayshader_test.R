# =============================================================================
# 3D Animated GIF: Sub-canopy temperature predictions over Haltiala
# v4: NE corner at top, 60% temp overlay, expanded bbox, OSM visible
# =============================================================================

library(terra)
library(rayshader)
library(magick)
library(grDevices)
library(maptiles)
library(png)

# =============================================================================
# USER SETTINGS
# =============================================================================

PRED_DIR    <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictions/testday/"
DTM_PATH    <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/DTM_10m_Helsinki.tif"
DATE_STR    <- "20240601"
OUT_GIF     <- "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/figures/haltiala_temperature_day.gif"

# Haltiala bounding box in EPSG:3879 — expanded ~500m N, ~200m other sides
BBOX <- ext(
  25494386.282804944,   # xmin  (-200m)
  25496361.098983757,   # xmax  (+200m)
  6681537.543951329,    # ymin  (-200m)
  6683649.691991909     # ymax  (+500m north)
)

UTC_HOURS   <- 1:23
FRAME_DELAY <- 50        # 1/100ths of a second (~1.5s per frame)
GIF_WIDTH   <- 900  
GIF_HEIGHT  <- 800

# 3D render settings
Z_SCALE     <- 5         # natural for ~47m relief
PHI         <- 25        # slightly higher angle suits diagonal view
THETA       <- 10        # NE corner at top
ZOOM        <- 1.2
WINDOWSIZE <- c(1400, 1200)

# 60% temp overlay — OSM roads and features clearly visible underneath
TEMP_ALPHA  <- 0.60

TEMP_PALETTE <- rev(RColorBrewer::brewer.pal(11, "RdYlBu"))

# =============================================================================
# HELPERS
# =============================================================================

utc_to_local_label <- function(utc_h) {
  local_h <- (utc_h + 3) %% 24
  sprintf("%02d:00 local  |  %02d:00 UTC", local_h, utc_h)
}

norm01 <- function(x, lo, hi) {
  out <- (x - lo) / (hi - lo)
  out[is.na(out)] <- 0.5
  pmin(pmax(out, 0), 1)
}

temp_to_rgb_array <- function(temp_rast, temp_min, temp_max, palette) {
  vals    <- as.vector(as.matrix(temp_rast, wide = TRUE))
  normed  <- norm01(vals, temp_min, temp_max)
  pal_fun <- colorRampPalette(palette)
  cols    <- pal_fun(256)[round(normed * 255) + 1]
  rgb_mat <- col2rgb(cols) / 255
  nr <- nrow(temp_rast)
  nc <- ncol(temp_rast)
  out      <- array(0, dim = c(nr, nc, 3))
  out[,,1] <- matrix(rgb_mat[1,], nrow = nr, ncol = nc, byrow = FALSE)
  out[,,2] <- matrix(rgb_mat[2,], nrow = nr, ncol = nc, byrow = FALSE)
  out[,,3] <- matrix(rgb_mat[3,], nrow = nr, ncol = nc, byrow = FALSE)
  out
}

render_legend <- function(temp_min, temp_max, palette, width = 800, height = 80) {
  tmp <- tempfile(fileext = ".png")
  png(tmp, width = width, height = height, bg = "#0d1117")
  par(mar = c(3.5, 2, 0.5, 2), bg = "#0d1117", fg = "white",
      col.axis = "white", col.lab = "white")
  n       <- 256
  pal_fun <- colorRampPalette(palette)
  image(1:n, 1, matrix(1:n, n, 1), col = pal_fun(n), axes = FALSE, xlab = "", ylab = "")
  axis(1,
       at     = seq(1, n, length.out = 6),
       labels = sprintf("%.1f C", seq(temp_min, temp_max, length.out = 6)),
       col = "white", col.ticks = "white", cex.axis = 0.95)
  title(xlab = "Sub-canopy temperature", col.lab = "white", cex.lab = 1.0)
  box(col = "grey50")
  dev.off()
  image_read(tmp)
}

# =============================================================================
# 1. LOAD AND CLIP DTM
# =============================================================================
message("Loading and clipping DTM...")
dtm_full <- rast(DTM_PATH)
dtm_clip <- crop(dtm_full, BBOX)

dtm_mat <- matrix(
  as.vector(t(as.matrix(dtm_clip, wide = TRUE))),
  nrow = ncol(dtm_clip),
  ncol = nrow(dtm_clip)
)

message(sprintf("DTM: %d x %d cells | elevation %.1f - %.1f m",
  ncol(dtm_clip), nrow(dtm_clip),
  global(dtm_clip, "min", na.rm = TRUE)[1,1],
  global(dtm_clip, "max", na.rm = TRUE)[1,1]))

# =============================================================================
# 2. FETCH OSM BASEMAP
# =============================================================================
message("Fetching OSM basemap tile...")
bbox_3879  <- as.polygons(BBOX, crs = "EPSG:3879")
bbox_wgs84 <- project(bbox_3879, "EPSG:4326")

osm_tile <- get_tiles(
  x        = sf::st_as_sf(bbox_wgs84),
  provider = "OpenStreetMap",
  zoom     = 15,
  crop     = TRUE
)

osm_3879 <- resample(project(osm_tile, crs(dtm_clip), method = "bilinear"),
                     dtm_clip, method = "bilinear")

osm_arr <- array(0, dim = c(nrow(dtm_clip), ncol(dtm_clip), 3))
for (b in 1:3) {
  band <- as.matrix(osm_3879[[b]], wide = TRUE)
  band <- band / 255
  band[is.na(band)] <- 0.5
  osm_arr[,,b] <- band
}
message("OSM tile ready.")

# =============================================================================
# 3. GLOBAL TEMPERATURE RANGE
# =============================================================================
message("Scanning prediction files for global temp range...")
all_files <- file.path(PRED_DIR, sprintf("pred_%s_%02d00.tif", DATE_STR, UTC_HOURS))
missing   <- all_files[!file.exists(all_files)]
if (length(missing) > 0) stop("Missing files:\n", paste(missing, collapse = "\n"))

temp_range <- sapply(all_files, function(f) {
  r <- crop(rast(f), BBOX)
  c(global(r, "min", na.rm = TRUE)[1,1],
    global(r, "max", na.rm = TRUE)[1,1])
})
TEMP_MIN <- floor(min(temp_range[1,]))
TEMP_MAX <- ceiling(max(temp_range[2,]))
message(sprintf("Global temp range: %.1f - %.1f C", TEMP_MIN, TEMP_MAX))

# =============================================================================
# 4. STATIC HILLSHADE
# =============================================================================
message("Computing hillshade and ambient occlusion...")
hillshade_mat <- ray_shade(dtm_mat,     zscale = Z_SCALE, multicore = TRUE)
ambient_mat   <- ambient_shade(dtm_mat, zscale = Z_SCALE, multicore = TRUE)

# =============================================================================
# 5. LEGEND
# =============================================================================
legend_img <- render_legend(TEMP_MIN, TEMP_MAX, TEMP_PALETTE)

# =============================================================================
# 6. RENDER FRAMES
# =============================================================================
message(sprintf("Rendering %d frames...", length(UTC_HOURS)))
frame_files <- character(length(UTC_HOURS))

for (i in seq_along(UTC_HOURS)) {
  h <- UTC_HOURS[i]
  message(sprintf("  [%02d/%02d] UTC %02d:00 -> %02d:00 local",
                  i, length(UTC_HOURS), h, (h + 3) %% 24))

  # Load and resample prediction raster
  pred_rast <- resample(crop(rast(all_files[i]), BBOX), dtm_clip, method = "bilinear")

  # Build temperature RGB array
  temp_arr  <- temp_to_rgb_array(pred_rast, TEMP_MIN, TEMP_MAX, TEMP_PALETTE)

# Extract OSM luminance (perceptual weights for R/G/B)
  osm_lum <- osm_arr[,,1] * 0.299 + osm_arr[,,2] * 0.587 + osm_arr[,,3] * 0.114

# Normalize luminance to a subtle modulation range (0.7–1.0)
# so it darkens features slightly without washing out colors
  osm_mod <- 0.7 + 0.3 * (osm_lum - min(osm_lum)) / (max(osm_lum) - min(osm_lum))

# Apply luminance modulation to temperature colors — no alpha blending at all
  blended      <- array(0, dim = c(nrow(dtm_clip), ncol(dtm_clip), 3))
  blended[,,1] <- pmin(temp_arr[,,1] * osm_mod, 1)
  blended[,,2] <- pmin(temp_arr[,,2] * osm_mod, 1)
  blended[,,3] <- pmin(temp_arr[,,3] * osm_mod, 1)

  # Write blended overlay to temp PNG — required for new rayimage backend
  overlay_png <- tempfile(fileext = ".png")
  png::writePNG(aperm(blended, c(2, 1, 3)), overlay_png)

  frame_png <- tempfile(fileext = ".png")

  sphere_shade(dtm_mat, texture = "bw", zscale = Z_SCALE) |>
    add_overlay(overlay_png, alphalayer = 1.0) |>
    add_shadow(hillshade_mat, max_darken = 0.5) |>
    add_shadow(ambient_mat,   max_darken = 0.3) |>
    plot_3d(
      dtm_mat,
      zscale          = Z_SCALE,
      fov             = 0,
      theta           = THETA,
      phi             = PHI,
      zoom            = ZOOM,
      windowsize      = WINDOWSIZE,
      background      = "#0d1117",
      shadow          = TRUE,
      shadow_darkness = 0.7,
      solidcolor      = "grey15",
      solidlinecolor  = "grey15",
      baseshape = "rectangle",
      water           = FALSE
    )

  Sys.sleep(0.3)
  #render_highquality(
  #  frame_png,
  #  samples     = 64,    # higher = smoother but slower, 64 is faster
  #  width       = GIF_WIDTH,
  #  height      = GIF_HEIGHT,
  #  interactive = FALSE
  #)
  render_snapshot(frame_png, software_render = TRUE, clear = TRUE)
  rgl::rgl.close()
  file.remove(overlay_png)

  # Annotate with timestamp and title
  frame_img <- image_scale(image_read(frame_png), paste0(GIF_WIDTH, "x", GIF_HEIGHT))
  frame_img <- image_trim(image_read(frame_png))
  frame_img <- image_scale(frame_img, paste0(GIF_WIDTH, "x", GIF_HEIGHT))
  frame_img <- image_annotate(frame_img,
   text     = sprintf("Haltiala - Near-ground air temperature | %s", utc_to_local_label(h)),
   gravity  = "North",
   location = "+0+12",
   color    = "white",
   size     = 20,
   font     = "DejaVu-Sans-Bold",
   boxcolor = adjustcolor("#0d1117", alpha.f = 0.7))

  legend_scaled  <- image_scale(legend_img, paste0(GIF_WIDTH, "x80"))
  final_frame    <- image_append(c(frame_img, legend_scaled), stack = TRUE)
  frame_files[i] <- tempfile(fileext = ".png")
  image_write(final_frame, frame_files[i])
}

# =============================================================================
# 7. ASSEMBLE GIF
# =============================================================================
message("Assembling GIF...")
animation <- image_animate(image_read(frame_files),
                           delay = FRAME_DELAY, loop = 0, optimize = TRUE)
image_write(animation, OUT_GIF)
file.remove(frame_files)

message(sprintf("Done! -> %s  (%.1f MB)", OUT_GIF, file.size(OUT_GIF) / 1e6))
