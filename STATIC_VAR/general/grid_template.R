
dtm <- rast("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/topo_metrics/topometrics/DTM_10m_Helsinki.tif")

# snap extent to clean 10 m grid
e <- ext(dtm)

xmin_new <- floor(xmin(e) / 10) * 10
ymin_new <- floor(ymin(e) / 10) * 10
xmax_new <- ceiling(xmax(e) / 10) * 10
ymax_new <- ceiling(ymax(e) / 10) * 10

master_ext <- ext(xmin_new, xmax_new, ymin_new, ymax_new)

master_10m <- rast(
  master_ext,
  resolution = 10,
  crs = crs(dtm)
)
values(master_10m) <- NA

writeRaster(master_10m, "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/MASTER_TEMPLATE_10m.tif", overwrite=TRUE)
