# =========================================================
# Robust ALS bulk downloader for City of Helsinki server
# =========================================================
# this is a bit more intricate, it creates a download log, reuses the same
# curl connection to avoid excessive throttling and has a progress bar
# input is a txt with the tiles that should be downloaded
# from the initial brute force attempt, i compiled a new tile index that
# can be used to download all valid tiles (much more efficiently)

# effective runtime: 3-4 hours.
# =========================================================

# -------------------------
# Config
# -------------------------
fn <- readLines(
  "C:/Users/terschan/Downloads/helsinki_lidar/custom_index.txt"
)

path    <- "https://ptp.hel.fi/DataHandlers/Lidar_kaikki/Default.ashx"
out_dir <- "C:/Users/terschan/Downloads/helsinki_lidar/tiles/"
year    <- 2021

MIN_SIZE <- 1e6  # 1 MB threshold for LAZ files

# -------------------------
# Setup
# -------------------------
set.seed(1)
fn <- sample(fn)

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

library(curl)
library(pbapply)

h <- new_handle(
  followlocation = TRUE,
  timeout = 60,
  useragent = "Helsingin yliopisto – ALS bulk download"
)

# -------------------------
# Results table
# -------------------------
results <- data.frame(
  tile      = fn,
  exists    = FALSE,
  file_size = NA_real_,
  status    = NA_character_,
  timestamp = as.POSIXct(NA),
  stringsAsFactors = FALSE
)

# -------------------------
# Progress bar
# -------------------------
pb <- pbapply::startpb(0, length(fn))
on.exit(pbapply::closepb(pb))

# -------------------------
# Download loop
# -------------------------
for (i in seq_along(fn)) {
  
  tile <- fn[i]
  url  <- paste0(path, "?q=", tile, "&y=", year)
  
  final <- file.path(out_dir, paste0(tile, ".laz"))
  tmp   <- paste0(final, ".tmp")
  
  # ---- resume-safe skip ----
  if (file.exists(final)) {
    size <- file.info(final)$size
    if (!is.na(size) && size >= MIN_SIZE) {
      results$exists[i]    <- TRUE
      results$file_size[i] <- size
      results$status[i]    <- "already_present"
      results$timestamp[i] <- Sys.time()
      pbapply::setpb(pb, i)
      next
    }
  }
  
  if (file.exists(tmp)) unlink(tmp)
  
  ok <- tryCatch(
    {
      curl_download(
        url,
        destfile = tmp,
        handle = h,
        quiet = TRUE
      )
      TRUE
    },
    error = function(e) FALSE
  )
  
  if (ok && file.exists(tmp)) {
    size <- file.info(tmp)$size
    
    if (!is.na(size) && size >= MIN_SIZE) {
      file.rename(tmp, final)
      results$exists[i]    <- TRUE
      results$file_size[i] <- size
      results$status[i]    <- "downloaded"
    } else {
      unlink(tmp)
      results$status[i] <- "not_available"
    }
  } else {
    results$status[i] <- "error"
  }
  
  results$timestamp[i] <- Sys.time()
  
  pbapply::setpb(pb, i)
  
  # polite throttling only for real attempts
  if (results$status[i] %in% c("downloaded", "error")) {
    Sys.sleep(0.1)
  }
}

# -------------------------
# Write CSV log
# -------------------------
write.csv(
  results,
  file = "C:/Users/terschan/Downloads/helsinki_lidar/als_download_log.csv",
  row.names = FALSE
)

# -------------------------
# Done
# -------------------------
message("Complete download finished. Check download log")

# =========================================================
# Post-Download file handling
# =========================================================
# =========================================================
library (dplyr)
library(lidR)
library(sf)

# read the als download log and reexport it with only valid tiles
# output can be used as a tile index to skip the brute force attempt
log <- read.csv("C:/Users/terschan/Downloads/helsinki_lidar/als_download_log_appended.csv", sep = ";")
log_filtered <- log %>%
  filter(exists == TRUE)
write.csv2(log_filtered, "C:/Users/terschan/Downloads/helsinki_lidar/als_existing_tiles.csv")

# read as lascatalog
ctg <- readLAScatalog(
  "C:/Users/terschan/Downloads/helsinki_lidar/tiles/"
)
#plot(ctg, map = TRUE, col = "grey80", border = "black")

# create and export tile extent as polygons
ext <- st_as_sf(ctg)
st_crs(ext) <- 3879 # assign project CRS

st_write(
  ext,
  "C:/Users/terschan/Downloads/helsinki_lidar/als_tile_outlines.gpkg",
  layer = "als_tiles",
  delete_layer = TRUE
)

###########################################################
####### DEPRECATED SINCE BACKEND LOCATION MOVED ###########
###########################################################
# this is the original script my colleague gave me,
# it does no longer work since the backend moved. 

# Download laserscanning data from Helsinki Map Service at:
# https://kartta.hel.fi/?setlanguage=en


# Read filenames from file (2021)
# Make list based on the file names in paths below
#fn <- read.csv("C:/Users/terschan/Downloads/helsinki_lidar/tiles_2021.txt")
#fn <- unlist(fn, use.names=FALSE)

# Download files!
# TryCatch is needed as some files do not exist
#path <- "https://kartta.hel.fi/helshares/Laserkeilausaineistot/Pintamalli/2017_LAZ/"
#path <- "https://kartta.hel.fi/helshares/Laserkeilausaineistot/Pintamalli/2015_LAZ/"
#path <- "https://kartta.hel.fi/helshares/Laserkeilausaineistot/Pintamalli/2021_LAZ/"
#path <- "https://kartta.hel.fi/link/aw4R8i"

#for(file in fn) {
#  url    <- paste0(path, file)
#output <- paste0("/scratch/project_2002648/helsinki_2015/", file) #remember "/" in the end of path!
#output <- paste0("/scratch/project_2002648/helsinki_2017/west/", file)
#  output <- paste0("C:/Users/terschan/Downloads/helsinki_lidar/tiles/", file)
#  tryCatch(
#    download.file(url, destfile=output, mode="wb"),
#    error = function(e) print(paste(file, 'did not work out')))
#}


# If you had to stop downloading, update 'fn' to continue
#fn.done <- list.files("E:/Helsinki_lidar/2021/")
#fn <- setdiff(fn, fn.done)
# Continue by running the previous for-loop.

#########################################################
###############CREATE CONSERVATIVE INDEX#################
#########################################################
# to create a conservative tilename index for downloading and test which 
# files exist by brute force

#tiles   <- 667489:687489
#letters <- c("a", "b", "c", "d")

#index <- as.vector(outer(tiles, letters, paste0))

#writeLines(index, "C:/Users/terschan/Downloads/helsinki_lidar/als_mapsheet_index.txt")

###########################################################
###################DEPRECATED: NEW BACKEND #################
###########################################################
# uses tilename index to download data
# deprecated because each download is a separate network connection attempt
# which quickly leads to the server throttling download
# but essentially, this would work and is quite simple

#fn <- readLines("C:/Users/terschan/Downloads/helsinki_lidar/als_mapsheet_index.txt")

# city of helsinki sneakily changed the endpoint where the files are without documentation
# but the new endpoint can be reverse engineered from the client-request:
# https://ptp.hel.fi/DataHandlers/Lidar_kaikki/Default.ashx?q=676495c&y=2021
# in the URL: q = file id (coordinate + abcd, y = year)

#fn <- gsub("^rgb_|\\.laz$", "", fn) # trim strings to correct length, not needed with new file

# new download hander base
#path <- "https://ptp.hel.fi/DataHandlers/Lidar_kaikki/Default.ashx"

#for (tile in fn) {
#  url <- paste0(path, "?q=", tile, "&y=2021")
#  output <- paste0(
#    "C:/Users/terschan/Downloads/helsinki_lidar/tiles/", # change to output folder
#    tile,
#    ".laz"
#  )

#  tryCatch(
#    download.file(
#      url,
#      destfile = output,
#      mode = "wb",
#      method = "libcurl"
#    ),
#    error = function(e) message(tile, " did not work out")
#  )
#  Sys.sleep(0.1) # artificial lag to prevent ddosing helsinki server hihihi, probably superfluyous :)
#}

