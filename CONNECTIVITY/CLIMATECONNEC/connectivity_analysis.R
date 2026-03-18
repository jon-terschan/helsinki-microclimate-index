## CONNECTIVTY ANALYSIS
## ORIGINAL AUTHOR: IRIS STARCK, ADAPTED TO THIS STUDY

library(terra)
library(sf)
library(igraph)
library(dplyr)
library(purrr)
library(lwgeom)
library(tidyverse)

setwd("D:/POSTDOC/Data")

rm(list=ls())

#Define folders
forest_folder <- "./Forest_2010"
elev_folder <- "./Elevation_STRM"
macro_folder <- "./ERA5_Land/past"
micro_folder <- "./MIcroclimate_Zhimin/Microclimate_mean/past/mean"
micro_245_folder <- "./MIcroclimate_Zhimin/Microclimate_mean/ssp245/mean"
micro_585_folder <- "./MIcroclimate_Zhimin/Microclimate_mean/ssp585/mean"
macro_245_folder <- "./ERA5_Land/ssp245"
macro_585_folder <- "./ERA5_Land/ssp585"
amazon_shp_folder <- "./Amazon_boundry"
result_folder <- "./Outputs"

#Read files

forest_files <- list.files(forest_folder, pattern = "^treecover.*\\.tif$", full.names = TRUE)
amazon <- vect(paste0(amazon_shp_folder, "./amazon_bio.shp"))
micro_t <- rast(paste0(micro_folder,"./cropped_masked_predicted_multiband_image_2013_2022.tif"))
macro_t <- rast(paste0(macro_folder, "./mean_temperature_2013_2022.tif"))
elev <- rast(paste0(elev_folder, "/Elevation_merged.tif"))

######################################################################################### 
#                                                                                       #
#                      WORKFLOW FOR CALCULATING CLIMATE CONNECTIVITY                    #
#                                                                                       #
######################################################################################### 
# 1. Merge forest files, resample all rasters to EPSG:5880 and desired resolution

# Read all the forest rasters into a list of SpatRaster objects
ras_list <- lapply(forest_files, rast)

# Merge all forest rasters into one
treecover <- do.call(merge, ras_list)

#Crop forest cover to Amazon boundary
treecover <- crop(treecover, amazon)

#writeRaster(treecover, paste0(forest_folder, "/forest_cover_merged.tif"), overwrite = TRUE)

#Crop elevation to Amazon boundary
elev <- mask(elev, amazon)

#Define resolution
#reso = 1000

#Resample all rasters and Amazon polygon

amazon <- project(amazon, "EPSG:5880")
micro_t_resampled <- project(micro_t, "EPSG:5880", method = "bilinear")
macro_t_resampled <- project(macro_t, micro_t_resampled, method = "bilinear")
elev <- project(elev, micro_t_resampled, method = "bilinear")
#elev <- project(elev, macro_t_resampled, method = "bilinear")
forest <- project(treecover, micro_t_resampled, method = "bilinear")
#forest <- project(treecover, macro_t_resampled, method = "bilinear")
forest_masked <- mask(forest, amazon)

#writeRaster(macro_t_resampled, paste0(macro_folder, "/macro_t_resampled_5880_1km.tif"), overwrite = TRUE)
#writeRaster(micro_t_resampled, paste0(micro_folder, "/micro_t_resampled_5880.tif"), overwrite = TRUE)
#writeRaster(elev, paste0(elev_folder, "/elev_micro_res_5880.tif"), overwrite = TRUE)
#writeRaster(forest_masked, paste0(forest_folder, "/forest_masked_era5_res_5880.tif"), overwrite = TRUE)

# 1. Reclassify to forest an non-forest

# Reclassify to forest and non-forest: <=50% is non-forest, >50% is forest.
forest_rec <- ifel(forest_masked > 50, 1, NA)

#Write raster
#writeRaster(forest_rec, paste0(forest_folder,"/forest_mask_classified_era5_res.tif"), overwrite = TRUE)

# 2. Crop and remove high-elevation pixels from the elevation file

#writeRaster(elev, "elev_mask.tif", overwrite = TRUE)

#Remove pixels >600 meters
elev[elev > 600] <- NA
#writeRaster(elev, paste0(elev_folder, "/elev_mask_600m_era5_res.tif"), overwrite = TRUE)

# 4. Mask non-forest and high-elevation pixels from the temperature map

# Mask using the forest pixels
micro_t_forest <- mask(micro_t_resampled, forest_rec)
#macro_t_forest <- mask(macro_t_resampled, forest_rec)

# Mask high-elevation pixels
micro_t_forest <- mask(micro_t_forest, elev)
#macro_t_forest <- mask(macro_t_forest, elev)

#Write final temperature raster
writeRaster(micro_t_forest, paste0(micro_folder, "./micro_past_forest_elev_masked.tif"))

# 5. Reclassify temperature pixels with 0.5 degree intervals and create polygons

#Create working file
#temp <- macro_t_forest #Change according to your temperature data, all future processes are done using "temp"
temp <- micro_t_forest

# Get min and max temperatures
rmin <- floor(minmax(temp)[1,1])
rmax <- ceiling(minmax(temp)[2,1])

# Create 0.5 interval breakpoints (always at .0 and .5)
breaks <- seq(rmin, rmax, by = 0.5)

#Build reclassification matrix
reclass_matrix <- cbind(breaks[-length(breaks)], breaks[-1], 1:(length(breaks)-1))

#Reclassify temperature raster
temp_classes <- classify(temp, reclass_matrix, include.lowest = TRUE)

#writeRaster(temp_classes, "macro_t_0.5deg_classes.tif", overwrite = TRUE)

# Convert raster to polygons
# You need to first aggregate and then disaggregate, otherwise you lose information
temp_polygons <- as.polygons(temp_classes, aggregate = TRUE) #you need to first aggregate
#Disaggregate
temp_polygons_disagg <- disagg(temp_polygons)

#Add unique polygon ID column
temp_polygons_disagg[["ID"]] <- 1:nrow(temp_polygons_disagg)

#Rename value column
names(temp_polygons_disagg) <- c("Class","ID")


# 6. Find neighbors within distance for each polygon and merge into final forest patches

#Define maximum distance of two patches to count as one
dist = 2000

#Get unique temperature classes to treat each one separately
classes <- unique(temp_polygons_disagg$Class)

# Convert SpatVect polygons to sf
sf_polygons <- st_as_sf(temp_polygons_disagg)

# Disable S2 geometry (for planar distances)
sf_use_s2(FALSE)

# Compute neighbors within a distance
sf_polygons$patch_id <- NA_integer_  # initialize
next_patch_id <- 1            # global counter

for (class in classes) {
  
  print(paste0("Class = ", class))
  #Get class subset
  p <- sf_polygons[sf_polygons$Class == class, ]
  #Find polygons within distance
  print("Finding neighbors")
  nb <- st_is_within_distance(p, p, dist = dist)
  
  # build graph and get connected components
  print("Building graph")
  g <- graph_from_adj_list(nb, mode = "all")
  cl <- components(g)
  
  # assign global patch IDs
  print("Assigning patch IDs")
  p$patch_id <- cl$membership + next_patch_id - 1
  
  # update the full object
  print("Updating original polygons")
  sf_polygons$patch_id[sf_polygons$Class == class] <- p$patch_id
  
  # increment the counter
  next_patch_id <- max(sf_polygons$patch_id, na.rm = TRUE) + 1
}

#st_write(sf_polygons, "sf_polygons.shp") 

# Dissolve polygons based on patch ID
polygons_dissolved <- sf_polygons |>
  dplyr::group_by(patch_id) |>
  dplyr::summarise(Class = mean(Class))

#st_write(polygons_dissolved, "micro_polygons_dissolved.shp") 
#st_write(polygons_dissolved, "macro_polygons_micro_res_dissolved.shp") 

# Add area column
polygons_dissolved$area_m2 <- round(st_area(polygons_dissolved),0)

#Convert area to km2
polygons_dissolved <- polygons_dissolved %>%
  mutate(area_km2 = as.numeric(st_area(geometry)) / 1e6)

#Remove polygons with area smaller than 10 km2
polygons_filtered <- polygons_dissolved %>%
  filter(area_km2 >= 10)

#st_write(polygons_filtered, paste0(result_folder, "./micro_polygons_filtered_10km2.shp")) 

# Now, you have the final forest patches defined.

# 7. Calculate connectivity

# First, get current and future temperatures for each patch

#Read future scenario rasters and reproject them
micro_245 <- rast(paste0(micro_245_folder,"./cropped_masked_processed_multiband_image_ssp245_2093-2100.tif"))
micro_585 <- rast(paste0(micro_585_folder,"./cropped_masked_processed_multiband_image_ssp585_2093-2100.tif"))
micro_245<- project(micro_245, micro_t_resampled, method = "bilinear")
micro_585<- project(micro_585, micro_t_resampled, method = "bilinear")

macro_245 <- rast(paste0(macro_245_folder,"./CMIP6_MeanTemperature_ssp245_2093-2100.tif"))
macro_585 <- rast(paste0(macro_585_folder,"./CMIP6_MeanTemperature_ssp585_2093-2100.tif"))
macro_245<- project(macro_245, macro_t_resampled, method = "bilinear")
macro_585<- project(macro_585, macro_t_resampled, method = "bilinear")

#writeRaster(micro_245, paste0(micro_245_folder, "/micro_245_micro_res_2093-2100_resampled.tif"))
#writeRaster(micro_585, paste0(micro_585_folder, "/micro_585_micro_res_2093-2100_resampled.tif"))
#writeRaster(macro_245, paste0(macro_245_folder, "/macro_245_micro_res_2093-2100_resampled.tif"))
#writeRaster(macro_585, paste0(macro_585_folder, "/macro_585_micro_res_2093-2100_resampled.tif"))

#Extract current and future temperatures for polygons

rasters <- c(micro_t_resampled, micro_245, micro_585)
#rasters <- c(macro_t_resampled, macro_245, macro_585)

# Extract mean temperature values of current, ssp245 and ssp585 temperatures per polygon
mean_values <- terra::extract(rasters, vect(polygons_filtered), fun = mean, na.rm = TRUE)

#Add mean temperatures as new columns in polygons
polygons_filtered$current <- round(mean_values[,2],1)
polygons_filtered$ssp245 <- round(mean_values[,3],1) 
polygons_filtered$ssp585 <- round(mean_values[,4],1)  

#st_write(polygons_filtered, "macro_final_polygons_micro_res.shp", overwrite=TRUE) 

# 9. Calculate connectivity

#Create new ID column for final patches
polygons_filtered$new_patch_id <- seq_len(nrow(polygons_filtered))

# Find neighbors for each polygon
neighbors_list <- st_relate(polygons_filtered, polygons_filtered, pattern = "F***1****")

#Make into dataframe (to match ArcGIS output)
neighbor_table <- do.call(rbind, lapply(seq_along(neighbors_list), function(i) {
  if (length(neighbors_list[[i]]) == 0) return(NULL)
  data.frame(
    src = polygons_filtered$new_patch_id[i],
    nbr = polygons_filtered$new_patch_id[neighbors_list[[i]]]
  )
}))

# From now on, the code is from McGuire et al.

#First we will determine the Origin (warmer) & Dest (cooler) Cores for all adjacent core pairs
neighbors<-cbind(neighbor_table,Origin=0, Dest=0)

#joining the temperature of each set of cores (Core1 & Core2)
colnames(neighbors)[1]<-"new_patch_id"
#neighbors<-join(neighbors, temp, type= "inner", by= "patch_id_p")
neighbors <- neighbors %>%
  inner_join(polygons_filtered %>% select(new_patch_id, current), by = "new_patch_id")
colnames(neighbors)[1]<-"new_patch_id_src"

colnames(neighbors)[2]<-"new_patch_id"
#neighbors<-join(neighbors, temp, type= "inner", by= "patch_id_f")
neighbors <- neighbors %>%
  inner_join(polygons_filtered %>% select(new_patch_id, current), by = "new_patch_id")
colnames(neighbors)[2]<-"new_patch_id_nbr"

#Remove geometry columns
neighbors <- neighbors[,-c(6,8)]

#Rename columns
colnames(neighbors)[colnames(neighbors)=="current.x"]<-"Mean1"
colnames(neighbors)[colnames(neighbors)=="new_patch_id_src"]<-"Cores1"
colnames(neighbors)[colnames(neighbors)=="current.y"]<-"Mean2"
colnames(neighbors)[colnames(neighbors)=="new_patch_id_nbr"]<-"Cores2"

n<-nrow(neighbors)

#lists the hotter core # as the "Origin" & cooler core # as the "Dest"
for (i in 1:n)
{
  if (neighbors$Mean1[i]>neighbors$Mean2[i])
  {neighbors$Origin[i]<-neighbors$Cores1[i]
  neighbors$Dest[i]<-neighbors$Cores2[i]}
  
  else {neighbors$Origin[i]<-neighbors$Cores2[i]
  neighbors$Dest[i]<-neighbors$Cores1[i]}
}

colnames(neighbors)[5]<-"MeanO"
colnames(neighbors)[6]<-"MeanD"

#Identify neighbors
#First, transform polygons to dataframe to avoid issues with geometry
polygons_df <- st_drop_geometry(polygons_filtered)

#FIX:
connections <- setNames(
  lapply(polygons_df$new_patch_id, function(pid) {
    neighbors$Dest[neighbors$Origin == pid]
  }),
  polygons_df$new_patch_id
)

tail(connections)

## Determine the ﬁnal, coolest destination patch
#Iterate over unique patch temperatures from cooler to hotter
uniquetemps <- sort(unique(polygons_df$current)) 
uniquetemps

# Set up output columns 
polygons_df$dest <- NA 
polygons_df$dest_temp <- NA 
polygons_df$inter_patch <- NA

# Copy the original dataframe
running <- polygons_df[, c("new_patch_id", "current")]

# New:

for (i in 1:length(uniquetemps)) {
  
  this_temp <- uniquetemps[i]
  inds <- which(running$current == this_temp)
  
  for (j in 1:length(inds)) {
    
    # ✔ NEW: always get neighbors by patch ID — not row index
    this_pid  <- running$new_patch_id[inds[j]]
    dest_pids <- connections[[as.character(this_pid)]]
    
    if (length(dest_pids) > 0) { 
      
      # ✔ NEW: convert destination patch IDs → row indices
      dest_inds <- match(dest_pids, polygons_df$new_patch_id)
      
      # minimum destination temperature
      t <- min(running$current[dest_inds], na.rm = TRUE)
      min_ind <- dest_inds[which.min(running$current[dest_inds])]
      
      # ---- everything below here is YOUR existing logic ----
      
      polygons_df$dest_temp[inds[j]] <- t 
      polygons_df$dest[inds[j]]      <- running$new_patch_id[min_ind]
      
      running$current[inds[j]]       <- t
      running$new_patch_id[inds[j]]  <- running$new_patch_id[min_ind]
      
      inter_patch <- polygons_df$inter_patch[min_ind]
      if (is.na(inter_patch)) {
        polygons_df$inter_patch[inds[j]] <- polygons_df$new_patch_id[min_ind]
      } else {
        polygons_df$inter_patch[inds[j]] <-
          paste(
            polygons_df$new_patch_id[min_ind],
            inter_patch[!is.na(inter_patch)],
            sep = ";"
          )
      }
      
    } else {
      # no neighbors → stays where it started
      polygons_df$dest[inds[j]]      <- this_pid
      polygons_df$dest_temp[inds[j]] <- this_temp
    }
  }
}

# Connectivity

#Assign destination future temperature
polygons_df$dest_ftemp_ssp245 <-
  vapply(1:nrow(polygons_df), function(x){
    dest <- polygons_df$dest[x]
    dest_ftemp <- polygons_df$ssp245[polygons_df$new_patch_id == dest] 
    return(dest_ftemp)
  }, FUN.VALUE = numeric(1))

polygons_df$dest_ftemp_ssp585 <-
  vapply(1:nrow(polygons_df), function(x){
    dest <- polygons_df$dest[x]
    dest_ftemp <- polygons_df$ssp585[polygons_df$new_patch_id == dest] 
    return(dest_ftemp)
  }, FUN.VALUE = numeric(1))

polygons_df$clim_conn_245 <- round(polygons_df$current - polygons_df$dest_ftemp_ssp245,1)
polygons_df$clim_conn_585 <- round(polygons_df$current - polygons_df$dest_ftemp_ssp585,1)

write.csv(polygons_df, paste0(result_folder, "/micro_connectivity_era5_res.csv"))

# Merge connectivity information to forest polygons

polygons_connectivity <- polygons_filtered %>%
  left_join(polygons_df %>%
              select(new_patch_id, clim_conn_245, clim_conn_585),
            by = "new_patch_id")

# Save new shapefile
st_write(polygons_connectivity, paste0(result_folder, "/polygons_micro_connectivity_era5_res.shp"))

# Convert to raster

connectivity_245 <- rasterize(polygons_connectivity, temp_classes, field="clim_conn_245", fun=mean)
connectivity_585 <- rasterize(polygons_connectivity, temp_classes, field="clim_conn_585", fun=mean)

plot(connectivity_245)
plot(connectivity_585)

#writeRaster(connectivity_245, paste0(result_folder, "/micro_connectivity_245_era5_res.tif"))
#writeRaster(connectivity_585,  paste0(result_folder, "/micro_connectivity_585_era5_res.tif"))