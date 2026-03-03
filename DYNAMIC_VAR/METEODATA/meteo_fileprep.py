import xarray as xr
import os
import numpy as np
from glob import glob
# import dask

# INPUTS
com_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\era5_combined"
folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5"
files = sorted(glob(os.path.join(folder, "ERA5l_*.netcdf")))

#concatenate
datasets = [xr.open_dataset(f) for f in files]
era = xr.concat(datasets, dim="valid_time")
era = era.sortby("valid_time")

# convert temps to celsius
era["t2m"] = era["t2m"] - 273.15
era["t2m"].attrs["units"] = "degC"

# hourly resolve accumulated variables
# Compute diffs
ssrd_diff = era["ssrd"].diff("valid_time")
tp_diff   = era["tp"].diff("valid_time")

# align datasat to diff timestamps
era = era.sel(valid_time=ssrd_diff.valid_time)

# assign transformed values
era["ssrd"] = ssrd_diff.clip(min=0) / 3600.0
era["ssrd"].attrs["units"] = "W m-2"

era["tp"] = tp_diff.clip(min=0) * 1000.0
era["tp"].attrs["units"] = "mm"

# drop season restart (first time stamp, because it will be NA)
era = era.where(
    ~((era.valid_time.dt.month == 5) &
      (era.valid_time.dt.day == 1) &
      (era.valid_time.dt.hour == 0)),
    drop=True
)

# remove unused coords
era = era.reset_coords(drop=True)

# fill single coastal cell ###
####
# this one is necessary for a full prediction grid, but doesnt fully exist

# target cell center
target_lat_approx = 60.2
target_lon_approx = 25.2
# snap to actual grid
target_lat = float(era.latitude.sel(latitude=target_lat_approx, method="nearest"))
target_lon = float(era.longitude.sel(longitude=target_lon_approx, method="nearest"))
print("Target cell:", target_lat, target_lon)

## fill variables by interpolation
lat_vals = era.latitude.values
lon_vals = era.longitude.values

lat_idx = np.where(lat_vals == target_lat)[0][0]
lon_idx = np.where(lon_vals == target_lon)[0][0]

vars_to_fill = ["t2m", "ssrd", "tp", "u10", "v10"]

for v in vars_to_fill:
    
    data = era[v].values  # (time, lat, lon)
    
    # Extract 4-neighbor time series
    north = data[:, lat_idx-1, lon_idx]
    south = data[:, lat_idx+1, lon_idx]
    west  = data[:, lat_idx, lon_idx-1]
    east  = data[:, lat_idx, lon_idx+1]
    # Stack neighbors
    neighbors = np.stack([north, south, west, east], axis=0)
    # Compute mean ignoring NaNs
    filled_series = np.nanmean(neighbors, axis=0)
    # Assign back to missing cell
    data[:, lat_idx, lon_idx] = filled_series
    era[v].values = data


# convert wind components to wind speed
# this has to be done here so that wind speed is also calculated for the interpolated
# grid cell
era["wind_s"] = np.sqrt(era["u10"]**2 + era["v10"]**2)
era["wind_s"].attrs["units"] = "m s-1"

# mask out rows with NaN grid cells
valid_mask = era["t2m"].notnull().all("valid_time")
era = era.where(valid_mask, drop=True)

# export
out = os.path.join(com_folder, "ERA5l_SUMMER_24_25_HEL.netcdf")
era.to_netcdf(out, mode="w")

# check #############
#####################
# validity ##########
era = xr.open_dataset(out)

print("Stacked missing fraction:")
print(era[["t2m","ssrd","tp","wind_s"]].to_array().isnull().mean())

print("Physical checks:")
print("Mean T2M:", float(era["t2m"].mean()))
print("Max SSRD:", float(era["ssrd"].max()))
print("Max TP:", float(era["tp"].max()))

print("Per-variable missing:")
for v in ["t2m","ssrd","tp","wind_s"]:
    print(v, float(era[v].isnull().mean()))

print("Timesteps:", len(era.valid_time))
print("First timestep NaN counts:")
print(era["t2m"].isnull().sum(dim=["latitude","longitude"]).values[:10])

era["t2m"].isnull().sum("valid_time")

####################
# export valid grid
####################
import geopandas as gpd
from shapely.geometry import box

# 1. Get valid grid mask (strict: valid at all timesteps)
valid_mask = era["t2m"].notnull().all("valid_time")

# 2. Extract coordinates
lats = era.latitude.values
lons = era.longitude.values

# 3. Estimate grid resolution
dlat = float(abs(lats[1] - lats[0]))
dlon = float(abs(lons[1] - lons[0]))

polygons = []
lat_list = []
lon_list = []

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        if valid_mask.values[i, j]:
            # Build cell bounds (centered grid)
            poly = box(
                lon - dlon/2,
                lat - dlat/2,
                lon + dlon/2,
                lat + dlat/2
            )
            polygons.append(poly)
            lat_list.append(lat)
            lon_list.append(lon)

# 4. Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    {
        "latitude": lat_list,
        "longitude": lon_list,
    },
    geometry=polygons,
    crs="EPSG:4326"
)

# 5. Export
gpkg_out = os.path.join(com_folder, "ERA5l_SUMMER_24_25_HEL_coverage.gpkg")
gdf.to_file(gpkg_out, driver="GPKG")

##########################################
# transforming era5_land to a raster logic, be refactored later for prediction logic.
###########################################
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
# select timestamp, step 1
ts = era.valid_time.values[0]
da = era["t2m"].sel(valid_time=ts)
# prep as raster
# ERA5 coords must be named x/y for rioxarray
da = da.rename({"longitude": "x", "latitude": "y"})
# write CRS
da = da.rio.write_crs("EPSG:4326")
# Ensure north -> south ordering
da = da.sortby("y", ascending=False)

# reproject to whatever CRS is needed
# EPSG:3067 = ETRS-TM35FIN (meters, correct for Helsinki)

da_utm = da.rio.reproject(
    "EPSG:3067",
    resolution=10,                     # 10 meters
    resampling=Resampling.bilinear
)

#export 
output_file = "era5_test_10m_t2m_utm.tif"
da_utm.rio.to_raster(output_file)

# CONCAT TWO FILES DEPRECATED
# import netcdf
#era24 = xr.open_dataset(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\ERA\ERA_SUMMER_24_05_HEL.netcdf")
# concat 24/25 data into one ds and export
#era_c = xr.concat([era24, era25], dim="valid_time")
#era_c.to_netcdf(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\ERA_SUMMER_24_25_HEL.netcdf")

# crop CERRA data to same area as ERA5
# DEPRECATED because we no longer use CERRA
#ds_cerra = xr.open_dataset(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\CERRA_SUMMER_24_25.netcdf")
#cerra_crop = ds_cerra.where(
#    (ds_cerra.latitude <= 60.30) &
#    (ds_cerra.latitude >= 60.05) &
#    (ds_cerra.longitude >= 24.70) &
#    (ds_cerra.longitude <= 25.28),
#    drop=True
#)
#cerra_crop.to_netcdf(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\CERRA_SUMMER_24_25_HEL.netcdf")