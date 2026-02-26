import xarray as xr
import os
from glob import glob
# import dask
import numpy as np

# purpose: concatenate ERA5 files, and add windspeed, physical transformation of variables:

# define inputs
com_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\era5_combined"
folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5"
files = sorted(glob(os.path.join(folder, "ERA5l_*.netcdf")))

datasets = [xr.open_dataset(f) for f in files]
era = xr.concat(datasets, dim="valid_time") # concat files
# era = xr.open_mfdataset(files, combine="by_coords", join="inner") # concat alternative but uses dask
era = era.sortby("valid_time")

# -------------------------
# metric transformations

# temp from kelvin to C
era["t2m"] = era["t2m"] - 273.15
era["t2m"].attrs["units"] = "degC"

# wind speed
era["wind_s"] = np.sqrt(era["u10"]**2 + era["v10"]**2)
era["wind_s"].attrs["units"] = "m s-1"

# 3) SSRD from accumulated J/m2 to W/m2 per hour
era["ssrd"] = (
    era["ssrd"]
    .diff("valid_time")
    .clip(min=0)
    .fillna(0)
    / 3600.0
)
era["ssrd"].attrs["units"] = "W m-2"

# 4) total precipitation accumulated m into mm per hour
era["tp"] = (
    era["tp"]
    .diff("valid_time")
    .clip(min=0)
    .fillna(0)
    * 1000.0
)
era["tp"].attrs["units"] = "mm"

# IMPORTANT:
# MAKE SURE TO DROP THE FIRST TIMESTAMP OF FIRST OF MAY (00:00) IN BOTH YEARS
# #transformations will be nonsensical there (no data on timestep before)
# drop first timestep overall
era = era.isel(valid_time=slice(1, None))

# drop explicit seasonal restart (May 1st 00:00)
era = era.where(
    ~((era.valid_time.dt.month == 5) &
      (era.valid_time.dt.day == 1) &
      (era.valid_time.dt.hour == 0)),
    drop=True
)

# export file
out = os.path.join(com_folder, "ERA5l_SUMMER_24_25_HEL.netcdf") # output path
era.to_netcdf(out) 

# CHECK INTEGRITY OF FILE
era = xr.open_dataset(out)
print(era[["t2m","ssrd","tp","wind_s"]].to_array().isnull().mean())
print(float(era["t2m"].mean()))
print(float(era["ssrd"].max()))
print(float(era["tp"].max()))

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