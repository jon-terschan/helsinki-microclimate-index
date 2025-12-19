import xarray as xr
import os
from glob import glob
import dask
import numpy as np

# purpose: concatenate ERA5 files, and add windspeed:

# define inputs
com_folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\ERA"
folder = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\ERA\HELNORTH"
files = sorted(glob(os.path.join(folder, "ERA_SUMMER_*.netcdf")))

datasets = [xr.open_dataset(f) for f in files]
era = xr.concat(datasets, dim="valid_time") # concat files
# era = xr.open_mfdataset(files, combine="by_coords", join="inner") # concat alternative but uses dask

# calculate wind speed
era["wind_s"] = np.sqrt(era["u10"]**2 + era["v10"]**2)

# export file
out = os.path.join(com_folder, "combined", "ERA_SUMMER_24_25_HEL.netcdf") # output path
era.to_netcdf(out) 
print("imdone")

# CHECK INTEGRITY OF FILE
path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\ERA\combined\ERA_SUMMER_24_25_HEL.netcdf"
era = xr.open_dataset(path)

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