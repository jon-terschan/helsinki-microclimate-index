import xarray as xr

era = xr.open_dataset(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\preprocessed\ERA5L_H1_pre.nc")
cov_polygon = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\preprocessed\ERA5L_H1_pre_coverage.gpkg"
tst_raster = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\preprocessed\ERA5L_H1_pre_testraster.tif"
print(era)

import xarray as xr
import rioxarray

# take one timestep
da = era["t2m"].isel(valid_time=0)

# assign CRS
da = da.rio.write_crs("EPSG:4326")

# write directly
da.rio.to_raster(tst_raster)