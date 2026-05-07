import xarray as xr

era = xr.open_dataset(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\climatology\era5land_jja_tmax_climatology_interpol.nc")
                      
cov_polygon =r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\climatology\baseline_coverage.gpkg"
tst_raster = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\climatology\baseline_raster.tif"
print(era)

import xarray as xr
import rioxarray

# take one timestep
#da = era["t2m"].isel(valid_time=0)
da = era["t2m"]

# assign CRS
da = da.rio.write_crs("EPSG:4326")

# write directly
da.rio.to_raster(tst_raster)