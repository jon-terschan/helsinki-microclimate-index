import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
import rasterio

# --------------------------------------------------
# Paths
# --------------------------------------------------
era5_path = "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/11.25/ERA/combined/ERA_SUMMER_24_25_HEL.netcdf"
template_path = "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/full_stack/pred_stack_10m.tif"
out_path = "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/04_predictions/era_files/era5_2024-07-15_16_10m.tif"

target = "2024-07-15T16:00:00"
prev   = "2024-07-15T15:00:00"

# --------------------------------------------------
# Open data
# --------------------------------------------------
ds = xr.open_dataset(era5_path)
template = rioxarray.open_rasterio(template_path)

# Use template as reprojection target
template_geom = template

# --------------------------------------------------
# Helper: prepare + reproject ERA variable
# --------------------------------------------------
def prepare_and_reproject(da):
    da = da.squeeze(drop=True)
    da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    da.rio.write_crs("EPSG:4326", inplace=True)
    da = da.rio.reproject_match(
        template_geom,
        resampling=Resampling.nearest   # IMPORTANT: prevent NA bleeding
    )
    return da.squeeze(drop=True)

# --------------------------------------------------
# Accumulated variable handler
# --------------------------------------------------
def diff_accum(var, scale=1.0):
    now  = ds[var].sel(valid_time=target)
    prev = ds[var].sel(valid_time=prev)
    out  = (now - prev).fillna(0)
    return out * scale

# --------------------------------------------------
# Extract variables
# --------------------------------------------------

# Temperature (K → °C)
t2m = ds["t2m"].sel(valid_time=target) - 273.15
t2m = prepare_and_reproject(t2m)

# SSRD (J/m² → W/m²)
ssrd = diff_accum("ssrd", scale=1/3600.0)
ssrd = prepare_and_reproject(ssrd)

# Precipitation (m → mm/hour)
tp = diff_accum("tp", scale=1000.0)
tp = prepare_and_reproject(tp)

# Wind
u10 = prepare_and_reproject(ds["u10"].sel(valid_time=target))
v10 = prepare_and_reproject(ds["v10"].sel(valid_time=target))
wind_s = np.sqrt(u10**2 + v10**2)

# --------------------------------------------------
# Time encodings
# --------------------------------------------------
dt = pd.to_datetime(target)
hour = dt.hour
doy  = dt.dayofyear

hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)
doy_sin  = np.sin(2*np.pi*doy/365)
doy_cos  = np.cos(2*np.pi*doy/365)

def constant_layer(value):
    arr = template_geom.isel(band=0).copy()
    arr[:] = value
    return arr

hour_sin_r = constant_layer(hour_sin)
hour_cos_r = constant_layer(hour_cos)
doy_sin_r  = constant_layer(doy_sin)
doy_cos_r  = constant_layer(doy_cos)

# --------------------------------------------------
# Stack as numpy (clean structure)
# --------------------------------------------------
layers = [
    t2m.values,
    ssrd.values,
    tp.values,
    u10.values,
    v10.values,
    wind_s.values,
    hour_sin_r.values,
    hour_cos_r.values,
    doy_sin_r.values,
    doy_cos_r.values,
]

stack = np.stack(layers, axis=0)

stack_da = xr.DataArray(
    stack,
    dims=("band", "y", "x"),
    coords={
        "band": np.arange(1, 11),
        "y": template_geom.y,
        "x": template_geom.x,
    },
)

stack_da.rio.write_crs(template_geom.rio.crs, inplace=True)

# --------------------------------------------------
# Export raster
# --------------------------------------------------
stack_da.rio.to_raster(out_path)

# --------------------------------------------------
# Explicit band names (terra-safe)
# --------------------------------------------------
band_names = [
    "t2m","ssrd","tp",
    "u10","v10","wind_s",
    "hour_sin","hour_cos",
    "doy_sin","doy_cos"
]

with rasterio.open(out_path, "r+") as dst:
    for i, name in enumerate(band_names, start=1):
        dst.set_band_description(i, name)

print("ERA5 stack written correctly with consistent geometry and coverage.")