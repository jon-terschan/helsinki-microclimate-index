import geopandas as gpd
from shapely.geometry import box
import numpy as np
import xarray as xr 

# purpose: create ERA5 grid cell center points as gpkg to check in QGIS

# import output
path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\ERA\HELNORTH\ERA_SUMMER_24_05.netcdf"
out_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\ERA\era5_centers.gpkg"

era = xr.open_dataset(path)

# extract lat/long
lats = era.latitude.values
lons = era.longitude.values

# create grid cell center points
lon2d, lat2d = np.meshgrid(lons, lats)
lon_flat = lon2d.ravel()
lat_flat = lat2d.ravel()
era_centers = gpd.GeoDataFrame(
    {
        "tile_i": np.repeat(np.arange(len(lats)), len(lons)),
        "tile_j": np.tile(np.arange(len(lons)), len(lats)),
        "lat": lat_flat,
        "lon": lon_flat,
    },
    geometry=gpd.points_from_xy(lon_flat, lat_flat),
    crs="EPSG:4326"
)

# export
era_centers.to_file(
    out_path,
    layer="era5_centers",
    driver="GPKG"
)
