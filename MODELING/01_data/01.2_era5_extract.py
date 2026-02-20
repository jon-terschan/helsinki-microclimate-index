import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow

# ----------------- paths -----------------
era5_path = r"//ad.helsinki.fi/home/t/terschan/Desktop/paper1/data/11.25/ERA/combined/ERA_SUMMER_24_25_HEL.netcdf"
gpkg_path  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\modeling\01_traindataprep\04_training_static.gpkg"

# ============================================================
# 1) LOAD ERA5 DATASET AND STATION LOCATIONS
# ============================================================
ds = xr.open_dataset(era5_path) # load era5 as xaray

stations = gpd.read_file(gpkg_path)[["sensor_id", "geometry"]].drop_duplicates("sensor_id").reset_index(drop=True) # load localized training df
stations = stations.to_crs("EPSG:4326") # convert to lat/lon
stations["sensor_id"] = stations["sensor_id"].astype(str) # enforce sensor id as string

# ============================================================
# 2) DETECT COORDINATE / TIME NAMES IN NETCDF
# ============================================================
lat_name = "latitude" if "latitude" in ds.coords else [c for c in ds.coords if "lat" in c.lower()][0]
lon_name = "longitude" if "longitude" in ds.coords else [c for c in ds.coords if "lon" in c.lower()][0]
time_name = "valid_time" if "valid_time" in ds.dims else [d for d in ds.dims if "time" in d.lower() or "valid" in d.lower()][0]

# ============================================================
# 3) BUILD ERA5 VALIDITY MASK
# ============================================================
# which grid cells have data?
var_for_mask = "t2m" if "t2m" in ds.data_vars else list(ds.data_vars)[0]
valid_mask = ~ds[var_for_mask].isnull().all(dim=time_name).values   # shape (nlat, nlon)

lat_vals = ds[lat_name].values
lon_vals = ds[lon_name].values

# ============================================================
# 4) MAP EACH STATION TO ITS NEAREST ERA5 GRID CELL (BY INDEX)
# ============================================================
stations_ll = stations.copy()
lat_idx = np.abs(lat_vals[:, None] - stations_ll.geometry.y.values).argmin(axis=0)
lon_idx = np.abs(lon_vals[:, None] - stations_ll.geometry.x.values).argmin(axis=0)
stations_ll["lat_idx"] = lat_idx
stations_ll["lon_idx"] = lon_idx

# ============================================================
# 5) EXTRACT ERA5 TIME SERIES PER STATION
#  
# ============================================================
era5_list = []
max_radius = 1   # how far (in grid-cells) we allow searching, in case bigger datasets should be used at some point but not needed here

for _, row in stations_ll.iterrows():
    sid = row.sensor_id
    lat_i = int(row.lat_idx); lon_i = int(row.lon_idx)

    # if the chosen cell is invalid, expand search radius
    if not valid_mask[lat_i, lon_i]:
        found = False
        nlat, nlon = valid_mask.shape
        for r in range(1, max_radius + 1):
            i0 = max(0, lat_i - r); i1 = min(nlat, lat_i + r + 1)
            j0 = max(0, lon_i - r); j1 = min(nlon, lon_i + r + 1)
            sub = valid_mask[i0:i1, j0:j1]
            if sub.any():
                # choose nearest valid within this window by Euclidean dist on lat/lon
                sub_lats = lat_vals[i0:i1]
                sub_lons = lon_vals[j0:j1]
                dlat = (sub_lats[:, None] - row.geometry.y)**2
                dlon = (sub_lons[None, :] - row.geometry.x)**2
                dist2 = dlat + dlon
                dist2 = np.where(sub, dist2, np.inf)  # invalid => inf
                flat = np.argmin(dist2)
                ii, jj = np.unravel_index(flat, dist2.shape)
                lat_i = i0 + int(ii); lon_i = j0 + int(jj)
                found = True
                break
        if not found:
            print(f"WARNING: no valid ERA5 tile found within radius {max_radius} for sensor {sid}; skipping")
            continue

    # extract tile; if extra dims like 'number' or 'expver' exist, take first index
    pt = ds.isel(latitude=lat_i, longitude=lon_i)
    if "number" in pt.dims:
        pt = pt.isel(number=0)
    if "expver" in pt.dims:
        pt = pt.isel(expver=0)
    pt = pt.squeeze(drop=True)

    # convert to dataframe (rename time dim if named valid_time)
    df = pt.to_dataframe().reset_index()
    if "valid_time" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"valid_time": "time"})
    df["sensor_id"] = sid
    era5_list.append(df)
    print(f"Extracting ERA5 for sensor {sid} from grid ({lat_i},{lon_i})")

# ============================================================
# 6) COMBINE ALL STATIONS INTO ONE TABLE
# ============================================================
era5_stations = pd.concat(era5_list, ignore_index=True)
# drop unnecessary columns we dont need
era5_stations = era5_stations.drop(
    columns=["latitude", "longitude", "spatial_ref", "number", "expver"],
    errors="ignore"
)

era5_stations = era5_stations.sort_values(["sensor_id", "time"]).reset_index(drop=True)

# ============================================================
# 7) PHYSICAL UNIT CONVERSIONS
# ============================================================
# convert era5 vars to physiccally meaningful units
era5_phys = era5_stations.copy()

# temperature from kelvin to celsius
era5_phys["t2m"] = era5_phys["t2m"] - 273.15

# SSRD: J/m² to W/m² (hourly)
era5_phys = era5_phys.sort_values(["sensor_id", "time"])

era5_phys["ssrd"] = (
    era5_phys
    .groupby("sensor_id")["ssrd"]
    .diff()
    .clip(lower=0)
    .fillna(0)
    / 3600.0
)
# total precipitation (review later)
era5_phys["tp"] = (
    era5_phys
    .groupby("sensor_id")["tp"]
    .diff()
    .clip(lower=0)
    .fillna(0)
    * 1000.0
)


### EXTRA LAGGED PREDICTORS, REMOVE IF THEY DONT INCREASE PERFORMANCE
# t2m lags
era5_phys["t2m_lag1"] = era5_phys.groupby("sensor_id")["t2m"].shift(1)
era5_phys["t2m_lag3"] = era5_phys.groupby("sensor_id")["t2m"].shift(3)
era5_phys["t2m_lag6"] = era5_phys.groupby("sensor_id")["t2m"].shift(6)

era5_phys["ssrd_roll3"] = (
    era5_phys
    .groupby("sensor_id")["ssrd"]
    .transform(lambda x: x.rolling(3, min_periods=1).sum())
)

era5_phys["ssrd_roll6"] = (
    era5_phys
    .groupby("sensor_id")["ssrd"]
    .transform(lambda x: x.rolling(6, min_periods=1).sum())
)

era5_phys["t2m_lag24"] = (
    era5_phys
    .groupby("sensor_id")["t2m"]
    .shift(24)
)

era5_phys.head()

# ============================================================
#  SOME POST CALCULATION CHECKS
# ============================================================
bad = era5_phys.groupby("sensor_id")["t2m"].apply(lambda x: x.isna().all()) # should be empty (sensor points without era5)
print("Sensors with all-NaN t2m:", bad[bad].index.tolist())
era5_phys.groupby("sensor_id")["time"].agg(["min", "max"]) # sensor time coverage
era5_phys.isna().mean().sort_values(ascending=False) # missingness rates

# ============================================================
#  EXPORT FOR REUSE IN R
# ============================================================
era5_phys["time"] = era5_phys["time"].dt.tz_localize("UTC")

era5_phys.to_parquet(r"//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/01_traindataprep/05_era5_variables.parquet", engine="fastparquet", index=False)

# ======================================
# ======================
# 9) VISUAL DIAGNOSTIC: GRID TILE ↔ STATION MAPPING
# ============================================================
# unique integer per ERA5 grid cell
nlon = ds.dims["longitude"]
stations_ll["tile_id"] = stations_ll["lat_idx"] * nlon + stations_ll["lon_idx"]
import numpy as np

lat_n = ds.dims["latitude"]
lon_n = ds.dims["longitude"]

tile_grid = np.full((lat_n, lon_n), np.nan)

# fill only tiles that are actually used by stations
for _, r in stations_ll.iterrows():
    tile_grid[r.lat_idx, r.lon_idx] = r.tile_id
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))

# ERA5 grid tiles
im = ax.pcolormesh(
    ds.longitude.values,
    ds.latitude.values,
    tile_grid,
    shading="nearest",
    cmap="tab20"
)

# station points, colored by assigned tile
sc = ax.scatter(
    stations_ll.geometry.x,
    stations_ll.geometry.y,
    c=stations_ll.tile_id,
    cmap="tab20",
    edgecolor="black",
    s=40,
    zorder=3
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("ERA5 grid cell attribution to stations")

plt.tight_layout()
plt.savefig("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/01_traindataprep/era5_tile_station_mapping.png", dpi=300)
plt.close()

# how many stations per era5_tile
stations_ll.groupby("tile_id").size().sort_values(ascending=False)
