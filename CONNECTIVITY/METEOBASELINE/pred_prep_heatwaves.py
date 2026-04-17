import xarray as xr
import pandas as pd
import numpy as np

events = {
    "H1":[("2010","07",["24","25","26","27","28","29","30"])],
    "H2":[
        ("2018","07",["21","22","23","24","25","26","27","28","29","30","31"]),
        ("2018","08",["01","02","03","04"])
    ],
    "H3":[("2021","07",["09","10","11","12","13","14","15","16","17","18","19","20"])]
}

def expected_hours(event_list):
    return pd.DatetimeIndex(pd.Index([]).append([
        pd.date_range(f"{y}-{m}-{d} 00:00", f"{y}-{m}-{d} 23:00", freq="h")
        for y, m, days in event_list
        for d in days
    ]))

expected_map = {k: expected_hours(v) for k, v in events.items()}

files = {
    "H1": r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\ERA5L_hourly_H1.nc",
    "H2": r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\ERA5L_hourly_H2.nc",
    "H3": r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\ERA5L_hourly_H3.nc"
}

for key, path in files.items():
    time = xr.open_dataset(path, decode_times=True)["valid_time"].values
    time = pd.to_datetime(time).tz_localize(None).floor("h")

    expected = expected_map[key]

    missing = expected[~np.isin(expected.values, time)]

    print(f"{key}: missing {len(missing)}"
          

for key, path in files.items():
    time = xr.open_dataset(path, decode_times=True)["valid_time"].values
    time = pd.to_datetime(time).tz_localize(None)

    counts = pd.Series(time).dt.floor("D").value_counts().sort_index()

    bad_days = counts[counts != 24]

    if bad_days.empty:
        print(f"{key}: all days have 24 hours")
    else:
        print(f"{key}: days with !=24 hours")
        print(bad_days)




files = [
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\ERA5L_hourly_H2_2018_07.nc",
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\ERA5L_hourly_H2_2018_08.nc"
]
datasets = [xr.open_dataset(f) for f in files]

# IMPORTANT: use valid_time, not time
datasets = [ds.rename({"time": "valid_time"}) if "time" in ds.dims else ds for ds in datasets]

era = xr.concat(datasets, dim="valid_time").sortby("valid_time")


import xarray as xr
import numpy as np
import os

# INPUT
file = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\ERA5L_hourly_H2.nc"
era = xr.open_dataset(file)

out  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\preprocessed\ERA5L_H2_pre.nc"

if "time" in era.dims:
    era = era.isel(time=0)
    
era["t2m"] = era["t2m"] - 273.15
era["t2m"].attrs["units"] = "degC"

ssrd_diff = era["ssrd"].diff("valid_time")
tp_diff   = era["tp"].diff("valid_time")

ssrd_hourly = ssrd_diff.clip(min=0) / 3600.0
tp_hourly   = tp_diff.clip(min=0) * 1000.0

era = era.isel(valid_time=slice(1, None))

era["ssrd"] = ssrd_hourly
era["ssrd"].attrs["units"] = "W m-2"

era["tp"] = tp_hourly
era["tp"].attrs["units"] = "mm"

# -------------------------------
# TARGET GRID CELL (missing tile)
# -------------------------------
target_lat_approx = 60.2
target_lon_approx = 25.2

target_lat = era.latitude.sel(latitude=target_lat_approx, method="nearest").item()
target_lon = era.longitude.sel(longitude=target_lon_approx, method="nearest").item()

lat_vals = era.latitude.values
lon_vals = era.longitude.values

lat_idx = int(np.argmin(np.abs(lat_vals - target_lat)))
lon_idx = int(np.argmin(np.abs(lon_vals - target_lon)))

print("Target cell:", target_lat, target_lon)

# -------------------------------
# INTERPOLATE SINGLE CELL
# (ONLY north + west)
# -------------------------------
vars_to_fill = ["t2m", "ssrd", "tp", "u10", "v10"]

for v in vars_to_fill:
    data = era[v].values

    north = data[:, lat_idx-1, lon_idx]
    west  = data[:, lat_idx, lon_idx-1]

    # average only available values
    filled = np.where(
        np.isnan(north) & np.isnan(west),
        np.nan,
        np.nanmean(np.stack([north, west], axis=0), axis=0)
    )

    data[:, lat_idx, lon_idx] = filled
    era[v].values = data

# -------------------------------
# WIND SPEED
# -------------------------------
era["wind_s"] = np.sqrt(era["u10"]**2 + era["v10"]**2)
era["wind_s"].attrs["units"] = "m s-1"

# -------------------------------
# EXPORT
# -------------------------------
era.to_netcdf(out)

# -------------------------------
# QUICK CHECKS
# -------------------------------
era = xr.open_dataset(out)

print("Timesteps:", len(era.valid_time))

nan_map = era.to_array().isnull().sum(dim=("variable", "valid_time"))
print(nan_map)

print(era.valid_time.sel(valid_time=slice("2018-07-21", "2018-07-23")).values)



#### SINGLE HEATWAVE ##############
#### this does the same thing without concat, for our corrected heatwave 2 period # 
import xarray as xr
import numpy as np
import os

# -------------------------------
# INPUT / OUTPUT
# -------------------------------
file = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\ERA5L_hourly_H2_corrected.nc"
out  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\preprocessed\ERA5L_H2_corrected_pre.nc"

era = xr.open_dataset(file)

# -------------------------------
# UNIT CONVERSIONS
# -------------------------------

# Temperature: K → °C
era["t2m"] = era["t2m"] - 273.15
era["t2m"].attrs["units"] = "degC"

# Accumulated → hourly
ssrd_diff = era["ssrd"].diff("valid_time")
tp_diff   = era["tp"].diff("valid_time")

era = era.isel(valid_time=slice(1, None))  # drop first timestep

# Radiation: J/m² → W/m²
era["ssrd"] = (ssrd_diff.clip(min=0) / 3600.0)
era["ssrd"].attrs["units"] = "W m-2"

# Precip: m → mm
era["tp"] = (tp_diff.clip(min=0) * 1000.0)
era["tp"].attrs["units"] = "mm"

# -------------------------------
# FILL MISSING GRID CELL
# -------------------------------
target_lat_approx = 60.2
target_lon_approx = 25.2

target_lat = era.latitude.sel(latitude=target_lat_approx, method="nearest").item()
target_lon = era.longitude.sel(longitude=target_lon_approx, method="nearest").item()

lat_vals = era.latitude.values
lon_vals = era.longitude.values

lat_idx = int(np.argmin(np.abs(lat_vals - target_lat)))
lon_idx = int(np.argmin(np.abs(lon_vals - target_lon)))

vars_to_fill = ["t2m", "ssrd", "tp", "u10", "v10"]

for v in vars_to_fill:
    data = era[v].values

    north = data[:, lat_idx-1, lon_idx]
    west  = data[:, lat_idx, lon_idx-1]

    filled = np.where(
        np.isnan(north) & np.isnan(west),
        np.nan,
        np.nanmean(np.stack([north, west], axis=0), axis=0)
    )

    data[:, lat_idx, lon_idx] = filled
    era[v].values = data

# -------------------------------
# WIND SPEED
# -------------------------------
era["wind_s"] = np.sqrt(era["u10"]**2 + era["v10"]**2)
era["wind_s"].attrs["units"] = "m s-1"

# -------------------------------
# EXPORT
# -------------------------------
era.to_netcdf(out)

# -------------------------------
# QUICK CHECK
# -------------------------------
print("Timesteps:", len(era.valid_time))
print(era)

nan_map = era.to_array().isnull().sum(dim="valid_time")
print(nan_map)