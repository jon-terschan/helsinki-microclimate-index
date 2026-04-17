import os
import glob
import numpy as np
import xarray as xr

# ----------------------------
# Paths
# ----------------------------
input_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\hourly\hourly_nc"
output_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\climatology"
os.makedirs(output_dir, exist_ok=True)

out_nc = os.path.join(output_dir, "era5land_jja_tmax_climatology.nc")

# ----------------------------
# Load monthly NetCDF files
# ----------------------------
files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))
if not files:
    raise FileNotFoundError(f"No NetCDF files found in: {input_dir}")

ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    parallel=True,
    chunks={"valid_time": 744}
)

if "expver" in ds:
    ds = ds.drop_vars("expver")

ds = ds.sortby("valid_time")

# ----------------------------
# Deaccumulation helper
# ----------------------------
def deaccumulate(acc: xr.DataArray) -> xr.DataArray:
    first = acc.isel(valid_time=0)
    diff = acc.diff("valid_time")
    diff = xr.where(diff < 0, acc.isel(valid_time=slice(1, None)), diff)
    out = xr.concat([first, diff], dim="valid_time")
    out = out.assign_coords(valid_time=acc["valid_time"])
    return out

# ----------------------------
# Derived predictors
# ----------------------------
t2m_c = ds["t2m"] - 273.15
u10 = ds["u10"]
v10 = ds["v10"]
wind_s = np.sqrt(u10**2 + v10**2)

# Deaccumulate (keep your logic)
ssrd_hourly = deaccumulate(ds["ssrd"])
tp_hourly = deaccumulate(ds["tp"])

# ----------------------------
# Unit conversions (MATCH TRAINING)
# ----------------------------
# SSRD: J/m² → W/m²
ssrd_hourly = ssrd_hourly / 3600.0
ssrd_hourly.attrs["units"] = "W m-2"

# TP: m → mm
tp_hourly = tp_hourly * 1000.0
tp_hourly.attrs["units"] = "mm"

# ----------------------------
# UTC encodings on valid_time
# ----------------------------
hour = ds["valid_time"].dt.hour
doy = ds["valid_time"].dt.dayofyear
doy = xr.where((ds["valid_time"].dt.is_leap_year) & (doy > 59), doy - 1, doy)

hour_sin = np.sin(2 * np.pi * hour / 24.0)
hour_cos = np.cos(2 * np.pi * hour / 24.0)
doy_sin = np.sin(2 * np.pi * doy / 365.0)
doy_cos = np.cos(2 * np.pi * doy / 365.0)

# ----------------------------
# Assemble predictor dataset
# ----------------------------
pred = xr.Dataset(
    {
        "t2m": t2m_c,              # renamed here
        "u10": u10,
        "v10": v10,
        "wind_s": wind_s,
        "ssrd": ssrd_hourly,      # renamed here
        "tp": tp_hourly,          # renamed here
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
    }
)

# ----------------------------
# JJA only
# ----------------------------
pred_jja = pred.sel(valid_time=pred["valid_time"].dt.month.isin([6, 7, 8]))

# Add day and hour coordinates
pred_jja = pred_jja.assign_coords(
    day=pred_jja["valid_time"].dt.floor("D"),
    hour=pred_jja["valid_time"].dt.hour
)

# ----------------------------
# Reshape to (day, hour, lat, lon)
# ----------------------------
pred_dayhour = pred_jja.set_index(valid_time=["day", "hour"]).unstack("valid_time")
pred_dayhour = pred_dayhour.sortby("day")

# ----------------------------
# Find Tmax hour for each day and grid cell
# ----------------------------
t2m = pred_dayhour["t2m"]

# valid mask: cells with at least one non-NaN value along hour
valid = t2m.notnull().any("hour")

# fill NaNs so argmax never sees an all-NaN slice
tmax_idx = t2m.fillna(-1e30).argmax("hour").compute()

# Select all predictors at Tmax hour
daily_tmax = pred_dayhour.isel(hour=tmax_idx)

# restore missing cells to NaN
daily_tmax = daily_tmax.where(valid)

# ----------------------------
# Mean climatology across all JJA days
# ----------------------------
climatology = daily_tmax.mean("day", skipna=True)

# Keep only final RF predictors
climatology = climatology[[
    "t2m",
    "u10",
    "v10",
    "wind_s",
    "ssrd",
    "tp",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]]

# ----------------------------
# Save
# ----------------------------
import numpy as np

for var in climatology.data_vars:
    da = climatology[var]

    print(f"\n=== {var} ===")
    print(f"min:   {float(da.min())}")
    print(f"max:   {float(da.max())}")
    print(f"mean:  {float(da.mean())}")
    print(f"std:   {float(da.std())}")


hour_angle = np.arctan2(climatology["hour_sin"], climatology["hour_cos"])
hour = (hour_angle % (2*np.pi)) * 24 / (2*np.pi)

print("\nSSRD sanity (W/m²):", float(climatology["ssrd"].mean()))
print("TP sanity (mm):", float(climatology["tp"].mean()))



# ----------------------------
# OPTIONAL: reload from disk (clean separation)
# ----------------------------
climatology.to_netcdf(out_nc)
climatology = xr.open_dataset(out_nc)
out_nc = os.path.join(
    output_dir,
    "era5land_jja_tmax_climatology_interpol.nc"
)

# ----------------------------
# FILL MISSING GRID CELL (north + west)
# ----------------------------
target_lat_approx = 60.2
target_lon_approx = 25.2

target_lat = float(climatology.latitude.sel(latitude=target_lat_approx, method="nearest"))
target_lon = float(climatology.longitude.sel(longitude=target_lon_approx, method="nearest"))

lat_vals = climatology.latitude.values
lon_vals = climatology.longitude.values

lat_idx = int(np.argmin(np.abs(lat_vals - target_lat)))
lon_idx = int(np.argmin(np.abs(lon_vals - target_lon)))

print("Filling cell:", target_lat, target_lon)

# --- 1. interpolate physical variables ---
vars_interp = ["t2m", "u10", "v10", "wind_s", "ssrd", "tp"]

for v in vars_interp:
    data = climatology[v].values

    north = data[lat_idx - 1, lon_idx]
    west  = data[lat_idx, lon_idx - 1]

    filled = np.nanmean([north, west])
    data[lat_idx, lon_idx] = filled

    climatology[v].values = data

# --- 2. assign encodings directly (no spatial meaning) ---
for v in ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]:
    data = climatology[v].values

    # just copy from nearest valid neighbor (north)
    data[lat_idx, lon_idx] = data[lat_idx - 1, lon_idx]

    climatology[v].values = data

import pandas as pd

# FAKE DUMMY TIMESTAMP FOR R INDEXING
# ----------------------------
# FORCE UNIX TIMESTAMP (R-compatible)
# ----------------------------
ts = pd.Timestamp("2000-07-15 13:00:00", tz="UTC")
seconds_since_epoch = int(ts.timestamp())

climatology = climatology.expand_dims(
    valid_time=[seconds_since_epoch]
)

# enforce NetCDF time encoding
climatology["valid_time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"
climatology["valid_time"].attrs["calendar"] = "standard"

# (optional but safer)
climatology["valid_time"].encoding = {
    "dtype": "int64",
    "units": "seconds since 1970-01-01 00:00:00"
}
climatology.to_netcdf(out_nc)

print(f"Saved climatology to: {out_nc}")
print(climatology.dims)



################################
### new climatology baseline ###
import os
import glob
import numpy as np
import xarray as xr
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
input_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\hourly\hourly_nc"
output_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\climatology"
os.makedirs(output_dir, exist_ok=True)

out_nc = os.path.join(output_dir, "era5land_jja_climatology_13_15_17.nc")

target_local_hours = [13, 15, 17]
utc_offset = 3  # Helsinki in JJA

# ----------------------------
# Load files safely, one by one
# ----------------------------
files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))
if not files:
    raise FileNotFoundError(f"No NetCDF files found in: {input_dir}")

datasets = []
for f in files:
    with xr.open_dataset(f, engine="netcdf4") as ds0:
        ds0 = ds0.load()  # detach from file handle immediately
        if "expver" in ds0:
            ds0 = ds0.drop_vars("expver")
        datasets.append(ds0)

ds = xr.concat(datasets, dim="valid_time").sortby("valid_time")

# ----------------------------
# Deaccumulation helper
# ----------------------------
def deaccumulate(acc: xr.DataArray) -> xr.DataArray:
    first = acc.isel(valid_time=0)
    diff = acc.diff("valid_time")
    diff = xr.where(diff < 0, acc.isel(valid_time=slice(1, None)), diff)
    out = xr.concat([first, diff], dim="valid_time")
    out = out.assign_coords(valid_time=acc["valid_time"])
    return out

# ----------------------------
# Predictors
# ----------------------------
t2m_c = ds["t2m"] - 273.15
u10 = ds["u10"]
v10 = ds["v10"]
wind_s = np.sqrt(u10**2 + v10**2)

ssrd = deaccumulate(ds["ssrd"]) / 3600.0
tp = deaccumulate(ds["tp"]) * 1000.0

hour = ds["valid_time"].dt.hour
doy = ds["valid_time"].dt.dayofyear
doy = xr.where((ds["valid_time"].dt.is_leap_year) & (doy > 59), doy - 1, doy)

hour_sin = np.sin(2 * np.pi * hour / 24.0)
hour_cos = np.cos(2 * np.pi * hour / 24.0)
doy_sin = np.sin(2 * np.pi * doy / 365.0)
doy_cos = np.cos(2 * np.pi * doy / 365.0)

pred = xr.Dataset(
    {
        "t2m": t2m_c,
        "u10": u10,
        "v10": v10,
        "wind_s": wind_s,
        "ssrd": ssrd,
        "tp": tp,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
    }
)

pred = pred.sel(valid_time=pred["valid_time"].dt.month.isin([6, 7, 8]))

# ----------------------------
# Build 3-hour climatology
# ----------------------------
clim_list = []
time_list = []

for local_hour in target_local_hours:
    utc_hour = (local_hour - utc_offset) % 24

    sub = pred.sel(valid_time=pred["valid_time"].dt.hour == utc_hour)
    if sub.sizes["valid_time"] == 0:
        raise ValueError(f"No data found for UTC hour {utc_hour} (local {local_hour})")

    clim = sub.mean("valid_time", skipna=True)

    clim = clim[[
        "t2m", "u10", "v10", "wind_s", "ssrd", "tp",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos"
    ]]

    # overwrite encodings so they are constant fields for this hour
    clim["hour_sin"] = xr.full_like(clim["t2m"], np.sin(2 * np.pi * utc_hour / 24.0))
    clim["hour_cos"] = xr.full_like(clim["t2m"], np.cos(2 * np.pi * utc_hour / 24.0))
    clim["doy_sin"] = xr.full_like(clim["t2m"], float(sub["doy_sin"].mean()))
    clim["doy_cos"] = xr.full_like(clim["t2m"], float(sub["doy_cos"].mean()))

    # ----------------------------
    # Fill missing cell
    # ----------------------------
    lat_idx = int(np.argmin(np.abs(clim.latitude.values - 60.2)))
    lon_idx = int(np.argmin(np.abs(clim.longitude.values - 25.2)))

    for v in ["t2m", "u10", "v10", "wind_s", "ssrd", "tp"]:
        data = clim[v].values
        data[lat_idx, lon_idx] = np.nanmean([
            data[lat_idx - 1, lon_idx],
            data[lat_idx, lon_idx - 1]
        ])
        clim[v].values = data

    for v in ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]:
        data = clim[v].values
        data[lat_idx, lon_idx] = data[lat_idx - 1, lon_idx]
        clim[v].values = data

    # ----------------------------
    # UNIX timestamp for R
    # ----------------------------
    ts = pd.Timestamp(f"2000-07-15 {utc_hour:02d}:00:00", tz="UTC")
    time_list.append(np.int64(ts.timestamp()))
    clim_list.append(clim)

# ----------------------------
# Concatenate to one dataset
# ----------------------------
clim_all = xr.concat(clim_list, dim="valid_time")

clim_all = clim_all.assign_coords(
    valid_time=("valid_time", np.array(time_list, dtype="int64"))
)

clim_all["valid_time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"
clim_all["valid_time"].attrs["calendar"] = "standard"
clim_all["valid_time"].encoding = {
    "dtype": "int64",
    "units": "seconds since 1970-01-01 00:00:00",
    "calendar": "standard",
}

# ----------------------------
# Save
# ----------------------------
clim_all.to_netcdf(out_nc)

print("Saved:", out_nc)
print("Dims:", clim_all.dims)
print("Times:", clim_all.valid_time.values)

# ----------------------------
# DEFENSIVE CHECK
# ----------------------------

# 1. Check timestamps
print("\n--- Timestamp check ---")
for t in clim_all.valid_time.values:
    print("Unix:", int(t), "| UTC:", pd.to_datetime(int(t), unit="s", utc=True))

# 2. Check spatial means per timestep
print("\n--- Spatial mean per timestep ---")
spatial_mean = clim_all.mean(dim=["latitude", "longitude"])

for i, t in enumerate(clim_all.valid_time.values):
    ts_readable = pd.to_datetime(int(t), unit="s", utc=True)
    print(f"\nTime: {ts_readable}")

    for var in ["t2m","u10","v10","wind_s","ssrd","tp"]:
        val = float(spatial_mean[var].isel(valid_time=i).values)
        print(f"  {var:8s}: {val:8.3f}")


################################
### extended climatology baseline ###
import os
import glob
import numpy as np
import xarray as xr
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
input_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\hourly\hourly_nc"
output_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\climatology"
os.makedirs(output_dir, exist_ok=True)

# local hours: 10–18 inclusive
target_local_hours = list(range(10, 19))
utc_offset = 3  # Helsinki JJA

# ----------------------------
# Load files safely
# ----------------------------
files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))
if not files:
    raise FileNotFoundError(f"No NetCDF files found in: {input_dir}")

datasets = []
for f in files:
    with xr.open_dataset(f, engine="netcdf4") as ds0:
        ds0 = ds0.load()
        if "expver" in ds0:
            ds0 = ds0.drop_vars("expver")
        datasets.append(ds0)

ds = xr.concat(datasets, dim="valid_time").sortby("valid_time")

# ----------------------------
# Deaccumulation helper
# ----------------------------
def deaccumulate(acc):
    first = acc.isel(valid_time=0)
    diff = acc.diff("valid_time")
    diff = xr.where(diff < 0, acc.isel(valid_time=slice(1, None)), diff)
    out = xr.concat([first, diff], dim="valid_time")
    out = out.assign_coords(valid_time=acc["valid_time"])
    return out

# ----------------------------
# Predictors
# ----------------------------
t2m_c = ds["t2m"] - 273.15
u10 = ds["u10"]
v10 = ds["v10"]
wind_s = np.sqrt(u10**2 + v10**2)

ssrd = deaccumulate(ds["ssrd"]) / 3600.0
tp   = deaccumulate(ds["tp"]) * 1000.0

hour = ds["valid_time"].dt.hour
doy  = ds["valid_time"].dt.dayofyear
doy  = xr.where((ds["valid_time"].dt.is_leap_year) & (doy > 59), doy - 1, doy)

hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)
doy_sin  = np.sin(2*np.pi*doy/365)
doy_cos  = np.cos(2*np.pi*doy/365)

pred_all = xr.Dataset({
    "t2m": t2m_c,
    "u10": u10,
    "v10": v10,
    "wind_s": wind_s,
    "ssrd": ssrd,
    "tp": tp,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "doy_sin": doy_sin,
    "doy_cos": doy_cos
})

# ----------------------------
# FUNCTION: build climatology
# ----------------------------
def build_climatology(pred, months, out_path):

    pred = pred.sel(valid_time=pred["valid_time"].dt.month.isin(months))

    clim_list = []
    time_list = []

    for local_hour in target_local_hours:

        utc_hour = (local_hour - utc_offset) % 24

        sub = pred.sel(valid_time=pred["valid_time"].dt.hour == utc_hour)
        if sub.sizes["valid_time"] == 0:
            raise ValueError(f"No data for UTC {utc_hour}")

        clim = sub.mean("valid_time", skipna=True)

        clim = clim[[
            "t2m","u10","v10","wind_s","ssrd","tp",
            "hour_sin","hour_cos","doy_sin","doy_cos"
        ]]

        # overwrite encodings
        clim["hour_sin"] = xr.full_like(clim["t2m"], np.sin(2*np.pi*utc_hour/24))
        clim["hour_cos"] = xr.full_like(clim["t2m"], np.cos(2*np.pi*utc_hour/24))
        clim["doy_sin"]  = xr.full_like(clim["t2m"], float(sub["doy_sin"].mean()))
        clim["doy_cos"]  = xr.full_like(clim["t2m"], float(sub["doy_cos"].mean()))

        # fill missing cell
        lat_idx = int(np.argmin(np.abs(clim.latitude.values - 60.2)))
        lon_idx = int(np.argmin(np.abs(clim.longitude.values - 25.2)))

        for v in ["t2m","u10","v10","wind_s","ssrd","tp"]:
            data = clim[v].values
            data[lat_idx, lon_idx] = np.nanmean([
                data[lat_idx-1, lon_idx],
                data[lat_idx, lon_idx-1]
            ])
            clim[v].values = data

        for v in ["hour_sin","hour_cos","doy_sin","doy_cos"]:
            data = clim[v].values
            data[lat_idx, lon_idx] = data[lat_idx-1, lon_idx]
            clim[v].values = data

        # UNIX timestamp
        ts = pd.Timestamp(f"2000-07-15 {utc_hour:02d}:00:00", tz="UTC")
        time_list.append(np.int64(ts.timestamp()))

        clim_list.append(clim)

    # concat
    clim_all = xr.concat(clim_list, dim="valid_time")

    clim_all = clim_all.assign_coords(
        valid_time=("valid_time", np.array(time_list, dtype="int64"))
    )

    clim_all["valid_time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"
    clim_all["valid_time"].attrs["calendar"] = "standard"
    clim_all["valid_time"].encoding = {
        "dtype": "int64",
        "units": "seconds since 1970-01-01 00:00:00",
        "calendar": "standard"
    }

    clim_all.to_netcdf(out_path)

    print("\nSaved:", out_path)
    print("Dims:", clim_all.dims)
    print("Times:", clim_all.valid_time.values)

    return clim_all


# ----------------------------
# RUN: JJA climatology
# ----------------------------
out_jja = os.path.join(output_dir, "era5land_climatology_JJA_10_18.nc")
clim_jja = build_climatology(pred_all, [6,7,8], out_jja)

# ----------------------------
# RUN: JULY climatology
# ----------------------------
out_jul = os.path.join(output_dir, "era5land_climatology_JULY_10_18.nc")
clim_jul = build_climatology(pred_all, [7], out_jul)