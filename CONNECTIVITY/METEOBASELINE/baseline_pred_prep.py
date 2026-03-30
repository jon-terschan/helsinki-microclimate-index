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

climatology = climatology.expand_dims(
    valid_time=[np.datetime64("2000-07-15T13:00")] # fake dummy timestamp for compatibility with training data structuree
)

climatology.to_netcdf(out_nc)

print(f"Saved climatology to: {out_nc}")
print(climatology.dims)