################################
### extended climatology baseline (P90) ###
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
# FUNCTION: build climatology (P90)
# ----------------------------
def build_climatology_p90(pred, months, out_path):

    pred = pred.sel(valid_time=pred["valid_time"].dt.month.isin(months))

    clim_list = []
    time_list = []

    for local_hour in target_local_hours:

        utc_hour = (local_hour - utc_offset) % 24

        sub = pred.sel(valid_time=pred["valid_time"].dt.hour == utc_hour)
        if sub.sizes["valid_time"] == 0:
            raise ValueError(f"No data for UTC {utc_hour}")

        # ----------------------------
        # P90 instead of mean
        # ----------------------------
        clim = sub.quantile(0.9, dim="valid_time", skipna=True)

        # safe removal of quantile dim (only if it exists)
        if "quantile" in clim.dims:
            clim = clim.squeeze("quantile", drop=True)

        clim = clim[[
            "t2m","u10","v10","wind_s","ssrd","tp",
            "hour_sin","hour_cos","doy_sin","doy_cos"
        ]]

        # overwrite encodings
        clim["hour_sin"] = xr.full_like(clim["t2m"], np.sin(2*np.pi*utc_hour/24))
        clim["hour_cos"] = xr.full_like(clim["t2m"], np.cos(2*np.pi*utc_hour/24))
        clim["doy_sin"]  = xr.full_like(clim["t2m"], float(sub["doy_sin"].mean()))
        clim["doy_cos"]  = xr.full_like(clim["t2m"], float(sub["doy_cos"].mean()))

        # ----------------------------
        # fill missing cell
        # ----------------------------
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

        # ----------------------------
        # UNIX timestamp
        # ----------------------------
        ts = pd.Timestamp(f"2000-07-15 {utc_hour:02d}:00:00", tz="UTC")
        time_list.append(np.int64(ts.timestamp()))

        clim_list.append(clim)

    # ----------------------------
    # concat
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
        "calendar": "standard"
    }

    clim_all.to_netcdf(out_path)

    print("\nSaved:", out_path)
    print("Dims:", clim_all.dims)
    print("Times:", clim_all.valid_time.values)

    return clim_all


# ----------------------------
# RUN: JJA P90 climatology
# ----------------------------
out_jja = os.path.join(output_dir, "era5land_climatology_JJA_10_18_P90.nc")
clim_jja = build_climatology_p90(pred_all, [6,7,8], out_jja)

# ----------------------------
# RUN: JULY P90 climatology
# ----------------------------
out_jul = os.path.join(output_dir, "era5land_climatology_JULY_10_18_P90.nc")
clim_jul = build_climatology_p90(pred_all, [7], out_jul)

print(clim_jul["t2m"].mean().values)
print(clim_jul["t2m"].max().values)