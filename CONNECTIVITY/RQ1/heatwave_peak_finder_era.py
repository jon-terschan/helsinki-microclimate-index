# the purpose of this script is to find the peak hour of the heatwaves in local time 
# we did this to find the correct raster layer from the hourly predicted data for the connectivitiy analysis.
# since our heatwave identification is based on daily max temperatures, the datetime handle does not contain hourly information
# which is why this is a necessary step

import cdsapi
import os
import xarray as xr
import pandas as pd

# ---------------------------
# SETTINGS
# ---------------------------
target_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly\peak_days"

dates = [
    "2010-07-28",
    "2018-07-17",
    "2021-07-14"
]

area = [60.5, 24.7, 60.0, 25.5]
hours = [f"{h:02d}:00" for h in range(24)]

# ---------------------------
# DOWNLOAD ERA5-LAND HOURLY
# ---------------------------
client = cdsapi.Client()

for date in dates:
    y, m, d = date.split("-")
    filename = f"ERA5l_hourly_{date}.nc"
    target = os.path.join(target_dir, filename)

    if os.path.exists(target):
        print(f"Skipping {date}, already exists")
        continue

    request = {
        "variable": "2m_temperature",
        "year": y,
        "month": m,
        "day": d,
        "time": hours,
        "data_format": "netcdf",
        "area": area
    }

    print(f"Downloading {date}")
    client.retrieve("reanalysis-era5-land", request, target)

# ---------------------------
# ANALYSIS: PEAK HOUR (LOCAL TIME, TZ-SAFE)
# ---------------------------
import zipfile
import os

def open_era5_file(path):
    with open(path, "rb") as f:
        sig = f.read(4)

    if sig.startswith(b'PK'):
        with zipfile.ZipFile(path, 'r') as z:
            nc_files = [f for f in z.namelist() if f.endswith(".nc")]

            if not nc_files:
                raise RuntimeError(f"No NetCDF inside ZIP: {path}")

            # create unique output name
            out_name = os.path.basename(path).replace(".nc", "_extracted.nc")
            out_path = os.path.join(os.path.dirname(path), out_name)

            # overwrite safely
            if os.path.exists(out_path):
                os.remove(out_path)

            with z.open(nc_files[0]) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

            return xr.open_dataset(out_path)

    else:
        return xr.open_dataset(path)

for date in dates:
    filename = f"ERA5l_hourly_{date}.nc"
    path = os.path.join(target_dir, filename)

    ds = open_era5_file(path)

    # Convert Kelvin → Celsius
    t = ds['t2m'] - 273.15

    # Get timestamp of max temperature per grid cell
    peak_time = t.idxmax(dim="valid_time")

    # Convert to local hour (UTC+3 for July)
    peak_hour_local = (peak_time.dt.hour + 3) % 24

    # Flatten and compute mode
    flat = peak_hour_local.values.flatten()
    mode_hour = pd.Series(flat).mode()[0]

    # Distribution
    counts = pd.Series(flat).value_counts().sort_index()

    print(f"\n{date}")
    print(f"Dominant peak hour (LOCAL): {mode_hour}")
    print("Hour distribution:")
    print(counts)