# ERA5-LAND - MULTI - MONTH/YEAR DOWNLOAD
import cdsapi
import os
import pandas as pd
import xarray as xr

client = cdsapi.Client()

## TMAX -DAILY MAX TEMP ####
############################
target_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\1990-2020"

dataset = "derived-era5-land-daily-statistics"
years = [str(y) for y in range(1989, 2022)]
months = ["05", "06", "07", "08", "09"]

for year in years:
    for month in months:
        filename = f"ERA5l_tmax_{year}_{month}.nc"
        target = os.path.join(target_dir, filename)

        if os.path.exists(target):
            print(f"Skipping {year}-{month} because the file already exists!")
            continue

        request = {
            "variable": "2m_temperature",
            "daily_statistic": ["daily_maximum"],
            "year": year,
            "month": month,
            "day": [
                "01","02","03","04","05","06","07","08","09","10",
                "11","12","13","14","15","16","17","18","19","20",
                "21","22","23","24","25","26","27","28","29","30","31"
            ],
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "data_format": "netcdf",
            "area": [60.5, 24.7, 60.0, 25.5]
        }

        print(f"Requesting Tmax {year}-{month}!")
        try:
            client.retrieve(dataset, request, target)
        except Exception as e:
            print(f"  FAILED {year}-{month}: {e}")

### TMEAN -DAILY MN TEMP ###
############################
# not sure if necessary but why not just in case

client = cdsapi.Client() # create new client

target_dir_mn = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\mean"

dataset = "derived-era5-land-daily-statistics"
years = [str(y) for y in range(1961, 1991)]  # 1961-1990
months = ["05", "06", "07", "08", "09"]

for year in years:
    for month in months:
        request = {
            "variable": "2m_temperature",
            "daily_statistic": ["daily_maximum"],
            "year": year,
            "month": month,
            "day": [
                "01","02","03","04","05","06","07","08","09","10",
                "11","12","13","14","15","16","17","18","19","20",
                "21","22","23","24","25","26","27","28","29","30","31"
            ],
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "data_format": "netcdf",
            "area": [60.5, 24.7, 60.0, 25.5]
        }
        filename = f"ERA5l_tmn_{year}_{month}.nc"
        target = os.path.join(target_dir_mn, filename)
        print(f"Requesting Tmean {year}-{month}!")
        client.retrieve(dataset, request, target)


# ERA5-MODEL HOURLY HEATWAVE DOWNLOADER (JUST FOR PREDICTIONS)
# also this downloads zips for some reason
import os
import cdsapi
import zipfile
from pathlib import Path

client = cdsapi.Client()

target_dir = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves_hourly")
target_dir.mkdir(exist_ok=True)

dataset = "reanalysis-era5-land"

variables = [
    "2m_temperature",
    "surface_solar_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation"
]

times = [f"{h:02d}:00" for h in range(24)]

events = {
    #H1":[("2010","07",["24","25","26","27","28","29","30"])],
    "H2":[
        ("2018","07",["21","22","23","24","25","26","27","28","29","30","31"]),
        ("2018","08",["01","02","03","04"])
    ],
    #"H3":[("2021","07",["09","10","11","12","13","14","15","16","17","18","19","20"])]
}

for event, parts in events.items():

    for year, month, days in parts:

        out_file = target_dir / f"ERA5L_hourly_{event}_{year}_{month}.nc"

        if out_file.exists():
            print(f"Skipping {out_file.name}")
            continue

        request = {
            "product_type": "reanalysis",
            "variable": variables,
            "year": year,
            "month": month,
            "day": days,
            "time": times,
            "format": "netcdf",
            "area": [60.5, 24.7, 60.0, 25.5]
        }

        print(f"Downloading {event} {year}-{month}")
        client.retrieve(dataset, request, out_file)

        # handle CDS zip wrapping
        if zipfile.is_zipfile(out_file):
            with zipfile.ZipFile(out_file) as z:
                ncfile = [m for m in z.namelist() if m.endswith(".nc")][0]
                z.extract(ncfile, target_dir)

            extracted = target_dir / ncfile
            out_file.unlink()
            extracted.rename(out_file)

        print(f"Saved {out_file}")

print("All downloads completed.")


#### download baseline data

import cdsapi
import os

client = cdsapi.Client()

target_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\hourly"

dataset = "reanalysis-era5-land"

years = [str(y) for y in range(1989, 2021)]
months = ["05", "06", "07", "08", "09"]

variables = [
    "2m_temperature",
    "surface_solar_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation"
]

times = [f"{h:02d}:00" for h in range(24)]

for year in years:
    for month in months:
        filename = f"ERA5l_hourly_{year}_{month}.zip"
        target = os.path.join(target_dir, filename)

        if os.path.exists(target):
            print(f"Skipping {year}-{month}, already exists")
            continue

        request = {
            "variable": variables,
            "year": year,
            "month": month,
            "day": [
                "01","02","03","04","05","06","07","08","09","10",
                "11","12","13","14","15","16","17","18","19","20",
                "21","22","23","24","25","26","27","28","29","30","31"
            ],
            "time": times,   # important: zipped output
            "format": "netcdf",
            "area": [60.5, 24.7, 60.0, 25.5]
        }

        print(f"Requesting {year}-{month}")
        try:
            client.retrieve(dataset, request, target)
        except Exception as e:
            print(f"FAILED {year}-{month}: {e}")