# ERA5-LAND - MULTI - MONTH/YEAR DOWNLOAD
import cdsapi
import os
import pandas as pd
import xarray as xr

client = cdsapi.Client()

## TMAX -DAILY MAX TEMP ####
############################
target_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\recent_tmax"

dataset = "derived-era5-land-daily-statistics"
years = [str(y) for y in range(2010, 2025)]
months = ["06", "07", "08"]

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
import xarray as xr
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
    "H1":[("2010","07",["24","25","26","27","28","29","30"])],
    "H2":[
        ("2018","07",["21","22","23","24","25","26","27","28","29","30","31"]),
        ("2018","08",["01","02","03","04"])
    ],
    "H3":[("2021","07",["09","10","11","12","13","14","15","16","17","18","19","20"])]
}

for event, parts in events.items():

    final_file = target_dir / f"ERA5L_hourly_{event}.nc"
    if final_file.exists():
        print(f"Skipping {event}, already finished")
        continue

    monthly_files = []

    for year, month, days in parts:

        tmp = target_dir / f"{event}_{year}_{month}.nc"

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
        client.retrieve(dataset, request, tmp)

        # unzip if CDS wrapped file
        if zipfile.is_zipfile(tmp):
            with zipfile.ZipFile(tmp) as z:
                ncfile = [m for m in z.namelist() if m.endswith(".nc")][0]
                z.extract(ncfile, target_dir)

            extracted = target_dir / ncfile
            tmp.unlink()
            extracted.rename(tmp)

        monthly_files.append(tmp)

    # merge if multiple months
    if len(monthly_files) == 1:
        monthly_files[0].rename(final_file)
    else:
        print(f"Merging {event}")
        datasets = [xr.open_dataset(f) for f in monthly_files]
        xr.concat(datasets, dim="time").to_netcdf(final_file)

        for ds in datasets:
            ds.close()

        for f in monthly_files:
            f.unlink()

    print(f"Saved {final_file}")

print("All events completed.")