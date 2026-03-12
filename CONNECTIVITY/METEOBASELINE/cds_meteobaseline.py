# ERA5-LAND - MULTI - MONTH/YEAR DOWNLOAD
import cdsapi
import os

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