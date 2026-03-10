# ERA5-LAND - MULTI - MONTH/YEAR DOWNLOAD
import cdsapi
import os

client = cdsapi.Client()
target_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline"

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
        filename = f"ERA5l_tmax_{year}_{month}.nc"
        target = os.path.join(target_dir, filename)
        print(f"Downloading Tmax {year}-{month}...")
        client.retrieve(dataset, request, target)