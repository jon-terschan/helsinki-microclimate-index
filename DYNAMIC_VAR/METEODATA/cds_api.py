# Purpose: download data from copernicus data store using cdsapi
# REFERENCE: https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+Windows
# import api, only works if the corresponding key file has been set up correctly
# ERA5-Land downloads seem to be very strictly resource capped, at least I have not
# been able to download more than a single month of data at once

import cdsapi
import os # for filepath concat

# initialize API 
client = cdsapi.Client()

# ERA5-LAND - SINGLE MONTH DOWNLOAD
target_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5"
dataset_era = "reanalysis-era5-land" #era5-land
era_request = {
    "variable": [
        "2m_temperature",
        "surface_solar_radiation_downwards",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_precipitation"
    ],
    "year": ["2024"],
    "month": ["05"], # "06","07" "08", "09"
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [60.5, 24.7, 60.0, 25.5] # generous bounding box
}
# "area": [60.3242, 24.7828, 60.1247, 25.3059] bounding box of master template
# "area": [60.5, 24.7, 60.0, 25.5] # nicer bounding box 
# ERA5 LAND INTERPOLATION Bug: will be fixed 25.2 according to the forums: 
# https://forum.ecmwf.int/t/software-upgrade-for-data-extraction-of-a-geographical-area-from-selected-era5-and-seasonal-forecast-datasets/14583
# ERA5 LAND INTERPOLATION BUg HAS BEEN FIXED 25.02
#"area": [60.02, 24.7, 60.35, 25.30] initial request
#"area": [59.9, 24.5, 60.4, 25.4] # regular grid

# extract year and month from request 
year = era_request["year"][0]
month = era_request["month"][0]
filename = f"ERA5l_{year}_{month}_5.netcdf" # name filename
era_target = os.path.join(target_dir, filename)
client.retrieve(dataset_era, era_request, era_target) # execute API request

# ERA5-LAND - MULTI - MONTH/YEAR DOWNLOAD
import cdsapi
import os # for filepath concat

# initialize API 
client = cdsapi.Client()
target_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5"
dataset_era = "reanalysis-era5-land"
years = ["2024", "2025"]
months = ["05", "06", "07", "08", "09"]

for year in years:
    for month in months:
        era_request = {
            "variable": [
                "2m_temperature",
                "surface_solar_radiation_downwards",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "total_precipitation"
            ],
            "year": [year],
            "month": [month],
            "day": [
                "01","02","03","04","05","06","07","08","09","10",
                "11","12","13","14","15","16","17","18","19","20",
                "21","22","23","24","25","26","27","28","29","30","31"
            ],
            "time": [
                "00:00","01:00","02:00","03:00","04:00","05:00",
                "06:00","07:00","08:00","09:00","10:00","11:00",
                "12:00","13:00","14:00","15:00","16:00","17:00",
                "18:00","19:00","20:00","21:00","22:00","23:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [60.5, 24.7, 60.0, 25.5] 
        }

        filename = f"ERA5l_{year}_{month}.netcdf"
        era_target = os.path.join(target_dir, filename)

        print(f"Downloading {year}-{month}!")
        client.retrieve(dataset_era, era_request, era_target)

# CERRA (not needed here)
#dataset = "reanalysis-cerra-single-levels" #CERRA
# CERRA API REQUEST
#cerra_request = {
#    "variable": ["2m_temperature"],
#    "level_type": "surface_or_atmosphere",
#    "data_type": ["reanalysis"],
#    "product_type": "analysis",
#    "year": ["2024", "2025"],
#    "month": ["05", "06", "07", "08", "09"],
#    "day": [
#        "01", "02", "03",
#        "04", "05", "06",
#        "07", "08", "09",
#        "10", "11", "12",
#        "13", "14", "15",
#        "16", "17", "18",
#        "19", "20", "21",
#        "22", "23", "24",
#        "25", "26", "27",
#        "28", "29", "30",
#        "31"
#    ],
#    "time": [
#        "00:00", "03:00", "06:00", 
#        "09:00", "12:00", "15:00",
#        "18:00", "21:00"
#    ],
#    "data_format": "netcdf",
    #"download_format": "unarchived",
    #"area": [60.05, 24.7, 60.3, 25.28] #CERRA is on some weird grid and shouldnt be handled like that
#}

#cerra_target = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\CERRA_SUMMER_24_25.netcdf" # target folder and name
#client.retrieve(dataset, cerra_request, cerra_target) # execute API request
