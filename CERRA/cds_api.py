# Purpose: download data from copernicus data store using cdsapi
# REFERENCE: https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+Windows
# import api, only works if the corresponding key file has been set up correctly
import cdsapi

# initialize API 
client = cdsapi.Client()

# CERRA 
dataset = "reanalysis-cerra-single-levels" #CERRA
# CERRA API REQUEST
cerra_request = {
    "variable": ["2m_temperature"],
    "level_type": "surface_or_atmosphere",
    "data_type": ["reanalysis"],
    "product_type": "analysis",
    "year": ["2024", "2025"],
    "month": ["05", "06", "07", "08", "09"],
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
        "00:00", "03:00", "06:00", 
        "09:00", "12:00", "15:00",
        "18:00", "21:00"
    ],
    "data_format": "netcdf",
    #"download_format": "unarchived",
    #"area": [60.05, 24.7, 60.3, 25.28] #CERRA is on some weird grid and shouldnt be handled like that
}

#cerra_target = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\CERRA_SUMMER_24_25.netcdf" # target folder and name
#client.retrieve(dataset, cerra_request, cerra_target) # execute API request

# ERA5-LAND REQUEST
dataset_era = "reanalysis-era5-land" #era5-land
era_request = {
    "variable": [
        "2m_temperature",
        "surface_solar_radiation_downwards",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_precipitation"
    ],
    "year": ["2025"],
    "month": ["09"], # "06","07" "08", "09"
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
    "area": [59.9, 24.5, 60.4, 25.4] # regular grid
}

#"area": [60.02, 24.7, 60.35, 25.30] initial request

era_target = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\ERA\HELNORTH\ERA_SUMMER_25_09.netcdf" # target folder and name
client.retrieve(dataset_era, era_request, era_target) # execute API request

# dummy testing code from documentation
#import cdsapi
#client = cdsapi.Client()
#dataset = 'reanalysis-era5-pressure-levels'
#request = {
#    'product_type': ['reanalysis'],
#    'variable': ['geopotential'],
#    'year': ['2024'],
#    'month': ['03'],
#    'day': ['01'],
#    'time': ['13:00'],
#    'pressure_level': ['1000'],
#    'data_format': 'grib',
#}
#target = 'download.grib'
#client.retrieve(dataset, request, target)
