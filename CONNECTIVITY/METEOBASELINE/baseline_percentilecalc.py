import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# paths
baseline_dir = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline_tmax")
output_dir   = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\thresholds")
output_file  = output_dir / "CTX90_thresholds.nc"
output_dir.mkdir(parents=True, exist_ok=True)

# load them all
files = sorted(baseline_dir.glob("ERA5l_tmax_*.nc"))
ds = xr.open_mfdataset(files, combine="by_coords")

# celsius conversion
tmax = ds["t2m"]
if tmax.mean() > 200:
    tmax = tmax - 273.15
    print("Converted K -> C")

# build datetime coordinate
season_days = pd.date_range("2001-05-01", "2001-09-30")
doy_labels  = [d.strftime("%m-%d") for d in season_days]

# assign md coordinate
tmax_md = tmax.assign_coords(md=("valid_time", tmax.valid_time.dt.strftime("%m-%d").values))

# calculate values
threshold_list = []
mean_list = []

for i, md in enumerate(doy_labels):
    window_indices = range(max(0, i - 7), min(len(doy_labels), i + 8))
    window_mds = [doy_labels[j] for j in window_indices]

    window_data = tmax_md.sel(valid_time=tmax_md.md.isin(window_mds))

    # p90
    p90 = window_data.quantile(0.90, dim="valid_time").drop_vars("quantile")
    p90 = p90.expand_dims(md=[md])
    threshold_list.append(p90)

    # average tmax
    mean = window_data.mean(dim="valid_time")
    mean = mean.expand_dims(md=[md])
    mean_list.append(mean)

    if i % 20 == 0:
        print(f"  Processed {md} ({i+1}/{len(doy_labels)})...")

#  combine and save
thresholds = xr.concat(threshold_list, dim="md")
thresholds.name = "ctx90_thresh"
thresholds.attrs["description"] = "CTX90 90th percentile Tmax, 15-day clipped window, baseline 1961-1990"
thresholds.attrs["units"] = "degC"

mean_tmax = xr.concat(mean_list, dim="md")
mean_tmax.name = "mn_tmax"
mean_tmax.attrs["description"] = "Mean Tmax, 15-day window, baseline 1961-1990"
mean_tmax.attrs["units"] = "degC"

out_ds = xr.Dataset({"ctx90_threshold": thresholds, "mean_tmax": mean_tmax})
out_ds.to_netcdf(output_file)

# plot
p90_mean   = thresholds.mean(dim=["latitude", "longitude"])
tmax_mean  = mean_tmax.mean(dim=["latitude", "longitude"])
x = range(153)
xticks = range(0, 153, 15)
xlabels = [doy_labels[i] for i in xticks]

plt.figure(figsize=(12, 4))
plt.plot(x, p90_mean.values, label="Tmax p90 (CTX90 threshold)", color="tomato")
plt.plot(x, tmax_mean.values, label="Tmax mean", color="steelblue")
plt.fill_between(x, tmax_mean.values, p90_mean.values, alpha=0.15, color="tomato")
plt.xticks(ticks=xticks, labels=xlabels, rotation=45)
plt.ylabel("Temperature (°C)")
plt.title("CTX90 threshold vs average max temperature during Helsinki summers Helsinki, 1961–1990")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()