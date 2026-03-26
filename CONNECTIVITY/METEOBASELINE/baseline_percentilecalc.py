import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# paths
baseline_dir = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\1990-2020")
output_dir   = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\thresholds")
output_file  = output_dir / "CTX90_thresholds_1990-2020.nc"
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
thresholds.attrs["description"] = "CTX90 90th percentile Tmax, 15-day clipped window, baseline 1990-2020"
thresholds.attrs["units"] = "degC"

mean_tmax = xr.concat(mean_list, dim="md")
mean_tmax.name = "mn_tmax"
mean_tmax.attrs["description"] = "Mean Tmax, 15-day window, baseline 1990-2020"
mean_tmax.attrs["units"] = "degC"

out_ds = xr.Dataset({"ctx90_threshold": thresholds, "mean_tmax": mean_tmax})
out_ds.to_netcdf(output_file)

# plot
p90_mean   = thresholds.mean(dim=["latitude", "longitude"])
tmax_mean  = mean_tmax.mean(dim=["latitude", "longitude"])
x = range(153)
xticks = range(0, 153, 15)
xlabels = [doy_labels[i] for i in xticks]

# check baseline
plt.figure(figsize=(12, 4))
plt.plot(x, p90_mean.values, label="Tmax p90 1990-2020)", color="tomato")
plt.plot(x, tmax_mean.values, label="Tmax mean 1990-2020", color="steelblue")
plt.fill_between(x, tmax_mean.values, p90_mean.values, alpha=0.15, color="tomato")
plt.xticks(ticks=xticks, labels=xlabels, rotation=45)
plt.ylabel("Temperature (°C)")
plt.title("CTX90 threshold vs average max temperature during Helsinki summers Helsinki, 1990–2020")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# compare old and new climatology
old_file = output_dir / "CTX90_thresholds_1960-1990.nc"
old_ds = xr.open_dataset(old_file)

old_thresh = old_ds["ctx90_threshold"]
old_mean   = old_ds["mean_tmax"]

# spatial averages
old_p90_mean  = old_thresh.mean(dim=["latitude", "longitude"])
old_tmax_mean = old_mean.mean(dim=["latitude", "longitude"])


plt.figure(figsize=(12, 4))
# 1990–2020
plt.plot(x, p90_mean.values, label="Tmax p90 1990–2020", color="tomato")
plt.plot(x, tmax_mean.values, label="Tmax mean 1990–2020", color="steelblue")
# 1960–1990
plt.plot(x, old_p90_mean.values, label="Tmax p90 1960–1990", linestyle="--", color="darkred")
plt.plot(x, old_tmax_mean.values, label="Tmax mean 1960–1990", linestyle="--", color="navy")

plt.fill_between(x, tmax_mean.values, p90_mean.values, alpha=0.15, color="tomato") # shading 
plt.xticks(ticks=xticks, labels=xlabels, rotation=45)
plt.ylabel("Temperature (°C)")
plt.title("mean, 90p of maximum 2m air temperature in Helsinki")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\appendix\climatology_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()


#######################################################
# FIGURE CODE AGAIN, BUT LOADING THE FINISHED TIME IN
#######################################################

# load finished 1990–2020 thresholds file
new_ds = xr.open_dataset(output_file)

p90 = new_ds["ctx90_threshold"]
mean = new_ds["mean_tmax"]

# spatial averages
p90_mean  = p90.mean(dim=["latitude", "longitude"])
tmax_mean = mean.mean(dim=["latitude", "longitude"])

# x-axis setup
x = range(len(p90_mean))
xticks = range(0, len(p90_mean), 15)
xlabels = [p90.md.values[i] for i in xticks]


# ---- plot baseline only ----
plt.figure(figsize=(12, 4))
plt.plot(x, p90_mean.values, label="Tmax p90 1990–2020", color="tomato")
plt.plot(x, tmax_mean.values, label="Tmax mean 1990–2020", color="steelblue")
plt.fill_between(x, tmax_mean.values, p90_mean.values, alpha=0.15, color="tomato")

plt.xticks(ticks=xticks, labels=xlabels, rotation=45)
plt.ylabel("Temperature (°C)")
plt.title("CTX90 threshold vs average max temperature (Helsinki, 1990–2020)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ---- compare with old climatology ----
old_file = output_dir / "CTX90_thresholds_1960-1990.nc"
old_ds = xr.open_dataset(old_file)

old_p90_mean  = old_ds["ctx90_threshold"].mean(dim=["latitude", "longitude"])
old_tmax_mean = old_ds["mean_tmax"].mean(dim=["latitude", "longitude"])

plt.figure(figsize=(14, 6))  # wider + taller

# increase global font sizes
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

# plotting (same as before)
plt.plot(x, p90_mean.values, label="Tmax p90 1990–2020", color="tomato", linewidth=2)
plt.plot(x, tmax_mean.values, label="Tmax mean 1990–2020", color="steelblue", linewidth=2)

plt.plot(x, old_p90_mean.values, label="Tmax p90 1960–1990", linestyle="--", color="darkred", linewidth=2)
plt.plot(x, old_tmax_mean.values, label="Tmax mean 1960–1990", linestyle="--", color="navy", linewidth=2)

plt.fill_between(x, tmax_mean.values, p90_mean.values, alpha=0.2, color="tomato")

plt.xticks(ticks=xticks, labels=xlabels, rotation=45)
plt.ylabel("Temperature (°C)")
plt.title("Mean and 90th percentile Tmax (Helsinki)")
plt.legend()

plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\appendix\climatology_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()