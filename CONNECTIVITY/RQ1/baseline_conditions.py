import os
import numpy as np
import rasterio
import xarray as xr
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

# =============================
# USER SETTINGS
# =============================
baseline_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\baseline\15cm_July_allday"
era5_path    = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\climatology\era5land_climatology_JULY_10_18.nc"

tree_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\TREE_FRAC_10m.tif"
nwn_path  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\NWN_FRAC_10m.tif"

target_local_hour = 13
utc_offset = 3

# =============================
# TIME HANDLING
# =============================
utc_hour = (target_local_hour - utc_offset) % 24
hh_str = f"{utc_hour:02d}00"

temp_path = os.path.join(baseline_dir, f"pred_20000715_{hh_str}.tif")

# =============================
# LOAD TEMPERATURE
# =============================
with rasterio.open(temp_path) as src:
    temp = src.read(1).astype("float32")
    nodata = src.nodata
    transform = src.transform
    crs = src.crs

if nodata is not None:
    temp[temp == nodata] = np.nan

mask_valid = ~np.isnan(temp)

# =============================
# LOAD FRACTIONS
# =============================
with rasterio.open(tree_path) as src:
    tree = src.read(1).astype("float32")

with rasterio.open(nwn_path) as src:
    nwn = src.read(1).astype("float32")

tree = np.clip(tree, 0, 1)
nwn  = np.clip(nwn, 0, 1)

# residual urban fraction
urban = np.clip(1 - (tree + nwn), 0, 1)

# =============================
# BASIC STATS
# =============================
def mean_std(arr):
    return np.nanmean(arr), np.nanstd(arr)

mean_all, std_all = mean_std(temp)

# =============================
# AREA-WEIGHTED STATS (NO THRESHOLDS)
# =============================
def weighted_stats(values, weights):
    mask = (~np.isnan(values)) & (~np.isnan(weights))
    v = values[mask]
    w = weights[mask]

    if np.sum(w) == 0:
        return np.nan, np.nan

    mean = np.sum(w * v) / np.sum(w)
    var  = np.sum(w * (v - mean)**2) / np.sum(w)
    return mean, np.sqrt(var)

mean_tree_w, std_tree_w = weighted_stats(temp, tree)
mean_nwn_w,  std_nwn_w  = weighted_stats(temp, nwn)
mean_urban_w, std_urban_w = weighted_stats(temp, urban)

# =============================
# DOMINANT CLASS (ARGMAX)
# =============================
stack = np.stack([tree, nwn, urban], axis=0)
dominant = np.argmax(stack, axis=0)

tree_dom  = (dominant == 0) & mask_valid
nwn_dom   = (dominant == 1) & mask_valid
urban_dom = (dominant == 2) & mask_valid

def masked_stats(arr, mask):
    vals = arr[mask]
    return np.nanmean(vals), np.nanstd(vals)

mean_tree_d, std_tree_d = masked_stats(temp, tree_dom)
mean_nwn_d,  std_nwn_d  = masked_stats(temp, nwn_dom)
mean_urban_d, std_urban_d = masked_stats(temp, urban_dom)

# =============================
# PRINT RESULTS
# =============================
print("=== OVERALL ===")
print(f"{mean_all:.2f} ± {std_all:.2f} °C\n")

print("=== AREA-WEIGHTED ===")
print(f"Tree:  {mean_tree_w:.2f} ± {std_tree_w:.2f} °C")
print(f"NWN:   {mean_nwn_w:.2f} ± {std_nwn_w:.2f} °C")
print(f"Urban: {mean_urban_w:.2f} ± {std_urban_w:.2f} °C\n")

print("=== DOMINANT CLASS ===")
print(f"Tree:  {mean_tree_d:.2f} ± {std_tree_d:.2f} °C")
print(f"NWN:   {mean_nwn_d:.2f} ± {std_nwn_d:.2f} °C")
print(f"Urban: {mean_urban_d:.2f} ± {std_urban_d:.2f} °C")

# =============================
# VISUALIZATION (DISTRIBUTIONS)
# =============================
plt.figure(figsize=(8,6))

plt.hist(temp[tree_dom], bins=50, alpha=0.4, label="Tree-dominated", density=True)
plt.hist(temp[nwn_dom],  bins=50, alpha=0.4, label="NWN-dominated", density=True)
plt.hist(temp[urban_dom],bins=50, alpha=0.4, label="Urban-dominated", density=True)

plt.xlabel("Temperature (°C)")
plt.ylabel("Density")
plt.legend()
plt.title("Temperature distributions by dominant land cover (13:00)")
plt.tight_layout()
plt.show()



##### 90th percentile
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# =============================
# USER SETTINGS
# =============================
baseline_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\baseline\15cm_July_allday_p90"

tree_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\TREE_FRAC_10m.tif"
nwn_path  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\NWN_FRAC_10m.tif"

target_local_hour = 13
utc_offset = 3

# =============================
# TIME HANDLING
# =============================
utc_hour = (target_local_hour - utc_offset) % 24
hh_str = f"{utc_hour:02d}00"

p90_path = os.path.join(baseline_dir, f"pred_20000715_{hh_str}.tif")

# =============================
# LOAD P90 TEMPERATURE
# =============================
with rasterio.open(p90_path) as src:
    temp_p90 = src.read(1).astype("float32")
    nodata = src.nodata

if nodata is not None:
    temp_p90[temp_p90 == nodata] = np.nan

mask_valid = ~np.isnan(temp_p90)

# =============================
# LOAD FRACTIONS
# =============================
with rasterio.open(tree_path) as src:
    tree = src.read(1).astype("float32")

with rasterio.open(nwn_path) as src:
    nwn = src.read(1).astype("float32")

tree = np.clip(tree, 0, 1)
nwn  = np.clip(nwn, 0, 1)

# residual urban fraction
urban = np.clip(1 - (tree + nwn), 0, 1)

# =============================
# BASIC STATS
# =============================
def mean_std(arr):
    return np.nanmean(arr), np.nanstd(arr)

mean_all, std_all = mean_std(temp_p90)

# =============================
# AREA-WEIGHTED STATS
# =============================
def weighted_stats(values, weights):
    mask = (~np.isnan(values)) & (~np.isnan(weights))
    v = values[mask]
    w = weights[mask]

    if np.sum(w) == 0:
        return np.nan, np.nan

    mean = np.sum(w * v) / np.sum(w)
    var  = np.sum(w * (v - mean)**2) / np.sum(w)
    return mean, np.sqrt(var)

mean_tree_w, std_tree_w   = weighted_stats(temp_p90, tree)
mean_nwn_w,  std_nwn_w    = weighted_stats(temp_p90, nwn)
mean_urban_w, std_urban_w = weighted_stats(temp_p90, urban)

# =============================
# DOMINANT CLASS (ARGMAX)
# =============================
stack = np.stack([tree, nwn, urban], axis=0)
dominant = np.argmax(stack, axis=0)

tree_dom  = (dominant == 0) & mask_valid
nwn_dom   = (dominant == 1) & mask_valid
urban_dom = (dominant == 2) & mask_valid

def masked_stats(arr, mask):
    vals = arr[mask]
    return np.nanmean(vals), np.nanstd(vals)

mean_tree_d, std_tree_d   = masked_stats(temp_p90, tree_dom)
mean_nwn_d,  std_nwn_d    = masked_stats(temp_p90, nwn_dom)
mean_urban_d, std_urban_d = masked_stats(temp_p90, urban_dom)

# =============================
# PRINT RESULTS
# =============================
print("=== OVERALL (P90 FIELD) ===")
print(f"{mean_all:.2f} ± {std_all:.2f} °C\n")

print("=== AREA-WEIGHTED (P90 FIELD) ===")
print(f"Tree:  {mean_tree_w:.2f} ± {std_tree_w:.2f} °C")
print(f"NWN:   {mean_nwn_w:.2f} ± {std_nwn_w:.2f} °C")
print(f"Urban: {mean_urban_w:.2f} ± {std_urban_w:.2f} °C\n")

print("=== DOMINANT CLASS (P90 FIELD) ===")
print(f"Tree:  {mean_tree_d:.2f} ± {std_tree_d:.2f} °C")
print(f"NWN:   {mean_nwn_d:.2f} ± {std_nwn_d:.2f} °C")
print(f"Urban: {mean_urban_d:.2f} ± {std_urban_d:.2f} °C")

# =============================
# VISUALIZATION
# =============================
plt.figure(figsize=(8,6))

plt.hist(temp_p90[tree_dom],  bins=50, alpha=0.4, label="Tree-dominated", density=True)
plt.hist(temp_p90[nwn_dom],   bins=50, alpha=0.4, label="NWN-dominated", density=True)
plt.hist(temp_p90[urban_dom], bins=50, alpha=0.4, label="Urban-dominated", density=True)

plt.xlabel("Temperature (°C, P90)")
plt.ylabel("Density")
plt.legend()
plt.title("P90 temperature distributions by dominant land cover (13:00)")
plt.tight_layout()
plt.show() 