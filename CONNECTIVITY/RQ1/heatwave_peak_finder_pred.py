import rasterio
import numpy as np
import pandas as pd
import glob
import os

# -----------------------
# PATHS
# -----------------------
h2010 = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\2010\20100728"
h2018 = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\2018\20180717"
h2021 = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\2021\20210714"

# -----------------------
# PEAK HOUR FUNCTION
# -----------------------
def get_peak_hour(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))

    stack = []
    hours = []

    for f in files:
        hour = int(os.path.basename(f).split("_")[-1][:2])
        hours.append((hour + 3) % 24)  # UTC → local

        with rasterio.open(f) as src:
            arr = src.read(1).astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            stack.append(arr)

    stack = np.stack(stack, axis=0)
    hours = np.array(hours)

    valid_mask = np.any(~np.isnan(stack), axis=0)

    peak_idx = np.full(stack.shape[1:], np.nan)
    peak_idx[valid_mask] = np.nanargmax(stack[:, valid_mask], axis=0)

    peak_hour = np.full_like(peak_idx, np.nan)
    peak_hour[valid_mask] = hours[peak_idx[valid_mask].astype(int)]

    flat = peak_hour.flatten()
    flat = flat[~np.isnan(flat)]

    flat_series = pd.Series(flat)

    median_hour = np.median(flat)
    mean_hour   = np.mean(flat)
    mode_hour   = flat_series.mode()[0]
    counts      = flat_series.value_counts().sort_index()

    return mode_hour, median_hour, mean_hour, counts

# -----------------------
# DIAGNOSTIC FUNCTION
# -----------------------
def diagnose_peak_hour(folder, utc_to_local_shift=3, tol_list=(0.0, 0.1, 0.25, 0.5)):
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))
    print("nfiles:", len(files))
    print("first/last:", os.path.basename(files[0]), os.path.basename(files[-1]))

    hours_utc = [int(os.path.basename(f).split("_")[-1][:2]) for f in files]
    hours_local = [((h + utc_to_local_shift) % 24) for h in hours_utc]

    print("UTC hours:   ", hours_utc)
    print("LOCAL hours: ", hours_local)
    print("missing UTC:  ", sorted(set(range(24)) - set(hours_utc)))
    print("missing LOCAL:", sorted(set(range(24)) - set(hours_local)))

    stack = []
    for f in files:
        with rasterio.open(f) as src:
            arr = src.read(1).astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            stack.append(arr)

    stack = np.stack(stack, axis=0)
    print("stack shape:", stack.shape)
    print("NaN fraction:", np.isnan(stack).mean())

    valid_mask = np.any(~np.isnan(stack), axis=0)
    print("valid pixels:", valid_mask.sum(), "/", valid_mask.size)

    hours_local = np.array(hours_local)

    maxv = np.nanmax(stack, axis=0)

    strict_idx = np.full(stack.shape[1:], np.nan)
    strict_idx[valid_mask] = np.nanargmax(stack[:, valid_mask], axis=0)

    strict_hour = np.full_like(strict_idx, np.nan)
    strict_hour[valid_mask] = hours_local[strict_idx[valid_mask].astype(int)]

    flat = pd.Series(strict_hour[valid_mask].ravel())

    print("\nSTRICT argmax:")
    print("mode  :", flat.mode().iloc[0])
    print("median:", flat.median())
    print("mean  :", flat.mean())
    print(flat.value_counts().sort_index())

    for tol in tol_list:
        near = stack >= (maxv - tol)
        near = np.where(np.isnan(stack), False, near)

        idx = np.argmax(near, axis=0)
        hour = hours_local[idx][valid_mask]
        s = pd.Series(hour.ravel())

        print(f"\nTOL = {tol}")
        print("mode  :", s.mode().iloc[0])
        print("median:", s.median())
        print("mean  :", s.mean())
        print(s.value_counts().sort_index())

# -----------------------
# RUN ANALYSIS
# -----------------------
for label, path in [
    ("2010-07-28", h2010),
    ("2018-07-17", h2018),
    ("2021-07-14", h2021)
]:
    mode, median, mean, counts = get_peak_hour(path)

    print(f"\n{label}")
    print(f"Mode   (MODEL): {mode}")
    print(f"Median (MODEL): {median}")
    print(f"Mean   (MODEL): {mean}")
    print("Hour distribution:")
    print(counts)

# -----------------------
# RUN DIAGNOSTICS
# -----------------------
for label, path in [
    ("2010-07-28", h2010),
    ("2018-07-17", h2018),
    ("2021-07-14", h2021)
]:
    print(f"\n===== {label} =====")
    diagnose_peak_hour(path)