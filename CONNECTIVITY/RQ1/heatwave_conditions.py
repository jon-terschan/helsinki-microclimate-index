"""
=========================================================
URBAN TEMPERATURE ANALYSIS: BASELINE + P90 + HEATWAVES
=========================================================

This script computes temperature statistics for:

1) Baseline climatology (mean and p90)
2) Three heatwave events (2010, 2018, 2021)

Metrics computed:
- Solar noon temperature (≈13:00 local, 10:00 UTC)
- Intra-day temperature range (10:00–18:00 local)

Statistics:
- Overall mean ± std
- Area-weighted mean ± std (tree, NWN, urban)

Outputs:
- Console summaries (detailed diagnostics)
- Unified table (CSV + Excel + formatted Excel)

IMPORTANT:
- Results in this script use AREA-WEIGHTED statistics
- Do not mix with dominant-class results elsewhere

Author: mostly chatgpt lool and a little bit of me 
=========================================================
"""

# IMPORTS
# =============================
import os
import numpy as np
import rasterio
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# USER SETTINGS
# =============================
baseline_mean_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\baseline\15cm_July_allday"
baseline_p90_dir  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\baseline\15cm_July_allday_p90"

events = {
    "HW2010": r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\2010\20100728",
    "HW2018": r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\2018\20180717",
    "HW2021": r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\2021\20210714",
}

tree_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\TREE_FRAC_10m.tif"
nwn_path  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\NWN_FRAC_10m.tif"

out_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\tables"

utc_offset = 3
local_hours = list(range(10, 19))  # 10–18 local

# LOAD LAND COVER
# =============================
print("Loading land cover fractions...")

with rasterio.open(tree_path) as src:
    tree = np.clip(src.read(1).astype("float32"), 0, 1)

with rasterio.open(nwn_path) as src:
    nwn = np.clip(src.read(1).astype("float32"), 0, 1)

urban = np.clip(1 - (tree + nwn), 0, 1)

# HELPER FUNCTIONS
# =============================
def load_raster(path):
    """Load raster and apply nodata mask."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def mean_std(arr):
    """Return mean and std ignoring NaNs."""
    return np.nanmean(arr), np.nanstd(arr)


def weighted_stats(values, weights):
    """Compute area-weighted mean and std."""
    mask = (~np.isnan(values)) & (~np.isnan(weights))
    v = values[mask]
    w = weights[mask]

    if np.sum(w) == 0:
        return np.nan, np.nan

    mean = np.sum(w * v) / np.sum(w)
    var  = np.sum(w * (v - mean)**2) / np.sum(w)
    return mean, np.sqrt(var)


def fmt(m, s):
    """Format mean ± std string."""
    return f"{m:.2f} ± {s:.2f}"


def compute_stats(arr):
    """Compute all area-weighted statistics."""
    m_all, s_all = mean_std(arr)

    m_tree, s_tree = weighted_stats(arr, tree)
    m_nwn,  s_nwn  = weighted_stats(arr, nwn)
    m_urban,s_urban= weighted_stats(arr, urban)

    return {
        "overall": fmt(m_all, s_all),
        "tree":    fmt(m_tree, s_tree),
        "nwn":     fmt(m_nwn, s_nwn),
        "urban":   fmt(m_urban, s_urban),
    }


def load_time_stack(folder):
    """Load all hourly rasters for range calculation."""
    temps = []
    date_str = os.path.basename(folder)

    for h in local_hours:
        utc_hour = (h - utc_offset) % 24
        hh = f"{utc_hour:02d}00"
        path = os.path.join(folder, f"pred_{date_str}_{hh}.tif")

        if os.path.exists(path):
            temps.append(load_raster(path))

    return np.stack(temps)

# BASELINE PROCESSING
# =============================
def compute_baseline_row(label, base_dir):
    """Compute stats for baseline datasets."""
    temps = []

    for h in local_hours:
        utc_hour = (h - utc_offset) % 24
        hh = f"{utc_hour:02d}00"
        path = os.path.join(base_dir, f"pred_20000715_{hh}.tif")
        temps.append(load_raster(path))

    temps = np.stack(temps)

    # Solar noon
    noon_idx = local_hours.index(13)
    temp_noon = temps[noon_idx]

    # Daily range
    temp_range = np.nanmax(temps, axis=0) - np.nanmin(temps, axis=0)

    s_noon = compute_stats(temp_noon)
    s_range = compute_stats(temp_range)

    return {
        "Case": label,
        "T (°C)": s_noon["overall"],
        "Tree (°C)": s_noon["tree"],
        "NWN (°C)": s_noon["nwn"],
        "Urban (°C)": s_noon["urban"],
        "Range (°C)": s_range["overall"],
        "Range Tree (°C)": s_range["tree"],
        "Range NWN (°C)": s_range["nwn"],
        "Range Urban (°C)": s_range["urban"],
    }

# HEATWAVE PROCESSING
# =============================
def compute_event_row(label, folder):
    """Compute stats for heatwave events."""
    date_str = os.path.basename(folder)

    # Solar noon (10 UTC)
    noon_path = os.path.join(folder, f"pred_{date_str}_1000.tif")
    temp_noon = load_raster(noon_path)

    # Range
    stack = load_time_stack(folder)
    temp_range = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)

    s_noon = compute_stats(temp_noon)
    s_range = compute_stats(temp_range)

    return {
        "Case": label,
        "T (°C)": s_noon["overall"],
        "Tree (°C)": s_noon["tree"],
        "NWN (°C)": s_noon["nwn"],
        "Urban (°C)": s_noon["urban"],
        "Range (°C)": s_range["overall"],
        "Range Tree (°C)": s_range["tree"],
        "Range NWN (°C)": s_range["nwn"],
        "Range Urban (°C)": s_range["urban"],
    }

# BUILD TABLE
# =============================
print("Computing statistics...")

rows = []

rows.append(compute_baseline_row("Baseline (mean)", baseline_mean_dir))
rows.append(compute_baseline_row("Baseline (p90)", baseline_p90_dir))

for label, folder in events.items():
    rows.append(compute_event_row(label, folder))

df = pd.DataFrame(rows)

print("\nFinal table:\n")
print(df)

# EXPORT
# =============================
os.makedirs(out_dir, exist_ok=True)

# CSV + Excel
df.to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)
df.to_excel(os.path.join(out_dir, "summary_table.xlsx"), index=False)

# Formatted Excel
wb = Workbook()
ws = wb.active
ws.title = "Summary"

for r in dataframe_to_rows(df, index=False, header=True):
    ws.append(r)

for col in ws.columns:
    max_length = max(len(str(cell.value)) for cell in col)
    ws.column_dimensions[col[0].column_letter].width = max_length + 2

wb.save(os.path.join(out_dir, "summary_table_formatted.xlsx"))

print("\nExport complete.")