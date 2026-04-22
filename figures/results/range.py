"""
Script: Daily Temperature Range Analysis (Mean vs P90)

Description
-----------
This script computes and visualizes intra-day temperature variability (10:00–18:00 local time)
for two baseline scenarios:
    1) Mean climatological conditions
    2) Hot conditions (P90)

Outputs:
    - Spatial maps of:
        • Mean daily temperature range
        • P90 daily temperature range
        • Difference (P90 − Mean)
    - Numerical summaries (overall + area-weighted + dominant class)

Key concepts:
    - Range = max(T) − min(T) over daytime hours
    - Area-weighted stats use fractional land cover
    - Dominant class stats use argmax classification
"""

# =============================
# IMPORTS
# =============================
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import plotting_extent
import contextily as ctx
from matplotlib.colors import Normalize, TwoSlopeNorm

# =============================
# USER SETTINGS
# =============================
baseline_mean_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\baseline\15cm_July_allday"
baseline_p90_dir  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\baseline\15cm_July_allday_p90"

tree_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\TREE_FRAC_10m.tif"
nwn_path  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\NWN_FRAC_10m.tif"

utc_offset = 3
local_hours = list(range(10, 19))  # 10–18 local

export_figures = True
out_base = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\drafts"

# =============================
# LOAD LAND COVER
# =============================
with rasterio.open(tree_path) as src:
    tree = np.clip(src.read(1).astype("float32"), 0, 1)

with rasterio.open(nwn_path) as src:
    nwn = np.clip(src.read(1).astype("float32"), 0, 1)

urban = np.clip(1 - (tree + nwn), 0, 1)
veg   = np.clip(tree + nwn, 0, 1)

# transparency (same visual logic as before)
alpha = 0.08 + 0.92 * (veg ** 3)

# =============================
# LOAD STACK FUNCTION
# =============================
def load_stack(base_dir):
    temps = []

    for h in local_hours:
        utc_hour = (h - utc_offset) % 24
        hh = f"{utc_hour:02d}00"
        path = os.path.join(base_dir, f"pred_20000715_{hh}.tif")

        with rasterio.open(path) as src:
            t = src.read(1).astype("float32")
            if src.nodata is not None:
                t[t == src.nodata] = np.nan

            if 'extent' not in globals():
                global extent, crs
                extent = plotting_extent(src)
                crs = src.crs

        temps.append(t)

    return np.stack(temps)

# =============================
# LOAD DATA
# =============================
temps_mean = load_stack(baseline_mean_dir)
temps_p90  = load_stack(baseline_p90_dir)

# =============================
# COMPUTE DAILY RANGE
# =============================
range_mean = np.nanmax(temps_mean, axis=0) - np.nanmin(temps_mean, axis=0)
range_p90  = np.nanmax(temps_p90,  axis=0) - np.nanmin(temps_p90,  axis=0)
range_diff = range_p90 - range_mean

# =============================
# STATISTICS FUNCTIONS
# =============================
def mean_std(arr):
    return np.nanmean(arr), np.nanstd(arr)

def weighted_stats(values, weights):
    mask = (~np.isnan(values)) & (~np.isnan(weights))
    v, w = values[mask], weights[mask]

    if np.sum(w) == 0:
        return np.nan, np.nan

    mean = np.sum(w * v) / np.sum(w)
    var  = np.sum(w * (v - mean)**2) / np.sum(w)
    return mean, np.sqrt(var)

def masked_stats(arr, mask):
    vals = arr[mask]
    return np.nanmean(vals), np.nanstd(vals)

# =============================
# COMPUTE + PRINT STATS
# =============================
def compute_stats(arr, label):
    mask_valid = ~np.isnan(arr)

    stack = np.stack([tree, nwn, urban], axis=0)
    dominant = np.argmax(stack, axis=0)

    tree_dom  = (dominant == 0) & mask_valid
    nwn_dom   = (dominant == 1) & mask_valid
    urban_dom = (dominant == 2) & mask_valid

    mean_all, std_all = mean_std(arr)

    mean_tree_w, std_tree_w   = weighted_stats(arr, tree)
    mean_nwn_w,  std_nwn_w    = weighted_stats(arr, nwn)
    mean_urban_w, std_urban_w = weighted_stats(arr, urban)

    mean_tree_d, std_tree_d   = masked_stats(arr, tree_dom)
    mean_nwn_d,  std_nwn_d    = masked_stats(arr, nwn_dom)
    mean_urban_d, std_urban_d = masked_stats(arr, urban_dom)

    print(f"\n=== {label} ===")
    print(f"Overall: {mean_all:.2f} ± {std_all:.2f} °C\n")

    print("Area-weighted:")
    print(f"Tree:  {mean_tree_w:.2f} ± {std_tree_w:.2f}")
    print(f"NWN:   {mean_nwn_w:.2f} ± {std_nwn_w:.2f}")
    print(f"Urban: {mean_urban_w:.2f} ± {std_urban_w:.2f}\n")

    print("Dominant class:")
    print(f"Tree:  {mean_tree_d:.2f} ± {std_tree_d:.2f}")
    print(f"NWN:   {mean_nwn_d:.2f} ± {std_nwn_d:.2f}")
    print(f"Urban: {mean_urban_d:.2f} ± {std_urban_d:.2f}")

# =============================
# PRINT RESULTS
# =============================
compute_stats(range_mean, "MEAN DAILY RANGE")
compute_stats(range_p90,  "P90 DAILY RANGE")
compute_stats(range_diff, "P90 − MEAN RANGE DIFFERENCE")

# =============================
# MAP DRAWING FUNCTION
# =============================
def draw_map(arr, norm, cmap, title, filename, cbar_label):

    rgba = cmap(norm(arr))
    rgba[..., -1] = alpha
    rgba[np.isnan(arr), -1] = 0

    fig, ax = plt.subplots(figsize=(8, 10))
    fig.patch.set_facecolor("#f7f7f7")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

    ctx.add_basemap(
        ax,
        crs=crs,
        source=ctx.providers.CartoDB.PositronNoLabels,
        alpha=0.35,
        zorder=0
    )

    ax.imshow(rgba, extent=extent, interpolation="bicubic", zorder=10)
    ax.axis("off")

    ax.text(0, 1.05, title, transform=ax.transAxes, fontsize=11)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.05)
    cbar.set_label(cbar_label)

    plt.tight_layout()

    if export_figures:
        fig.savefig(os.path.join(out_base, filename + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(out_base, filename + ".pdf"), bbox_inches="tight")

    plt.close(fig)

# =============================
# COLOR NORMALIZATION
# =============================
combined = np.concatenate([
    range_mean[~np.isnan(range_mean)],
    range_p90[~np.isnan(range_p90)]
])

norm_range = Normalize(
    vmin=np.percentile(combined, 7),
    vmax=np.percentile(combined, 93)
)

valid_diff = range_diff[~np.isnan(range_diff)]
dmax = np.percentile(np.abs(valid_diff), 90)

norm_diff = TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)

# =============================
# EXPORT MAPS (SEPARATE FILES)
# =============================
draw_map(
    range_mean,
    norm_range,
    plt.get_cmap("magma"),
    "Mean daily temperature range (10–18 local)",
    "range_mean",
    "Temperature range (°C)"
)

draw_map(
    range_p90,
    norm_range,
    plt.get_cmap("magma"),
    "P90 daily temperature range (10–18 local)",
    "range_p90",
    "Temperature range (°C)"
)

draw_map(
    range_diff,
    norm_diff,
    plt.get_cmap("RdBu_r"),
    "Difference in daily range (P90 − mean)",
    "range_diff",
    "Δ temperature range (°C)"
)