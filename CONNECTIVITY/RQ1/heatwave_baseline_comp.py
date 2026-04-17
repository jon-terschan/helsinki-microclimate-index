import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from matplotlib.colors import Normalize

# ----------------------------
# PATHS
# ----------------------------
pred_base = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions"
clim_base = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\baseline\15cm_July_allday"
out_base  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\anomalies"

tree_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\TREE_FRAC_10m.tif"
nwn_path  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\NWN_FRAC_10m.tif"

events = {
    2010: "20100728",
    2018: "20180717",
    2021: "20210714",
}

target_local_hours = [12, 15, 17]
utc_offset = 3
sigma = 0.6

os.makedirs(out_base, exist_ok=True)

# ----------------------------
# LOAD VEGETATION FRACTION
# ----------------------------
with rasterio.open(tree_path) as src:
    tree = src.read(1).astype("float32")

with rasterio.open(nwn_path) as src:
    nwn = src.read(1).astype("float32")

veg = np.clip(tree + nwn, 0, 1)

alpha_base = veg ** 3
alpha_base = 0.05 + 0.95 * alpha_base

# ----------------------------
# FIRST PASS
# ----------------------------
anoms = {}
all_vals = []

for year, date_str in events.items():
    event_folder = os.path.join(pred_base, str(year), date_str)
    out_dir = os.path.join(out_base, str(year))
    os.makedirs(out_dir, exist_ok=True)

    for h_local in target_local_hours:
        utc_hour = (h_local - utc_offset) % 24
        hh_str = f"{utc_hour:02d}00"

        hw_path = os.path.join(event_folder, f"pred_{date_str}_{hh_str}.tif")
        clim_path = os.path.join(clim_base, f"pred_20000715_{hh_str}.tif")

        if not (os.path.exists(hw_path) and os.path.exists(clim_path)):
            print("Missing:", year, h_local)
            anoms[(year, h_local)] = None
            continue

        with rasterio.open(hw_path) as src:
            hw = src.read(1).astype("float32")
            profile = src.profile
            extent = (
                src.bounds.left,
                src.bounds.right,
                src.bounds.bottom,
                src.bounds.top,
            )

        with rasterio.open(clim_path) as src:
            clim = src.read(1).astype("float32")

        # ----------------------------
        # RAW ANOMALY
        # ----------------------------
        anom = hw - clim

        # ----------------------------
        # SAVE RAW ANOMALY (ONLY IF NOT EXISTS)
        # ----------------------------
        out_path = os.path.join(
            out_dir,
            f"anom_{year}_{date_str}_{hh_str}.tif"
        )

        if not os.path.exists(out_path):
            profile.update(
                dtype="float32",
                count=1,
                compress="lzw",
                nodata=np.nan
            )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(anom.astype("float32"), 1)

            print("Saved RAW anomaly:", out_path)
        else:
            print("Exists, skipping:", out_path)

        # ----------------------------
        # SMOOTHING (VISUAL ONLY)
        # ----------------------------
        mask = ~np.isnan(anom)
        filled = np.where(mask, anom, 0)

        smooth = gaussian_filter(filled, sigma=sigma)
        smooth_mask = gaussian_filter(mask.astype(float), sigma=sigma)

        anom_smooth = np.divide(
            smooth,
            smooth_mask,
            out=np.full_like(smooth, np.nan),
            where=smooth_mask > 0
        )

        # ----------------------------
        # STRUCTURE (edges)
        # ----------------------------
        edges = np.hypot(
            sobel(anom_smooth, axis=0),
            sobel(anom_smooth, axis=1)
        )
        if np.nanmax(edges) > 0:
            edges = edges / np.nanmax(edges)

        anoms[(year, h_local)] = (anom_smooth, edges)
        all_vals.append(anom_smooth[~np.isnan(anom_smooth)])

# ----------------------------
# GLOBAL COLOR SCALE
# ----------------------------
if len(all_vals) == 0:
    raise RuntimeError("No valid data found. Check input paths.")

all_vals = np.concatenate(all_vals)

vmin = np.percentile(all_vals, 2)
vmax = np.percentile(all_vals, 98)

norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.inferno

print("Color scale:", vmin, vmax)

# ----------------------------
# PLOT
# ----------------------------
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
fig.patch.set_facecolor("#f7f7f7")

years = list(events.keys())

for i, year in enumerate(years):
    for j, h_local in enumerate(target_local_hours):

        ax = axes[i, j]
        entry = anoms.get((year, h_local))

        if entry is None:
            ax.axis("off")
            continue

        anom, edges = entry

        rgba = cmap(norm(anom))
        rgba[..., -1] = alpha_base
        rgba[np.isnan(anom), -1] = 0

        ax.imshow(rgba, extent=extent, interpolation="bilinear", zorder=10)
        ax.imshow(edges, extent=extent, cmap="gray", alpha=0.08, zorder=15)

        ax.axis("off")

        if i == 0:
            ax.set_title(f"{h_local}:00", fontsize=11)

        if j == 0:
            ax.text(
                -0.05, 0.5, str(year),
                transform=ax.transAxes,
                rotation=90,
                va="center", ha="right",
                fontsize=11, fontweight="semibold"
            )

# ----------------------------
# COLORBAR
# ----------------------------
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

cbar = fig.colorbar(
    sm,
    ax=axes.ravel().tolist(),
    orientation="vertical",
    fraction=0.035,
    pad=0.02
)

cbar.set_label("Temperature anomaly (°C)", fontsize=10)
cbar.ax.tick_params(labelsize=8)

cbar.outline.set_visible(False)
for spine in cbar.ax.spines.values():
    spine.set_visible(False)

# ----------------------------
# LAYOUT
# ----------------------------
plt.subplots_adjust(
    left=0.08,
    right=0.86,
    top=0.92,
    bottom=0.08,
    wspace=0.02,
    hspace=0.05
)

plt.show()

# spatial variability
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from matplotlib.colors import Normalize

# ----------------------------
# PATHS
# ----------------------------
pred_base = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions"
clim_base = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\baseline\15cm_July_allday"

tree_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\TREE_FRAC_10m.tif"
nwn_path  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\NWN_FRAC_10m.tif"

events = {
    2010: "20100728",
    2018: "20180717",
    2021: "20210714",
}

h_local = 12
utc_offset = 3
sigma = 0.6

# ----------------------------
# LOAD VEGETATION FRACTION
# ----------------------------
with rasterio.open(tree_path) as src:
    tree = src.read(1).astype("float32")

with rasterio.open(nwn_path) as src:
    nwn = src.read(1).astype("float32")

veg = np.clip(tree + nwn, 0, 1)

alpha_base = veg ** 3
alpha_base = 0.05 + 0.95 * alpha_base

# ----------------------------
# LOAD + PROCESS
# ----------------------------
anoms = {}
all_vals = []

utc_hour = (h_local - utc_offset) % 24
hh_str = f"{utc_hour:02d}00"

for year, date_str in events.items():
    event_folder = os.path.join(pred_base, str(year), date_str)

    hw_path = os.path.join(event_folder, f"pred_{date_str}_{hh_str}.tif")
    clim_path = os.path.join(clim_base, f"pred_20000715_{hh_str}.tif")

    if not (os.path.exists(hw_path) and os.path.exists(clim_path)):
        print("Missing:", year)
        anoms[year] = None
        continue

    with rasterio.open(hw_path) as src:
        hw = src.read(1).astype("float32")
        extent = (
            src.bounds.left,
            src.bounds.right,
            src.bounds.bottom,
            src.bounds.top,
        )

    with rasterio.open(clim_path) as src:
        clim = src.read(1).astype("float32")

    # anomaly
    anom = hw - clim

    # ----------------------------
    # SMOOTHING (visual only)
    # ----------------------------
    mask = ~np.isnan(anom)
    filled = np.where(mask, anom, 0)

    smooth = gaussian_filter(filled, sigma=sigma)
    smooth_mask = gaussian_filter(mask.astype(float), sigma=sigma)

    anom_smooth = np.divide(
        smooth,
        smooth_mask,
        out=np.full_like(smooth, np.nan),
        where=smooth_mask > 0
    )

    # edges
    edges = np.hypot(
        sobel(anom_smooth, axis=0),
        sobel(anom_smooth, axis=1)
    )
    if np.nanmax(edges) > 0:
        edges = edges / np.nanmax(edges)

    anoms[year] = (anom_smooth, edges)

    all_vals.append(anom_smooth[~np.isnan(anom_smooth)])

# ----------------------------
# COLOR SCALE
# ----------------------------
all_vals = np.concatenate(all_vals)

vmin = np.percentile(all_vals, 2)
vmax = np.percentile(all_vals, 98)

norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.inferno

print("Color scale:", vmin, vmax)

# ----------------------------
# PLOT (1 x 3)
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.patch.set_facecolor("#f7f7f7")

for i, (year, data) in enumerate(anoms.items()):
    ax = axes[i]

    if data is None:
        ax.axis("off")
        continue

    anom, edges = data

    rgba = cmap(norm(anom))
    rgba[..., -1] = alpha_base
    rgba[np.isnan(anom), -1] = 0

    ax.imshow(rgba, extent=extent, interpolation="bilinear", zorder=10)
    ax.imshow(edges, extent=extent, cmap="gray", alpha=0.08, zorder=15)

    ax.axis("off")
    ax.set_title(f"{year} (12:00)", fontsize=11)

# ----------------------------
# COLORBAR
# ----------------------------
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

cbar = fig.colorbar(
    sm,
    ax=axes,
    orientation="vertical",
    fraction=0.04,
    pad=0.02
)

cbar.set_label("Temperature anomaly (°C)", fontsize=10)
cbar.ax.tick_params(labelsize=8)

cbar.outline.set_visible(False)
for spine in cbar.ax.spines.values():
    spine.set_visible(False)

# ----------------------------
# LAYOUT
# ----------------------------
plt.subplots_adjust(
    left=0.05,
    right=0.88,
    top=0.90,
    bottom=0.05,
    wspace=0.02
)

plt.show()


# els

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from matplotlib.colors import TwoSlopeNorm

# ----------------------------
# PATHS
# ----------------------------
pred_base = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions"

tree_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\TREE_FRAC_10m.tif"
nwn_path  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\NWN_FRAC_10m.tif"

events = {
    2010: "20100728",
    2018: "20180717",
    2021: "20210714",
}

h_local = 12
utc_offset = 3
sigma = 0.6

# ----------------------------
# LOAD VEGETATION FRACTION
# ----------------------------
with rasterio.open(tree_path) as src:
    tree = src.read(1).astype("float32")

with rasterio.open(nwn_path) as src:
    nwn = src.read(1).astype("float32")

veg = np.clip(tree + nwn, 0, 1)

alpha_base = veg ** 3
alpha_base = 0.05 + 0.95 * alpha_base

mask_veg = veg > 0.3  # focus reference on vegetated areas

# ----------------------------
# LOAD + PROCESS
# ----------------------------
anoms = {}
all_vals = []

utc_hour = (h_local - utc_offset) % 24
hh_str = f"{utc_hour:02d}00"

for year, date_str in events.items():
    event_folder = os.path.join(pred_base, str(year), date_str)

    hw_path = os.path.join(event_folder, f"pred_{date_str}_{hh_str}.tif")

    if not os.path.exists(hw_path):
        print("Missing:", year)
        anoms[year] = None
        continue

    with rasterio.open(hw_path) as src:
        hw = src.read(1).astype("float32")
        extent = (
            src.bounds.left,
            src.bounds.right,
            src.bounds.bottom,
            src.bounds.top,
        )

    # ----------------------------
    # RELATIVE ANOMALY
    # ----------------------------
    mean_hw = np.nanmean(hw[mask_veg])
    anom_rel = hw - mean_hw

    # ----------------------------
    # SMOOTHING (visual only)
    # ----------------------------
    mask = ~np.isnan(anom_rel)
    filled = np.where(mask, anom_rel, 0)

    smooth = gaussian_filter(filled, sigma=sigma)
    smooth_mask = gaussian_filter(mask.astype(float), sigma=sigma)

    anom_smooth = np.divide(
        smooth,
        smooth_mask,
        out=np.full_like(smooth, np.nan),
        where=smooth_mask > 0
    )

    # ----------------------------
    # STRUCTURE
    # ----------------------------
    edges = np.hypot(
        sobel(anom_smooth, axis=0),
        sobel(anom_smooth, axis=1)
    )
    if np.nanmax(edges) > 0:
        edges = edges / np.nanmax(edges)

    anoms[year] = (anom_smooth, edges)
    all_vals.append(anom_smooth[~np.isnan(anom_smooth)])

# ----------------------------
# COLOR SCALE (CENTERED)
# ----------------------------
all_vals = np.concatenate(all_vals)

dmax = np.percentile(np.abs(all_vals), 95)

norm = TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)
cmap = plt.cm.RdBu_r

print("Relative range:", -dmax, dmax)

# ----------------------------
# PLOT
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.patch.set_facecolor("#f7f7f7")

for i, (year, data) in enumerate(anoms.items()):
    ax = axes[i]

    if data is None:
        ax.axis("off")
        continue

    anom, edges = data

    rgba = cmap(norm(anom))
    rgba[..., -1] = alpha_base
    rgba[np.isnan(anom), -1] = 0

    ax.imshow(rgba, extent=extent, interpolation="bilinear", zorder=10)
    ax.imshow(edges, extent=extent, cmap="gray", alpha=0.08, zorder=15)

    ax.axis("off")
    ax.set_title(f"{year} (12:00)", fontsize=11)

# ----------------------------
# COLORBAR
# ----------------------------
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

cbar = fig.colorbar(
    sm,
    ax=axes,
    orientation="vertical",
    fraction=0.04,
    pad=0.02
)

cbar.set_label("Relative temperature (°C)", fontsize=10)
cbar.ax.tick_params(labelsize=8)

cbar.outline.set_visible(False)
for spine in cbar.ax.spines.values():
    spine.set_visible(False)

# ----------------------------
# LAYOUT
# ----------------------------
plt.subplots_adjust(
    left=0.05,
    right=0.88,
    top=0.90,
    bottom=0.05,
    wspace=0.02
)

plt.show()



