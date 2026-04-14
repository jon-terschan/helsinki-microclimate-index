### changed cmap on left figure
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import plotting_extent
import contextily as ctx
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.cm as cm
import xarray as xr
from rasterio.warp import reproject, Resampling
from matplotlib.colors import TwoSlopeNorm

# -----------------------------
# SETTINGS
# -----------------------------
temp_path  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\baseline\baseline_pred.tif"
tree_path  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\TREE_FRAC_10m.tif"
nwn_path   = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\NWN_FRAC_10m.tif"
era5_path  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\climatology\era5land_jja_tmax_climatology_interpol.nc"

sigma = 0.5

# -----------------------------
# LOAD TEMPERATURE
# -----------------------------
with rasterio.open(temp_path) as src:
    temp = src.read(1).astype("float32")
    nodata = src.nodata
    extent = plotting_extent(src)
    crs = src.crs
    transform = src.transform

if nodata is not None:
    temp[temp == nodata] = np.nan

# -----------------------------
# LOAD FRACTIONS
# -----------------------------
with rasterio.open(tree_path) as src:
    tree = src.read(1).astype("float32")

with rasterio.open(nwn_path) as src:
    nwn = src.read(1).astype("float32")

veg = np.clip(tree + nwn, 0, 1)

# -----------------------------
# SMOOTH TEMPERATURE
# -----------------------------
mask_valid = ~np.isnan(temp)
temp_filled = np.where(mask_valid, temp, 0)

smooth_temp = gaussian_filter(temp_filled, sigma=sigma)
smooth_mask = gaussian_filter(mask_valid.astype(float), sigma=sigma)

temp_smooth = np.divide(
    smooth_temp,
    smooth_mask,
    out=np.full_like(smooth_temp, np.nan, dtype=np.float32),
    where=smooth_mask > 0
)

temp_smooth[~mask_valid] = np.nan

# -----------------------------
# LOAD + REPROJECT ERA5
# -----------------------------
ds = xr.open_dataset(era5_path)

if "t2m" in ds.data_vars:
    era5_da = ds["t2m"].squeeze()
else:
    data_vars = [v for v in ds.data_vars if ds[v].ndim >= 2]
    era5_da = ds[data_vars[0]].squeeze()

y_name, x_name = era5_da.dims[-2], era5_da.dims[-1]
y = np.asarray(era5_da[y_name].values)
x = np.asarray(era5_da[x_name].values)
era5 = np.asarray(era5_da.values).astype("float32")

if np.nanmean(era5) > 100:
    era5 -= 273.15

if y[0] < y[-1]:
    era5 = np.flipud(era5)
    y = y[::-1]

dx = np.median(np.diff(x))
dy = np.median(np.diff(y))

src_transform = rasterio.transform.from_origin(
    float(x.min() - abs(dx)/2),
    float(y.max() + abs(dy)/2),
    float(abs(dx)),
    float(abs(dy))
)

src_crs = "EPSG:4326"

era5_reproj = np.full_like(temp_smooth, np.nan, dtype=np.float32)

reproject(
    source=era5,
    destination=era5_reproj,
    src_transform=src_transform,
    src_crs=src_crs,
    dst_transform=transform,
    dst_crs=crs,
    resampling=Resampling.bilinear,
    src_nodata=np.nan,
    dst_nodata=np.nan
)

era5_reproj[~mask_valid] = np.nan

# -----------------------------
# DELTA
# -----------------------------
delta = temp_smooth - era5_reproj

# -----------------------------
# MODEL COLOR SCALING (LEFT)
# -----------------------------
valid = temp_smooth[~np.isnan(temp_smooth)]
vmin = np.percentile(valid, 7)
vmax = np.percentile(valid, 93)

norm = plt.Normalize(vmin=vmin, vmax=vmax)

# >>> CHANGED HERE <<<
cmap_left = cm.get_cmap("cividis")
rgba = cmap_left(norm(temp_smooth))

# vegetation alpha
alpha = veg ** 3
alpha = 0.08 + 0.92 * alpha

rgba[..., -1] = alpha
rgba[np.isnan(temp_smooth), -1] = 0

# -----------------------------
# STRUCTURE OVERLAY
# -----------------------------
edges = np.hypot(
    sobel(temp_smooth, axis=0),
    sobel(temp_smooth, axis=1)
)
edges = edges / np.nanmax(edges)

# -----------------------------
# DELTA COLOR SCALING (RIGHT)
# -----------------------------
valid_d = delta[~np.isnan(delta)]
dmax = np.percentile(np.abs(valid_d), 90)

norm_d = TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)

cmap_right = cm.get_cmap("RdBu_r")
rgba_d = cmap_right(norm_d(delta))

rgba_d[..., -1] = alpha
rgba_d[np.isnan(delta), -1] = 0

# -----------------------------
# PLOT
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax in axes:
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

# LEFT
axes[0].imshow(rgba, extent=extent, interpolation="bicubic", zorder=10)
axes[0].imshow(edges, extent=extent, cmap="gray", alpha=0.10, zorder=15)
axes[0].axis("off")

# RIGHT
axes[1].imshow(rgba_d, extent=extent, interpolation="bicubic", zorder=10)
axes[1].axis("off")

# -----------------------------
# COLORBARS
# -----------------------------
# LEFT (UPDATED CMAP)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_left)

ticks = np.arange(np.floor(vmin*2)/2, np.ceil(vmax*2)/2 + 0.01, 0.5)

cbar1 = fig.colorbar(
    sm, ax=axes[0],
    orientation="horizontal",
    fraction=0.05, pad=0.05,
    ticks=ticks
)

cbar1.set_label("Temperature (°C)", fontweight="semibold", fontsize=10)
cbar1.ax.tick_params(labelsize=8, length=2)

# RIGHT
sm_d = plt.cm.ScalarMappable(norm=norm_d, cmap=cmap_right)

ticks_d = np.arange(-np.ceil(dmax*2)/2, np.ceil(dmax*2)/2 + 0.01, 0.5)
if len(ticks_d) > 11:
    ticks_d = np.arange(-np.ceil(dmax), np.ceil(dmax)+0.01, 1.0)

cbar2 = fig.colorbar(
    sm_d, ax=axes[1],
    orientation="horizontal",
    fraction=0.05, pad=0.05,
    ticks=ticks_d
)

cbar2.set_label("Offset (°C)", fontweight="semibold", fontsize=10)
cbar2.ax.tick_params(labelsize=8, length=2)

# interpretive labels
cbar2.ax.text(0.0, -1.5, "Cooler than macroclimate",
              transform=cbar2.ax.transAxes,
              ha="left", va="top", fontsize=9)

cbar2.ax.text(1.0, -1.5, "Warmer than macroclimate",
              transform=cbar2.ax.transAxes,
              ha="right", va="top", fontsize=9)

# remove outlines
cbar1.outline.set_visible(False)
cbar2.outline.set_visible(False)
for spine in cbar1.ax.spines.values():
    spine.set_visible(False)
for spine in cbar2.ax.spines.values():
    spine.set_visible(False)

# -----------------------------
# TITLES
# -----------------------------
axes[0].text(
    0.0, 1.08,
    "An average Summer day in Helsinki's urban green areas:",
    transform=axes[0].transAxes,
    ha="left", va="bottom",
    fontsize=12, fontweight="semibold"
)

axes[0].text(
    0.0, 1.01,
    "a) daily max. 15cm temperature predicted from \navg. ERA5-Land Tmax conditions (JJA, 1990–2020)",
    transform=axes[0].transAxes,
    ha="left", va="bottom",
    fontsize=9, alpha=0.85
)

axes[1].text(
    0.0, 1.08,
    "Deviation from regional climate:",
    transform=axes[1].transAxes,
    ha="left", va="bottom",
    fontsize=12, fontweight="semibold"
)

axes[1].text(
    0.0, 1.01,
    "b) difference between predicted 15cm and ERA5-Land climatology \n2m air temperature (JJA, 1990–2020)",
    transform=axes[1].transAxes,
    ha="left", va="bottom",
    fontsize=9, alpha=0.85
)
fig.patch.set_facecolor("#f7f7f7")
plt.tight_layout(rect=[0, 0, 1, 0.82])

# -----------------------------
# EXPORT
# -----------------------------
out_path_png = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\drafts\hel_t_baseline.png"
out_path_pdf = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\drafts\hel_t_baseline.pdf"

fig.savefig(
    out_path_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    out_path_pdf,
    bbox_inches="tight"
)
plt.show()



# -----------------------------
# VERTICAL VERSION (2x1, HIGH RES) — FIXED
# -----------------------------
fig_v, axes_v = plt.subplots(2, 1, figsize=(8, 14))

# match horizontal background
fig_v.patch.set_facecolor("#f7f7f7")

for ax in axes_v:
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

# -----------------------------
# MAPS
# -----------------------------
axes_v[0].imshow(rgba, extent=extent, interpolation="bicubic", zorder=10)
axes_v[0].imshow(edges, extent=extent, cmap="gray", alpha=0.10, zorder=15)
axes_v[0].axis("off")

axes_v[1].imshow(rgba_d, extent=extent, interpolation="bicubic", zorder=10)
axes_v[1].axis("off")

# -----------------------------
# COLORBARS (MATCH STYLE)
# -----------------------------
# LEFT
sm_v = plt.cm.ScalarMappable(norm=norm, cmap=cmap_left)

ticks_v = np.arange(np.floor(vmin*2)/2, np.ceil(vmax*2)/2 + 0.01, 0.5)

cbar1_v = fig_v.colorbar(
    sm_v, ax=axes_v[0],
    orientation="horizontal",
    fraction=0.05, pad=0.05,
    ticks=ticks_v
)

cbar1_v.set_label("Temperature (°C)", fontweight="semibold", fontsize=10)
cbar1_v.ax.tick_params(labelsize=8, length=2)
cbar1_v.outline.set_visible(False)
for spine in cbar1_v.ax.spines.values():
    spine.set_visible(False)

# RIGHT
sm_d_v = plt.cm.ScalarMappable(norm=norm_d, cmap=cmap_right)

ticks_d_v = np.arange(-np.ceil(dmax*2)/2, np.ceil(dmax*2)/2 + 0.01, 0.5)
if len(ticks_d_v) > 11:
    ticks_d_v = np.arange(-np.ceil(dmax), np.ceil(dmax)+0.01, 1.0)

cbar2_v = fig_v.colorbar(
    sm_d_v, ax=axes_v[1],
    orientation="horizontal",
    fraction=0.05, pad=0.05,
    ticks=ticks_d_v
)

cbar2_v.set_label("Offset (°C)", fontweight="semibold", fontsize=10)
cbar2_v.ax.tick_params(labelsize=8, length=2)
cbar2_v.outline.set_visible(False)

for spine in cbar2_v.ax.spines.values():
    spine.set_visible(False)

cbar2_v.ax.text(
    0.0, -1.5, "Cooler than macroclimate",
    transform=cbar2_v.ax.transAxes,
    ha="left", va="top", fontsize=9
)

cbar2_v.ax.text(
    1.0, -1.5, "Warmer than macroclimate",
    transform=cbar2_v.ax.transAxes,
    ha="right", va="top", fontsize=9
)

# -----------------------------
# TITLES (EXACT MATCH)
# -----------------------------
axes_v[0].text(
    0.0, 1.08,
    "An average Summer day in Helsinki's urban green areas:",
    transform=axes_v[0].transAxes,
    ha="left", va="bottom",
    fontsize=12, fontweight="semibold"
)

axes_v[0].text(
    0.0, 1.01,
    "a) daily max. 15cm temperature predicted from \navg. ERA5-Land Tmax conditions (JJA, 1990–2020)",
    transform=axes_v[0].transAxes,
    ha="left", va="bottom",
    fontsize=9, alpha=0.85
)

axes_v[1].text(
    0.0, 1.08,
    "Deviation from regional climate:",
    transform=axes_v[1].transAxes,
    ha="left", va="bottom",
    fontsize=12, fontweight="semibold"
)

axes_v[1].text(
    0.0, 1.01,
    "b) difference between predicted 15cm and ERA5-Land climatology \n2m air temperature (JJA, 1990–2020)",
    transform=axes_v[1].transAxes,
    ha="left", va="bottom",
    fontsize=9, alpha=0.85
)

plt.tight_layout(rect=[0, 0, 1, 0.82])

# -----------------------------
# EXPORT (HIGH RES)
# -----------------------------
out_path_png_v = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\drafts\hel_t_baseline_vertical.png"
out_path_pdf_v = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\drafts\hel_t_baseline_vertical.pdf"

fig_v.savefig(out_path_png_v, dpi=400, bbox_inches="tight")
fig_v.savefig(out_path_pdf_v, bbox_inches="tight")
