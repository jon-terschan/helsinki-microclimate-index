import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import plotting_extent
import contextily as ctx
from scipy.ndimage import gaussian_filter, sobel
import xarray as xr
from rasterio.warp import reproject, Resampling
from matplotlib.colors import TwoSlopeNorm

# =============================
# USER SETTINGS
# =============================
baseline_dir = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\baseline\15cm_July_allday"
era5_path    = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\baseline\climatology\era5land_climatology_JULY_10_18.nc"

tree_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\TREE_FRAC_10m.tif"
nwn_path  = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\NWN_FRAC_10m.tif"

target_local_hour = 12
utc_offset = 3

apply_smoothing = False
sigma = 0.5
show_edges = False

export_figures = True
out_base = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\drafts"

# =============================
# TIME
# =============================
utc_hour = (target_local_hour - utc_offset) % 24
hh_str = f"{utc_hour:02d}00"

# =============================
# LOAD TEMPERATURE
# =============================
temp_path = os.path.join(baseline_dir, f"pred_20000715_{hh_str}.tif")

with rasterio.open(temp_path) as src:
    temp = src.read(1).astype("float32")
    nodata = src.nodata
    extent = plotting_extent(src)
    crs = src.crs
    transform = src.transform

if nodata is not None:
    temp[temp == nodata] = np.nan

mask_valid = ~np.isnan(temp)

# =============================
# VEGETATION
# =============================
with rasterio.open(tree_path) as src:
    tree = src.read(1).astype("float32")

with rasterio.open(nwn_path) as src:
    nwn = src.read(1).astype("float32")

veg = np.clip(tree + nwn, 0, 1)

# =============================
# SMOOTHING
# =============================
if apply_smoothing:
    temp_filled = np.where(mask_valid, temp, 0)
    smooth_temp = gaussian_filter(temp_filled, sigma=sigma)
    smooth_mask = gaussian_filter(mask_valid.astype(float), sigma=sigma)

    temp_vis = np.divide(
        smooth_temp,
        smooth_mask,
        out=np.full_like(smooth_temp, np.nan),
        where=smooth_mask > 0
    )
else:
    temp_vis = temp.copy()

temp_vis[~mask_valid] = np.nan

# =============================
# ERA5
# =============================
ds = xr.open_dataset(era5_path)

era5_da = ds["t2m"] if "t2m" in ds else list(ds.data_vars.values())[0]

time_dim = [d for d in era5_da.dims if "time" in d.lower()][0]
era5_da = era5_da.sel({time_dim: era5_da[time_dim].dt.hour == utc_hour})
era5_da = era5_da.mean(dim=time_dim)

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

# =============================
# REPROJECT
# =============================
era5_reproj = np.full_like(temp_vis, np.nan)

reproject(
    source=era5,
    destination=era5_reproj,
    src_transform=src_transform,
    src_crs="EPSG:4326",
    dst_transform=transform,
    dst_crs=crs,
    resampling=Resampling.bilinear,
    src_nodata=np.nan,
    dst_nodata=np.nan
)

era5_reproj[~mask_valid] = np.nan

# =============================
# DELTA
# =============================
delta = temp_vis - era5_reproj

# =============================
# COLOR
# =============================
valid = temp_vis[~np.isnan(temp_vis)]
vmin = np.percentile(valid, 7)
vmax = np.percentile(valid, 93)

norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap_left = plt.get_cmap("cividis")

valid_d = delta[~np.isnan(delta)]
dmax = np.percentile(np.abs(valid_d), 90)

norm_d = TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)
cmap_right = plt.get_cmap("RdBu_r")

# =============================
# RGBA
# =============================
alpha = 0.08 + 0.92 * (veg ** 3)

rgba = cmap_left(norm(temp_vis))
rgba[..., -1] = alpha
rgba[np.isnan(temp_vis), -1] = 0

rgba_d = cmap_right(norm_d(delta))
rgba_d[..., -1] = alpha
rgba_d[np.isnan(delta), -1] = 0

# =============================
# EDGES
# =============================
edges = None
if show_edges:
    edges_input = gaussian_filter(temp_vis, sigma=0.7) if not apply_smoothing else temp_vis
    edges = np.hypot(
        sobel(edges_input, axis=0),
        sobel(edges_input, axis=1)
    )
    if np.nanmax(edges) > 0:
        edges /= np.nanmax(edges)

# =============================
# DRAW FUNCTION
# =============================
def draw_map(ax, img, with_edges=False):
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

    ax.imshow(img, extent=extent, interpolation="bicubic", zorder=10)

    if with_edges and edges is not None:
        ax.imshow(edges, extent=extent, cmap="gray", alpha=0.05, zorder=15)

    ax.axis("off")

# =============================
# HORIZONTAL FIGURE
# =============================
fig_h, axes_h = plt.subplots(1, 2, figsize=(14, 7))
fig_h.patch.set_facecolor("#f7f7f7")

draw_map(axes_h[0], rgba, with_edges=show_edges)
draw_map(axes_h[1], rgba_d)

# titles
axes_h[0].text(0, 1.08, "An average Summer day in Helsinki's urban green areas:",
               transform=axes_h[0].transAxes, fontsize=12, fontweight="semibold")

axes_h[0].text(0, 1.01,
               f"a) 15 cm temperature predicted from ERA5-Land July climatology ({target_local_hour}:00)",
               transform=axes_h[0].transAxes, fontsize=9, alpha=0.85)

axes_h[1].text(0, 1.08, "Deviation from regional climate:",
               transform=axes_h[1].transAxes, fontsize=12, fontweight="semibold")

axes_h[1].text(0, 1.01,
               f"b) difference between predicted 15 cm and ERA5-Land 2 m air temperature ({target_local_hour}:00)",
               transform=axes_h[1].transAxes, fontsize=9, alpha=0.85)

# colorbars
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_left)
cbar1 = fig_h.colorbar(sm, ax=axes_h[0], orientation="horizontal", fraction=0.05, pad=0.05)
cbar1.set_label("Temperature (°C)")

sm_d = plt.cm.ScalarMappable(norm=norm_d, cmap=cmap_right)
cbar2 = fig_h.colorbar(sm_d, ax=axes_h[1], orientation="horizontal", fraction=0.05, pad=0.05)
cbar2.set_label("Offset (°C)")

cbar2.ax.text(0, -1.5, "Cooler than macroclimate", transform=cbar2.ax.transAxes)
cbar2.ax.text(1, -1.5, "Warmer than macroclimate", transform=cbar2.ax.transAxes, ha="right")

plt.tight_layout()

if export_figures:
    fig_h.savefig(os.path.join(out_base, f"baseline_horizontal_{target_local_hour}.png"), dpi=300, bbox_inches="tight")
    fig_h.savefig(os.path.join(out_base, f"baseline_horizontal_{target_local_hour}.pdf"), bbox_inches="tight")

# =============================
# VERTICAL FIGURE
# =============================
fig_v, axes_v = plt.subplots(2, 1, figsize=(8, 14))
fig_v.patch.set_facecolor("#f7f7f7")

draw_map(axes_v[0], rgba, with_edges=show_edges)
draw_map(axes_v[1], rgba_d)

# titles (same style)
axes_v[0].text(0, 1.08, "An average Summer day in Helsinki's urban green areas:",
               transform=axes_v[0].transAxes, fontsize=12, fontweight="semibold")

axes_v[0].text(0, 1.01,
               f"a) 15 cm temperature predicted from ERA5-Land July climatology ({target_local_hour}:00)",
               transform=axes_v[0].transAxes, fontsize=9, alpha=0.85)

axes_v[1].text(0, 1.08, "Deviation from regional climate:",
               transform=axes_v[1].transAxes, fontsize=12, fontweight="semibold")

axes_v[1].text(0, 1.01,
               f"b) difference between predicted 15 cm and ERA5-Land 2 m air temperature ({target_local_hour}:00)",
               transform=axes_v[1].transAxes, fontsize=9, alpha=0.85)

# colorbars
cbar1_v = fig_v.colorbar(sm, ax=axes_v[0], orientation="horizontal", fraction=0.05, pad=0.05)
cbar1_v.set_label("Temperature (°C)")

cbar2_v = fig_v.colorbar(sm_d, ax=axes_v[1], orientation="horizontal", fraction=0.05, pad=0.05)
cbar2_v.set_label("Offset (°C)")

cbar2_v.ax.text(0, -1.5, "Cooler than macroclimate", transform=cbar2_v.ax.transAxes)
cbar2_v.ax.text(1, -1.5, "Warmer than macroclimate", transform=cbar2_v.ax.transAxes, ha="right")

plt.tight_layout()

if export_figures:
    fig_v.savefig(os.path.join(out_base, f"baseline_vertical_{target_local_hour}.png"), dpi=400, bbox_inches="tight")
    fig_v.savefig(os.path.join(out_base, f"baseline_vertical_{target_local_hour}.pdf"), bbox_inches="tight")

plt.show()