# ============================================================
# Minimal monochrome sensor location map with ocean raster mask
# transparent background, no labels
# CRS: EPSG:3879 / ETRS-TM35FIN
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import rasterio
from rasterio.mask import mask
from rasterio.plot import plotting_extent
from shapely.geometry import Point, box


# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

target_crs = "EPSG:3879"

peruspiiri_path = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\figures\offset_figure\peruspiiri_WFS.gpkg"
)

logger_current_path = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\modeling\01_traindataprep\site_locations\helmostatus_11.25.gpkg"
)

logger_original_path = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\modeling\01_traindataprep\site_locations\helmostatus_original.gpkg"
)

botanical_sensors_path = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\VALIDATION\botanical_sensors.gpkg"
)

ocean_raster_path = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\OCEAN_FRAC_10m_Helsinki.tif"
)

water_raster_path = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\WATER_FRAC_10m_Helsinki.tif"
)

tree_raster_path = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\TREE_FRAC_10m.tif"
)

nwn_raster_path = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\NWN_FRAC_10m.tif"
)

out_png = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\figures\sensor_location_map_mono_oceanmask_transparent.png"
)

out_svg = out_png.with_suffix(".svg")


# ------------------------------------------------------------
# LOAD PERUSPIIRI POLYGONS
# ------------------------------------------------------------

peruspiiri = gpd.read_file(peruspiiri_path).to_crs(target_crs)


# ------------------------------------------------------------
# LOAD LOGGER LOCATIONS
# ------------------------------------------------------------

loggers_current = (
    gpd.read_file(logger_current_path)
    .to_crs(target_crs)
    .assign(sensor_id=lambda x: x["SERIAL"].astype(str))
    [["sensor_id", "geometry"]]
)

loggers_original = (
    gpd.read_file(logger_original_path)
    .to_crs(target_crs)
    .assign(sensor_id=lambda x: x["SERIAL"].astype(str))
    [["sensor_id", "geometry"]]
)

missing_from_current = ~loggers_original["sensor_id"].isin(
    loggers_current["sensor_id"]
)

loggers = gpd.GeoDataFrame(
    pd.concat(
        [
            loggers_current,
            loggers_original.loc[missing_from_current],
        ],
        ignore_index=True,
    ),
    geometry="geometry",
    crs=target_crs,
)

loggers = loggers[
    loggers["sensor_id"].notna()
    & loggers.geometry.notna()
    & ~loggers.geometry.is_empty
].copy()

loggers["geometry"] = loggers.geometry.centroid
loggers = loggers.drop_duplicates(subset="sensor_id", keep="first")

loggers = (
    gpd.sjoin(
        loggers,
        peruspiiri[["geometry"]],
        how="inner",
        predicate="within",
    )
    .drop(columns=["index_right"])
    .copy()
)


# ------------------------------------------------------------
# LOAD AND CLIP OCEAN RASTER TO HELSINKI POLYGON
# ------------------------------------------------------------

with rasterio.open(ocean_raster_path) as src:
    raster_crs = src.crs

    peruspiiri_for_mask = peruspiiri.to_crs(raster_crs)

    shapes = [
        geom
        for geom in peruspiiri_for_mask.geometry
        if geom is not None and not geom.is_empty
    ]

    ocean_clip, ocean_transform = mask(
        src,
        shapes,
        crop=True,
        filled=True,
        nodata=0,
    )

    ocean = ocean_clip[0]

ocean_mask = ocean > 0

# Soft, light-blue ocean tint that stays subtle
ocean_rgba = np.zeros((ocean_mask.shape[0], ocean_mask.shape[1], 4), dtype=float)
ocean_rgba[ocean_mask, 0] = 0.78
ocean_rgba[ocean_mask, 1] = 0.84
ocean_rgba[ocean_mask, 2] = 0.92
ocean_rgba[ocean_mask, 3] = 0.55

ocean_extent = plotting_extent(ocean, ocean_transform)


# ------------------------------------------------------------
# LOAD VEGETATION RASTERS AND BUILD GREEN SHADES
# ------------------------------------------------------------

def clip_raster_to_city(raster_path: Path, city_gdf: gpd.GeoDataFrame):
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        city_for_mask = city_gdf.to_crs(raster_crs)
        shapes = [
            geom
            for geom in city_for_mask.geometry
            if geom is not None and not geom.is_empty
        ]
        clip_arr, transform = mask(
            src,
            shapes,
            crop=True,
            filled=True,
            nodata=0,
        )
        return clip_arr[0], transform


def make_rgba_mask(mask_arr: np.ndarray, color: tuple[float, float, float], alpha_base: float):
    rgba = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=float)
    mask = mask_arr > 0.7
    values = mask_arr[mask]
    if values.size == 0:
        return rgba

    scaled = np.clip(values / 1.0, 0.0, 1.0)
    alpha = alpha_base + 0.18 * scaled
    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]
    rgba[mask, 3] = alpha
    return rgba


tree_arr, tree_transform = clip_raster_to_city(tree_raster_path, peruspiiri)
nwn_arr, nwn_transform = clip_raster_to_city(nwn_raster_path, peruspiiri)
water_arr, water_transform = clip_raster_to_city(water_raster_path, peruspiiri)

tree_rgba = make_rgba_mask(tree_arr, (0.20, 0.50, 0.24), 0.48)
nwn_rgba = make_rgba_mask(nwn_arr, (0.55, 0.74, 0.42), 0.30)
water_rgba = make_rgba_mask(water_arr, (0.54, 0.74, 0.86), 0.60)

tree_extent = plotting_extent(tree_arr, tree_transform)
nwn_extent = plotting_extent(nwn_arr, nwn_transform)
water_extent = plotting_extent(water_arr, water_transform)


# ------------------------------------------------------------
# CROP VECTOR LAYERS TO RASTER EXTENT
# ------------------------------------------------------------

left, right, bottom, top = ocean_extent

raster_box = gpd.GeoDataFrame(
    geometry=[box(left, bottom, right, top)],
    crs=target_crs,
)

peruspiiri_plot = gpd.clip(peruspiiri, raster_box)
loggers_plot = gpd.clip(loggers, raster_box)


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 6,
        "axes.linewidth": 0.4,
        "savefig.dpi": 600,
        "font.family": ["Tahoma", "DejaVu Sans", "sans-serif"],
    }
)

fig, ax = plt.subplots(figsize=(2.2, 2.2))

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor("none")

# Land / administrative shape
peruspiiri_plot.plot(
    ax=ax,
    facecolor="white",
    edgecolor=(0.15, 0.15, 0.15, 0.85),
    linewidth=0.30,
    zorder=1,
)

# Vegetation overlays (green shades)
ax.imshow(
    tree_rgba,
    extent=tree_extent,
    origin="upper",
    zorder=2,
)

ax.imshow(
    nwn_rgba,
    extent=nwn_extent,
    origin="upper",
    zorder=3,
)

# Water overlays (light blue shades)
ax.imshow(
    water_rgba,
    extent=water_extent,
    origin="upper",
    zorder=4,
)

# Ocean mask
ax.imshow(
    ocean_rgba,
    extent=ocean_extent,
    origin="upper",
    zorder=5,
)

# Sensor locations
loggers_plot.plot(
    ax=ax,
    color="black",
    markersize=3.2,
    marker="o",
    linewidth=0,
    zorder=8,
)

# Extent based on raster coverage
pad_x = (right - left) * 0.02
pad_y = (top - bottom) * 0.02

ax.set_xlim(left - pad_x, right + pad_x)
ax.set_ylim(bottom - pad_y, top + pad_y)

ax.set_axis_off()
ax.set_aspect("equal")

plt.tight_layout(pad=0)

fig.savefig(
    out_png,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.01,
    transparent=True,
)

fig.savefig(
    out_svg,
    bbox_inches="tight",
    pad_inches=0.01,
    transparent=True,
)

plt.close(fig)


# ------------------------------------------------------------
# LOG
# ------------------------------------------------------------

print(f"Saved PNG to: {out_png}")
print(f"Saved SVG to: {out_svg}")
print(f"Number of plotted loggers: {len(loggers_plot)}")
print(f"Ocean raster cells plotted: {int(ocean_mask.sum())}")