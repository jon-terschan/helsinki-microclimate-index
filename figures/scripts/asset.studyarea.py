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
from shapely.geometry import box


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

ocean_raster_path = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictorstack\OCEAN_FRAC_10m_Helsinki.tif"
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

# Transparent everywhere except ocean
ocean_rgba = np.zeros((ocean_mask.shape[0], ocean_mask.shape[1], 4), dtype=float)
ocean_rgba[ocean_mask, 0] = 0.88
ocean_rgba[ocean_mask, 1] = 0.88
ocean_rgba[ocean_mask, 2] = 0.88
ocean_rgba[ocean_mask, 3] = 1.0

ocean_extent = plotting_extent(ocean, ocean_transform)


# ------------------------------------------------------------
# CROP VECTOR LAYERS TO RASTER EXTENT
# ------------------------------------------------------------

left, right, bottom, top = ocean_extent

raster_box = gpd.GeoDataFrame(
    geometry=[box(left, bottom, right, top)],
    crs=target_crs,
)

peruspiiri_plot = gpd.clip(peruspiiri, raster_box)
helsinki_outline_plot = peruspiiri_plot.dissolve()
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
helsinki_outline_plot.plot(
    ax=ax,
    facecolor="white",
    edgecolor="black",
    linewidth=0.85,
    zorder=1,
)

# Ocean mask
ax.imshow(
    ocean_rgba,
    extent=ocean_extent,
    origin="upper",
    zorder=2,
)

# Internal peruspiiri boundaries
peruspiiri_plot.boundary.plot(
    ax=ax,
    color="black",
    linewidth=0.22,
    alpha=0.45,
    zorder=3,
)

# Outer outline on top
helsinki_outline_plot.boundary.plot(
    ax=ax,
    color="black",
    linewidth=1.00,
    zorder=4,
)

# Sensor locations
loggers_plot.plot(
    ax=ax,
    color="black",
    markersize=3.2,
    marker="o",
    linewidth=0,
    zorder=5,
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