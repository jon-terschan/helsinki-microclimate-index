#!/usr/bin/env python3
"""
Haltiala case study: bivariate heatwave response with elevation, canopy
height and land-cover context, all cropped to one small AOI.

Panel A is the response: the same continuous, rank-based bivariate
classification used in the full-city map (see
f3_2_bivariate_heatwave_response_continuous_map.py in this folder), just
cropped to the AOI for display. Ranks are computed city-wide so the color
scale matches the full-city figure; only the crop is local. Behind the map,
a muted scatter of the same city-wide (THW, delta) distribution -- colored
with the same bivariate rank colors -- shows through wherever the map itself
has no data (buildings, roads, water), linking this local map back to the
citywide pattern it was drawn from.

Panels B-D are quiet context layers (elevation, canopy height, land cover),
kept visually subordinate to panel A: monochrome grey, monochrome grey, and
the same categorical land-cover colors used in f3_2_patch_scatter_by_lulc.py.
Land cover is smoothed the same way as the response map so the two maps read
as comparable in style.

Rasters are read with a buffer around the AOI (BUFFER_M) so bilinear
interpolation has real neighboring pixels at the edges; the displayed extent
and figure dimensions are still exactly the AOI, unaffected by the buffer.

To reuse this script for a different area, change AOI_PATH below. Every
raster used here (target domain, DTM, CHM, heatwave and baseline
predictions) shares one 10 m grid, so no reprojection is needed.

Output: one SVG per panel plus one small SVG for the bivariate key, written
to output/<aoi name>_case_study/.
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows as rio_windows
from matplotlib.patches import Patch, Rectangle
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from scipy.ndimage import uniform_filter

# =============================================================================
# CONFIG -- change AOI_PATH to regenerate this figure for a different area
# =============================================================================

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = SCRIPTS_ROOT / "DATA"
STYLE_DIR = SCRIPTS_ROOT / "figures" / "2_results" / "figures"
HERE = Path(__file__).resolve().parent

AOI_PATH = HERE / "haltiala_aoi_v2.gpkg"
OUTPUT_DIR = HERE / "output" / f"{AOI_PATH.stem}_case_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_RASTER = STYLE_DIR / "f4" / "rasters" / "p90_loss_target_domain_tree_veg_nwn_pm1p0deg.tif"
DTM_RASTER = DATA_DIR / "predictorstack" / "DTM_10m_Helsinki.tif"
CHM_RASTER = DATA_DIR / "predictorstack" / "CHM_10m_MAX.tif"

HEATWAVE_DIRS = {
    "2010": DATA_DIR / "predictions" / "2010" / "20100728",
    "2018": DATA_DIR / "predictions" / "2018" / "20180717",
    "2021": DATA_DIR / "predictions" / "2021" / "20210714",
}
BASELINE_DIR = DATA_DIR / "predictions" / "baseline" / "15cm_July_allday"
LOCAL_UTC_OFFSET_HOURS = 3
PEAK_LOCAL_HOUR = 13
TARGET_HOUR_UTC = (PEAK_LOCAL_HOUR - LOCAL_UTC_OFFSET_HOURS) % 24
MIN_VALID_TEMP_C = 15.0  # same floor as the full-city map

BUFFER_M = 500  # extra context read around the AOI; displayed extent is unchanged

LULC_PATHS = {
    "Trees 2\u201310 m": DATA_DIR / "LULC" / "lc_trees2-10.gpkg",
    "Trees 10\u201315 m": DATA_DIR / "LULC" / "lc_trees10-15.gpkg",
    "Trees 15\u201320 m": DATA_DIR / "LULC" / "lc_trees15-20.gpkg",
    "Trees >20 m": DATA_DIR / "LULC" / "lc_trees_o20.gpkg",
    "Fields": DATA_DIR / "LULC" / "lc_fields.gpkg",
    "Other vegetation": DATA_DIR / "LULC" / "lc_otherveg.gpkg",
}
LULC_ORDER = list(LULC_PATHS)

# Same categorical colors as f3_2_patch_scatter_by_lulc.py.
LULC_COLORS = {
    "Trees 2\u201310 m": "#1966D2",
    "Trees 10\u201315 m": "#1B5E20",
    "Trees 15\u201320 m": "#F5B041",
    "Trees >20 m": "#8B4513",
    "Fields": "#C62828",
    "Other vegetation": "#6A1B9A",
}

# Same four corner colors as the full-city bivariate map.
COLOR_COOL_STABLE = "#1b9e77"
COLOR_HOT_STABLE = "#fdae61"
COLOR_COOL_AMPLIFYING = "#7b3294"
COLOR_HOT_AMPLIFYING = "#d7191c"

MONOCHROME_CMAP = "Greys"
LEGEND_SHADE_COLOR = "white"
LEGEND_SHADE_ALPHA = 0.85

FONT = "DejaVu Sans"
TEXT_COLOR = "#222222"
MAP_HEIGHT_CM = 9.0
AXES_MARGIN = 0.015
DPI = 400

BACKGROUND_SCATTER_N = 6000
BACKGROUND_SCATTER_ALPHA = 0.15
BACKGROUND_SCATTER_SIZE = 6
RANDOM_SEED = 0

# =============================================================================
# STYLE
# =============================================================================

def apply_style() -> None:
    plt.rcParams.update({
        "font.family": FONT,
        "svg.fonttype": "none",
        "savefig.transparent": True,
    })

# =============================================================================
# AOI WINDOW (shared 10 m grid, no reprojection needed anywhere)
# =============================================================================

def _window_from_bounds(bounds: tuple[float, float, float, float], transform) -> Window:
    minx, miny, maxx, maxy = bounds
    win = from_bounds(minx, miny, maxx, maxy, transform=transform)
    win = win.round_lengths(op="ceil").round_offsets(op="floor")
    return Window(int(win.col_off), int(win.row_off), int(win.width), int(win.height))


def aoi_window(aoi_path: Path, reference_raster: Path, buffer_m: float = 0.0):
    """Return the buffered read window/profile, plus the AOI's own (unbuffered)
    display bounds. The read window feeds imshow extra context at the edges;
    the display bounds keep the shown extent and figure size unchanged."""
    with rasterio.open(reference_raster) as src:
        ref_crs, ref_transform = src.crs, src.transform

    aoi_gdf = gpd.read_file(aoi_path)
    if aoi_gdf.crs != ref_crs:
        aoi_gdf = aoi_gdf.to_crs(ref_crs)

    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    display_bounds = (minx, miny, maxx, maxy)
    read_bounds = (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)

    read_window = _window_from_bounds(read_bounds, ref_transform)
    read_profile = {
        "crs": ref_crs,
        "transform": rio_windows.transform(read_window, ref_transform),
        "width": read_window.width,
        "height": read_window.height,
    }
    return read_window, read_profile, display_bounds, aoi_gdf


def extent(profile: dict) -> tuple[float, float, float, float]:
    t = profile["transform"]
    left, top = t.c, t.f
    right = left + t.a * profile["width"]
    bottom = top + t.e * profile["height"]
    return left, right, bottom, top

# =============================================================================
# RASTER IO
# =============================================================================

def read_band(path: Path, window: Window | None = None) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1, window=window).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def discover_tifs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))


def file_hour_utc(path: Path) -> int:
    match = re.search(r"_(\d{4})\.tiff?$", path.name, flags=re.IGNORECASE)
    return int(match.group(1)[:2])


def pick_hour(folder: Path, hour_utc: int) -> Path:
    matches = [p for p in discover_tifs(folder) if file_hour_utc(p) == hour_utc]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {hour_utc:02d}:00 UTC raster in {folder}")
    return matches[0]


def robust_vlim(arr: np.ndarray) -> tuple[float, float]:
    """2nd/98th percentile stretch, so a few extreme outliers don't wash out
    the color scale for the rest of the map (common for elevation/canopy)."""
    finite = arr[np.isfinite(arr)]
    return float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98))

# =============================================================================
# BIVARIATE RESPONSE -- same method as the full-city map
# =============================================================================

def rank01(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    n = len(values)
    return (ranks - 1.0) / (n - 1.0) if n > 1 else np.zeros_like(values)


def hex_to_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def bivariate_rgb(rank_temp: np.ndarray, rank_delta: np.ndarray) -> np.ndarray:
    c00, c10 = hex_to_rgb(COLOR_COOL_STABLE), hex_to_rgb(COLOR_HOT_STABLE)
    c01, c11 = hex_to_rgb(COLOR_COOL_AMPLIFYING), hex_to_rgb(COLOR_HOT_AMPLIFYING)
    x = np.clip(rank_temp, 0, 1)[..., None]
    y = np.clip(rank_delta, 0, 1)[..., None]
    return (1 - x) * (1 - y) * c00 + x * (1 - y) * c10 + (1 - x) * y * c01 + x * y * c11


def smooth_field(field: np.ndarray, valid: np.ndarray, size: int = 3) -> np.ndarray:
    """Weighted box smoothing: averages only over valid neighbors, so pixels
    near the edge of the valid domain are not biased toward zero. Used both
    for the response ranks and (for a comparable look) for land cover."""
    weights = valid.astype(float)
    values = np.where(valid & np.isfinite(field), field, 0.0)
    num = uniform_filter(values, size=size, mode="nearest")
    den = uniform_filter(weights, size=size, mode="nearest")
    out = np.full(field.shape, np.nan)
    good = den > 0
    out[good] = num[good] / den[good]
    out[~valid] = np.nan
    return out


def compute_city_wide_response():
    """City-wide THW, delta, and their ranks. Computed once and reused for
    both the cropped map and the background scatter."""
    target = read_band(REFERENCE_RASTER)
    domain = np.isfinite(target) & (target > 0)

    heatwave, baseline = [], []
    for label, folder in HEATWAVE_DIRS.items():
        hw_path = pick_hour(folder, TARGET_HOUR_UTC)
        base_path = pick_hour(BASELINE_DIR, TARGET_HOUR_UTC)
        heatwave.append(np.where(domain, read_band(hw_path), np.nan))
        baseline.append(np.where(domain, read_band(base_path), np.nan))
        print(f"  heatwave {label}: {hw_path.name}")

    thw = np.where(domain, np.nanmean(np.stack(heatwave), axis=0), np.nan)
    tmean = np.where(domain, np.nanmean(np.stack(baseline), axis=0), np.nan)
    delta = np.where(domain, thw - tmean, np.nan)

    valid = domain & np.isfinite(thw) & np.isfinite(delta) & (thw >= MIN_VALID_TEMP_C)
    print(f"  city-wide classified pixels: {int(valid.sum()):,}")

    rank_t = np.full_like(thw, np.nan)
    rank_d = np.full_like(delta, np.nan)
    rank_t[valid] = rank01(thw[valid])
    rank_d[valid] = rank01(delta[valid])
    return thw, delta, valid, rank_t, rank_d


def crop_response_rgba(rank_t: np.ndarray, rank_d: np.ndarray, valid: np.ndarray, window: Window) -> np.ndarray:
    """Smooth city-wide (so AOI-edge pixels get full neighborhood context),
    then crop to the read window for display."""
    rank_t_smooth = smooth_field(rank_t, valid)
    rank_d_smooth = smooth_field(rank_d, valid)
    rgb = bivariate_rgb(rank_t_smooth, rank_d_smooth)

    rgba = np.zeros((*rank_t.shape, 4))
    finite = valid & np.isfinite(rank_t_smooth) & np.isfinite(rank_d_smooth)
    rgba[..., :3] = np.where(finite[..., None], rgb, 0.0)
    rgba[..., 3] = np.where(finite, 1.0, 0.0)

    rows = slice(window.row_off, window.row_off + window.height)
    cols = slice(window.col_off, window.col_off + window.width)
    return rgba[rows, cols]


def sample_background_scatter(thw, delta, rank_t, rank_d, valid):
    """A muted city-wide sample, colored with the same bivariate rank colors
    as panel A, used as a faint background echo of the citywide pattern."""
    rows, cols = np.nonzero(valid)
    rng = np.random.default_rng(RANDOM_SEED)
    if rows.size > BACKGROUND_SCATTER_N:
        pick = rng.choice(rows.size, size=BACKGROUND_SCATTER_N, replace=False)
        rows, cols = rows[pick], cols[pick]
    x = thw[rows, cols]
    y = delta[rows, cols]
    colors = bivariate_rgb(rank_t[rows, cols], rank_d[rows, cols])
    return x, y, colors

# =============================================================================
# LAND COVER
# =============================================================================

def landcover_classes(aoi_gdf: gpd.GeoDataFrame, profile: dict) -> np.ndarray:
    shape = (profile["height"], profile["width"])
    class_arr = np.zeros(shape, dtype=np.uint8)
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    bounds = (minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M)
    already = np.zeros(shape, dtype=bool)

    for i, name in enumerate(LULC_ORDER, start=1):
        gdf = gpd.read_file(LULC_PATHS[name], bbox=bounds)
        if gdf.empty:
            continue
        if gdf.crs != aoi_gdf.crs:
            gdf = gdf.to_crs(aoi_gdf.crs)
        geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
        if not geoms:
            continue
        mask = rasterize(
            ((g, 1) for g in geoms),
            out_shape=shape, transform=profile["transform"],
            fill=0, dtype=np.uint8,
        ).astype(bool)
        mask &= ~already
        class_arr[mask] = i
        already |= mask
    return class_arr


def landcover_rgba(class_arr: np.ndarray) -> np.ndarray:
    """Convert categorical land-cover classes directly to RGBA.

    No spatial smoothing or color blending is applied. Each raster cell keeps
    the exact color assigned to its land-cover class.
    """
    valid = class_arr > 0

    rgba = np.zeros((*class_arr.shape, 4), dtype=float)

    for i, name in enumerate(LULC_ORDER, start=1):
        mask = class_arr == i
        rgba[mask, :3] = hex_to_rgb(LULC_COLORS[name])
        rgba[mask, 3] = 1.0

    rgba[~valid, 3] = 0.0
    return rgba

# =============================================================================
# SHARED PANEL FRAME
# =============================================================================

def new_map_axes(display_bounds: tuple[float, float, float, float]):
    """One consistent figure/axes pair for every panel: same margin and
    typography, sized from the AOI's true (unbuffered) aspect ratio, no
    border/frame, and a transparent axes background so layered panels (the
    response map over its background scatter) actually show through."""
    left, bottom, right, top = display_bounds
    aspect = (right - left) / (top - bottom)
    width_cm = MAP_HEIGHT_CM * aspect

    fig = plt.figure(figsize=(width_cm / 2.54, MAP_HEIGHT_CM / 2.54), dpi=DPI)
    m = AXES_MARGIN
    ax = fig.add_axes([m, m, 1 - 2 * m, 1 - 2 * m])
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.axis("off")
    ax.patch.set_alpha(0)
    return fig, ax


def save(fig, name: str) -> None:
    path = OUTPUT_DIR / f"{name}.svg"
    fig.savefig(path, transparent=True)
    plt.close(fig)
    print(f"wrote {path}")

# =============================================================================
# PANELS
# =============================================================================

def make_response_panel(rgba, display_bounds, read_profile, scatter_x, scatter_y, scatter_colors) -> None:
    fig, ax = new_map_axes(display_bounds)

    # Faint background echo of the citywide distribution, in (THW, delta)
    # value space -- not geographic -- so it shows through only where the
    # map itself has no data (buildings, roads, water).
    bg_ax = fig.add_axes(ax.get_position(), zorder=0)
    bg_ax.set_xlim(scatter_x.min(), scatter_x.max())
    bg_ax.set_ylim(scatter_y.min(), scatter_y.max())
    bg_ax.scatter(scatter_x, scatter_y, c=scatter_colors, s=BACKGROUND_SCATTER_SIZE,
                  alpha=BACKGROUND_SCATTER_ALPHA, edgecolor="none", linewidths=0)
    bg_ax.axis("off")
    bg_ax.patch.set_alpha(0)

    ax.imshow(rgba, extent=extent(read_profile), interpolation="bilinear", zorder=1)
    save(fig, "panelA_response")


def make_response_key() -> None:
    """Bare bivariate key: color square only, no ticks, labels, or border."""
    fig = plt.figure(figsize=(3.0 / 2.54, 3.0 / 2.54), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    grid = np.linspace(0, 1, 100)
    xx, yy = np.meshgrid(grid, grid)
    ax.imshow(bivariate_rgb(xx, yy), extent=(0, 1, 0, 1), origin="lower", interpolation="bilinear")
    ax.axis("off")
    save(fig, "panelA_response_key")


def _colorbar_on_top(fig, ax, im, label: str) -> None:
    """Compact inset colorbar, sized 30% larger than a plain default inset,
    with its label above the bar and a shaded backing box (matching the
    land-cover legend) so it stays legible over any part of the map."""
    x0, y0, width, height = 0.06, 0.10, 0.40, 0.05
    pad_x, pad_below, pad_above = 0.02, 0.02, 0.14

    backing = Rectangle(
        (x0 - pad_x, y0 - pad_below), width + 2 * pad_x, height + pad_below + pad_above,
        transform=ax.transAxes, facecolor=LEGEND_SHADE_COLOR, edgecolor="none",
        alpha=LEGEND_SHADE_ALPHA, zorder=1.5,
    )
    ax.add_patch(backing)

    cax = ax.inset_axes([x0, y0, width, height])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.set_ticks_position("top")
    cb.set_label(label, fontsize=9.1, fontfamily=FONT, color=TEXT_COLOR, labelpad=2)
    cb.ax.tick_params(labelsize=7.8, length=2, colors=TEXT_COLOR)


def make_elevation_panel(read_window: Window, read_profile: dict, display_bounds) -> None:
    arr = read_band(DTM_RASTER, read_window)
    vmin, vmax = robust_vlim(arr)
    fig, ax = new_map_axes(display_bounds)
    im = ax.imshow(arr, extent=extent(read_profile), cmap=MONOCHROME_CMAP,
                    vmin=vmin, vmax=vmax, interpolation="bilinear", zorder=1)
    _colorbar_on_top(fig, ax, im, "elevation (m)")
    save(fig, "panelB_elevation")


def make_canopy_panel(read_window: Window, read_profile: dict, display_bounds) -> None:
    arr = read_band(CHM_RASTER, read_window)
    vmin, vmax = robust_vlim(arr)
    fig, ax = new_map_axes(display_bounds)
    im = ax.imshow(arr, extent=extent(read_profile), cmap=MONOCHROME_CMAP,
                    vmin=vmin, vmax=vmax, interpolation="bilinear", zorder=1)
    _colorbar_on_top(fig, ax, im, "canopy height (m)")
    save(fig, "panelC_canopy_height")


def make_landcover_panel(
    class_arr: np.ndarray,
    read_profile: dict,
    display_bounds,
) -> None:
    rgba = landcover_rgba(class_arr)

    fig, ax = new_map_axes(display_bounds)

    ax.imshow(
        rgba,
        extent=extent(read_profile),
        interpolation="nearest",
        zorder=1,
    )

    save(fig, "panelD_landcover")

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    apply_style()
    print(f"AOI: {AOI_PATH.name}")
    print(f"Output: {OUTPUT_DIR}")

    read_window, read_profile, display_bounds, aoi_gdf = aoi_window(AOI_PATH, REFERENCE_RASTER, BUFFER_M)
    print(f"Read window (buffered {BUFFER_M} m): {read_window}")

    print("Response panel (city-wide ranks)...")
    thw, delta, valid, rank_t, rank_d = compute_city_wide_response()
    rgba = crop_response_rgba(rank_t, rank_d, valid, read_window)
    scatter_x, scatter_y, scatter_colors = sample_background_scatter(thw, delta, rank_t, rank_d, valid)
    make_response_panel(rgba, display_bounds, read_profile, scatter_x, scatter_y, scatter_colors)
    make_response_key()

    print("Elevation panel...")
    make_elevation_panel(read_window, read_profile, display_bounds)

    print("Canopy height panel...")
    make_canopy_panel(read_window, read_profile, display_bounds)

    print("Land cover panel...")
    class_arr = landcover_classes(aoi_gdf, read_profile)
    make_landcover_panel(class_arr, read_profile, display_bounds)


if __name__ == "__main__":
    main()
