#!/usr/bin/env python3
"""
Figure f3_2: 4×4 bivariate heatwave-response map.

Purpose
-------
Classify vegetated prediction-domain pixels by two rank-normalized axes:

1. Absolute mean heatwave temperature, THW
2. Heatwave amplification from average conditions, THW - Tavg

Both axes are split at the median rank within the valid mapping domain. The
resulting 4 × 4 bivariate classification shows a graded joint pattern of
absolute heatwave temperature and heatwave amplification from average conditions.

Inputs expected
---------------
- Heatwave prediction rasters for 2010, 2018, 2021 at 10:00 UTC / 13:00 local
- Baseline mean prediction raster at the matching hour
- Valid mapping-domain raster from the f4 connectivity-loss workflow:
  figures/results/figures/f4/rasters/p90_loss_target_domain_tree_veg_nwn_pm1p0deg.tif
- Helsinki peruspiiri layer for outer boundary
- LULC vector layers for optional class summary

Outputs
-------
\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures\f3_2
  f3_2_bivariate_heatwave_response_typology_pm1p0deg.svg
  rasters/f3_2_bivariate_heatwave_response_typology_pm1p0deg.tif
  tables/f3_2_bivariate_heatwave_response_typology_summary_pm1p0deg.csv
  tables/f3_2_bivariate_heatwave_response_typology_by_lulc_pm1p0deg.csv
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from scipy.ndimage import label as ndi_label, uniform_filter
from rasterio.features import rasterize
from rasterio.warp import reproject
from shapely.geometry import box

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
FIGURES_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures")
WORKDIR = FIGURES_DIR / "f3_2"
WORKDIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = WORKDIR / "tables"
RASTER_OUT_DIR = WORKDIR / "rasters"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
RASTER_OUT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SETTINGS = FIGURES_DIR / "global_plotting_settings.py"
PERUSPIIRI_PATH = DATA_DIR / "figures" / "offset_figure" / "peruspiiri_WFS.gpkg"

HEATWAVE_INPUTS = {
    "2010": DATA_DIR / "predictions" / "2010" / "20100728",
    "2018": DATA_DIR / "predictions" / "2018" / "20180717",
    "2021": DATA_DIR / "predictions" / "2021" / "20210714",
}
BASELINE_MEAN_INPUT = DATA_DIR / "predictions" / "baseline" / "15cm_July_allday" / "pred_20000715_1000.tif"

# Valid vegetation / prediction domain from the f4 connectivity-loss workflow.
MAP_TOLERANCE = 1.0
TOL_LABEL = str(MAP_TOLERANCE).replace(".", "p")
TARGET_DOMAIN_RASTER = (
    FIGURES_DIR
    / "f4"
    / "rasters"
    / f"p90_loss_target_domain_tree_veg_nwn_pm{TOL_LABEL}deg.tif"
)

LULC_LAYERS = [
    ("Fields", [DATA_DIR / "LULC" / "lc_fields.gpkg"]),
    ("Trees 2–10 m", [DATA_DIR / "LULC" / "lc_trees2-10.gpkg"]),
    ("Trees 10–15 m", [DATA_DIR / "LULC" / "lc_trees10-15.gpkg"]),
    ("Trees >15 m", [
        DATA_DIR / "LULC" / "lc_trees15-20.gpkg",
        DATA_DIR / "LULC" / "lc_trees_o20.gpkg",
    ]),
    ("Other vegetation", [DATA_DIR / "LULC" / "lc_otherveg.gpkg"]),
]
PLOT_ORDER = ["Trees 2–10 m", "Trees 10–15 m", "Trees >15 m", "Fields", "Other vegetation"]
DISPLAY_LABEL_MAP = {
    "Trees 2–10 m": "Trees 2–10 m",
    "Trees 10–15 m": "Trees 10–15 m",
    "Trees >15 m": "Trees >15 m",
    "Fields": "Fields",
    "Other vegetation": "Other veg.",
}

# =============================================================================
# SETTINGS
# =============================================================================

LOCAL_UTC_OFFSET_HOURS = 3
EXPECTED_PEAK_LOCAL_HOUR = 13
TARGET_HEATWAVE_UTC_HOUR = (EXPECTED_PEAK_LOCAL_HOUR - LOCAL_UTC_OFFSET_HOURS) % 24
AGGREGATION = "mean"
ASSUME_VECTOR_EPSG_IF_MISSING = 3879
MIN_VALID_ABSOLUTE_TEMP_C = 15.0

OUTPUT_BASENAME = f"f3_2_bivariate_heatwave_response_continuous_mean_corners_pm{TOL_LABEL}deg"
OUT_RASTER = RASTER_OUT_DIR / f"{OUTPUT_BASENAME}.tif"
OUT_SUMMARY_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_summary.csv"
OUT_LULC_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_by_lulc.csv"

# Class IDs written to raster.
# 0 = no data / outside valid domain. IDs 1..16 encode a 4 × 4 bivariate grid:
# class_id = offset_quartile * 4 + temp_quartile + 1, where quartiles are 0..3
# for low, low-mid, high-mid, high.
N_CLASSES = 4
CLASS_BREAKS = [0.25, 0.50, 0.75]

# Sequential × sequential bivariate palette. Rows increase upward in heatwave
# amplification (THW - Tavg); columns increase rightward in absolute THW.
# The corners retain the intended semantics: cool/stable = green,
# hot/stable = orange, cool/amplifying = purple, hot/amplifying = red.
BIVAR_COLORS = {
    1:  "#1b9e77", 2:  "#73b86f", 3:  "#d9b365", 4:  "#fdae61",
    5:  "#4f9b8c", 6:  "#8eaa7d", 7:  "#c69b72", 8:  "#e58a62",
    9:  "#6d679f", 10: "#967896", 11: "#bd6f82", 12: "#d95f5f",
    13: "#7b3294", 14: "#9f3f8e", 15: "#bd3b79", 16: "#d7191c",
}

BIN_LABELS = {0: "low", 1: "", 2: "", 3: "high"}


# Continuous bivariate display corner colors. The rendered map interpolates
# between these four corners using the two rank surfaces.
COLOR_LOW_T_LOW_D = "#1b9e77"   # low absolute heat, low amplification
COLOR_HIGH_T_LOW_D = "#fdae61"  # high absolute heat, low amplification
COLOR_LOW_T_HIGH_D = "#7b3294"  # low absolute heat, high amplification
COLOR_HIGH_T_HIGH_D = "#d7191c" # high absolute heat, high amplification

# Fixed standalone panel layout, respecting the general style but keeping the map
# compact enough for a single f3_2 panel.
PANEL_WIDTH_CM = 20.0
PANEL_HEIGHT_CM = 14.0
MAP_AXES_RECT = [0.02, 0.03, 0.74, 0.94]
LEGEND_AXES_RECT = [0.755, 0.15, 0.235, 0.70]
BOUNDARY_COLOR = "black"
BOUNDARY_LINEWIDTH = 0.42
BOUNDARY_ALPHA = 0.45

# The raw pixel classification is intentionally written to disk unchanged, but a
# publication map can become unreadable at native 10 m resolution. The display
# raster below is a mode-aggregated version of the typology, restricted to the
# same valid domain. Increase to 4 or 5 if the map is still too salt-and-pepper;
# set to 1 for native pixels.
DISPLAY_BLOCK_SIZE = 1

# Display is continuous RGB, not the categorical raster. Smooth the continuous
# rank surfaces only for display; the saved raster and CSVs stay unsmoothed.
DISPLAY_RANK_SMOOTH_SIZE = 3
DISPLAY_VALID_DOMAIN_ALPHA = 0.18
DISPLAY_MAP_ALPHA = 0.96

# Kept for the raw categorical helper functions; not used in the continuous map display.
DISPLAY_MIN_PATCH_PIXELS = 0
DISPLAY_MODE_FILTER_SIZE = 1

# If True, the map extent is set to the raster bounds, not to the Helsinki
# boundary. This prevents the southern city outline from stretching the plot.
CROP_TO_RASTER_BOUNDS = True

# =============================================================================
# GLOBAL STYLE
# =============================================================================

def import_global_plotting_settings():
    if not GLOBAL_SETTINGS.exists():
        raise FileNotFoundError(f"Missing global plotting settings: {GLOBAL_SETTINGS}")
    settings_dir = str(GLOBAL_SETTINGS.parent)
    if settings_dir not in sys.path:
        sys.path.insert(0, settings_dir)
    module_name = "global_plotting_settings"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def ensure_global_style() -> None:
    global gps, STYLE
    gps_local = import_global_plotting_settings()
    if not hasattr(gps_local, "STYLE"):
        raise AttributeError("global_plotting_settings.py must define STYLE = FigureStyle(...).")
    STYLE = replace(
        gps_local.STYLE,
        export_png=False,
        export_pdf=False,
        export_svg=True,
        use_tight_bbox=False,
        pad_inches=0.0,
    )
    gps = gps_local
    gps.apply_style(STYLE)
    mpl.rcParams["svg.fonttype"] = "none"

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class RasterSurface:
    label: str
    path: Path
    hour_utc: int
    array: np.ndarray
    profile: dict

# =============================================================================
# IO / GRID HELPERS
# =============================================================================

def check_required_files() -> None:
    required = [
        GLOBAL_SETTINGS,
        TARGET_DOMAIN_RASTER,
        BASELINE_MEAN_INPUT,
        PERUSPIIRI_PATH,
        *HEATWAVE_INPUTS.values(),
    ]
    for _, paths in LULC_LAYERS:
        required.extend(paths)
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print(f"  {p}")
        raise FileNotFoundError("One or more required inputs are missing.")


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = src.profile.copy()
        profile.update(crs=src.crs, transform=src.transform, width=src.width, height=src.height, nodata=np.nan)
    return arr, profile


def same_grid(a: dict, b: dict) -> bool:
    return (
        a["crs"] == b["crs"]
        and a["transform"] == b["transform"]
        and a["width"] == b["width"]
        and a["height"] == b["height"]
    )


def reproject_to_match(arr: np.ndarray, src_profile: dict, dst_profile: dict, *, resampling=Resampling.bilinear) -> np.ndarray:
    if same_grid(src_profile, dst_profile):
        return arr
    dst = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype="float64")
    reproject(
        source=arr,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=np.nan,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


def file_hour_utc(path: Path) -> int:
    m = re.search(r"_(\d{4})\.tiff?$", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not extract UTC hour from filename: {path.name}")
    return int(m.group(1)[:2])


def discover_tifs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
    raise FileNotFoundError(f"Input path does not exist: {path}")


def select_heatwave_surface(label: str, input_path: Path) -> RasterSurface:
    tif_paths = discover_tifs(input_path)
    if not tif_paths:
        raise FileNotFoundError(f"No GeoTIFF files found for {label}: {input_path}")
    matching = [p for p in tif_paths if file_hour_utc(p) == TARGET_HEATWAVE_UTC_HOUR]
    if len(matching) != 1:
        available = ", ".join(f"{file_hour_utc(p):02d}:00 UTC ({p.name})" for p in tif_paths)
        raise FileNotFoundError(
            f"Expected exactly one {TARGET_HEATWAVE_UTC_HOUR:02d}:00 UTC raster for {label}. Available: {available}"
        )
    chosen = matching[0]
    arr, profile = read_raster(chosen)
    print(f"Selected {label}: {chosen.name} at {file_hour_utc(chosen):02d}:00 UTC")
    return RasterSurface(label=label, path=chosen, hour_utc=file_hour_utc(chosen), array=arr, profile=profile)


def matching_baseline_for_hour(baseline_input: Path, hour_utc: int) -> Path:
    candidates = discover_tifs(baseline_input)
    same_hour = [p for p in candidates if file_hour_utc(p) == hour_utc]
    if len(same_hour) == 1:
        return same_hour[0]
    if baseline_input.is_file() and baseline_input.parent.exists():
        parent_candidates = sorted(baseline_input.parent.glob("*.tif")) + sorted(baseline_input.parent.glob("*.tiff"))
        same_hour = [p for p in parent_candidates if file_hour_utc(p) == hour_utc]
        if len(same_hour) == 1:
            return same_hour[0]
    available = ", ".join(f"{file_hour_utc(p):02d}:00 UTC ({p.name})" for p in candidates)
    raise FileNotFoundError(f"No unique p90 baseline raster for {hour_utc:02d}:00 UTC. Available: {available}")


def aggregate_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    with np.errstate(all="ignore"):
        if AGGREGATION == "mean":
            return np.nanmean(stack, axis=0)
        if AGGREGATION == "max":
            return np.nanmax(stack, axis=0)
    raise ValueError(f"Unsupported AGGREGATION: {AGGREGATION}")


def compute_temperature_surfaces(target_profile: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heatwave_arrays = []
    mean_arrays = []

    for label, input_path in HEATWAVE_INPUTS.items():
        surface = select_heatwave_surface(label, input_path)
        heatwave = reproject_to_match(surface.array, surface.profile, target_profile, resampling=Resampling.bilinear)

        mean_path = matching_baseline_for_hour(BASELINE_MEAN_INPUT, surface.hour_utc)
        mean_arr, mean_profile = read_raster(mean_path)
        tmean = reproject_to_match(mean_arr, mean_profile, target_profile, resampling=Resampling.bilinear)

        heatwave_arrays.append(heatwave)
        mean_arrays.append(tmean)
        print(f"Matched mean baseline for {label}: {mean_path.name}")

    thw = aggregate_arrays(heatwave_arrays)
    tmean = aggregate_arrays(mean_arrays)
    offset = thw - tmean
    return thw, tmean, offset


def write_geotiff(path: Path, arr: np.ndarray, profile: dict) -> None:
    out_profile = profile.copy()
    out_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    out = np.where(np.isfinite(arr), arr, 0).astype(np.uint8)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(out, 1)

# =============================================================================
# RANKING / CLASSIFICATION
# =============================================================================

def rank01(values: np.ndarray) -> np.ndarray:
    """Return empirical rank scaled to 0..1, using average ranks for ties."""
    s = pd.Series(values)
    ranks = s.rank(method="average").to_numpy(dtype=float)
    n = len(values)
    if n <= 1:
        return np.zeros_like(values, dtype=float)
    return (ranks - 1.0) / (n - 1.0)


def classify_typology(thw: np.ndarray, offset: np.ndarray, valid_domain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = (
        valid_domain
        & np.isfinite(thw)
        & np.isfinite(offset)
        & (thw >= MIN_VALID_ABSOLUTE_TEMP_C)
    )
    if int(valid.sum()) == 0:
        raise ValueError("No valid pixels for bivariate classification.")

    r_thw = np.full_like(thw, np.nan, dtype=float)
    r_offset = np.full_like(offset, np.nan, dtype=float)
    r_thw[valid] = rank01(thw[valid])
    r_offset[valid] = rank01(offset[valid])

    temp_cls = np.full(thw.shape, -1, dtype=np.int8)
    off_cls = np.full(thw.shape, -1, dtype=np.int8)
    temp_cls[valid] = np.digitize(r_thw[valid], bins=CLASS_BREAKS, right=False)
    off_cls[valid] = np.digitize(r_offset[valid], bins=CLASS_BREAKS, right=False)

    bivar = np.zeros(thw.shape, dtype=np.uint8)
    bivar[valid] = (off_cls[valid] * N_CLASSES + temp_cls[valid] + 1).astype(np.uint8)
    return bivar, r_thw, r_offset, temp_cls, off_cls

# =============================================================================
# VECTOR / SUMMARY HELPERS
# =============================================================================

def raster_extent(profile: dict) -> tuple[float, float, float, float]:
    t = profile["transform"]
    left = t.c
    top = t.f
    right = left + t.a * profile["width"]
    bottom = top + t.e * profile["height"]
    return left, right, bottom, top


def raster_bounds_polygon(profile: dict):
    left, right, bottom, top = raster_extent(profile)
    return box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))


def raster_bbox_in_vector_crs(path: Path, target_profile: dict):
    try:
        info = gpd.read_file(path, rows=1)
        crs = info.crs
        if crs is None:
            crs = f"EPSG:{ASSUME_VECTOR_EPSG_IF_MISSING}"
        rb = gpd.GeoDataFrame(geometry=[raster_bounds_polygon(target_profile)], crs=target_profile["crs"]).to_crs(crs)
        return tuple(rb.total_bounds)
    except Exception:
        return None


def read_vector_fast(path: Path, target_profile: dict, label: str, *, use_bbox: bool = True) -> gpd.GeoDataFrame:
    bbox = raster_bbox_in_vector_crs(path, target_profile) if use_bbox else None
    try:
        gdf = gpd.read_file(path, bbox=bbox) if bbox is not None else gpd.read_file(path)
    except TypeError:
        gdf = gpd.read_file(path)

    if gdf.empty:
        print(f"WARNING: {label} is empty after bbox read; retrying full layer.")
        gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"{label} contains no features: {path}")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError(f"{label} has no non-empty geometries: {path}")

    if gdf.crs is None:
        print(f"WARNING: {label} has no CRS; assuming EPSG:{ASSUME_VECTOR_EPSG_IF_MISSING}.")
        gdf = gdf.set_crs(epsg=ASSUME_VECTOR_EPSG_IF_MISSING, allow_override=True)
    if gdf.crs != target_profile["crs"]:
        gdf = gdf.to_crs(target_profile["crs"])
    return gdf


def rasterize_gdf(gdf: gpd.GeoDataFrame, profile: dict, *, all_touched: bool = False) -> np.ndarray:
    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        return np.zeros((profile["height"], profile["width"]), dtype=bool)
    arr = rasterize(
        ((geom, 1) for geom in geoms),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        all_touched=all_touched,
        dtype=np.uint8,
    )
    return arr.astype(bool)


def load_outer_boundary(profile: dict) -> gpd.GeoDataFrame:
    gdf = read_vector_fast(PERUSPIIRI_PATH, profile, "peruspiiri", use_bbox=True)
    dissolved = gpd.GeoDataFrame(geometry=[gdf.unary_union], crs=gdf.crs)

    # Clip the dissolved boundary to the raster extent. Otherwise the southern
    # Helsinki outline can extend far beyond the mapped prediction domain and
    # force the map to shrink inside the panel.
    bounds_geom = raster_bounds_polygon(profile)
    bounds_gdf = gpd.GeoDataFrame(geometry=[bounds_geom], crs=profile["crs"])
    try:
        clipped = gpd.clip(dissolved, bounds_gdf)
    except Exception:
        clipped = dissolved.copy()
        clipped["geometry"] = clipped.geometry.intersection(bounds_geom)
        clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()

    return clipped if not clipped.empty else dissolved


def load_lulc_masks(profile: dict, valid_domain: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for class_name, paths in LULC_LAYERS:
        class_mask = np.zeros_like(valid_domain, dtype=bool)
        for path in paths:
            gdf = read_vector_fast(path, profile, f"LULC {class_name}", use_bbox=True)
            class_mask |= rasterize_gdf(gdf, profile, all_touched=False)
        class_mask &= valid_domain
        masks[class_name] = class_mask
        print(f"LULC {class_name}: {int(class_mask.sum())} valid raster cells")

    exclusive: dict[str, np.ndarray] = {}
    already = np.zeros_like(valid_domain, dtype=bool)
    for class_name, _ in LULC_LAYERS:
        m = masks[class_name] & (~already)
        exclusive[class_name] = m
        already |= m
        print(f"LULC {class_name} exclusive cells: {int(m.sum())}")
    return exclusive


def cell_area_m2(profile: dict) -> float:
    return abs(float(profile["transform"].a) * float(profile["transform"].e))


def summarize_typology(typology: np.ndarray, thw: np.ndarray, offset: np.ndarray, profile: dict) -> pd.DataFrame:
    area = cell_area_m2(profile)
    total = int(np.sum(typology > 0))
    rows = []
    for class_id in range(1, N_CLASSES * N_CLASSES + 1):
        temp_bin = (class_id - 1) % N_CLASSES
        offset_bin = (class_id - 1) // N_CLASSES
        mask = typology == class_id
        count = int(mask.sum())
        rows.append({
            "bivariate_id": class_id,
            "temp_bin": temp_bin,
            "offset_bin": offset_bin,
            "pixel_count": count,
            "area_m2": count * area,
            "percent_valid_domain": 100.0 * count / total if total else np.nan,
            "mean_heatwave_temp_degC": float(np.nanmean(thw[mask])) if count else np.nan,
            "median_heatwave_temp_degC": float(np.nanmedian(thw[mask])) if count else np.nan,
            "mean_hw_minus_mean_degC": float(np.nanmean(offset[mask])) if count else np.nan,
            "median_hw_minus_mean_degC": float(np.nanmedian(offset[mask])) if count else np.nan,
        })
    return pd.DataFrame.from_records(rows)


def summarize_by_lulc(typology: np.ndarray, lulc_masks: dict[str, np.ndarray], profile: dict) -> pd.DataFrame:
    area = cell_area_m2(profile)
    rows = []
    type_totals = {class_id: int(np.sum(typology == class_id)) for class_id in range(1, N_CLASSES * N_CLASSES + 1)}
    for lulc in PLOT_ORDER:
        lulc_mask = lulc_masks[lulc] & (typology > 0)
        lulc_total = int(lulc_mask.sum())
        for class_id in range(1, N_CLASSES * N_CLASSES + 1):
            temp_bin = (class_id - 1) % N_CLASSES
            offset_bin = (class_id - 1) // N_CLASSES
            mask = lulc_mask & (typology == class_id)
            count = int(mask.sum())
            rows.append({
                "lulc_class": lulc,
                "lulc_label": DISPLAY_LABEL_MAP.get(lulc, lulc),
                "bivariate_id": class_id,
                "temp_bin": temp_bin,
                "offset_bin": offset_bin,
                "pixel_count": count,
                "area_m2": count * area,
                "percent_of_lulc_class": 100.0 * count / lulc_total if lulc_total else np.nan,
                "percent_of_bivariate_class": 100.0 * count / type_totals[class_id] if type_totals[class_id] else np.nan,
            })
    return pd.DataFrame.from_records(rows)

# =============================================================================
# PLOTTING
# =============================================================================

def make_typology_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    colors = [BIVAR_COLORS[i] for i in range(1, N_CLASSES * N_CLASSES + 1)]
    cmap = ListedColormap(colors, name="heatwave_response_bivariate4x4_mean")
    norm = BoundaryNorm(np.arange(0.5, N_CLASSES * N_CLASSES + 1.5, 1.0), cmap.N)
    return cmap, norm


def save_svg(fig: plt.Figure, basename: str) -> None:
    gps.make_transparent(fig)
    out = WORKDIR / f"{basename}.svg"
    fig.savefig(out, transparent=STYLE.transparent, facecolor="none", edgecolor="none", bbox_inches=None, pad_inches=0)
    width_cm = fig.get_figwidth() * gps.CM_PER_INCH
    height_cm = fig.get_figheight() * gps.CM_PER_INCH
    print(f"[OK] wrote {out} ({width_cm:.2f} × {height_cm:.2f} cm fixed canvas)")


def _hex_to_rgb01(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def bivariate_rgb_from_ranks(r_temp: np.ndarray, r_delta: np.ndarray) -> np.ndarray:
    """Bilinear interpolation between four corner colors."""
    c00 = _hex_to_rgb01(COLOR_LOW_T_LOW_D)
    c10 = _hex_to_rgb01(COLOR_HIGH_T_LOW_D)
    c01 = _hex_to_rgb01(COLOR_LOW_T_HIGH_D)
    c11 = _hex_to_rgb01(COLOR_HIGH_T_HIGH_D)
    x = np.clip(r_temp, 0, 1)[..., None]
    y = np.clip(r_delta, 0, 1)[..., None]
    return (1 - x) * (1 - y) * c00 + x * (1 - y) * c10 + (1 - x) * y * c01 + x * y * c11


def plot_bivariate_legend(ax) -> None:
    """Draw a square continuous bivariate legend with four corner callouts.

    The map itself uses a continuous bivariate RGB interpolation. The legend is
    therefore drawn as a continuous square, with the four interpretive corner
    labels placed around the grid rather than inside it.
    """
    ax.set_xlim(-0.88, 1.92)
    ax.set_ylim(-0.86, 1.84)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    n = 180
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y)
    rgb = bivariate_rgb_from_ranks(xx, yy)

    # Square legend grid in data coordinates. With equal aspect, this stays
    # physically square regardless of the legend axis rectangle.
    ax.imshow(rgb, extent=(0, 1, 0, 1), origin="lower", interpolation="bilinear", zorder=1)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="white", linewidth=0.9, zorder=2))

    line_col = getattr(STYLE, "col_grey", "#888888")
    txt_col = STYLE.col_black
    fs_corner = max(5.8, STYLE.fs_legend * 0.58)
    fs_tick = max(6.2, STYLE.fs_tick * 0.70)
    fs_axis = max(7.2, STYLE.fs_axis * 0.82)

    def callout(text, xy, xytext, ha, va):
        ax.plot([xy[0], xytext[0]], [xy[1], xytext[1]], color=line_col, linewidth=0.55, alpha=0.80, zorder=0)
        ax.text(
            xytext[0], xytext[1], text,
            ha=ha, va=va,
            fontsize=fs_corner,
            fontfamily=STYLE.font_family,
            color=txt_col,
            linespacing=1.05,
        )

    # Four corner interpretations. These are deliberately outside the color
    # square so the square remains readable.
    callout("cool +\nstable",       (0.02, 0.02), (-0.36, -0.26), "right", "top")
    callout("hot +\nstable",        (0.98, 0.02), (1.36, -0.26), "left",  "top")
    callout("cool +\namplifying",   (0.02, 0.98), (-0.36, 1.26), "right", "bottom")
    callout("hot +\namplifying",    (0.98, 0.98), (1.36, 1.26), "left",  "bottom")

    # Axis end labels and axis names, with generous offsets.
    ax.text(0.00, -0.095, "low", ha="left", va="top", fontsize=fs_tick,
            fontfamily=STYLE.font_family, color=txt_col)
    ax.text(1.00, -0.095, "high", ha="right", va="top", fontsize=fs_tick,
            fontfamily=STYLE.font_family, color=txt_col)
    ax.text(0.50, -0.30, "absolute heatwave T", ha="center", va="top",
            fontsize=fs_axis, fontfamily=STYLE.font_family, color=txt_col, fontweight="bold")

    ax.text(-0.095, 0.00, "low", ha="right", va="bottom", fontsize=fs_tick,
            fontfamily=STYLE.font_family, color=txt_col, rotation=90)
    ax.text(-0.095, 1.00, "high", ha="right", va="top", fontsize=fs_tick,
            fontfamily=STYLE.font_family, color=txt_col, rotation=90)
    ax.text(-0.36, 0.50, "ΔT to mean", ha="center", va="center",
            fontsize=fs_axis, fontfamily=STYLE.font_family, color=txt_col,
            fontweight="bold", rotation=90)


def mode_filter_display(arr: np.ndarray, size: int) -> np.ndarray:
    if size is None or size <= 1:
        return arr
    pad = size // 2
    padded = np.pad(arr, pad_width=pad, mode="constant", constant_values=0)
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            window = padded[i:i+size, j:j+size].ravel()
            window = window[window > 0]
            if window.size:
                counts = np.bincount(window, minlength=N_CLASSES * N_CLASSES + 1)
                out[i, j] = int(np.argmax(counts))
    return out


def remove_small_display_patches(typology: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels is None or min_pixels <= 1:
        return typology
    out = typology.copy()
    structure = np.ones((3, 3), dtype=int)
    for class_id in range(1, N_CLASSES * N_CLASSES + 1):
        mask = out == class_id
        labeled, n_labels = ndi_label(mask, structure=structure)
        if n_labels == 0:
            continue
        counts = np.bincount(labeled.ravel())
        drop = np.where(counts < min_pixels)[0]
        drop = drop[drop != 0]
        if drop.size:
            out[np.isin(labeled, drop)] = 0
    return out


def block_mode_display(typology: np.ndarray, block_size: int) -> np.ndarray:
    if block_size is None or block_size <= 1:
        return typology
    h, w = typology.shape
    h2 = (h // block_size) * block_size
    w2 = (w // block_size) * block_size
    cropped = typology[:h2, :w2]
    blocks = cropped.reshape(h2 // block_size, block_size, w2 // block_size, block_size)
    out = np.zeros((h2 // block_size, w2 // block_size), dtype=np.uint8)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            vals = blocks[i, :, j, :].ravel()
            vals = vals[vals > 0]
            if vals.size:
                counts = np.bincount(vals, minlength=N_CLASSES * N_CLASSES + 1)
                out[i, j] = int(np.argmax(counts))
    return out


def block_extent(profile: dict, block_size: int, display_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    left, right, bottom, top = raster_extent(profile)
    if block_size is None or block_size <= 1:
        return left, right, bottom, top
    # The mode aggregation crops the bottom/right edge to a multiple of block_size.
    height, width = display_shape
    new_right = left + width * block_size * profile["transform"].a
    new_bottom = top + height * block_size * profile["transform"].e
    return left, new_right, new_bottom, top


def smooth_rank_for_display(rank_arr: np.ndarray, valid_mask: np.ndarray, size: int) -> np.ndarray:
    if size is None or size <= 1:
        return rank_arr.copy()
    weights = valid_mask.astype(float)
    values = np.where(valid_mask & np.isfinite(rank_arr), rank_arr, 0.0)
    num = uniform_filter(values, size=size, mode="nearest")
    den = uniform_filter(weights, size=size, mode="nearest")
    out = np.full(rank_arr.shape, np.nan, dtype=float)
    good = den > 0
    out[good] = num[good] / den[good]
    out[~valid_mask] = np.nan
    return out


def build_continuous_rgba(r_thw: np.ndarray, r_offset: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rt = smooth_rank_for_display(r_thw, valid_mask, DISPLAY_RANK_SMOOTH_SIZE)
    rd = smooth_rank_for_display(r_offset, valid_mask, DISPLAY_RANK_SMOOTH_SIZE)
    rgb = bivariate_rgb_from_ranks(rt, rd)
    rgba = np.zeros((r_thw.shape[0], r_thw.shape[1], 4), dtype=float)
    finite = valid_mask & np.isfinite(rt) & np.isfinite(rd)
    rgba[..., :3] = np.where(finite[..., None], rgb, 0.0)
    rgba[..., 3] = np.where(finite, DISPLAY_MAP_ALPHA, 0.0)
    return rgba

def plot_map(typology: np.ndarray, r_thw: np.ndarray, r_offset: np.ndarray, valid_domain: np.ndarray, profile: dict, boundary: gpd.GeoDataFrame) -> None:
    fig = plt.figure(figsize=(PANEL_WIDTH_CM / gps.CM_PER_INCH, PANEL_HEIGHT_CM / gps.CM_PER_INCH), dpi=STYLE.dpi_export)
    ax = fig.add_axes(MAP_AXES_RECT)
    legend_ax = fig.add_axes(LEGEND_AXES_RECT)

    left, right, bottom, top = raster_extent(profile)

    # faint domain backdrop so the masking does not dominate the map visually
    domain_rgba = np.zeros((valid_domain.shape[0], valid_domain.shape[1], 4), dtype=float)
    domain_rgba[..., :3] = _hex_to_rgb01("#ebe8df")
    domain_rgba[..., 3] = np.where(valid_domain, DISPLAY_VALID_DOMAIN_ALPHA, 0.0)
    ax.imshow(domain_rgba, extent=(left, right, bottom, top), interpolation="nearest", zorder=0)

    rgba = build_continuous_rgba(r_thw, r_offset, valid_domain)
    ax.imshow(rgba, extent=(left, right, bottom, top), interpolation="bilinear", zorder=1)

    if not boundary.empty:
        boundary.boundary.plot(ax=ax, color=BOUNDARY_COLOR, linewidth=BOUNDARY_LINEWIDTH, alpha=BOUNDARY_ALPHA, zorder=3)

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.axis("off")

    plot_bivariate_legend(legend_ax)
    save_svg(fig, OUTPUT_BASENAME)
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_global_style()
    print("Resolved global plotting style:")
    print(f"  global file: {GLOBAL_SETTINGS}")
    print(f"  output dir:  {WORKDIR}")
    print(f"  target domain: {TARGET_DOMAIN_RASTER}")
    print(f"  aggregation: {AGGREGATION}")
    print(f"  analysis hour: {EXPECTED_PEAK_LOCAL_HOUR:02d}:00 local = {TARGET_HEATWAVE_UTC_HOUR:02d}:00 UTC")

    check_required_files()

    target_arr, target_profile = read_raster(TARGET_DOMAIN_RASTER)
    valid_domain = np.isfinite(target_arr) & (target_arr > 0)
    print(f"Valid domain pixels: {int(valid_domain.sum()):,}")

    thw, tmean, offset = compute_temperature_surfaces(target_profile)
    typology, r_thw, r_offset, temp_cls, off_cls = classify_typology(thw, offset, valid_domain)

    write_geotiff(OUT_RASTER, typology, target_profile)
    print(f"[OK] wrote {OUT_RASTER}")

    summary_df = summarize_typology(typology, thw, offset, target_profile)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"[OK] wrote {OUT_SUMMARY_CSV}")
    print(summary_df.to_string(index=False))

    lulc_masks = load_lulc_masks(target_profile, typology > 0)
    lulc_df = summarize_by_lulc(typology, lulc_masks, target_profile)
    lulc_df.to_csv(OUT_LULC_CSV, index=False)
    print(f"[OK] wrote {OUT_LULC_CSV}")

    boundary = load_outer_boundary(target_profile)
    plot_map(typology, r_thw, r_offset, valid_domain, target_profile, boundary)


if __name__ == "__main__":
    main()
