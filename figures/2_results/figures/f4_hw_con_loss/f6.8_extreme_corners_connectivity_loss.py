#!/usr/bin/env python3
"""
Figure f3_2 companion: p90 connectivity loss in the four extreme bivariate heatwave-response classes.

This script is intentionally separate from the bivariate map script. It recomputes the same
4×4 rank-quartile bivariate classification from:
  x-axis: mean heatwave temperature, THW
  y-axis: heatwave amplification from mean conditions, THW - Tavg

It then extracts the four corner classes and summarizes p90→heatwave connectivity loss from f4.

Extreme classes:
  cool-stable      = low THW, low (THW - Tavg)
  cool-amplifying  = low THW, high (THW - Tavg)
  hot-stable       = high THW, low (THW - Tavg)
  hot-amplifying   = high THW, high (THW - Tavg)

Outputs are written to:
  figures/2_results/figures/f4_hw_con_loss/
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
from matplotlib.lines import Line2D
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy.ndimage import label as ndi_label
from shapely.geometry import box

# =============================================================================
# PATHS
# =============================================================================


def infer_scripts_root(start: Path) -> Path:
    """Walk upward from this file to find the repo scripts root (has DATA/ and figures/)."""
    for parent in [start, *start.parents]:
        if (parent / "DATA").exists() and (parent / "figures").exists():
            return parent
    raise FileNotFoundError(f"Could not locate scripts root above {start}")


ANALYSIS_DIR = Path(__file__).resolve().parent  # figures/2_results/figures/f4_hw_con_loss
SCRIPTS_ROOT = infer_scripts_root(ANALYSIS_DIR)
DATA_DIR = SCRIPTS_ROOT / "DATA"
FIGURES_DIR = SCRIPTS_ROOT / "figures" / "2_results" / "figures"

# This script lives directly inside the f4 panel folder; reuse its existing
# tables/ and rasters/ subfolders rather than nesting a new output tree.
WORKDIR = ANALYSIS_DIR
TABLE_DIR = WORKDIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SETTINGS = FIGURES_DIR / "global_plotting_settings.py"

HEATWAVE_INPUTS = {
    "2010": DATA_DIR / "predictions" / "2010" / "20100728",
    "2018": DATA_DIR / "predictions" / "2018" / "20180717",
    "2021": DATA_DIR / "predictions" / "2021" / "20210714",
}
BASELINE_MEAN_INPUT = DATA_DIR / "predictions" / "baseline" / "15cm_July_allday" / "pred_20000715_1000.tif"
BASELINE_P90_INPUT = DATA_DIR / "predictions" / "baseline" / "15cm_July_allday_p90" / "pred_20000715_1000.tif"

MAP_TOLERANCE = 1.0
TOL_LABEL = str(MAP_TOLERANCE).replace(".", "p")
TARGET_DOMAIN_RASTER = WORKDIR / "rasters" / f"p90_loss_target_domain_tree_veg_nwn_pm{TOL_LABEL}deg.tif"
CONNECTIVITY_LOSS_RAW_RASTER = WORKDIR / "rasters" / f"p90_to_heatwave_mean_connectivity_loss_raw_pm{TOL_LABEL}deg.tif"

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
N_CLASSES = 4
CLASS_BREAKS = [0.25, 0.50, 0.75]
CONNECTIVITY = 8

OUTPUT_BASENAME = f"f3_2_extreme_corners_p90_connectivity_loss_pm{TOL_LABEL}deg"
OUT_PROFILE_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_profiles.csv"
OUT_LULC_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_lulc_composition.csv"
OUT_PATCH_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_patch_metrics.csv"
OUT_VALUES_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_loss_values_sampled.csv"

# SVG figure settings.
PANEL_WIDTH_CM = 15.5
PANEL_HEIGHT_CM = 8.4
AXES_RECT = [0.13, 0.24, 0.82, 0.62]
Y_LIM = (0, 70)
Y_TICKS = [0, 10, 20, 30, 40, 50, 60, 70]
BOX_WIDTH = 0.42
X_TICK_SPACING = 0.62
MAX_POINTS_PER_CLASS = 700
RANDOM_SEED = 42

CORNER_CLASSES = {
    "cool_stable": {
        "label": "cool\nstable",
        "long_label": "Cool-stable",
        "temp_bin": 0,
        "offset_bin": 0,
        "color": "#1b9e77",
    },
    "cool_amplifying": {
        "label": "cool\namplifying",
        "long_label": "Cool-amplifying",
        "temp_bin": 0,
        "offset_bin": 3,
        "color": "#7b3294",
    },
    "hot_stable": {
        "label": "hot\nstable",
        "long_label": "Hot-stable",
        "temp_bin": 3,
        "offset_bin": 0,
        "color": "#fdae61",
    },
    "hot_amplifying": {
        "label": "hot\namplifying",
        "long_label": "Hot-amplifying",
        "temp_bin": 3,
        "offset_bin": 3,
        "color": "#d7191c",
    },
}
CORNER_ORDER = ["cool_stable", "cool_amplifying", "hot_stable", "hot_amplifying"]

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
# DATA CLASSES / IO HELPERS
# =============================================================================

@dataclass(frozen=True)
class RasterSurface:
    label: str
    path: Path
    hour_utc: int
    array: np.ndarray
    profile: dict


def check_required_files() -> None:
    required = [
        GLOBAL_SETTINGS,
        TARGET_DOMAIN_RASTER,
        CONNECTIVITY_LOSS_RAW_RASTER,
        BASELINE_MEAN_INPUT,
        BASELINE_P90_INPUT,
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
    matching = [p for p in tif_paths if file_hour_utc(p) == TARGET_HEATWAVE_UTC_HOUR]
    if len(matching) != 1:
        available = ", ".join(f"{file_hour_utc(p):02d}:00 UTC ({p.name})" for p in tif_paths)
        raise FileNotFoundError(f"Expected one {TARGET_HEATWAVE_UTC_HOUR:02d}:00 UTC raster for {label}. Available: {available}")
    path = matching[0]
    arr, profile = read_raster(path)
    print(f"Selected {label}: {path.name} at {file_hour_utc(path):02d}:00 UTC")
    return RasterSurface(label=label, path=path, hour_utc=file_hour_utc(path), array=arr, profile=profile)


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
    raise FileNotFoundError(f"No unique baseline raster for {hour_utc:02d}:00 UTC under {baseline_input}")


def aggregate_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    with np.errstate(all="ignore"):
        if AGGREGATION == "mean":
            return np.nanmean(stack, axis=0)
        if AGGREGATION == "max":
            return np.nanmax(stack, axis=0)
    raise ValueError(f"Unsupported AGGREGATION: {AGGREGATION}")


def compute_temperature_surfaces(target_profile: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    heatwave_arrays = []
    mean_arrays = []
    p90_arrays = []

    for label, input_path in HEATWAVE_INPUTS.items():
        surface = select_heatwave_surface(label, input_path)
        heatwave = reproject_to_match(surface.array, surface.profile, target_profile, resampling=Resampling.bilinear)

        mean_path = matching_baseline_for_hour(BASELINE_MEAN_INPUT, surface.hour_utc)
        mean_arr, mean_profile = read_raster(mean_path)
        tmean = reproject_to_match(mean_arr, mean_profile, target_profile, resampling=Resampling.bilinear)

        p90_path = matching_baseline_for_hour(BASELINE_P90_INPUT, surface.hour_utc)
        p90_arr, p90_profile = read_raster(p90_path)
        tp90 = reproject_to_match(p90_arr, p90_profile, target_profile, resampling=Resampling.bilinear)

        heatwave_arrays.append(heatwave)
        mean_arrays.append(tmean)
        p90_arrays.append(tp90)
        print(f"Matched mean baseline for {label}: {mean_path.name}")
        print(f"Matched p90 baseline for {label}: {p90_path.name}")

    thw = aggregate_arrays(heatwave_arrays)
    tmean = aggregate_arrays(mean_arrays)
    tp90 = aggregate_arrays(p90_arrays)
    hw_minus_mean = thw - tmean
    hw_minus_p90 = thw - tp90
    return thw, tmean, tp90, hw_minus_mean, hw_minus_p90

# =============================================================================
# CLASSIFICATION
# =============================================================================

def rank01(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values)
    ranks = s.rank(method="average").to_numpy(dtype=float)
    n = len(values)
    if n <= 1:
        return np.zeros_like(values, dtype=float)
    return (ranks - 1.0) / (n - 1.0)


def classify_bivariate(thw: np.ndarray, offset: np.ndarray, valid_domain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (
        valid_domain
        & np.isfinite(thw)
        & np.isfinite(offset)
        & (thw >= MIN_VALID_ABSOLUTE_TEMP_C)
    )
    if int(valid.sum()) == 0:
        raise ValueError("No valid pixels for bivariate classification.")

    r_thw = np.full(thw.shape, np.nan, dtype=float)
    r_offset = np.full(offset.shape, np.nan, dtype=float)
    r_thw[valid] = rank01(thw[valid])
    r_offset[valid] = rank01(offset[valid])

    temp_bin = np.full(thw.shape, -1, dtype=np.int8)
    offset_bin = np.full(thw.shape, -1, dtype=np.int8)
    temp_bin[valid] = np.digitize(r_thw[valid], bins=CLASS_BREAKS, right=False)
    offset_bin[valid] = np.digitize(r_offset[valid], bins=CLASS_BREAKS, right=False)
    return temp_bin, offset_bin, valid


def corner_masks(temp_bin: np.ndarray, offset_bin: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    masks = {}
    for key, info in CORNER_CLASSES.items():
        masks[key] = valid & (temp_bin == info["temp_bin"]) & (offset_bin == info["offset_bin"])
        print(f"{info['long_label']}: {int(masks[key].sum()):,} pixels")
    return masks

# =============================================================================
# VECTOR / LULC HELPERS
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

# =============================================================================
# SUMMARY HELPERS
# =============================================================================

def cell_area_m2(profile: dict) -> float:
    return abs(float(profile["transform"].a) * float(profile["transform"].e))


def patch_structure() -> np.ndarray:
    if CONNECTIVITY == 8:
        return np.ones((3, 3), dtype=int)
    if CONNECTIVITY == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    raise ValueError("CONNECTIVITY must be 4 or 8.")


def patch_metrics(mask: np.ndarray, profile: dict) -> dict[str, float]:
    area = cell_area_m2(profile)
    labeled, n_labels = ndi_label(mask, structure=patch_structure())
    if n_labels == 0:
        return {
            "n_patches": 0,
            "median_patch_area_m2": np.nan,
            "largest_patch_area_m2": np.nan,
            "patches_per_km2": np.nan,
        }
    counts = np.bincount(labeled.ravel())[1:]
    areas = counts.astype(float) * area
    total_area_km2 = float(mask.sum() * area / 1_000_000.0)
    return {
        "n_patches": int(n_labels),
        "median_patch_area_m2": float(np.median(areas)),
        "largest_patch_area_m2": float(np.max(areas)),
        "patches_per_km2": float(n_labels / total_area_km2) if total_area_km2 > 0 else np.nan,
    }


def finite_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask]
    return vals[np.isfinite(vals)]


def summarize_profiles(
    masks: dict[str, np.ndarray],
    lulc_masks: dict[str, np.ndarray],
    profile: dict,
    *,
    thw: np.ndarray,
    tmean: np.ndarray,
    tp90: np.ndarray,
    hw_minus_mean: np.ndarray,
    hw_minus_p90: np.ndarray,
    connectivity_loss: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    area = cell_area_m2(profile)
    profile_rows = []
    lulc_rows = []
    patch_rows = []
    value_rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for key in CORNER_ORDER:
        info = CORNER_CLASSES[key]
        mask = masks[key]
        count = int(mask.sum())
        class_area_m2 = count * area

        loss = finite_values(connectivity_loss, mask)
        thw_vals = finite_values(thw, mask)
        tmean_vals = finite_values(tmean, mask)
        tp90_vals = finite_values(tp90, mask)
        dmean_vals = finite_values(hw_minus_mean, mask)
        dp90_vals = finite_values(hw_minus_p90, mask)

        # LULC composition.
        lulc_counts = {}
        for lulc in PLOT_ORDER:
            c = int(np.sum(mask & lulc_masks[lulc]))
            lulc_counts[lulc] = c
            lulc_rows.append({
                "corner_class": key,
                "corner_label": info["long_label"],
                "lulc_class": lulc,
                "lulc_label": DISPLAY_LABEL_MAP.get(lulc, lulc),
                "pixel_count": c,
                "area_m2": c * area,
                "percent_of_corner": 100.0 * c / count if count else np.nan,
            })
        dominant_lulc = max(lulc_counts, key=lulc_counts.get) if count else None
        dominant_pct = 100.0 * lulc_counts[dominant_lulc] / count if count and dominant_lulc else np.nan

        pm = patch_metrics(mask, profile)
        patch_rows.append({"corner_class": key, "corner_label": info["long_label"], **pm})

        profile_rows.append({
            "corner_class": key,
            "corner_label": info["long_label"],
            "temp_bin": info["temp_bin"],
            "offset_bin": info["offset_bin"],
            "pixel_count": count,
            "area_m2": class_area_m2,
            "area_km2": class_area_m2 / 1_000_000.0,
            "dominant_lulc": dominant_lulc,
            "dominant_lulc_percent": dominant_pct,
            "median_heatwave_temp_degC": float(np.nanmedian(thw_vals)) if thw_vals.size else np.nan,
            "mean_heatwave_temp_degC": float(np.nanmean(thw_vals)) if thw_vals.size else np.nan,
            "median_mean_baseline_temp_degC": float(np.nanmedian(tmean_vals)) if tmean_vals.size else np.nan,
            "mean_mean_baseline_temp_degC": float(np.nanmean(tmean_vals)) if tmean_vals.size else np.nan,
            "median_p90_temp_degC": float(np.nanmedian(tp90_vals)) if tp90_vals.size else np.nan,
            "mean_p90_temp_degC": float(np.nanmean(tp90_vals)) if tp90_vals.size else np.nan,
            "median_hw_minus_mean_degC": float(np.nanmedian(dmean_vals)) if dmean_vals.size else np.nan,
            "mean_hw_minus_mean_degC": float(np.nanmean(dmean_vals)) if dmean_vals.size else np.nan,
            "median_hw_minus_p90_degC": float(np.nanmedian(dp90_vals)) if dp90_vals.size else np.nan,
            "mean_hw_minus_p90_degC": float(np.nanmean(dp90_vals)) if dp90_vals.size else np.nan,
            "median_p90_connectivity_loss_percent": float(np.nanmedian(loss)) if loss.size else np.nan,
            "mean_p90_connectivity_loss_percent": float(np.nanmean(loss)) if loss.size else np.nan,
            "q25_p90_connectivity_loss_percent": float(np.nanpercentile(loss, 25)) if loss.size else np.nan,
            "q75_p90_connectivity_loss_percent": float(np.nanpercentile(loss, 75)) if loss.size else np.nan,
            "share_area_loss_gt_10pct": float(np.mean(loss > 10.0)) if loss.size else np.nan,
            "share_area_loss_gt_20pct": float(np.mean(loss > 20.0)) if loss.size else np.nan,
            **pm,
        })

        # Sample values for the compact plot / potential checking.
        if loss.size:
            sample = loss
            if sample.size > MAX_POINTS_PER_CLASS:
                idx = rng.choice(sample.size, size=MAX_POINTS_PER_CLASS, replace=False)
                sample = sample[idx]
            for v in sample:
                value_rows.append({
                    "corner_class": key,
                    "corner_label": info["long_label"],
                    "connectivity_loss_percent": float(v),
                })

    return (
        pd.DataFrame.from_records(profile_rows),
        pd.DataFrame.from_records(lulc_rows),
        pd.DataFrame.from_records(patch_rows),
        pd.DataFrame.from_records(value_rows),
    )

# =============================================================================
# PLOT
# =============================================================================

def save_svg(fig: plt.Figure, basename: str) -> None:
    gps.make_transparent(fig)
    out = WORKDIR / f"{basename}.svg"
    fig.savefig(out, transparent=STYLE.transparent, facecolor="none", edgecolor="none", bbox_inches=None, pad_inches=0)
    width_cm = fig.get_figwidth() * gps.CM_PER_INCH
    height_cm = fig.get_figheight() * gps.CM_PER_INCH
    print(f"[OK] wrote {out} ({width_cm:.2f} × {height_cm:.2f} cm fixed canvas)")


def plot_connectivity_loss(profile_df: pd.DataFrame, values_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(PANEL_WIDTH_CM / gps.CM_PER_INCH, PANEL_HEIGHT_CM / gps.CM_PER_INCH), dpi=STYLE.dpi_export)
    ax = fig.add_axes(AXES_RECT)
    gps.style_axis(ax, STYLE, grid_y=True, grid_x=False)
    ax.grid(axis="x", visible=False)  # style_axis's grid_x=False is overridden by matplotlib when kwargs are passed

    positions = np.arange(1, len(CORNER_ORDER) + 1) * X_TICK_SPACING

    for x, key in zip(positions, CORNER_ORDER):
        info = CORNER_CLASSES[key]
        vals = values_df.loc[values_df["corner_class"] == key, "connectivity_loss_percent"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            bp = ax.boxplot(
                [vals],
                positions=[x],
                widths=BOX_WIDTH,
                showfliers=False,
                patch_artist=True,
                zorder=4,
            )
            for box in bp["boxes"]:
                box.set_facecolor(info["color"])
                box.set_alpha(0.55)
                box.set_edgecolor(STYLE.col_black)
                box.set_linewidth(1.0)
            for element in ("whiskers", "caps"):
                for line in bp[element]:
                    line.set_color(STYLE.col_black)
                    line.set_linewidth(1.0)
            for line in bp["medians"]:
                line.set_color(STYLE.col_black)
                line.set_linewidth(1.7)

    ax.axhline(0, color=STYLE.col_black, linewidth=0.8, alpha=0.55, zorder=1)
    ax.set_xlim(positions[0] - 0.9 * X_TICK_SPACING, positions[-1] + 0.9 * X_TICK_SPACING)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.set_xticks(positions)
    ax.set_xticklabels([CORNER_CLASSES[k]["label"] for k in CORNER_ORDER], fontsize=max(9.0, STYLE.fs_tick * 0.95))
    gps.set_axis_labels(ax, xlabel=None, ylabel="Connectivity loss (%)", style=STYLE)

    save_svg(fig, OUTPUT_BASENAME)
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_global_style()
    print("Resolved global plotting style:")
    print(f"  output dir: {WORKDIR}")
    print(f"  valid domain: {TARGET_DOMAIN_RASTER}")
    print(f"  connectivity loss: {CONNECTIVITY_LOSS_RAW_RASTER}")

    check_required_files()

    target_arr, target_profile = read_raster(TARGET_DOMAIN_RASTER)
    valid_domain = np.isfinite(target_arr) & (target_arr > 0)
    print(f"Valid domain pixels: {int(valid_domain.sum()):,}")

    loss_arr, loss_profile = read_raster(CONNECTIVITY_LOSS_RAW_RASTER)
    if not same_grid(loss_profile, target_profile):
        loss_arr = reproject_to_match(loss_arr, loss_profile, target_profile, resampling=Resampling.bilinear)

    thw, tmean, tp90, hw_minus_mean, hw_minus_p90 = compute_temperature_surfaces(target_profile)
    temp_bin, offset_bin, valid = classify_bivariate(thw, hw_minus_mean, valid_domain)
    masks = corner_masks(temp_bin, offset_bin, valid)

    lulc_masks = load_lulc_masks(target_profile, valid)
    profile_df, lulc_df, patch_df, values_df = summarize_profiles(
        masks,
        lulc_masks,
        target_profile,
        thw=thw,
        tmean=tmean,
        tp90=tp90,
        hw_minus_mean=hw_minus_mean,
        hw_minus_p90=hw_minus_p90,
        connectivity_loss=loss_arr,
    )

    profile_df.to_csv(OUT_PROFILE_CSV, index=False)
    lulc_df.to_csv(OUT_LULC_CSV, index=False)
    patch_df.to_csv(OUT_PATCH_CSV, index=False)
    values_df.to_csv(OUT_VALUES_CSV, index=False)
    print(f"[OK] wrote {OUT_PROFILE_CSV}")
    print(f"[OK] wrote {OUT_LULC_CSV}")
    print(f"[OK] wrote {OUT_PATCH_CSV}")
    print(f"[OK] wrote {OUT_VALUES_CSV}")

    print("\n[EXTREME CLASS PROFILES]")
    cols = [
        "corner_label", "dominant_lulc", "dominant_lulc_percent",
        "median_heatwave_temp_degC", "median_hw_minus_mean_degC",
        "median_p90_connectivity_loss_percent", "share_area_loss_gt_20pct",
        "n_patches", "median_patch_area_m2", "largest_patch_area_m2", "patches_per_km2",
    ]
    print(profile_df[cols].to_string(index=False))

    plot_connectivity_loss(profile_df, values_df)


if __name__ == "__main__":
    main()
