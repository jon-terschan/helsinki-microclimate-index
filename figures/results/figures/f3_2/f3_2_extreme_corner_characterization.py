#!/usr/bin/env python3
"""
Figure f3_2 companion: characterize the four extreme bivariate heatwave-response classes.

This script uses the same valid vegetated prediction domain and same bivariate
classification logic as the f3_2 4×4 heatwave-response map:

  x axis: absolute mean heatwave temperature, THW
  y axis: heatwave amplification from average conditions, THW - Tavg

It extracts only the four corner classes:
  1  = low THW, low ΔT to mean       (cool-stable corner)
  4  = high THW, low ΔT to mean      (hot-stable corner)
  13 = low THW, high ΔT to mean      (cool-amplifying corner)
  16 = high THW, high ΔT to mean     (hot-amplifying corner)

Outputs are descriptive tables plus one optional stacked composition SVG. It does
not include connectivity loss; that is intended as a separate final step.
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
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy.ndimage import label as ndi_label
from shapely.geometry import box

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
FIGURES_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures")
WORKDIR = FIGURES_DIR / "f3_2"
WORKDIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = WORKDIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SETTINGS = FIGURES_DIR / "global_plotting_settings.py"

HEATWAVE_INPUTS = {
    "2010": DATA_DIR / "predictions" / "2010" / "20100728",
    "2018": DATA_DIR / "predictions" / "2018" / "20180717",
    "2021": DATA_DIR / "predictions" / "2021" / "20180714",  # corrected below if missing
}
# Keep the intended 2021 path explicit in case the typo line above is edited accidentally.
HEATWAVE_INPUTS["2021"] = DATA_DIR / "predictions" / "2021" / "20210714"

BASELINE_MEAN_INPUT = DATA_DIR / "predictions" / "baseline" / "15cm_July_allday" / "pred_20000715_1000.tif"
BASELINE_P90_INPUT = DATA_DIR / "predictions" / "baseline" / "15cm_July_allday_p90" / "pred_20000715_1000.tif"

MAP_TOLERANCE = 1.0
TOL_LABEL = str(MAP_TOLERANCE).replace(".", "p")
TARGET_DOMAIN_RASTER = (
    FIGURES_DIR / "f4" / "rasters" / f"p90_loss_target_domain_tree_veg_nwn_pm{TOL_LABEL}deg.tif"
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
LULC_COLORS = {
    "Trees 2–10 m": "#1966D2",
    "Trees 10–15 m": "#1B5E20",
    "Trees >15 m": "#F5B041",
    "Fields": "#C62828",
    "Other vegetation": "#6A1B9A",
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
EXPORT_SVG = True

OUTPUT_BASENAME = f"f3_2_extreme_corner_characterization_mean_bivar4x4_pm{TOL_LABEL}deg"
OUT_PROFILE_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_profiles.csv"
OUT_LULC_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_lulc_composition.csv"
OUT_PATCH_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_patch_metrics.csv"
OUT_SVG = WORKDIR / f"{OUTPUT_BASENAME}_lulc_composition.svg"

EXTREME_CLASSES = {
    1: {
        "corner": "cool-stable",
        "label": "Cool-stable",
        "temp_bin": "low",
        "amplification_bin": "low",
        "description": "low heatwave T, low ΔT to mean",
    },
    4: {
        "corner": "hot-stable",
        "label": "Hot-stable",
        "temp_bin": "high",
        "amplification_bin": "low",
        "description": "high heatwave T, low ΔT to mean",
    },
    13: {
        "corner": "cool-amplifying",
        "label": "Cool-amplifying",
        "temp_bin": "low",
        "amplification_bin": "high",
        "description": "low heatwave T, high ΔT to mean",
    },
    16: {
        "corner": "hot-amplifying",
        "label": "Hot-amplifying",
        "temp_bin": "high",
        "amplification_bin": "high",
        "description": "high heatwave T, high ΔT to mean",
    },
}
CORNER_ORDER = [1, 13, 4, 16]

PANEL_WIDTH_CM = 15.0
PANEL_HEIGHT_CM = 7.0
AX_RECT = [0.12, 0.24, 0.82, 0.58]

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
# RASTER HELPERS
# =============================================================================

@dataclass(frozen=True)
class RasterSurface:
    label: str
    path: Path
    hour_utc: int
    array: np.ndarray
    profile: dict


def check_required_files() -> None:
    required = [GLOBAL_SETTINGS, TARGET_DOMAIN_RASTER, BASELINE_MEAN_INPUT, BASELINE_P90_INPUT, *HEATWAVE_INPUTS.values()]
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
    return a["crs"] == b["crs"] and a["transform"] == b["transform"] and a["width"] == b["width"] and a["height"] == b["height"]


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
    chosen = matching[0]
    arr, profile = read_raster(chosen)
    print(f"Selected {label}: {chosen.name}")
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
    raise FileNotFoundError(f"No unique baseline raster for {hour_utc:02d}:00 UTC in {baseline_input}")


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
        print(f"Matched baselines for {label}: mean={mean_path.name}, p90={p90_path.name}")

    thw = aggregate_arrays(heatwave_arrays)
    tmean = aggregate_arrays(mean_arrays)
    tp90 = aggregate_arrays(p90_arrays)
    dmean = thw - tmean
    dp90 = thw - tp90
    return thw, tmean, tp90, dmean, dp90

# =============================================================================
# RANKING / CLASSIFICATION
# =============================================================================

def rank01(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values)
    ranks = s.rank(method="average").to_numpy(dtype=float)
    n = len(values)
    if n <= 1:
        return np.zeros_like(values, dtype=float)
    return (ranks - 1.0) / (n - 1.0)


def classify_bivar(thw: np.ndarray, dmean: np.ndarray, valid_domain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = valid_domain & np.isfinite(thw) & np.isfinite(dmean) & (thw >= MIN_VALID_ABSOLUTE_TEMP_C)
    if int(valid.sum()) == 0:
        raise ValueError("No valid pixels for bivariate classification.")
    r_thw = np.full_like(thw, np.nan, dtype=float)
    r_dmean = np.full_like(dmean, np.nan, dtype=float)
    r_thw[valid] = rank01(thw[valid])
    r_dmean[valid] = rank01(dmean[valid])
    temp_cls = np.full(thw.shape, -1, dtype=np.int8)
    amp_cls = np.full(thw.shape, -1, dtype=np.int8)
    temp_cls[valid] = np.digitize(r_thw[valid], bins=CLASS_BREAKS, right=False)
    amp_cls[valid] = np.digitize(r_dmean[valid], bins=CLASS_BREAKS, right=False)
    bivar = np.zeros(thw.shape, dtype=np.uint8)
    bivar[valid] = (amp_cls[valid] * N_CLASSES + temp_cls[valid] + 1).astype(np.uint8)
    return bivar, r_thw, r_dmean

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
        crs = info.crs or f"EPSG:{ASSUME_VECTOR_EPSG_IF_MISSING}"
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
        print(f"WARNING: {label} empty after bbox read; retrying full layer.")
        gdf = gpd.read_file(path)
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
# CHARACTERIZATION
# =============================================================================

def cell_area_m2(profile: dict) -> float:
    return abs(float(profile["transform"].a) * float(profile["transform"].e))


def patch_structure() -> np.ndarray:
    if CONNECTIVITY == 8:
        return np.ones((3, 3), dtype=int)
    if CONNECTIVITY == 4:
        return np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)
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
    patch_areas = counts.astype(float) * area
    total_area_km2 = mask.sum() * area / 1_000_000.0
    return {
        "n_patches": int(n_labels),
        "median_patch_area_m2": float(np.nanmedian(patch_areas)),
        "largest_patch_area_m2": float(np.nanmax(patch_areas)),
        "patches_per_km2": float(n_labels / total_area_km2) if total_area_km2 > 0 else np.nan,
    }


def safe_median(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.nanmedian(vals)) if vals.size else np.nan


def safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.nanmean(vals)) if vals.size else np.nan


def characterize_extremes(
    bivar: np.ndarray,
    thw: np.ndarray,
    tmean: np.ndarray,
    tp90: np.ndarray,
    dmean: np.ndarray,
    dp90: np.ndarray,
    r_thw: np.ndarray,
    r_dmean: np.ndarray,
    lulc_masks: dict[str, np.ndarray],
    profile: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    area = cell_area_m2(profile)
    profile_rows = []
    lulc_rows = []
    patch_rows = []

    for class_id in CORNER_ORDER:
        info = EXTREME_CLASSES[class_id]
        mask = bivar == class_id
        count = int(mask.sum())
        total_area_m2 = count * area

        # LULC composition within the corner class.
        comp_counts: dict[str, int] = {}
        for lulc in PLOT_ORDER:
            c = int(np.sum(mask & lulc_masks[lulc]))
            comp_counts[lulc] = c
            lulc_rows.append({
                "corner_id": class_id,
                "corner": info["corner"],
                "corner_label": info["label"],
                "description": info["description"],
                "lulc_class": lulc,
                "lulc_label": DISPLAY_LABEL_MAP.get(lulc, lulc),
                "pixel_count": c,
                "area_m2": c * area,
                "percent_of_corner": 100.0 * c / count if count else np.nan,
            })
        dominant_lulc = max(comp_counts, key=comp_counts.get) if comp_counts else None
        dominant_percent = 100.0 * comp_counts[dominant_lulc] / count if dominant_lulc and count else np.nan

        pm = patch_metrics(mask, profile)
        patch_rows.append({
            "corner_id": class_id,
            "corner": info["corner"],
            "corner_label": info["label"],
            "description": info["description"],
            "pixel_count": count,
            "total_area_m2": total_area_m2,
            **pm,
        })

        row = {
            "corner_id": class_id,
            "corner": info["corner"],
            "corner_label": info["label"],
            "description": info["description"],
            "temp_bin": info["temp_bin"],
            "amplification_bin": info["amplification_bin"],
            "pixel_count": count,
            "total_area_m2": total_area_m2,
            "total_area_km2": total_area_m2 / 1_000_000.0,
            "dominant_lulc": dominant_lulc,
            "dominant_lulc_label": DISPLAY_LABEL_MAP.get(dominant_lulc, dominant_lulc) if dominant_lulc else None,
            "dominant_lulc_percent": dominant_percent,
            "median_heatwave_temp_degC": safe_median(thw, mask),
            "mean_heatwave_temp_degC": safe_mean(thw, mask),
            "median_mean_baseline_temp_degC": safe_median(tmean, mask),
            "mean_mean_baseline_temp_degC": safe_mean(tmean, mask),
            "median_hw_minus_mean_degC": safe_median(dmean, mask),
            "mean_hw_minus_mean_degC": safe_mean(dmean, mask),
            "median_p90_baseline_temp_degC": safe_median(tp90, mask),
            "mean_p90_baseline_temp_degC": safe_mean(tp90, mask),
            "median_hw_minus_p90_degC": safe_median(dp90, mask),
            "mean_hw_minus_p90_degC": safe_mean(dp90, mask),
            "median_heatwave_rank": safe_median(r_thw, mask),
            "median_amplification_rank": safe_median(r_dmean, mask),
            **pm,
        }
        for lulc in PLOT_ORDER:
            row[f"pct_{lulc.replace(' ', '_').replace('–', '-').replace('>', 'gt').replace('<', 'lt')}"] = (
                100.0 * comp_counts[lulc] / count if count else np.nan
            )
        profile_rows.append(row)

    return (
        pd.DataFrame.from_records(profile_rows),
        pd.DataFrame.from_records(lulc_rows),
        pd.DataFrame.from_records(patch_rows),
    )

# =============================================================================
# PLOT
# =============================================================================

def save_svg(fig: plt.Figure, path: Path) -> None:
    gps.make_transparent(fig)
    fig.savefig(path, transparent=STYLE.transparent, facecolor="none", edgecolor="none", bbox_inches=None, pad_inches=0)
    width_cm = fig.get_figwidth() * gps.CM_PER_INCH
    height_cm = fig.get_figheight() * gps.CM_PER_INCH
    print(f"[OK] wrote {path} ({width_cm:.2f} × {height_cm:.2f} cm fixed canvas)")


def plot_lulc_composition(lulc_df: pd.DataFrame) -> None:
    if not EXPORT_SVG:
        return
    fig = plt.figure(figsize=(PANEL_WIDTH_CM / gps.CM_PER_INCH, PANEL_HEIGHT_CM / gps.CM_PER_INCH), dpi=STYLE.dpi_export)
    ax = fig.add_axes(AX_RECT)
    gps.style_axis(ax, STYLE, grid_y=False, grid_x=True)

    corners = [EXTREME_CLASSES[c]["label"] for c in CORNER_ORDER]
    y = np.arange(len(corners), dtype=float)
    left = np.zeros(len(corners), dtype=float)
    for lulc in PLOT_ORDER:
        vals = []
        for class_id in CORNER_ORDER:
            sub = lulc_df[(lulc_df["corner_id"] == class_id) & (lulc_df["lulc_class"] == lulc)]
            vals.append(float(sub["percent_of_corner"].iloc[0]) if not sub.empty else 0.0)
        vals_arr = np.array(vals)
        ax.barh(y, vals_arr, left=left, color=LULC_COLORS[lulc], edgecolor="none", height=0.62, label=DISPLAY_LABEL_MAP.get(lulc, lulc))
        left += vals_arr

    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(corners, fontsize=STYLE.fs_tick, fontfamily=STYLE.font_family)
    ax.invert_yaxis()
    gps.set_axis_labels(ax, xlabel="Land-cover composition of corner class (%)", ylabel=None, style=STYLE)

    handles = [Patch(facecolor=LULC_COLORS[c], edgecolor="none", label=DISPLAY_LABEL_MAP.get(c, c)) for c in PLOT_ORDER]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.965),
        ncol=5,
        frameon=False,
        fontsize=max(7, STYLE.fs_legend * 0.78),
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    save_svg(fig, OUT_SVG)
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_global_style()
    print("Resolved style and paths:")
    print(f"  output dir: {WORKDIR}")
    print(f"  target domain: {TARGET_DOMAIN_RASTER}")
    print(f"  analysis hour: {EXPECTED_PEAK_LOCAL_HOUR:02d}:00 local = {TARGET_HEATWAVE_UTC_HOUR:02d}:00 UTC")
    print("  NOTE: connectivity loss is intentionally not included in this characterization step.")

    check_required_files()
    target_arr, target_profile = read_raster(TARGET_DOMAIN_RASTER)
    valid_domain = np.isfinite(target_arr) & (target_arr > 0)
    print(f"Valid domain pixels: {int(valid_domain.sum()):,}")

    thw, tmean, tp90, dmean, dp90 = compute_temperature_surfaces(target_profile)
    bivar, r_thw, r_dmean = classify_bivar(thw, dmean, valid_domain)
    lulc_masks = load_lulc_masks(target_profile, valid_domain & (bivar > 0))

    profile_df, lulc_df, patch_df = characterize_extremes(
        bivar=bivar,
        thw=thw,
        tmean=tmean,
        tp90=tp90,
        dmean=dmean,
        dp90=dp90,
        r_thw=r_thw,
        r_dmean=r_dmean,
        lulc_masks=lulc_masks,
        profile=target_profile,
    )

    profile_df.to_csv(OUT_PROFILE_CSV, index=False)
    lulc_df.to_csv(OUT_LULC_CSV, index=False)
    patch_df.to_csv(OUT_PATCH_CSV, index=False)
    print(f"[OK] wrote {OUT_PROFILE_CSV}")
    print(f"[OK] wrote {OUT_LULC_CSV}")
    print(f"[OK] wrote {OUT_PATCH_CSV}")

    print("\n[EXTREME CORNER PROFILES]")
    display_cols = [
        "corner_label", "dominant_lulc_label", "dominant_lulc_percent",
        "median_heatwave_temp_degC", "median_hw_minus_mean_degC",
        "median_p90_baseline_temp_degC", "median_hw_minus_p90_degC",
        "n_patches", "median_patch_area_m2", "largest_patch_area_m2", "patches_per_km2",
    ]
    print(profile_df[display_cols].to_string(index=False))

    plot_lulc_composition(lulc_df)


if __name__ == "__main__":
    main()
