#!/usr/bin/env python3
"""
Figure f4 panel D/E: patch-level mechanism checks for p90 connectivity change.

This standalone script reads the existing f4 p90 connectivity-change raster and
exports two compact two-axis scatter panels:

Panel D:
  left  = connectivity change vs mean heatwave T
  right = connectivity change vs ΔT to p90 baseline (heatwave - p90)

Panel E:
  left  = connectivity change vs mean p90 T
  right = connectivity change vs ΔT to mean baseline (p90 - mean)

Positive connectivity-change values are losses. Negative values are local
current gains / redistribution. The raw, unclipped connectivity-change raster is
used so negative values are retained.
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
from scipy.stats import spearmanr
from shapely.geometry import box

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
FIGURES_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures")
WORKDIR = FIGURES_DIR / "f4"
RASTER_DIR = WORKDIR / "rasters"
TABLE_DIR = WORKDIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SETTINGS = FIGURES_DIR / "global_plotting_settings.py"

HEATWAVE_INPUTS = {
    "2010": DATA_DIR / "predictions" / "2010" / "20100728",
    "2018": DATA_DIR / "predictions" / "2018" / "20180717",
    "2021": DATA_DIR / "predictions" / "2021" / "20210714",
}
BASELINE_P90_INPUT = DATA_DIR / "predictions" / "baseline" / "15cm_July_allday_p90" / "pred_20000715_1000.tif"
BASELINE_MEAN_INPUT = DATA_DIR / "predictions" / "baseline" / "15cm_July_allday" / "pred_20000715_1000.tif"

PERUSPIIRI_PATH = DATA_DIR / "figures" / "offset_figure" / "peruspiiri_WFS.gpkg"
BAREGROUND_VECTOR_PATH = DATA_DIR / "LULC" / "lc_bareground.gpkg"

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


ensure_global_style()

# =============================================================================
# SETTINGS
# =============================================================================

MAP_TOLERANCE = 1.0
LOCAL_UTC_OFFSET_HOURS = 3
EXPECTED_PEAK_LOCAL_HOUR = 13
TARGET_HEATWAVE_UTC_HOUR = (EXPECTED_PEAK_LOCAL_HOUR - LOCAL_UTC_OFFSET_HOURS) % 24
AGGREGATION = "mean"
ASSUME_VECTOR_EPSG_IF_MISSING = 3879

MIN_PATCH_PIXELS = 9
CONNECTIVITY = 8
MIN_VALID_ABSOLUTE_TEMP_C = 15.0

OUTPUT_BASENAME_HW_P90 = "f4_panel_d_connectivity_change_mechanism_patch_scatter_hw_to_p90_pm1p0deg"
OUTPUT_BASENAME_P90_MEAN = "f4_panel_e_connectivity_change_mechanism_patch_scatter_p90_to_mean_pm1p0deg"
# Keep this disabled until the matching mean-baseline -> p90 Omniscape comparison exists.
RUN_P90_TO_MEAN_COMPARISON = False

# Fixed compact canvas. These dimensions are independent of STYLE.panel_width_cm.
FIG_WIDTH_CM = 16.0
FIG_HEIGHT_CM = 7.2
PANEL_AXES_LEFT = [0.090, 0.205, 0.385, 0.610]
PANEL_AXES_RIGHT = [0.555, 0.205, 0.385, 0.610]

POINT_ALPHA = 0.22
MAX_PLOT_PATCHES_PER_CLASS = 180
POINT_SIZE_MIN = 7
POINT_SIZE_MAX = 24
CLASS_MEDIAN_SIZE = 58
DRAW_ZERO_LINE = True
PLOT_YLIM = (-20, 70)
PLOT_YTICKS = [-20, 0, 20, 40, 60]
ABS_TEMP_XLIM = None
OFFSET_XLIM = None

FS_LEGEND_LOCAL = max(6.2, STYLE.fs_legend * 0.62)
FS_AXIS_LOCAL = max(7.2, STYLE.fs_axis * 0.68)
FS_TICK_LOCAL = max(6.7, STYLE.fs_tick * 0.74)
FS_ANNOT_LOCAL = max(6.4, STYLE.fs_legend * 0.68)

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

def tolerance_label(value: float) -> str:
    return str(value).replace(".", "p")


def loss_raster_path() -> Path:
    tol_label = tolerance_label(MAP_TOLERANCE)
    return RASTER_DIR / f"p90_to_heatwave_mean_connectivity_loss_raw_pm{tol_label}deg.tif"


def target_domain_path() -> Path:
    tol_label = tolerance_label(MAP_TOLERANCE)
    return RASTER_DIR / f"p90_loss_target_domain_tree_veg_nwn_pm{tol_label}deg.tif"


def check_required_files() -> None:
    required = [
        GLOBAL_SETTINGS,
        loss_raster_path(),
        target_domain_path(),
        BASELINE_P90_INPUT,
        BASELINE_MEAN_INPUT,
        PERUSPIIRI_PATH,
        BAREGROUND_VECTOR_PATH,
        *HEATWAVE_INPUTS.values(),
    ]
    for _, paths in LULC_LAYERS:
        required.extend(paths)
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print(f"  {p}")
        raise FileNotFoundError("One or more required inputs are missing. Run the f4 connectivity-loss script first.")


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


def select_peak_surface(label: str, input_path: Path) -> RasterSurface:
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


def matching_baseline_for_hour(baseline_input: Path, hour_utc: int, label: str) -> Path:
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
    raise FileNotFoundError(f"No unique {label} baseline raster for {hour_utc:02d}:00 UTC. Available: {available}")


def aggregate_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    with np.errstate(all="ignore"):
        if AGGREGATION == "mean":
            return np.nanmean(stack, axis=0)
        if AGGREGATION == "max":
            return np.nanmax(stack, axis=0)
    raise ValueError(f"Unsupported AGGREGATION: {AGGREGATION}")


def compute_temperature_surfaces(target_profile: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    heatwave_arrays: list[np.ndarray] = []
    p90_arrays: list[np.ndarray] = []
    mean_arrays: list[np.ndarray] = []
    hw_minus_p90_arrays: list[np.ndarray] = []
    p90_minus_mean_arrays: list[np.ndarray] = []

    for label, input_path in HEATWAVE_INPUTS.items():
        surface = select_peak_surface(label, input_path)
        heatwave = reproject_to_match(surface.array, surface.profile, target_profile, resampling=Resampling.bilinear)

        p90_path = matching_baseline_for_hour(BASELINE_P90_INPUT, surface.hour_utc, "p90")
        p90_arr, p90_profile = read_raster(p90_path)
        p90 = reproject_to_match(p90_arr, p90_profile, target_profile, resampling=Resampling.bilinear)

        mean_path = matching_baseline_for_hour(BASELINE_MEAN_INPUT, surface.hour_utc, "mean")
        mean_arr, mean_profile = read_raster(mean_path)
        tmean = reproject_to_match(mean_arr, mean_profile, target_profile, resampling=Resampling.bilinear)

        heatwave_arrays.append(heatwave)
        p90_arrays.append(p90)
        mean_arrays.append(tmean)
        hw_minus_p90_arrays.append(heatwave - p90)
        p90_minus_mean_arrays.append(p90 - tmean)

        print(f"Matched p90 baseline for {label}: {p90_path.name}")
        print(f"Matched mean baseline for {label}: {mean_path.name}")

    agg_heatwave = aggregate_arrays(heatwave_arrays)
    agg_p90 = aggregate_arrays(p90_arrays)
    agg_mean = aggregate_arrays(mean_arrays)
    agg_hw_minus_p90 = aggregate_arrays(hw_minus_p90_arrays)
    agg_p90_minus_mean = aggregate_arrays(p90_minus_mean_arrays)
    return agg_heatwave, agg_p90, agg_mean, agg_hw_minus_p90, agg_p90_minus_mean

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

    print(f"Loaded {label}: {len(gdf)} features")
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


def load_lulc_masks(profile: dict, target_domain: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for class_name, paths in LULC_LAYERS:
        class_mask = np.zeros_like(target_domain, dtype=bool)
        for path in paths:
            gdf = read_vector_fast(path, profile, f"LULC {class_name}", use_bbox=True)
            class_mask |= rasterize_gdf(gdf, profile, all_touched=False)
        class_mask &= target_domain
        masks[class_name] = class_mask
        print(f"LULC {class_name}: {int(class_mask.sum())} target-domain cells")

    exclusive: dict[str, np.ndarray] = {}
    already = np.zeros_like(target_domain, dtype=bool)
    for class_name, _ in LULC_LAYERS:
        m = masks[class_name] & (~already)
        exclusive[class_name] = m
        already |= m
        print(f"LULC {class_name} exclusive cells: {int(m.sum())}")
    return exclusive

# =============================================================================
# PATCH AGGREGATION
# =============================================================================

def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    out = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        out &= np.isfinite(arr)
    return out


def patch_structure() -> np.ndarray:
    if CONNECTIVITY == 8:
        return np.ones((3, 3), dtype=int)
    if CONNECTIVITY == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    raise ValueError("CONNECTIVITY must be 4 or 8.")


def build_patch_table(
    *,
    loss_raw: np.ndarray,
    heatwave: np.ndarray,
    p90: np.ndarray,
    tmean: np.ndarray,
    hw_minus_p90: np.ndarray,
    p90_minus_mean: np.ndarray,
    lulc_masks: dict[str, np.ndarray],
    profile: dict,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    base_valid = (
        finite_mask(loss_raw, heatwave, p90, tmean, hw_minus_p90, p90_minus_mean)
        & (heatwave >= MIN_VALID_ABSOLUTE_TEMP_C)
        & (p90 >= MIN_VALID_ABSOLUTE_TEMP_C)
        & (tmean >= MIN_VALID_ABSOLUTE_TEMP_C)
    )

    structure = patch_structure()
    cell_area_m2 = abs(float(profile["transform"].a) * float(profile["transform"].e))

    for class_name in PLOT_ORDER:
        mask = lulc_masks[class_name] & base_valid
        labeled, n_labels = ndi_label(mask, structure=structure)
        print(f"{class_name}: {n_labels} labelled candidate patches")

        if n_labels == 0:
            continue

        labels = labeled[mask].astype(np.int32, copy=False)
        if labels.size == 0:
            continue

        counts = np.bincount(labels, minlength=n_labels + 1)
        keep_label = counts >= MIN_PATCH_PIXELS
        keep = keep_label[labels]
        if not np.any(keep):
            print(f"  retained 0 patches after MIN_PATCH_PIXELS={MIN_PATCH_PIXELS}")
            continue

        labels = labels[keep]
        tmp = pd.DataFrame({
            "patch_id": labels,
            "connectivity_change_percent": loss_raw[mask][keep].astype(np.float32, copy=False),
            "heatwave_temp_degC": heatwave[mask][keep].astype(np.float32, copy=False),
            "p90_temp_degC": p90[mask][keep].astype(np.float32, copy=False),
            "mean_temp_degC": tmean[mask][keep].astype(np.float32, copy=False),
            "hw_minus_p90_degC": hw_minus_p90[mask][keep].astype(np.float32, copy=False),
            "p90_minus_mean_degC": p90_minus_mean[mask][keep].astype(np.float32, copy=False),
        })

        grouped = tmp.groupby("patch_id", sort=False, observed=True)
        out = pd.DataFrame({
            "class": class_name,
            "patch_id": grouped.size().index.to_numpy(),
            "n_pixels": grouped.size().to_numpy(dtype=np.int32),
            "connectivity_change_mean_percent": grouped["connectivity_change_percent"].mean().to_numpy(),
            "connectivity_change_median_percent": grouped["connectivity_change_percent"].median().to_numpy(),
            "connectivity_change_p25_percent": grouped["connectivity_change_percent"].quantile(0.25).to_numpy(),
            "connectivity_change_p75_percent": grouped["connectivity_change_percent"].quantile(0.75).to_numpy(),
            "heatwave_temp_mean_degC": grouped["heatwave_temp_degC"].mean().to_numpy(),
            "heatwave_temp_median_degC": grouped["heatwave_temp_degC"].median().to_numpy(),
            "p90_temp_mean_degC": grouped["p90_temp_degC"].mean().to_numpy(),
            "p90_temp_median_degC": grouped["p90_temp_degC"].median().to_numpy(),
            "mean_temp_mean_degC": grouped["mean_temp_degC"].mean().to_numpy(),
            "mean_temp_median_degC": grouped["mean_temp_degC"].median().to_numpy(),
            "hw_minus_p90_mean_degC": grouped["hw_minus_p90_degC"].mean().to_numpy(),
            "hw_minus_p90_median_degC": grouped["hw_minus_p90_degC"].median().to_numpy(),
            "p90_minus_mean_mean_degC": grouped["p90_minus_mean_degC"].mean().to_numpy(),
            "p90_minus_mean_median_degC": grouped["p90_minus_mean_degC"].median().to_numpy(),
        })
        out["area_m2"] = out["n_pixels"].astype(float) * cell_area_m2
        records.append(out)
        print(f"  retained {len(out)} patches after MIN_PATCH_PIXELS={MIN_PATCH_PIXELS}")

    if not records:
        raise ValueError("Patch table is empty. Try lowering MIN_PATCH_PIXELS or check LULC masks.")

    return pd.concat(records, ignore_index=True)


def build_class_summary(patch_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_name in PLOT_ORDER:
        sub = patch_df[patch_df["class"] == class_name]
        if sub.empty:
            continue
        rows.append({
            "class": class_name,
            "n_patches": int(len(sub)),
            "total_area_m2": float(sub["area_m2"].sum()),
            "median_patch_area_m2": float(sub["area_m2"].median()),
            "median_connectivity_change_percent": float(sub["connectivity_change_median_percent"].median()),
            "mean_connectivity_change_percent": float(np.average(sub["connectivity_change_mean_percent"], weights=sub["area_m2"])),
            "median_heatwave_temp_degC": float(sub["heatwave_temp_median_degC"].median()),
            "median_p90_temp_degC": float(sub["p90_temp_median_degC"].median()),
            "median_mean_temp_degC": float(sub["mean_temp_median_degC"].median()),
            "median_hw_minus_p90_degC": float(sub["hw_minus_p90_median_degC"].median()),
            "median_p90_minus_mean_degC": float(sub["p90_minus_mean_median_degC"].median()),
        })
    return pd.DataFrame.from_records(rows)


def safe_spearman(df: pd.DataFrame, x: str, y: str, label: str) -> dict[str, object]:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3:
        return {"group": label, "x": x, "y": y, "n": len(sub), "rho": np.nan, "p_value": np.nan}
    rho, p = spearmanr(sub[x].to_numpy(), sub[y].to_numpy())
    return {"group": label, "x": x, "y": y, "n": len(sub), "rho": float(rho), "p_value": float(p)}


def build_spearman_summary(patch_df: pd.DataFrame, class_df: pd.DataFrame) -> pd.DataFrame:
    y = "connectivity_change_median_percent"
    rows = [
        safe_spearman(patch_df, "heatwave_temp_median_degC", y, "panel d: mean heatwave T"),
        safe_spearman(patch_df, "hw_minus_p90_median_degC", y, "panel d: heatwave minus p90 baseline"),
        safe_spearman(patch_df, "p90_temp_median_degC", y, "panel e: mean p90 T"),
        safe_spearman(patch_df, "p90_minus_mean_median_degC", y, "panel e: p90 minus mean baseline"),
    ]

    if len(class_df) >= 3:
        class_tmp = class_df.rename(columns={
            "median_connectivity_change_percent": "connectivity_change_median_percent",
            "median_heatwave_temp_degC": "heatwave_temp_median_degC",
            "median_p90_temp_degC": "p90_temp_median_degC",
            "median_hw_minus_p90_degC": "hw_minus_p90_median_degC",
            "median_p90_minus_mean_degC": "p90_minus_mean_median_degC",
        })
        rows.extend([
            safe_spearman(class_tmp, "heatwave_temp_median_degC", y, "class medians: mean heatwave T"),
            safe_spearman(class_tmp, "hw_minus_p90_median_degC", y, "class medians: heatwave minus p90 baseline"),
            safe_spearman(class_tmp, "p90_temp_median_degC", y, "class medians: mean p90 T"),
            safe_spearman(class_tmp, "p90_minus_mean_median_degC", y, "class medians: p90 minus mean baseline"),
        ])
    return pd.DataFrame.from_records(rows)

# =============================================================================
# PLOTTING
# =============================================================================

def class_colors() -> dict[str, str]:
    return {
        "Trees 2–10 m": getattr(STYLE, "col_blue", "#2b6cb0"),
        "Trees 10–15 m": getattr(STYLE, "col_hist", "#b7791f"),
        "Trees >15 m": getattr(STYLE, "col_red", "#b83232"),
        "Fields": getattr(STYLE, "col_grey", "#777777"),
        "Other vegetation": getattr(STYLE, "col_black", "#222222"),
    }


def scale_sizes(area: pd.Series) -> np.ndarray:
    vals = np.sqrt(area.to_numpy(dtype=float))
    if np.nanmax(vals) <= np.nanmin(vals):
        return np.full_like(vals, 0.5 * (POINT_SIZE_MIN + POINT_SIZE_MAX))
    scaled = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
    return POINT_SIZE_MIN + scaled * (POINT_SIZE_MAX - POINT_SIZE_MIN)


def add_regression_line(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return
    xx = x[valid]
    yy = y[valid]
    try:
        coef = np.polyfit(xx, yy, deg=1)
    except np.linalg.LinAlgError:
        return
    xline = np.linspace(np.nanpercentile(xx, 2), np.nanpercentile(xx, 98), 100)
    yline = coef[0] * xline + coef[1]
    ax.plot(xline, yline, color=color, linewidth=1.25, alpha=0.68, zorder=2)


def annotate_rho(ax, df: pd.DataFrame, xcol: str, ycol: str) -> None:
    sub = df[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3:
        txt = "ρ = NA"
    else:
        rho, _p = spearmanr(sub[xcol].to_numpy(), sub[ycol].to_numpy())
        txt = f"ρ = {rho:.2f}"
    ax.text(
        0.035, 0.945, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=FS_ANNOT_LOCAL,
        fontfamily=STYLE.font_family,
        color=STYLE.col_black,
    )


def plot_sample(patch_df: pd.DataFrame) -> pd.DataFrame:
    if MAX_PLOT_PATCHES_PER_CLASS is None:
        return patch_df
    rng = np.random.default_rng(42)
    parts = []
    for class_name in PLOT_ORDER:
        sub = patch_df[patch_df["class"] == class_name].copy()
        if len(sub) <= MAX_PLOT_PATCHES_PER_CLASS:
            parts.append(sub)
        else:
            idx = rng.choice(sub.index.to_numpy(), size=MAX_PLOT_PATCHES_PER_CLASS, replace=False)
            parts.append(sub.loc[idx])
    return pd.concat(parts, ignore_index=True) if parts else patch_df.iloc[0:0].copy()


def new_compact_figure() -> plt.Figure:
    return plt.figure(figsize=(FIG_WIDTH_CM / gps.CM_PER_INCH, FIG_HEIGHT_CM / gps.CM_PER_INCH), dpi=STYLE.dpi_export)


def save_svg(fig: plt.Figure, basename: str) -> None:
    gps.make_transparent(fig)
    out = WORKDIR / f"{basename}.svg"
    fig.savefig(out, transparent=STYLE.transparent, facecolor="none", edgecolor="none", bbox_inches=None, pad_inches=0)
    width_cm = fig.get_figwidth() * gps.CM_PER_INCH
    height_cm = fig.get_figheight() * gps.CM_PER_INCH
    print(f"[OK] wrote {out} ({width_cm:.2f} × {height_cm:.2f} cm fixed canvas)")


def class_median_x(row: pd.Series, xcol: str) -> float:
    mapping = {
        "heatwave_temp_median_degC": "median_heatwave_temp_degC",
        "hw_minus_p90_median_degC": "median_hw_minus_p90_degC",
        "p90_temp_median_degC": "median_p90_temp_degC",
        "p90_minus_mean_median_degC": "median_p90_minus_mean_degC",
    }
    return float(row[mapping[xcol]])


def plot_scatter_panel(
    ax,
    patch_df: pd.DataFrame,
    class_df: pd.DataFrame,
    *,
    xcol: str,
    xlabel: str,
    show_ylabel: bool,
    xlim: tuple[float, float] | None = None,
) -> None:
    colors = class_colors()
    ycol = "connectivity_change_median_percent"

    gps.style_axis(ax, STYLE, grid_y=True, grid_x=True)
    ax.tick_params(axis="both", labelsize=FS_TICK_LOCAL)
    if DRAW_ZERO_LINE:
        ax.axhline(0, color=STYLE.col_black, linewidth=0.75, alpha=0.55, zorder=1)

    for class_name in PLOT_ORDER:
        sub = patch_df[patch_df["class"] == class_name]
        if sub.empty:
            continue
        ax.scatter(
            sub[xcol],
            sub[ycol],
            s=scale_sizes(sub["area_m2"]),
            color=colors[class_name],
            alpha=POINT_ALPHA,
            linewidths=0,
            zorder=3,
        )

    for _, row in class_df.iterrows():
        class_name = row["class"]
        ax.scatter(
            [class_median_x(row, xcol)],
            [row["median_connectivity_change_percent"]],
            s=CLASS_MEDIAN_SIZE,
            facecolor=colors.get(class_name, STYLE.col_black),
            edgecolor=STYLE.col_black,
            linewidth=0.65,
            alpha=0.98,
            zorder=5,
        )

    add_regression_line(ax, patch_df[xcol].to_numpy(dtype=float), patch_df[ycol].to_numpy(dtype=float), STYLE.col_black)
    annotate_rho(ax, patch_df, xcol, ycol)

    ax.set_xlabel(xlabel, fontsize=FS_AXIS_LOCAL, fontfamily=STYLE.font_family, labelpad=3.5)
    if show_ylabel:
        ax.set_ylabel("Connectivity loss (%)", fontsize=FS_AXIS_LOCAL, fontfamily=STYLE.font_family, labelpad=4)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(*PLOT_YLIM)
    ax.set_yticks(PLOT_YTICKS)
    if xlim is not None:
        ax.set_xlim(*xlim)


def plot_scatter_variant(
    patch_df: pd.DataFrame,
    class_df: pd.DataFrame,
    *,
    left_xcol: str,
    left_xlabel: str,
    right_xcol: str,
    right_xlabel: str,
    output_basename: str,
) -> None:
    plot_df = plot_sample(patch_df)
    print(f"[PLOT] drawing {len(plot_df):,} sampled patches of {len(patch_df):,} total patches")

    fig = new_compact_figure()
    ax_left = fig.add_axes(PANEL_AXES_LEFT)
    ax_right = fig.add_axes(PANEL_AXES_RIGHT)

    plot_scatter_panel(
        ax_left,
        plot_df,
        class_df,
        xcol=left_xcol,
        xlabel=left_xlabel,
        show_ylabel=True,
        xlim=ABS_TEMP_XLIM,
    )
    plot_scatter_panel(
        ax_right,
        plot_df,
        class_df,
        xcol=right_xcol,
        xlabel=right_xlabel,
        show_ylabel=False,
        xlim=OFFSET_XLIM,
    )

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o", linestyle="None", markersize=4.6,
            markerfacecolor=class_colors()[name],
            markeredgecolor=STYLE.col_black,
            markeredgewidth=0.4,
            label=DISPLAY_LABEL_MAP.get(name, name),
        )
        for name in PLOT_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=5,
        frameon=False,
        fontsize=FS_LEGEND_LOCAL,
        handletextpad=0.22,
        columnspacing=0.50,
        borderaxespad=0.0,
        labelspacing=0.20,
    )

    save_svg(fig, output_basename)
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_global_style()
    print("Resolved global plotting style:")
    print(f"  global file: {GLOBAL_SETTINGS}")
    print(f"  output dir:  {WORKDIR}")
    print(f"  canvas:      {FIG_WIDTH_CM} × {FIG_HEIGHT_CM} cm")
    print(f"  tolerance:   ±{MAP_TOLERANCE:g} °C")
    print(f"  analysis hour: {EXPECTED_PEAK_LOCAL_HOUR:02d}:00 local = {TARGET_HEATWAVE_UTC_HOUR:02d}:00 UTC")
    print(f"  min patch:   {MIN_PATCH_PIXELS} pixels")
    print("  loss raster: raw values; negative values are local current gains")

    check_required_files()

    loss_raw, loss_profile = read_raster(loss_raster_path())
    target_domain_arr, target_profile = read_raster(target_domain_path())
    if not same_grid(loss_profile, target_profile):
        target_domain_arr = reproject_to_match(target_domain_arr, target_profile, loss_profile, resampling=Resampling.nearest)
    target_domain = np.isfinite(target_domain_arr) & (target_domain_arr > 0)

    heatwave, p90, tmean, hw_minus_p90, p90_minus_mean = compute_temperature_surfaces(loss_profile)
    lulc_masks = load_lulc_masks(loss_profile, target_domain)

    patch_df = build_patch_table(
        loss_raw=loss_raw,
        heatwave=heatwave,
        p90=p90,
        tmean=tmean,
        hw_minus_p90=hw_minus_p90,
        p90_minus_mean=p90_minus_mean,
        lulc_masks=lulc_masks,
        profile=loss_profile,
    )
    class_df = build_class_summary(patch_df)
    rho_df = build_spearman_summary(patch_df, class_df)

    tol_label = tolerance_label(MAP_TOLERANCE)
    patch_csv = TABLE_DIR / f"f4_panel_d_patch_mechanism_dual_summary_pm{tol_label}deg.csv"
    class_csv = TABLE_DIR / f"f4_panel_d_class_mechanism_dual_summary_pm{tol_label}deg.csv"
    rho_csv = TABLE_DIR / f"f4_panel_d_spearman_dual_summary_pm{tol_label}deg.csv"

    patch_df.to_csv(patch_csv, index=False)
    class_df.to_csv(class_csv, index=False)
    rho_df.to_csv(rho_csv, index=False)

    print("\n[CLASS SUMMARY]")
    print(class_df.to_string(index=False))
    print("\n[SPEARMAN SUMMARY]")
    print(rho_df.to_string(index=False))
    print(f"\n[OK] wrote {patch_csv}")
    print(f"[OK] wrote {class_csv}")
    print(f"[OK] wrote {rho_csv}")

    plot_scatter_variant(
        patch_df,
        class_df,
        left_xcol="heatwave_temp_median_degC",
        left_xlabel="Mean heatwave T (°C)",
        right_xcol="hw_minus_p90_median_degC",
        right_xlabel="ΔT to p90 baseline (°C)",
        output_basename=OUTPUT_BASENAME_HW_P90,
    )

    plot_scatter_variant(
        patch_df,
        class_df,
        left_xcol="p90_temp_median_degC",
        left_xlabel="Mean p90 T (°C)",
        right_xcol="p90_minus_mean_median_degC",
        right_xlabel="ΔT to mean baseline (°C)",
        output_basename=OUTPUT_BASENAME_P90_MEAN,
    )


if __name__ == "__main__":
    main()
