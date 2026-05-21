#!/usr/bin/env python3
"""
Figure f3_1: two separate LULC result panels.

Main changes in this version
----------------------------
- Keeps the simpler rasterization workflow for robustness. Mixed-pixel
  exclusion is not enforced here, because a strict full-coverage approach would
  add extra complexity and runtime.
- Panel A uses three fully opaque, same-width overlaid bar series in this draw
  order: HW first, then T p90, then T mean.
- Panel B starts the y-axis at 0, shows full x-axis labels with meters, removes
  the legend, and instead adds very transparent per-category ribbons showing
  the boxplot whisker ranges for the two offset baselines, labelled
  "from p90" and "from mean".
- No n-labels are drawn.
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
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FuncFormatter, MultipleLocator
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from shapely.geometry import box

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
FIGURES_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures")
WORKDIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures\f3_1\new")
WORKDIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SETTINGS = FIGURES_DIR / "global_plotting_settings.py"
PERUSPIIRI_PATH = DATA_DIR / "figures" / "offset_figure" / "peruspiiri_WFS.gpkg"

HEATWAVE_INPUTS = {
    "2010": DATA_DIR / "predictions" / "2010" / "20100728",
    "2018": DATA_DIR / "predictions" / "2018" / "20180717",
    "2021": DATA_DIR / "predictions" / "2021" / "20210714",
}
BASELINE_MEAN_INPUT = DATA_DIR / "predictions" / "baseline" / "15cm_July_allday" / "pred_20000715_1000.tif"
BASELINE_P90_INPUT = DATA_DIR / "predictions" / "baseline" / "15cm_July_allday_p90" / "pred_20000715_1000.tif"

TREE_PATH = DATA_DIR / "predictorstack" / "TREE_FRAC_10m.tif"
NWN_PATH = DATA_DIR / "predictorstack" / "NWN_FRAC_10m.tif"
WATER_PATH = DATA_DIR / "predictorstack" / "WATER_FRAC_10m_Helsinki.tif"
OCEAN_PATH = DATA_DIR / "predictorstack" / "OCEAN_FRAC_10m_Helsinki.tif"
IMPERVIOUS_PATH = DATA_DIR / "predictorstack" / "IMPERV_FRAC_10m_Helsinki.tif"
BUILDING_PATH = DATA_DIR / "predictorstack" / "BLDG_FRAC_10m.tif"
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
# SETTINGS
# =============================================================================

LOCAL_UTC_OFFSET_HOURS = 3
EXPECTED_PEAK_LOCAL_HOUR = 13
TARGET_HEATWAVE_UTC_HOUR = (EXPECTED_PEAK_LOCAL_HOUR - LOCAL_UTC_OFFSET_HOURS) % 24
AGGREGATION = "mean"

VEGETATION_MIN_FRACTION = 0.05
WATER_MIN_FRACTION = 0.05
BUILDING_EXCLUDE_FRACTION = 0.0
IMPERVIOUS_EXCLUDE_FRACTION = 0.5
ASSUME_VECTOR_EPSG_IF_MISSING = 3879

PANEL_AXES_RECT = [0.11, 0.22, 0.84, 0.70]
MAX_VIOLIN_POINTS_PER_CLASS = 60000
RANDOM_SEED = 42
DISPLAY_CLIP_PERCENTILES = (1.0, 99.0)
MIN_VALID_ABSOLUTE_TEMP_C = 15.0
VIOLIN_WIDTH_BACK = 0.78
VIOLIN_WIDTH_FRONT = 0.56
BAR_WIDTH = 0.58
ABS_VIOLIN_OFFSET = 0.24
ABS_VIOLIN_WIDTH = 0.22
RIBBON_WIDTH_BACK = 0.86
RIBBON_WIDTH_FRONT = 0.62
RIBBON_ALPHA_BACK = 0.26
RIBBON_ALPHA_FRONT = 0.28
RIBBON_LABEL_FS = 19

OUTPUT_PANEL_A = "f3_1_panel_a_lulc_absolute_temperatures_v19_1300local"
OUTPUT_PANEL_B = "f3_1_panel_b_lulc_offset_violins_v19_1300local"
OUTPUT_PANEL_C = "f3_1_panel_c_lulc_absolute_temperature_violins_v19_1300local"

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
# HELPERS
# =============================================================================

def check_required_files() -> None:
    required = [
        GLOBAL_SETTINGS,
        BASELINE_MEAN_INPUT,
        BASELINE_P90_INPUT,
        PERUSPIIRI_PATH,
        TREE_PATH,
        NWN_PATH,
        WATER_PATH,
        OCEAN_PATH,
        IMPERVIOUS_PATH,
        BUILDING_PATH,
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
        raise FileNotFoundError("One or more required inputs are missing.")


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = src.profile.copy()
        profile.update(crs=src.crs, transform=src.transform, width=src.width, height=src.height, nodata=np.nan)
    return arr, profile


def reproject_to_match(arr: np.ndarray, src_profile: dict, dst_profile: dict, *, resampling=Resampling.bilinear) -> np.ndarray:
    if (
        src_profile["crs"] == dst_profile["crs"]
        and src_profile["transform"] == dst_profile["transform"]
        and src_profile["width"] == dst_profile["width"]
        and src_profile["height"] == dst_profile["height"]
    ):
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


def read_and_match_fraction(path: Path, target_profile: dict, label: str) -> np.ndarray:
    arr, profile = read_raster(path)
    arr = reproject_to_match(arr, profile, target_profile, resampling=Resampling.bilinear)
    out = np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)
    print(f"Loaded {label}: min={np.nanmin(out):.3f}, max={np.nanmax(out):.3f}")
    return out


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


def spatial_mean(path: Path) -> float:
    arr, _ = read_raster(path)
    return float(np.nanmean(arr))


def select_peak_surface(label: str, input_path: Path) -> RasterSurface:
    """Select the 13:00 local heatwave raster for this event.

    File timestamps are interpreted as UTC. For Helsinki summer time in these
    events, LOCAL_UTC_OFFSET_HOURS = 3, so 13:00 local corresponds to 10:00 UTC.
    The function now selects the matching UTC-hour file explicitly instead of
    choosing the spatially warmest file from the folder.
    """
    tif_paths = discover_tifs(input_path)
    if not tif_paths:
        raise FileNotFoundError(f"No GeoTIFF files found for {label}: {input_path}")

    matching = [p for p in tif_paths if file_hour_utc(p) == TARGET_HEATWAVE_UTC_HOUR]
    if len(matching) == 1:
        chosen = matching[0]
    elif len(matching) > 1:
        raise ValueError(
            f"Multiple {TARGET_HEATWAVE_UTC_HOUR:02d}:00 UTC rasters found for {label}: "
            + ", ".join(str(p) for p in matching)
        )
    else:
        available = ", ".join(f"{file_hour_utc(p):02d}:00 UTC ({p.name})" for p in tif_paths)
        raise FileNotFoundError(
            f"No raster found for {label} at {TARGET_HEATWAVE_UTC_HOUR:02d}:00 UTC "
            f"({EXPECTED_PEAK_LOCAL_HOUR:02d}:00 local). Available: {available}"
        )

    hour = file_hour_utc(chosen)
    local_hour = (hour + LOCAL_UTC_OFFSET_HOURS) % 24
    if local_hour != EXPECTED_PEAK_LOCAL_HOUR:
        raise ValueError(
            f"Selected {label} is {hour:02d}:00 UTC ({local_hour:02d}:00 local), "
            f"not {EXPECTED_PEAK_LOCAL_HOUR:02d}:00 local."
        )

    arr, profile = read_raster(chosen)
    print(f"Selected {label}: {chosen} at {hour:02d}:00 UTC ({local_hour:02d}:00 local)")
    return RasterSurface(label=label, path=chosen, hour_utc=hour, array=arr, profile=profile)


def matching_baseline_for_hour(baseline_input: Path, hour_utc: int) -> Path:
    """Return the baseline raster for exactly the same UTC hour.

    This is strict by design: a missing 10:00 UTC baseline should fail rather
    than silently falling back to another hour.
    """
    candidates = discover_tifs(baseline_input)
    same_hour = [p for p in candidates if file_hour_utc(p) == hour_utc]
    if same_hour:
        if len(same_hour) > 1:
            raise ValueError(
                f"Multiple baseline rasters found for {hour_utc:02d}:00 UTC: "
                + ", ".join(str(p) for p in same_hour)
            )
        return same_hour[0]

    if baseline_input.is_file() and baseline_input.parent.exists():
        parent_candidates = sorted(baseline_input.parent.glob("*.tif")) + sorted(baseline_input.parent.glob("*.tiff"))
        same_hour = [p for p in parent_candidates if file_hour_utc(p) == hour_utc]
        if same_hour:
            if len(same_hour) > 1:
                raise ValueError(
                    f"Multiple baseline rasters found for {hour_utc:02d}:00 UTC: "
                    + ", ".join(str(p) for p in same_hour)
                )
            return same_hour[0]

    available = ", ".join(f"{file_hour_utc(p):02d}:00 UTC ({p.name})" for p in candidates)
    raise FileNotFoundError(
        f"No baseline raster found for {hour_utc:02d}:00 UTC. "
        f"Expected this to match the 13:00 local heatwave rasters. Available: {available}"
    )


def read_baseline_matched_to_surface(surface: RasterSurface, baseline_input: Path) -> np.ndarray:
    baseline_path = matching_baseline_for_hour(baseline_input, surface.hour_utc)
    baseline_arr, baseline_profile = read_raster(baseline_path)
    baseline_arr = reproject_to_match(baseline_arr, baseline_profile, surface.profile)
    print(f"Matched baseline for {surface.label}: {baseline_path.name}")
    return baseline_arr


def aggregate_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    with np.errstate(all="ignore"):
        if AGGREGATION == "mean":
            return np.nanmean(stack, axis=0)
        if AGGREGATION == "mean_positive":
            return np.nanmean(np.where(stack > 0, stack, np.nan), axis=0)
        if AGGREGATION == "max":
            return np.nanmax(stack, axis=0)
    raise ValueError(f"Unsupported AGGREGATION: {AGGREGATION}")


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
        if bbox is not None:
            gdf = gpd.read_file(path, bbox=bbox)
        else:
            gdf = gpd.read_file(path)
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


def rasterize_full_cover_gdf(gdf: gpd.GeoDataFrame, profile: dict) -> np.ndarray:
    """Rasterize only pixels that are fully covered by the polygon geometry.

    This is approximated conservatively by shrinking polygons inward by half the
    cell diagonal before normal center-based rasterization. A pixel is retained
    only if its center lies inside this eroded polygon, which implies the full
    pixel lies inside the original polygon for square raster cells.
    """
    dx = abs(profile["transform"].a)
    dy = abs(profile["transform"].e)
    inward = 0.5 * np.hypot(dx, dy)
    if inward <= 0:
        return rasterize_gdf(gdf, profile, all_touched=False)

    eroded = gdf.copy()
    eroded["geometry"] = eroded.geometry.buffer(-inward)
    eroded = eroded[eroded.geometry.notna() & ~eroded.geometry.is_empty].copy()
    if eroded.empty:
        return np.zeros((profile["height"], profile["width"]), dtype=bool)
    return rasterize_gdf(eroded, profile, all_touched=False)


def build_city_mask(target_profile: dict) -> np.ndarray:
    outline = read_vector_fast(PERUSPIIRI_PATH, target_profile, "Helsinki outline", use_bbox=True)
    city_mask = rasterize_gdf(outline, target_profile, all_touched=False)
    if not np.any(city_mask):
        raise ValueError("City mask from peruspiiri outline is empty.")
    return city_mask


def rasterize_bareground(target_profile: dict) -> np.ndarray:
    bare = read_vector_fast(BAREGROUND_VECTOR_PATH, target_profile, "bareground", use_bbox=True)
    return rasterize_gdf(bare, target_profile, all_touched=False).astype(float)


def build_target_domain(profile: dict, city_mask: np.ndarray) -> np.ndarray:
    tree = read_and_match_fraction(TREE_PATH, profile, "TREE")
    nwn = read_and_match_fraction(NWN_PATH, profile, "NWN")
    water = read_and_match_fraction(WATER_PATH, profile, "WATER")
    ocean = read_and_match_fraction(OCEAN_PATH, profile, "OCEAN")
    impervious = read_and_match_fraction(IMPERVIOUS_PATH, profile, "IMPERVIOUS")
    building = read_and_match_fraction(BUILDING_PATH, profile, "BUILDING")
    bareground = rasterize_bareground(profile)

    nwn_veg = np.where(bareground > 0, 0.0, nwn)
    vegetation_fraction = np.clip(tree + nwn_veg, 0.0, 1.0)
    vegetation_domain = vegetation_fraction >= VEGETATION_MIN_FRACTION
    water_domain = city_mask & ((water >= WATER_MIN_FRACTION) | (ocean >= WATER_MIN_FRACTION))
    building_excluded = building > BUILDING_EXCLUDE_FRACTION
    impervious_excluded = impervious > IMPERVIOUS_EXCLUDE_FRACTION

    target_domain = city_mask & vegetation_domain & (~water_domain) & (~building_excluded) & (~impervious_excluded)
    print("[TARGET DOMAIN]")
    print(f"  city cells                  = {int(np.sum(city_mask))}")
    print(f"  vegetation cells in city    = {int(np.sum(vegetation_domain & city_mask))}")
    print(f"  water/ocean cells in city   = {int(np.sum(water_domain))}")
    print(f"  building-excluded in city   = {int(np.sum(building_excluded & city_mask))}")
    print(f"  impervious-excluded in city = {int(np.sum(impervious_excluded & city_mask))}")
    print(f"  target-domain cells         = {int(np.sum(target_domain))}")
    if not np.any(target_domain):
        raise ValueError("Target-domain mask is empty. Check predictor raster grids and thresholds.")
    return target_domain


def finite_values_for_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask]
    return vals[np.isfinite(vals)]

# =============================================================================
# DATA EXTRACTION
# =============================================================================

def compute_surfaces() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    surfaces = [select_peak_surface(label, path) for label, path in HEATWAVE_INPUTS.items()]
    target_profile = surfaces[0].profile
    if target_profile.get("crs") is None:
        raise ValueError("Raster CRS is missing; cannot align vectors.")

    city_mask = build_city_mask(target_profile)
    target_domain = build_target_domain(target_profile, city_mask)

    heatwave_arrays = []
    baseline_mean_arrays = []
    baseline_p90_arrays = []
    mean_offsets = []
    p90_offsets = []

    for surface in surfaces:
        heatwave = reproject_to_match(surface.array, surface.profile, target_profile)
        baseline_mean = read_baseline_matched_to_surface(surface, BASELINE_MEAN_INPUT)
        baseline_p90 = read_baseline_matched_to_surface(surface, BASELINE_P90_INPUT)
        baseline_mean = reproject_to_match(baseline_mean, surface.profile, target_profile)
        baseline_p90 = reproject_to_match(baseline_p90, surface.profile, target_profile)

        heatwave_arrays.append(heatwave)
        baseline_mean_arrays.append(baseline_mean)
        baseline_p90_arrays.append(baseline_p90)
        mean_offsets.append(heatwave - baseline_mean)
        p90_offsets.append(heatwave - baseline_p90)

    agg_heatwave = np.where(target_domain, aggregate_arrays(heatwave_arrays), np.nan)
    agg_tmean = np.where(target_domain, aggregate_arrays(baseline_mean_arrays), np.nan)
    agg_tp90 = np.where(target_domain, aggregate_arrays(baseline_p90_arrays), np.nan)
    agg_dmean = np.where(target_domain, aggregate_arrays(mean_offsets), np.nan)
    agg_dp90 = np.where(target_domain, aggregate_arrays(p90_offsets), np.nan)

    return agg_heatwave, agg_tmean, agg_tp90, agg_dmean, agg_dp90, target_profile, target_domain


def load_lulc_masks(profile: dict, target_domain: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for class_name, paths in LULC_LAYERS:
        class_mask = np.zeros_like(target_domain, dtype=bool)
        for path in paths:
            gdf = read_vector_fast(path, profile, f"LULC {class_name}", use_bbox=True)
            class_mask |= rasterize_gdf(gdf, profile, all_touched=False)
        class_mask &= target_domain
        masks[class_name] = class_mask
        print(f"LULC {class_name}: {int(class_mask.sum())} target-domain raster cells")

    exclusive: dict[str, np.ndarray] = {}
    already = np.zeros_like(target_domain, dtype=bool)
    for class_name, _ in LULC_LAYERS:
        m = masks[class_name] & (~already)
        exclusive[class_name] = m
        already |= m
        print(f"LULC {class_name} exclusive cells: {int(m.sum())}")
    return exclusive


def extract_lulc_data(
    heatwave_arr: np.ndarray,
    tmean_arr: np.ndarray,
    tp90_arr: np.ndarray,
    dmean_arr: np.ndarray,
    dp90_arr: np.ndarray,
    lulc_masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    summary = []

    for class_name, _ in LULC_LAYERS:
        class_mask = lulc_masks[class_name]

        # Remove erroneous absolute-temperature pixels before any statistics or plotting.
        # A pixel is retained only if all absolute temperature surfaces used in
        # panels A/C are finite and >= MIN_VALID_ABSOLUTE_TEMP_C. The same valid
        # pixel mask is also used for the offset values in panel B, so the three
        # panels are based on the same cleaned set of pixels.
        valid_abs_mask = (
            class_mask
            & np.isfinite(heatwave_arr)
            & np.isfinite(tmean_arr)
            & np.isfinite(tp90_arr)
            & (heatwave_arr >= MIN_VALID_ABSOLUTE_TEMP_C)
            & (tmean_arr >= MIN_VALID_ABSOLUTE_TEMP_C)
            & (tp90_arr >= MIN_VALID_ABSOLUTE_TEMP_C)
        )

        dropped = int(np.sum(class_mask) - np.sum(valid_abs_mask))
        if dropped > 0:
            print(
                f"{class_name}: dropped {dropped:,} pixels with absolute "
                f"temperature < {MIN_VALID_ABSOLUTE_TEMP_C:g} °C or non-finite values"
            )

        v_hw = finite_values_for_mask(heatwave_arr, valid_abs_mask)
        v_tm = finite_values_for_mask(tmean_arr, valid_abs_mask)
        v_tp = finite_values_for_mask(tp90_arr, valid_abs_mask)
        v_dm = finite_values_for_mask(dmean_arr, valid_abs_mask)
        v_dp = finite_values_for_mask(dp90_arr, valid_abs_mask)

        summary.append({
            "class": class_name,
            "n_pixels": int(v_hw.size),
            "dropped_low_abs_temp_pixels": dropped,
            "min_valid_absolute_temp_degC": MIN_VALID_ABSOLUTE_TEMP_C,
            "mean_hw": float(np.nanmean(v_hw)) if v_hw.size else np.nan,
            "mean_tmean": float(np.nanmean(v_tm)) if v_tm.size else np.nan,
            "mean_tp90": float(np.nanmean(v_tp)) if v_tp.size else np.nan,
            "mean_dmean": float(np.nanmean(v_dm)) if v_dm.size else np.nan,
            "mean_dp90": float(np.nanmean(v_dp)) if v_dp.size else np.nan,
        })

        records.extend({"class": class_name, "series": "Δ mean", "value_degC": float(v)} for v in v_dm)
        records.extend({"class": class_name, "series": "Δ p90", "value_degC": float(v)} for v in v_dp)
        records.extend({"class": class_name, "series": "HW", "value_degC": float(v)} for v in v_hw)
        records.extend({"class": class_name, "series": "T p90", "value_degC": float(v)} for v in v_tp)
        records.extend({"class": class_name, "series": "T mean", "value_degC": float(v)} for v in v_tm)

    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(summary)

# =============================================================================
# PLOTTING
# =============================================================================

def save_svg(fig: plt.Figure, basename: str) -> None:
    gps.make_transparent(fig)
    out = WORKDIR / f"{basename}.svg"
    fig.savefig(out, transparent=STYLE.transparent, facecolor="none", edgecolor="none", bbox_inches=None, pad_inches=0)
    width_cm = fig.get_figwidth() * gps.CM_PER_INCH
    height_cm = fig.get_figheight() * gps.CM_PER_INCH
    print(f"[OK] wrote {out} ({width_cm:.2f} × {height_cm:.2f} cm fixed canvas)")


def _sample_for_violin(vals: np.ndarray) -> np.ndarray:
    vals = vals[np.isfinite(vals)]
    if vals.size <= MAX_VIOLIN_POINTS_PER_CLASS:
        return vals
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(vals.size, size=MAX_VIOLIN_POINTS_PER_CLASS, replace=False)
    return vals[idx]


def _clip_for_display(vals: np.ndarray) -> np.ndarray:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return vals
    lo, hi = np.nanpercentile(vals, DISPLAY_CLIP_PERCENTILES)
    clipped = vals[(vals >= lo) & (vals <= hi)]
    return clipped if clipped.size else vals


def _draw_violin_summary(ax, x: float, vals_raw: np.ndarray, color: str, half_width: float) -> None:
    q25, med, q75 = np.nanpercentile(vals_raw, [25, 50, 75])
    mean = float(np.nanmean(vals_raw))
    ax.plot([x, x], [q25, q75], color=STYLE.col_black, linewidth=1.9, solid_capstyle="round", zorder=7)
    ax.plot([x - half_width, x + half_width], [med, med], color=color, linewidth=2.1, zorder=8)
    ax.scatter([x], [mean], s=22, color=color, zorder=9)


def _boxplot_whiskers(vals: np.ndarray) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    q1, q3 = np.nanpercentile(vals, [25, 75])
    iqr = q3 - q1
    low_lim = q1 - 1.5 * iqr
    high_lim = q3 + 1.5 * iqr
    vmin = float(np.nanmin(vals[vals >= low_lim])) if np.any(vals >= low_lim) else float(np.nanmin(vals))
    vmax = float(np.nanmax(vals[vals <= high_lim])) if np.any(vals <= high_lim) else float(np.nanmax(vals))
    return vmin, vmax


def _ordered_classes(summary_df: pd.DataFrame) -> list[str]:
    available = summary_df["class"].tolist()
    ordered = [c for c in PLOT_ORDER if c in available and int(summary_df.loc[summary_df["class"] == c, "n_pixels"].iloc[0]) > 0]
    # fallback for any unexpected class names
    ordered.extend([c for c in available if c not in ordered and int(summary_df.loc[summary_df["class"] == c, "n_pixels"].iloc[0]) > 0])
    return ordered


def _add_axis_break_marker(ax) -> None:
    kw = dict(transform=ax.transAxes, color=STYLE.col_black, clip_on=False, linewidth=1.2)
    ax.plot((-0.012, +0.012), (0.030, 0.050), **kw)
    ax.plot((-0.012, +0.012), (0.055, 0.075), **kw)


def _add_class_background_shading(ax, x_positions: np.ndarray, *, alpha: float = 0.08) -> None:
    for i, xi in enumerate(x_positions):
        if i % 2 == 0:
            ax.axvspan(xi - 0.5, xi + 0.5, color=STYLE.col_grey, alpha=alpha, zorder=0)


def plot_panel_a_temperatures(summary_df: pd.DataFrame) -> None:
    order = _ordered_classes(summary_df)
    x = np.arange(1, len(order) + 1, dtype=float)
    labels = [DISPLAY_LABEL_MAP.get(c, c) for c in order]

    y_hw = np.array([float(summary_df.loc[summary_df["class"] == c, "mean_hw"].iloc[0]) for c in order])
    y_tp = np.array([float(summary_df.loc[summary_df["class"] == c, "mean_tp90"].iloc[0]) for c in order])
    y_tm = np.array([float(summary_df.loc[summary_df["class"] == c, "mean_tmean"].iloc[0]) for c in order])

    fig = gps.new_panel_figure(STYLE)
    ax = fig.add_axes(PANEL_AXES_RECT)
    gps.style_axis(ax, STYLE, grid_y=True, grid_x=False)

    ax.bar(x, y_hw, width=BAR_WIDTH, color=STYLE.col_hist_fill, edgecolor=STYLE.col_hist, linewidth=1.0, alpha=1.0, zorder=2)
    ax.bar(x, y_tp, width=BAR_WIDTH, color=STYLE.col_red_fill, edgecolor=STYLE.col_red, linewidth=1.0, alpha=1.0, zorder=3)
    ax.bar(x, y_tm, width=BAR_WIDTH, color=STYLE.col_blue_fill, edgecolor=STYLE.col_blue, linewidth=1.0, alpha=1.0, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=STYLE.fs_tick, fontfamily=STYLE.font_family)
    gps.set_axis_labels(ax, xlabel=None, ylabel="Mean temperature (°C)", style=STYLE)

    ymax = max(float(np.nanmax(y_hw)), float(np.nanmax(y_tp)), float(np.nanmax(y_tm)))
    ymin = 10.0
    ax.set_ylim(ymin, ymax * 1.14)
    _add_axis_break_marker(ax)

    # 25 °C guide line.
    ax.axhline(25.0, color=STYLE.col_black, linewidth=1.2, linestyle="--", alpha=0.90, zorder=8)

    # Make bottom tick label read 0 without moving the axis.
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: "0" if abs(val - ymin) < 1e-6 else f"{val:.0f}"))

    # 25 °C guide line, matching panel A.
    ax.axhline(25.0, color=STYLE.col_black, linewidth=1.2, linestyle="--", alpha=0.90, zorder=8)

    legend_handles = [
        Patch(facecolor=STYLE.col_hist_fill, edgecolor=STYLE.col_hist, alpha=0.82, label="HW avg."),
        Patch(facecolor=STYLE.col_red_fill, edgecolor=STYLE.col_red, alpha=0.72, label="base p90"),
        Patch(facecolor=STYLE.col_blue_fill, edgecolor=STYLE.col_blue, alpha=0.82, label="base mean"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=0.70,
        facecolor="white",
        edgecolor="#B8B8B8",
        fontsize=STYLE.fs_legend,
        borderpad=0.30,
        labelspacing=0.25,
        columnspacing=0.90,
        handlelength=1.25,
        handletextpad=0.35,
    )

    ax.set_xlim(0.45, len(order) + 0.55)
    save_svg(fig, OUTPUT_PANEL_A)
    plt.close(fig)


def plot_panel_b_violins(values_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    order = _ordered_classes(summary_df)
    x = np.arange(1, len(order) + 1, dtype=float)
    labels = [DISPLAY_LABEL_MAP.get(c, c) for c in order]

    raw_dmean = [values_df.loc[(values_df["class"] == c) & (values_df["series"] == "Δ mean"), "value_degC"].to_numpy(dtype=float) for c in order]
    raw_dp90 = [values_df.loc[(values_df["class"] == c) & (values_df["series"] == "Δ p90"), "value_degC"].to_numpy(dtype=float) for c in order]
    disp_dmean = [_clip_for_display(_sample_for_violin(v)) for v in raw_dmean]
    disp_dp90 = [_clip_for_display(_sample_for_violin(v)) for v in raw_dp90]

    fig = gps.new_panel_figure(STYLE)
    ax = fig.add_axes(PANEL_AXES_RECT)
    gps.style_axis(ax, STYLE, grid_y=True, grid_x=False)

    # Two global background ribbons.
    p90_ranges = [_boxplot_whiskers(vals) for vals in raw_dp90]
    mean_ranges = [_boxplot_whiskers(vals) for vals in raw_dmean]
    low_p90 = min(lo for lo, _ in p90_ranges)
    high_p90 = max(hi for _, hi in p90_ranges)
    low_mean = min(lo for lo, _ in mean_ranges)
    high_mean = max(hi for _, hi in mean_ranges)

    x0 = 0.55
    x1 = len(order) + 0.45
    rect_p90 = Rectangle((x0, low_p90), x1 - x0, high_p90 - low_p90,
                         facecolor=STYLE.col_red_fill, edgecolor=STYLE.col_red, linewidth=0.8,
                         alpha=RIBBON_ALPHA_BACK, zorder=1)
    rect_mean = Rectangle((x0, low_mean), x1 - x0, high_mean - low_mean,
                          facecolor=STYLE.col_blue_fill, edgecolor=STYLE.col_blue, linewidth=0.8,
                          alpha=RIBBON_ALPHA_FRONT, zorder=2)
    ax.add_patch(rect_p90)
    ax.add_patch(rect_mean)

    parts_back = ax.violinplot(disp_dp90, positions=x, widths=VIOLIN_WIDTH_BACK, showmeans=False, showmedians=False, showextrema=False, bw_method="scott")
    for body in parts_back["bodies"]:
        body.set_facecolor(STYLE.col_red_fill)
        body.set_edgecolor(STYLE.col_red)
        body.set_linewidth(1.05)
        body.set_alpha(1.0)
        body.set_zorder(5)

    parts_front = ax.violinplot(disp_dmean, positions=x, widths=VIOLIN_WIDTH_FRONT, showmeans=False, showmedians=False, showextrema=False, bw_method="scott")
    for body in parts_front["bodies"]:
        body.set_facecolor(STYLE.col_blue_fill)
        body.set_edgecolor(STYLE.col_blue)
        body.set_linewidth(1.10)
        body.set_alpha(1.0)
        body.set_zorder(6)

    for xi, vals in zip(x, raw_dp90):
        _draw_violin_summary(ax, xi, vals, STYLE.col_red, 0.12)
    for xi, vals in zip(x, raw_dmean):
        _draw_violin_summary(ax, xi, vals, STYLE.col_blue, 0.08)

    ax.axhline(0, color=STYLE.col_black, linewidth=0.9, alpha=0.45, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=STYLE.fs_tick, fontfamily=STYLE.font_family)
    gps.set_axis_labels(ax, xlabel=None, ylabel="Heatwave Δ T (°C)", style=STYLE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"+{val:.0f}"))

    combined_display = np.concatenate([v for v in disp_dmean + disp_dp90 if v.size > 0])
    y_hi = float(np.nanmax(combined_display))
    y_top = max(y_hi, high_p90, high_mean) * 1.14
    ax.set_ylim(0, y_top)
    ax.set_xlim(0.45, len(order) + 0.55)

    # Ribbon titles near the middle (around Trees >15 m), raised to avoid overlap.
    try:
        x_text = x[order.index("Trees >15 m")]
    except ValueError:
        x_text = x[len(x) // 2]
    ax.text(x_text, min(high_mean + 0.18 * (y_top - high_mean), y_top * 0.96), "from mean",
            ha="center", va="bottom", fontsize=RIBBON_LABEL_FS, fontfamily=STYLE.font_family,
            fontweight="bold", color=STYLE.col_blue, zorder=7)
    ax.text(x_text, 2.8, "from p90",
            ha="center", va="bottom", fontsize=RIBBON_LABEL_FS, fontfamily=STYLE.font_family,
            fontweight="bold", color=STYLE.col_red, zorder=7)

    save_svg(fig, OUTPUT_PANEL_B)
    plt.close(fig)


def plot_panel_c_temperature_violins(values_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    order = _ordered_classes(summary_df)
    x = np.arange(1, len(order) + 1, dtype=float)
    labels = [DISPLAY_LABEL_MAP.get(c, c) for c in order]

    series_names = ["HW", "T p90", "T mean"]
    offsets = [-ABS_VIOLIN_OFFSET, 0.0, ABS_VIOLIN_OFFSET]
    widths = ABS_VIOLIN_WIDTH
    colors = [
        (STYLE.col_hist_fill, STYLE.col_hist, 0.82),
        (STYLE.col_red_fill, STYLE.col_red, 0.72),
        (STYLE.col_blue_fill, STYLE.col_blue, 0.82),
    ]

    fig = gps.new_panel_figure(STYLE)
    ax = fig.add_axes(PANEL_AXES_RECT)
    gps.style_axis(ax, STYLE, grid_y=True, grid_x=False)
    _add_class_background_shading(ax, x, alpha=0.06)

    all_vals = []
    for sname, off, (fcol, ecol, alpha) in zip(series_names, offsets, colors):
        data = [values_df.loc[(values_df["class"] == c) & (values_df["series"] == sname), "value_degC"].to_numpy(dtype=float) for c in order]
        sampled = [_sample_for_violin(v) for v in data]
        positions = x + off
        parts = ax.violinplot(sampled, positions=positions, widths=widths, showmeans=False, showmedians=False, showextrema=False, bw_method="scott")
        for body in parts["bodies"]:
            body.set_facecolor(fcol)
            body.set_edgecolor(ecol)
            body.set_linewidth(1.0)
            body.set_alpha(alpha)
            body.set_zorder(4)
        for xi, vals in zip(positions, data):
            if vals.size == 0:
                continue
            _draw_violin_summary(ax, xi, vals, ecol, 0.06)
        all_vals.extend([v for v in sampled if v.size > 0])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=STYLE.fs_tick, fontfamily=STYLE.font_family)
    gps.set_axis_labels(ax, xlabel=None, ylabel="Temperature (°C)", style=STYLE)

    ymin = 15.0
    if all_vals:
        y_lo = min(float(np.nanmin(v)) for v in all_vals)
        y_hi = max(float(np.nanmax(v)) for v in all_vals)
        yr = y_hi - y_lo if y_hi > y_lo else 1.0
        ax.set_ylim(ymin, y_hi + 0.04 * yr)
        _add_axis_break_marker(ax)
        ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: "0" if abs(val - ymin) < 1e-6 else f"{val:.0f}"))

    legend_handles = [
        Patch(facecolor=STYLE.col_hist_fill, edgecolor=STYLE.col_hist, alpha=0.82, label="HW avg."),
        Patch(facecolor=STYLE.col_red_fill, edgecolor=STYLE.col_red, alpha=0.72, label="base p90"),
        Patch(facecolor=STYLE.col_blue_fill, edgecolor=STYLE.col_blue, alpha=0.82, label="base mean"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=False,
        fancybox=False,
        framealpha=0.70,
        facecolor="white",
        edgecolor="#B8B8B8",
        fontsize=STYLE.fs_legend,
        borderpad=0.30,
        labelspacing=0.25,
        columnspacing=0.90,
        handlelength=1.25,
        handletextpad=0.35,
    )

    ax.set_xlim(0.45, len(order) + 0.55)
    save_svg(fig, OUTPUT_PANEL_C)
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_global_style()
    print("Resolved global plotting style:")
    print(f"  global file: {GLOBAL_SETTINGS}")
    print(f"  output dir:  {WORKDIR}")
    print(f"  canvas:      {STYLE.panel_width_cm} × {STYLE.panel_height_cm} cm")
    print("  exports:     SVG only")
    print(f"  analysis hour: {EXPECTED_PEAK_LOCAL_HOUR:02d}:00 local = {TARGET_HEATWAVE_UTC_HOUR:02d}:00 UTC")
    print("  LULC extraction: simple rasterized class masks (mixed-pixel exclusion omitted for simplicity)")
    print(f"  violin clipping: {DISPLAY_CLIP_PERCENTILES[0]:.0f}th–{DISPLAY_CLIP_PERCENTILES[1]:.0f}th percentiles")
    print(f"  absolute-temperature filter: remove pixels < {MIN_VALID_ABSOLUTE_TEMP_C:g} °C before plotting/statistics")

    check_required_files()
    heatwave_arr, tmean_arr, tp90_arr, dmean_arr, dp90_arr, profile, target_domain = compute_surfaces()
    lulc_masks = load_lulc_masks(profile, target_domain)
    values_df, summary_df = extract_lulc_data(heatwave_arr, tmean_arr, tp90_arr, dmean_arr, dp90_arr, lulc_masks)
    print(summary_df.to_string(index=False))

    plot_panel_a_temperatures(summary_df)
    plot_panel_b_violins(values_df, summary_df)
    plot_panel_c_temperature_violins(values_df, summary_df)


if __name__ == "__main__":
    main()
