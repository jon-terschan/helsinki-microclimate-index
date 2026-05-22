#!/usr/bin/env python3
"""
Patch-level predictor profiling: cool-stable vs cool-amplifying vegetation.

Purpose
-------
Characterize which environmental / urban-context predictors distinguish the two
cool bivariate-response extremes:

  cool-stable      = low absolute heatwave T, low HW - mean amplification
  cool-amplifying  = low absolute heatwave T, high HW - mean amplification

The analysis is patch-level, not pixel-level. Connected patches are labelled
within each extreme class, and predictor rasters are summarized by patch median.
The output ranks predictors by standardized median difference between
cool-amplifying and cool-stable patches.

Inputs expected
---------------
- Valid target-domain raster from f4 connectivity workflow
- Heatwave and baseline mean rasters used by the bivariate map
- Predictor stack rasters in DATA/predictorstack
- Optional LULC vector layers for class-composition summaries

Outputs
-------
figures/results/figures/f3_2/tables/
  f3_2_cool_stable_vs_amplifying_predictor_patch_values_pm1p0deg.csv
  f3_2_cool_stable_vs_amplifying_predictor_effects_pm1p0deg.csv
  f3_2_cool_stable_vs_amplifying_predictor_lulc_composition_pm1p0deg.csv

figures/results/figures/f3_2/
  f3_2_cool_stable_vs_amplifying_predictor_effects_pm1p0deg.svg
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

DATA_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
FIGURES_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures")
WORKDIR = FIGURES_DIR / "f5"
TABLE_DIR = WORKDIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
WORKDIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SETTINGS = FIGURES_DIR / "global_plotting_settings.py"
PREDICTOR_DIR = DATA_DIR / "predictorstack"

HEATWAVE_INPUTS = {
    "2010": DATA_DIR / "predictions" / "2010" / "20100728",
    "2018": DATA_DIR / "predictions" / "2018" / "20180717",
    "2021": DATA_DIR / "predictions" / "2021" / "20210714",
}
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

# =============================================================================
# SETTINGS
# =============================================================================

LOCAL_UTC_OFFSET_HOURS = 3
EXPECTED_PEAK_LOCAL_HOUR = 13
TARGET_HEATWAVE_UTC_HOUR = (EXPECTED_PEAK_LOCAL_HOUR - LOCAL_UTC_OFFSET_HOURS) % 24
AGGREGATION = "mean"
ASSUME_VECTOR_EPSG_IF_MISSING = 3879
MIN_VALID_ABSOLUTE_TEMP_C = 15.0

# 4 × 4 bivariate classes: class_id = offset_quartile * 4 + temp_quartile + 1
N_CLASSES = 4
CLASS_BREAKS = [0.25, 0.50, 0.75]
CORNER_CLASSES = {
    "cool-stable": {"class_id": 1, "label": "cool stable", "color": "#1b9e77"},       # low T, low amplification
    "cool-amplifying": {"class_id": 13, "label": "cool amplifying", "color": "#7b3294"}, # low T, high amplification
}

CONNECTIVITY = 8
MIN_PATCH_PIXELS = 9
MAX_PATCHES_PER_CLASS_FOR_PLOT = None  # keep None; plot shows predictor effects, not patches

# Predictor selection for the single, interpretable model.
# tuple: model_name, display_label, group, candidate filenames/patterns, patch_summary, model_transform
# patch_summary:
#   median          = median patch value (distance, terrain, canopy height)
#   fraction_mean   = mean fraction across patch pixels, clipped to 0..1
#   presence_share  = share of patch pixels where raster value > PRESENCE_THRESHOLD
# model_transform:
#   identity, log1p
PRESENCE_THRESHOLD = 1e-6
MODEL_PREDICTORS = [
    # Regional / location context
    ("inland_position", "Inland position\n(distance to ocean)", "location", ["OCEAN_DIST_10m_Helsinki.tif", "*OCEAN*DIST*.tif"], "median", "log1p"),
    ("elevation", "Elevation\n(DTM)", "terrain/exposure", ["DTM_10m_Helsinki.tif", "*DTM*.tif"], "median", "identity"),
    ("slope", "Slope", "terrain/exposure", ["SLOPE_10m_Helsinki.tif", "*SLOPE*.tif"], "median", "identity"),
    ("tpi_50m", "TPI\n50 m", "terrain/exposure", ["TPI_50m_10m_Helsinki.tif", "*TPI*50*.tif"], "median", "identity"),
    ("southness", "Southness", "terrain/exposure", ["SOUTHNESS_10m_Helsinki.tif", "*SOUTHNESS*.tif"], "median", "identity"),
    # Canopy / surface context
    ("canopy_height_max", "Canopy height\nmax", "vegetation/canopy", ["CHM_10m_MAX.tif", "*CHM*MAX*.tif", "*canopy*height*.tif"], "median", "identity"),
    ("rock_presence_10m", "Rock presence\n10 m", "surface", ["ROCK_FRAC_10m_Helsinki.tif", "ROCK_FRAC_10m.tif", "*ROCK*FRAC*10*.tif", "*rock*frac*10*.tif"], "presence_share", "identity"),
    # Built context
    ("urban_distance", "Urban distance\n(distance to buildings)", "built context", ["BLDG_DIST.tif", "*BLDG*DIST*.tif", "*building*dist*.tif"], "median", "log1p"),
    ("building_50m", "Building fraction\n50 m", "built context", ["BLDG_FRAC_MEAN_50m.tif", "*BLDG*FRAC*50*.tif", "*building*frac*50*.tif"], "fraction_mean", "identity"),
    ("impervious_50m", "Impervious fraction\n50 m", "built context", ["IMPREV_FRAC_50m_Helsinki.tif", "*IMPREV*50*.tif", "*imperv*50*.tif"], "fraction_mean", "identity"),
]

OUTPUT_BASENAME = f"f3_2_cool_stable_vs_amplifying_spatial_adjusted_contrasts_twopanel_pm{TOL_LABEL}deg"
PATCH_VALUES_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_patch_values.csv"
EFFECTS_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_effects.csv"
MEDIANS_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_group_medians.csv"
CONTEXT_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_adjustment_context.csv"

# Plot settings
PANEL_WIDTH_CM = 17.0
PANEL_HEIGHT_CM = 10.5
AX_RAW_RECT = [0.36, 0.57, 0.58, 0.34]
AX_ADJ_RECT = [0.36, 0.13, 0.58, 0.34]
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

# Variables used only to absorb broad city geography in the adjusted local-context contrasts.
# These are not treated as causal controls; they are a conservative spatial/geographic baseline.
ADJUSTMENT_CONTROLS = ["centroid_x", "centroid_y", "inland_position", "elevation", "slope"]
REGIONAL_VARIABLES = ["inland_position", "elevation", "slope"]
LOCAL_VARIABLES = ["tpi_50m", "southness", "canopy_height_max", "rock_presence_10m", "urban_distance", "building_50m", "impervious_50m"]
STABLE_COLOR = "#1b9e77"
AMPLIFYING_COLOR = "#7b3294"
ZERO_COLOR = "#606060"

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
    raise FileNotFoundError(f"No unique baseline raster for {hour_utc:02d}:00 UTC. Available: {available}")


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
    return thw, tmean, tp90, thw - tmean, thw - tp90

# =============================================================================
# CLASSIFICATION / PATCHES
# =============================================================================

def rank01(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values)
    ranks = s.rank(method="average").to_numpy(dtype=float)
    n = len(values)
    if n <= 1:
        return np.zeros_like(values, dtype=float)
    return (ranks - 1.0) / (n - 1.0)


def classify_bivar(thw: np.ndarray, offset: np.ndarray, valid_domain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = valid_domain & np.isfinite(thw) & np.isfinite(offset) & (thw >= MIN_VALID_ABSOLUTE_TEMP_C)
    r_thw = np.full(thw.shape, np.nan, dtype=float)
    r_offset = np.full(thw.shape, np.nan, dtype=float)
    r_thw[valid] = rank01(thw[valid])
    r_offset[valid] = rank01(offset[valid])
    temp_cls = np.full(thw.shape, -1, dtype=np.int8)
    off_cls = np.full(thw.shape, -1, dtype=np.int8)
    temp_cls[valid] = np.digitize(r_thw[valid], bins=CLASS_BREAKS, right=False)
    off_cls[valid] = np.digitize(r_offset[valid], bins=CLASS_BREAKS, right=False)
    bivar = np.zeros(thw.shape, dtype=np.uint8)
    bivar[valid] = (off_cls[valid] * N_CLASSES + temp_cls[valid] + 1).astype(np.uint8)
    return bivar, r_thw, r_offset


def patch_structure() -> np.ndarray:
    if CONNECTIVITY == 8:
        return np.ones((3, 3), dtype=int)
    if CONNECTIVITY == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    raise ValueError("CONNECTIVITY must be 4 or 8")


def cell_area_m2(profile: dict) -> float:
    return abs(float(profile["transform"].a) * float(profile["transform"].e))


def label_corner_patches(bivar: np.ndarray, profile: dict) -> tuple[np.ndarray, pd.DataFrame]:
    patch_id_raster = np.zeros(bivar.shape, dtype=np.int32)
    records = []
    next_patch_id = 1
    structure = patch_structure()
    area = cell_area_m2(profile)

    for corner_name, info in CORNER_CLASSES.items():
        class_id = info["class_id"]
        mask = bivar == class_id
        labeled, n_labels = ndi_label(mask, structure=structure)
        if n_labels == 0:
            continue
        counts = np.bincount(labeled.ravel())
        for label_id in range(1, n_labels + 1):
            n_pixels = int(counts[label_id])
            if n_pixels < MIN_PATCH_PIXELS:
                continue
            patch_mask = labeled == label_id
            patch_id = next_patch_id
            next_patch_id += 1
            patch_id_raster[patch_mask] = patch_id
            records.append({
                "patch_id": patch_id,
                "corner": corner_name,
                "corner_label": info["label"],
                "bivariate_id": class_id,
                "n_pixels": n_pixels,
                "area_m2": n_pixels * area,
            })
    patch_df = pd.DataFrame.from_records(records)
    if patch_df.empty:
        raise RuntimeError("No corner patches retained. Try lowering MIN_PATCH_PIXELS.")
    return patch_id_raster, patch_df


def add_patch_centroids(patch_id_raster: np.ndarray, patch_df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Add patch centroid coordinates in raster CRS using pixel centers."""
    valid = patch_id_raster > 0
    if not np.any(valid):
        return patch_df
    rows, cols = np.nonzero(valid)
    patch_ids = patch_id_raster[valid].astype(np.int32)
    transform = profile["transform"]
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    tmp = pd.DataFrame({"patch_id": patch_ids, "centroid_x": xs, "centroid_y": ys})
    cent = tmp.groupby("patch_id", sort=False)[["centroid_x", "centroid_y"]].mean().reset_index()
    return patch_df.merge(cent, on="patch_id", how="left")

# =============================================================================
# LULC HELPERS
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
        gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError(f"{label} has no non-empty geometries: {path}")
    if gdf.crs is None:
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
            if not path.exists():
                print(f"[WARN] missing LULC layer, skipping: {path}")
                continue
            gdf = read_vector_fast(path, profile, f"LULC {class_name}", use_bbox=True)
            class_mask |= rasterize_gdf(gdf, profile, all_touched=False)
        class_mask &= valid_domain
        masks[class_name] = class_mask

    exclusive: dict[str, np.ndarray] = {}
    already = np.zeros_like(valid_domain, dtype=bool)
    for class_name, _ in LULC_LAYERS:
        m = masks[class_name] & (~already)
        exclusive[class_name] = m
        already |= m
    return exclusive


def summarize_lulc_composition(bivar: np.ndarray, lulc_masks: dict[str, np.ndarray], profile: dict) -> pd.DataFrame:
    area = cell_area_m2(profile)
    rows = []
    for corner_name, info in CORNER_CLASSES.items():
        corner_mask = bivar == info["class_id"]
        corner_total = int(corner_mask.sum())
        for lulc in PLOT_ORDER:
            count = int(np.sum(corner_mask & lulc_masks[lulc]))
            rows.append({
                "corner": corner_name,
                "corner_label": info["label"],
                "lulc_class": lulc,
                "lulc_label": DISPLAY_LABEL_MAP.get(lulc, lulc),
                "pixel_count": count,
                "area_m2": count * area,
                "percent_of_corner": 100.0 * count / corner_total if corner_total else np.nan,
            })
    return pd.DataFrame.from_records(rows)

# =============================================================================
# PREDICTOR EXTRACTION / EFFECTS
# =============================================================================

def resolve_predictor_path(candidates: list[str]) -> Path | None:
    for cand in candidates:
        direct = PREDICTOR_DIR / cand
        if direct.exists():
            return direct
        matches = sorted(PREDICTOR_DIR.glob(cand))
        if matches:
            return matches[0]
    return None


def read_predictor_to_grid(candidates: list[str], target_profile: dict, name: str) -> np.ndarray | None:
    path = resolve_predictor_path(candidates)
    if path is None:
        print(f"[WARN] predictor missing, skipping {name}: {candidates}")
        return None
    arr, profile = read_raster(path)
    print(f"Loaded predictor {name}: {path.name}")
    return reproject_to_match(arr, profile, target_profile, resampling=Resampling.bilinear)


def grouped_summary_by_patch(values: np.ndarray, patch_id_raster: np.ndarray, summary: str) -> pd.DataFrame:
    valid = (patch_id_raster > 0) & np.isfinite(values)
    if not np.any(valid):
        return pd.DataFrame(columns=["patch_id", "value"])

    vals = values[valid].astype(np.float32)
    if summary == "fraction_mean":
        # Fraction rasters are expected to represent 0..1 local fractions.
        # Clip tiny numeric overshoots after reprojection and summarize by patch mean.
        vals = np.clip(vals, 0.0, 1.0)
    elif summary == "presence_share":
        # For 10 m fraction/presence rasters, the meaningful patch-level quantity is
        # how much of the patch has any presence, not the median of 0/1 cells.
        vals = (vals > PRESENCE_THRESHOLD).astype(np.float32)
    elif summary != "median":
        raise ValueError(f"Unknown patch summary: {summary}")

    tmp = pd.DataFrame({
        "patch_id": patch_id_raster[valid].astype(np.int32),
        "value": vals,
    })
    if summary in {"fraction_mean", "presence_share"}:
        return tmp.groupby("patch_id", sort=False)["value"].mean().reset_index()
    return tmp.groupby("patch_id", sort=False)["value"].median().reset_index()


def build_patch_predictor_table(
    patch_id_raster: np.ndarray,
    patch_df: pd.DataFrame,
    target_profile: dict,
    thw: np.ndarray,
    tmean: np.ndarray,
    tp90: np.ndarray,
    hw_minus_mean: np.ndarray,
    hw_minus_p90: np.ndarray,
) -> pd.DataFrame:
    out = patch_df.copy()

    thermal_predictors = [
        ("heatwave_temp", "Heatwave T", "thermal", thw),
        ("mean_baseline_temp", "Mean baseline T", "thermal", tmean),
        ("p90_baseline_temp", "p90 baseline T", "thermal", tp90),
        ("hw_minus_mean", "HW − mean", "thermal", hw_minus_mean),
        ("hw_minus_p90", "HW − p90", "thermal", hw_minus_p90),
    ]
    predictor_meta = []

    for var, label, group, arr in thermal_predictors:
        med = grouped_summary_by_patch(arr, patch_id_raster, "median").rename(columns={"value": var})
        out = out.merge(med, on="patch_id", how="left")
        predictor_meta.append({"variable": var, "label": label, "group": group})

    for var, label, group, candidates, patch_summary, model_transform in MODEL_PREDICTORS:
        arr = read_predictor_to_grid(candidates, target_profile, var)
        if arr is None:
            continue
        summary_df = grouped_summary_by_patch(arr, patch_id_raster, patch_summary).rename(columns={"value": var})
        out = out.merge(summary_df, on="patch_id", how="left")
        predictor_meta.append({
            "variable": var,
            "label": label,
            "group": group,
            "patch_summary": patch_summary,
            "model_transform": model_transform,
            "source_candidates": ";".join(candidates),
        })

    meta_df = pd.DataFrame.from_records(predictor_meta)
    out.attrs["predictor_meta"] = meta_df
    return out



# =============================================================================
# SPATIALLY ADJUSTED DESCRIPTIVE CONTRASTS
# =============================================================================

def transform_series(s: pd.Series, transform: str) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").astype(float)
    if transform == "identity":
        return vals
    if transform == "log1p":
        return np.log1p(vals.clip(lower=0))
    raise ValueError(f"Unknown transform: {transform}")


def robust_standardize(vals: pd.Series) -> tuple[pd.Series, float, float, str]:
    """Return robust z-like values using IQR; fall back to SD if needed."""
    x = pd.to_numeric(vals, errors="coerce").astype(float)
    med = float(np.nanmedian(x))
    q25, q75 = np.nanpercentile(x, [25, 75])
    scale = float(q75 - q25)
    method = "IQR"
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(x, ddof=0))
        method = "SD fallback"
    if not np.isfinite(scale) or scale <= 0:
        return pd.Series(np.nan, index=x.index), med, np.nan, "constant"
    return (x - med) / scale, med, scale, method


def fit_linear_residual(y: pd.Series, controls: pd.DataFrame) -> pd.Series:
    """Residualize standardized y against standardized controls using OLS."""
    data = pd.concat([y.rename("y"), controls], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    resid = pd.Series(np.nan, index=y.index, dtype=float)
    if len(data) < controls.shape[1] + 5:
        return resid
    X = np.column_stack([np.ones(len(data)), data[controls.columns].to_numpy(dtype=float)])
    yy = data["y"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    fitted = X @ coef
    resid.loc[data.index] = yy - fitted
    return resid


def median_contrast(values: pd.Series, groups: pd.Series) -> float:
    stable = values[groups == "cool-stable"].dropna().to_numpy(dtype=float)
    amp = values[groups == "cool-amplifying"].dropna().to_numpy(dtype=float)
    if stable.size == 0 or amp.size == 0:
        return np.nan
    return float(np.nanmedian(amp) - np.nanmedian(stable))


def bootstrap_contrast(values: pd.Series, groups: pd.Series, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    stable = values[groups == "cool-stable"].dropna().to_numpy(dtype=float)
    amp = values[groups == "cool-amplifying"].dropna().to_numpy(dtype=float)
    if stable.size < 5 or amp.size < 5:
        return np.nan, np.nan
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        s = rng.choice(stable, size=stable.size, replace=True)
        a = rng.choice(amp, size=amp.size, replace=True)
        boots[i] = np.nanmedian(a) - np.nanmedian(s)
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def get_predictor_meta(predictor_meta: pd.DataFrame, var: str) -> dict:
    recs = predictor_meta[predictor_meta["variable"] == var].to_dict("records")
    if not recs:
        return {"variable": var, "label": var, "group": "unknown", "model_transform": "identity", "patch_summary": ""}
    return recs[0]


def build_standardized_analysis_table(patch_table: pd.DataFrame, predictor_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = patch_table[patch_table["corner"].isin(["cool-stable", "cool-amplifying"])].copy()
    df = df.reset_index(drop=True)

    # Prepare standardized adjustment controls. Centroid coordinates are standardized directly;
    # distance controls use log1p to avoid letting extreme distances dominate.
    control_z = pd.DataFrame(index=df.index)
    control_info = []
    for var in ADJUSTMENT_CONTROLS:
        if var not in df.columns:
            continue
        transform = "log1p" if var in {"inland_position", "urban_distance", "building_distance", "ocean_distance"} else "identity"
        z, center, scale, method = robust_standardize(transform_series(df[var], transform))
        if z.notna().sum() > 0 and np.isfinite(scale):
            control_z[var] = z
            control_info.append({"control": var, "transform": transform, "center": center, "scale": scale, "scale_method": method})

    rows = []
    med_rows = []
    variables = REGIONAL_VARIABLES + LOCAL_VARIABLES
    for var in variables:
        if var not in df.columns:
            print(f"[WARN] variable missing from patch table; skipping {var}")
            continue
        rec = get_predictor_meta(predictor_meta, var)
        transform = rec.get("model_transform", "identity")
        x_raw = pd.to_numeric(df[var], errors="coerce").astype(float)
        x_trans = transform_series(x_raw, transform)
        x_z, center, scale, scale_method = robust_standardize(x_trans)
        if x_z.notna().sum() == 0 or not np.isfinite(scale):
            print(f"[WARN] variable constant/invalid; skipping {var}")
            continue

        raw_eff = median_contrast(x_z, df["corner"])
        raw_lo, raw_hi = bootstrap_contrast(x_z, df["corner"])

        adjusted_eff = np.nan
        adjusted_lo = np.nan
        adjusted_hi = np.nan
        adjusted_used = False
        if var in LOCAL_VARIABLES and len(control_z.columns) >= 3:
            # Do not use the response variable itself as a control if a variable is duplicated.
            controls = control_z[[c for c in control_z.columns if c != var]].copy()
            resid = fit_linear_residual(x_z, controls)
            adjusted_eff = median_contrast(resid, df["corner"])
            adjusted_lo, adjusted_hi = bootstrap_contrast(resid, df["corner"])
            adjusted_used = True

        stable_raw = x_raw[df["corner"] == "cool-stable"].dropna()
        amp_raw = x_raw[df["corner"] == "cool-amplifying"].dropna()
        stable_z = x_z[df["corner"] == "cool-stable"].dropna()
        amp_z = x_z[df["corner"] == "cool-amplifying"].dropna()
        med_rows.append({
            "variable": var,
            "label": rec.get("label", var),
            "group": rec.get("group", ""),
            "patch_summary": rec.get("patch_summary", ""),
            "transform": transform,
            "cool_stable_median_raw": float(np.nanmedian(stable_raw)) if len(stable_raw) else np.nan,
            "cool_amplifying_median_raw": float(np.nanmedian(amp_raw)) if len(amp_raw) else np.nan,
            "raw_median_difference": float(np.nanmedian(amp_raw) - np.nanmedian(stable_raw)) if len(stable_raw) and len(amp_raw) else np.nan,
        })
        rows.append({
            "variable": var,
            "label": rec.get("label", var),
            "group": rec.get("group", ""),
            "panel": "regional raw" if var in REGIONAL_VARIABLES else "local adjusted",
            "patch_summary": rec.get("patch_summary", ""),
            "transform": transform,
            "scale_center": center,
            "scale": scale,
            "scale_method": scale_method,
            "n_cool_stable": int((df["corner"] == "cool-stable").sum()),
            "n_cool_amplifying": int((df["corner"] == "cool-amplifying").sum()),
            "n_finite_cool_stable": int(stable_z.size),
            "n_finite_cool_amplifying": int(amp_z.size),
            "raw_effect": raw_eff,
            "raw_ci_low": raw_lo,
            "raw_ci_high": raw_hi,
            "adjusted_effect": adjusted_eff,
            "adjusted_ci_low": adjusted_lo,
            "adjusted_ci_high": adjusted_hi,
            "adjusted_used": adjusted_used,
        })
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(med_rows), pd.DataFrame.from_records(control_info)


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


def _effect_color(effect: float) -> str:
    if not np.isfinite(effect) or abs(effect) < 1e-12:
        return ZERO_COLOR
    return AMPLIFYING_COLOR if effect > 0 else STABLE_COLOR


def _make_plot_rows(effects: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    rows = []
    if mode == "raw":
        use = effects.copy()
        for _, row in use.iterrows():
            eff, lo, hi = row["raw_effect"], row["raw_ci_low"], row["raw_ci_high"]
            if np.isfinite(eff):
                rows.append({"label": row["label"], "effect": eff, "ci_low": lo, "ci_high": hi, "variable": row["variable"]})
    elif mode == "adjusted":
        use = effects[effects["variable"].isin(LOCAL_VARIABLES) & effects["adjusted_used"]].copy()
        for _, row in use.iterrows():
            eff, lo, hi = row["adjusted_effect"], row["adjusted_ci_low"], row["adjusted_ci_high"]
            if np.isfinite(eff):
                rows.append({"label": row["label"], "effect": eff, "ci_low": lo, "ci_high": hi, "variable": row["variable"]})
    else:
        raise ValueError(mode)
    out = pd.DataFrame.from_records(rows)
    if out.empty:
        return out
    out["abs_effect"] = out["effect"].abs()
    # Put stronger cool-amplifying associations toward the top, stable toward bottom.
    return out.sort_values("effect", ascending=True).reset_index(drop=True)


def _plot_panel(ax, plot_df: pd.DataFrame, *, panel_label: str, xlim: tuple[float, float]) -> None:
    gps.style_axis(ax, STYLE, grid_y=False, grid_x=True)
    ax.axvline(0, color=STYLE.col_black, linewidth=0.85, alpha=0.50, zorder=1)
    if plot_df.empty:
        ax.axis("off")
        return

    y = np.arange(len(plot_df), dtype=float)
    for yi, (_, row) in zip(y, plot_df.iterrows()):
        x = float(row["effect"])
        lo = float(row["ci_low"]) if np.isfinite(row["ci_low"]) else np.nan
        hi = float(row["ci_high"]) if np.isfinite(row["ci_high"]) else np.nan
        color = _effect_color(x)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [yi, yi], color=color, linewidth=1.35, alpha=0.80, zorder=2)
        ax.scatter([x], [yi], s=32, color=color, edgecolor=STYLE.col_black, linewidth=0.35, zorder=3)

    clean_labels = [str(s).replace("\n", " ") for s in plot_df["label"].tolist()]
    ax.set_yticks(y)
    ax.set_yticklabels(clean_labels, fontsize=max(7.0, STYLE.fs_tick * 0.74), fontfamily=STYLE.font_family)
    ax.tick_params(axis="x", labelsize=max(7.0, STYLE.fs_tick * 0.76))
    ax.set_xlim(*xlim)
    ax.text(0.0, 1.02, panel_label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=max(7.6, STYLE.fs_axis * 0.74), fontfamily=STYLE.font_family,
            color=STYLE.col_black, fontweight="bold")


def plot_contrasts(effects: pd.DataFrame) -> None:
    raw_df = _make_plot_rows(effects, mode="raw")
    adj_df = _make_plot_rows(effects, mode="adjusted")
    if raw_df.empty and adj_df.empty:
        raise RuntimeError("No effects available for plotting.")

    all_eff = []
    for df in [raw_df, adj_df]:
        if not df.empty:
            all_eff.extend(df["effect"].to_numpy(dtype=float))
            all_eff.extend(df["ci_low"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float))
            all_eff.extend(df["ci_high"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float))
    lim = max(0.75, min(2.5, float(np.nanmax(np.abs(all_eff))) * 1.12)) if all_eff else 1.0
    xlim = (-lim, lim)

    fig = plt.figure(figsize=(PANEL_WIDTH_CM / gps.CM_PER_INCH, PANEL_HEIGHT_CM / gps.CM_PER_INCH), dpi=STYLE.dpi_export)
    ax_raw = fig.add_axes(AX_RAW_RECT)
    ax_adj = fig.add_axes(AX_ADJ_RECT)

    _plot_panel(ax_raw, raw_df, panel_label="A  raw patch contrast", xlim=xlim)
    _plot_panel(ax_adj, adj_df, panel_label="B  adjusted local/context contrast", xlim=xlim)
    ax_adj.set_xlabel("Standardized median difference", fontsize=max(7.6, STYLE.fs_axis * 0.74), fontfamily=STYLE.font_family)

    # Direction labels only; colors encode side of association.
    fig.text(AX_RAW_RECT[0], 0.965, "cool-stable", ha="left", va="center",
             fontsize=max(8.0, STYLE.fs_legend * 0.78), fontfamily=STYLE.font_family,
             color=STABLE_COLOR, fontweight="bold")
    fig.text(AX_RAW_RECT[0] + AX_RAW_RECT[2], 0.965, "cool-amplifying", ha="right", va="center",
             fontsize=max(8.0, STYLE.fs_legend * 0.78), fontfamily=STYLE.font_family,
             color=AMPLIFYING_COLOR, fontweight="bold")

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
    print(f"  predictor stack: {PREDICTOR_DIR}")
    print("  analysis: spatially adjusted descriptive contrasts, not a multivariate classifier")
    print("  fraction handling: 50 m fractions = patch mean; 10 m rock fraction = patch presence share")
    print("  local/context contrasts adjusted for: centroid x/y, ocean distance, elevation, slope")

    target_arr, target_profile = read_raster(TARGET_DOMAIN_RASTER)
    valid_domain = np.isfinite(target_arr) & (target_arr > 0)

    thw, tmean, tp90, hw_minus_mean, hw_minus_p90 = compute_temperature_surfaces(target_profile)
    bivar, r_thw, r_offset = classify_bivar(thw, hw_minus_mean, valid_domain)
    patch_id_raster, patch_df = label_corner_patches(bivar, target_profile)
    patch_df = add_patch_centroids(patch_id_raster, patch_df, target_profile)
    print(f"Retained patches: {len(patch_df):,}")

    patch_table = build_patch_predictor_table(
        patch_id_raster,
        patch_df,
        target_profile,
        thw,
        tmean,
        tp90,
        hw_minus_mean,
        hw_minus_p90,
    )
    predictor_meta = patch_table.attrs.get("predictor_meta", pd.DataFrame())

    # Some scripts/tables use ocean_distance/building_distance names; keep aliases explicit.
    if "inland_position" not in patch_table.columns and "ocean_distance" in patch_table.columns:
        patch_table["inland_position"] = patch_table["ocean_distance"]
    if "urban_distance" not in patch_table.columns and "building_distance" in patch_table.columns:
        patch_table["urban_distance"] = patch_table["building_distance"]

    effects_df, medians_df, context_df = build_standardized_analysis_table(patch_table, predictor_meta)

    patch_table.to_csv(PATCH_VALUES_CSV, index=False)
    effects_df.to_csv(EFFECTS_CSV, index=False)
    medians_df.to_csv(MEDIANS_CSV, index=False)
    context_df.to_csv(CONTEXT_CSV, index=False)

    print(f"[OK] wrote {PATCH_VALUES_CSV}")
    print(f"[OK] wrote {EFFECTS_CSV}")
    print(f"[OK] wrote {MEDIANS_CSV}")
    print(f"[OK] wrote {CONTEXT_CSV}")

    print("\n[EFFECTS]")
    show_cols = ["variable", "group", "raw_effect", "raw_ci_low", "raw_ci_high", "adjusted_effect", "adjusted_ci_low", "adjusted_ci_high", "adjusted_used"]
    print(effects_df[show_cols].to_string(index=False))

    plot_contrasts(effects_df)


if __name__ == "__main__":
    main()
