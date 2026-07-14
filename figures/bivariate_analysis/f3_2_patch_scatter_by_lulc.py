#!/usr/bin/env python3
"""
Figure f3_2 companion: pixel-level scatterplot across the full bivariate domain.

Uses the same valid vegetated prediction domain, hard domain cut, and bivariate
classification logic as the f3_2 4x4 heatwave-response map:

  x axis (raw value): absolute mean heatwave temperature, THW (°C)  -> cool vs hot
  y axis (raw value): heatwave amplification, THW - Tavg (ΔT, °C)   -> stable vs amplifying

Earlier versions of this figure plotted one point per connected patch (mean
value per patch). That aggregation, combined with the fact that THW and ΔT are
moderately correlated in this domain (Pearson r ≈ 0.44), made some regions of
the plot look artificially "empty" even though the underlying pixel data is a
continuum. Plotting individual pixels instead preserves that continuum. To
avoid rendering ~1.5 million points (slow, illegible, huge file), a uniform
random sample of pixels is drawn from the full classification-valid domain
(see MAX_SAMPLE_POINTS). Uniform random sampling (rather than sampling equal
counts per class or per LULC category) is deliberate: it preserves the real,
observed joint density of THW/ΔT and the real land-cover composition, so the
plot reads as a faithful (if sparser) picture of the true distribution rather
than an artificially rebalanced one.

Each sampled pixel becomes one point in the scatterplot:
  - x position = absolute heatwave temperature at that pixel (°C)
  - y position = heatwave amplification at that pixel (°C)
  - color      = vegetation land-cover class at that pixel

Pixels with no assigned vegetation class ("unclassified") are dropped entirely
rather than plotted in a neutral color. The four extreme bivariate corners
(cool-stable / hot-stable / cool-amplifying / hot-amplifying) are outlined with
a dashed rectangle at the same rank-quartile breakpoints (raw °C) used by the
f3_2 map classification, computed from the full population (not the sample).

Output
------
figures/bivariate_analysis/output/f3_2_patch_scatter/
  f3_2_patch_scatter_by_lulc_pm1p0deg.svg
  f3_2_patch_scatter_by_lulc_pm1p0deg.png
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
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from shapely.geometry import box

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_BASENAME = "f3_2_patch_scatter_by_lulc.py"


def infer_script_path() -> Path:
    """Resolve the script path for both file-run and interactive selection run."""
    file_obj = globals().get("__file__")
    if file_obj:
        return Path(file_obj).resolve()

    cwd = Path.cwd().resolve()
    candidates = [
        cwd / "figures" / "bivariate_analysis" / SCRIPT_BASENAME,
        cwd / SCRIPT_BASENAME,
        cwd / "bivariate_analysis" / SCRIPT_BASENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (cwd / SCRIPT_BASENAME).resolve()


def infer_scripts_root(script_path: Path) -> Path:
    """Find repository scripts root (expects DATA/ and figures/ folders)."""
    for parent in [script_path.parent, *script_path.parents]:
        if (parent / "DATA").exists() and (parent / "figures").exists():
            return parent
    return Path.cwd().resolve()


def initialize_runtime_paths() -> None:
    global SCRIPT_PATH, SCRIPTS_ROOT, DATA_DIR, FIGURES_ROOT
    global FIGURES_RESULTS_DIR, FIGURES_STYLE_DIR
    global ANALYSIS_DIR, ANALYSIS_OUTPUT_DIR, WORKDIR, GLOBAL_SETTINGS

    SCRIPT_PATH = infer_script_path()
    SCRIPTS_ROOT = infer_scripts_root(SCRIPT_PATH)
    DATA_DIR = SCRIPTS_ROOT / "DATA"
    FIGURES_ROOT = SCRIPTS_ROOT / "figures"
    FIGURES_RESULTS_DIR = FIGURES_ROOT / "results" / "figures"
    FIGURES_STYLE_DIR = FIGURES_ROOT / "2_results" / "figures"
    ANALYSIS_DIR = SCRIPT_PATH.parent
    ANALYSIS_OUTPUT_DIR = ANALYSIS_DIR / "output"

    WORKDIR = ANALYSIS_OUTPUT_DIR / "f3_2_patch_scatter"
    WORKDIR.mkdir(parents=True, exist_ok=True)

    GLOBAL_SETTINGS = FIGURES_STYLE_DIR / "global_plotting_settings.py"


initialize_runtime_paths()

HEATWAVE_INPUTS = {
    "2010": DATA_DIR / "predictions" / "2010" / "20100728",
    "2018": DATA_DIR / "predictions" / "2018" / "20180717",
    "2021": DATA_DIR / "predictions" / "2021" / "20210714",
}
BASELINE_MEAN_INPUT = DATA_DIR / "predictions" / "baseline" / "15cm_July_allday" / "pred_20000715_1000.tif"

MAP_TOLERANCE = 1.0
TOL_LABEL = str(MAP_TOLERANCE).replace(".", "p")
TARGET_DOMAIN_RASTER = (
    FIGURES_STYLE_DIR / "f4" / "rasters" / f"p90_loss_target_domain_tree_veg_nwn_pm{TOL_LABEL}deg.tif"
)

LULC_LAYERS = [
    ("Fields", [DATA_DIR / "LULC" / "lc_fields.gpkg"]),
    ("Trees 2–10 m", [DATA_DIR / "LULC" / "lc_trees2-10.gpkg"]),
    ("Trees 10–15 m", [DATA_DIR / "LULC" / "lc_trees10-15.gpkg"]),
    ("Trees 15–20 m", [DATA_DIR / "LULC" / "lc_trees15-20.gpkg"]),
    ("Trees >20 m", [DATA_DIR / "LULC" / "lc_trees_o20.gpkg"]),
    ("Other vegetation", [DATA_DIR / "LULC" / "lc_otherveg.gpkg"]),
]
PLOT_ORDER = ["Trees 2–10 m", "Trees 10–15 m", "Trees 15–20 m", "Trees >20 m", "Fields", "Other vegetation"]
DISPLAY_LABEL_MAP = {
    "Trees 2–10 m": "Trees 2–10 m",
    "Trees 10–15 m": "Trees 10–15 m",
    "Trees 15–20 m": "Trees 15–20 m",
    "Trees >20 m": "Trees >20 m",
    "Fields": "Fields",
    "Other vegetation": "Other veg.",
}
LULC_COLORS = {
    "Trees 2–10 m": "#1966D2",
    "Trees 10–15 m": "#1B5E20",
    "Trees 15–20 m": "#F5B041",
    "Trees >20 m": "#8B4513",
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

# Uniform random sample of individual pixels (not patches) drawn from the full
# classification-valid domain, kept small enough to render quickly and stay
# legible while still reading as a continuum.
MAX_SAMPLE_POINTS = 20000
RANDOM_SEED = 42

EXTREME_CLASSES = {
    1: {"label": "Cool-stable", "description": "low heatwave T, low amplification"},
    4: {"label": "Hot-stable", "description": "high heatwave T, low amplification"},
    13: {"label": "Cool-amplifying", "description": "low heatwave T, high amplification"},
    16: {"label": "Hot-amplifying", "description": "high heatwave T, high amplification"},
}

OUTPUT_BASENAME = f"f3_2_patch_scatter_by_lulc_pm{TOL_LABEL}deg"
LAND_COVER_LEGEND_OUTPUT_BASENAME = (
    f"f3_2_patch_scatter_land_cover_legend_pm{TOL_LABEL}deg"
)

# When True, the land-cover legend is removed from the scatterplot and
# exported as its own transparent SVG and PNG. When False, the legend remains
# attached to the scatterplot and no standalone legend file is created.
EXPORT_STANDALONE_LAND_COVER_LEGEND = True

PANEL_SIZE_CM = 15.0
AX_RECT = [0.14, 0.13, 0.70, 0.74]
LEGEND_LOC = "center left"
LEGEND_BBOX = (1.02, 0.5)

POINT_SIZE = 5.0
POINT_ALPHA = 0.35

# Standalone legend canvas size. The saved output is tightly cropped around
# the legend, so these values mainly provide a comfortable rendering canvas.
LEGEND_FIGSIZE_IN = (4.0, 2.8)

# =============================================================================
# STYLE (best-effort; falls back to plain matplotlib defaults)
# =============================================================================

def import_global_plotting_settings():
    if not GLOBAL_SETTINGS.exists():
        return None
    settings_dir = str(GLOBAL_SETTINGS.parent)
    if settings_dir not in sys.path:
        sys.path.insert(0, settings_dir)
    module_name = "global_plotting_settings"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def setup_style():
    global gps, STYLE
    initialize_runtime_paths()
    gps_local = import_global_plotting_settings()
    if gps_local is not None and hasattr(gps_local, "STYLE"):
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
    else:
        gps = None
        STYLE = None
        plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    mpl.rcParams["svg.fonttype"] = "none"

# =============================================================================
# RASTER HELPERS (duplicated intentionally to keep this script standalone)
# =============================================================================

@dataclass(frozen=True)
class RasterSurface:
    label: str
    path: Path
    hour_utc: int
    array: np.ndarray
    profile: dict


def check_required_files() -> None:
    required = [TARGET_DOMAIN_RASTER, BASELINE_MEAN_INPUT, *HEATWAVE_INPUTS.values()]
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
    raise FileNotFoundError(f"No unique baseline raster for {hour_utc:02d}:00 UTC. Available: {available}")


def aggregate_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    with np.errstate(all="ignore"):
        if AGGREGATION == "mean":
            return np.nanmean(stack, axis=0)
        if AGGREGATION == "max":
            return np.nanmax(stack, axis=0)
    raise ValueError(f"Unsupported AGGREGATION: {AGGREGATION}")


def hard_cut_to_domain(arr: np.ndarray, valid_domain: np.ndarray) -> np.ndarray:
    """Force all outside-domain cells to NaN before any downstream analysis."""
    out = arr.astype("float64", copy=True)
    out[~valid_domain] = np.nan
    return out


def compute_temperature_surfaces(target_profile: dict, valid_domain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heatwave_arrays = []
    mean_arrays = []

    for label, input_path in HEATWAVE_INPUTS.items():
        surface = select_heatwave_surface(label, input_path)
        heatwave = reproject_to_match(surface.array, surface.profile, target_profile, resampling=Resampling.bilinear)

        mean_path = matching_baseline_for_hour(BASELINE_MEAN_INPUT, surface.hour_utc)
        mean_arr, mean_profile = read_raster(mean_path)
        tmean = reproject_to_match(mean_arr, mean_profile, target_profile, resampling=Resampling.bilinear)

        heatwave = hard_cut_to_domain(heatwave, valid_domain)
        tmean = hard_cut_to_domain(tmean, valid_domain)

        heatwave_arrays.append(heatwave)
        mean_arrays.append(tmean)
        print(f"Matched mean baseline for {label}: {mean_path.name}")

    thw = aggregate_arrays(heatwave_arrays)
    tmean = aggregate_arrays(mean_arrays)
    offset = thw - tmean
    thw = hard_cut_to_domain(thw, valid_domain)
    tmean = hard_cut_to_domain(tmean, valid_domain)
    offset = hard_cut_to_domain(offset, valid_domain)
    return thw, tmean, offset

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


def classify_typology(thw: np.ndarray, offset: np.ndarray, valid_domain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = (
        valid_domain
        & np.isfinite(thw)
        & np.isfinite(offset)
        & (thw >= MIN_VALID_ABSOLUTE_TEMP_C)
    )
    if int(valid.sum()) == 0:
        raise ValueError("No valid pixels for bivariate classification.")

    print(
        f"Classification scope: valid-domain cells={int(valid_domain.sum()):,}; "
        f"ranked cells used for quartiles={int(valid.sum()):,}"
    )

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
    return bivar, r_thw, r_offset, valid

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
# PIXEL SAMPLING
# =============================================================================

def build_pixel_sample(
    thw: np.ndarray,
    offset: np.ndarray,
    lulc_masks: dict[str, np.ndarray],
    class_valid_mask: np.ndarray,
    max_points: int,
    seed: int,
) -> pd.DataFrame:
    """Uniform random sample of individual pixels across the whole domain.

    Every classification-valid pixel with an assigned vegetation class is a
    candidate. Sampling is unweighted and unstratified on purpose: a plain
    random sample from the true population preserves both the real joint
    density of (THW, offset) and the real land-cover composition, rather than
    forcing equal counts per class or per LULC category (which would distort
    both and hide the real correlation between the two axes).

    Pixels with no assigned vegetation class ("unclassified") are excluded
    from the candidate pool entirely, rather than sampled and shown in a
    neutral color.
    """
    lulc_id = np.full(thw.shape, -1, dtype=np.int16)
    for i, lulc in enumerate(PLOT_ORDER):
        lulc_id[lulc_masks[lulc]] = i

    candidate_mask = class_valid_mask & (lulc_id >= 0)
    rows, cols = np.nonzero(candidate_mask)
    n_candidates = rows.size
    print(f"Classified, samplable pixels: {n_candidates:,}")

    rng = np.random.default_rng(seed)
    if n_candidates > max_points:
        sel = rng.choice(n_candidates, size=max_points, replace=False)
        rows = rows[sel]
        cols = cols[sel]

    x_vals = thw[rows, cols]
    y_vals = offset[rows, cols]
    lulc_idx = lulc_id[rows, cols]
    lulc_names = np.array(PLOT_ORDER, dtype=object)[lulc_idx]

    return pd.DataFrame({
        "thw_degC": x_vals,
        "offset_degC": y_vals,
        "lulc": lulc_names,
        "lulc_label": [DISPLAY_LABEL_MAP.get(v, v) for v in lulc_names],
    })

# =============================================================================
# PLOTTING
# =============================================================================

def save_figure(fig: plt.Figure, basename: str) -> None:
    svg = WORKDIR / f"{basename}.svg"
    png = WORKDIR / f"{basename}.png"
    fig.savefig(svg, transparent=True, facecolor="none", edgecolor="none", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, dpi=300, transparent=True, facecolor="none", edgecolor="none", bbox_inches="tight", pad_inches=0.05)
    print(f"[OK] wrote {svg}")
    print(f"[OK] wrote {png}")


def draw_corner_boundaries(ax, x_breaks: np.ndarray, y_breaks: np.ndarray) -> None:
    """Outline the four extreme bivariate corners.

    Uses the same rank-quartile breakpoints (0/25/75/100th percentile of the
    classification domain, expressed as raw °C) that define the discrete
    cool-stable / hot-stable / cool-amplifying / hot-amplifying classes on the
    f3_2 map, so viewers can see where those four classes sit within the full,
    continuous patch distribution plotted here.
    """
    x_min, x_p25, x_p75, x_max = x_breaks
    y_min, y_p25, y_p75, y_max = y_breaks
    boxes = {
        "cool-\nstable": (x_min, x_p25, y_min, y_p25),
        "hot-\nstable": (x_p75, x_max, y_min, y_p25),
        "cool-\namplifying": (x_min, x_p25, y_p75, y_max),
        "hot-\namplifying": (x_p75, x_max, y_p75, y_max),
    }
    for label, (x0, x1, y0, y1) in boxes.items():
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor="#404040", linestyle="--", linewidth=1.0, zorder=1,
        ))
        ax.text(
            (x0 + x1) / 2.0, (y0 + y1) / 2.0, label,
            ha="center", va="center", fontsize=11.5, color="#404040",
            style="italic", zorder=1, linespacing=0.95,
        )


def build_land_cover_legend_handles() -> list[Patch]:
    """Build legend handles shared by the main and standalone figures."""
    return [
        Patch(
            facecolor=LULC_COLORS[class_name],
            edgecolor="black",
            linewidth=0.3,
            label=DISPLAY_LABEL_MAP.get(class_name, class_name),
        )
        for class_name in PLOT_ORDER
    ]


def export_land_cover_legend() -> None:
    """Export the land-cover legend as a standalone transparent SVG and PNG."""
    handles = build_land_cover_legend_handles()
    fig = plt.figure(figsize=LEGEND_FIGSIZE_IN, dpi=300)

    legend = fig.legend(
        handles=handles,
        title="Land cover",
        loc="center",
        frameon=False,
        fontsize=11,
        title_fontsize=11.5,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

    svg = WORKDIR / f"{LAND_COVER_LEGEND_OUTPUT_BASENAME}.svg"
    png = WORKDIR / f"{LAND_COVER_LEGEND_OUTPUT_BASENAME}.png"

    fig.savefig(
        svg,
        transparent=True,
        facecolor="none",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.05,
        bbox_extra_artists=[legend],
    )
    fig.savefig(
        png,
        dpi=300,
        transparent=True,
        facecolor="none",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.05,
        bbox_extra_artists=[legend],
    )

    print(f"[OK] wrote {svg}")
    print(f"[OK] wrote {png}")
    plt.close(fig)


def plot_pixel_scatter(
    sample_df: pd.DataFrame,
    median_thw: float,
    median_offset: float,
    x_breaks: np.ndarray,
    y_breaks: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(PANEL_SIZE_CM / 2.54, PANEL_SIZE_CM / 2.54), dpi=300)
    ax = fig.add_axes(AX_RECT)

    # Reference crosshair at the domain median (equivalent to the rank-0.5
    # split used for classification), now expressed in raw degrees so the
    # value ranges of both axes stay visible.
    ax.axvline(median_thw, color="#b0b0b0", linewidth=0.9, alpha=0.8, zorder=0)
    ax.axhline(median_offset, color="#b0b0b0", linewidth=0.9, alpha=0.8, zorder=0)

    x_vals = sample_df["thw_degC"].to_numpy(dtype=float)
    y_vals = sample_df["offset_degC"].to_numpy(dtype=float)
    colors = [LULC_COLORS[lulc] for lulc in sample_df["lulc"]]

    # rasterized=True keeps the exported SVG a manageable size and fast to
    # open, since a few thousand individually-editable vector markers would
    # otherwise bloat the file; no marker edge keeps dense overlapping pixels
    # from turning into a solid black smear.
    ax.scatter(
        x_vals,
        y_vals,
        s=POINT_SIZE,
        c=colors,
        alpha=POINT_ALPHA,
        edgecolor="none",
        linewidth=0.0,
        zorder=2,
        rasterized=True,
    )

    x_pad = 0.06 * (np.nanmax(x_vals) - np.nanmin(x_vals) + 1e-6)
    y_pad = 0.06 * (np.nanmax(y_vals) - np.nanmin(y_vals) + 1e-6)
    ax.set_xlim(np.nanmin(x_vals) - x_pad, np.nanmax(x_vals) + x_pad)
    ax.set_ylim(np.nanmin(y_vals) - y_pad, np.nanmax(y_vals) + y_pad)

    draw_corner_boundaries(ax, x_breaks, y_breaks)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)

    fs_axis = 14
    ax.text(0.5, -0.10, "heatwave T (°C)", ha="center", va="top",
            fontsize=fs_axis, fontweight="bold", transform=ax.transAxes)

    ax.text(-0.14, 0.5, "heatwave ΔT to July mean (°C)", ha="center", va="center",
            fontsize=fs_axis, fontweight="bold", rotation=90, transform=ax.transAxes)

    # A single option controls both outputs: enabling standalone legend export
    # automatically suppresses the legend in the scatterplot.
    if not EXPORT_STANDALONE_LAND_COVER_LEGEND:
        ax.legend(
            handles=build_land_cover_legend_handles(),
            title="Land cover",
            loc=LEGEND_LOC,
            bbox_to_anchor=LEGEND_BBOX,
            frameon=False,
            fontsize=11,
            title_fontsize=11.5,
            handletextpad=0.5,
            borderaxespad=0.0,
        )

    save_figure(fig, OUTPUT_BASENAME)
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    setup_style()
    print("Resolved paths:")
    print(f"  output dir:  {WORKDIR}")
    print(f"  target domain: {TARGET_DOMAIN_RASTER}")
    print(f"  analysis hour: {EXPECTED_PEAK_LOCAL_HOUR:02d}:00 local = {TARGET_HEATWAVE_UTC_HOUR:02d}:00 UTC")

    check_required_files()

    target_arr, target_profile = read_raster(TARGET_DOMAIN_RASTER)
    valid_domain = np.isfinite(target_arr) & (target_arr > 0)
    print(f"Valid domain pixels: {int(valid_domain.sum()):,}")

    thw, tmean, offset = compute_temperature_surfaces(target_profile, valid_domain)
    bivar, r_thw, r_offset, class_valid_mask = classify_typology(thw, offset, valid_domain)

    median_thw = float(np.nanmedian(thw[class_valid_mask]))
    median_offset = float(np.nanmedian(offset[class_valid_mask]))
    print(f"Domain median absolute heatwave T = {median_thw:.2f} degC; median amplification = {median_offset:.2f} degC")

    x_breaks = np.nanpercentile(thw[class_valid_mask], [0, 25, 75, 100])
    y_breaks = np.nanpercentile(offset[class_valid_mask], [0, 25, 75, 100])

    lulc_masks = load_lulc_masks(target_profile, class_valid_mask)

    sample_df = build_pixel_sample(thw, offset, lulc_masks, class_valid_mask, MAX_SAMPLE_POINTS, RANDOM_SEED)
    print(f"Sampled pixels plotted: {len(sample_df):,}")
    print(sample_df.groupby("lulc_label")["thw_degC"].count().to_string())

    plot_pixel_scatter(sample_df, median_thw, median_offset, x_breaks, y_breaks)

    if EXPORT_STANDALONE_LAND_COVER_LEGEND:
        export_land_cover_legend()


if __name__ == "__main__":
    main()
