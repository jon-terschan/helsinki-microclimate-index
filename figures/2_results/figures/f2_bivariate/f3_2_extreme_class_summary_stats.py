#!/usr/bin/env python3
"""
Figure f3_2 companion: summary statistics for the four extreme bivariate classes.

Uses the same valid vegetated prediction domain, hard domain cut, and bivariate
classification logic as the f3_2 4x4 heatwave-response map:

  x axis: absolute mean heatwave temperature, THW
  y axis: heatwave amplification from average conditions, THW - Tavg (ΔT)

For each of the four extreme corner classes:
  1  = low THW, low ΔT   (cool-stable)
  4  = high THW, low ΔT  (hot-stable)
  13 = low THW, high ΔT  (cool-amplifying)
  16 = high THW, high ΔT (hot-amplifying)

this script reports:
  - average / median normal (baseline mean) temperature
  - average / median heatwave temperature
  - average / median amplification (THW - Tavg)
  - pixel count and area

Output
------
figures/bivariate_analysis/output/f3_2_extreme_summary/tables/
  f3_2_extreme_class_summary_stats_pm1p0deg.csv
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_BASENAME = "f3_2_extreme_class_summary_stats.py"


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
    global ANALYSIS_DIR, ANALYSIS_OUTPUT_DIR, WORKDIR, TABLE_DIR

    SCRIPT_PATH = infer_script_path()
    SCRIPTS_ROOT = infer_scripts_root(SCRIPT_PATH)
    DATA_DIR = SCRIPTS_ROOT / "DATA"
    FIGURES_ROOT = SCRIPTS_ROOT / "figures"
    FIGURES_RESULTS_DIR = FIGURES_ROOT / "results" / "figures"
    FIGURES_STYLE_DIR = FIGURES_ROOT / "2_results" / "figures"
    ANALYSIS_DIR = SCRIPT_PATH.parent
    ANALYSIS_OUTPUT_DIR = ANALYSIS_DIR / "output"

    WORKDIR = ANALYSIS_OUTPUT_DIR / "f3_2_extreme_summary"
    WORKDIR.mkdir(parents=True, exist_ok=True)

    TABLE_DIR = WORKDIR / "tables"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


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

# =============================================================================
# SETTINGS
# =============================================================================

LOCAL_UTC_OFFSET_HOURS = 3
EXPECTED_PEAK_LOCAL_HOUR = 13
TARGET_HEATWAVE_UTC_HOUR = (EXPECTED_PEAK_LOCAL_HOUR - LOCAL_UTC_OFFSET_HOURS) % 24
AGGREGATION = "mean"
MIN_VALID_ABSOLUTE_TEMP_C = 15.0

N_CLASSES = 4
CLASS_BREAKS = [0.25, 0.50, 0.75]

OUTPUT_BASENAME = f"f3_2_extreme_class_summary_stats_pm{TOL_LABEL}deg"
OUT_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}.csv"

# Corner class IDs. class_id = offset_quartile * N_CLASSES + temp_quartile + 1
EXTREME_CLASSES = {
    1: {"label": "Cool-stable", "description": "low heatwave T, low amplification"},
    4: {"label": "Hot-stable", "description": "high heatwave T, low amplification"},
    13: {"label": "Cool-amplifying", "description": "low heatwave T, high amplification"},
    16: {"label": "Hot-amplifying", "description": "high heatwave T, high amplification"},
}
CORNER_ORDER = [1, 13, 4, 16]

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

        # Hard domain cut before aggregation, so invalid/outside cells can never
        # leak into temperature offsets or rank-based class boundaries.
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
    """Return empirical rank scaled to 0..1, using average ranks for ties."""
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
# SUMMARY
# =============================================================================

def cell_area_m2(profile: dict) -> float:
    return abs(float(profile["transform"].a) * float(profile["transform"].e))


def safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.nanmean(vals)) if vals.size else np.nan


def safe_median(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.nanmedian(vals)) if vals.size else np.nan


def summarize_extreme_classes(
    bivar: np.ndarray,
    thw: np.ndarray,
    tmean: np.ndarray,
    offset: np.ndarray,
    profile: dict,
) -> pd.DataFrame:
    area = cell_area_m2(profile)
    rows = []
    for class_id in CORNER_ORDER:
        info = EXTREME_CLASSES[class_id]
        mask = bivar == class_id
        count = int(mask.sum())
        rows.append({
            "corner_id": class_id,
            "corner_label": info["label"],
            "description": info["description"],
            "pixel_count": count,
            "area_m2": count * area,
            "area_km2": (count * area) / 1_000_000.0,
            "mean_normal_temp_degC": safe_mean(tmean, mask),
            "median_normal_temp_degC": safe_median(tmean, mask),
            "mean_heatwave_temp_degC": safe_mean(thw, mask),
            "median_heatwave_temp_degC": safe_median(thw, mask),
            "mean_amplification_degC": safe_mean(offset, mask),
            "median_amplification_degC": safe_median(offset, mask),
        })
    return pd.DataFrame.from_records(rows)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    initialize_runtime_paths()
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

    summary_df = summarize_extreme_classes(bivar, thw, tmean, offset, target_profile)
    summary_df.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV}")

    print("\n[EXTREME CLASS SUMMARY STATISTICS]")
    display_cols = [
        "corner_label", "pixel_count", "area_km2",
        "mean_normal_temp_degC", "mean_heatwave_temp_degC", "mean_amplification_degC",
        "median_normal_temp_degC", "median_heatwave_temp_degC", "median_amplification_degC",
    ]
    print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
