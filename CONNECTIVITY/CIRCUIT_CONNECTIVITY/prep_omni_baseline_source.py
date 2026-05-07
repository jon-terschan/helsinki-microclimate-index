#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import geometry_mask

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")

PREDICTIONS_DIR = BASE_DIR / "predictions"
PREDICTORSTACK_DIR = BASE_DIR / "predictorstack"

OUT_DIR = BASE_DIR / "omniscape" / "sources"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_ROOT = PREDICTIONS_DIR / "baseline" / "15cm_July_allday"
P90_ROOT = PREDICTIONS_DIR / "baseline" / "15cm_July_allday_p90"

TREE_PATH = PREDICTORSTACK_DIR / "TREE_FRAC_10m.tif"
NWN_PATH = PREDICTORSTACK_DIR / "NWN_FRAC_10m.tif"

NA_CLIPPER_PATH = BASE_DIR / "NA_clipper.gpkg"

TARGET_UTC_HOURS = {9, 10, 11, 12, 13}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RasterRef:
    profile: dict
    transform: object
    crs: object
    width: int
    height: int
    nodata: float | int | None


# =============================================================================
# HELPERS
# =============================================================================

def parse_pred_file(path: Path) -> Tuple[str, int, int]:

    m = re.match(r"pred_(\d{8})_(\d{2})(\d{2})\.tif$", path.name)

    if not m:
        raise ValueError(f"Unexpected filename: {path.name}")

    return m.group(1), int(m.group(2)), int(m.group(3))


def find_prediction_files(root: Path, target_hours: set[int]) -> List[Path]:

    files = []

    for tif in root.rglob("pred_*.tif"):

        _, hour, _ = parse_pred_file(tif)

        if hour in target_hours:
            files.append(tif)

    return sorted(files)


def read_raster(path: Path):

    with rasterio.open(path) as src:

        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

        ref = RasterRef(
            profile=src.profile.copy(),
            transform=src.transform,
            crs=src.crs,
            width=src.width,
            height=src.height,
            nodata=src.nodata,
        )

    return arr, ref


def average_rasters(paths: List[Path], clip_mask=None):

    arrays = []
    ref = None

    for p in paths:

        arr, this_ref = read_raster(p)

        if clip_mask is not None:
            arr = np.where(clip_mask, np.nan, arr)

        if ref is None:
            ref = this_ref

        arrays.append(arr)

    stack = np.stack(arrays)

    mean = np.nanmean(stack, axis=0)

    return mean, ref


def read_mask(path: Path, ref: RasterRef):

    with rasterio.open(path) as src:

        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

    return arr


def mask_from_gpkg(path: Path, ref: RasterRef):

    gdf = gpd.read_file(path)

    if gdf.crs != ref.crs:
        gdf = gdf.to_crs(ref.crs)

    geoms = [
        geom.__geo_interface__
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    ]

    return geometry_mask(
        geoms,
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        invert=True,
    )


def write_geotiff(path: Path, arr: np.ndarray, ref: RasterRef):

    profile = ref.profile.copy()

    profile.update(
        dtype="float32",
        count=1,
        compress="deflate",
        tiled=False,
        nodata=-9999.0,
    )

    out = np.where(np.isnan(arr), -9999.0, arr).astype(np.float32)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def plot_raster(
    arr: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str,
    vmin=None,
    vmax=None,
):

    plt.figure(figsize=(8, 8))

    im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.title(title)
    plt.axis("off")

    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def robust_normalize(arr, mask):

    vals = arr[mask & np.isfinite(arr)]

    p1 = np.nanpercentile(vals, 1)
    p99 = np.nanpercentile(vals, 99)

    out = (arr - p1) / (p99 - p1)

    return np.clip(out, 0, 1)


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # LOAD REFERENCE GRID
    # -------------------------------------------------------------------------

    first_file = find_prediction_files(
        BASELINE_ROOT,
        TARGET_UTC_HOURS,
    )[0]

    _, ref = read_raster(first_file)

    # -------------------------------------------------------------------------
    # LOAD MASKS
    # -------------------------------------------------------------------------

    na_mask = mask_from_gpkg(NA_CLIPPER_PATH, ref)

    tree = read_mask(TREE_PATH, ref)
    nwn = read_mask(NWN_PATH, ref)

    valid_mask = (
        (np.nan_to_num(tree) > 0) |
        (np.nan_to_num(nwn) > 0)
    )

    analysis_mask = valid_mask & (~na_mask)

    # -------------------------------------------------------------------------
    # LOAD + AVERAGE BASELINE
    # -------------------------------------------------------------------------

    baseline_files = find_prediction_files(
        BASELINE_ROOT,
        TARGET_UTC_HOURS,
    )

    baseline_mean, _ = average_rasters(
        baseline_files,
        clip_mask=na_mask,
    )

    baseline_mean = np.where(
        analysis_mask,
        baseline_mean,
        np.nan,
    )

    # -------------------------------------------------------------------------
    # LOAD + AVERAGE P90
    # -------------------------------------------------------------------------

    p90_files = find_prediction_files(
        P90_ROOT,
        TARGET_UTC_HOURS,
    )

    p90_mean, _ = average_rasters(
        p90_files,
        clip_mask=na_mask,
    )

    p90_mean = np.where(
        analysis_mask,
        p90_mean,
        np.nan,
    )

    # -------------------------------------------------------------------------
    # AGGREGATED SOLAR PEAK MEAN SURFACE
    # -------------------------------------------------------------------------

    solar_peak_mean = (baseline_mean + p90_mean) / 2.0

    solar_peak_mean = np.where(
        analysis_mask,
        solar_peak_mean,
        np.nan,
    )

    # -------------------------------------------------------------------------
    # THERMAL AMPLIFICATION
    # -------------------------------------------------------------------------

    delta = p90_mean - baseline_mean

    delta = np.where(
        analysis_mask,
        delta,
        np.nan,
    )

    # -------------------------------------------------------------------------
    # COMPONENTS
    # -------------------------------------------------------------------------

    # Low baseline temperatures = high score
    baseline_norm = robust_normalize(
        baseline_mean,
        analysis_mask,
    )

    baseline_coolness = 1.0 - baseline_norm

    # Low p90 temperatures = highest importance
    p90_norm = robust_normalize(
        p90_mean,
        analysis_mask,
    )

    extreme_coolness = 1.0 - p90_norm

    # Low amplification = high score
    delta_norm = robust_normalize(
        delta,
        analysis_mask,
    )

    stability = 1.0 - delta_norm

    # -------------------------------------------------------------------------
    # FINAL SOURCE STRENGTH
    # -------------------------------------------------------------------------

    source = (
        0.60 * extreme_coolness +
        0.25 * baseline_coolness +
        0.15 * stability
    )

    source = np.clip(source, 0, 1)

    source = np.where(
        analysis_mask,
        source,
        np.nan,
    )

    # -------------------------------------------------------------------------
    # EXPORT RASTERS
    # -------------------------------------------------------------------------

    baseline_out = OUT_DIR / "baseline_peak_mean.tif"
    p90_out = OUT_DIR / "p90_peak_mean.tif"

    solar_peak_out = OUT_DIR / "solar_peak_mean_surface.tif"

    delta_out = OUT_DIR / "baseline_vs_p90_delta.tif"

    baseline_cool_out = OUT_DIR / "baseline_coolness.tif"
    extreme_cool_out = OUT_DIR / "extreme_coolness.tif"
    stability_out = OUT_DIR / "thermal_stability.tif"

    source_out = OUT_DIR / "source_relative_coolness.tif"

    write_geotiff(baseline_out, baseline_mean, ref)
    write_geotiff(p90_out, p90_mean, ref)

    write_geotiff(solar_peak_out, solar_peak_mean, ref)

    write_geotiff(delta_out, delta, ref)

    write_geotiff(baseline_cool_out, baseline_coolness, ref)
    write_geotiff(extreme_cool_out, extreme_coolness, ref)
    write_geotiff(stability_out, stability, ref)

    write_geotiff(source_out, source, ref)

    print(f"[OK] wrote {baseline_out}")
    print(f"[OK] wrote {p90_out}")
    print(f"[OK] wrote {solar_peak_out}")
    print(f"[OK] wrote {delta_out}")
    print(f"[OK] wrote {baseline_cool_out}")
    print(f"[OK] wrote {extreme_cool_out}")
    print(f"[OK] wrote {stability_out}")
    print(f"[OK] wrote {source_out}")

    # -------------------------------------------------------------------------
    # EXPORT FIGURES
    # -------------------------------------------------------------------------

    plot_raster(
        baseline_mean,
        OUT_DIR / "qc_baseline_peak_mean.png",
        title="Baseline peak mean",
        cmap="inferno",
    )

    plot_raster(
        p90_mean,
        OUT_DIR / "qc_p90_peak_mean.png",
        title="P90 peak mean",
        cmap="inferno",
    )

    plot_raster(
        solar_peak_mean,
        OUT_DIR / "qc_solar_peak_mean_surface.png",
        title="Aggregated solar peak mean surface",
        cmap="inferno",
    )

    plot_raster(
        delta,
        OUT_DIR / "qc_delta_relative_coolness.png",
        title="Thermal amplification (p90 - baseline)",
        cmap="coolwarm",
    )

    plot_raster(
        baseline_coolness,
        OUT_DIR / "qc_baseline_coolness.png",
        title="Baseline coolness",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plot_raster(
        extreme_coolness,
        OUT_DIR / "qc_extreme_coolness.png",
        title="Extreme-state coolness",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plot_raster(
        stability,
        OUT_DIR / "qc_thermal_stability.png",
        title="Thermal stability",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plot_raster(
        source,
        OUT_DIR / "qc_source_strength.png",
        title="Final source strength",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    print("[OK] wrote QC figures")


if __name__ == "__main__":
    main()