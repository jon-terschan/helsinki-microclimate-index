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
from scipy.stats import rankdata

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")

PREDICTIONS_DIR = BASE_DIR / "predictions"
PREDICTORSTACK_DIR = BASE_DIR / "predictorstack"

SOURCE_OUT_DIR = BASE_DIR / "omniscape" / "sources" 
CONDITIONS_OUT_DIR = BASE_DIR / "omniscape" / "conditions" 

SOURCE_OUT_DIR.mkdir(parents=True, exist_ok=True)
CONDITIONS_OUT_DIR.mkdir(parents=True, exist_ok=True)

AVG_ROOT = PREDICTIONS_DIR / "baseline" / "15cm_July_allday"
P90_ROOT = PREDICTIONS_DIR / "baseline" / "15cm_July_allday_p90"

HEATWAVE_ROOTS = {
    "condition_heatwave_2010": PREDICTIONS_DIR / "2010",
    "condition_heatwave_2018": PREDICTIONS_DIR / "2018",
    "condition_heatwave_2021": PREDICTIONS_DIR / "2021",
}

TREE_PATH = PREDICTORSTACK_DIR / "TREE_FRAC_10m.tif"
NWN_PATH = PREDICTORSTACK_DIR / "NWN_FRAC_10m.tif"

NA_CLIPPER_PATH = BASE_DIR / "NA_clipper.gpkg"
if not NA_CLIPPER_PATH.exists():
    NA_CLIPPER_PATH = BASE_DIR / "NA_CLIPPER.gpkg"

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

    if not arrays:
        raise FileNotFoundError("No rasters found for averaging.")

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


def percentile_rank_01(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=np.float32)
    valid = mask & np.isfinite(arr)
    vals = arr[valid]

    if vals.size == 0:
        return out

    if vals.size == 1:
        out[valid] = 0.0
        return out

    ranks = rankdata(vals, method="average").astype(np.float32)
    pr = (ranks - 1.0) / (vals.size - 1.0)
    out[valid] = pr
    return out


def export_condition_surface(
    name: str,
    arr: np.ndarray,
    ref: RasterRef,
    title: str,
) -> None:
    folder = CONDITIONS_OUT_DIR / name
    folder.mkdir(parents=True, exist_ok=True)

    tif_path = folder / f"{name}.tif"
    png_path = folder / f"qc_{name}.png"

    write_geotiff(tif_path, arr, ref)
    plot_raster(
        arr,
        png_path,
        title=title,
        cmap="inferno",
    )

    print(f"[OK] wrote {tif_path}")
    print(f"[OK] wrote {png_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # LOAD REFERENCE GRID
    # -------------------------------------------------------------------------

    first_file = find_prediction_files(AVG_ROOT, TARGET_UTC_HOURS)[0]
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
    # CONDITION SURFACES
    # -------------------------------------------------------------------------

    scenario_means = {}

    avg_files = find_prediction_files(AVG_ROOT, TARGET_UTC_HOURS)
    avg_mean, _ = average_rasters(avg_files, clip_mask=na_mask)
    avg_mean = np.where(analysis_mask, avg_mean, np.nan)
    scenario_means["condition_average"] = avg_mean

    p90_files = find_prediction_files(P90_ROOT, TARGET_UTC_HOURS)
    p90_mean, _ = average_rasters(p90_files, clip_mask=na_mask)
    p90_mean = np.where(analysis_mask, p90_mean, np.nan)
    scenario_means["condition_p90"] = p90_mean

    for name, root in HEATWAVE_ROOTS.items():
        files = find_prediction_files(root, TARGET_UTC_HOURS)
        mean, _ = average_rasters(files, clip_mask=na_mask)
        mean = np.where(analysis_mask, mean, np.nan)
        scenario_means[name] = mean

    # export all condition surfaces + figures
    export_condition_surface(
        "condition_average",
        scenario_means["condition_average"],
        ref,
        "Condition surface: average mean peak temperature",
    )

    export_condition_surface(
        "condition_p90",
        scenario_means["condition_p90"],
        ref,
        "Condition surface: p90 mean peak temperature",
    )

    export_condition_surface(
        "condition_heatwave_2010",
        scenario_means["condition_heatwave_2010"],
        ref,
        "Condition surface: heatwave 2010 mean peak temperature",
    )

    export_condition_surface(
        "condition_heatwave_2018",
        scenario_means["condition_heatwave_2018"],
        ref,
        "Condition surface: heatwave 2018 mean peak temperature",
    )

    export_condition_surface(
        "condition_heatwave_2021",
        scenario_means["condition_heatwave_2021"],
        ref,
        "Condition surface: heatwave 2021 mean peak temperature",
    )

    # -------------------------------------------------------------------------
    # SOURCE INPUTS
    # -------------------------------------------------------------------------

    condition_average = avg_mean
    condition_p90 = p90_mean

    delta = condition_p90 - condition_average
    delta = np.where(analysis_mask, delta, np.nan)

    # -------------------------------------------------------------------------
    # SIMPLE DEFENSIBLE SOURCE
    #
    # Source = sqrt(C_E * S)
    #
    # C_E = 1 - percentile_rank(p90 temperature)
    # S   = 1 - percentile_rank(delta)
    #
    # High source = cool in the extreme state AND stable relative to average
    # -------------------------------------------------------------------------

    rank_E = percentile_rank_01(condition_p90, analysis_mask)
    rank_D = percentile_rank_01(delta, analysis_mask)

    C_E = 1.0 - rank_E
    S = 1.0 - rank_D

    source = np.sqrt(np.clip(C_E * S, 0.0, 1.0))
    source = np.where(analysis_mask, source, np.nan)

    # -------------------------------------------------------------------------
    # EXPORT SOURCE TIFF
    # -------------------------------------------------------------------------

    source_out = SOURCE_OUT_DIR / "source_p90_coolness_stability.tif"
    write_geotiff(source_out, source, ref)
    print(f"[OK] wrote {source_out}")

    # -------------------------------------------------------------------------
    # QC FIGURES FOR SOURCE
    # -------------------------------------------------------------------------

    plot_raster(
        condition_average,
        SOURCE_OUT_DIR / "qc_condition_average_surface.png",
        title="Condition surface used for source: average mean peak temperature",
        cmap="inferno",
    )

    plot_raster(
        condition_p90,
        SOURCE_OUT_DIR / "qc_condition_p90_surface.png",
        title="Condition surface used for source: p90 mean peak temperature",
        cmap="inferno",
    )

    plot_raster(
        delta,
        SOURCE_OUT_DIR / "qc_delta_p90_minus_average.png",
        title="Thermal amplification: p90 - average",
        cmap="coolwarm",
    )

    plot_raster(
        rank_E,
        SOURCE_OUT_DIR / "qc_rank_p90_temperature.png",
        title="Percentile rank of p90 temperature",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plot_raster(
        rank_D,
        SOURCE_OUT_DIR / "qc_rank_delta.png",
        title="Percentile rank of thermal amplification",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plot_raster(
        C_E,
        SOURCE_OUT_DIR / "qc_extreme_coolness_score.png",
        title="Extreme-state coolness score",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plot_raster(
        S,
        SOURCE_OUT_DIR / "qc_stability_score.png",
        title="Thermal stability score",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plot_raster(
        source,
        SOURCE_OUT_DIR / "qc_source_p90_coolness_stability.png",
        title="Final source: sqrt(extreme coolness × stability)",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    print("[OK] wrote source QC figures")


if __name__ == "__main__":
    main()