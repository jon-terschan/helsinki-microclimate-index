#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import geometry_mask

warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

BASE_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
PREDICTIONS_DIR = BASE_DIR / "predictions"
PREDICTORSTACK_DIR = BASE_DIR / "predictorstack"
SOURCE_OUT_DIR = BASE_DIR / "omniscape" / "sources"

NWN_PATH = PREDICTORSTACK_DIR / "NWN_FRAC_10m.tif"
TREE_PATH = PREDICTORSTACK_DIR / "TREE_FRAC_10m.tif"
NA_CLIPPER_PATH = BASE_DIR / "NA_clipper.gpkg"
AOI_PATH = BASE_DIR / "aoi_outer_buffer.gpkg"

SCENARIO_ROOTS = {
    "baseline_avg": PREDICTIONS_DIR / "baseline" / "15cm_July_allday",
    "baseline_p90": PREDICTIONS_DIR / "baseline" / "15cm_July_allday_p90",
    "h2010": PREDICTIONS_DIR / "2010",
    "h2018": PREDICTIONS_DIR / "2018",
    "h2021": PREDICTIONS_DIR / "2021",
}

TARGET_UTC_HOURS = {9, 10, 11, 12, 13} # which hours to include in the source calculation (local time 12-16)
REFERENCE_PERCENTILES = [75, 90, 95] # which percentiles to use as reference temperatures for source calculation

VALID_MASK_THRESHOLD = 0.0  # minimum per pixel fraction of valid green area required to keep a cell in the source

@dataclass
class RasterRef:
    profile: dict
    transform: object
    crs: object
    width: int
    height: int
    nodata: float | int | None


# -----------------------------------------------------------------------------
# IO HELPERS
# -----------------------------------------------------------------------------

def parse_pred_file(path: Path) -> Tuple[str, int, int]:
    m = re.match(r"pred_(\d{8})_(\d{2})(\d{2})\.tif$", path.name)
    if not m:
        raise ValueError(f"Unexpected prediction filename format: {path.name}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def find_prediction_files(root: Path, target_hours: set[int]) -> List[Path]:
    files = []
    for tif in root.rglob("pred_*.tif"):
        _, hour, _ = parse_pred_file(tif)
        if hour in target_hours:
            files.append(tif)
    return sorted(files)


def read_raster(path: Path, clip_mask: np.ndarray | None = None) -> Tuple[np.ndarray, RasterRef]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        if clip_mask is not None:
            arr = np.where(clip_mask, np.nan, arr)

        ref = RasterRef(src.profile.copy(), src.transform, src.crs,
                        src.width, src.height, src.nodata)
    return arr, ref


def average_rasters(paths: List[Path], clip_mask=None):
    arrays = []
    ref = None

    for p in paths:
        arr, this_ref = read_raster(p, clip_mask)
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

    geoms = [g.__geo_interface__ for g in gdf.geometry if g is not None]

    return geometry_mask(
        geoms,
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        invert=True,
    )


# -----------------------------------------------------------------------------
# CORE
# -----------------------------------------------------------------------------

def build_valid_domain(nwn, tree):
    return (np.nan_to_num(nwn) > 0) | (np.nan_to_num(tree) > 0)


def write_geotiff(path, arr, ref):
    profile = ref.profile.copy()
    profile.update(dtype="float32", count=1)

    out = np.where(np.isnan(arr), -9999, arr).astype(np.float32)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def scale_global(arr, scale):
    return np.clip(arr / scale, 0, 1)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():

    SOURCE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    first_root = next(iter(SCENARIO_ROOTS.values()))
    first_file = find_prediction_files(first_root, TARGET_UTC_HOURS)[0]
    _, ref = read_raster(first_file)

    nwn = read_mask(NWN_PATH, ref)
    tree = read_mask(TREE_PATH, ref)
    valid = build_valid_domain(nwn, tree)

    na_mask = mask_from_gpkg(NA_CLIPPER_PATH, ref)
    aoi_mask = mask_from_gpkg(AOI_PATH, ref)

    analysis_mask = valid & (~na_mask)

    # -------------------------------------------------------------------------
    # SCENARIO MEANS
    # -------------------------------------------------------------------------
    scenario_means = {}

    for name, root in SCENARIO_ROOTS.items():
        files = find_prediction_files(root, TARGET_UTC_HOURS)
        mean, _ = average_rasters(files, na_mask)
        scenario_means[name] = mean

    # -------------------------------------------------------------------------
    # GLOBAL REFERENCE TEMPERATURES
    # -------------------------------------------------------------------------
    pooled = np.concatenate([
        scenario_means[s][analysis_mask & np.isfinite(scenario_means[s])]
        for s in scenario_means
    ])

    reference_temps = {
        p: float(np.nanpercentile(pooled, p))
        for p in REFERENCE_PERCENTILES
    }

    print("Reference temps:", reference_temps)

    # -------------------------------------------------------------------------
    # RAW SOURCES
    # -------------------------------------------------------------------------
    raw_sources = {}

    for s, temp in scenario_means.items():
        for p in REFERENCE_PERCENTILES:

            delta = reference_temps[p] - temp

            # --- CHANGE 1: SOFT THRESHOLD ---
            # Instead of hard truncation (max(delta,0)),
            # allow negative values but dampen them
            raw = np.where(delta > 0, delta, delta * 0.25)

            raw = np.where(analysis_mask, raw, np.nan)

            raw_sources[(s, p)] = raw

    # -------------------------------------------------------------------------
    # GLOBAL SCALING
    # -------------------------------------------------------------------------

    pooled_raw = np.concatenate([
        arr[analysis_mask & np.isfinite(arr)]
        for arr in raw_sources.values()
    ])

    # --- CHANGE 2: ROBUST SCALING ---
    # Use percentile instead of max to avoid extreme compression
    scale = float(np.nanpercentile(pooled_raw, 99))

    print("Scaling value (p99):", scale)

    # -------------------------------------------------------------------------
    # WRITE OUTPUTS
    # -------------------------------------------------------------------------
    written = {}

    for s in scenario_means:
        written[s] = {}

        for p in REFERENCE_PERCENTILES:

            out_dir = SOURCE_OUT_DIR / f"ref_p{p}" / s
            out_dir.mkdir(parents=True, exist_ok=True)

            src = scale_global(raw_sources[(s, p)], scale)
            src = np.where(aoi_mask, src, np.nan)

            out = out_dir / f"omni_source_{s}_refp{p}.tif"
            write_geotiff(out, src, ref)

            written[s][p] = out

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------
    for s in written:
        fig, axes = plt.subplots(1, len(REFERENCE_PERCENTILES), figsize=(15,5))

        for ax, p in zip(axes, REFERENCE_PERCENTILES):
            with rasterio.open(written[s][p]) as src:
                arr = src.read(1)
                arr[arr == src.nodata] = np.nan

            im = ax.imshow(arr, vmin=0, vmax=1)
            ax.set_title(f"p{p}")
            ax.axis("off")
            plt.colorbar(im, ax=ax)

        plt.savefig(SOURCE_OUT_DIR / f"{s}_qc.png")
        plt.close()

    with open(SOURCE_OUT_DIR / "reference_temps.json", "w") as f:
        json.dump(reference_temps, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()