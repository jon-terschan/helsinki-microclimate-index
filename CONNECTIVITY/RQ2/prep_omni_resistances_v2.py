#creates a permissive and a conservative resistance surface for omniscape
# context specific urban penality: large urban patches get stronger penalized, small isolated pixels less so.

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import geometry_mask
from scipy.ndimage import uniform_filter

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
STACK = BASE / "predictorstack"
OUT = BASE / "omniscape" / "resistance"
OUT.mkdir(parents=True, exist_ok=True)

AOI_PATH = BASE / "aoi_outer_nobuffer.gpkg"
NA_CLIPPER_PATH = BASE / "NA_clipper.gpkg"

FILES = {
    "bldg": STACK / "BLDG_FRAC_10m.tif",
    "imperv": STACK / "IMPERV_FRAC_10m_Helsinki.tif",
    "tree": STACK / "TREE_FRAC_10m.tif",
    "nwn": STACK / "NWN_FRAC_10m.tif",
    "water": STACK / "WATER_FRAC_10m_Helsinki.tif",
    "ocean": STACK / "OCEAN_FRAC_10m_Helsinki.tif",
    "rock": STACK / "ROCK_FRAC_10m_Helsinki.tif",
}

# =============================================================================
# CHANGEABLE SECTION: SCENARIOS
# =============================================================================
# Higher weight = higher resistance.
# Tree / NWN / rock are subtracted because they reduce resistance.

SCENARIOS: Dict[str, Dict[str, float]] = {
    "permissive": {
        "bldg": 8.0,
        "imperv": 6.0,
        "water": 15.0,
        "ocean": 15.0,
        "tree": 1.5,
        "nwn": 1.0,
        "rock": 0.5,
    },
    "conservative": {
        "bldg": 13.0,
        "imperv": 11.0,
        "water": 15.0,
        "ocean": 15.0,
        "tree": 2.5,
        "nwn": 2.0,
        "rock": 1.0,
    },
}

# =============================================================================
# CHANGEABLE SECTION: OPTIONAL THIRD SCENARIO
# =============================================================================

ENABLE_CUSTOM_SCENARIO = False
CUSTOM_SCENARIO_NAME = "custom"

CUSTOM_SCENARIO_WEIGHTS = {
    "bldg": 4.0,
    "imperv": 4.0,
    "water": 2.5,
    "ocean": 6.0,
    "tree": 1.5,
    "nwn": 1.2,
    "rock": 0.4,
}

# =============================================================================
# CHANGEABLE SECTION: CONTEXT / SCALING SETTINGS
# =============================================================================

# --- CHANGE: CONTEXT-AWARE URBAN PENALTY ---
# Neighborhood multiplier for buildings and imperviousness.
# Small isolated road-like pixels keep close to original value.
# Large continuous urban patches get a bounded boost.
CONTEXT_SIZE = 7
URBAN_CONTEXT_BLDG = 0.9
URBAN_CONTEXT_IMPERV = 0.75

WATER_CONTEXT = 0.35   # much weaker than urban
# --- CHANGE: LAND-ONLY GLOBAL SCALING ---
# Land surfaces are scaled to this range.
LAND_SCALE_MIN = 1.0
LAND_SCALE_MAX = 95.0

# --- CHANGE: WATER/OCEAN OVERRIDE AFTER SCALING ---
WATER_VALUE = 97.0
OCEAN_VALUE = 100.0

# =============================================================================
# HELPERS
# =============================================================================

@dataclass
class RasterRef:
    profile: dict
    transform: object
    crs: object
    width: int
    height: int
    nodata: float | int | None


def read_raster(path: Path) -> Tuple[np.ndarray, RasterRef]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        ref = RasterRef(
            profile=profile,
            transform=src.transform,
            crs=src.crs,
            width=src.width,
            height=src.height,
            nodata=src.nodata,
        )
    return arr, ref


def make_geometry_mask(path: Path, ref: RasterRef, invert: bool) -> np.ndarray:
    gdf = gpd.read_file(path)
    if gdf.crs != ref.crs:
        gdf = gdf.to_crs(ref.crs)

    geoms = [geom.__geo_interface__ for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        raise ValueError(f"No valid geometries found in {path}")

    return geometry_mask(
        geoms,
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        invert=invert,
    )


def write_raster(path: Path, arr: np.ndarray, profile: dict) -> None:
    p = profile.copy()
    p.update(
        dtype="float32",
        count=1,
        compress="deflate",
        tiled=False,
        nodata=-9999.0,
    )
    out = np.where(np.isnan(arr), -9999.0, arr).astype(np.float32)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out, 1)


def load_layer(path: Path, ref: RasterRef, keep_mask: np.ndarray, remove_mask: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as src:
        if (src.width, src.height) != (ref.width, ref.height) or src.transform != ref.transform or src.crs != ref.crs:
            raise ValueError(f"Grid mismatch for {path.name}")

        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

    arr = np.where(keep_mask, arr, np.nan)
    arr = np.where(remove_mask, np.nan, arr)
    return arr


# =============================================================================
# CHANGE: CONTEXT-AWARE URBAN PENALTY
# =============================================================================
# This is bounded and multiplicative, so it does not flatten the surface.
# It only boosts urban layers where there is neighborhood support.

def apply_context_multiplier(arr: np.ndarray, size: int, beta: float) -> np.ndarray:
    arr_filled = np.nan_to_num(arr, nan=0.0)
    local_mean = uniform_filter(arr_filled, size=size, mode="nearest")
    local_mean = np.clip(local_mean, 0.0, 1.0)
    return arr_filled * (1.0 + beta * local_mean)

def build_resistance(layers: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    B = np.nan_to_num(layers["bldg"], nan=0.0)
    I = np.nan_to_num(layers["imperv"], nan=0.0)
    T = np.nan_to_num(layers["tree"], nan=0.0)
    N = np.nan_to_num(layers["nwn"], nan=0.0)
    W = np.nan_to_num(layers["water"], nan=0.0)
    O = np.nan_to_num(layers["ocean"], nan=0.0)
    R = np.nan_to_num(layers["rock"], nan=0.0)

    # --- context only where it makes sense ---
    B_adj = apply_context_multiplier(B, size=CONTEXT_SIZE, beta=URBAN_CONTEXT_BLDG)
    I_adj = apply_context_multiplier(I, size=CONTEXT_SIZE, beta=URBAN_CONTEXT_IMPERV)

    # --- OPTIONAL: very mild context for water ---
    # This increases resistance for large lakes slightly,
    # but keeps small streams/perforations relatively permeable
    W_adj = apply_context_multiplier(W, size=CONTEXT_SIZE, beta=WATER_CONTEXT)

    # --- full resistance model (NO overrides here) ---
    raw = (
        weights["bldg"] * B_adj +
        weights["imperv"] * I_adj +
        weights["water"] * W_adj -     # <-- water now participates normally
        weights["tree"] * T -
        weights["nwn"] * N -
        weights["rock"] * R
    )

    # --- EXCLUDE ONLY OCEAN from scaling ---
    raw = np.where(O > 0, np.nan, raw)

    raw = np.where(np.isfinite(raw), raw, np.nan)
    return raw.astype(np.float32)


# =============================================================================
# CHANGE: LAND-ONLY GLOBAL SCALING
# =============================================================================

def scale_global_land(arr: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
    valid = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=np.float32)

    if np.isclose(global_min, global_max):
        out[valid] = LAND_SCALE_MIN
        return out

    out[valid] = LAND_SCALE_MIN + (LAND_SCALE_MAX - LAND_SCALE_MIN) * (
        (arr[valid] - global_min) / (global_max - global_min)
    )
    return out


def qc_plot(results: Dict[str, np.ndarray], out_path: Path) -> None:
    names = list(results.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), constrained_layout=True)

    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        arr = results[name]
        im = ax.imshow(arr, cmap="viridis", vmin=LAND_SCALE_MIN, vmax=OCEAN_VALUE)
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Resistance")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    first_key = next(iter(FILES))
    _, ref = read_raster(FILES[first_key])

    aoi_keep_mask = make_geometry_mask(AOI_PATH, ref, invert=True)
    na_remove_mask = make_geometry_mask(NA_CLIPPER_PATH, ref, invert=True)

    layers: Dict[str, np.ndarray] = {}
    for key, path in FILES.items():
        layers[key] = load_layer(path, ref, keep_mask=aoi_keep_mask, remove_mask=na_remove_mask)

    water_mask = np.nan_to_num(layers["water"], nan=0.0) > 0
    ocean_mask = np.nan_to_num(layers["ocean"], nan=0.0) > 0

    raw_results: Dict[str, np.ndarray] = {}
    for name, weights in SCENARIOS.items():
        raw_results[name] = build_resistance(layers, weights)

    if ENABLE_CUSTOM_SCENARIO:
        raw_results[CUSTOM_SCENARIO_NAME] = build_resistance(layers, CUSTOM_SCENARIO_WEIGHTS)

    pooled_vals = np.concatenate([
        arr[np.isfinite(arr)]
        for arr in raw_results.values()
        if np.isfinite(arr).any()
    ])

    if pooled_vals.size == 0:
        raise ValueError("No valid values available across raw land resistance surfaces.")

    global_min = float(np.min(pooled_vals))
    global_max = float(np.max(pooled_vals))

    print(f"[INFO] Global land scaling range: min={global_min:.3f}, max={global_max:.3f}")

    results: Dict[str, np.ndarray] = {}
    for name, raw in raw_results.items():
        res = scale_global_land(raw, global_min, global_max)

        # --- CHANGE: WATER/OCEAN OVERRIDE AFTER SCALING ---
        #res = np.where(water_mask, WATER_VALUE, res)
        res = np.where(ocean_mask, OCEAN_VALUE, res)

        res = np.where(aoi_keep_mask, res, np.nan)
        results[name] = res

        out_path = OUT / f"resistance_{name}.tif"
        write_raster(out_path, res, ref.profile)
        print(f"[OK] wrote {out_path}")

    qc_path = OUT / "qc_resistance_maps.png"
    qc_plot(results, qc_path)
    print(f"[OK] wrote {qc_path}")


if __name__ == "__main__":
    main()