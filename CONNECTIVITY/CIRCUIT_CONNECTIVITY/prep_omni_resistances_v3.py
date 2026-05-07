#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import geometry_mask, rasterize

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
STACK = BASE / "predictorstack"
OUT = BASE / "omniscape" / "resistance"
OUT.mkdir(parents=True, exist_ok=True)

AOI_PATH = BASE / "aoi_outer_nobuffer.gpkg"
NA_CLIPPER_PATH = BASE / "NA_clipper.gpkg"
FIELDS_VECTOR_PATH = BASE / "LULC" / "lc_fields.gpkg"

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
# CLASS RESISTANCE VALUES
# =============================================================================
# 1 = easiest movement
# 50 = highest resistance

RESISTANCE_VALUES = {
    "forest": 1.0,
    "nwn": 2.0,
    "fields": 4.0,
    "bare": 4.0,
    "rock": 8.0,
    "imperv": 30.0,
    "water": 40.0,
    "bldg": 50.0,
    "ocean": 50.0,
}

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


def rasterize_fields(path: Path, ref: RasterRef, clip_mask: np.ndarray) -> np.ndarray:
    gdf = gpd.read_file(path)
    if gdf.crs != ref.crs:
        gdf = gdf.to_crs(ref.crs)

    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    shapes = [(geom, 1.0) for geom in geoms]

    arr = rasterize(
        shapes=shapes,
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        fill=0.0,
        all_touched=False,
        dtype=np.float32,
    )

    arr = np.where(clip_mask, arr, np.nan)
    return arr


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


def qc_plot(arr: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(arr, cmap="viridis", vmin=1, vmax=50)
    plt.title("Resistance surface")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04, label="Resistance")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN RESISTANCE BUILDING
# =============================================================================

def build_resistance(layers: Dict[str, np.ndarray], fields: np.ndarray) -> np.ndarray:
    # Clamp all fractions to [0, 1]
    bldg = np.clip(np.nan_to_num(layers["bldg"], nan=0.0), 0.0, 1.0)
    imperv = np.clip(np.nan_to_num(layers["imperv"], nan=0.0), 0.0, 1.0)
    tree = np.clip(np.nan_to_num(layers["tree"], nan=0.0), 0.0, 1.0)
    nwn = np.clip(np.nan_to_num(layers["nwn"], nan=0.0), 0.0, 1.0)
    water = np.clip(np.nan_to_num(layers["water"], nan=0.0), 0.0, 1.0)
    ocean = np.clip(np.nan_to_num(layers["ocean"], nan=0.0), 0.0, 1.0)
    rock = np.clip(np.nan_to_num(layers["rock"], nan=0.0), 0.0, 1.0)
    fields = np.clip(np.nan_to_num(fields, nan=0.0), 0.0, 1.0)

    # --- CHANGE 1: subtract fields from NWN ---
    nwn_eff = np.clip(nwn - fields, 0.0, 1.0)

    # --- CHANGE 2: mixed pixels handled as a convex mixture of class resistances ---
    # Bare/residual area gets a moderate value.
    bare = np.clip(
        1.0 - (tree + nwn_eff + fields + rock + imperv + bldg + water + ocean),
        0.0,
        1.0,
    )

    w_forest = tree
    w_nwn = nwn_eff
    w_fields = fields
    w_rock = rock
    w_imperv = imperv
    w_bldg = bldg
    w_water = water
    w_ocean = ocean
    w_bare = bare

    numerator = (
        w_forest * RESISTANCE_VALUES["forest"] +
        w_nwn * RESISTANCE_VALUES["nwn"] +
        w_fields * RESISTANCE_VALUES["fields"] +
        w_rock * RESISTANCE_VALUES["rock"] +
        w_imperv * RESISTANCE_VALUES["imperv"] +
        w_bldg * RESISTANCE_VALUES["bldg"] +
        w_water * RESISTANCE_VALUES["water"] +
        w_ocean * RESISTANCE_VALUES["ocean"] +
        w_bare * RESISTANCE_VALUES["bare"]
    )

    denominator = w_forest + w_nwn + w_fields + w_rock + w_imperv + w_bldg + w_water + w_ocean + w_bare
    resistance = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)

    # Keep inside the intended range.
    resistance = np.clip(resistance, 1.0, 50.0)
    return resistance.astype(np.float32)


def main() -> None:
    first_key = next(iter(FILES))
    _, ref = read_raster(FILES[first_key])

    # Keep only AOI, remove NA clip polygon.
    aoi_keep_mask = make_geometry_mask(AOI_PATH, ref, invert=True)
    na_remove_mask = make_geometry_mask(NA_CLIPPER_PATH, ref, invert=True)
    clip_mask = aoi_keep_mask & (~na_remove_mask)

    layers: Dict[str, np.ndarray] = {}
    for key, path in FILES.items():
        layers[key] = load_layer(path, ref, keep_mask=clip_mask, remove_mask=np.zeros((ref.height, ref.width), dtype=bool))

    fields = rasterize_fields(FIELDS_VECTOR_PATH, ref, clip_mask=clip_mask)

    resistance = build_resistance(layers, fields)
    resistance = np.where(clip_mask, resistance, np.nan)

    out_tif = OUT / "resistance_parsimonious.tif"
    write_raster(out_tif, resistance, ref.profile)
    print(f"[OK] wrote {out_tif}")

    qc_png = OUT / "qc_resistance_parsimonious.png"
    qc_plot(resistance, qc_png)
    print(f"[OK] wrote {qc_png}")


if __name__ == "__main__":
    main()