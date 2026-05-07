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
BAREGROUND_VECTOR_PATH = BASE / "LULC" / "lc_bareground.gpkg"

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
    "nwn": 3.0,
    "residual_bare": 6.0,
    "bareground": 8.0,
    "rock": 12.0,
    "fields": 20.0,
    "imperv": 32.0,
    "water": 42.0,
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


def rasterize_polygons(path: Path, ref: RasterRef, clip_mask: np.ndarray) -> np.ndarray:
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
# RESISTANCE BUILDING
# =============================================================================

def build_resistance(layers: Dict[str, np.ndarray], fields: np.ndarray, bareground: np.ndarray) -> np.ndarray:
    # Clamp all fractions to [0, 1]
    bldg = np.clip(np.nan_to_num(layers["bldg"], nan=0.0), 0.0, 1.0)
    imperv = np.clip(np.nan_to_num(layers["imperv"], nan=0.0), 0.0, 1.0)
    tree = np.clip(np.nan_to_num(layers["tree"], nan=0.0), 0.0, 1.0)
    nwn = np.clip(np.nan_to_num(layers["nwn"], nan=0.0), 0.0, 1.0)
    water = np.clip(np.nan_to_num(layers["water"], nan=0.0), 0.0, 1.0)
    ocean = np.clip(np.nan_to_num(layers["ocean"], nan=0.0), 0.0, 1.0)
    rock = np.clip(np.nan_to_num(layers["rock"], nan=0.0), 0.0, 1.0)
    fields = np.clip(np.nan_to_num(fields, nan=0.0), 0.0, 1.0)
    bareground = np.clip(np.nan_to_num(bareground, nan=0.0), 0.0, 1.0)

    # Remove explicit land-cover classes from NWN so they do not get double-counted.
    nwn_eff = np.clip(nwn - fields - bareground, 0.0, 1.0)

    # Residual area not captured by the explicit fractions.
    residual_bare = np.clip(
        1.0 - (tree + nwn_eff + fields + bareground + rock + imperv + bldg + water + ocean),
        0.0,
        1.0,
    )

    w_forest = tree
    w_nwn = nwn_eff
    w_residual_bare = residual_bare
    w_bareground = bareground
    w_rock = rock
    w_fields = fields
    w_imperv = imperv
    w_water = water
    w_bldg = bldg
    w_ocean = ocean

    numerator = (
        w_forest * RESISTANCE_VALUES["forest"] +
        w_nwn * RESISTANCE_VALUES["nwn"] +
        w_residual_bare * RESISTANCE_VALUES["residual_bare"] +
        w_bareground * RESISTANCE_VALUES["bareground"] +
        w_rock * RESISTANCE_VALUES["rock"] +
        w_fields * RESISTANCE_VALUES["fields"] +
        w_imperv * RESISTANCE_VALUES["imperv"] +
        w_water * RESISTANCE_VALUES["water"] +
        w_bldg * RESISTANCE_VALUES["bldg"] +
        w_ocean * RESISTANCE_VALUES["ocean"]
    )

    denominator = (
        w_forest + w_nwn + w_residual_bare + w_bareground + w_rock +
        w_fields + w_imperv + w_water + w_bldg + w_ocean
    )

    resistance = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )

    resistance = np.clip(resistance, 1.0, 50.0)
    return resistance.astype(np.float32)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    first_key = next(iter(FILES))
    _, ref = read_raster(FILES[first_key])

    # Keep only AOI, remove NA clip polygon.
    aoi_keep_mask = make_geometry_mask(AOI_PATH, ref, invert=True)
    na_remove_mask = make_geometry_mask(NA_CLIPPER_PATH, ref, invert=True)
    clip_mask = aoi_keep_mask & (~na_remove_mask)

    layers: Dict[str, np.ndarray] = {}
    for key, path in FILES.items():
        layers[key] = load_layer(
            path,
            ref,
            keep_mask=clip_mask,
            remove_mask=np.zeros((ref.height, ref.width), dtype=bool),
        )

    fields = rasterize_polygons(FIELDS_VECTOR_PATH, ref, clip_mask=clip_mask)
    bareground = rasterize_polygons(BAREGROUND_VECTOR_PATH, ref, clip_mask=clip_mask)

    resistance = build_resistance(layers, fields, bareground)
    resistance = np.where(clip_mask, resistance, np.nan)

    out_tif = OUT / "resistance_parsimonious_v2.tif"
    write_raster(out_tif, resistance, ref.profile)
    print(f"[OK] wrote {out_tif}")

    qc_png = OUT / "qc_resistance_parsimonious_v2.png"
    qc_plot(resistance, qc_png)
    print(f"[OK] wrote {qc_png}")


if __name__ == "__main__":
    main()



# exclude buildings:
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
BAREGROUND_VECTOR_PATH = BASE / "LULC" / "lc_bareground.gpkg"

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
#
# Buildings are NOT assigned a resistance value in this version.
# They are converted to nodata/excluded cells.

RESISTANCE_VALUES = {
    "forest": 1.0,
    "nwn": 3.0,
    "residual_bare": 6.0,
    "bareground": 8.0,
    "rock": 12.0,
    "fields": 20.0,
    "imperv": 32.0,
    "water": 42.0,
    "ocean": 50.0,
}

# Buildings greater than this fraction are excluded entirely.
BUILDING_NA_THRESHOLD = 0.0
OCEAN_NA_THRESHOLD = 0.0

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


def rasterize_polygons(path: Path, ref: RasterRef, clip_mask: np.ndarray) -> np.ndarray:
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
# RESISTANCE BUILDING
# =============================================================================

def build_resistance(layers: Dict[str, np.ndarray], fields: np.ndarray, bareground: np.ndarray) -> np.ndarray:
    bldg = np.clip(np.nan_to_num(layers["bldg"], nan=0.0), 0.0, 1.0)
    imperv = np.clip(np.nan_to_num(layers["imperv"], nan=0.0), 0.0, 1.0)
    tree = np.clip(np.nan_to_num(layers["tree"], nan=0.0), 0.0, 1.0)
    nwn = np.clip(np.nan_to_num(layers["nwn"], nan=0.0), 0.0, 1.0)
    water = np.clip(np.nan_to_num(layers["water"], nan=0.0), 0.0, 1.0)
    ocean = np.clip(np.nan_to_num(layers["ocean"], nan=0.0), 0.0, 1.0)
    rock = np.clip(np.nan_to_num(layers["rock"], nan=0.0), 0.0, 1.0)
    fields = np.clip(np.nan_to_num(fields, nan=0.0), 0.0, 1.0)
    bareground = np.clip(np.nan_to_num(bareground, nan=0.0), 0.0, 1.0)

    # Buildings are excluded entirely from the circuit network.
    building_mask = bldg > BUILDING_NA_THRESHOLD
    ocean_mask = ocean > OCEAN_NA_THRESHOLD

    # Remove explicit classes from NWN so they are not double-counted.
    nwn_eff = np.clip(nwn - fields - bareground, 0.0, 1.0)

    residual_bare = np.clip(
        1.0 - (tree + nwn_eff + fields + bareground + rock + imperv + water + ocean),
        0.0,
        1.0,
    )

    w_forest = tree
    w_nwn = nwn_eff
    w_residual_bare = residual_bare
    w_bareground = bareground
    w_rock = rock
    w_fields = fields
    w_imperv = imperv
    w_water = water
    #w_ocean = ocean

    numerator = (
        w_forest * RESISTANCE_VALUES["forest"] +
        w_nwn * RESISTANCE_VALUES["nwn"] +
        w_residual_bare * RESISTANCE_VALUES["residual_bare"] +
        w_bareground * RESISTANCE_VALUES["bareground"] +
        w_rock * RESISTANCE_VALUES["rock"] +
        w_fields * RESISTANCE_VALUES["fields"] +
        w_imperv * RESISTANCE_VALUES["imperv"] +
        w_water * RESISTANCE_VALUES["water"] #+
        #w_ocean * RESISTANCE_VALUES["ocean"]
    )

    denominator = (
        w_forest + w_nwn + w_residual_bare + w_bareground + w_rock +
        w_fields + w_imperv + w_water #+ w_ocean
    )

    resistance = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )

    resistance = np.clip(resistance, 1.0, 50.0)

    # Exclude building cells entirely from the final surface.
    resistance = np.where(building_mask, np.nan, resistance)
    resistance = np.where(ocean_mask, np.nan, resistance) # remove oceans

    return resistance.astype(np.float32)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    first_key = next(iter(FILES))
    _, ref = read_raster(FILES[first_key])

    aoi_keep_mask = make_geometry_mask(AOI_PATH, ref, invert=True)
    na_remove_mask = make_geometry_mask(NA_CLIPPER_PATH, ref, invert=True)
    clip_mask = aoi_keep_mask & (~na_remove_mask)

    layers: Dict[str, np.ndarray] = {}
    for key, path in FILES.items():
        layers[key] = load_layer(
            path,
            ref,
            keep_mask=clip_mask,
            remove_mask=np.zeros((ref.height, ref.width), dtype=bool),
        )

    fields = rasterize_polygons(FIELDS_VECTOR_PATH, ref, clip_mask=clip_mask)
    bareground = rasterize_polygons(BAREGROUND_VECTOR_PATH, ref, clip_mask=clip_mask)

    resistance = build_resistance(layers, fields, bareground)
    resistance = np.where(clip_mask, resistance, np.nan)

    out_tif = OUT / "resistance_parsimonious_v4.tif"
    write_raster(out_tif, resistance, ref.profile)
    print(f"[OK] wrote {out_tif}")

    qc_png = OUT / "qc_resistance_parsimonious_v4.png"
    qc_plot(resistance, qc_png)
    print(f"[OK] wrote {qc_png}")


if __name__ == "__main__":
    main()





# exclude ONLY BUILDINGS no ocean
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
BAREGROUND_VECTOR_PATH = BASE / "LULC" / "lc_bareground.gpkg"

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
#
# Buildings are NOT assigned a resistance value in this version.
# They are converted to nodata/excluded cells.

RESISTANCE_VALUES = {
    "forest": 1.0,
    "nwn": 3.0,
    "residual_bare": 6.0,
    "bareground": 8.0,
    "rock": 12.0,
    "fields": 20.0,
    "imperv": 32.0,
    "water": 42.0,
    "ocean": 50.0,
}

# Buildings greater than this fraction are excluded entirely.
BUILDING_NA_THRESHOLD = 0.0
#OCEAN_NA_THRESHOLD = 0.0

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


def rasterize_polygons(path: Path, ref: RasterRef, clip_mask: np.ndarray) -> np.ndarray:
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
# RESISTANCE BUILDING
# =============================================================================

def build_resistance(layers: Dict[str, np.ndarray], fields: np.ndarray, bareground: np.ndarray) -> np.ndarray:
    bldg = np.clip(np.nan_to_num(layers["bldg"], nan=0.0), 0.0, 1.0)
    imperv = np.clip(np.nan_to_num(layers["imperv"], nan=0.0), 0.0, 1.0)
    tree = np.clip(np.nan_to_num(layers["tree"], nan=0.0), 0.0, 1.0)
    nwn = np.clip(np.nan_to_num(layers["nwn"], nan=0.0), 0.0, 1.0)
    water = np.clip(np.nan_to_num(layers["water"], nan=0.0), 0.0, 1.0)
    ocean = np.clip(np.nan_to_num(layers["ocean"], nan=0.0), 0.0, 1.0)
    rock = np.clip(np.nan_to_num(layers["rock"], nan=0.0), 0.0, 1.0)
    fields = np.clip(np.nan_to_num(fields, nan=0.0), 0.0, 1.0)
    bareground = np.clip(np.nan_to_num(bareground, nan=0.0), 0.0, 1.0)

    # Buildings are excluded entirely from the circuit network.
    building_mask = bldg > BUILDING_NA_THRESHOLD
    #ocean_mask = ocean > OCEAN_NA_THRESHOLD

    # Remove explicit classes from NWN so they are not double-counted.
    nwn_eff = np.clip(nwn - fields - bareground, 0.0, 1.0)

    residual_bare = np.clip(
        1.0 - (tree + nwn_eff + fields + bareground + rock + imperv + water + ocean),
        0.0,
        1.0,
    )

    w_forest = tree
    w_nwn = nwn_eff
    w_residual_bare = residual_bare
    w_bareground = bareground
    w_rock = rock
    w_fields = fields
    w_imperv = imperv
    w_water = water
    w_ocean = ocean

    numerator = (
        w_forest * RESISTANCE_VALUES["forest"] +
        w_nwn * RESISTANCE_VALUES["nwn"] +
        w_residual_bare * RESISTANCE_VALUES["residual_bare"] +
        w_bareground * RESISTANCE_VALUES["bareground"] +
        w_rock * RESISTANCE_VALUES["rock"] +
        w_fields * RESISTANCE_VALUES["fields"] +
        w_imperv * RESISTANCE_VALUES["imperv"] +
        w_water * RESISTANCE_VALUES["water"] +
        w_ocean * RESISTANCE_VALUES["ocean"]
    )

    denominator = (
        w_forest + w_nwn + w_residual_bare + w_bareground + w_rock +
        w_fields + w_imperv + w_water + w_ocean
    )

    resistance = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )

    resistance = np.clip(resistance, 1.0, 50.0)

    # Exclude building cells entirely from the final surface.
    resistance = np.where(building_mask, np.nan, resistance)
    #resistance = np.where(ocean_mask, np.nan, resistance) # remove oceans

    return resistance.astype(np.float32)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    first_key = next(iter(FILES))
    _, ref = read_raster(FILES[first_key])

    aoi_keep_mask = make_geometry_mask(AOI_PATH, ref, invert=True)
    na_remove_mask = make_geometry_mask(NA_CLIPPER_PATH, ref, invert=True)
    clip_mask = aoi_keep_mask & (~na_remove_mask)

    layers: Dict[str, np.ndarray] = {}
    for key, path in FILES.items():
        layers[key] = load_layer(
            path,
            ref,
            keep_mask=clip_mask,
            remove_mask=np.zeros((ref.height, ref.width), dtype=bool),
        )

    fields = rasterize_polygons(FIELDS_VECTOR_PATH, ref, clip_mask=clip_mask)
    bareground = rasterize_polygons(BAREGROUND_VECTOR_PATH, ref, clip_mask=clip_mask)

    resistance = build_resistance(layers, fields, bareground)
    resistance = np.where(clip_mask, resistance, np.nan)

    out_tif = OUT / "resistance_parsimonious_v3_nobuild.tif"
    write_raster(out_tif, resistance, ref.profile)
    print(f"[OK] wrote {out_tif}")

    qc_png = OUT / "qc_resistance_parsimonious_v3_nobuild.png"
    qc_plot(resistance, qc_png)
    print(f"[OK] wrote {qc_png}")


if __name__ == "__main__":
    main()