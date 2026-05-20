"""
Figure f3_1: official Helsinki subdivisions, heatwave offset from July mean baseline.

What it does
------------
1. Uses suurpiirit as the outer grouping sectors.
2. Uses kaupunginosat as the individual radial bars.
3. Finds the peak UTC hour for each heatwave event from available GeoTIFFs.
4. Subtracts the same-hour July mean baseline raster from each heatwave raster.
5. Aggregates the event offsets, computes kaupunginosa zonal means, and exports SVG.

Default aggregation is the mean heatwave offset from the July mean baseline, computed only inside the full model target domain: tree + non-natural woody vegetation, excluding water/ocean, buildings, and highly impervious pixels.

The center map shows raw mean heatwave offset from the July mean baseline. The outer ring uses a stronger neighborhood metric by default: the 90th percentile of target-domain offsets within each kaupunginosa, which increases between-neighborhood separation while remaining comparable across areas.
"""

from __future__ import annotations

import importlib.util
import re
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import rasterio
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import reproject
from shapely.geometry import box
from shapely.validation import make_valid
from scipy.ndimage import gaussian_filter

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
WORKDIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures\f3_1")
OUTPUT_SVG = WORKDIR / "f3_1_suurpiirit_kaupunginosat_heatwave_offset.svg"
OUTPUT_CSV = WORKDIR / "f3_1_suurpiirit_kaupunginosat_heatwave_offset_values.csv"

GLOBAL_SETTINGS = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures\global_plotting_settings.py")
SUURPIIRIT_GEOJSON = WORKDIR / "suurpiirit_WFS_20260520_135903.geojson"
KAUPUNGINOSAT_GEOJSON = WORKDIR / "kaupunginosat_20260520_135904.geojson"

HEATWAVE_INPUTS = {
    "2010": Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\2010\20100728"),
    "2018": Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\2018\20180717\pred_20180717_0700.tif"),
    "2021": Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\2021\20210714\pred_20210714_0700.tif"),
}

BASELINE_MEAN_INPUT = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\baseline\15cm_July_allday\pred_20000715_0900.tif")
BASELINE_P90_INPUT = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\predictions\baseline\15cm_July_allday_p90\pred_20000715_0900.tif")

# -----------------------------------------------------------------------------
# Figure choices
# -----------------------------------------------------------------------------
LOCAL_UTC_OFFSET_HOURS = 3  # Helsinki July = EEST = UTC+3
EXPECTED_PEAK_LOCAL_HOUR = 13
AGGREGATION = "mean"  # one of: mean, mean_positive, max
BAR_LABEL_MAX_N = 80          # above this, labels are suppressed to avoid unreadable SVGs
MAP_SMOOTH_SIGMA = 1.35       # display-only smoothing for the center map
BAR_GROUP_GAP_SLOTS = 0.8      # small constant gap between suurpiirit while keeping equal bar widths
RING_METRIC = "q90"            # one of: mean, q90, q95, max
RING_DISPLAY_FLOOR_PAD = 0.18  # display floor below min; bars still start at inner ring
# Fixed display constants. These are intentionally absolute so changes remain
# visible even when project rcParams set large default font sizes.
POLAR_BOX = [0.06, 0.06, 0.88, 0.88]  # make the whole figure footprint larger on the fixed canvas
RING_RADIUS_SCALE = 3.20      # strongly scale bar heights so the outer ring becomes much more legible
HOLE_R = 18.0                 # preserve a large central hole even with the stronger ring scaling
GROUP_LABEL_RADIUS = 0.0      # curved suurpiiri labels live on the unified inner axis
GROUP_LABEL_OFFSET_PTS = -11.0   # move suurpiiri labels inside, away from the axis
GROUP_LABEL_FONT_SIZE = 10.4
NEIGHBORHOOD_LABEL_RADIUS = 0.0   # anchor neighborhood labels on the axis itself
NEIGHBORHOOD_LABEL_OFFSET_PTS = 50  # move labels outward so they begin near the top of the bar area
NEIGHBORHOOD_FONT_SIZE = 9
VALUE_LABEL_OFFSET_PTS = 5.5
VALUE_LABEL_FONT_SIZE = 6.6
TEXT_OUTLINE_WIDTH = 2.4
VALUE_LABEL_MIN_SEP_PX = 13.0

# Target-domain definition copied from the f3 map logic.
DATA_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
TREE_PATH = DATA_DIR / "predictorstack" / "TREE_FRAC_10m.tif"
NWN_PATH = DATA_DIR / "predictorstack" / "NWN_FRAC_10m.tif"
WATER_PATH = DATA_DIR / "predictorstack" / "WATER_FRAC_10m_Helsinki.tif"
OCEAN_PATH = DATA_DIR / "predictorstack" / "OCEAN_FRAC_10m_Helsinki.tif"
IMPERVIOUS_PATH = DATA_DIR / "predictorstack" / "IMPERV_FRAC_10m_Helsinki.tif"
BUILDING_PATH = DATA_DIR / "predictorstack" / "BLDG_FRAC_10m.tif"
BAREGROUND_VECTOR_PATH = DATA_DIR / "LULC" / "lc_bareground.gpkg"
VEGETATION_MIN_FRACTION = 0.05
WATER_MIN_FRACTION = 0.05
BUILDING_EXCLUDE_FRACTION = 0.0
IMPERVIOUS_EXCLUDE_FRACTION = 0.5
MIN_BAR_LABEL_VALUE = 0.05
CMAP = "inferno"

# Column guesses. The script prints available columns if none match.
SUURPIIRI_NAME_COLUMNS = [
    "hel:nimi_fi", "nimi_fi", "NIMI_FI", "nimi", "NIMI", "name", "NAME",
    "suurpiiri", "SUURPIIRI", "SUURPIIRIN_NIMI", "Suurpiiri",
]
KAUPUNGINOSA_NAME_COLUMNS = [
    "hel:nimi_fi", "nimi_fi", "NIMI_FI", "nimi", "NIMI", "name", "NAME",
    "kaupunginosa", "KAUPUNGINOSA", "KAUPUNGINOSAN_NIMI", "Kaupunginosa",
]


@dataclass(frozen=True)
class RasterSurface:
    label: str
    path: Path
    hour_utc: int
    array: np.ndarray
    profile: dict


def apply_global_plotting_settings(settings_path: Path) -> None:
    """Best-effort import of the project-wide plotting settings."""
    if not settings_path.exists():
        print(f"WARNING: global plotting settings not found: {settings_path}")
        return

    ns = runpy.run_path(str(settings_path))

    for dict_name in ["RC_PARAMS", "rcParams", "MPL_RC_PARAMS", "GLOBAL_RC_PARAMS"]:
        value = ns.get(dict_name)
        if isinstance(value, dict):
            mpl.rcParams.update(value)

    for func_name in [
        "apply_global_plotting_settings",
        "apply_plot_style",
        "set_plot_style",
        "setup_matplotlib",
        "configure_matplotlib",
    ]:
        func = ns.get(func_name)
        if callable(func):
            try:
                func()
                break
            except TypeError:
                # Some project functions require axes/figure arguments. Ignore those here.
                pass

    # Ensure SVG text remains editable in vector software unless project settings override later.
    mpl.rcParams.setdefault("svg.fonttype", "none")



def clean_geometries(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """Repair invalid geometries and drop empty/null features."""
    out = gdf.copy()
    out = out[out.geometry.notna()].copy()

    def _fix(geom):
        if geom is None or geom.is_empty:
            return None
        try:
            if not geom.is_valid:
                geom = make_valid(geom)
        except Exception:
            try:
                geom = geom.buffer(0)
            except Exception:
                return None
        if geom is None or geom.is_empty:
            return None
        return geom

    out["geometry"] = out.geometry.apply(_fix)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    if out.empty:
        raise ValueError(f"All {label} geometries became empty after repair.")
    return out


def raster_bounds_polygon(profile: dict):
    left, right, bottom, top = raster_extent(profile)
    return box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))


def vector_raster_overlap_score(gdf: gpd.GeoDataFrame, profile: dict) -> float:
    """Approximate overlap area between vector layer and raster bounds."""
    rb = raster_bounds_polygon(profile)
    try:
        # Bounds-level check is enough for choosing the right CRS interpretation.
        gb = box(*gdf.total_bounds)
        if not gb.is_valid or gb.is_empty:
            return 0.0
        inter = gb.intersection(rb)
        return float(inter.area) if not inter.is_empty else 0.0
    except Exception:
        return 0.0


def align_vectors_to_raster_crs(gdf: gpd.GeoDataFrame, target_profile: dict, label: str) -> gpd.GeoDataFrame:
    """Convert vectors to raster CRS, with safeguards for misdeclared/no CRS GeoJSONs."""
    target_crs = target_profile["crs"]
    original = clean_geometries(gdf, label)

    candidates: list[tuple[str, gpd.GeoDataFrame]] = []

    if original.crs is not None:
        try:
            candidates.append((f"declared CRS {original.crs}", clean_geometries(original.to_crs(target_crs), label)))
        except Exception as exc:
            print(f"WARNING: could not transform {label} from declared CRS {original.crs}: {exc}")

    # Helsinki city WFS and common GeoJSON/raster CRS fallbacks. These are only used
    # to recover from missing or misleading CRS metadata.
    for epsg in [4326, 3067, 3879, 3857]:
        try:
            cand = original.set_crs(epsg=epsg, allow_override=True).to_crs(target_crs)
            cand = clean_geometries(cand, label)
            candidates.append((f"forced EPSG:{epsg}", cand))
        except Exception:
            pass

    if not candidates:
        raise ValueError(f"Could not align {label} to raster CRS {target_crs}.")

    scored = [(vector_raster_overlap_score(cand, target_profile), desc, cand) for desc, cand in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_desc, best = scored[0]

    print(f"{label}: selected {best_desc}; bounds overlap score={best_score:.3f}")
    if best_score <= 0:
        print(
            f"WARNING: {label} bounds do not overlap the raster bounds after CRS alignment. "
            "Zonal means may be NaN. Check vector/raster CRS metadata."
        )
    return best

def first_existing_column(gdf: gpd.GeoDataFrame, candidates: Iterable[str], label: str) -> str:
    """Return the first matching name column, including namespaced WFS fields.

    Helsinki WFS exports commonly prefix fields as e.g. ``hel:nimi_fi``.
    This helper first checks exact candidates and then falls back to the
    suffix after the namespace separator, so ``hel:nimi_fi`` matches
    ``nimi_fi``.
    """
    cols = list(gdf.columns)

    for col in candidates:
        if col in cols:
            return col

    candidate_set = {c.lower() for c in candidates}
    for col in cols:
        suffix = col.split(":")[-1].lower()
        if suffix in candidate_set:
            return col

    # Last-resort heuristic for Helsinki area-name columns. Prefer Finnish names.
    for preferred_suffix in ["nimi_fi", "namn_fi", "name_fi", "nimi"]:
        for col in cols:
            if col.split(":")[-1].lower() == preferred_suffix:
                return col

    raise ValueError(
        f"Could not identify a {label} name column. Available columns: {list(gdf.columns)}"
    )


def file_hour_utc(path: Path) -> int:
    """Extract hour from filenames like pred_20180717_0700.tif."""
    m = re.search(r"_(\d{4})\.tif$", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not extract UTC hour from filename: {path.name}")
    return int(m.group(1)[:2])


def discover_tifs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
    raise FileNotFoundError(f"Input path does not exist: {path}")


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        profile = src.profile.copy()
        profile.update(
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            nodata=np.nan,
        )
    return arr, profile


def reproject_to_match(arr: np.ndarray, src_profile: dict, dst_profile: dict) -> np.ndarray:
    same_grid = (
        src_profile["crs"] == dst_profile["crs"]
        and src_profile["transform"] == dst_profile["transform"]
        and src_profile["width"] == dst_profile["width"]
        and src_profile["height"] == dst_profile["height"]
    )
    if same_grid:
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
        resampling=Resampling.bilinear,
    )
    return dst


def spatial_mean(path: Path) -> float:
    arr, _ = read_raster(path)
    return float(np.nanmean(arr))


def select_peak_surface(label: str, input_path: Path) -> RasterSurface:
    tif_paths = discover_tifs(input_path)
    if not tif_paths:
        raise FileNotFoundError(f"No GeoTIFF files found for {label}: {input_path}")

    if len(tif_paths) == 1:
        chosen = tif_paths[0]
    else:
        scored = [(spatial_mean(p), p) for p in tif_paths]
        chosen = max(scored, key=lambda x: x[0])[1]

    hour = file_hour_utc(chosen)
    local_hour = (hour + LOCAL_UTC_OFFSET_HOURS) % 24
    if local_hour != EXPECTED_PEAK_LOCAL_HOUR:
        print(
            f"NOTE: selected peak for {label} is {hour:02d}:00 UTC "
            f"({local_hour:02d}:00 local), not {EXPECTED_PEAK_LOCAL_HOUR:02d}:00 local."
        )

    arr, profile = read_raster(chosen)
    print(f"Selected {label}: {chosen.name} at {hour:02d}:00 UTC")
    return RasterSurface(label=label, path=chosen, hour_utc=hour, array=arr, profile=profile)


def matching_baseline_for_hour(baseline_input: Path, hour_utc: int) -> Path:
    """Find same-hour baseline. If only one file is available, use it and print a warning."""
    candidates = discover_tifs(baseline_input)
    same_hour = [p for p in candidates if file_hour_utc(p) == hour_utc]
    if same_hour:
        return same_hour[0]

    parent_candidates = []
    if baseline_input.is_file() and baseline_input.parent.exists():
        parent_candidates = sorted(baseline_input.parent.glob("*.tif")) + sorted(baseline_input.parent.glob("*.tiff"))
        same_hour = [p for p in parent_candidates if file_hour_utc(p) == hour_utc]
        if same_hour:
            return same_hour[0]

    if candidates:
        fallback = candidates[0]
        print(
            f"WARNING: no baseline found for {hour_utc:02d}:00 UTC; "
            f"using {fallback.name}. Prefer supplying all hourly baseline rasters."
        )
        return fallback

    raise FileNotFoundError(f"No baseline GeoTIFFs found: {baseline_input}")


def event_offset(surface: RasterSurface, baseline_input: Path) -> np.ndarray:
    baseline_path = matching_baseline_for_hour(baseline_input, surface.hour_utc)
    baseline_arr, baseline_profile = read_raster(baseline_path)
    baseline_arr = reproject_to_match(baseline_arr, baseline_profile, surface.profile)
    return surface.array - baseline_arr


def aggregate_offsets(offsets: list[np.ndarray]) -> np.ndarray:
    """Aggregate selected heatwave-minus-baseline rasters cellwise.

    The default is a mean across events. This avoids the misleading
    interpretation of a cumulative event count and keeps the plotted unit in °C.
    """
    stack = np.stack(offsets, axis=0)
    with np.errstate(all="ignore"):
        if AGGREGATION == "mean":
            return np.nanmean(stack, axis=0)
        if AGGREGATION == "mean_positive":
            return np.nanmean(np.where(stack > 0, stack, np.nan), axis=0)
        if AGGREGATION == "max":
            return np.nanmax(stack, axis=0)
    raise ValueError(f"Unsupported AGGREGATION: {AGGREGATION}")


def rasterize_polygons_to_mask(gdf: gpd.GeoDataFrame, profile: dict) -> np.ndarray:
    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        raise ValueError("No geometries available for rasterization.")
    arr = rasterize(
        shapes=[(geom, 1) for geom in geoms],
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        all_touched=False,
        dtype=np.uint8,
    )
    return arr.astype(bool)


def raster_box_gdf(profile: dict) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[raster_bounds_polygon(profile)], crs=profile["crs"])


def clip_vectors_to_raster_extent(gdf: gpd.GeoDataFrame, profile: dict, label: str) -> gpd.GeoDataFrame:
    """Crop vector subdivisions to the model raster extent before plotting/zonal stats."""
    clipped = gpd.clip(clean_geometries(gdf, label), raster_box_gdf(profile), keep_geom_type=False)
    clipped = clean_geometries(clipped, f"{label} clipped to raster extent")
    print(f"{label}: clipped to raster extent; features retained={len(clipped)}")
    return clipped


def read_and_match_fraction(path: Path, target_profile: dict, label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing target-domain raster for {label}: {path}")
    arr, profile = read_raster(path)
    arr = reproject_to_match(arr, profile, target_profile)
    return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)


def rasterize_bareground(target_profile: dict) -> np.ndarray:
    if not BAREGROUND_VECTOR_PATH.exists():
        raise FileNotFoundError(f"Missing bareground vector: {BAREGROUND_VECTOR_PATH}")
    bare = gpd.read_file(BAREGROUND_VECTOR_PATH)
    bare = align_vectors_to_raster_crs(bare, target_profile, "bareground")
    return rasterize_polygons_to_mask(bare, target_profile).astype(float)


def build_target_domain(profile: dict, city_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ecological model target domain and water/ocean context mask.

    Target domain = city ∩ (tree + non-natural woody vegetation) ∩ non-water
    ∩ non-building ∩ non-high-impervious. Non-natural woody vegetation over
    bareground is removed, matching the f3 map logic.
    """
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

    target_domain = (
        city_mask
        & vegetation_domain
        & (~water_domain)
        & (~building_excluded)
        & (~impervious_excluded)
    )

    print("[TARGET DOMAIN]")
    print(f"  city cells                 = {int(np.sum(city_mask))}")
    print(f"  vegetation cells in city   = {int(np.sum(vegetation_domain & city_mask))}")
    print(f"  water/ocean cells in city  = {int(np.sum(water_domain))}")
    print(f"  building-excluded in city  = {int(np.sum(building_excluded & city_mask))}")
    print(f"  impervious-excluded in city= {int(np.sum(impervious_excluded & city_mask))}")
    print(f"  target-domain cells        = {int(np.sum(target_domain))}")
    if not np.any(target_domain):
        raise ValueError("Target-domain mask is empty. Check predictor raster grids and thresholds.")
    return target_domain, water_domain


def zonal_mean_for_geometries(
    gdf: gpd.GeoDataFrame,
    arr: np.ndarray,
    profile: dict,
    domain_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean raster value per geometry, optionally restricted to target-domain cells."""
    means = []
    counts = []
    transform = profile["transform"]
    out_shape = (profile["height"], profile["width"])

    for geom in gdf.geometry:
        try:
            geom = make_valid(geom) if geom is not None and not geom.is_valid else geom
            if geom is None or geom.is_empty:
                means.append(np.nan); counts.append(0); continue
            mask = geometry_mask([geom], transform=transform, invert=True, out_shape=out_shape)
            if domain_mask is not None:
                mask &= domain_mask
            if not mask.any():
                means.append(np.nan); counts.append(0); continue
            vals = arr[mask]
            valid = np.isfinite(vals)
            counts.append(int(np.sum(valid)))
            means.append(float(np.nanmean(vals)) if valid.any() else np.nan)
        except Exception:
            means.append(np.nan); counts.append(0)

    return np.array(means, dtype="float64"), np.array(counts, dtype=int)


def zonal_metric_for_geometries(
    gdf: gpd.GeoDataFrame,
    arr: np.ndarray,
    profile: dict,
    *,
    domain_mask: np.ndarray | None = None,
    metric: str = "q90",
) -> tuple[np.ndarray, np.ndarray]:
    """Neighborhood statistic per geometry, optionally restricted to a domain."""
    stats = []
    counts = []
    transform = profile["transform"]
    out_shape = (profile["height"], profile["width"])

    for geom in gdf.geometry:
        try:
            geom = make_valid(geom) if geom is not None and not geom.is_valid else geom
            if geom is None or geom.is_empty:
                stats.append(np.nan); counts.append(0); continue
            mask = geometry_mask([geom], transform=transform, invert=True, out_shape=out_shape)
            if domain_mask is not None:
                mask &= domain_mask
            if not mask.any():
                stats.append(np.nan); counts.append(0); continue
            vals = arr[mask]
            vals = vals[np.isfinite(vals)]
            counts.append(int(vals.size))
            if vals.size == 0:
                stats.append(np.nan); continue
            if metric == "mean":
                stats.append(float(np.nanmean(vals)))
            elif metric == "q90":
                stats.append(float(np.nanpercentile(vals, 90)))
            elif metric == "q95":
                stats.append(float(np.nanpercentile(vals, 95)))
            elif metric == "max":
                stats.append(float(np.nanmax(vals)))
            else:
                raise ValueError(f"Unsupported ring metric: {metric}")
        except Exception:
            stats.append(np.nan); counts.append(0)

    return np.array(stats, dtype="float64"), np.array(counts, dtype=int)


def assign_kaupunginosat_to_suurpiirit(
    kaupunginosat: gpd.GeoDataFrame,
    suurpiirit: gpd.GeoDataFrame,
    suurpiiri_name_col: str,
) -> gpd.GeoDataFrame:
    """Assign each kaupunginosa to the suurpiiri with the largest intersection area."""
    k = clean_geometries(kaupunginosat, "kaupunginosat").copy()
    s = clean_geometries(suurpiirit[[suurpiiri_name_col, "geometry"]].copy(), "suurpiirit")
    k["_kid"] = np.arange(len(k))
    s = s.rename(columns={suurpiiri_name_col: "suurpiiri"})

    try:
        intersections = gpd.overlay(
            k[["_kid", "geometry"]],
            s,
            how="intersection",
            keep_geom_type=False,
            make_valid=True,
        )
        intersections = intersections[intersections.geometry.notna() & ~intersections.geometry.is_empty].copy()
        intersections["_area"] = intersections.geometry.area
        best = intersections.sort_values("_area", ascending=False).drop_duplicates("_kid")
        mapping = best.set_index("_kid")["suurpiiri"]
        k["suurpiiri"] = k["_kid"].map(mapping)
    except Exception as exc:
        print(f"WARNING: overlay assignment failed ({exc}); falling back to representative-point spatial join.")
        pts = k.copy()
        pts["geometry"] = pts.geometry.representative_point()
        joined = gpd.sjoin(pts[["_kid", "geometry"]], s[["suurpiiri", "geometry"]], how="left", predicate="within")
        mapping = joined.drop_duplicates("_kid").set_index("_kid")["suurpiiri"]
        k["suurpiiri"] = k["_kid"].map(mapping)

    missing = int(k["suurpiiri"].isna().sum())
    if missing:
        print(f"WARNING: {missing} kaupunginosat could not be assigned to a suurpiiri; labelled Unassigned.")
    return k.drop(columns=["_kid"])


def raster_extent(profile: dict) -> tuple[float, float, float, float]:
    transform = profile["transform"]
    width = profile["width"]
    height = profile["height"]
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return left, right, bottom, top


def smooth_nan_array(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that respects NaN masks, used for display only."""
    if sigma <= 0:
        return arr
    valid = np.isfinite(arr)
    if not np.any(valid):
        return arr
    filled = np.where(valid, arr, 0.0)
    weights = valid.astype(float)
    smoothed_vals = gaussian_filter(filled, sigma=sigma)
    smoothed_wts = gaussian_filter(weights, sigma=sigma)
    return np.divide(
        smoothed_vals,
        smoothed_wts,
        out=np.full_like(arr, np.nan, dtype=float),
        where=smoothed_wts > 0,
    )


def prettify_neighborhood_name(name: str) -> str:
    """Convert neighborhood names to proper capitalization and shorten a few long labels."""
    s = str(name).strip()
    s = s.lower().title()
    if "Mustikkamaa-Korkeasaari" in s:
        return "Mustik.-Kork."
    if "Vanhankaupunki" in s or "Vanhakaupunki" in s:
        return "Vanhankaup."
    if "Kaartinkaupunki" in s or "Kaartinaupunki" in s or "Kaartinkaupun" in s:
        return "Kaartinkaup."
    return s


def radial_offset_xy(theta: float, offset_points: float) -> tuple[float, float]:
    """Point offset along the local radial direction in display space."""
    display_angle = (np.pi / 2 - theta) % (2 * np.pi)
    return (offset_points * np.cos(display_angle), offset_points * np.sin(display_angle))


def outline_effects(width: float = TEXT_OUTLINE_WIDTH):
    return [pe.withStroke(linewidth=width, foreground="white")]


def label_too_close(ax, placed_xy, theta: float, radius: float, dx: float, dy: float, min_sep_px: float) -> bool:
    """Greedy screen-space collision check for annotations."""
    x0, y0 = ax.transData.transform((theta, radius))
    x, y = x0 + dx * ax.figure.dpi / 72.0, y0 + dy * ax.figure.dpi / 72.0
    for px, py in placed_xy:
        if (x - px) ** 2 + (y - py) ** 2 < min_sep_px ** 2:
            return True
    placed_xy.append((x, y))
    return False


def draw_text_on_arc(
    ax,
    text: str,
    radius: float,
    theta_center: float,
    theta_span: float,
    *,
    fontsize: float | None = None,
    fontweight: str | None = None,
    color: str = "black",
    zorder: int = 6,
    radial_offset_points: float = 0.0,
) -> None:
    """Draw text character-by-character along an invisible circular arc.

    Characters are aligned tangentially; positive radial offsets move them
    outward, negative offsets inward, in display-space points.
    """
    text = str(text)
    if not text:
        return

    chars = list(text)
    n_chars = len(chars)
    if n_chars == 1:
        thetas = np.array([theta_center], dtype=float)
    else:
        char_step = min(0.060, max(0.026, theta_span / max(n_chars + 1, 2)))
        total_span = char_step * (n_chars - 1)
        offsets = np.linspace(-total_span / 2, total_span / 2, n_chars)
        thetas = theta_center + offsets

    display_angle_center = (np.pi / 2 - theta_center) % (2 * np.pi)
    baseline_rotation_center = np.degrees(display_angle_center) - 90
    if np.cos(np.deg2rad(baseline_rotation_center)) < 0:
        chars = chars[::-1]
        thetas = thetas[::-1]

    for ch, theta in zip(chars, thetas):
        if ch == " ":
            continue
        display_angle = (np.pi / 2 - theta) % (2 * np.pi)
        rotation = np.degrees(display_angle) - 90
        dx, dy = radial_offset_xy(theta, radial_offset_points)
        ann = ax.annotate(
            ch,
            xy=(theta, radius),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            zorder=zorder,
            annotation_clip=False,
        )
        ann.set_path_effects(outline_effects())



def plot_figure(
    city_outline: gpd.GeoDataFrame,
    suurpiirit: gpd.GeoDataFrame,
    kaupunginosat: gpd.GeoDataFrame,
    agg_arr: np.ndarray,
    profile: dict,
    name_col: str,
    target_domain: np.ndarray,
) -> None:
    finite_vals = kaupunginosat["ring_metric_value"].to_numpy(dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if finite_vals.size == 0:
        raise ValueError("No finite ring-metric values were computed for kaupunginosat.")

    preferred_group_order = [
        "POHJOINEN",
        "ÖSTERSUNDOM",
        "ITÄINEN",
        "KOILLINEN",
        "KAAKKOINEN",
        "KESKINEN",
        "ETELÄINEN",
        "LÄNTINEN",
    ]
    available_groups = list(kaupunginosat["suurpiiri"].fillna("Unassigned").astype(str).unique())

    def _norm_group(x: str) -> str:
        return str(x).strip().upper()

    group_names = [g for pref in preferred_group_order for g in available_groups if _norm_group(g) == pref]
    group_names += [g for g in available_groups if g not in group_names]

    ordered_groups = []
    for group_name in group_names:
        group = kaupunginosat[kaupunginosat["suurpiiri"].fillna("Unassigned").astype(str) == str(group_name)]
        if len(group):
            ordered_groups.append(group.sort_values("ring_metric_value", ascending=False))

    bars_gdf = gpd.GeoDataFrame(data=np.concatenate([g.index.to_numpy() for g in ordered_groups]), columns=["idx"])
    bars_gdf = kaupunginosat.loc[bars_gdf["idx"].to_numpy()].reset_index(drop=True)
    colors = dict(zip(group_names, plt.get_cmap("tab20")(np.linspace(0, 1, len(group_names)))))

    values = bars_gdf["ring_metric_value"].to_numpy(dtype=float)
    names = bars_gdf[name_col].map(prettify_neighborhood_name).to_numpy()
    groups = bars_gdf["suurpiiri"].fillna("Unassigned").astype(str).to_numpy()

    fig = plt.figure(figsize=(9.6, 9.6), facecolor="white")
    ax_polar = fig.add_axes(POLAR_BOX, polar=True)
    ax_polar.set_zorder(3)
    ax_polar.set_facecolor("none")

    map_arr = np.where(target_domain, np.maximum(agg_arr, 0.0), np.nan)
    map_display = smooth_nan_array(map_arr, MAP_SMOOTH_SIGMA)
    map_display = np.where(target_domain, map_display, np.nan)
    masked = np.ma.masked_invalid(map_display)
    vals_for_scale = map_display[np.isfinite(map_display)]
    positive_vals = vals_for_scale[vals_for_scale > 0]
    if positive_vals.size:
        vmin = float(np.nanpercentile(positive_vals, 8))
        vmax = float(np.nanpercentile(positive_vals, 99))
    else:
        vmin, vmax = 0.0, 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)

    n = len(values)
    data_min = float(np.nanmin(values))
    data_max = float(np.nanmax(values))
    data_range = data_max - data_min
    if np.isclose(data_range, 0.0):
        data_range = max(0.1, data_max * 0.1 if np.isfinite(data_max) else 0.1)

    ring_display_floor = max(0.0, data_min - RING_DISPLAY_FLOOR_PAD * data_range)
    display_values = values - ring_display_floor
    display_values = np.where(np.isfinite(display_values), np.maximum(display_values, 0.0), np.nan)
    plot_values = display_values * RING_RADIUS_SCALE

    inner_ring_r = 0.0
    outer_pad = max(0.08, 0.12 * float(np.nanmax(plot_values)))
    outer_radius = float(np.nanmax(plot_values) + outer_pad)
    hole_r = HOLE_R
    rlim_max = outer_radius + outer_pad * 1.12

    ax_polar.set_rorigin(-hole_r)
    ax_polar.set_rlim(0, rlim_max)
    ax_polar.set_xticks([])
    ax_polar.set_yticks([])
    ax_polar.spines["polar"].set_visible(False)
    ax_polar.grid(False)
    ax_polar.set_theta_offset(np.pi / 2)
    ax_polar.set_theta_direction(-1)

    gap = float(BAR_GROUP_GAP_SLOTS)
    total_slots = n + gap * len(group_names)
    slot = 2 * np.pi / total_slots
    bar_width = slot * 0.88

    angles = np.empty(n)
    widths = np.full(n, bar_width)
    group_bounds = []
    cursor = 0.0
    for group_name in group_names:
        idx = np.where(groups == str(group_name))[0]
        if len(idx) == 0:
            continue
        start_angle = cursor
        group_angles = cursor + (np.arange(len(idx)) + 0.5) * slot
        angles[idx] = group_angles
        span = len(idx) * slot
        group_bounds.append((start_angle, span, str(group_name), idx))
        cursor += span + gap * slot

    if group_bounds:
        first_start, first_span, _, _ = group_bounds[0]
        desired_center = 0.0
        current_center = first_start + first_span / 2
        rotation_shift = desired_center - current_center
        angles = (angles + rotation_shift) % (2 * np.pi)
        group_bounds = [((a0 + rotation_shift) % (2 * np.pi), span, name, idx) for a0, span, name, idx in group_bounds]

    theta_dense = np.linspace(0, 2 * np.pi, 720)

    # Compute the map axes from the same fixed polar geometry. With a fixed-size
    # canvas (no bbox_inches='tight'), this aligns the circular crop to the
    # unified inner ring much more reliably.
    polar_left, polar_bottom, polar_w, polar_h = POLAR_BOX
    inner_fraction = hole_r / (hole_r + rlim_max)
    map_size = polar_w * inner_fraction * 1.045
    map_left = polar_left + (polar_w - map_size) / 2
    map_bottom = polar_bottom + (polar_h - map_size) / 2 + 0.015
    ax_map = fig.add_axes([map_left, map_bottom, map_size, map_size])
    ax_map.set_zorder(1)

    im = ax_map.imshow(
        masked,
        extent=raster_extent(profile),
        origin="upper",
        cmap=CMAP,
        norm=norm,
        zorder=1,
        interpolation="bilinear",
    )

    clip_circle = Circle((0.5, 0.5), 0.5, transform=ax_map.transAxes)
    im.set_clip_path(clip_circle)

    # Only internal map boundaries. No outer city outline.
    kaupunginosat.boundary.plot(ax=ax_map, color="white", linewidth=0.10, alpha=0.18, zorder=3)
    for coll in ax_map.collections:
        coll.set_clip_path(clip_circle)
    for line in ax_map.lines:
        line.set_clip_path(clip_circle)

    minx, miny, maxx, maxy = city_outline.total_bounds
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    side = max(maxx - minx, maxy - miny) * 1.01
    ax_map.set_xlim(cx - side / 2, cx + side / 2)
    ax_map.set_ylim(cy - side / 2, cy + side / 2)
    ax_map.set_aspect("equal")
    ax_map.axis("off")

    cax = fig.add_axes([map_left + 0.22 * map_size, map_bottom + 0.15 * map_size, 0.56 * map_size, 0.018])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.set_label(colorbar_label(), labelpad=2.0, fontsize=8.8)
    cbar.ax.tick_params(labelsize=7.8, pad=0.8)

    ax_polar.plot(
        theta_dense,
        np.full_like(theta_dense, inner_ring_r),
        color="black",
        linewidth=1.0,
        alpha=0.95,
        zorder=5,
    )

    ring_candidates = np.linspace(data_min, data_max, 5)
    for val in ring_candidates:
        r = max(0.0, val - ring_display_floor) * RING_RADIUS_SCALE
        if r <= 0:
            continue
        ax_polar.plot(
            theta_dense,
            np.full_like(theta_dense, r),
            color="0.65",
            linewidth=0.40,
            alpha=0.50,
            zorder=0,
            linestyle="--",
        )
        ringtxt = ax_polar.text(np.pi / 2, r, f"{val:.1f}", ha="center", va="bottom", fontsize=7.4)
        ringtxt.set_path_effects(outline_effects(1.8))

    n_layers = 16
    for angle, width, height_total, group in zip(angles, widths, plot_values, groups):
        if not np.isfinite(height_total) or height_total <= 0:
            continue
        base = np.array(mpl.colors.to_rgb(colors[group]))
        for i in range(n_layers):
            f0 = i / n_layers
            f1 = (i + 1) / n_layers
            bottom = inner_ring_r + height_total * f0
            height = height_total * (f1 - f0)
            t = f1 ** 1.35
            color = (1 - t) * np.ones(3) + t * base
            ax_polar.bar(angle, height, width=width, bottom=bottom, color=color, edgecolor="none", linewidth=0, zorder=1)

    ax_polar.bar(
        angles,
        np.nan_to_num(plot_values, nan=0.0),
        width=widths,
        bottom=inner_ring_r,
        color="none",
        edgecolor="white",
        linewidth=0.55,
        zorder=2,
    )

    show_bar_labels = n <= BAR_LABEL_MAX_N
    if show_bar_labels:
        for angle, name, height_total in zip(angles, names, plot_values):
            if not np.isfinite(height_total) or height_total < MIN_BAR_LABEL_VALUE:
                continue
            display_angle = (np.pi / 2 - angle) % (2 * np.pi)
            rotation = np.degrees(display_angle)
            ha = "left"
            if np.pi / 2 < display_angle < 3 * np.pi / 2:
                rotation += 180
                ha = "right"
            dx, dy = radial_offset_xy(angle, NEIGHBORHOOD_LABEL_OFFSET_PTS)
            ann = ax_polar.annotate(
                name,
                xy=(angle, NEIGHBORHOOD_LABEL_RADIUS),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=NEIGHBORHOOD_FONT_SIZE,
                zorder=7,
                annotation_clip=False,
            )

    for start, span, group_name, idx in group_bounds:
        ax_polar.plot(
            [start, start],
            [inner_ring_r, outer_radius],
            color="black",
            linewidth=0.75,
            zorder=4,
        )
        label_angle = (start + span / 2) % (2 * np.pi)
        draw_text_on_arc(
            ax_polar,
            group_name,
            GROUP_LABEL_RADIUS,
            label_angle,
            span,
            fontsize=GROUP_LABEL_FONT_SIZE,
            fontweight="bold",
            color="black",
            zorder=8,
            radial_offset_points=GROUP_LABEL_OFFSET_PTS,
        )

    placed_value_label_xy = []
    for angle, value, height_total in zip(angles, values, plot_values):
        if not np.isfinite(value) or not np.isfinite(height_total):
            continue
        display_angle = (np.pi / 2 - angle) % (2 * np.pi)
        rotation = np.degrees(display_angle)
        ha = "left"
        if np.pi / 2 < display_angle < 3 * np.pi / 2:
            rotation += 180
            ha = "right"
        dx, dy = radial_offset_xy(angle, VALUE_LABEL_OFFSET_PTS)
        if label_too_close(ax_polar, placed_value_label_xy, angle, height_total, dx, dy, VALUE_LABEL_MIN_SEP_PX):
            continue
        ann = ax_polar.annotate(
            f"{value:.1f}",
            xy=(angle, height_total),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=VALUE_LABEL_FONT_SIZE,
            zorder=8,
            annotation_clip=False,
        )
        ann.set_path_effects(outline_effects(1.8))

    fig.savefig(OUTPUT_SVG, format="svg", facecolor="white")
    plt.close(fig)
    print(f"Wrote SVG: {OUTPUT_SVG}")


def colorbar_label() -> str:
    if AGGREGATION == "mean":
        return "Mean ΔT vs avg (°C)"
    if AGGREGATION == "mean_positive":
        return "+Mean ΔT vs avg (°C)"
    if AGGREGATION == "max":
        return "Max ΔT vs avg (°C)"
    return "ΔT vs avg (°C)"


def main() -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    apply_global_plotting_settings(GLOBAL_SETTINGS)

    suurpiirit = gpd.read_file(SUURPIIRIT_GEOJSON)
    kaupunginosat = gpd.read_file(KAUPUNGINOSAT_GEOJSON)

    # Select event peak rasters first; use the first selected raster grid as target CRS/grid.
    surfaces = [select_peak_surface(label, path) for label, path in HEATWAVE_INPUTS.items()]
    target_profile = surfaces[0].profile
    if target_profile.get("crs") is None:
        raise ValueError("Raster CRS is missing; cannot align vector boundaries.")

    suurpiirit = align_vectors_to_raster_crs(suurpiirit, target_profile, "suurpiirit")
    kaupunginosat = align_vectors_to_raster_crs(kaupunginosat, target_profile, "kaupunginosat")

    suurpiiri_name_col = first_existing_column(suurpiirit, SUURPIIRI_NAME_COLUMNS, "suurpiiri")
    kaupunginosa_name_col = first_existing_column(kaupunginosat, KAUPUNGINOSA_NAME_COLUMNS, "kaupunginosa")

    # Crop the official subdivisions to the prediction raster extent before
    # computing zones and drawing boundaries. This removes the southern
    # out-of-domain district geometry visible in the previous SVG.
    suurpiirit = clip_vectors_to_raster_extent(suurpiirit, target_profile, "suurpiirit")
    kaupunginosat = clip_vectors_to_raster_extent(kaupunginosat, target_profile, "kaupunginosat")

    kaupunginosat = assign_kaupunginosat_to_suurpiirit(kaupunginosat, suurpiirit, suurpiiri_name_col)
    city_mask = rasterize_polygons_to_mask(suurpiirit.dissolve(), target_profile)
    target_domain, water_domain = build_target_domain(target_profile, city_mask)

    # Reproject all event offsets to the target grid before aggregation.
    offsets = []
    for surface in surfaces:
        off = event_offset(surface, BASELINE_MEAN_INPUT)
        off = reproject_to_match(off, surface.profile, target_profile)
        offsets.append(off)

    agg_arr = aggregate_offsets(offsets)
    agg_arr = np.where(target_domain, agg_arr, np.nan)
    city_mean = float(np.nanmean(agg_arr[target_domain]))
    print(f"Helsinki-wide target-domain mean offset above July mean baseline = {city_mean:.3f} °C")

    means, counts = zonal_mean_for_geometries(kaupunginosat, agg_arr, target_profile, domain_mask=target_domain)
    ring_vals, ring_counts = zonal_metric_for_geometries(kaupunginosat, agg_arr, target_profile, domain_mask=target_domain, metric=RING_METRIC)
    kaupunginosat["raw_offset_value"] = means
    kaupunginosat["deviation_from_city_mean"] = means - city_mean
    kaupunginosat["ring_metric_value"] = ring_vals
    kaupunginosat["target_pixel_count"] = counts
    kaupunginosat["ring_metric_pixel_count"] = ring_counts
    kaupunginosat["offset_value"] = kaupunginosat["ring_metric_value"]

    output_cols = [kaupunginosa_name_col, "suurpiiri", "raw_offset_value", "deviation_from_city_mean", "ring_metric_value", "target_pixel_count", "ring_metric_pixel_count"]
    kaupunginosat[output_cols].sort_values(["suurpiiri", "ring_metric_value"], ascending=[True, False]).to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote values CSV: {OUTPUT_CSV}")

    if not np.isfinite(kaupunginosat["ring_metric_value"].to_numpy(dtype=float)).any():
        print("Raster bounds:", raster_extent(target_profile))
        print("Suurpiirit bounds:", tuple(suurpiirit.total_bounds))
        print("Kaupunginosat bounds:", tuple(kaupunginosat.total_bounds))
        raise ValueError("No finite zonal means were computed. The vector boundaries still do not overlap finite raster cells; check CRS and raster extent.")

    valid_count = int(np.isfinite(kaupunginosat["ring_metric_value"].to_numpy(dtype=float)).sum())
    print(f"Neighborhood ring-metric values computed for {valid_count}/{len(kaupunginosat)} kaupunginosat.")
    print(f"Map values are raw mean offsets from the July mean baseline; the ring uses {RING_METRIC} over target-domain pixels within each neighborhood, with the v10 ring geometry restored. raw_offset_value and deviation_from_city_mean are still written to the CSV.")

    city_outline = clean_geometries(suurpiirit, "city outline").dissolve()
    plot_figure(city_outline, suurpiirit, kaupunginosat, agg_arr, target_profile, kaupunginosa_name_col, target_domain)


if __name__ == "__main__":
    main()
