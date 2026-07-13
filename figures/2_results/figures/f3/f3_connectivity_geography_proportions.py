#!/usr/bin/env python3
"""
script quantifies which landscape elements disproportionately structure modeled 
thermal movement in the green network, in two levels:

LEVEL 1: vegetation proportionality
  To find out whether certain vegetation types carry more thermal current than their area predicts?
  metric: disproportionality ratio = (% of current) / (% of area)
  interpretation:
    - > 1.0: Vegetation is overrepresented in high-current pixels
    - < 1.0: Vegetation is underrepresented in high-current pixels
  
LEVEL 2: bottleneck geography and corridor analysis
Where do thermal bottlenecks (95th percentile current) occur?
  Categories: 
    - overlapping geography (island/peninsula/mainland)
    - vegetation structure (tree corridors <30m-50m width, non-woody, other)
  Metrics:
    - Density: % of category pixels that are bottlenecks (constraint severity)
    - Proportion: % of all bottleneck pixels in category (contribution to total constraint)

first tested all of this individually, then refactored into one script with copilot. 
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt


# =============================================================================
# CONFIGURATION
# =============================================================================
# change all the stuff to whatever is needed

DATA = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
OMNI = DATA / "omniscape"

# Input rasters
CUM_CURRENT_PATH = OMNI / "output" / "conditional_multiruns" / "condition_average__pm0p5deg" / "cum_currmap.tif"
NORMALIZED_CURRENT_PATH = OMNI / "output" / "conditional_multiruns" / "condition_average__pm0p5deg" / "normalized_cum_currmap.tif"

TREE_PATH = DATA / "predictorstack" / "TREE_FRAC_10m.tif"
NWN_PATH = DATA / "predictorstack" / "NWN_FRAC_10m.tif"
WATER_PATH = DATA / "predictorstack" / "WATER_FRAC_10m_Helsinki.tif"
OCEAN_PATH = DATA / "predictorstack" / "OCEAN_FRAC_10m_Helsinki.tif"
BAREGROUND_VECTOR_PATH = DATA / "LULC" / "lc_bareground.gpkg"
PERUSPIIRI_PATH = DATA / "figures" / "offset_figure" / "peruspiiri_WFS.gpkg"

# Landscape mask vectors (EPSG:3879)
ISLAND_MASK_PATH = DATA / "hel_island_mask.gpkg"
PENINSULA_MASK_PATH = DATA / "hel_penin_mask.gpkg"

# Output
FIG_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\2_results\figures\f3")
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = FIG_DIR / "f3_connectivity_geography_proportions.csv"

# Analysis parameters
BOTTLENECK_PERCENTILE = 95
CORRIDOR_WIDTH_PIXELS = 3  # 30m at 10m resolution
PIXEL_SIZE_M = 10.0
CRS_EPSG = "EPSG:3879"

# Target domain parameters
VEGETATION_MIN_FRACTION = 0.05
WATER_MIN_FRACTION = 0.05
BUILDING_EXCLUDE_FRACTION = 0.0
IMPERVIOUS_EXCLUDE_FRACTION = 0.5

IMPERVIOUS_PATH = DATA / "predictorstack" / "IMPERV_FRAC_10m_Helsinki.tif"
BUILDING_PATH = DATA / "predictorstack" / "BLDG_FRAC_10m.tif"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RasterInfo:
    """Metadata about a raster."""
    array: np.ndarray
    profile: dict
    transform: object
    crs: object
    width: int
    height: int
    nodata: float | int | None


# =============================================================================
# RASTER I/O
# =============================================================================

def load_raster(path: Path) -> RasterInfo:
    """Load a single raster band and metadata."""
    with rasterio.open(path) as src:
        array = src.read(1)
        return RasterInfo(
            array=array,
            profile=src.profile,
            transform=src.transform,
            crs=src.crs,
            width=src.width,
            height=src.height,
            nodata=src.nodata,
        )


# =============================================================================
# VECTOR RASTERIZATION
# =============================================================================
# we need to rasterize the vector masks to the general predictor extent

def rasterize_vector_mask(
    vector_path: Path,
    reference_raster: RasterInfo,
) -> np.ndarray:
    """Rasterize a vector layer to match reference raster extent."""
    gdf = gpd.read_file(vector_path)
    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(reference_raster.height, reference_raster.width),
        transform=reference_raster.transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )
    return mask.astype(bool)


def rasterize_peruspiiri_boundary(reference_raster: RasterInfo) -> np.ndarray:
    """Rasterize peruspiiri (district) boundary."""
    return rasterize_vector_mask(PERUSPIIRI_PATH, reference_raster)


def rasterize_bareground(reference_raster: RasterInfo) -> np.ndarray:
    """Rasterize bareground mask."""
    return rasterize_vector_mask(BAREGROUND_VECTOR_PATH, reference_raster)


# =============================================================================
# TARGET DOMAIN CONSTRUCTION
# ============================================================================
def build_target_domain(
    tree: np.ndarray,
    nwn: np.ndarray,
    water: np.ndarray,
    ocean: np.ndarray,
    impervious: np.ndarray,
    building: np.ndarray,
    bareground: np.ndarray,
    peruspiiri_mask: np.ndarray,
) -> np.ndarray:
    """
    build the analysis target domains, this is essentially same thing stolen from earlier analysis just refactored
    into a function
    
    Include pixels that are:
    - Within peruspiiri (Helsinki boundaries)
    - contain sufficient vegetation
    - are not water/ocean
    - are not buildings
    - do not have high impervious cover
    """
    tree = np.clip(np.nan_to_num(tree, nan=0.0), 0.0, 1.0)
    nwn = np.clip(np.nan_to_num(nwn, nan=0.0), 0.0, 1.0)
    water = np.clip(np.nan_to_num(water, nan=0.0), 0.0, 1.0)
    ocean = np.clip(np.nan_to_num(ocean, nan=0.0), 0.0, 1.0)
    impervious = np.clip(np.nan_to_num(impervious, nan=0.0), 0.0, 1.0)
    building = np.clip(np.nan_to_num(building, nan=0.0), 0.0, 1.0)
    
    # vegetation definition: tree + non-woody (excluding bareground pixels)
    nwn_veg = np.where(bareground > 0, 0.0, nwn)
    nwn_veg = np.clip(nwn_veg, 0.0, 1.0)
    vegetation_fraction = np.clip(tree + nwn_veg, 0.0, 1.0)
    
    vegetation_domain = (
        np.isfinite(vegetation_fraction)
        & (vegetation_fraction >= VEGETATION_MIN_FRACTION)
    )
    
    water_domain = (water >= WATER_MIN_FRACTION) | (ocean >= WATER_MIN_FRACTION)
    building_excluded = building > BUILDING_EXCLUDE_FRACTION
    impervious_excluded = impervious > IMPERVIOUS_EXCLUDE_FRACTION
    
    target_domain = (
        peruspiiri_mask
        & vegetation_domain
        & (~water_domain)
        & (~building_excluded)
        & (~impervious_excluded)
    )
    
    return target_domain


# =============================================================================
# LEVEL 1: VEGETATION DISPROPORTIONALITY
# =============================================================================

def compute_vegetation_disproportionality(
    vegetation_frac: np.ndarray,
    cumulative_current: np.ndarray,
    target_domain: np.ndarray,
) -> dict:
    """
    compute disproportionality of vegetation in high-current areas.
    disproportionality = (share of current) / (share of area) (weighted average)
    
    Interpretation:
      > 1: Vegetation carries MORE current than area predicts (overrepresented)
      < 1: Vegetation carries LESS current than area predicts (underrepresented)
    """
    # Only use pixels with valid current within target domain
    valid = target_domain & np.isfinite(cumulative_current) & np.isfinite(vegetation_frac)
    
    if np.sum(valid) == 0:
        return {
            "share_of_current_pct": np.nan,
            "share_of_area_pct": np.nan,
            "disproportionality_ratio": np.nan,
        }
    
    total_current = np.sum(cumulative_current[valid])
    
    if total_current <= 0:
        return {
            "share_of_current_pct": np.nan,
            "share_of_area_pct": np.nan,
            "disproportionality_ratio": np.nan,
        }
    
    # Weighted average of vegetation fraction (current-weighted)
    weights = cumulative_current[valid]
    veg_values = vegetation_frac[valid]
    
    share_of_current = np.sum(veg_values * weights) / total_current
    share_of_area = np.mean(veg_values)
    
    disproportionality = share_of_current / share_of_area if share_of_area > 0 else np.nan
    
    return {
        "share_of_current_pct": share_of_current * 100.0,
        "share_of_area_pct": share_of_area * 100.0,
        "disproportionality_ratio": disproportionality,
    }


# =============================================================================
# LEVEL 2: BOTTLENECK DETECTION & GEOGRAPHY
# =============================================================================

def detect_bottlenecks(
    normalized_current: np.ndarray,
    target_domain: np.ndarray,
) -> np.ndarray:
    """
    identify bottleneck pixels as those in the upper tail (95th percentile)
    of normalized current within the target domain.
    """
    valid = normalized_current[target_domain & np.isfinite(normalized_current)]
    if valid.size == 0:
        return np.zeros_like(target_domain, dtype=bool)
    
    threshold = np.nanpercentile(valid, BOTTLENECK_PERCENTILE)
    bottleneck_mask = (normalized_current >= threshold) & target_domain
    
    return bottleneck_mask


def detect_tree_corridors(
    tree_frac: np.ndarray,
    target_domain: np.ndarray,
    corridor_width_pixels: float = None,
) -> np.ndarray:
    """
    detect narrow linear patches of trees (corridors < specified width).
    tuned this a bit, should now be robust
    Method:
    1. Identify tree-dominant pixels (tree_frac > 0.5)
    2. compute distance to nearest non-tree pixel
    3. threshold: pixels with distance < corridor_width_pixels/2
    
    Args:
        tree_frac: Tree fraction raster
        target_domain: Boolean mask of analysis domain
        corridor_width_pixels: Width threshold in pixels (if None, uses CORRIDOR_WIDTH_PIXELS)
    """
    if corridor_width_pixels is None:
        corridor_width_pixels = CORRIDOR_WIDTH_PIXELS
        
    tree_dominant = (tree_frac > 0.5) & target_domain
    
    if not np.any(tree_dominant):
        return np.zeros_like(tree_dominant, dtype=bool)
    
    dist_transform = distance_transform_edt(tree_dominant)
    corridor_threshold = corridor_width_pixels / 2.0
    corridor_mask = tree_dominant & (dist_transform < corridor_threshold)
    
    return corridor_mask


def classify_landscape_categories(
    tree_frac: np.ndarray,
    nwn_frac: np.ndarray,
    island_mask: np.ndarray,
    peninsula_mask: np.ndarray,
    mainland_mask: np.ndarray,
    target_domain: np.ndarray,
    corridor_width_pixels: float = None,
) -> dict[str, np.ndarray]:
    """
    Classify target domain using overlapping categories.
    Two independent dimensions:
    1. VEGETATION: tree_corridors, nwn_dominant, other_veg (independent)
    2. GEOGRAPHY: island, peninsula, mainland (mutually exclusive)
    Creates 9 combinations to show where bottlenecks oppur
    
    Args:
        corridor_width_pixels: Width threshold in pixels for corridor detection, std. 30m
    """
    # Vegetation classification
    tree_corridors = detect_tree_corridors(tree_frac, target_domain, corridor_width_pixels)
    nwn_dominant = (nwn_frac > tree_frac) & target_domain
    other_veg = target_domain & ~tree_corridors & ~nwn_dominant
    
    # Geography classification
    islands = island_mask & target_domain
    peninsulas = peninsula_mask & target_domain
    mainland = mainland_mask & target_domain
    
    # Overlapping combinations
    categories = {
        "tree_corridor_island": tree_corridors & islands,
        "tree_corridor_peninsula": tree_corridors & peninsulas,
        "tree_corridor_mainland": tree_corridors & mainland,
        "nwn_island": nwn_dominant & islands,
        "nwn_peninsula": nwn_dominant & peninsulas,
        "nwn_mainland": nwn_dominant & mainland,
        "other_island": other_veg & islands,
        "other_peninsula": other_veg & peninsulas,
        "other_mainland": other_veg & mainland,
    }
    
    return categories


def compute_category_metrics(
    category_name: str,
    category_mask: np.ndarray,
    cumulative_current: np.ndarray,
    bottleneck_mask: np.ndarray,
    pixel_area_m2: float = PIXEL_SIZE_M ** 2,
) -> dict:
    """
    compute bottleneck metrics for a landscape category.
    
    metrics:
    - total_pixels: Number of pixels in category
    - bottleneck_pixels: Number of bottleneck pixels in category
    - bottleneck_density: % of category pixels that are bottlenecks
    - bottleneck_proportion: % of all bottlenecks in this category
    """
    category_bottlenecks = category_mask & bottleneck_mask
    total_pixels = int(np.sum(category_mask))
    bottleneck_pixels = int(np.sum(category_bottlenecks))
    
    return {
        "category": category_name,
        "total_pixels": total_pixels,
        "total_area_m2": round(total_pixels * pixel_area_m2, 1),
        "bottleneck_pixels": bottleneck_pixels,
        "bottleneck_area_m2": round(bottleneck_pixels * pixel_area_m2, 1),
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print()
    print("=" * 80)
    print("THERMAL CONNECTIVITY METRICS ANALYSIS - HELSINKI URBAN GREEN AREAS")
    print("=" * 80)
    print()
    
    # Load data
    print("[1/7] Loading rasters...")
    current_info = load_raster(CUM_CURRENT_PATH)
    norm_current_info = load_raster(NORMALIZED_CURRENT_PATH)
    tree_info = load_raster(TREE_PATH)
    nwn_info = load_raster(NWN_PATH)
    water_info = load_raster(WATER_PATH)
    ocean_info = load_raster(OCEAN_PATH)
    impervious_info = load_raster(IMPERVIOUS_PATH)
    building_info = load_raster(BUILDING_PATH)
    
    # Ensure cumulative current is positive (handle negative values from model output)
    current_info.array = np.maximum(current_info.array, 0.0)
    
    # Rasterize vector masks
    print("[2/7] Rasterizing landscape masks...")
    bareground = rasterize_bareground(current_info)
    peruspiiri = rasterize_peruspiiri_boundary(current_info)
    island_mask = rasterize_vector_mask(ISLAND_MASK_PATH, current_info)
    peninsula_mask = rasterize_vector_mask(PENINSULA_MASK_PATH, current_info)
    mainland_mask = peruspiiri & ~island_mask & ~peninsula_mask
    
    print(f"  Peruspiiri pixels: {int(np.sum(peruspiiri)):,}")
    print(f"  Island pixels: {int(np.sum(island_mask)):,}")
    print(f"  Peninsula pixels: {int(np.sum(peninsula_mask)):,}")
    print(f"  Mainland pixels: {int(np.sum(mainland_mask)):,}")
    print()
    
    # Build target domain
    print("[3/7] Building target domain...")
    target_domain = build_target_domain(
        tree_info.array,
        nwn_info.array,
        water_info.array,
        ocean_info.array,
        impervious_info.array,
        building_info.array,
        bareground,
        peruspiiri,
    )
    print(f"  Target domain pixels: {int(np.sum(target_domain)):,}")
    print()
    
    # LEVEL 1: Vegetation disproportionality
    print("[4/7] LEVEL 1 - Computing vegetation disproportionality...")
    tree_dispr = compute_vegetation_disproportionality(
        tree_info.array, current_info.array, target_domain
    )
    nwn_dispr = compute_vegetation_disproportionality(
        nwn_info.array, current_info.array, target_domain
    )
    print(f"  Trees: {tree_dispr['share_of_current_pct']:.1f}% of current, "
          f"{tree_dispr['share_of_area_pct']:.1f}% of area -> "
          f"{tree_dispr['disproportionality_ratio']:.2f}x disproportionate")
    print(f"  NWN:  {nwn_dispr['share_of_current_pct']:.1f}% of current, "
          f"{nwn_dispr['share_of_area_pct']:.1f}% of area -> "
          f"{nwn_dispr['disproportionality_ratio']:.2f}x disproportionate")
    print()
    
    # LEVEL 2: Bottleneck geography
    print("[5/7] Detecting bottlenecks (95th percentile)...")
    bottleneck_mask = detect_bottlenecks(norm_current_info.array, target_domain)
    total_bn = int(np.sum(bottleneck_mask))
    print(f"  Total bottleneck pixels: {total_bn:,}")
    print()
    
    print("[6/7] Classifying landscape categories...")
    categories = classify_landscape_categories(
        tree_info.array,
        nwn_info.array,
        island_mask,
        peninsula_mask,
        mainland_mask,
        target_domain,
    )
    
    # Compute metrics
    print("[7/7] Computing category metrics...")
    category_metrics = []
    for cat_name, cat_mask in categories.items():
        metrics = compute_category_metrics(
            cat_name, cat_mask, current_info.array, bottleneck_mask
        )
        category_metrics.append(metrics)
    print()
    
    # Sensitivity analysis: corridor width variation (30-50m)
    print("[8/8] Sensitivity analysis - Corridor width variation...")
    sensitivity_results = []
    corridor_widths_pixels = [3, 4, 5]  # 30m, 40m, 50m at 10m resolution
    
    for width_px in corridor_widths_pixels:
        width_m = int(width_px * PIXEL_SIZE_M)
        categories_varied = classify_landscape_categories(
            tree_info.array,
            nwn_info.array,
            island_mask,
            peninsula_mask,
            mainland_mask,
            target_domain,
            corridor_width_pixels=width_px,
        )
        
        # Count tree corridor bottlenecks at this width
        tree_corridors_mask = categories_varied["tree_corridor_island"] | \
                             categories_varied["tree_corridor_peninsula"] | \
                             categories_varied["tree_corridor_mainland"]
        corridor_bn = int(np.sum(tree_corridors_mask & bottleneck_mask))
        corridor_proportion = (100.0 * corridor_bn / total_bn) if total_bn > 0 else 0
        
        sensitivity_results.append({
            "width_m": width_m,
            "width_px": width_px,
            "corridor_bottleneck_pixels": corridor_bn,
            "corridor_bottleneck_proportion": corridor_proportion,
        })
        print(f"  {width_m}m: {corridor_bn:,} bottleneck pixels ({corridor_proportion:.1f}%)")
    print()
    
    # =========================================================================
    # OUTPUT GENERATION
    # =========================================================================
    
    output_rows = []
    
    # Metadata section
    output_rows.append({
        "section": "METADATA",
        "metric": "Analysis Date",
        "value": "2026-07-13",
        "unit": "",
    })
    output_rows.append({
        "section": "METADATA",
        "metric": "Target Domain Pixels",
        "value": int(np.sum(target_domain)),
        "unit": "pixels",
    })
    output_rows.append({
        "section": "METADATA",
        "metric": "Bottleneck Threshold",
        "value": BOTTLENECK_PERCENTILE,
        "unit": "percentile",
    })
    output_rows.append({
        "section": "METADATA",
        "metric": "Corridor Width Definition",
        "value": CORRIDOR_WIDTH_PIXELS * PIXEL_SIZE_M,
        "unit": "meters",
    })
    output_rows.append({
        "section": "METADATA",
        "metric": "Pixel Resolution",
        "value": PIXEL_SIZE_M,
        "unit": "meters",
    })
    
    # LEVEL 1: Disproportionality
    output_rows.append({
        "section": "LEVEL 1: VEGETATION DISPROPORTIONALITY",
        "metric": "Tree - Share of Current",
        "value": round(tree_dispr['share_of_current_pct'], 2),
        "unit": "% of thermal current",
    })
    output_rows.append({
        "section": "LEVEL 1: VEGETATION DISPROPORTIONALITY",
        "metric": "Tree - Share of Area",
        "value": round(tree_dispr['share_of_area_pct'], 2),
        "unit": "% of green area",
    })
    output_rows.append({
        "section": "LEVEL 1: VEGETATION DISPROPORTIONALITY",
        "metric": "Tree - Disproportionality Ratio",
        "value": round(tree_dispr['disproportionality_ratio'], 2),
        "unit": "fold change",
    })
    output_rows.append({
        "section": "LEVEL 1: VEGETATION DISPROPORTIONALITY",
        "metric": "NWN - Share of Current",
        "value": round(nwn_dispr['share_of_current_pct'], 2),
        "unit": "% of thermal current",
    })
    output_rows.append({
        "section": "LEVEL 1: VEGETATION DISPROPORTIONALITY",
        "metric": "NWN - Share of Area",
        "value": round(nwn_dispr['share_of_area_pct'], 2),
        "unit": "% of green area",
    })
    output_rows.append({
        "section": "LEVEL 1: VEGETATION DISPROPORTIONALITY",
        "metric": "NWN - Disproportionality Ratio",
        "value": round(nwn_dispr['disproportionality_ratio'], 2),
        "unit": "fold change",
    })
    
    # LEVEL 2: Bottleneck categories
    output_rows.append({
        "section": "LEVEL 2: BOTTLENECK GEOGRAPHY & CORRIDORS",
        "metric": "Total Bottleneck Pixels",
        "value": total_bn,
        "unit": "pixels",
    })
    
    for metrics in category_metrics:
        density_pct = (100.0 * metrics['bottleneck_pixels'] / metrics['total_pixels']
                      if metrics['total_pixels'] > 0 else 0)
        proportion_pct = (100.0 * metrics['bottleneck_pixels'] / total_bn
                         if total_bn > 0 else 0)
        
        output_rows.append({
            "section": "LEVEL 2: BOTTLENECK GEOGRAPHY & CORRIDORS",
            "metric": f"{metrics['category']} - Total Pixels",
            "value": metrics['total_pixels'],
            "unit": "pixels",
        })
        output_rows.append({
            "section": "LEVEL 2: BOTTLENECK GEOGRAPHY & CORRIDORS",
            "metric": f"{metrics['category']} - Bottleneck Pixels",
            "value": metrics['bottleneck_pixels'],
            "unit": "pixels",
        })
        output_rows.append({
            "section": "LEVEL 2: BOTTLENECK GEOGRAPHY & CORRIDORS",
            "metric": f"{metrics['category']} - Bottleneck Density",
            "value": round(density_pct, 2),
            "unit": "% of category",
        })
        output_rows.append({
            "section": "LEVEL 2: BOTTLENECK GEOGRAPHY & CORRIDORS",
            "metric": f"{metrics['category']} - Bottleneck Proportion",
            "value": round(proportion_pct, 2),
            "unit": "% of total bottlenecks",
        })
    
    # SENSITIVITY ANALYSIS: Corridor width variation
    for result in sensitivity_results:
        output_rows.append({
            "section": "SENSITIVITY ANALYSIS: CORRIDOR WIDTH VARIATION",
            "metric": f"Tree corridors ({result['width_m']}m) - Bottleneck Pixels",
            "value": result['corridor_bottleneck_pixels'],
            "unit": "pixels",
        })
        output_rows.append({
            "section": "SENSITIVITY ANALYSIS: CORRIDOR WIDTH VARIATION",
            "metric": f"Tree corridors ({result['width_m']}m) - Bottleneck Proportion",
            "value": round(result['corridor_bottleneck_proportion'], 2),
            "unit": "% of total bottlenecks",
        })
    
    # Write CSV
    print("=" * 80)
    print(f"Writing results to: {OUTPUT_CSV}")
    print("=" * 80)
    
    df = pd.DataFrame(output_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print()
    print("Output CSV structure:")
    print("  - section: Analysis section (metadata, Level 1, Level 2)")
    print("  - metric: Specific measurement or category")
    print("  - value: Numerical result")
    print("  - unit: Unit of measurement")
    print()
    print(f"✓ Analysis complete. Results saved to: {OUTPUT_CSV}")
    print()


if __name__ == "__main__":
    main()
