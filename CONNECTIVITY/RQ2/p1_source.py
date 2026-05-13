#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

from rasterio.features import geometry_mask
from scipy.ndimage import label, find_objects, binary_dilation, distance_transform_edt


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
OMNI = BASE / "omniscape"
STACK = BASE / "predictorstack"

# Main source raster to diagnose
SOURCE_FILE = OMNI / "sources" / "source_p90_coolness_stability.tif"

# If your source is in a subfolder, use this instead:
# SOURCE_FILE = OMNI / "sources" / "thermal_refugia_simple_v2" / "source_p90_coolness_stability.tif"

OUT_DIR = OMNI / "diagnostics" / "source_pattern_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Land-cover / predictor rasters
LULC_FILES = {
    "tree": STACK / "TREE_FRAC_10m.tif",
    "nwn": STACK / "NWN_FRAC_10m.tif",
    "impervious": STACK / "IMPERV_FRAC_10m_Helsinki.tif",
    "building": STACK / "BLDG_FRAC_10m.tif",
    "water": STACK / "WATER_FRAC_10m_Helsinki.tif",
    "ocean": STACK / "OCEAN_FRAC_10m_Helsinki.tif",
    "rock": STACK / "ROCK_FRAC_10m_Helsinki.tif",
}

# Optional auxiliary data.
# If you have district/neighbourhood polygons, set this to a gpkg/shp path.
# The script will then summarize source values by polygon.
OPTIONAL_ZONE_POLYGONS: Optional[Path] = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\figures\offset_figure\peruspiiri_WFS.gpkg")
OPTIONAL_ZONE_NAME_FIELD = "nimi_fi"  # e.g. "nimi", "name", "district"


# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

HIGH_PERCENTILE = 90
VERY_HIGH_PERCENTILE = 95
LOW_PERCENTILE = 10

# For patch detection, use very high source pixels.
PATCH_PERCENTILE = VERY_HIGH_PERCENTILE

# Pixel connectivity:
# 1 = 4-neighbour connectivity
# 2 = 8-neighbour connectivity
CONNECTIVITY = 2

# Coastal diagnostic:
# cells within this distance from ocean are considered coastal.
COASTAL_DISTANCE_M = 300.0


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
    pixel_width: float
    pixel_height: float
    pixel_area_m2: float


# =============================================================================
# IO HELPERS
# =============================================================================

def load_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

        transform = src.transform
        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)
        pixel_area_m2 = pixel_width * pixel_height

        ref = RasterRef(
            profile=src.profile.copy(),
            transform=transform,
            crs=src.crs,
            width=src.width,
            height=src.height,
            nodata=src.nodata,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            pixel_area_m2=pixel_area_m2,
        )

    return arr, ref


def load_aligned_raster(path: Path, ref: RasterRef):
    with rasterio.open(path) as src:
        if src.width != ref.width or src.height != ref.height:
            raise ValueError(f"Grid size mismatch: {path.name}")
        if src.transform != ref.transform:
            raise ValueError(f"Transform mismatch: {path.name}")
        if src.crs != ref.crs:
            raise ValueError(f"CRS mismatch: {path.name}")

        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

    return arr


def write_geotiff(path: Path, arr: np.ndarray, ref: RasterRef):
    profile = ref.profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
    )

    out = np.where(np.isnan(arr), -9999.0, arr).astype(np.float32)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def plot_raster(arr, title, outpath, cmap="viridis", vmin=None, vmax=None):
    plt.figure(figsize=(9, 9))
    im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close()


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def valid_values(arr: np.ndarray):
    return arr[np.isfinite(arr)]


def classify_source(source: np.ndarray, valid_mask: np.ndarray):
    vals = source[valid_mask]

    p_low = np.nanpercentile(vals, LOW_PERCENTILE)
    p_high = np.nanpercentile(vals, HIGH_PERCENTILE)
    p_very_high = np.nanpercentile(vals, VERY_HIGH_PERCENTILE)

    low_mask = valid_mask & (source <= p_low)
    high_mask = valid_mask & (source >= p_high)
    very_high_mask = valid_mask & (source >= p_very_high)

    return {
        "p_low": p_low,
        "p_high": p_high,
        "p_very_high": p_very_high,
        "low_mask": low_mask,
        "high_mask": high_mask,
        "very_high_mask": very_high_mask,
    }


def make_structure(connectivity: int):
    if connectivity == 1:
        return np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]],
            dtype=np.uint8,
        )

    return np.ones((3, 3), dtype=np.uint8)


def patch_stats(
    patch_mask: np.ndarray,
    source: np.ndarray,
    ref: RasterRef,
    lulc: Dict[str, np.ndarray],
):
    structure = make_structure(CONNECTIVITY)
    labels, n_labels = label(patch_mask, structure=structure)

    rows = []

    objects = find_objects(labels)

    for patch_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        patch_pixels = labels[slc] == patch_id

        rows_idx, cols_idx = np.where(patch_pixels)

        # Convert local slice indices to global row/col
        global_rows = rows_idx + slc[0].start
        global_cols = cols_idx + slc[1].start

        n_pix = len(global_rows)
        area_m2 = n_pix * ref.pixel_area_m2
        area_ha = area_m2 / 10_000.0

        src_vals = source[global_rows, global_cols]

        xs, ys = rasterio.transform.xy(
            ref.transform,
            global_rows,
            global_cols,
            offset="center",
        )

        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))

        lulc_means = {}
        for name, arr in lulc.items():
            vals = arr[global_rows, global_cols]
            lulc_means[f"mean_{name}"] = float(np.nanmean(vals))

        dominant_lulc = max(
            lulc.keys(),
            key=lambda k: lulc_means.get(f"mean_{k}", np.nan),
        )

        rows.append({
            "patch_id": patch_id,
            "n_pixels": int(n_pix),
            "area_m2": float(area_m2),
            "area_ha": float(area_ha),
            "mean_source": float(np.nanmean(src_vals)),
            "max_source": float(np.nanmax(src_vals)),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "dominant_lulc": dominant_lulc,
            **lulc_means,
        })

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df.sort_values(
            ["area_ha", "mean_source"],
            ascending=False,
        ).reset_index(drop=True)

    return labels, df


def summarize_lulc_by_mask(lulc: Dict[str, np.ndarray], mask: np.ndarray, label_name: str):
    rows = []

    for name, arr in lulc.items():
        vals = arr[mask & np.isfinite(arr)]

        if vals.size == 0:
            mean_val = np.nan
            median_val = np.nan
        else:
            mean_val = float(np.nanmean(vals))
            median_val = float(np.nanmedian(vals))

        rows.append({
            "group": label_name,
            "landscape_variable": name,
            "mean_fraction": mean_val,
            "median_fraction": median_val,
        })

    return rows


def dominant_class_map(lulc: Dict[str, np.ndarray], valid_mask: np.ndarray):
    names = list(lulc.keys())
    stack = np.stack([
        np.nan_to_num(lulc[name], nan=-9999.0)
        for name in names
    ])

    idx = np.argmax(stack, axis=0).astype(np.float32)
    idx[~valid_mask] = np.nan

    lookup = {i: name for i, name in enumerate(names)}
    return idx, lookup


def city_sector_summary(source: np.ndarray, valid_mask: np.ndarray, ref: RasterRef):
    rows, cols = np.where(valid_mask)

    xs, ys = rasterio.transform.xy(
        ref.transform,
        rows,
        cols,
        offset="center",
    )

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    vals = source[rows, cols]

    x_mid = np.nanmedian(xs)
    y_mid = np.nanmedian(ys)

    sectors = np.full(vals.shape, "unknown", dtype=object)

    sectors[(xs < x_mid) & (ys >= y_mid)] = "northwest"
    sectors[(xs >= x_mid) & (ys >= y_mid)] = "northeast"
    sectors[(xs < x_mid) & (ys < y_mid)] = "southwest"
    sectors[(xs >= x_mid) & (ys < y_mid)] = "southeast"

    out = []

    for sector in ["northwest", "northeast", "southwest", "southeast"]:
        svals = vals[sectors == sector]
        out.append({
            "sector": sector,
            "n_pixels": int(svals.size),
            "mean_source": float(np.nanmean(svals)),
            "median_source": float(np.nanmedian(svals)),
            "p90_source": float(np.nanpercentile(svals, 90)),
            "p95_source": float(np.nanpercentile(svals, 95)),
        })

    return pd.DataFrame(out)


def coastal_summary(source, valid_mask, ocean, ref: RasterRef):
    if ocean is None:
        return None

    ocean_mask = np.nan_to_num(ocean, nan=0.0) > 0
    ocean_mask = ocean_mask & valid_mask

    if ocean_mask.sum() == 0:
        return None

    # Distance to ocean in pixels, then meters.
    # distance_transform_edt computes distance to nearest False, so invert ocean.
    distance_pix = distance_transform_edt(~ocean_mask)
    distance_m = distance_pix * ref.pixel_width

    coastal_mask = valid_mask & (distance_m <= COASTAL_DISTANCE_M)
    inland_mask = valid_mask & (distance_m > COASTAL_DISTANCE_M)

    rows = []

    for name, mask in [
        (f"coastal_within_{int(COASTAL_DISTANCE_M)}m", coastal_mask),
        (f"inland_beyond_{int(COASTAL_DISTANCE_M)}m", inland_mask),
    ]:
        vals = source[mask]
        rows.append({
            "zone": name,
            "n_pixels": int(vals.size),
            "mean_source": float(np.nanmean(vals)),
            "median_source": float(np.nanmedian(vals)),
            "p90_source": float(np.nanpercentile(vals, 90)),
            "p95_source": float(np.nanpercentile(vals, 95)),
        })

    return pd.DataFrame(rows)


def optional_zone_summary(source, ref: RasterRef, zones_path: Optional[Path], name_field: Optional[str]):
    if zones_path is None:
        return None

    if not zones_path.exists():
        print(f"[WARN] optional zone file does not exist: {zones_path}")
        return None

    gdf = gpd.read_file(zones_path)

    if gdf.crs != ref.crs:
        gdf = gdf.to_crs(ref.crs)

    rows = []

    for idx, row in gdf.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        mask = geometry_mask(
            [geom.__geo_interface__],
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            invert=True,
        )

        vals = source[mask & np.isfinite(source)]

        if vals.size == 0:
            continue

        zone_name = str(row[name_field]) if name_field and name_field in gdf.columns else str(idx)

        rows.append({
            "zone": zone_name,
            "n_pixels": int(vals.size),
            "mean_source": float(np.nanmean(vals)),
            "median_source": float(np.nanmedian(vals)),
            "p90_source": float(np.nanpercentile(vals, 90)),
            "p95_source": float(np.nanpercentile(vals, 95)),
        })

    if not rows:
        return None

    return pd.DataFrame(rows).sort_values("mean_source", ascending=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Missing source raster: {SOURCE_FILE}\n"
            "Update SOURCE_FILE at the top of the script."
        )

    source, ref = load_raster(SOURCE_FILE)

    valid_mask = np.isfinite(source)

    print(f"[INFO] valid source pixels: {valid_mask.sum():,}")
    print(f"[INFO] pixel size: {ref.pixel_width:.2f} x {ref.pixel_height:.2f} m")
    print(f"[INFO] pixel area: {ref.pixel_area_m2:.2f} m²")

    # -------------------------------------------------------------------------
    # LOAD LANDSCAPE LAYERS
    # -------------------------------------------------------------------------

    lulc = {}

    for name, path in LULC_FILES.items():
        if not path.exists():
            print(f"[WARN] missing LULC layer, skipping: {name}: {path}")
            continue

        lulc[name] = load_aligned_raster(path, ref)

    if not lulc:
        raise RuntimeError("No LULC rasters loaded. Check LULC_FILES.")

    # -------------------------------------------------------------------------
    # SOURCE CLASSES
    # -------------------------------------------------------------------------

    cls = classify_source(source, valid_mask)

    low_mask = cls["low_mask"]
    high_mask = cls["high_mask"]
    very_high_mask = cls["very_high_mask"]

    class_map = np.full_like(source, np.nan, dtype=np.float32)
    class_map[low_mask] = 1
    class_map[high_mask] = 2
    class_map[very_high_mask] = 3

    write_geotiff(
        OUT_DIR / "source_classes_low_high_veryhigh.tif",
        class_map,
        ref,
    )

    # -------------------------------------------------------------------------
    # PATCHES FROM VERY HIGH SOURCE
    # -------------------------------------------------------------------------

    patch_labels, patches = patch_stats(
        very_high_mask,
        source,
        ref,
        lulc,
    )

    write_geotiff(
        OUT_DIR / "very_high_source_patch_labels.tif",
        patch_labels.astype(np.float32),
        ref,
    )

    patches.to_csv(
        OUT_DIR / "very_high_source_patch_stats.csv",
        index=False,
    )

    top5 = patches.head(5).copy()

    top5.to_csv(
        OUT_DIR / "top5_source_patches.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # LANDSCAPE COMPOSITION
    # -------------------------------------------------------------------------

    rows = []
    rows.extend(summarize_lulc_by_mask(lulc, valid_mask, "all_valid_source_pixels"))
    rows.extend(summarize_lulc_by_mask(lulc, low_mask, f"low_source_bottom_{LOW_PERCENTILE}pct"))
    rows.extend(summarize_lulc_by_mask(lulc, high_mask, f"high_source_top_{100 - HIGH_PERCENTILE}pct"))
    rows.extend(summarize_lulc_by_mask(lulc, very_high_mask, f"very_high_source_top_{100 - VERY_HIGH_PERCENTILE}pct"))

    lulc_summary = pd.DataFrame(rows)

    lulc_summary.to_csv(
        OUT_DIR / "source_landscape_composition.csv",
        index=False,
    )

    # dominant land-cover category
    dominant_map, dominant_lookup = dominant_class_map(lulc, valid_mask)

    write_geotiff(
        OUT_DIR / "dominant_landscape_class.tif",
        dominant_map,
        ref,
    )

    pd.DataFrame([
        {"class_id": k, "class_name": v}
        for k, v in dominant_lookup.items()
    ]).to_csv(
        OUT_DIR / "dominant_landscape_class_lookup.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # CITY-WIDE DISTRIBUTION
    # -------------------------------------------------------------------------

    sector_df = city_sector_summary(source, valid_mask, ref)
    sector_df.to_csv(
        OUT_DIR / "source_by_city_sector.csv",
        index=False,
    )

    ocean = lulc.get("ocean")
    coastal_df = coastal_summary(source, valid_mask, ocean, ref)

    if coastal_df is not None:
        coastal_df.to_csv(
            OUT_DIR / "source_coastal_vs_inland.csv",
            index=False,
        )

    zone_df = optional_zone_summary(
        source,
        ref,
        OPTIONAL_ZONE_POLYGONS,
        OPTIONAL_ZONE_NAME_FIELD,
    )

    if zone_df is not None:
        zone_df.to_csv(
            OUT_DIR / "source_by_optional_zones.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # SUMMARY METRICS
    # -------------------------------------------------------------------------

    vals = source[valid_mask]

    patch_area_ha_total = float(patches["area_ha"].sum()) if len(patches) else 0.0
    largest_patch_ha = float(patches["area_ha"].max()) if len(patches) else 0.0
    n_patches = int(len(patches))

    summary = {
        "source_file": str(SOURCE_FILE),
        "valid_pixels": int(valid_mask.sum()),
        "valid_area_ha": float(valid_mask.sum() * ref.pixel_area_m2 / 10_000),
        "source_min": float(np.nanmin(vals)),
        "source_mean": float(np.nanmean(vals)),
        "source_median": float(np.nanmedian(vals)),
        "source_p10": float(np.nanpercentile(vals, 10)),
        "source_p90": float(np.nanpercentile(vals, 90)),
        "source_p95": float(np.nanpercentile(vals, 95)),
        "low_threshold_p10": float(cls["p_low"]),
        "high_threshold_p90": float(cls["p_high"]),
        "very_high_threshold_p95": float(cls["p_very_high"]),
        "very_high_patch_count": n_patches,
        "very_high_patch_total_area_ha": patch_area_ha_total,
        "largest_very_high_patch_ha": largest_patch_ha,
        "largest_patch_fraction_of_very_high_area": (
            largest_patch_ha / patch_area_ha_total if patch_area_ha_total > 0 else np.nan
        ),
    }

    pd.DataFrame([summary]).to_csv(
        OUT_DIR / "source_pattern_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # FIGURES
    # -------------------------------------------------------------------------

    plot_raster(
        source,
        "Source strength",
        OUT_DIR / "qc_source_strength.png",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plot_raster(
        class_map,
        "Source classes: low / high / very high",
        OUT_DIR / "qc_source_classes.png",
        cmap="viridis",
    )

    plot_raster(
        np.where(very_high_mask, source, np.nan),
        f"Very high source pixels: top {100 - VERY_HIGH_PERCENTILE}%",
        OUT_DIR / "qc_very_high_source_pixels.png",
        cmap="inferno",
        vmin=cls["p_very_high"],
        vmax=1,
    )

    plot_raster(
        patch_labels.astype(np.float32),
        f"Very high source patches: top {100 - VERY_HIGH_PERCENTILE}%",
        OUT_DIR / "qc_very_high_source_patch_labels.png",
        cmap="tab20",
    )

    # LULC barplot for high source composition
    fig_df = lulc_summary[
        lulc_summary["group"].isin([
            "all_valid_source_pixels",
            f"high_source_top_{100 - HIGH_PERCENTILE}pct",
            f"very_high_source_top_{100 - VERY_HIGH_PERCENTILE}pct",
        ])
    ].copy()

    pivot = fig_df.pivot(
        index="landscape_variable",
        columns="group",
        values="mean_fraction",
    )

    pivot.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Mean fraction")
    plt.title("Landscape composition of source classes")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_landscape_composition_barplot.png", dpi=250)
    plt.close()

    # Sector plot
    sector_df.plot(
        x="sector",
        y="mean_source",
        kind="bar",
        legend=False,
        figsize=(8, 5),
    )
    plt.ylabel("Mean source strength")
    plt.title("Mean source strength by city sector")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_source_by_city_sector.png", dpi=250)
    plt.close()

    print(f"[OK] diagnostics written to: {OUT_DIR}")
    print("[OK] key outputs:")
    print(f"     {OUT_DIR / 'source_pattern_summary.csv'}")
    print(f"     {OUT_DIR / 'source_landscape_composition.csv'}")
    print(f"     {OUT_DIR / 'very_high_source_patch_stats.csv'}")
    print(f"     {OUT_DIR / 'top5_source_patches.csv'}")


if __name__ == "__main__":
    main()



# -------------------------------------------------------------------------
# AUTOMATED INTERPRETATION SUMMARY
# -------------------------------------------------------------------------

def dominant_landscape_terms(lulc_summary: pd.DataFrame):
    """
    Extracts the dominant landscape variables for high and very-high source pixels.
    """
    out = {}

    for group in [
        f"high_source_top_{100 - HIGH_PERCENTILE}pct",
        f"very_high_source_top_{100 - VERY_HIGH_PERCENTILE}pct",
        f"low_source_bottom_{LOW_PERCENTILE}pct",
    ]:
        sub = lulc_summary[lulc_summary["group"] == group].copy()

        if sub.empty:
            out[group] = []
            continue

        sub = sub.sort_values("mean_fraction", ascending=False)
        out[group] = list(
            zip(
                sub["landscape_variable"].head(5),
                sub["mean_fraction"].head(5),
            )
        )

    return out


def format_landscape_list(items):
    if not items:
        return "no available landscape composition data"

    return ", ".join([
        f"{name} ({value:.2f})"
        for name, value in items
        if np.isfinite(value)
    ])


def summarize_optional_zones(zone_df: pd.DataFrame | None):
    if zone_df is None or zone_df.empty:
        return {
            "available": False,
            "top_mean": [],
            "top_p95": [],
        }

    top_mean = zone_df.sort_values("mean_source", ascending=False).head(5)
    top_p95 = zone_df.sort_values("p95_source", ascending=False).head(5)

    return {
        "available": True,
        "top_mean": list(zip(top_mean["zone"], top_mean["mean_source"])),
        "top_p95": list(zip(top_p95["zone"], top_p95["p95_source"])),
    }


def format_zone_list(items):
    if not items:
        return "no zone summary available"

    return ", ".join([
        f"{zone} ({value:.3f})"
        for zone, value in items
        if np.isfinite(value)
    ])


def interpret_patchiness(summary: dict):
    n_patches = summary["very_high_patch_count"]
    largest_fraction = summary["largest_patch_fraction_of_very_high_area"]

    if n_patches == 0:
        return "No very-high-source patches were detected."

    if largest_fraction >= 0.50:
        return (
            "The very-high-source pattern is strongly clustered, with the largest patch "
            f"containing {largest_fraction * 100:.1f}% of all very-high-source area."
        )

    if largest_fraction >= 0.25:
        return (
            "The very-high-source pattern is moderately clustered, with one large patch "
            f"containing {largest_fraction * 100:.1f}% of all very-high-source area, "
            "but with additional secondary patches also present."
        )

    return (
        "The very-high-source pattern is relatively patchy or dispersed, with the largest patch "
        f"containing only {largest_fraction * 100:.1f}% of all very-high-source area."
    )


def write_interpretation_report(
    out_path: Path,
    summary: dict,
    lulc_summary: pd.DataFrame,
    patches: pd.DataFrame,
    zone_df: pd.DataFrame | None,
    coastal_df: pd.DataFrame | None,
):
    landscape_terms = dominant_landscape_terms(lulc_summary)
    zone_summary = summarize_optional_zones(zone_df)

    high_group = f"high_source_top_{100 - HIGH_PERCENTILE}pct"
    very_high_group = f"very_high_source_top_{100 - VERY_HIGH_PERCENTILE}pct"
    low_group = f"low_source_bottom_{LOW_PERCENTILE}pct"

    patchiness_text = interpret_patchiness(summary)

    top5_text = "No source patches were detected."

    if patches is not None and not patches.empty:
        top5 = patches.head(5).copy()

        top5_lines = []
        for _, row in top5.iterrows():
            top5_lines.append(
                f"- Patch {int(row['patch_id'])}: "
                f"{row['area_ha']:.2f} ha, "
                f"mean source {row['mean_source']:.3f}, "
                f"max source {row['max_source']:.3f}, "
                f"dominant class: {row['dominant_lulc']}, "
                f"centroid: ({row['centroid_x']:.1f}, {row['centroid_y']:.1f})"
            )

        top5_text = "\n".join(top5_lines)

    coastal_text = "No coastal/inland comparison was available."

    if coastal_df is not None and not coastal_df.empty:
        coastal_lines = []
        for _, row in coastal_df.iterrows():
            coastal_lines.append(
                f"- {row['zone']}: mean source {row['mean_source']:.3f}, "
                f"median {row['median_source']:.3f}, "
                f"p95 {row['p95_source']:.3f}"
            )
        coastal_text = "\n".join(coastal_lines)

    report = f"""
SOURCE PATTERN INTERPRETATION
=============================

Input source raster
-------------------
{summary["source_file"]}

Overall source distribution
---------------------------
Valid source area: {summary["valid_area_ha"]:.2f} ha
Mean source value: {summary["source_mean"]:.3f}
Median source value: {summary["source_median"]:.3f}
90th percentile source threshold: {summary["source_p90"]:.3f}
95th percentile source threshold: {summary["source_p95"]:.3f}

Patch structure of very-high-source areas
-----------------------------------------
Very-high-source definition: top {100 - VERY_HIGH_PERCENTILE}% of source values.
Number of very-high-source patches: {summary["very_high_patch_count"]}
Total very-high-source patch area: {summary["very_high_patch_total_area_ha"]:.2f} ha
Largest very-high-source patch: {summary["largest_very_high_patch_ha"]:.2f} ha
Largest patch share of very-high-source area: {summary["largest_patch_fraction_of_very_high_area"] * 100:.1f}%

Interpretation:
{patchiness_text}

Landscape composition
---------------------
Dominant variables in high-source pixels, top {100 - HIGH_PERCENTILE}%:
{format_landscape_list(landscape_terms[high_group])}

Dominant variables in very-high-source pixels, top {100 - VERY_HIGH_PERCENTILE}%:
{format_landscape_list(landscape_terms[very_high_group])}

Dominant variables in low-source pixels, bottom {LOW_PERCENTILE}%:
{format_landscape_list(landscape_terms[low_group])}

Optional zone / district summary
--------------------------------
Top zones by mean source strength:
{format_zone_list(zone_summary["top_mean"])}

Top zones by p95 source strength:
{format_zone_list(zone_summary["top_p95"])}

Coastal versus inland summary
-----------------------------
{coastal_text}

Five most prominent very-high-source patches
--------------------------------------------
{top5_text}

Suggested paragraph skeleton
----------------------------
Candidate microrefugia were spatially concentrated rather than evenly distributed across the study area. 
The top {100 - VERY_HIGH_PERCENTILE}% of source values formed {summary["very_high_patch_count"]} discrete patches covering {summary["very_high_patch_total_area_ha"]:.2f} ha, with the largest patch accounting for {summary["largest_patch_fraction_of_very_high_area"] * 100:.1f}% of the very-high-source area. 
The strongest source areas were associated primarily with {format_landscape_list(landscape_terms[very_high_group])}. 
Low-source areas were associated primarily with {format_landscape_list(landscape_terms[low_group])}. 
At the district/peruspiiri scale, high refuge potential was most evident in {format_zone_list(zone_summary["top_p95"])}. 
Together, these results indicate that persistent thermal refuge potential is concentrated in a limited set of landscape structures rather than being evenly distributed across the urban matrix.
""".strip()

    out_path.write_text(report, encoding="utf-8")

    print(f"[OK] wrote interpretation report: {out_path}")


write_interpretation_report(
        out_path=OUT_DIR / "source_pattern_interpretation.txt",
        summary=summary,
        lulc_summary=lulc_summary,
        patches=patches,
        zone_df=zone_df,
        coastal_df=coastal_df,
    )



#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import geometry_mask
from scipy.ndimage import label, find_objects, distance_transform_edt


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
OMNI = BASE / "omniscape"
STACK = BASE / "predictorstack"

SOURCE_FILE = OMNI / "sources" / "source_p90_coolness_stability.tif"

OUT_DIR = OMNI / "diagnostics" / "source_pattern_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LULC_FILES = {
    "tree": STACK / "TREE_FRAC_10m.tif",
    "nwn": STACK / "NWN_FRAC_10m.tif",
    "impervious": STACK / "IMPERV_FRAC_10m_Helsinki.tif",
    "building": STACK / "BLDG_FRAC_10m.tif",
    "water": STACK / "WATER_FRAC_10m_Helsinki.tif",
    "ocean": STACK / "OCEAN_FRAC_10m_Helsinki.tif",
    "rock": STACK / "ROCK_FRAC_10m_Helsinki.tif",
}

# Optional zone / district polygons.
OPTIONAL_ZONE_POLYGONS: Optional[Path] = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\figures\offset_figure\peruspiiri_WFS.gpkg"
)
OPTIONAL_ZONE_NAME_FIELD = "nimi_fi"


# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

HIGH_PERCENTILE = 90
VERY_HIGH_PERCENTILE = 95
LOW_PERCENTILE = 10

PATCH_PERCENTILE = VERY_HIGH_PERCENTILE
CONNECTIVITY = 2

COASTAL_DISTANCE_M = 300.0


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
    pixel_width: float
    pixel_height: float
    pixel_area_m2: float


# =============================================================================
# IO HELPERS
# =============================================================================

def load_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

        transform = src.transform
        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)
        pixel_area_m2 = pixel_width * pixel_height

        ref = RasterRef(
            profile=src.profile.copy(),
            transform=transform,
            crs=src.crs,
            width=src.width,
            height=src.height,
            nodata=src.nodata,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            pixel_area_m2=pixel_area_m2,
        )

    return arr, ref


def load_aligned_raster(path: Path, ref: RasterRef):
    with rasterio.open(path) as src:
        if src.width != ref.width or src.height != ref.height:
            raise ValueError(f"Grid size mismatch: {path.name}")
        if src.transform != ref.transform:
            raise ValueError(f"Transform mismatch: {path.name}")
        if src.crs != ref.crs:
            raise ValueError(f"CRS mismatch: {path.name}")

        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

    return arr


def write_geotiff(path: Path, arr: np.ndarray, ref: RasterRef):
    profile = ref.profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
    )

    out = np.where(np.isnan(arr), -9999.0, arr).astype(np.float32)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def plot_raster(arr, title, outpath, cmap="viridis", vmin=None, vmax=None):
    plt.figure(figsize=(9, 9))
    im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close()


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def classify_source(source: np.ndarray, valid_mask: np.ndarray):
    vals = source[valid_mask]

    p_low = np.nanpercentile(vals, LOW_PERCENTILE)
    p_high = np.nanpercentile(vals, HIGH_PERCENTILE)
    p_very_high = np.nanpercentile(vals, VERY_HIGH_PERCENTILE)

    low_mask = valid_mask & (source <= p_low)
    high_mask = valid_mask & (source >= p_high)
    very_high_mask = valid_mask & (source >= p_very_high)

    return {
        "p_low": p_low,
        "p_high": p_high,
        "p_very_high": p_very_high,
        "low_mask": low_mask,
        "high_mask": high_mask,
        "very_high_mask": very_high_mask,
    }


def make_structure(connectivity: int):
    if connectivity == 1:
        return np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]],
            dtype=np.uint8,
        )

    return np.ones((3, 3), dtype=np.uint8)


def patch_stats(
    patch_mask: np.ndarray,
    source: np.ndarray,
    ref: RasterRef,
    lulc: Dict[str, np.ndarray],
):
    structure = make_structure(CONNECTIVITY)
    labels, n_labels = label(patch_mask, structure=structure)

    rows = []
    objects = find_objects(labels)

    for patch_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        patch_pixels = labels[slc] == patch_id

        rows_idx, cols_idx = np.where(patch_pixels)

        global_rows = rows_idx + slc[0].start
        global_cols = cols_idx + slc[1].start

        n_pix = len(global_rows)
        area_m2 = n_pix * ref.pixel_area_m2
        area_ha = area_m2 / 10_000.0

        src_vals = source[global_rows, global_cols]

        xs, ys = rasterio.transform.xy(
            ref.transform,
            global_rows,
            global_cols,
            offset="center",
        )

        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))

        lulc_means = {}
        for name, arr in lulc.items():
            vals = arr[global_rows, global_cols]
            lulc_means[f"mean_{name}"] = float(np.nanmean(vals))

        dominant_lulc = max(
            lulc.keys(),
            key=lambda k: lulc_means.get(f"mean_{k}", np.nan),
        )

        rows.append({
            "patch_id": patch_id,
            "n_pixels": int(n_pix),
            "area_m2": float(area_m2),
            "area_ha": float(area_ha),
            "mean_source": float(np.nanmean(src_vals)),
            "max_source": float(np.nanmax(src_vals)),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "dominant_lulc": dominant_lulc,
            **lulc_means,
        })

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df.sort_values(
            ["area_ha", "mean_source"],
            ascending=False,
        ).reset_index(drop=True)

    return labels, df


def summarize_lulc_by_mask(lulc: Dict[str, np.ndarray], mask: np.ndarray, label_name: str):
    rows = []

    for name, arr in lulc.items():
        vals = arr[mask & np.isfinite(arr)]

        if vals.size == 0:
            mean_val = np.nan
            median_val = np.nan
        else:
            mean_val = float(np.nanmean(vals))
            median_val = float(np.nanmedian(vals))

        rows.append({
            "group": label_name,
            "landscape_variable": name,
            "mean_fraction": mean_val,
            "median_fraction": median_val,
        })

    return rows


def dominant_class_map(lulc: Dict[str, np.ndarray], valid_mask: np.ndarray):
    names = list(lulc.keys())

    stack = np.stack([
        np.nan_to_num(lulc[name], nan=-9999.0)
        for name in names
    ])

    idx = np.argmax(stack, axis=0).astype(np.float32)
    idx[~valid_mask] = np.nan

    lookup = {i: name for i, name in enumerate(names)}
    return idx, lookup


def city_sector_summary(source: np.ndarray, valid_mask: np.ndarray, ref: RasterRef):
    rows, cols = np.where(valid_mask)

    xs, ys = rasterio.transform.xy(
        ref.transform,
        rows,
        cols,
        offset="center",
    )

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    vals = source[rows, cols]

    x_mid = np.nanmedian(xs)
    y_mid = np.nanmedian(ys)

    sectors = np.full(vals.shape, "unknown", dtype=object)

    sectors[(xs < x_mid) & (ys >= y_mid)] = "northwest"
    sectors[(xs >= x_mid) & (ys >= y_mid)] = "northeast"
    sectors[(xs < x_mid) & (ys < y_mid)] = "southwest"
    sectors[(xs >= x_mid) & (ys < y_mid)] = "southeast"

    out = []

    for sector in ["northwest", "northeast", "southwest", "southeast"]:
        svals = vals[sectors == sector]

        out.append({
            "sector": sector,
            "n_pixels": int(svals.size),
            "mean_source": float(np.nanmean(svals)),
            "median_source": float(np.nanmedian(svals)),
            "p90_source": float(np.nanpercentile(svals, 90)),
            "p95_source": float(np.nanpercentile(svals, 95)),
        })

    return pd.DataFrame(out)


def coastal_summary(source, valid_mask, ocean, ref: RasterRef):
    if ocean is None:
        return None

    ocean_mask = np.nan_to_num(ocean, nan=0.0) > 0

    if ocean_mask.sum() == 0:
        return None

    distance_pix = distance_transform_edt(~ocean_mask)
    distance_m = distance_pix * ref.pixel_width

    coastal_mask = valid_mask & (distance_m <= COASTAL_DISTANCE_M)
    inland_mask = valid_mask & (distance_m > COASTAL_DISTANCE_M)

    rows = []

    for name, mask in [
        (f"coastal_within_{int(COASTAL_DISTANCE_M)}m", coastal_mask),
        (f"inland_beyond_{int(COASTAL_DISTANCE_M)}m", inland_mask),
    ]:
        vals = source[mask]

        if vals.size == 0:
            continue

        rows.append({
            "zone": name,
            "n_pixels": int(vals.size),
            "mean_source": float(np.nanmean(vals)),
            "median_source": float(np.nanmedian(vals)),
            "p90_source": float(np.nanpercentile(vals, 90)),
            "p95_source": float(np.nanpercentile(vals, 95)),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


def optional_zone_summary(
    source,
    ref: RasterRef,
    zones_path: Optional[Path],
    name_field: Optional[str],
):
    if zones_path is None:
        return None

    zones_path = Path(zones_path)

    if not zones_path.exists():
        print(f"[WARN] optional zone file does not exist: {zones_path}")
        return None

    gdf = gpd.read_file(zones_path)

    if gdf.crs != ref.crs:
        gdf = gdf.to_crs(ref.crs)

    rows = []

    for idx, row in gdf.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        mask = geometry_mask(
            [geom.__geo_interface__],
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            invert=True,
        )

        vals = source[mask & np.isfinite(source)]

        if vals.size == 0:
            continue

        zone_name = (
            str(row[name_field])
            if name_field and name_field in gdf.columns
            else str(idx)
        )

        rows.append({
            "zone": zone_name,
            "n_pixels": int(vals.size),
            "mean_source": float(np.nanmean(vals)),
            "median_source": float(np.nanmedian(vals)),
            "p90_source": float(np.nanpercentile(vals, 90)),
            "p95_source": float(np.nanpercentile(vals, 95)),
        })

    if not rows:
        return None

    return pd.DataFrame(rows).sort_values("mean_source", ascending=False)


# =============================================================================
# AUTOMATED INTERPRETATION HELPERS
# =============================================================================

def dominant_landscape_terms(lulc_summary: pd.DataFrame):
    out = {}

    for group in [
        f"high_source_top_{100 - HIGH_PERCENTILE}pct",
        f"very_high_source_top_{100 - VERY_HIGH_PERCENTILE}pct",
        f"low_source_bottom_{LOW_PERCENTILE}pct",
    ]:
        sub = lulc_summary[lulc_summary["group"] == group].copy()

        if sub.empty:
            out[group] = []
            continue

        sub = sub.sort_values("mean_fraction", ascending=False)

        out[group] = list(
            zip(
                sub["landscape_variable"].head(5),
                sub["mean_fraction"].head(5),
            )
        )

    return out


def format_landscape_list(items):
    if not items:
        return "no available landscape composition data"

    filtered = [
        f"{name} ({value:.2f})"
        for name, value in items
        if np.isfinite(value)
    ]

    if not filtered:
        return "no available landscape composition data"

    return ", ".join(filtered)


def summarize_optional_zones(zone_df: pd.DataFrame | None):
    if zone_df is None or zone_df.empty:
        return {
            "available": False,
            "top_mean": [],
            "top_p95": [],
        }

    top_mean = zone_df.sort_values("mean_source", ascending=False).head(5)
    top_p95 = zone_df.sort_values("p95_source", ascending=False).head(5)

    return {
        "available": True,
        "top_mean": list(zip(top_mean["zone"], top_mean["mean_source"])),
        "top_p95": list(zip(top_p95["zone"], top_p95["p95_source"])),
    }


def format_zone_list(items):
    if not items:
        return "no zone summary available"

    filtered = [
        f"{zone} ({value:.3f})"
        for zone, value in items
        if np.isfinite(value)
    ]

    if not filtered:
        return "no zone summary available"

    return ", ".join(filtered)


def interpret_patchiness(summary: dict):
    n_patches = summary["very_high_patch_count"]
    largest_fraction = summary["largest_patch_fraction_of_very_high_area"]

    if n_patches == 0:
        return "No very-high-source patches were detected."

    if not np.isfinite(largest_fraction):
        return "Very-high-source patches were detected, but the largest-patch fraction could not be calculated."

    if largest_fraction >= 0.50:
        return (
            "The very-high-source pattern is strongly clustered, with the largest patch "
            f"containing {largest_fraction * 100:.1f}% of all very-high-source area."
        )

    if largest_fraction >= 0.25:
        return (
            "The very-high-source pattern is moderately clustered, with one large patch "
            f"containing {largest_fraction * 100:.1f}% of all very-high-source area, "
            "but with additional secondary patches also present."
        )

    return (
        "The very-high-source pattern is relatively patchy or dispersed, with the largest patch "
        f"containing only {largest_fraction * 100:.1f}% of all very-high-source area."
    )


def write_interpretation_report(
    out_path: Path,
    summary: dict,
    lulc_summary: pd.DataFrame,
    patches: pd.DataFrame,
    zone_df: pd.DataFrame | None,
    coastal_df: pd.DataFrame | None,
):
    landscape_terms = dominant_landscape_terms(lulc_summary)
    zone_summary = summarize_optional_zones(zone_df)

    high_group = f"high_source_top_{100 - HIGH_PERCENTILE}pct"
    very_high_group = f"very_high_source_top_{100 - VERY_HIGH_PERCENTILE}pct"
    low_group = f"low_source_bottom_{LOW_PERCENTILE}pct"

    patchiness_text = interpret_patchiness(summary)

    top5_text = "No source patches were detected."

    if patches is not None and not patches.empty:
        top5 = patches.head(5).copy()

        top5_lines = []

        for _, row in top5.iterrows():
            top5_lines.append(
                f"- Patch {int(row['patch_id'])}: "
                f"{row['area_ha']:.2f} ha, "
                f"mean source {row['mean_source']:.3f}, "
                f"max source {row['max_source']:.3f}, "
                f"dominant class: {row['dominant_lulc']}, "
                f"centroid: ({row['centroid_x']:.1f}, {row['centroid_y']:.1f})"
            )

        top5_text = "\n".join(top5_lines)

    coastal_text = "No coastal/inland comparison was available."

    if coastal_df is not None and not coastal_df.empty:
        coastal_lines = []

        for _, row in coastal_df.iterrows():
            coastal_lines.append(
                f"- {row['zone']}: mean source {row['mean_source']:.3f}, "
                f"median {row['median_source']:.3f}, "
                f"p95 {row['p95_source']:.3f}"
            )

        coastal_text = "\n".join(coastal_lines)

    report = f"""
SOURCE PATTERN INTERPRETATION
=============================

Input source raster
-------------------
{summary["source_file"]}

Overall source distribution
---------------------------
Valid source area: {summary["valid_area_ha"]:.2f} ha
Mean source value: {summary["source_mean"]:.3f}
Median source value: {summary["source_median"]:.3f}
90th percentile source threshold: {summary["source_p90"]:.3f}
95th percentile source threshold: {summary["source_p95"]:.3f}

Patch structure of very-high-source areas
-----------------------------------------
Very-high-source definition: top {100 - VERY_HIGH_PERCENTILE}% of source values.
Number of very-high-source patches: {summary["very_high_patch_count"]}
Total very-high-source patch area: {summary["very_high_patch_total_area_ha"]:.2f} ha
Largest very-high-source patch: {summary["largest_very_high_patch_ha"]:.2f} ha
Largest patch share of very-high-source area: {summary["largest_patch_fraction_of_very_high_area"] * 100:.1f}%

Interpretation:
{patchiness_text}

Landscape composition
---------------------
Dominant variables in high-source pixels, top {100 - HIGH_PERCENTILE}%:
{format_landscape_list(landscape_terms[high_group])}

Dominant variables in very-high-source pixels, top {100 - VERY_HIGH_PERCENTILE}%:
{format_landscape_list(landscape_terms[very_high_group])}

Dominant variables in low-source pixels, bottom {LOW_PERCENTILE}%:
{format_landscape_list(landscape_terms[low_group])}

Optional zone / district summary
--------------------------------
Top zones by mean source strength:
{format_zone_list(zone_summary["top_mean"])}

Top zones by p95 source strength:
{format_zone_list(zone_summary["top_p95"])}

Coastal versus inland summary
-----------------------------
{coastal_text}

Five most prominent very-high-source patches
--------------------------------------------
{top5_text}

Suggested paragraph skeleton
----------------------------
Candidate microrefugia were spatially concentrated rather than evenly distributed across the study area. The top {100 - VERY_HIGH_PERCENTILE}% of source values formed {summary["very_high_patch_count"]} discrete patches covering {summary["very_high_patch_total_area_ha"]:.2f} ha, with the largest patch accounting for {summary["largest_patch_fraction_of_very_high_area"] * 100:.1f}% of the very-high-source area. The strongest source areas were associated primarily with {format_landscape_list(landscape_terms[very_high_group])}. Low-source areas were associated primarily with {format_landscape_list(landscape_terms[low_group])}. At the district/peruspiiri scale, high refuge potential was most evident in {format_zone_list(zone_summary["top_p95"])}. Together, these results indicate that persistent thermal refuge potential is concentrated in a limited set of landscape structures rather than being evenly distributed across the urban matrix.
""".strip()

    out_path.write_text(report, encoding="utf-8")
    print(f"[OK] wrote interpretation report: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Missing source raster: {SOURCE_FILE}\n"
            "Update SOURCE_FILE at the top of the script."
        )

    source, ref = load_raster(SOURCE_FILE)
    valid_mask = np.isfinite(source)

    print(f"[INFO] valid source pixels: {valid_mask.sum():,}")
    print(f"[INFO] pixel size: {ref.pixel_width:.2f} x {ref.pixel_height:.2f} m")
    print(f"[INFO] pixel area: {ref.pixel_area_m2:.2f} m²")

    # -------------------------------------------------------------------------
    # LOAD LANDSCAPE LAYERS
    # -------------------------------------------------------------------------

    lulc = {}

    for name, path in LULC_FILES.items():
        if not path.exists():
            print(f"[WARN] missing LULC layer, skipping: {name}: {path}")
            continue

        lulc[name] = load_aligned_raster(path, ref)

    if not lulc:
        raise RuntimeError("No LULC rasters loaded. Check LULC_FILES.")

    # -------------------------------------------------------------------------
    # SOURCE CLASSES
    # -------------------------------------------------------------------------

    cls = classify_source(source, valid_mask)

    low_mask = cls["low_mask"]
    high_mask = cls["high_mask"]
    very_high_mask = cls["very_high_mask"]

    class_map = np.full_like(source, np.nan, dtype=np.float32)
    class_map[low_mask] = 1
    class_map[high_mask] = 2
    class_map[very_high_mask] = 3

    write_geotiff(
        OUT_DIR / "source_classes_low_high_veryhigh.tif",
        class_map,
        ref,
    )

    # -------------------------------------------------------------------------
    # PATCHES FROM VERY HIGH SOURCE
    # -------------------------------------------------------------------------

    patch_labels, patches = patch_stats(
        very_high_mask,
        source,
        ref,
        lulc,
    )

    write_geotiff(
        OUT_DIR / "very_high_source_patch_labels.tif",
        patch_labels.astype(np.float32),
        ref,
    )

    patches.to_csv(
        OUT_DIR / "very_high_source_patch_stats.csv",
        index=False,
    )

    top5 = patches.head(5).copy()

    top5.to_csv(
        OUT_DIR / "top5_source_patches.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # LANDSCAPE COMPOSITION
    # -------------------------------------------------------------------------

    rows = []
    rows.extend(summarize_lulc_by_mask(lulc, valid_mask, "all_valid_source_pixels"))
    rows.extend(summarize_lulc_by_mask(lulc, low_mask, f"low_source_bottom_{LOW_PERCENTILE}pct"))
    rows.extend(summarize_lulc_by_mask(lulc, high_mask, f"high_source_top_{100 - HIGH_PERCENTILE}pct"))
    rows.extend(summarize_lulc_by_mask(lulc, very_high_mask, f"very_high_source_top_{100 - VERY_HIGH_PERCENTILE}pct"))

    lulc_summary = pd.DataFrame(rows)

    lulc_summary.to_csv(
        OUT_DIR / "source_landscape_composition.csv",
        index=False,
    )

    dominant_map, dominant_lookup = dominant_class_map(lulc, valid_mask)

    write_geotiff(
        OUT_DIR / "dominant_landscape_class.tif",
        dominant_map,
        ref,
    )

    pd.DataFrame([
        {"class_id": k, "class_name": v}
        for k, v in dominant_lookup.items()
    ]).to_csv(
        OUT_DIR / "dominant_landscape_class_lookup.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # CITY-WIDE DISTRIBUTION
    # -------------------------------------------------------------------------

    sector_df = city_sector_summary(source, valid_mask, ref)

    sector_df.to_csv(
        OUT_DIR / "source_by_city_sector.csv",
        index=False,
    )

    ocean = lulc.get("ocean")

    coastal_df = coastal_summary(
        source,
        valid_mask,
        ocean,
        ref,
    )

    if coastal_df is not None:
        coastal_df.to_csv(
            OUT_DIR / "source_coastal_vs_inland.csv",
            index=False,
        )

    zone_df = optional_zone_summary(
        source,
        ref,
        OPTIONAL_ZONE_POLYGONS,
        OPTIONAL_ZONE_NAME_FIELD,
    )

    if zone_df is not None:
        zone_df.to_csv(
            OUT_DIR / "source_by_optional_zones.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # SUMMARY METRICS
    # -------------------------------------------------------------------------

    vals = source[valid_mask]

    patch_area_ha_total = float(patches["area_ha"].sum()) if len(patches) else 0.0
    largest_patch_ha = float(patches["area_ha"].max()) if len(patches) else 0.0
    n_patches = int(len(patches))

    largest_patch_fraction = (
        largest_patch_ha / patch_area_ha_total
        if patch_area_ha_total > 0
        else np.nan
    )

    summary = {
        "source_file": str(SOURCE_FILE),
        "valid_pixels": int(valid_mask.sum()),
        "valid_area_ha": float(valid_mask.sum() * ref.pixel_area_m2 / 10_000),
        "source_min": float(np.nanmin(vals)),
        "source_mean": float(np.nanmean(vals)),
        "source_median": float(np.nanmedian(vals)),
        "source_p10": float(np.nanpercentile(vals, 10)),
        "source_p90": float(np.nanpercentile(vals, 90)),
        "source_p95": float(np.nanpercentile(vals, 95)),
        "low_threshold_p10": float(cls["p_low"]),
        "high_threshold_p90": float(cls["p_high"]),
        "very_high_threshold_p95": float(cls["p_very_high"]),
        "very_high_patch_count": n_patches,
        "very_high_patch_total_area_ha": patch_area_ha_total,
        "largest_very_high_patch_ha": largest_patch_ha,
        "largest_patch_fraction_of_very_high_area": largest_patch_fraction,
    }

    pd.DataFrame([summary]).to_csv(
        OUT_DIR / "source_pattern_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # AUTOMATED INTERPRETATION REPORT
    # -------------------------------------------------------------------------

    write_interpretation_report(
        out_path=OUT_DIR / "source_pattern_interpretation.txt",
        summary=summary,
        lulc_summary=lulc_summary,
        patches=patches,
        zone_df=zone_df,
        coastal_df=coastal_df,
    )

    # -------------------------------------------------------------------------
    # FIGURES
    # -------------------------------------------------------------------------

    plot_raster(
        source,
        "Source strength",
        OUT_DIR / "qc_source_strength.png",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plot_raster(
        class_map,
        "Source classes: low / high / very high",
        OUT_DIR / "qc_source_classes.png",
        cmap="viridis",
    )

    plot_raster(
        np.where(very_high_mask, source, np.nan),
        f"Very high source pixels: top {100 - VERY_HIGH_PERCENTILE}%",
        OUT_DIR / "qc_very_high_source_pixels.png",
        cmap="inferno",
        vmin=cls["p_very_high"],
        vmax=1,
    )

    plot_raster(
        patch_labels.astype(np.float32),
        f"Very high source patches: top {100 - VERY_HIGH_PERCENTILE}%",
        OUT_DIR / "qc_very_high_source_patch_labels.png",
        cmap="tab20",
    )

    fig_df = lulc_summary[
        lulc_summary["group"].isin([
            "all_valid_source_pixels",
            f"high_source_top_{100 - HIGH_PERCENTILE}pct",
            f"very_high_source_top_{100 - VERY_HIGH_PERCENTILE}pct",
        ])
    ].copy()

    pivot = fig_df.pivot(
        index="landscape_variable",
        columns="group",
        values="mean_fraction",
    )

    pivot.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Mean fraction")
    plt.title("Landscape composition of source classes")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_landscape_composition_barplot.png", dpi=250)
    plt.close()

    sector_df.plot(
        x="sector",
        y="mean_source",
        kind="bar",
        legend=False,
        figsize=(8, 5),
    )

    plt.ylabel("Mean source strength")
    plt.title("Mean source strength by city sector")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_source_by_city_sector.png", dpi=250)
    plt.close()

    if zone_df is not None and not zone_df.empty:
        zone_plot = zone_df.sort_values("p95_source", ascending=False).head(15)

        zone_plot.plot(
            x="zone",
            y="p95_source",
            kind="bar",
            legend=False,
            figsize=(12, 5),
        )

        plt.ylabel("p95 source strength")
        plt.title("Top zones by p95 source strength")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "qc_top_zones_by_p95_source.png", dpi=250)
        plt.close()

    print(f"[OK] diagnostics written to: {OUT_DIR}")
    print("[OK] key outputs:")
    print(f"     {OUT_DIR / 'source_pattern_summary.csv'}")
    print(f"     {OUT_DIR / 'source_landscape_composition.csv'}")
    print(f"     {OUT_DIR / 'very_high_source_patch_stats.csv'}")
    print(f"     {OUT_DIR / 'top5_source_patches.csv'}")
    print(f"     {OUT_DIR / 'source_pattern_interpretation.txt'}")


if __name__ == "__main__":
    main()