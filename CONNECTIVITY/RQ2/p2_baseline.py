#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import geometry_mask
from scipy.ndimage import label, find_objects


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
OMNI = BASE / "omniscape"
STACK = BASE / "predictorstack"

RUN_ROOT = OMNI / "output" / "conditional_multiruns"

SOURCE_FILE = OMNI / "sources" / "source_p90_coolness_stability.tif"

OUT_DIR = OMNI / "diagnostics" / "baseline_connectivity_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPTIONAL_ZONE_POLYGONS: Optional[Path] = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\figures\offset_figure\peruspiiri_WFS.gpkg"
)
OPTIONAL_ZONE_NAME_FIELD = "nimi_fi"

LULC_FILES = {
    "tree": STACK / "TREE_FRAC_10m.tif",
    "nwn": STACK / "NWN_FRAC_10m.tif",
    "impervious": STACK / "IMPERV_FRAC_10m_Helsinki.tif",
    "building": STACK / "BLDG_FRAC_10m.tif",
    "water": STACK / "WATER_FRAC_10m_Helsinki.tif",
    "ocean": STACK / "OCEAN_FRAC_10m_Helsinki.tif",
    "rock": STACK / "ROCK_FRAC_10m_Helsinki.tif",
}


# =============================================================================
# SETTINGS
# =============================================================================

CONDITION_PREFIX = "condition_average"

NORM_CURRENT_NAME = "normalized_cum_currmap.tif"
CUM_CURRENT_NAME = "cum_currmap.tif"
FLOW_NAME = "flow_potential.tif"

HIGH_CURRENT_PERCENTILE = 95
VERY_HIGH_CURRENT_PERCENTILE = 99

SOURCE_HIGH_PERCENTILE = 95

CONNECTIVITY = 2  # 1 = 4-neighbour, 2 = 8-neighbour


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


def log_scale(arr: np.ndarray) -> np.ndarray:
    arr = np.where(arr <= 0, np.nan, arr)
    arr = np.log1p(arr)

    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        return arr

    p1, p99 = np.percentile(vals, [1, 99])
    arr = np.clip(arr, p1, p99)

    if np.isclose(p1, p99):
        return np.zeros_like(arr)

    return (arr - p1) / (p99 - p1)


# =============================================================================
# RUN DISCOVERY
# =============================================================================

def parse_tolerance_from_name(name: str) -> float | None:
    m = re.search(r"pm([0-9]+p?[0-9]*)deg", name)
    if not m:
        return None

    return float(m.group(1).replace("p", "."))


def find_file(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def discover_average_runs(run_root: Path):
    rows = []

    for d in sorted(run_root.iterdir()):
        if not d.is_dir():
            continue

        if d.name.endswith("_metadata"):
            continue

        if not d.name.startswith(f"{CONDITION_PREFIX}__pm"):
            continue

        tolerance = parse_tolerance_from_name(d.name)

        if tolerance is None:
            continue

        norm_path = find_file(d, NORM_CURRENT_NAME)

        if norm_path is None:
            print(f"[WARN] no normalized current found, skipping: {d}")
            continue

        rows.append({
            "run_name": d.name,
            "run_dir": d,
            "tolerance": tolerance,
            "norm_current": norm_path,
            "cum_current": find_file(d, CUM_CURRENT_NAME),
            "flow_potential": find_file(d, FLOW_NAME),
        })

    if not rows:
        raise FileNotFoundError(f"No average-condition runs found in {run_root}")

    return sorted(rows, key=lambda r: r["tolerance"])


# =============================================================================
# PATCH / SUMMARY HELPERS
# =============================================================================

def make_structure(connectivity: int):
    if connectivity == 1:
        return np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]],
            dtype=np.uint8,
        )

    return np.ones((3, 3), dtype=np.uint8)


def patch_stats(mask: np.ndarray, value_arr: np.ndarray, ref: RasterRef, label_name: str):
    structure = make_structure(CONNECTIVITY)
    labels, n_labels = label(mask, structure=structure)

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

        vals = value_arr[global_rows, global_cols]

        xs, ys = rasterio.transform.xy(
            ref.transform,
            global_rows,
            global_cols,
            offset="center",
        )

        rows.append({
            "patch_type": label_name,
            "patch_id": patch_id,
            "n_pixels": int(n_pix),
            "area_m2": float(area_m2),
            "area_ha": float(area_ha),
            "mean_value": float(np.nanmean(vals)),
            "max_value": float(np.nanmax(vals)),
            "centroid_x": float(np.mean(xs)),
            "centroid_y": float(np.mean(ys)),
        })

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df.sort_values(
            ["area_ha", "mean_value"],
            ascending=False,
        ).reset_index(drop=True)

    return labels, df


def summarize_lulc_by_mask(
    lulc: Dict[str, np.ndarray],
    mask: np.ndarray,
    group_name: str,
    tolerance: float,
):
    rows = []

    for name, arr in lulc.items():
        vals = arr[mask & np.isfinite(arr)]

        rows.append({
            "tolerance": tolerance,
            "group": group_name,
            "landscape_variable": name,
            "mean_fraction": float(np.nanmean(vals)) if vals.size else np.nan,
            "median_fraction": float(np.nanmedian(vals)) if vals.size else np.nan,
        })

    return rows


def summarize_zones(
    value_arr: np.ndarray,
    ref: RasterRef,
    zones_path: Optional[Path],
    name_field: Optional[str],
    tolerance: float,
):
    if zones_path is None:
        return None

    zones_path = Path(zones_path)

    if not zones_path.exists():
        print(f"[WARN] optional zone file not found: {zones_path}")
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

        vals = value_arr[mask & np.isfinite(value_arr)]

        if vals.size == 0:
            continue

        zone_name = (
            str(row[name_field])
            if name_field and name_field in gdf.columns
            else str(idx)
        )

        rows.append({
            "tolerance": tolerance,
            "zone": zone_name,
            "n_pixels": int(vals.size),
            "mean_current": float(np.nanmean(vals)),
            "median_current": float(np.nanmedian(vals)),
            "p90_current": float(np.nanpercentile(vals, 90)),
            "p95_current": float(np.nanpercentile(vals, 95)),
            "p99_current": float(np.nanpercentile(vals, 99)),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


def current_concentration_metrics(arr: np.ndarray, valid_mask: np.ndarray):
    vals = arr[valid_mask & np.isfinite(arr) & (arr > 0)]

    if vals.size == 0:
        return {
            "sum_current": 0.0,
            "top5_share_of_current": np.nan,
            "top1_share_of_current": np.nan,
        }

    total = float(np.nansum(vals))

    p95 = np.nanpercentile(vals, 95)
    p99 = np.nanpercentile(vals, 99)

    top5_sum = float(np.nansum(vals[vals >= p95]))
    top1_sum = float(np.nansum(vals[vals >= p99]))

    return {
        "sum_current": total,
        "top5_share_of_current": top5_sum / total if total > 0 else np.nan,
        "top1_share_of_current": top1_sum / total if total > 0 else np.nan,
    }


# =============================================================================
# INTERPRETATION REPORT
# =============================================================================

def write_interpretation_report(
    summary_df: pd.DataFrame,
    landscape_df: pd.DataFrame,
    zone_df: pd.DataFrame | None,
    patch_df: pd.DataFrame,
    out_path: Path,
):
    lines = []

    lines.append("BASELINE CONNECTIVITY INTERPRETATION")
    lines.append("====================================")
    lines.append("")

    lines.append("Scope")
    lines.append("-----")
    lines.append(
        "This report summarizes static reference connectivity for the average-condition thermal surface only. "
        "The source surface is fixed; differences among runs reflect the conditional thermal-similarity tolerance."
    )
    lines.append("")

    lines.append("Connectivity sensitivity to tolerance")
    lines.append("-------------------------------------")

    for _, row in summary_df.sort_values("tolerance").iterrows():
        lines.append(
            f"±{row['tolerance']:.1f}°C: "
            f"current-bearing area = {row['current_bearing_area_ha']:.2f} ha "
            f"({row['current_bearing_area_change_ha']:+.2f} ha; "
            f"{row['current_bearing_area_change_pct']:+.4f}% relative to strictest tolerance); "
            f"high-current area = {row['high_current_area_ha']:.2f} ha; "
            f"high-current patches = {int(row['high_current_patch_count'])}; "
            f"largest high-current patch = {row['largest_high_current_patch_ha']:.2f} ha; "
            f"top 1% current share = {row['top1_share_of_current']:.3f}; "
            f"top 5% current share = {row['top5_share_of_current']:.3f}."
        )

    lines.append("")

    first = summary_df.sort_values("tolerance").iloc[0]
    last = summary_df.sort_values("tolerance").iloc[-1]

    area_change_pct = last["current_bearing_area_change_pct"]

    if abs(area_change_pct) < 0.1:
        area_interp = (
            "The spatial extent of current-bearing area was effectively saturated and changed negligibly "
            f"across tolerances ({area_change_pct:+.4f}% from the strictest to broadest tolerance). "
            "Therefore, current-bearing area should not be interpreted as the main indicator of baseline "
            "connectivity sensitivity in this setup."
        )
    elif area_change_pct > 0:
        area_interp = (
            f"The current-bearing network expanded by {area_change_pct:.2f}% from the strictest to broadest "
            "tolerance, indicating that thermal similarity constraints affected the spatial extent of baseline connectivity."
        )
    else:
        area_interp = (
            f"The current-bearing network contracted by {abs(area_change_pct):.2f}% from the strictest to broadest "
            "tolerance, indicating non-monotonic sensitivity in the current-bearing area metric."
        )

    lines.append(area_interp)
    lines.append("")

    lines.append("Current concentration and bottleneck structure")
    lines.append("----------------------------------------------")
    lines.append(
        "Because the current-bearing area may become saturated, current concentration metrics are more informative "
        "than simple presence/absence of current. The top 1% and top 5% shares of cumulative current indicate "
        "how strongly flow is concentrated into a small fraction of the landscape. Higher values indicate stronger "
        "bottlenecking or pinch-point structure."
    )
    lines.append("")

    max_conc = summary_df.sort_values("top1_share_of_current", ascending=False).iloc[0]

    lines.append(
        f"The strongest current concentration occurred at ±{max_conc['tolerance']:.1f}°C, "
        f"where the top 1% of current-bearing pixels carried {max_conc['top1_share_of_current']:.3f} "
        "of total cumulative current."
    )
    lines.append("")

    lines.append("Patch structure")
    lines.append("---------------")
    lines.append(
        "High-current patch count and largest-patch area describe how the strongest current zones are arranged. "
        "Increasing patch counts or decreasing largest-patch area can indicate fragmentation of the highest-current "
        "corridor structure, even when total current-bearing area remains nearly unchanged."
    )
    lines.append("")

    first_patch = first["largest_high_current_patch_ha"]
    last_patch = last["largest_high_current_patch_ha"]

    lines.append(
        f"The largest high-current patch changed from {first_patch:.2f} ha at ±{first['tolerance']:.1f}°C "
        f"to {last_patch:.2f} ha at ±{last['tolerance']:.1f}°C."
    )
    lines.append("")

    lines.append("Landscape composition of high-current areas")
    lines.append("-------------------------------------------")

    if not landscape_df.empty:
        broad_tol = summary_df["tolerance"].max()

        sub = landscape_df[
            (landscape_df["tolerance"] == broad_tol) &
            (landscape_df["group"] == "high_current_top_5pct")
        ].sort_values("mean_fraction", ascending=False)

        if not sub.empty:
            desc = ", ".join([
                f"{r['landscape_variable']} ({r['mean_fraction']:.2f})"
                for _, r in sub.head(5).iterrows()
            ])

            lines.append(
                f"At the broadest tolerance tested, high-current areas were most associated with: {desc}."
            )
        else:
            lines.append("No landscape-composition summary was available for high-current areas.")
    else:
        lines.append("No landscape-composition summary was available.")

    lines.append("")

    lines.append("Top zones by current concentration")
    lines.append("----------------------------------")

    if zone_df is not None and not zone_df.empty:
        broad_tol = summary_df["tolerance"].max()

        z = zone_df[zone_df["tolerance"] == broad_tol].sort_values(
            "p95_current",
            ascending=False,
        ).head(10)

        for _, row in z.iterrows():
            lines.append(
                f"- {row['zone']}: p95 current = {row['p95_current']:.3f}, "
                f"mean current = {row['mean_current']:.3f}"
            )
    else:
        lines.append("No zone summary was available.")

    lines.append("")

    lines.append("Top high-current patches")
    lines.append("------------------------")

    if not patch_df.empty:
        top = patch_df.sort_values(
            ["tolerance", "area_ha"],
            ascending=[True, False],
        ).groupby("tolerance").head(3)

        for _, row in top.iterrows():
            lines.append(
                f"±{row['tolerance']:.1f}°C, patch {int(row['patch_id'])}: "
                f"{row['area_ha']:.2f} ha, mean current {row['mean_value']:.3f}, "
                f"centroid ({row['centroid_x']:.1f}, {row['centroid_y']:.1f})"
            )
    else:
        lines.append("No high-current patch summary was available.")

    lines.append("")
    lines.append("Suggested paragraph skeleton")
    lines.append("----------------------------")
    lines.append(
        "Under the average reference thermal condition, the spatial extent of the current-bearing network was "
        "largely saturated across the tested thermal similarity tolerances. Consequently, total current-bearing "
        "area was less informative than the distribution and concentration of current. High-current pixels were "
        "spatially concentrated into a subset of the landscape, indicating likely reference corridors and potential "
        "bottlenecks. Across tolerances, the most informative changes occurred in the patch structure of high-current "
        "areas and in the share of total current carried by the highest-current pixels. High-current areas were "
        "primarily associated with vegetated landscape structures, suggesting that baseline microrefugia connectivity "
        "is organized around green-network elements rather than the built matrix."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote interpretation report: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    runs = discover_average_runs(RUN_ROOT)

    source, source_ref = load_raster(SOURCE_FILE)
    source_valid = np.isfinite(source)
    source_vals = source[source_valid]
    source_thr = np.nanpercentile(source_vals, SOURCE_HIGH_PERCENTILE)
    high_source_mask = source_valid & (source >= source_thr)

    lulc = {}

    for name, path in LULC_FILES.items():
        if not path.exists():
            print(f"[WARN] missing LULC layer, skipping: {name}: {path}")
            continue

        lulc[name] = load_aligned_raster(path, source_ref)

    all_summary_rows = []
    all_patch_rows = []
    all_lulc_rows = []
    all_zone_rows = []

    for run in runs:
        tolerance = run["tolerance"]
        run_name = run["run_name"]

        print(f"[INFO] processing {run_name}")

        run_out = OUT_DIR / run_name
        run_out.mkdir(parents=True, exist_ok=True)

        norm_current, ref = load_raster(run["norm_current"])

        valid = np.isfinite(norm_current)
        current_bearing = valid & (norm_current > 0)

        vals = norm_current[current_bearing]

        if vals.size == 0:
            print(f"[WARN] no current-bearing pixels in {run_name}")
            continue

        high_thr = np.nanpercentile(vals, HIGH_CURRENT_PERCENTILE)
        very_high_thr = np.nanpercentile(vals, VERY_HIGH_CURRENT_PERCENTILE)

        high_current_mask = current_bearing & (norm_current >= high_thr)
        very_high_current_mask = current_bearing & (norm_current >= very_high_thr)

        high_labels, high_patch_df = patch_stats(
            high_current_mask,
            norm_current,
            ref,
            "high_current_top_5pct",
        )

        very_high_labels, very_high_patch_df = patch_stats(
            very_high_current_mask,
            norm_current,
            ref,
            "very_high_current_top_1pct",
        )

        high_patch_df["tolerance"] = tolerance
        high_patch_df["run_name"] = run_name

        very_high_patch_df["tolerance"] = tolerance
        very_high_patch_df["run_name"] = run_name

        if not high_patch_df.empty:
            all_patch_rows.append(high_patch_df)

        write_geotiff(
            run_out / "high_current_top5_patch_labels.tif",
            high_labels.astype(np.float32),
            ref,
        )

        write_geotiff(
            run_out / "very_high_current_top1_patch_labels.tif",
            very_high_labels.astype(np.float32),
            ref,
        )

        high_patch_df.to_csv(
            run_out / "high_current_top5_patch_stats.csv",
            index=False,
        )

        very_high_patch_df.to_csv(
            run_out / "very_high_current_top1_patch_stats.csv",
            index=False,
        )

        overlap_high_source_high_current = (
            np.sum(high_source_mask & high_current_mask) / np.sum(high_source_mask)
            if np.sum(high_source_mask) > 0 else np.nan
        )

        overlap_high_current_high_source = (
            np.sum(high_source_mask & high_current_mask) / np.sum(high_current_mask)
            if np.sum(high_current_mask) > 0 else np.nan
        )

        current_metrics = current_concentration_metrics(norm_current, current_bearing)

        largest_high_patch_ha = (
            float(high_patch_df["area_ha"].max())
            if not high_patch_df.empty else 0.0
        )

        high_patch_total_area_ha = (
            float(high_patch_df["area_ha"].sum())
            if not high_patch_df.empty else 0.0
        )

        largest_high_patch_share = (
            largest_high_patch_ha / high_patch_total_area_ha
            if high_patch_total_area_ha > 0 else np.nan
        )

        summary_row = {
            "run_name": run_name,
            "tolerance": tolerance,
            "norm_current_file": str(run["norm_current"]),
            "current_bearing_pixels": int(np.sum(current_bearing)),
            "current_bearing_area_ha": float(np.sum(current_bearing) * ref.pixel_area_m2 / 10_000),
            "mean_norm_current": float(np.nanmean(vals)),
            "median_norm_current": float(np.nanmedian(vals)),
            "p90_norm_current": float(np.nanpercentile(vals, 90)),
            "p95_norm_current": float(np.nanpercentile(vals, 95)),
            "p99_norm_current": float(np.nanpercentile(vals, 99)),
            "high_current_threshold_p95": float(high_thr),
            "very_high_current_threshold_p99": float(very_high_thr),
            "high_current_area_ha": float(np.sum(high_current_mask) * ref.pixel_area_m2 / 10_000),
            "very_high_current_area_ha": float(np.sum(very_high_current_mask) * ref.pixel_area_m2 / 10_000),
            "high_current_patch_count": int(len(high_patch_df)),
            "very_high_current_patch_count": int(len(very_high_patch_df)),
            "largest_high_current_patch_ha": largest_high_patch_ha,
            "largest_high_current_patch_share": largest_high_patch_share,
            "high_source_overlap_fraction": float(overlap_high_source_high_current),
            "high_current_overlap_with_high_source_fraction": float(overlap_high_current_high_source),
            **current_metrics,
        }

        if run["flow_potential"] is not None:
            flow = load_raster(run["flow_potential"])[0]
            flow_vals = flow[np.isfinite(flow) & (flow > 0)]

            summary_row.update({
                "flow_potential_mean": float(np.nanmean(flow_vals)) if flow_vals.size else np.nan,
                "flow_potential_median": float(np.nanmedian(flow_vals)) if flow_vals.size else np.nan,
                "flow_potential_p95": float(np.nanpercentile(flow_vals, 95)) if flow_vals.size else np.nan,
            })
        else:
            summary_row.update({
                "flow_potential_mean": np.nan,
                "flow_potential_median": np.nan,
                "flow_potential_p95": np.nan,
            })

        all_summary_rows.append(summary_row)

        if lulc:
            all_lulc_rows.extend(
                summarize_lulc_by_mask(
                    lulc,
                    current_bearing,
                    "all_current_bearing",
                    tolerance,
                )
            )
            all_lulc_rows.extend(
                summarize_lulc_by_mask(
                    lulc,
                    high_current_mask,
                    "high_current_top_5pct",
                    tolerance,
                )
            )
            all_lulc_rows.extend(
                summarize_lulc_by_mask(
                    lulc,
                    very_high_current_mask,
                    "very_high_current_top_1pct",
                    tolerance,
                )
            )

        zone_df = summarize_zones(
            norm_current,
            ref,
            OPTIONAL_ZONE_POLYGONS,
            OPTIONAL_ZONE_NAME_FIELD,
            tolerance,
        )

        if zone_df is not None:
            zone_df["run_name"] = run_name
            all_zone_rows.append(zone_df)

        plot_raster(
            log_scale(norm_current),
            f"Normalized current, ±{tolerance}°C",
            run_out / "qc_normalized_current_log.png",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        plot_raster(
            np.where(high_current_mask, norm_current, np.nan),
            f"High current: top {100 - HIGH_CURRENT_PERCENTILE}%, ±{tolerance}°C",
            run_out / "qc_high_current_top5.png",
            cmap="inferno",
        )

        plot_raster(
            np.where(very_high_current_mask, norm_current, np.nan),
            f"Very high current: top {100 - VERY_HIGH_CURRENT_PERCENTILE}%, ±{tolerance}°C",
            run_out / "qc_very_high_current_top1.png",
            cmap="inferno",
        )

        plot_raster(
            high_labels.astype(np.float32),
            f"High-current patches, ±{tolerance}°C",
            run_out / "qc_high_current_patch_labels.png",
            cmap="tab20",
        )

    # -------------------------------------------------------------------------
    # COMBINED OUTPUTS
    # -------------------------------------------------------------------------

    summary_df = pd.DataFrame(all_summary_rows).sort_values("tolerance")

    base_area = summary_df.iloc[0]["current_bearing_area_ha"]

    summary_df["current_bearing_area_change_ha"] = (
        summary_df["current_bearing_area_ha"] - base_area
    )

    summary_df["current_bearing_area_change_pct"] = (
        100 * summary_df["current_bearing_area_change_ha"] / base_area
    )

    summary_df.to_csv(
        OUT_DIR / "baseline_connectivity_summary.csv",
        index=False,
    )

    if all_patch_rows:
        patch_df = pd.concat(all_patch_rows, ignore_index=True)
    else:
        patch_df = pd.DataFrame()

    patch_df.to_csv(
        OUT_DIR / "top_corridor_patches_by_tolerance.csv",
        index=False,
    )

    if all_lulc_rows:
        landscape_df = pd.DataFrame(all_lulc_rows)
    else:
        landscape_df = pd.DataFrame()

    landscape_df.to_csv(
        OUT_DIR / "baseline_connectivity_landscape_composition.csv",
        index=False,
    )

    if all_zone_rows:
        zones_df = pd.concat(all_zone_rows, ignore_index=True)
        zones_df.to_csv(
            OUT_DIR / "baseline_connectivity_by_zone.csv",
            index=False,
        )
    else:
        zones_df = None

    # -------------------------------------------------------------------------
    # SENSITIVITY FIGURES
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary_df["tolerance"], summary_df["current_bearing_area_ha"], marker="o")
    ax.set_xlabel("Thermal similarity tolerance (± °C)")
    ax.set_ylabel("Current-bearing area (ha)")
    ax.set_title("Connectivity area sensitivity to tolerance")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_sensitivity_current_bearing_area.png", dpi=250)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary_df["tolerance"], summary_df["current_bearing_area_change_pct"], marker="o")
    ax.set_xlabel("Thermal similarity tolerance (± °C)")
    ax.set_ylabel("Change from strictest tolerance (%)")
    ax.set_title("Relative change in current-bearing area")
    ax.axhline(0, linewidth=1)
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_sensitivity_current_bearing_area_change_pct.png", dpi=250)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary_df["tolerance"], summary_df["high_current_patch_count"], marker="o")
    ax.set_xlabel("Thermal similarity tolerance (± °C)")
    ax.set_ylabel("High-current patch count")
    ax.set_title("High-current patch count sensitivity")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_sensitivity_high_current_patch_count.png", dpi=250)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary_df["tolerance"], summary_df["top1_share_of_current"], marker="o", label="Top 1%")
    ax.plot(summary_df["tolerance"], summary_df["top5_share_of_current"], marker="o", label="Top 5%")
    ax.set_xlabel("Thermal similarity tolerance (± °C)")
    ax.set_ylabel("Share of total cumulative current")
    ax.set_title("Current concentration sensitivity")
    ax.legend()
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_sensitivity_current_concentration.png", dpi=250)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary_df["tolerance"], summary_df["high_source_overlap_fraction"], marker="o")
    ax.set_xlabel("Thermal similarity tolerance (± °C)")
    ax.set_ylabel("Fraction of high-source pixels overlapping high current")
    ax.set_title("High-source / high-current overlap sensitivity")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_sensitivity_source_current_overlap.png", dpi=250)
    plt.close()

    # -------------------------------------------------------------------------
    # INTERPRETATION
    # -------------------------------------------------------------------------

    write_interpretation_report(
        summary_df=summary_df,
        landscape_df=landscape_df,
        zone_df=zones_df,
        patch_df=patch_df,
        out_path=OUT_DIR / "baseline_connectivity_interpretation.txt",
    )

    print(f"[OK] baseline connectivity diagnostics written to: {OUT_DIR}")
    print("[OK] key outputs:")
    print(f"     {OUT_DIR / 'baseline_connectivity_summary.csv'}")
    print(f"     {OUT_DIR / 'baseline_connectivity_landscape_composition.csv'}")
    print(f"     {OUT_DIR / 'baseline_connectivity_by_zone.csv'}")
    print(f"     {OUT_DIR / 'top_corridor_patches_by_tolerance.csv'}")
    print(f"     {OUT_DIR / 'baseline_connectivity_interpretation.txt'}")


if __name__ == "__main__":
    main()