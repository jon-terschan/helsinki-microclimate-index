#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

SOURCE_FILE = OMNI / "sources" / "source_p90_coolness_stability.tif"

RUN_ROOT = OMNI / "output" / "conditional_multiruns"

OUT_DIR = OMNI / "diagnostics" / "baseline_source_current_overlap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPTIONAL_ZONE_POLYGONS: Optional[Path] = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\figures\offset_figure\peruspiiri_WFS.gpkg"
)
OPTIONAL_ZONE_NAME_FIELD = "nimi_fi"


# =============================================================================
# SETTINGS
# =============================================================================

CONDITION_PREFIX = "condition_average"

NORM_CURRENT_NAME = "normalized_cum_currmap.tif"

HIGH_SOURCE_PERCENTILE = 95
HIGH_CURRENT_PERCENTILE = 95
HIGH_JOINT_PERCENTILE = 95

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
# IO
# =============================================================================

def load_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

        transform = src.transform
        ref = RasterRef(
            profile=src.profile.copy(),
            transform=transform,
            crs=src.crs,
            width=src.width,
            height=src.height,
            nodata=src.nodata,
            pixel_width=abs(transform.a),
            pixel_height=abs(transform.e),
            pixel_area_m2=abs(transform.a) * abs(transform.e),
        )

    return arr, ref


def check_alignment(ref_a: RasterRef, ref_b: RasterRef):
    if ref_a.width != ref_b.width or ref_a.height != ref_b.height:
        raise ValueError("Raster dimensions do not match.")
    if ref_a.transform != ref_b.transform:
        raise ValueError("Raster transforms do not match.")
    if ref_a.crs != ref_b.crs:
        raise ValueError("Raster CRS does not match.")


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


def log_scale01(arr: np.ndarray) -> np.ndarray:
    arr = np.where(arr <= 0, np.nan, arr)
    arr = np.log1p(arr)

    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return arr

    p1, p99 = np.nanpercentile(vals, [1, 99])
    arr = np.clip(arr, p1, p99)

    if np.isclose(p1, p99):
        return np.zeros_like(arr)

    return (arr - p1) / (p99 - p1)


def rescale01(arr: np.ndarray) -> np.ndarray:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return arr

    p1, p99 = np.nanpercentile(vals, [1, 99])
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

        current_path = find_file(d, NORM_CURRENT_NAME)

        if current_path is None:
            print(f"[WARN] missing normalized current, skipping: {d}")
            continue

        rows.append({
            "run_name": d.name,
            "run_dir": d,
            "tolerance": tolerance,
            "current_file": current_path,
        })

    if not rows:
        raise FileNotFoundError(f"No average-condition runs found in {run_root}")

    return sorted(rows, key=lambda r: r["tolerance"])


# =============================================================================
# PATCH / ZONE HELPERS
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


def patch_stats(mask: np.ndarray, value_arr: np.ndarray, source: np.ndarray, current: np.ndarray, ref: RasterRef):
    structure = make_structure(CONNECTIVITY)
    labels, n_labels = label(mask, structure=structure)

    rows = []
    objects = find_objects(labels)

    for patch_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        local_mask = labels[slc] == patch_id
        rr, cc = np.where(local_mask)

        global_rows = rr + slc[0].start
        global_cols = cc + slc[1].start

        n_pix = len(global_rows)
        area_ha = n_pix * ref.pixel_area_m2 / 10_000.0

        joint_vals = value_arr[global_rows, global_cols]
        source_vals = source[global_rows, global_cols]
        current_vals = current[global_rows, global_cols]

        xs, ys = rasterio.transform.xy(
            ref.transform,
            global_rows,
            global_cols,
            offset="center",
        )

        rows.append({
            "patch_id": patch_id,
            "n_pixels": int(n_pix),
            "area_ha": float(area_ha),
            "mean_joint": float(np.nanmean(joint_vals)),
            "max_joint": float(np.nanmax(joint_vals)),
            "mean_source": float(np.nanmean(source_vals)),
            "mean_current": float(np.nanmean(current_vals)),
            "max_source": float(np.nanmax(source_vals)),
            "max_current": float(np.nanmax(current_vals)),
            "centroid_x": float(np.mean(xs)),
            "centroid_y": float(np.mean(ys)),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["area_ha", "mean_joint"], ascending=False).reset_index(drop=True)

    return labels, df


def summarize_zones(
    joint: np.ndarray,
    source: np.ndarray,
    current: np.ndarray,
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

        vals = joint[mask & np.isfinite(joint)]

        if vals.size == 0:
            continue

        src_vals = source[mask & np.isfinite(source)]
        cur_vals = current[mask & np.isfinite(current)]

        zone_name = (
            str(row[name_field])
            if name_field and name_field in gdf.columns
            else str(idx)
        )

        rows.append({
            "tolerance": tolerance,
            "zone": zone_name,
            "n_pixels": int(vals.size),
            "mean_joint": float(np.nanmean(vals)),
            "p90_joint": float(np.nanpercentile(vals, 90)),
            "p95_joint": float(np.nanpercentile(vals, 95)),
            "p99_joint": float(np.nanpercentile(vals, 99)),
            "mean_source": float(np.nanmean(src_vals)) if src_vals.size else np.nan,
            "mean_current": float(np.nanmean(cur_vals)) if cur_vals.size else np.nan,
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Missing source raster: {SOURCE_FILE}")

    runs = discover_average_runs(RUN_ROOT)

    source, source_ref = load_raster(SOURCE_FILE)

    valid_source = np.isfinite(source)
    source_thr = np.nanpercentile(source[valid_source], HIGH_SOURCE_PERCENTILE)
    high_source_mask = valid_source & (source >= source_thr)

    all_summary = []
    all_patches = []
    all_zones = []

    for run in runs:
        tolerance = run["tolerance"]
        run_name = run["run_name"]

        print(f"[INFO] processing {run_name}")

        run_out = OUT_DIR / run_name
        run_out.mkdir(parents=True, exist_ok=True)

        current_raw, current_ref = load_raster(run["current_file"])
        check_alignment(source_ref, current_ref)

        # Use robustly scaled current so source and current are comparable in [0,1].
        current = log_scale01(current_raw)

        valid = np.isfinite(source) & np.isfinite(current)

        # Joint index: high only where both source quality and current importance are high.
        joint = np.sqrt(np.clip(source * current, 0.0, 1.0))
        joint = np.where(valid, joint, np.nan)

        current_vals = current[valid]
        joint_vals = joint[valid]

        current_thr = np.nanpercentile(current_vals, HIGH_CURRENT_PERCENTILE)
        joint_thr = np.nanpercentile(joint_vals, HIGH_JOINT_PERCENTILE)

        high_current_mask = valid & (current >= current_thr)
        high_joint_mask = valid & (joint >= joint_thr)

        overlap_high_source_current = (
            np.sum(high_source_mask & high_current_mask) / np.sum(high_source_mask)
            if np.sum(high_source_mask) else np.nan
        )

        overlap_high_current_source = (
            np.sum(high_source_mask & high_current_mask) / np.sum(high_current_mask)
            if np.sum(high_current_mask) else np.nan
        )

        labels, patches = patch_stats(
            high_joint_mask,
            joint,
            source,
            current,
            source_ref,
        )

        patches["tolerance"] = tolerance
        patches["run_name"] = run_name

        if not patches.empty:
            all_patches.append(patches)

        write_geotiff(run_out / "joint_source_current_index.tif", joint, source_ref)
        write_geotiff(run_out / "high_joint_source_current_top5.tif", np.where(high_joint_mask, joint, np.nan), source_ref)
        write_geotiff(run_out / "high_joint_patch_labels.tif", labels.astype(np.float32), source_ref)

        patches.to_csv(run_out / "high_joint_patch_stats.csv", index=False)

        zone_df = summarize_zones(
            joint=joint,
            source=source,
            current=current,
            ref=source_ref,
            zones_path=OPTIONAL_ZONE_POLYGONS,
            name_field=OPTIONAL_ZONE_NAME_FIELD,
            tolerance=tolerance,
        )

        if zone_df is not None:
            zone_df["run_name"] = run_name
            all_zones.append(zone_df)

        summary = {
            "run_name": run_name,
            "tolerance": tolerance,
            "current_file": str(run["current_file"]),
            "source_threshold_p95": float(source_thr),
            "current_threshold_p95": float(current_thr),
            "joint_threshold_p95": float(joint_thr),
            "high_source_area_ha": float(np.sum(high_source_mask) * source_ref.pixel_area_m2 / 10_000),
            "high_current_area_ha": float(np.sum(high_current_mask) * source_ref.pixel_area_m2 / 10_000),
            "high_joint_area_ha": float(np.sum(high_joint_mask) * source_ref.pixel_area_m2 / 10_000),
            "high_joint_patch_count": int(len(patches)),
            "largest_high_joint_patch_ha": float(patches["area_ha"].max()) if not patches.empty else 0.0,
            "mean_joint": float(np.nanmean(joint_vals)),
            "median_joint": float(np.nanmedian(joint_vals)),
            "p95_joint": float(np.nanpercentile(joint_vals, 95)),
            "p99_joint": float(np.nanpercentile(joint_vals, 99)),
            "fraction_high_source_overlapping_high_current": float(overlap_high_source_current),
            "fraction_high_current_overlapping_high_source": float(overlap_high_current_source),
        }

        all_summary.append(summary)

        # ---------------------------------------------------------------------
        # QC FIGURES
        # ---------------------------------------------------------------------

        plot_raster(
            source,
            "Source strength",
            run_out / "qc_source_strength.png",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        plot_raster(
            current,
            f"Baseline normalized current, ±{tolerance}°C",
            run_out / "qc_current_scaled.png",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        plot_raster(
            joint,
            f"Joint source × current index, ±{tolerance}°C",
            run_out / "qc_joint_source_current.png",
            cmap="magma",
            vmin=0,
            vmax=1,
        )

        plot_raster(
            np.where(high_joint_mask, joint, np.nan),
            f"High joint source-current areas, top {100 - HIGH_JOINT_PERCENTILE}%",
            run_out / "qc_high_joint_top5.png",
            cmap="inferno",
            vmin=joint_thr,
            vmax=1,
        )

        plot_raster(
            labels.astype(np.float32),
            f"High joint source-current patches, ±{tolerance}°C",
            run_out / "qc_high_joint_patch_labels.png",
            cmap="tab20",
        )

        # Overlay-style figure
        plt.figure(figsize=(10, 10))
        plt.imshow(current, cmap="Greys", alpha=0.45, vmin=0, vmax=1)
        plt.imshow(np.where(high_joint_mask, joint, np.nan), cmap="inferno", vmin=joint_thr, vmax=1)
        plt.title(f"High-source and high-current microrefugia, ±{tolerance}°C")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.savefig(run_out / "qc_joint_overlay.png", dpi=300, bbox_inches="tight")
        plt.close()

    # -------------------------------------------------------------------------
    # COMBINED OUTPUTS
    # -------------------------------------------------------------------------

    summary_df = pd.DataFrame(all_summary).sort_values("tolerance")
    summary_df.to_csv(OUT_DIR / "source_current_joint_summary.csv", index=False)

    if all_patches:
        patch_df = pd.concat(all_patches, ignore_index=True)
        patch_df.to_csv(OUT_DIR / "source_current_joint_patch_stats.csv", index=False)

        top_patches = (
            patch_df
            .sort_values(["tolerance", "area_ha", "mean_joint"], ascending=[True, False, False])
            .groupby("tolerance")
            .head(10)
            .reset_index(drop=True)
        )

        top_patches.to_csv(OUT_DIR / "top_joint_microrefugia_patches_by_tolerance.csv", index=False)
    else:
        patch_df = pd.DataFrame()

    if all_zones:
        zones_df = pd.concat(all_zones, ignore_index=True)
        zones_df.to_csv(OUT_DIR / "source_current_joint_by_zone.csv", index=False)

        top_zones = (
            zones_df
            .sort_values(["tolerance", "p95_joint"], ascending=[True, False])
            .groupby("tolerance")
            .head(10)
            .reset_index(drop=True)
        )

        top_zones.to_csv(OUT_DIR / "top_joint_microrefugia_zones_by_tolerance.csv", index=False)

    # -------------------------------------------------------------------------
    # SENSITIVITY FIGURES
    # -------------------------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["tolerance"], summary_df["high_joint_area_ha"], marker="o")
    plt.xlabel("Thermal similarity tolerance (± °C)")
    plt.ylabel("High joint source-current area (ha)")
    plt.title("High source-current area across baseline tolerances")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_joint_area_sensitivity.png", dpi=250)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["tolerance"], summary_df["high_joint_patch_count"], marker="o")
    plt.xlabel("Thermal similarity tolerance (± °C)")
    plt.ylabel("High joint patch count")
    plt.title("High source-current patch count across baseline tolerances")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_joint_patch_count_sensitivity.png", dpi=250)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["tolerance"], summary_df["fraction_high_source_overlapping_high_current"], marker="o")
    plt.xlabel("Thermal similarity tolerance (± °C)")
    plt.ylabel("Fraction of high-source pixels also high-current")
    plt.title("Overlap of high source and high current")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_high_source_high_current_overlap.png", dpi=250)
    plt.close()

    # -------------------------------------------------------------------------
    # INTERPRETATION FILE
    # -------------------------------------------------------------------------

    lines = []
    lines.append("SOURCE-CURRENT JOINT MICROREFUGIA DIAGNOSTIC")
    lines.append("============================================")
    lines.append("")
    lines.append("Definition")
    lines.append("----------")
    lines.append(
        "The joint index identifies pixels that combine high microrefugia source potential "
        "with high baseline connectivity importance. It is calculated as the geometric mean "
        "of source strength and scaled normalized current."
    )
    lines.append("")
    lines.append("Key interpretation")
    lines.append("------------------")
    lines.append(
        "High joint values indicate candidate microrefugia that are not only thermally suitable "
        "but also important within the baseline current network. These are the most relevant "
        "microrefugia for baseline connectivity interpretation."
    )
    lines.append("")
    lines.append("Tolerance summary")
    lines.append("-----------------")

    for _, row in summary_df.iterrows():
        lines.append(
            f"±{row['tolerance']:.1f}°C: "
            f"high joint area = {row['high_joint_area_ha']:.2f} ha; "
            f"patch count = {int(row['high_joint_patch_count'])}; "
            f"largest patch = {row['largest_high_joint_patch_ha']:.2f} ha; "
            f"high-source/high-current overlap = "
            f"{row['fraction_high_source_overlapping_high_current']:.3f}."
        )

    lines.append("")
    lines.append("Suggested paragraph sentence")
    lines.append("----------------------------")
    lines.append(
        "To identify the most functionally relevant microrefugia under baseline conditions, "
        "we combined local refuge quality with normalized current density. Areas with high joint "
        "source-current values represent locations that are both thermally suitable and important "
        "for maintaining connectivity among candidate microrefugia. These areas therefore indicate "
        "priority baseline microrefugia rather than merely locally cool patches."
    )

    (OUT_DIR / "source_current_joint_interpretation.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"[OK] outputs written to: {OUT_DIR}")
    print("[OK] key outputs:")
    print(f"     {OUT_DIR / 'source_current_joint_summary.csv'}")
    print(f"     {OUT_DIR / 'top_joint_microrefugia_patches_by_tolerance.csv'}")
    print(f"     {OUT_DIR / 'source_current_joint_by_zone.csv'}")
    print(f"     {OUT_DIR / 'source_current_joint_interpretation.txt'}")


if __name__ == "__main__":
    main()