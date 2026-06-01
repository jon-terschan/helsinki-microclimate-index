#!/usr/bin/env python3
"""
Figure f4 add-on panel: stacked dotplot of p90 heatwave connectivity loss by land-cover class.

Run this after the f4 connectivity-loss script. It uses the same p90 loss raster and
LULC masks as the violin-plot script, but renders a compact stacked dot distribution
inspired by the supplied ggplot example.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from rasterio.features import rasterize
from shapely.geometry import box

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
FIGURES_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures")
F4_DIR = FIGURES_DIR / "f4"
WORKDIR = F4_DIR
WORKDIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SETTINGS = FIGURES_DIR / "global_plotting_settings.py"
RASTER_DIR = F4_DIR / "rasters"
TABLE_DIR = F4_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

MAP_TOLERANCE = 1.0
USE_CLIPPED_LOSS_RASTER = True
TOL_LABEL = str(MAP_TOLERANCE).replace(".", "p")

LOSS_RASTER = RASTER_DIR / (
    f"p90_to_heatwave_mean_connectivity_loss_"
    f"{'clipped_0_100' if USE_CLIPPED_LOSS_RASTER else 'raw'}_pm{TOL_LABEL}deg.tif"
)
TARGET_DOMAIN_RASTER = RASTER_DIR / f"p90_loss_target_domain_tree_veg_nwn_pm{TOL_LABEL}deg.tif"

LULC_LAYERS = [
    ("Fields", [DATA_DIR / "LULC" / "lc_fields.gpkg"]),
    ("Trees 2–10 m", [DATA_DIR / "LULC" / "lc_trees2-10.gpkg"]),
    ("Trees 10–15 m", [DATA_DIR / "LULC" / "lc_trees10-15.gpkg"]),
    ("Trees >15 m", [
        DATA_DIR / "LULC" / "lc_trees15-20.gpkg",
        DATA_DIR / "LULC" / "lc_trees_o20.gpkg",
    ]),
    ("Other vegetation", [DATA_DIR / "LULC" / "lc_otherveg.gpkg"]),
]

PLOT_ORDER = ["Trees 2–10 m", "Trees 10–15 m", "Trees >15 m", "Fields", "Other vegetation"]
DISPLAY_LABEL_MAP = {
    "Trees 2–10 m": "Trees 2–10 m",
    "Trees 10–15 m": "Trees 10–15 m",
    "Trees >15 m": "Trees >15 m",
    "Fields": "Fields",
    "Other vegetation": "Other veg.",
}

PALETTE = {
    "Trees 2–10 m": "#1966D2",
    "Trees 10–15 m": "#1B5E20",
    "Trees >15 m": "#F5B041",
    "Fields": "#C62828",
    "Other vegetation": "#6A1B9A",
}

OUTPUT_BASENAME = f"f4_panel_f_p90_connectivity_loss_lulc_dotstack_pm{TOL_LABEL}deg"
SUMMARY_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_summary.csv"
VALUES_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_values_sampled.csv"

# =============================================================================
# SETTINGS
# =============================================================================

ASSUME_VECTOR_EPSG_IF_MISSING = 3879
SAVE_VALUES_SAMPLE_CSV = True
RANDOM_SEED = 42
SAMPLE_PER_CLASS = 100
BINWIDTH = 1.5
DOT_X_STEP = 0.22
DOT_SIZE = 16
DOT_ALPHA = 0.95
PANEL_AXES_RECT = [0.10, 0.23, 0.84, 0.58]
Y_LIM = (0, 65)
Y_TICKS = np.arange(0, 61, 10)
TOP_LEGEND_Y = 0.94

# =============================================================================
# GLOBAL STYLE
# =============================================================================

def import_global_plotting_settings():
    if not GLOBAL_SETTINGS.exists():
        raise FileNotFoundError(f"Missing global plotting settings: {GLOBAL_SETTINGS}")
    settings_dir = str(GLOBAL_SETTINGS.parent)
    if settings_dir not in sys.path:
        sys.path.insert(0, settings_dir)
    module_name = "global_plotting_settings"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def ensure_global_style() -> None:
    global gps, STYLE
    gps_local = import_global_plotting_settings()
    if not hasattr(gps_local, "STYLE"):
        raise AttributeError("global_plotting_settings.py must define STYLE = FigureStyle(...).")
    STYLE = replace(
        gps_local.STYLE,
        export_png=False,
        export_pdf=False,
        export_svg=True,
        use_tight_bbox=False,
        pad_inches=0.0,
    )
    gps = gps_local
    gps.apply_style(STYLE)
    mpl.rcParams["svg.fonttype"] = "none"

# =============================================================================
# IO / RASTERIZATION
# =============================================================================

def check_required_files() -> None:
    required = [GLOBAL_SETTINGS, LOSS_RASTER]
    if TARGET_DOMAIN_RASTER.exists():
        required.append(TARGET_DOMAIN_RASTER)
    for _, paths in LULC_LAYERS:
        required.extend(paths)
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print(f"  {p}")
        raise FileNotFoundError("One or more required inputs are missing.")


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = src.profile.copy()
        profile.update(crs=src.crs, transform=src.transform, width=src.width, height=src.height, nodata=np.nan)
    return arr, profile


def raster_extent(profile: dict) -> tuple[float, float, float, float]:
    t = profile["transform"]
    left = t.c
    top = t.f
    right = left + t.a * profile["width"]
    bottom = top + t.e * profile["height"]
    return left, right, bottom, top


def raster_bounds_polygon(profile: dict):
    left, right, bottom, top = raster_extent(profile)
    return box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))


def raster_bbox_in_vector_crs(path: Path, target_profile: dict):
    try:
        info = gpd.read_file(path, rows=1)
        crs = info.crs
        if crs is None:
            crs = f"EPSG:{ASSUME_VECTOR_EPSG_IF_MISSING}"
        rb = gpd.GeoDataFrame(geometry=[raster_bounds_polygon(target_profile)], crs=target_profile["crs"]).to_crs(crs)
        return tuple(rb.total_bounds)
    except Exception:
        return None


def read_vector_fast(path: Path, target_profile: dict, label: str, *, use_bbox: bool = True) -> gpd.GeoDataFrame:
    bbox = raster_bbox_in_vector_crs(path, target_profile) if use_bbox else None
    try:
        gdf = gpd.read_file(path, bbox=bbox) if bbox is not None else gpd.read_file(path)
    except TypeError:
        gdf = gpd.read_file(path)

    if gdf.empty:
        print(f"WARNING: {label} is empty after bbox read; retrying full layer.")
        gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"{label} contains no features: {path}")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError(f"{label} has no non-empty geometries: {path}")

    if gdf.crs is None:
        print(f"WARNING: {label} has no CRS; assuming EPSG:{ASSUME_VECTOR_EPSG_IF_MISSING}.")
        gdf = gdf.set_crs(epsg=ASSUME_VECTOR_EPSG_IF_MISSING, allow_override=True)

    if gdf.crs != target_profile["crs"]:
        gdf = gdf.to_crs(target_profile["crs"])

    print(f"Loaded {label}: {len(gdf)} features")
    return gdf


def rasterize_gdf(gdf: gpd.GeoDataFrame, profile: dict, *, all_touched: bool = False) -> np.ndarray:
    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        return np.zeros((profile["height"], profile["width"]), dtype=bool)
    arr = rasterize(
        ((geom, 1) for geom in geoms),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        all_touched=all_touched,
        dtype=np.uint8,
    )
    return arr.astype(bool)


def load_lulc_masks(profile: dict, valid_domain: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for class_name, paths in LULC_LAYERS:
        class_mask = np.zeros_like(valid_domain, dtype=bool)
        for path in paths:
            gdf = read_vector_fast(path, profile, f"LULC {class_name}", use_bbox=True)
            class_mask |= rasterize_gdf(gdf, profile, all_touched=False)
        class_mask &= valid_domain
        masks[class_name] = class_mask
        print(f"LULC {class_name}: {int(class_mask.sum())} valid raster cells")

    exclusive: dict[str, np.ndarray] = {}
    already = np.zeros_like(valid_domain, dtype=bool)
    for class_name, _ in LULC_LAYERS:
        m = masks[class_name] & (~already)
        exclusive[class_name] = m
        already |= m
        print(f"LULC {class_name} exclusive cells: {int(m.sum())}")
    return exclusive

# =============================================================================
# DATA EXTRACTION
# =============================================================================

def finite_values_for_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask]
    return vals[np.isfinite(vals)]


def extract_values(loss_arr: np.ndarray, lulc_masks: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    summary = []
    for class_name in PLOT_ORDER:
        mask = lulc_masks.get(class_name)
        if mask is None:
            continue
        vals = finite_values_for_mask(loss_arr, mask)
        if vals.size == 0:
            continue
        summary.append({
            "class": class_name,
            "label": DISPLAY_LABEL_MAP.get(class_name, class_name),
            "n_pixels": int(vals.size),
            "mean_loss_percent": float(np.nanmean(vals)),
            "median_loss_percent": float(np.nanmedian(vals)),
            "q25_loss_percent": float(np.nanpercentile(vals, 25)),
            "q75_loss_percent": float(np.nanpercentile(vals, 75)),
            "p90_loss_percent": float(np.nanpercentile(vals, 90)),
        })
        records.extend({"class": class_name, "loss_percent": float(v)} for v in vals)
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(summary)


def sample_values(values_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    sampled = []
    for class_name in PLOT_ORDER:
        sub = values_df.loc[values_df["class"] == class_name, ["class", "loss_percent"]].copy()
        if sub.empty:
            continue
        if len(sub) > SAMPLE_PER_CLASS:
            idx = rng.choice(sub.index.to_numpy(), size=SAMPLE_PER_CLASS, replace=False)
            sub = sub.loc[idx].copy()
        sampled.append(sub)
    if not sampled:
        return pd.DataFrame(columns=["class", "loss_percent"])
    return pd.concat(sampled, ignore_index=True)

# =============================================================================
# PLOTTING
# =============================================================================

def save_svg(fig: plt.Figure, basename: str) -> None:
    gps.make_transparent(fig)
    out = WORKDIR / f"{basename}.svg"
    fig.savefig(out, transparent=STYLE.transparent, facecolor="none", edgecolor="none", bbox_inches=None, pad_inches=0)
    width_cm = fig.get_figwidth() * gps.CM_PER_INCH
    height_cm = fig.get_figheight() * gps.CM_PER_INCH
    print(f"[OK] wrote {out} ({width_cm:.2f} × {height_cm:.2f} cm fixed canvas)")


def build_dotstack_coordinates(sampled_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    frames = []
    max_stack = 0
    for class_name in PLOT_ORDER:
        sub = sampled_df[sampled_df["class"] == class_name].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("loss_percent", kind="mergesort").reset_index(drop=True)
        bins = np.floor(sub["loss_percent"].to_numpy(dtype=float) / BINWIDTH).astype(int)
        x_positions = np.zeros(len(sub), dtype=float)
        counts: dict[int, int] = {}
        for i, b in enumerate(bins):
            c = counts.get(int(b), 0)
            x_positions[i] = c * DOT_X_STEP
            counts[int(b)] = c + 1
            max_stack = max(max_stack, c + 1)
        sub["x"] = x_positions
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True) if frames else sampled_df.copy()
    return out, float(max_stack)


def plot_dotstack(values_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    sampled_df = sample_values(values_df)
    if sampled_df.empty:
        raise RuntimeError("No sampled values available for plotting.")
    sampled_df, max_stack = build_dotstack_coordinates(sampled_df)

    fig = plt.figure(figsize=(13.5/2.54, 9.6/2.54), dpi=STYLE.dpi_export)
    ax = fig.add_axes(PANEL_AXES_RECT)
    gps.style_axis(ax, STYLE, grid_y=True, grid_x=False)

    for class_name in PLOT_ORDER:
        sub = sampled_df[sampled_df["class"] == class_name]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"],
            sub["loss_percent"],
            s=DOT_SIZE,
            c=PALETTE[class_name],
            alpha=DOT_ALPHA,
            linewidths=0,
            zorder=4,
        )

    ax.set_ylim(*Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.set_xlim(-0.25, max_stack * DOT_X_STEP + 4.8)
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines["bottom"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="None", markersize=6.2,
               markerfacecolor=PALETTE[c], markeredgecolor=PALETTE[c], markeredgewidth=0,
               label=DISPLAY_LABEL_MAP.get(c, c))
        for c in PLOT_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.06, TOP_LEGEND_Y),
        frameon=False,
        ncol=5,
        fontsize=max(7, STYLE.fs_legend * 0.86),
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )

    fig.text(
        0.10, 0.84,
        "Connectivity Loss (%) ↑",
        ha="left", va="center",
        fontsize=max(8, STYLE.fs_axis * 0.90),
        fontfamily=STYLE.font_family,
        color=STYLE.col_black,
    )

    mean_loss = float(summary_df["mean_loss_percent"].mean()) if not summary_df.empty else float(np.nan)
    n_points = int(len(sampled_df))
    fig.text(0.30, 0.12, "Mean Loss", ha="center", va="bottom",
             fontsize=max(8, STYLE.fs_axis * 0.88), fontfamily=STYLE.font_family, color=STYLE.col_black)
    fig.text(0.30, 0.08, f"{mean_loss:.1f}%", ha="center", va="top",
             fontsize=max(8, STYLE.fs_tick * 0.95), fontfamily=STYLE.font_family, color=STYLE.col_black)
    fig.text(0.72, 0.12, "Data Points", ha="center", va="bottom",
             fontsize=max(8, STYLE.fs_axis * 0.88), fontfamily=STYLE.font_family, color=STYLE.col_black)
    fig.text(0.72, 0.08, f"{n_points}", ha="center", va="top",
             fontsize=max(8, STYLE.fs_tick * 0.95), fontfamily=STYLE.font_family, color=STYLE.col_black)
    fig.lines.append(plt.Line2D([0.07, 0.93], [0.19, 0.19], transform=fig.transFigure,
                                color=STYLE.col_grey, alpha=0.35, linewidth=0.8))

    save_svg(fig, OUTPUT_BASENAME)
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_global_style()
    print("Resolved global plotting style:")
    print(f"  output dir:  {WORKDIR}")
    print(f"  input loss:  {LOSS_RASTER}")
    print(f"  sample/class: {SAMPLE_PER_CLASS}")
    print(f"  binwidth:     {BINWIDTH}")

    check_required_files()
    loss_arr, profile = read_raster(LOSS_RASTER)

    valid_domain = np.isfinite(loss_arr)
    if TARGET_DOMAIN_RASTER.exists():
        target_arr, target_profile = read_raster(TARGET_DOMAIN_RASTER)
        if (
            target_profile["crs"] != profile["crs"]
            or target_profile["transform"] != profile["transform"]
            or target_profile["width"] != profile["width"]
            or target_profile["height"] != profile["height"]
        ):
            raise ValueError("Target-domain raster is not aligned with the loss raster.")
        valid_domain &= np.isfinite(target_arr) & (target_arr > 0)
        print(f"Using target-domain mask: {TARGET_DOMAIN_RASTER}")

    lulc_masks = load_lulc_masks(profile, valid_domain)
    values_df, summary_df = extract_values(loss_arr, lulc_masks)
    if summary_df.empty:
        raise RuntimeError("No finite LULC connectivity-loss values were extracted.")

    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"[OK] wrote {SUMMARY_CSV}")

    sampled_df = sample_values(values_df)
    if SAVE_VALUES_SAMPLE_CSV:
        sampled_df.to_csv(VALUES_CSV, index=False)
        print(f"[OK] wrote {VALUES_CSV}")

    plot_dotstack(values_df, summary_df)


if __name__ == "__main__":
    main()
