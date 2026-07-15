#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f6.7_landscape_characteristics.py

Figure 4, panel C: landscape characteristics that distinguish cool-stable from
cool-amplifying vegetation patches.

This is the companion analysis requested by review: panel C should make the
ecological interpretation of the heatwave-response classes explicit. Cool-stable
patches remain relatively cool and show low thermal amplification during
heatwaves (the thermally desirable outcome); cool-amplifying patches warm up
disproportionately relative to their own average conditions. This script
quantifies which landscape attributes (canopy height, elevation, slope,
southness, eastness, topographic position, rock fraction, distance to ocean,
distance to inland water, building fraction, impervious fraction) are most
associated with each class.

Kept deliberately simple, matching the rest of this analysis:
- no residualization
- no adjusted/multivariable model
- raw patch-level medians with a non-parametric (Kruskal-Wallis) significance
  check per predictor

If an existing 4x4 bivariate raster is found, it uses it. If not, it rebuilds
the 4x4 bivariate classification from absolute mean heatwave temperature and
heatwave-minus-mean-baseline temperature, and writes the bivariate raster to
rasters/.

Positive standardized median differences mean higher values in cool-amplifying
patches. Negative standardized median differences mean higher values in
cool-stable patches.
"""

from __future__ import annotations

import re
import sys
import importlib
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import label as ndi_label
from scipy.stats import kruskal

# =============================================================================
# PATHS
# =============================================================================


def infer_scripts_root(start: Path) -> Path:
    """Walk upward from this file to find the repo scripts root (has DATA/ and figures/)."""
    for parent in [start, *start.parents]:
        if (parent / "DATA").exists() and (parent / "figures").exists():
            return parent
    raise FileNotFoundError(f"Could not locate scripts root above {start}")


ANALYSIS_DIR = Path(__file__).resolve().parent  # figures/2_results/figures/f4_hw_con_loss
SCRIPTS_ROOT = infer_scripts_root(ANALYSIS_DIR)
DATA_DIR = SCRIPTS_ROOT / "DATA"
FIGURES_ROOT = SCRIPTS_ROOT / "figures"
FIGURES_STYLE_DIR = FIGURES_ROOT / "2_results" / "figures"

# This script lives directly inside the f4 panel folder; reuse its existing
# tables/ and rasters/ subfolders rather than nesting a new output tree.
WORKDIR = ANALYSIS_DIR
TABLE_DIR = WORKDIR / "tables"
RASTER_DIR = WORKDIR / "rasters"
PREDICTOR_DIR = DATA_DIR / "predictorstack"
GLOBAL_SETTINGS = FIGURES_STYLE_DIR / "global_plotting_settings.py"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
RASTER_DIR.mkdir(parents=True, exist_ok=True)

HEATWAVE_INPUTS = {
    "2010": DATA_DIR / "predictions" / "2010" / "20100728",
    "2018": DATA_DIR / "predictions" / "2018" / "20180717",
    "2021": DATA_DIR / "predictions" / "2021" / "20210714",
}
BASELINE_MEAN_INPUT = DATA_DIR / "predictions" / "baseline" / "15cm_July_allday" / "pred_20000715_1000.tif"

MAP_TOLERANCE = 1.0
TOL_LABEL = str(MAP_TOLERANCE).replace(".", "p")
TARGET_DOMAIN_RASTER = RASTER_DIR / f"p90_loss_target_domain_tree_veg_nwn_pm{TOL_LABEL}deg.tif"

# Existing bivariate rasters to search before rebuilding.
# Earlier bivariate-map scripts wrote their 4x4 classification raster to f3_2.
BIVAR_SEARCH_DIRS = [
    RASTER_DIR,
    FIGURES_STYLE_DIR / "f3_2" / "rasters",
]
BIVAR_GLOBS = [
    "*bivariate*4x4*mean*.tif",
    "*bivariate*heatwave*response*.tif",
]

OUTPUT_BASENAME = f"f4_panel_c_cool_stable_vs_amplifying_landscape_characteristics_pm{TOL_LABEL}deg"
OUT_BIVAR_RASTER = RASTER_DIR / f"{OUTPUT_BASENAME}_bivariate4x4_mean.tif"
PATCH_VALUES_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_patch_values.csv"
EFFECTS_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_effects.csv"
GROUP_MEDIANS_CSV = TABLE_DIR / f"{OUTPUT_BASENAME}_group_medians.csv"

# =============================================================================
# SETTINGS
# =============================================================================

LOCAL_UTC_OFFSET_HOURS = 3
EXPECTED_PEAK_LOCAL_HOUR = 13
TARGET_HEATWAVE_UTC_HOUR = (EXPECTED_PEAK_LOCAL_HOUR - LOCAL_UTC_OFFSET_HOURS) % 24
AGGREGATION = "mean"
MIN_VALID_ABSOLUTE_TEMP_C = 15.0

N_CLASSES = 4
CLASS_BREAKS = [0.25, 0.50, 0.75]
COOL_STABLE_ID = 1       # low heatwave T, low HW-mean
COOL_AMPLIFYING_ID = 13  # low heatwave T, high HW-mean

CONNECTIVITY = 8
MIN_PATCH_PIXELS = 9
BOOTSTRAP_N = 2000
RANDOM_SEED = 42
PRESENCE_THRESHOLD = 1e-6

COLOR_STABLE = "#1b9e77"
COLOR_AMPLIFY = "#7b3294"
COLOR_ZERO = "#606060"

PREDICTORS = [
    # canopy
    ("canopy_height_max", "max. canopy height", "CHM_10m_MAX.tif", "median"),
    # topography
    ("elevation", "elevation", "DTM_10m_Helsinki.tif", "median"),
    ("slope", "slope", "SLOPE_10m_Helsinki.tif", "median"),
    ("tpi_50m", "TPI (50 m)", "TPI_50m_10m_Helsinki.tif", "median"),
    # water
    ("dist_ocean", "distance to sea", "OCEAN_DIST_10m_Helsinki.tif", "median"),
    ("dist_inland_water", "distance to inland water", "WATER_DIST_10m_Helsinki.tif", "median"),
    # urban
    ("building_50m", "building fraction (50 m)", "BLDG_FRAC_MEAN_50m.tif", "mean_fraction"),
    ("impervious_50m", "impervious fraction (50 m)", "IMPERV_FRAC_50m_Helsinki.tif", "mean_fraction"),
]

# Draw a faint divider line after these zero-based PREDICTORS indices, to keep
# thematically related rows (canopy | topography | water | urban) visually grouped.
PREDICTOR_GROUP_BREAKS_AFTER_INDEX = [0, 3, 5]

# =============================================================================
# STYLE
# =============================================================================

def import_global_plotting_settings():
    if not GLOBAL_SETTINGS.exists():
        return None
    settings_dir = str(GLOBAL_SETTINGS.parent)
    if settings_dir not in sys.path:
        sys.path.insert(0, settings_dir)
    module_name = "global_plotting_settings"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def setup_style():
    global gps, STYLE
    gps = import_global_plotting_settings()
    if gps is not None and hasattr(gps, "STYLE"):
        STYLE = replace(
            gps.STYLE,
            export_png=False,
            export_pdf=False,
            export_svg=True,
            use_tight_bbox=False,
            pad_inches=0.0,
        )
        gps.apply_style(STYLE)
    else:
        STYLE = None
        plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    mpl.rcParams["svg.fonttype"] = "none"

# =============================================================================
# RASTER HELPERS
# =============================================================================

@dataclass(frozen=True)
class RasterSurface:
    label: str
    path: Path
    hour_utc: int
    array: np.ndarray
    profile: dict


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = src.profile.copy()
        profile.update(crs=src.crs, transform=src.transform, width=src.width, height=src.height, nodata=np.nan)
    return arr, profile


def same_grid(a: dict, b: dict) -> bool:
    return (
        a["crs"] == b["crs"]
        and a["transform"] == b["transform"]
        and a["width"] == b["width"]
        and a["height"] == b["height"]
    )


def reproject_to_match(arr: np.ndarray, src_profile: dict, dst_profile: dict, *, resampling=Resampling.bilinear) -> np.ndarray:
    if same_grid(src_profile, dst_profile):
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
        resampling=resampling,
    )
    return dst


def file_hour_utc(path: Path) -> int:
    m = re.search(r"_(\d{4})\.tiff?$", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not extract UTC hour from filename: {path.name}")
    return int(m.group(1)[:2])


def discover_tifs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
    raise FileNotFoundError(f"Input path does not exist: {path}")


def select_heatwave_surface(label: str, input_path: Path) -> RasterSurface:
    tif_paths = discover_tifs(input_path)
    matching = [p for p in tif_paths if file_hour_utc(p) == TARGET_HEATWAVE_UTC_HOUR]
    if len(matching) != 1:
        available = ", ".join(f"{file_hour_utc(p):02d}:00 UTC ({p.name})" for p in tif_paths)
        raise FileNotFoundError(
            f"Expected exactly one {TARGET_HEATWAVE_UTC_HOUR:02d}:00 UTC raster for {label}. Available: {available}"
        )
    chosen = matching[0]
    arr, profile = read_raster(chosen)
    print(f"Selected {label}: {chosen.name} at {file_hour_utc(chosen):02d}:00 UTC")
    return RasterSurface(label=label, path=chosen, hour_utc=file_hour_utc(chosen), array=arr, profile=profile)


def matching_baseline_for_hour(baseline_input: Path, hour_utc: int) -> Path:
    candidates = discover_tifs(baseline_input)
    same_hour = [p for p in candidates if file_hour_utc(p) == hour_utc]
    if len(same_hour) == 1:
        return same_hour[0]
    if baseline_input.is_file() and baseline_input.parent.exists():
        parent_candidates = sorted(baseline_input.parent.glob("*.tif")) + sorted(baseline_input.parent.glob("*.tiff"))
        same_hour = [p for p in parent_candidates if file_hour_utc(p) == hour_utc]
        if len(same_hour) == 1:
            return same_hour[0]
    available = ", ".join(f"{file_hour_utc(p):02d}:00 UTC ({p.name})" for p in candidates)
    raise FileNotFoundError(f"No unique baseline raster for {hour_utc:02d}:00 UTC. Available: {available}")


def aggregate_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    with np.errstate(all="ignore"):
        if AGGREGATION == "mean":
            return np.nanmean(stack, axis=0)
        if AGGREGATION == "max":
            return np.nanmax(stack, axis=0)
    raise ValueError(f"Unsupported AGGREGATION: {AGGREGATION}")


def compute_temperature_surfaces(target_profile: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heatwave_arrays = []
    mean_arrays = []
    for label, input_path in HEATWAVE_INPUTS.items():
        surface = select_heatwave_surface(label, input_path)
        heatwave = reproject_to_match(surface.array, surface.profile, target_profile, resampling=Resampling.bilinear)

        mean_path = matching_baseline_for_hour(BASELINE_MEAN_INPUT, surface.hour_utc)
        mean_arr, mean_profile = read_raster(mean_path)
        tmean = reproject_to_match(mean_arr, mean_profile, target_profile, resampling=Resampling.bilinear)

        heatwave_arrays.append(heatwave)
        mean_arrays.append(tmean)
        print(f"Matched mean baseline for {label}: {mean_path.name}")

    thw = aggregate_arrays(heatwave_arrays)
    tmean = aggregate_arrays(mean_arrays)
    hw_minus_mean = thw - tmean
    return thw, tmean, hw_minus_mean


def write_bivar_raster(path: Path, bivar: np.ndarray, profile: dict) -> None:
    out_profile = profile.copy()
    out_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(np.where(np.isfinite(bivar), bivar, 0).astype(np.uint8), 1)
    print(f"[OK] wrote bivariate raster: {path}")

# =============================================================================
# CLASSIFICATION / PATCHES
# =============================================================================

def rank01(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values)
    ranks = s.rank(method="average").to_numpy(dtype=float)
    n = len(values)
    if n <= 1:
        return np.zeros_like(values, dtype=float)
    return (ranks - 1.0) / (n - 1.0)


def classify_bivar(thw: np.ndarray, offset: np.ndarray, valid_domain: np.ndarray) -> np.ndarray:
    valid = valid_domain & np.isfinite(thw) & np.isfinite(offset) & (thw >= MIN_VALID_ABSOLUTE_TEMP_C)
    if int(valid.sum()) == 0:
        raise ValueError("No valid pixels for bivariate classification.")
    r_thw = np.full(thw.shape, np.nan, dtype=float)
    r_offset = np.full(offset.shape, np.nan, dtype=float)
    r_thw[valid] = rank01(thw[valid])
    r_offset[valid] = rank01(offset[valid])
    temp_cls = np.full(thw.shape, -1, dtype=np.int8)
    off_cls = np.full(thw.shape, -1, dtype=np.int8)
    temp_cls[valid] = np.digitize(r_thw[valid], bins=CLASS_BREAKS, right=False)
    off_cls[valid] = np.digitize(r_offset[valid], bins=CLASS_BREAKS, right=False)
    bivar = np.zeros(thw.shape, dtype=np.uint8)
    bivar[valid] = (off_cls[valid] * N_CLASSES + temp_cls[valid] + 1).astype(np.uint8)
    return bivar


def find_existing_bivar() -> tuple[Path | None, np.ndarray | None, dict | None]:
    candidates = []
    for d in BIVAR_SEARCH_DIRS:
        if not d.exists():
            continue
        for glob in BIVAR_GLOBS:
            candidates.extend(sorted(d.glob(glob)))
    # Avoid using continuous display rasters if any were accidentally written with another name.
    candidates = [p for p in candidates if p.suffix.lower() in {".tif", ".tiff"}]
    if not candidates:
        return None, None, None
    chosen = candidates[0]
    arr, profile = read_raster(chosen)
    arr = np.where(np.isfinite(arr), arr, 0).astype(np.uint8)
    print(f"Using existing bivariate raster: {chosen}")
    return chosen, arr, profile


def patch_structure() -> np.ndarray:
    if CONNECTIVITY == 8:
        return np.ones((3, 3), dtype=int)
    if CONNECTIVITY == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    raise ValueError("CONNECTIVITY must be 4 or 8")


def cell_area_m2(profile: dict) -> float:
    return abs(float(profile["transform"].a) * float(profile["transform"].e))


def label_corner_patches(bivar: np.ndarray, profile: dict) -> tuple[np.ndarray, pd.DataFrame]:
    patch_id_raster = np.zeros(bivar.shape, dtype=np.int32)
    records = []
    next_patch_id = 1
    structure = patch_structure()
    area = cell_area_m2(profile)
    corners = [
        ("cool-stable", COOL_STABLE_ID),
        ("cool-amplifying", COOL_AMPLIFYING_ID),
    ]
    for corner_name, class_id in corners:
        mask = bivar == class_id
        labeled, n_labels = ndi_label(mask, structure=structure)
        if n_labels == 0:
            print(f"[WARN] no patches for {corner_name} class_id={class_id}")
            continue
        counts = np.bincount(labeled.ravel())
        for label_id in range(1, n_labels + 1):
            n_pixels = int(counts[label_id])
            if n_pixels < MIN_PATCH_PIXELS:
                continue
            patch_mask = labeled == label_id
            patch_id = next_patch_id
            next_patch_id += 1
            patch_id_raster[patch_mask] = patch_id
            records.append({
                "patch_id": patch_id,
                "corner": corner_name,
                "bivariate_id": class_id,
                "n_pixels": n_pixels,
                "area_m2": n_pixels * area,
            })
    patch_df = pd.DataFrame.from_records(records)
    if patch_df.empty:
        raise RuntimeError("No cool-stable/cool-amplifying patches retained. Check class IDs or MIN_PATCH_PIXELS.")
    return patch_id_raster, patch_df

# =============================================================================
# PREDICTOR EXTRACTION
# =============================================================================

def predictor_path(filename: str) -> Path:
    path = PREDICTOR_DIR / filename
    if path.exists():
        return path
    # fallback search if the exact case/name differs slightly
    stem = filename.replace(".tif", "").replace(".tiff", "")
    matches = sorted(PREDICTOR_DIR.glob(f"*{stem}*.tif")) + sorted(PREDICTOR_DIR.glob(f"*{stem}*.tiff"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Predictor not found: {path}")


def read_predictor_to_grid(filename: str, target_profile: dict, summary_type: str) -> np.ndarray:
    path = predictor_path(filename)
    arr, profile = read_raster(path)
    resampling = Resampling.nearest if summary_type == "presence_share" else Resampling.bilinear
    out = reproject_to_match(arr, profile, target_profile, resampling=resampling)
    print(f"Loaded predictor {filename} -> {path.name} [{summary_type}]")
    return out


def grouped_summary_by_patch(values: np.ndarray, patch_id_raster: np.ndarray, summary: str) -> pd.DataFrame:
    valid = (patch_id_raster > 0) & np.isfinite(values)
    if not np.any(valid):
        return pd.DataFrame(columns=["patch_id", "value"])
    vals = values[valid].astype(np.float32)
    if summary == "mean_fraction":
        vals = np.clip(vals, 0.0, 1.0)
    elif summary == "presence_share":
        vals = (vals > PRESENCE_THRESHOLD).astype(np.float32)
    elif summary != "median":
        raise ValueError(f"Unknown patch summary: {summary}")
    tmp = pd.DataFrame({"patch_id": patch_id_raster[valid].astype(np.int32), "value": vals})
    if summary in {"mean_fraction", "presence_share"}:
        return tmp.groupby("patch_id", sort=False)["value"].mean().reset_index()
    return tmp.groupby("patch_id", sort=False)["value"].median().reset_index()


def build_patch_predictor_table(patch_id_raster: np.ndarray, patch_df: pd.DataFrame, target_profile: dict) -> pd.DataFrame:
    out = patch_df.copy()
    for key, label, filename, summary in PREDICTORS:
        arr = read_predictor_to_grid(filename, target_profile, summary)
        vals = grouped_summary_by_patch(arr, patch_id_raster, summary).rename(columns={"value": key})
        out = out.merge(vals, on="patch_id", how="left")
    return out

# =============================================================================
# EFFECTS
# =============================================================================

def robust_scale(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    q25, q75 = np.nanpercentile(x, [25, 75])
    iqr = q75 - q25
    if np.isfinite(iqr) and iqr > 0:
        return float(iqr / 1.349)
    sd = np.nanstd(x, ddof=1)
    return float(sd) if np.isfinite(sd) and sd > 0 else np.nan


def standardized_median_difference(stable: np.ndarray, amp: np.ndarray) -> tuple[float, float, float, float, float]:
    stable = stable[np.isfinite(stable)]
    amp = amp[np.isfinite(amp)]
    if stable.size == 0 or amp.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    med_stable = float(np.nanmedian(stable))
    med_amp = float(np.nanmedian(amp))
    raw_diff = med_amp - med_stable
    scale = robust_scale(np.concatenate([stable, amp]))
    std_diff = raw_diff / scale if np.isfinite(scale) and scale > 0 else np.nan
    return med_stable, med_amp, raw_diff, std_diff, scale


def bootstrap_ci(stable: np.ndarray, amp: np.ndarray) -> tuple[float, float]:
    stable = stable[np.isfinite(stable)]
    amp = amp[np.isfinite(amp)]
    if stable.size < 5 or amp.size < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(RANDOM_SEED)
    vals = []
    for _ in range(BOOTSTRAP_N):
        s = rng.choice(stable, size=stable.size, replace=True)
        a = rng.choice(amp, size=amp.size, replace=True)
        *_unused, std = standardized_median_difference(s, a)[:4]
        if np.isfinite(std):
            vals.append(std)
    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def kruskal_test(stable: np.ndarray, amp: np.ndarray) -> tuple[float, float]:
    """Kruskal-Wallis omnibus test. With exactly two groups this is equivalent to a
    Mann-Whitney U / Wilcoxon rank-sum test, and gives the same p-value."""
    stable = stable[np.isfinite(stable)]
    amp = amp[np.isfinite(amp)]
    if stable.size < 2 or amp.size < 2:
        return np.nan, np.nan
    if np.allclose(np.std(stable), 0) and np.allclose(np.std(amp), 0) and np.allclose(np.median(stable), np.median(amp)):
        return np.nan, np.nan
    try:
        h_stat, p_value = kruskal(stable, amp)
    except ValueError:
        return np.nan, np.nan
    return float(h_stat), float(p_value)


def significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def build_effect_table(patch_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, label, filename, summary in PREDICTORS:
        stable = patch_table.loc[patch_table["corner"] == "cool-stable", key].to_numpy(dtype=float)
        amp = patch_table.loc[patch_table["corner"] == "cool-amplifying", key].to_numpy(dtype=float)
        med_stable, med_amp, raw_diff, std_diff, scale = standardized_median_difference(stable, amp)
        lo, hi = bootstrap_ci(stable, amp)
        h_stat, p_value = kruskal_test(stable, amp)
        rows.append({
            "variable": key,
            "label": label,
            "filename": filename,
            "patch_summary": summary,
            "n_cool_stable": int(np.isfinite(stable).sum()),
            "n_cool_amplifying": int(np.isfinite(amp).sum()),
            "median_cool_stable": med_stable,
            "median_cool_amplifying": med_amp,
            "raw_difference_amplifying_minus_stable": raw_diff,
            "standardized_median_difference": std_diff,
            "ci_low": lo,
            "ci_high": hi,
            "higher_in": "cool-amplifying" if np.isfinite(std_diff) and std_diff > 0 else "cool-stable" if np.isfinite(std_diff) and std_diff < 0 else "neutral",
            "scale": scale,
            "kruskal_h": h_stat,
            "kruskal_p": p_value,
            "significant_p05": bool(np.isfinite(p_value) and p_value < 0.05),
        })
    return pd.DataFrame.from_records(rows)

# =============================================================================
# PLOTTING
# =============================================================================
# SVG is the single reference export: a fixed canvas (no tight-bbox auto-crop),
# so the figure scales predictably when placed into the composite figure layout.

PANEL_WIDTH_CM = 18.0
PANEL_HEIGHT_CM = 16.0
AX_RECT = [0.46, 0.16, 0.48, 0.70]  # [left, bottom, width, height], figure fraction


def _style_val(attr: str, default):
    return getattr(STYLE, attr, default) if STYLE is not None else default


def save_figure(fig: plt.Figure, basename: str) -> None:
    svg = WORKDIR / f"{basename}.svg"
    fig.savefig(svg, transparent=True, facecolor="none", edgecolor="none", bbox_inches=None, pad_inches=0)
    cm_per_inch = gps.CM_PER_INCH if gps is not None else 2.54
    width_cm = fig.get_figwidth() * cm_per_inch
    height_cm = fig.get_figheight() * cm_per_inch
    print(f"[OK] wrote {svg} ({width_cm:.2f} x {height_cm:.2f} cm fixed canvas)")


def plot_panel_c(effects: pd.DataFrame) -> None:
    """Panel C: forest plot of standardized landscape-metric contrasts between
    cool-stable and cool-amplifying patches, grouped thematically (canopy,
    topography, water, urban) and annotated with Kruskal-Wallis significance."""
    fs_tick = _style_val("fs_tick", 15)
    fs_axis = _style_val("fs_axis", 18)
    fs_annotation = _style_val("fs_annotation", 20)
    font_family = _style_val("font_family", "DejaVu Sans")
    grid_color = _style_val("col_grid", "#D0D0D0")
    grid_lw = _style_val("grid_linewidth", 1.2)
    grid_alpha = _style_val("grid_alpha", 0.65)
    axes_lw = _style_val("axes_linewidth", 1.5)
    cm_per_inch = gps.CM_PER_INCH if gps is not None else 2.54

    plot_df = effects.copy()
    # Keep the user-specified (thematically grouped) order, top to bottom.
    plot_df["order"] = np.arange(len(plot_df))
    n = len(plot_df)
    plot_df = plot_df.sort_values("order", ascending=False).reset_index(drop=True)
    y = np.arange(n, dtype=float)

    fig = plt.figure(figsize=(PANEL_WIDTH_CM / cm_per_inch, PANEL_HEIGHT_CM / cm_per_inch), dpi=350)
    ax = fig.add_axes(AX_RECT)

    xmin = np.nanmin([plot_df["ci_low"].min(), plot_df["standardized_median_difference"].min(), -1.1])
    xmax = np.nanmax([plot_df["ci_high"].max(), plot_df["standardized_median_difference"].max(), 1.1])
    pad = 0.10 * (xmax - xmin)
    xlim = (xmin - pad, xmax + pad)

    # --- background shading: which side of zero favours which class ----------
    ax.axvspan(xlim[0], 0, color=COLOR_STABLE, alpha=0.15, zorder=0, linewidth=0)
    ax.axvspan(0, xlim[1], color=COLOR_AMPLIFY, alpha=0.15, zorder=0, linewidth=0)
    ax.axvline(0, color=COLOR_ZERO, linewidth=1.4, alpha=0.85, zorder=1)

    for yi, (_, row) in zip(y, plot_df.iterrows()):
        x = row["standardized_median_difference"]
        lo = row["ci_low"]
        hi = row["ci_high"]
        if not np.isfinite(x):
            color = "#8f8f8f"
        elif x > 0:
            color = COLOR_AMPLIFY
        elif x < 0:
            color = COLOR_STABLE
        else:
            color = COLOR_ZERO
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [yi, yi], color=color, linewidth=2.2, alpha=0.9, zorder=2)
        ax.scatter([x], [yi], s=90, color=color, edgecolor="black", linewidth=0.6, zorder=3)

    ax.set_yticks(y)
    ytick_labels = [
        f"{row['label']} {significance_stars(row['kruskal_p'])}".rstrip()
        for _, row in plot_df.iterrows()
    ]
    ax.set_yticklabels(ytick_labels, fontsize=fs_tick, fontfamily=font_family)
    ax.tick_params(axis="y", pad=3, length=0)
    ax.set_xlabel("Standardized median difference", fontsize=fs_axis, fontfamily=font_family)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.6, n - 1 + 0.6)

    ax.grid(axis="x", color=grid_color, linewidth=grid_lw, alpha=grid_alpha, zorder=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(axes_lw)
    ax.spines["bottom"].set_linewidth(axes_lw)
    ax.tick_params(axis="x", labelsize=fs_tick)

    # --- class labels: transparent, rotated, centered on their shaded panel ---
    y_center = (n - 1) / 2.0
    x_stable_center = xlim[0] / 2.0
    x_amplify_center = xlim[1] / 2.0
    ax.text(x_stable_center, y_center, "cool-stable", rotation=90,
            ha="center", va="center", fontsize=fs_annotation, fontweight="bold",
            fontfamily=font_family, color=COLOR_STABLE, alpha=0.5, zorder=1.8)
    ax.text(x_amplify_center, y_center, "cool-amplifying", rotation=90,
            ha="center", va="center", fontsize=fs_annotation, fontweight="bold",
            fontfamily=font_family, color=COLOR_AMPLIFY, alpha=0.5, zorder=1.8)

    fig.text(
        0.40, 0.045,
        "*  p < 0.05      **  p < 0.01      ***  p < 0.001",
        ha="left", va="top", fontsize=fs_tick * 0.8, fontfamily=font_family, color=_style_val("col_grey", "#606060"),
    )

    save_figure(fig, OUTPUT_BASENAME)
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    setup_style()
    print("Resolved settings:")
    print(f"  output dir:       {WORKDIR}")
    print(f"  tables dir:       {TABLE_DIR}")
    print(f"  predictor stack:  {PREDICTOR_DIR}")
    print(f"  target domain:    {TARGET_DOMAIN_RASTER}")
    print("  analysis: raw patch contrasts (cool-stable vs cool-amplifying); no adjustment/residual/model")
    print("  significance: Kruskal-Wallis per predictor")

    target_arr, target_profile = read_raster(TARGET_DOMAIN_RASTER)
    valid_domain = np.isfinite(target_arr) & (target_arr > 0)

    bivar_path, bivar, bivar_profile = find_existing_bivar()
    if bivar is None:
        print("[INFO] no existing bivariate raster found; rebuilding 4x4 classification from heatwave and mean baseline.")
        thw, tmean, hw_minus_mean = compute_temperature_surfaces(target_profile)
        bivar = classify_bivar(thw, hw_minus_mean, valid_domain)
        bivar_profile = target_profile
        write_bivar_raster(OUT_BIVAR_RASTER, bivar, target_profile)
    else:
        if not same_grid(bivar_profile, target_profile):
            warnings.warn("Existing bivariate raster grid differs from target domain; reprojecting class raster with nearest neighbour.")
            bivar = reproject_to_match(bivar.astype(float), bivar_profile, target_profile, resampling=Resampling.nearest).astype(np.uint8)
            bivar_profile = target_profile
        bivar = np.where(valid_domain, bivar, 0).astype(np.uint8)

    patch_id_raster, patch_df = label_corner_patches(bivar, target_profile)
    print(f"Retained patches: {len(patch_df):,}")
    print(patch_df.groupby("corner")["patch_id"].count().to_string())

    patch_table = build_patch_predictor_table(patch_id_raster, patch_df, target_profile)
    effects = build_effect_table(patch_table)

    patch_table.to_csv(PATCH_VALUES_CSV, index=False)
    effects.to_csv(EFFECTS_CSV, index=False)

    group_rows = []
    for key, label, filename, summary in PREDICTORS:
        group_rows.append({
            "variable": key,
            "label": label,
            "patch_summary": summary,
            "median_cool_stable": float(np.nanmedian(patch_table.loc[patch_table["corner"] == "cool-stable", key])),
            "median_cool_amplifying": float(np.nanmedian(patch_table.loc[patch_table["corner"] == "cool-amplifying", key])),
        })
    pd.DataFrame.from_records(group_rows).to_csv(GROUP_MEDIANS_CSV, index=False)

    print(f"[OK] wrote {PATCH_VALUES_CSV}")
    print(f"[OK] wrote {EFFECTS_CSV}")
    print(f"[OK] wrote {GROUP_MEDIANS_CSV}")
    print("\n[EFFECTS]")
    print(effects[["variable", "standardized_median_difference", "ci_low", "ci_high", "kruskal_p", "higher_in"]].to_string(index=False))

    plot_panel_c(effects)


if __name__ == "__main__":
    main()
