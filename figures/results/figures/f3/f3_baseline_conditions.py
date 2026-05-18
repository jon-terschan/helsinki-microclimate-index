#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import importlib
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import rasterio
from matplotlib.colors import Normalize, to_rgba
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter
from shapely.geometry import box


# =============================================================================
# PATHS
# =============================================================================

DATA = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
OMNI = DATA / "omniscape"

SOURCE_PATH = OMNI / "sources" / "source_p90_coolness_stability.tif"

BASELINE_RUN_DIR = (
    OMNI
    / "output"
    / "conditional_multiruns"
    / "condition_average__pm0p5deg"
)

CUM_CURRENT_PATH = BASELINE_RUN_DIR / "cum_currmap.tif"
NORMALIZED_CURRENT_PATH = BASELINE_RUN_DIR / "normalized_cum_currmap.tif"

TREE_PATH = DATA / "predictorstack" / "TREE_FRAC_10m.tif"
NWN_PATH = DATA / "predictorstack" / "NWN_FRAC_10m.tif"
WATER_PATH = DATA / "predictorstack" / "WATER_FRAC_10m_Helsinki.tif"
OCEAN_PATH = DATA / "predictorstack" / "OCEAN_FRAC_10m_Helsinki.tif"
IMPERVIOUS_PATH = DATA / "predictorstack" / "IMPERV_FRAC_10m_Helsinki.tif"
BUILDING_PATH = DATA / "predictorstack" / "BLDG_FRAC_10m.tif"
BAREGROUND_VECTOR_PATH = DATA / "LULC" / "lc_bareground.gpkg"

PERUSPIIRI_PATH = DATA / "figures" / "offset_figure" / "peruspiiri_WFS.gpkg"

FIG_DIR = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures\f3"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GLOBAL STYLE / EXPORT LOGIC
# =============================================================================

GLOBAL_STYLE_DIR = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures"
)

if str(GLOBAL_STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(GLOBAL_STYLE_DIR))

try:
    import global_plotting_settings as gps
    gps = importlib.reload(gps)
except ImportError as e:
    raise ImportError(
        "Could not import global_plotting_settings.py from:\n"
        f"  {GLOBAL_STYLE_DIR}"
    ) from e

if not hasattr(gps, "STYLE"):
    raise AttributeError(
        "global_plotting_settings.py must define STYLE = FigureStyle(...)."
    )

STYLE = replace(
    gps.STYLE,
    export_png=False,
    export_pdf=False,
    export_svg=True,
    use_tight_bbox=False,
    pad_inches=0.0,
)

# Figure-specific geometry only.
MAP_AXES_RECT = [0.20, 0.07, 0.76, 0.86]
MAP_COLORBAR_RECT = [0.13, 0.14, 0.028, 0.72]

# Two-panel vertical preview composite layout.
COMPOSITE_GUTTER_Y_CM = 0.8
COMPOSITE_MARGIN_X_CM = 0.4
COMPOSITE_MARGIN_Y_CM = 0.4

COMPOSITE_WIDTH_CM = STYLE.panel_width_cm + 2 * COMPOSITE_MARGIN_X_CM
COMPOSITE_HEIGHT_CM = (
    2 * STYLE.panel_height_cm
    + COMPOSITE_GUTTER_Y_CM
    + 2 * COMPOSITE_MARGIN_Y_CM
)

DRAW_PANEL_TITLES = False
DRAW_OUTER_PERUSPIIRI = True
DRAW_INTERNAL_PERUSPIIRI = False

INTERNAL_BOUNDARY_COLOR = "black"
INTERNAL_BOUNDARY_WIDTH = 0.14
INTERNAL_BOUNDARY_ALPHA = 0.10

OUTER_BOUNDARY_COLOR = "black"
OUTER_BOUNDARY_WIDTH = 0.38
OUTER_BOUNDARY_ALPHA = 0.24

PAD_FRACTION = 0.02
MAP_FIT_MODE = "crop"

# Target-domain definition.
VEGETATION_MIN_FRACTION = 0.05
WATER_MIN_FRACTION = 0.05

# Hard exclusion rules requested by user:
#   building > 0     -> gone
#   impervious > 0.5 -> gone
BUILDING_EXCLUDE_FRACTION = 0.0
IMPERVIOUS_EXCLUDE_FRACTION = 0.5

# Display-only transparency rule for remaining impervious mixed pixels:
#   impervious 0.1 -> almost fully visible
#   impervious 0.5 -> almost transparent
IMPERVIOUS_ALPHA_VISIBLE_AT = 0.10
IMPERVIOUS_ALPHA_TRANSPARENT_AT = 0.50
IMPERVIOUS_ALPHA_MIN_FACTOR = 0.08
DISPLAY_SMOOTH_SIGMA_IMPERVIOUS = 0.35

# Water/ocean background context.
DRAW_WATER_BACKGROUND = True
WATER_BACKGROUND_RGB = (0.82, 0.91, 0.97)
WATER_BACKGROUND_ALPHA = 0.58

# Display tuning for source-strength surface.
# Panel A should emphasize coherent low/high-refuge regions while remaining
# faithful to the data and avoiding patchy salt-and-pepper appearance.
# Strategy:
#   - use mild smoothing for the displayed color field
#   - derive the emphasis field from a more strongly smoothed source surface
#   - smooth alpha more strongly than the color field
#   - never let the middle quantiles disappear completely
#   - apply a mild center-compression to the displayed values so the tails read
#     a bit more clearly without distorting the map too much
DISPLAY_SMOOTH_SIGMA_SOURCE_VALUE = 0.35
DISPLAY_SMOOTH_SIGMA_SOURCE_RANK = 1.25
DISPLAY_SMOOTH_SIGMA_SOURCE_ALPHA = 1.80
SOURCE_ALPHA_MIN_MIDDLE = 0.24
SOURCE_ALPHA_MAX_EXTREME = 0.95
SOURCE_ALPHA_GAMMA = 1.15
SOURCE_MIDDLE_QUANTILE_LOW = 0.42
SOURCE_MIDDLE_QUANTILE_HIGH = 0.58
SOURCE_COLOR_CONTRAST_GAMMA = 0.88

# Display tuning for cumulative current.
DISPLAY_SMOOTH_SIGMA_CURRENT = 0.55
DISPLAY_SMOOTH_SIGMA_ALPHA = 0.55
DISPLAY_SMOOTH_SIGMA_BOTTLENECK = 0.70
DISPLAY_SMOOTH_SIGMA_TARGET_EDGE = 0.75

# Percentile clipping after log1p transform for cumulative-current display.
CURRENT_DISPLAY_PERCENTILES = (15, 98.5)

# Weak alpha modulation based on normalized current.
ALPHA_MIN_IMPEDED = 0.84
ALPHA_MAX_CONNECTED = 0.99

# Bottlenecks from upper normalized current, made clearly visible.
BOTTLENECK_PERCENTILE = 95
BOTTLENECK_COLOR = "#FF2A8A"
BOTTLENECK_ALPHA = 0.95
BOTTLENECK_VISIBILITY_CUTOFF = 0.06
# Contour disabled: filled raster overlay is safer with imshow(origin="upper").
BOTTLENECK_CONTOUR_COLOR = "#4A001F"
BOTTLENECK_CONTOUR_LINEWIDTH = 0.70

EXPORT_COMPOSITE_PREVIEW = False

gps.apply_style(STYLE)
CM_PER_INCH = gps.CM_PER_INCH
FONT_FAMILY = STYLE.font_family

print("Resolved global plotting style:")
print(f"  global file:  {Path(gps.__file__)}")
print(f"  output dir:   {FIG_DIR}")
print(f"  canvas:       {STYLE.panel_width_cm} × {STYLE.panel_height_cm} cm")
print(f"  dpi_export:   {STYLE.dpi_export}")
print(f"  exports:      PNG={STYLE.export_png}, SVG={STYLE.export_svg}, PDF={STYLE.export_pdf}")
print(f"  map axes:     {MAP_AXES_RECT}")
print(f"  colorbar:     {MAP_COLORBAR_RECT}")
print("  export mode:  fixed canvas, no tight bbox, no composite preview")


# =============================================================================
# PANEL SPECS
# =============================================================================

@dataclass(frozen=True)
class MapPanelSpec:
    panel_id: str
    basename: str
    raster_path: Path
    colorbar_label: str
    cmap: str
    panel_kind: str
    fixed_ticks: tuple[float, ...] | None = None
    low_note: str | None = None
    high_note: str | None = None


PANEL_SPECS = [
    MapPanelSpec(
        panel_id="A",
        basename="f3_panel_a_source_strength_surface_coherent_emphasis",
        raster_path=SOURCE_PATH,
        colorbar_label="Source strength",
        cmap="BrBG",
        panel_kind="source",
        fixed_ticks=(0.0, 1.0),
        low_note="low refuge",
        high_note="high refuge",
    ),
    MapPanelSpec(
        panel_id="B",
        basename="f3_panel_b_cumulative_current_final_masked_bottlenecks_gutterfixed",
        raster_path=CUM_CURRENT_PATH,
        colorbar_label="Cumulative current",
        cmap="viridis",
        panel_kind="cumulative_integrated",
        fixed_ticks=(),
        low_note="dispersed",
        high_note="concentrated",
    ),
]


# =============================================================================
# HELPERS
# =============================================================================

def check_required_files() -> None:
    required = [
        SOURCE_PATH,
        CUM_CURRENT_PATH,
        NORMALIZED_CURRENT_PATH,
        TREE_PATH,
        NWN_PATH,
        WATER_PATH,
        OCEAN_PATH,
        IMPERVIOUS_PATH,
        BUILDING_PATH,
        BAREGROUND_VECTOR_PATH,
        PERUSPIIRI_PATH,
    ]

    missing = [path for path in required if not path.exists()]
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  {path}")
        raise FileNotFoundError("One or more required inputs are missing.")


def new_panel_figure():
    return gps.new_panel_figure(STYLE)


def save_single_panel(fig, basename: str) -> None:
    """
    Save a fixed physical SVG canvas.

    Do not use bbox_inches='tight'. The left legend gutter is explicitly
    reserved inside the canvas, so labels are not clipped and both panels export
    with identical physical dimensions.
    """
    gps.make_transparent(fig)

    out_svg = FIG_DIR / f"{basename}.svg"
    fig.savefig(
        out_svg,
        transparent=STYLE.transparent,
        facecolor="none",
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )

    width_cm = fig.get_figwidth() * CM_PER_INCH
    height_cm = fig.get_figheight() * CM_PER_INCH
    print(f"[OK] wrote {out_svg} ({width_cm:.2f} × {height_cm:.2f} cm fixed canvas)")

def rect_within(panel_rect: list[float], inner_rect: list[float]) -> list[float]:
    px, py, pw, ph = panel_rect
    ix, iy, iw, ih = inner_rect
    return [
        px + ix * pw,
        py + iy * ph,
        iw * pw,
        ih * ph,
    ]


def composite_save(fig: plt.Figure, basename: str) -> None:
    gps.make_transparent(fig)

    save_kwargs = {
        "transparent": STYLE.transparent,
        "facecolor": "none",
        "edgecolor": "none",
    }

    if STYLE.use_tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = STYLE.pad_inches

    out_svg = FIG_DIR / f"{basename}.svg"
    fig.savefig(out_svg, **save_kwargs)
    width_cm = fig.get_figwidth() * CM_PER_INCH
    height_cm = fig.get_figheight() * CM_PER_INCH
    print(f"[OK] wrote {out_svg} ({width_cm:.2f} × {height_cm:.2f} cm canvas)")


def load_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = src.profile.copy()
    return arr, profile


def assert_same_grid(reference_profile: dict, other_profile: dict, label_name: str) -> None:
    checks = [
        ("width", reference_profile["width"], other_profile["width"]),
        ("height", reference_profile["height"], other_profile["height"]),
        ("transform", reference_profile["transform"], other_profile["transform"]),
        ("crs", reference_profile["crs"], other_profile["crs"]),
    ]
    for key, expected, observed in checks:
        if expected != observed:
            raise ValueError(
                f"Raster {key} mismatch for {label_name}: {observed} != {expected}"
            )


def raster_extent(profile: dict) -> tuple[float, float, float, float]:
    west, south, east, north = rasterio.transform.array_bounds(
        profile["height"],
        profile["width"],
        profile["transform"],
    )
    return west, east, south, north


def axes_physical_aspect(fig: plt.Figure, ax) -> float:
    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()
    return (pos.width * fig_w) / (pos.height * fig_h)


def bounds_to_match_axes_aspect(
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    target_aspect: float,
    mode: str = "crop",
) -> tuple[float, float, float, float]:
    width = right - left
    height = top - bottom

    if width <= 0 or height <= 0:
        raise ValueError("Invalid map bounds.")

    data_aspect = width / height
    cx = 0.5 * (left + right)
    cy = 0.5 * (bottom + top)

    if mode == "crop":
        if data_aspect > target_aspect:
            new_height = height
            new_width = height * target_aspect
        else:
            new_width = width
            new_height = width / target_aspect
    elif mode == "pad":
        if data_aspect > target_aspect:
            new_width = width
            new_height = width / target_aspect
        else:
            new_height = height
            new_width = height * target_aspect
    else:
        raise ValueError(f"Unknown MAP_FIT_MODE: {mode!r}")

    new_left = cx - 0.5 * new_width
    new_right = cx + 0.5 * new_width
    new_bottom = cy - 0.5 * new_height
    new_top = cy + 0.5 * new_height

    return new_left, new_right, new_bottom, new_top


def rasterize_polygons_to_mask(gdf: gpd.GeoDataFrame, profile: dict) -> np.ndarray:
    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        raise ValueError("No valid geometries for rasterization.")

    arr = rasterize(
        shapes=[(geom, 1) for geom in geoms],
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        all_touched=False,
        dtype=np.uint8,
    )
    return arr.astype(bool)


def rasterize_bareground(profile: dict) -> np.ndarray:
    gpd.options.io_engine = "pyogrio"

    bareground = gpd.read_file(BAREGROUND_VECTOR_PATH)
    if bareground.crs != profile["crs"]:
        bareground = bareground.to_crs(profile["crs"])

    geoms = [geom for geom in bareground.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        raise ValueError(f"No valid geometries found in {BAREGROUND_VECTOR_PATH}")

    arr = rasterize(
        shapes=[(geom, 1.0) for geom in geoms],
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0.0,
        all_touched=False,
        dtype=np.float32,
    )
    return np.clip(arr, 0.0, 1.0)


def prepare_boundaries_and_crop(profile: dict) -> dict:
    gpd.options.io_engine = "pyogrio"

    peruspiiri = gpd.read_file(PERUSPIIRI_PATH)
    if peruspiiri.crs != profile["crs"]:
        peruspiiri = peruspiiri.to_crs(profile["crs"])

    helsinki_outline = peruspiiri.dissolve()

    extent = raster_extent(profile)
    raster_left, raster_right, raster_bottom, raster_top = extent

    left, bottom, right, top = helsinki_outline.total_bounds

    left = max(left, raster_left)
    right = min(right, raster_right)
    bottom = max(bottom, raster_bottom)
    top = min(top, raster_top)

    if left >= right or bottom >= top:
        raise ValueError(
            "The peruspiiri outline bounds and raster bounds do not overlap."
        )

    raster_box = gpd.GeoDataFrame(
        geometry=[box(left, bottom, right, top)],
        crs=profile["crs"],
    )

    peruspiiri_plot = gpd.clip(peruspiiri, raster_box)
    outline_plot = peruspiiri_plot.dissolve()
    peruspiiri_mask = rasterize_polygons_to_mask(peruspiiri_plot, profile)

    pad_x = (right - left) * PAD_FRACTION
    pad_y = (top - bottom) * PAD_FRACTION

    return {
        "peruspiiri_plot": peruspiiri_plot,
        "outline_plot": outline_plot,
        "peruspiiri_mask": peruspiiri_mask,
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }


def smooth_display(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr

    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    weights = valid.astype(float)

    smooth_values = gaussian_filter(filled, sigma=sigma)
    smooth_weights = gaussian_filter(weights, sigma=sigma)

    return np.divide(
        smooth_values,
        smooth_weights,
        out=np.full_like(arr, np.nan, dtype=float),
        where=smooth_weights > 0,
    )


def build_target_domain(profile: dict, peruspiiri_mask: np.ndarray) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Target domain:
      peruspiiri ∩ vegetation ∩ non-water ∩ non-building ∩ non-high-impervious

    Hard rules:
      building > 0.0  -> excluded
      impervious > 0.5 -> excluded

    Soft display rule for remaining impervious cells:
      0.1 -> almost fully visible
      0.5 -> almost transparent
    """
    tree, tree_profile = load_raster(TREE_PATH)
    assert_same_grid(profile, tree_profile, "TREE")

    nwn, nwn_profile = load_raster(NWN_PATH)
    assert_same_grid(profile, nwn_profile, "NWN")

    water, water_profile = load_raster(WATER_PATH)
    assert_same_grid(profile, water_profile, "WATER")

    ocean, ocean_profile = load_raster(OCEAN_PATH)
    assert_same_grid(profile, ocean_profile, "OCEAN")

    impervious, imperv_profile = load_raster(IMPERVIOUS_PATH)
    assert_same_grid(profile, imperv_profile, "IMPERVIOUS")

    building, building_profile = load_raster(BUILDING_PATH)
    assert_same_grid(profile, building_profile, "BUILDING")

    tree = np.clip(np.nan_to_num(tree, nan=0.0), 0.0, 1.0)
    nwn = np.clip(np.nan_to_num(nwn, nan=0.0), 0.0, 1.0)
    water = np.clip(np.nan_to_num(water, nan=0.0), 0.0, 1.0)
    ocean = np.clip(np.nan_to_num(ocean, nan=0.0), 0.0, 1.0)
    impervious = np.clip(np.nan_to_num(impervious, nan=0.0), 0.0, 1.0)
    building = np.clip(np.nan_to_num(building, nan=0.0), 0.0, 1.0)

    bareground = rasterize_bareground(profile)
    nwn_veg = np.where(bareground > 0, 0.0, nwn)
    nwn_veg = np.clip(nwn_veg, 0.0, 1.0)

    vegetation_fraction = np.clip(tree + nwn_veg, 0.0, 1.0)
    vegetation_domain = (
        np.isfinite(vegetation_fraction)
        & (vegetation_fraction >= VEGETATION_MIN_FRACTION)
    )

    water_domain = peruspiiri_mask & (
        (water >= WATER_MIN_FRACTION)
        | (ocean >= WATER_MIN_FRACTION)
    )

    building_excluded = building > BUILDING_EXCLUDE_FRACTION
    impervious_excluded = impervious > IMPERVIOUS_EXCLUDE_FRACTION

    target_domain = (
        peruspiiri_mask
        & vegetation_domain
        & (~water_domain)
        & (~building_excluded)
        & (~impervious_excluded)
    )

    target_feather = gaussian_filter(
        target_domain.astype(float),
        sigma=DISPLAY_SMOOTH_SIGMA_TARGET_EDGE,
    )
    target_feather = np.clip(target_feather, 0.0, 1.0)

    imperv_smooth = smooth_display(impervious, DISPLAY_SMOOTH_SIGMA_IMPERVIOUS)
    imperv_rel = (
        (imperv_smooth - IMPERVIOUS_ALPHA_VISIBLE_AT)
        / (IMPERVIOUS_ALPHA_TRANSPARENT_AT - IMPERVIOUS_ALPHA_VISIBLE_AT)
    )
    imperv_rel = np.clip(imperv_rel, 0.0, 1.0)
    impervious_factor = 1.0 - imperv_rel * (1.0 - IMPERVIOUS_ALPHA_MIN_FACTOR)
    impervious_factor = np.clip(impervious_factor, IMPERVIOUS_ALPHA_MIN_FACTOR, 1.0)
    impervious_factor = np.where(target_domain, impervious_factor, 0.0)

    print("[TARGET DOMAIN]")
    print(f"  peruspiiri cells          = {int(np.sum(peruspiiri_mask))}")
    print(f"  vegetation cells          = {int(np.sum(vegetation_domain))}")
    print(f"  water/ocean cells         = {int(np.sum(water_domain))}")
    print(f"  building-excluded cells   = {int(np.sum(building_excluded & peruspiiri_mask))}")
    print(f"  impervious-excluded cells = {int(np.sum(impervious_excluded & peruspiiri_mask))}")
    print(f"  target-domain cells       = {int(np.sum(target_domain))}")

    return target_domain, target_feather, water_domain, impervious_factor


def make_water_background_rgba(water_domain: np.ndarray) -> np.ndarray | None:
    if not DRAW_WATER_BACKGROUND or not np.any(water_domain):
        return None

    rgba = np.zeros((water_domain.shape[0], water_domain.shape[1], 4), dtype=float)
    rgba[..., 0] = WATER_BACKGROUND_RGB[0]
    rgba[..., 1] = WATER_BACKGROUND_RGB[1]
    rgba[..., 2] = WATER_BACKGROUND_RGB[2]
    rgba[..., 3] = np.where(water_domain, WATER_BACKGROUND_ALPHA, 0.0)
    return rgba


def make_source_display(
    arr: np.ndarray,
    target_domain: np.ndarray,
    target_feather: np.ndarray,
    impervious_factor: np.ndarray,
    cmap,
):
    """
    Panel A source-strength display.

    Uses the same mask/crop/export machinery as Panel B.

    Implementation details:
      - alpha is baked directly into an RGBA image before imshow()
      - the displayed color field is only mildly smoothed
      - the emphasis field is derived from a more strongly smoothed source
        surface so coherent low/high-refuge regions stand out
      - alpha never drops to zero in the middle quantiles, which avoids the
        holey, patchy appearance from earlier versions
      - alpha is smoothed after impervious masking and then clipped back to the
        hard target domain
    """
    arr_target = np.where(target_domain, arr, np.nan)

    # Mild smoothing for the displayed color field.
    arr_display = smooth_display(arr_target, DISPLAY_SMOOTH_SIGMA_SOURCE_VALUE)
    arr_display = np.clip(arr_display, 0.0, 1.0)

    # Build a coarser, more spatially coherent emphasis field.
    arr_rank_field = smooth_display(arr_target, DISPLAY_SMOOTH_SIGMA_SOURCE_RANK)
    arr_rank_field = np.clip(arr_rank_field, 0.0, 1.0)

    # Mild center-compression in the displayed values so tails read more
    # clearly without introducing strong distortion.
    centered = 2.0 * (arr_display - 0.5)
    arr_color = 0.5 + 0.5 * np.sign(centered) * np.power(np.abs(centered), SOURCE_COLOR_CONTRAST_GAMMA)
    arr_color = np.clip(arr_color, 0.0, 1.0)

    norm = Normalize(vmin=0.0, vmax=1.0)

    vals_rank = arr_rank_field[np.isfinite(arr_rank_field)]
    alpha_extreme = np.full_like(arr_rank_field, SOURCE_ALPHA_MIN_MIDDLE, dtype=float)

    if vals_rank.size > 0:
        vals_sorted = np.sort(vals_rank)
        ranks = (
            np.searchsorted(vals_sorted, arr_rank_field, side='left')
            / max(len(vals_sorted) - 1, 1)
        )

        q_low = SOURCE_MIDDLE_QUANTILE_LOW
        q_high = SOURCE_MIDDLE_QUANTILE_HIGH

        low_side = np.clip((q_low - ranks) / max(q_low, 1e-6), 0.0, 1.0)
        high_side = np.clip((ranks - q_high) / max(1.0 - q_high, 1e-6), 0.0, 1.0)
        extreme_strength = np.maximum(low_side, high_side)

        alpha_extreme = (
            SOURCE_ALPHA_MIN_MIDDLE
            + (SOURCE_ALPHA_MAX_EXTREME - SOURCE_ALPHA_MIN_MIDDLE)
            * np.power(extreme_strength, SOURCE_ALPHA_GAMMA)
        )

        raw_alpha = np.where(
            np.isfinite(arr_display) & target_domain,
            alpha_extreme * impervious_factor,
            np.nan,
        )

        # Stronger alpha smoothing is the key anti-patchiness step.
        raw_alpha = smooth_display(raw_alpha, DISPLAY_SMOOTH_SIGMA_SOURCE_ALPHA)

        alpha = np.where(
            np.isfinite(arr_display) & target_domain,
            target_feather * raw_alpha,
            0.0,
        )
        alpha = np.clip(alpha, 0.0, SOURCE_ALPHA_MAX_EXTREME)

        print('[PANEL A SOURCE DISPLAY]')
        print('  source value percentiles p5/p25/p50/p75/p95 = ' + ', '.join(f'{v:.3f}' for v in np.nanpercentile(arr_display[np.isfinite(arr_display)], [5,25,50,75,95])))
        print('  rank-field percentiles   p5/p25/p50/p75/p95 = ' + ', '.join(f'{v:.3f}' for v in np.nanpercentile(vals_rank, [5,25,50,75,95])))
        valid_alpha = alpha[np.isfinite(arr_display)]
        if valid_alpha.size > 0:
            print('  baked alpha percentiles p5/p25/p50/p75/p95 = ' + ', '.join(f'{v:.3f}' for v in np.nanpercentile(valid_alpha, [5,25,50,75,95])))
            print(f'  de-emphasized quantile band = {q_low:.2f}–{q_high:.2f}')
            print(f'  alpha smoothing sigma = {DISPLAY_SMOOTH_SIGMA_SOURCE_ALPHA:g}')
    else:
        alpha = np.zeros_like(arr_display, dtype=float)

    rgba = cmap(norm(np.nan_to_num(arr_color, nan=0.0)))
    rgba[..., 3] = alpha

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    return norm, rgba, None, mappable

def make_cumulative_display(
    arr: np.ndarray,
    norm_arr: np.ndarray,
    target_domain: np.ndarray,
    target_feather: np.ndarray,
    impervious_factor: np.ndarray,
):
    arr_target = np.where(target_domain, arr, np.nan)
    arr_display = smooth_display(arr_target, DISPLAY_SMOOTH_SIGMA_CURRENT)

    transformed = np.full_like(arr_display, np.nan, dtype=float)
    positive_mask = np.isfinite(arr_display) & (arr_display > 0)
    transformed[positive_mask] = np.log1p(arr_display[positive_mask])

    vals_t = transformed[np.isfinite(transformed)]
    if vals_t.size == 0:
        norm = Normalize(vmin=0.0, vmax=1.0)
        display = np.ma.masked_invalid(transformed)
    else:
        p_low, p_high = np.nanpercentile(vals_t, CURRENT_DISPLAY_PERCENTILES)
        if np.isclose(p_low, p_high):
            p_high = p_low + 1e-6

        scaled = (transformed - p_low) / (p_high - p_low)
        scaled = np.clip(scaled, 0.0, 1.0)

        norm = Normalize(vmin=0.0, vmax=1.0)
        display = np.ma.masked_invalid(scaled)

    norm_target = np.where(target_domain, norm_arr, np.nan)
    norm_smooth = smooth_display(norm_target, DISPLAY_SMOOTH_SIGMA_ALPHA)

    alpha = np.zeros_like(norm_smooth, dtype=float)
    valid = np.isfinite(norm_smooth) & np.isfinite(arr_display) & (arr_display > 0)

    rel = np.clip(norm_smooth, 0.0, 1.0)
    alpha[valid] = (
        ALPHA_MIN_IMPEDED
        + (ALPHA_MAX_CONNECTED - ALPHA_MIN_IMPEDED) * rel[valid]
    )

    alpha *= target_feather * impervious_factor

    return norm, display, alpha


def make_bottleneck_overlay(
    norm_arr: np.ndarray,
    target_domain: np.ndarray,
    target_feather: np.ndarray,
    impervious_factor: np.ndarray,
) -> np.ndarray | None:
    """
    Return a filled RGBA bottleneck overlay.

    Bottlenecks are derived from the upper tail of normalized current, but are
    then hard-clipped back to the same target/display domain as the cumulative
    current layer. This prevents smoothing from leaking bottleneck signal into
    buildings, roads, water, or outside-domain cells.

    No contour is drawn here. A contour layer is visually risky because contour
    coordinate orientation can diverge from imshow(..., origin="upper").
    """
    norm_target = np.where(target_domain, norm_arr, np.nan)
    valid = norm_target[np.isfinite(norm_target) & (norm_target > 0)]

    if valid.size == 0:
        return None

    threshold = float(np.nanpercentile(valid, BOTTLENECK_PERCENTILE))
    mask = np.isfinite(norm_target) & (norm_target >= threshold)

    if not np.any(mask):
        return None

    mask_display = gaussian_filter(
        mask.astype(float),
        DISPLAY_SMOOTH_SIGMA_BOTTLENECK,
    )

    # Critical fix: after smoothing, force the overlay back into the same hard
    # ecological target domain, then apply the same softened edge and impervious
    # transparency factor as the cumulative-current raster.
    mask_display = np.where(target_domain, mask_display, 0.0)
    mask_display *= target_feather * impervious_factor

    alpha = np.where(
        mask_display > BOTTLENECK_VISIBILITY_CUTOFF,
        np.clip(mask_display, 0.0, 1.0),
        0.0,
    )
    alpha = np.where(target_domain, alpha, 0.0)

    rgba = np.zeros((norm_arr.shape[0], norm_arr.shape[1], 4), dtype=float)
    r, g, b, _ = to_rgba(BOTTLENECK_COLOR)

    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = alpha * BOTTLENECK_ALPHA

    return rgba


def prepare_panel_data() -> list[dict]:
    all_panels = []
    reference_profile = None

    for spec in PANEL_SPECS:
        arr, profile = load_raster(spec.raster_path)

        if reference_profile is None:
            reference_profile = profile
        else:
            assert_same_grid(reference_profile, profile, spec.basename)

        all_panels.append(
            {
                "spec": spec,
                "array": arr,
                "profile": profile,
            }
        )

    norm_arr, norm_profile = load_raster(NORMALIZED_CURRENT_PATH)
    assert_same_grid(reference_profile, norm_profile, "normalized current")

    boundary_data = prepare_boundaries_and_crop(reference_profile)
    target_domain, target_feather, water_domain, impervious_factor = build_target_domain(
        reference_profile,
        boundary_data["peruspiiri_mask"],
    )

    bottleneck_overlay = make_bottleneck_overlay(
        norm_arr,
        target_domain,
        target_feather,
        impervious_factor,
    )
    water_background_rgba = make_water_background_rgba(water_domain)

    for item in all_panels:
        item.update(boundary_data)
        item["normalized_array"] = norm_arr
        item["target_domain"] = target_domain
        item["target_feather"] = target_feather
        item["impervious_factor"] = impervious_factor
        item["water_background_rgba"] = water_background_rgba
        item["bottleneck_overlay"] = bottleneck_overlay

    return all_panels


def add_colorbar_function_labels(cax, low_text: str | None, high_text: str | None) -> None:
    x_text = -0.65
    fs = max(7.0, STYLE.fs_legend * 0.90)

    if high_text:
        cax.text(
            x_text,
            1.00,
            high_text,
            transform=cax.transAxes,
            rotation=90,
            ha="right",
            va="top",
            fontsize=fs,
            fontfamily=FONT_FAMILY,
            color="black",
            alpha=0.90,
        )

    if low_text:
        cax.text(
            x_text,
            0.00,
            low_text,
            transform=cax.transAxes,
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=fs,
            fontfamily=FONT_FAMILY,
            color="black",
            alpha=0.90,
        )


def draw_map_panel(fig: plt.Figure, ax, cax, item: dict) -> None:
    spec = item["spec"]
    arr = item["array"]
    profile = item["profile"]
    norm_arr = item["normalized_array"]
    target_domain = item["target_domain"]
    target_feather = item["target_feather"]
    impervious_factor = item["impervious_factor"]
    water_background_rgba = item["water_background_rgba"]
    peruspiiri_plot = item["peruspiiri_plot"]
    outline_plot = item["outline_plot"]
    bottleneck_overlay = item["bottleneck_overlay"]

    extent = raster_extent(profile)

    cmap = plt.get_cmap(spec.cmap).copy()
    cmap.set_bad((1, 1, 1, 0))
    colorbar_mappable = None

    if spec.panel_kind == "source":
        norm, display, alpha, colorbar_mappable = make_source_display(
            arr,
            target_domain,
            target_feather,
            impervious_factor,
            cmap,
        )
    elif spec.panel_kind == "cumulative_integrated":
        norm, display, alpha = make_cumulative_display(
            arr,
            norm_arr,
            target_domain,
            target_feather,
            impervious_factor,
        )
    else:
        raise ValueError(f"Unknown panel kind: {spec.panel_kind!r}")

    if water_background_rgba is not None:
        ax.imshow(
            water_background_rgba,
            interpolation="nearest",
            extent=extent,
            origin="upper",
            zorder=1.2,
        )

    if spec.panel_kind == "source":
        im = ax.imshow(
            display,
            interpolation="bilinear",
            extent=extent,
            origin="upper",
            zorder=2,
        )
    else:
        im = ax.imshow(
            display,
            cmap=cmap,
            norm=norm,
            interpolation="bilinear",
            extent=extent,
            origin="upper",
            zorder=2,
            alpha=alpha,
        )

    if spec.panel_kind == "cumulative_integrated" and bottleneck_overlay is not None:
        ax.imshow(
            bottleneck_overlay,
            interpolation="bilinear",
            extent=extent,
            origin="upper",
            zorder=2.8,
        )

    if DRAW_INTERNAL_PERUSPIIRI:
        peruspiiri_plot.boundary.plot(
            ax=ax,
            color=INTERNAL_BOUNDARY_COLOR,
            linewidth=INTERNAL_BOUNDARY_WIDTH,
            alpha=INTERNAL_BOUNDARY_ALPHA,
            zorder=3,
        )

    if DRAW_OUTER_PERUSPIIRI:
        outline_plot.boundary.plot(
            ax=ax,
            color=OUTER_BOUNDARY_COLOR,
            linewidth=OUTER_BOUNDARY_WIDTH,
            alpha=OUTER_BOUNDARY_ALPHA,
            zorder=4,
        )

    raw_left = item["left"] - item["pad_x"]
    raw_right = item["right"] + item["pad_x"]
    raw_bottom = item["bottom"] - item["pad_y"]
    raw_top = item["top"] + item["pad_y"]

    target_aspect = axes_physical_aspect(fig, ax)
    plot_left, plot_right, plot_bottom, plot_top = bounds_to_match_axes_aspect(
        left=raw_left,
        right=raw_right,
        bottom=raw_bottom,
        top=raw_top,
        target_aspect=target_aspect,
        mode=MAP_FIT_MODE,
    )

    ax.set_xlim(plot_left, plot_right)
    ax.set_ylim(plot_bottom, plot_top)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    cbar = fig.colorbar(colorbar_mappable if colorbar_mappable is not None else im, cax=cax)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label(
        spec.colorbar_label,
        fontsize=STYLE.fs_axis,
        fontfamily=FONT_FAMILY,
        labelpad=24,
    )
    cbar.ax.tick_params(
        labelsize=STYLE.fs_tick,
        length=STYLE.tick_length,
        width=STYLE.tick_width,
    )

    if spec.fixed_ticks is not None:
        if len(spec.fixed_ticks) == 0:
            cbar.set_ticks([])
        else:
            cbar.set_ticks(list(spec.fixed_ticks))
            cbar.set_ticklabels([f"{t:g}" for t in spec.fixed_ticks])

    add_colorbar_function_labels(
        cax,
        low_text=spec.low_note,
        high_text=spec.high_note,
    )


def make_standalone_panel_exports(panel_data: list[dict]) -> None:
    for item in panel_data:
        spec = item["spec"]
        fig = new_panel_figure()
        ax = fig.add_axes(MAP_AXES_RECT)
        cax = fig.add_axes(MAP_COLORBAR_RECT)

        draw_map_panel(fig, ax, cax, item)

        print(
            f"[GEOM] {spec.panel_id}: "
            f"fig={fig.get_size_inches()} in, "
            f"ax={ax.get_position().bounds}, "
            f"cax={cax.get_position().bounds}"
        )

        save_single_panel(fig, spec.basename)
        plt.close(fig)


def make_composite_preview(panel_data: list[dict]) -> None:
    if not EXPORT_COMPOSITE_PREVIEW:
        return

    fig = plt.figure(
        figsize=(
            gps.cm_to_in(COMPOSITE_WIDTH_CM),
            gps.cm_to_in(COMPOSITE_HEIGHT_CM),
        )
    )
    fig.patch.set_alpha(0.0)

    panel_w = STYLE.panel_width_cm / COMPOSITE_WIDTH_CM
    panel_h = STYLE.panel_height_cm / COMPOSITE_HEIGHT_CM

    left = COMPOSITE_MARGIN_X_CM / COMPOSITE_WIDTH_CM
    bottom_top = (
        COMPOSITE_MARGIN_Y_CM + STYLE.panel_height_cm + COMPOSITE_GUTTER_Y_CM
    ) / COMPOSITE_HEIGHT_CM
    bottom_bottom = COMPOSITE_MARGIN_Y_CM / COMPOSITE_HEIGHT_CM

    panel_rects = [
        [left, bottom_top, panel_w, panel_h],
        [left, bottom_bottom, panel_w, panel_h],
    ]

    for item, panel_rect in zip(panel_data, panel_rects):
        ax = fig.add_axes(rect_within(panel_rect, MAP_AXES_RECT))
        cax = fig.add_axes(rect_within(panel_rect, MAP_COLORBAR_RECT))
        draw_map_panel(fig, ax, cax, item)

    composite_name = "f3_baseline_connectivity_two_panel_final_mask_rules_bottleneck_fix_preview"
    composite_save(fig, composite_name)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    check_required_files()
    panel_data = prepare_panel_data()

    print("Using input rasters:")
    for item in panel_data:
        spec = item["spec"]
        print(f"  {spec.panel_id}: {spec.raster_path}")
    print(f"  bottleneck threshold: P{BOTTLENECK_PERCENTILE} normalized current")
    print("  display domain: peruspiiri ∩ vegetation ∩ non-water ∩ non-building ∩ impervious<=0.5")

    make_standalone_panel_exports(panel_data)

    print("\nDone.")
    print("Standalone panel exports:")
    for spec in PANEL_SPECS:
        print(f"  {FIG_DIR / (spec.basename + '.svg')}")


if __name__ == "__main__":
    main()



#### make sure the clipping really works in panel b
#### add a legend for bottlenecks
#### nudge panel a