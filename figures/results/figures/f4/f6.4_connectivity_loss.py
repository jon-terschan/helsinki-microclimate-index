#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import importlib

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import rasterio
from matplotlib import font_manager
from matplotlib.transforms import Bbox
from matplotlib.colors import LinearSegmentedColormap
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter, label
from shapely.geometry import box

# =============================================================================
# GLOBAL STYLE / EXPORT LOGIC
# =============================================================================
# Shared settings file:
#   \\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures\global_plotting_settings.py
#
# This script does not define the main font sizes, rcParams, canvas size, or
# export formats locally. Those come from global_plotting_settings.STYLE.

GLOBAL_STYLE_DIR = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures"
)

if str(GLOBAL_STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(GLOBAL_STYLE_DIR))

try:
    import global_plotting_settings as gps  # noqa: E402
    gps = importlib.reload(gps)
except ImportError as e:
    raise ImportError(
        "Could not import global_plotting_settings.py from:\n"
        f"  {GLOBAL_STYLE_DIR}\n\n"
        "Check that the file exists and that the directory is readable."
    ) from e

if not hasattr(gps, "STYLE"):
    raise AttributeError(
        "global_plotting_settings.py must define STYLE = FigureStyle(...)."
    )

# Use the global style directly, but force this figure to export only PNG + SVG.
STYLE = replace(
    gps.STYLE,
    export_png=False,
    export_pdf=False,
    export_svg=True,
)

gps.apply_style(STYLE)

FONT_FAMILY = STYLE.font_family
CM_PER_INCH = gps.CM_PER_INCH

# =============================================================================
# PATHS
# =============================================================================

DATA = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
OMNI = DATA / "omniscape"

FUTURE_RUN_PARENT = OMNI / "output" / "future_condition_multiruns"
BASELINE_RUN_PARENT = OMNI / "output" / "conditional_multiruns"

TREE_PATH = DATA / "predictorstack" / "TREE_FRAC_10m.tif"
NWN_PATH = DATA / "predictorstack" / "NWN_FRAC_10m.tif"

WATER_PATH = DATA / "predictorstack" / "WATER_FRAC_10m_Helsinki.tif"
OCEAN_PATH = DATA / "predictorstack" / "OCEAN_FRAC_10m_Helsinki.tif"

BAREGROUND_VECTOR_PATH = DATA / "LULC" / "lc_bareground.gpkg"
PERUSPIIRI_PATH = DATA / "figures" / "offset_figure" / "peruspiiri_WFS.gpkg"

FIG_DIR = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures\f4"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = FIG_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_DIR = FIG_DIR / "landcover_weights"
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

RASTER_OUT_DIR = FIG_DIR / "rasters"
RASTER_OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Resolved global plotting style for f6.4 connectivity loss:")
print(f"  global file:  {Path(gps.__file__)}")
print(f"  output dir:   {FIG_DIR}")
print(f"  canvas:       {STYLE.panel_width_cm} × {STYLE.panel_height_cm} cm")
print(f"  dpi_export:   {STYLE.dpi_export}")
print(f"  fs_tick:      {STYLE.fs_tick}")
print(f"  fs_axis:      {STYLE.fs_axis}")
print(f"  fs_legend:    {STYLE.fs_legend}")
print(f"  fs_annotation:{STYLE.fs_annotation}")
print(f"  exports:      PNG={STYLE.export_png}, SVG={STYLE.export_svg}, PDF={STYLE.export_pdf}")

# =============================================================================
# COMMON SETTINGS
# =============================================================================

FLOW_RASTER_NAME = "cum_currmap.tif"

HEATWAVES = [
    "condition_heatwave_2010",
    "condition_heatwave_2018",
    "condition_heatwave_2021",
]

TOLERANCES = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

WRITE_WEIGHT_RASTERS = True
SAVE_CSV = True
SHOW_PREVIEW = True

RUN_LINE_FIGURE = False
RUN_MAP_FIGURE = False
RUN_COMPOSITE_FIGURE = True

def cm_to_in(value_cm: float) -> float:
    return gps.cm_to_in(value_cm)


# =============================================================================
# EXPORT / LAYOUT SETTINGS
# =============================================================================

COMPOSITE_FIG_BASENAME = "f4_composite_connectivity_loss_and_map"
COMPOSITE_LINE_PANEL_BASENAME = "f4_panel_a_connectivity_loss_aligned"
COMPOSITE_MAP_PANEL_BASENAME = "f4_panel_b_connectivity_loss_map_aligned"

COMPOSITE_EXPORT_PNG = False
COMPOSITE_EXPORT_PDF = False
COMPOSITE_EXPORT_SVG = True
COMPOSITE_TRANSPARENT_BACKGROUND = STYLE.transparent
COMPOSITE_DPI = STYLE.dpi_export

# Fixed external-compositing panel canvas.
# The two individually exported panels are exactly this physical size.
PANEL_WIDTH_CM = STYLE.panel_width_cm
PANEL_HEIGHT_CM = STYLE.panel_height_cm
PANEL_WIDTH_IN = cm_to_in(PANEL_WIDTH_CM)
PANEL_HEIGHT_IN = cm_to_in(PANEL_HEIGHT_CM)

# Build the composite from the same physical panel size.
# This is critical: if the composite panel rectangles are not physically
# 20 × 15 cm, Matplotlib's equal-aspect map will render at a different scale
# in the composite than in the standalone panel exports.
COMPOSITE_GUTTER_CM = 1.0
COMPOSITE_MARGIN_X_CM = 0.5
COMPOSITE_MARGIN_Y_CM = 0.5

COMPOSITE_WIDTH_CM = (
    2 * PANEL_WIDTH_CM
    + COMPOSITE_GUTTER_CM
    + 2 * COMPOSITE_MARGIN_X_CM
)
COMPOSITE_HEIGHT_CM = PANEL_HEIGHT_CM + 2 * COMPOSITE_MARGIN_Y_CM
COMPOSITE_WIDTH_IN = cm_to_in(COMPOSITE_WIDTH_CM)
COMPOSITE_HEIGHT_IN = cm_to_in(COMPOSITE_HEIGHT_CM)

# Never use tight bbox for aligned external-compositing exports.
EXPORT_USE_TIGHT_BBOX = STYLE.use_tight_bbox
EXPORT_PAD_INCHES = STYLE.pad_inches

# Panel containers in the two-panel composite, in figure-relative coordinates.
# These are calculated from centimeters so the composite uses the exact same
# physical panel size as the standalone exports.
COMPOSITE_LINE_PANEL_RECT = [
    COMPOSITE_MARGIN_X_CM / COMPOSITE_WIDTH_CM,
    COMPOSITE_MARGIN_Y_CM / COMPOSITE_HEIGHT_CM,
    PANEL_WIDTH_CM / COMPOSITE_WIDTH_CM,
    PANEL_HEIGHT_CM / COMPOSITE_HEIGHT_CM,
]
COMPOSITE_MAP_PANEL_RECT = [
    (COMPOSITE_MARGIN_X_CM + PANEL_WIDTH_CM + COMPOSITE_GUTTER_CM) / COMPOSITE_WIDTH_CM,
    COMPOSITE_MARGIN_Y_CM / COMPOSITE_HEIGHT_CM,
    PANEL_WIDTH_CM / COMPOSITE_WIDTH_CM,
    PANEL_HEIGHT_CM / COMPOSITE_HEIGHT_CM,
]

# Axes within a single fixed panel canvas.
# Tune these values, not the external SVG/PDF scale.
# Line panel plotting box.
PANEL_LINE_AXES_RECT = [0.15, 0.22, 0.78, 0.69]

# Background tolerance bands and labels below the x-axis.
# More negative values push the shaded band/labels farther below the x ticks.
LINE_ZONE_YMIN = -0.20
LINE_ZONE_LABEL_Y = -0.14
LINE_XLABEL_PAD = 19

# Map panel plotting box.
# Lower and slightly larger so the map bottom aligns with the lower edge of the
# tolerance-band visual area in the line panel.
PANEL_MAP_AXES_RECT = [0.13, 0.07, 0.84, 0.86]

# The colorbar is matched to the line-plot y-axis height.
# Colorbar is deliberately close to the map: same vertical span as the line
# y-axis, but just outside the map plotting box.
PANEL_MAP_COLORBAR_RECT = [
    PANEL_MAP_AXES_RECT[0] - 0.050,
    PANEL_MAP_AXES_RECT[1],
    0.030,
    PANEL_MAP_AXES_RECT[3],
]

# "crop" fills the matched plotting box without distorting the map.
# "pad" shows the full map extent but may leave empty space.
MAP_FIT_MODE = "crop"

# Optional title bands drawn in figure coordinates rather than ax.set_title().
DRAW_MAP_TITLE = True
DRAW_LINE_TITLE = False
PANEL_TITLE_Y = PANEL_MAP_AXES_RECT[1] + PANEL_MAP_AXES_RECT[3] + 0.008

# Optional visible border around the exported panel canvas.
# Useful for checking alignment in Illustrator/Inkscape/PowerPoint.
DRAW_PANEL_DEBUG_FRAME = False

# Legacy standalone names/settings kept for compatibility.
LINE_FIG_BASENAME = "f4_connectivity_loss_with_landcover_band"
MAP_FIG_BASENAME = "f4_map_p90_heatwave_connectivity_loss_pm1deg_vegetated_peruspiiri"

LINE_EXPORT_PNG = False
LINE_EXPORT_PDF = False
LINE_EXPORT_SVG = True
LINE_TRANSPARENT_BACKGROUND = STYLE.transparent
LINE_DPI = STYLE.dpi_export

MAP_EXPORT_PNG = False
MAP_EXPORT_PDF = False
MAP_EXPORT_SVG = True
MAP_TRANSPARENT_BACKGROUND = STYLE.transparent
MAP_DPI = STYLE.dpi_export

# =============================================================================
# LINE FIGURE FONT SIZES
# =============================================================================

# Keep axis/tick fonts global, but tune annotation fonts locally.
# The line-panel callouts were originally positioned for much smaller text;
# using STYLE.fs_annotation directly makes them dominate the panel.
FS_TICK = STYLE.fs_tick
FS_AXIS = STYLE.fs_axis
FS_TITLE = STYLE.fs_title
FS_ZONE = max(7.0, STYLE.fs_legend * 0.70)
FS_BOX = max(7.5, STYLE.fs_annotation * 0.58)
FS_SMALL_BOX = max(7.0, STYLE.fs_legend * 0.65)
FS_LINE_LABEL = max(8.0, STYLE.fs_annotation * 0.65)

# =============================================================================
# LINE FIGURE COLORS / STYLE
# =============================================================================

COL_BLUE = STYLE.col_blue
COL_BLUE_FILL = STYLE.col_blue_fill

COL_RED = STYLE.col_red
COL_RED_FILL = STYLE.col_red_fill

COL_HIST = STYLE.col_hist
COL_HIST_FILL = STYLE.col_hist_fill

COL_GREY = STYLE.col_grey
COL_GRID = STYLE.col_grid

COL_BLACK = STYLE.col_black

COL_ZONE_STRICT = "#F7F7F7"
COL_ZONE_MODERATE = "#EFEFEF"
COL_ZONE_RELAXED = "#E7E7E7"

AVG_TOTAL_STYLE = {"marker": "o", "linestyle": "-", "linewidth": 3.0}
P90_TOTAL_STYLE = {"marker": "s", "linestyle": "-", "linewidth": 3.0}

# =============================================================================
# MAP FIGURE SETTINGS
# =============================================================================

MAP_PRESENT_CONDITION = "condition_p90"
MAP_TOLERANCE = 1.0

BASELINE_EPSILON = 1e-9
PIXEL_LOSS_BASELINE_MIN_PERCENTILE = None

VEGETATION_MIN_FRACTION = 0.05
WATER_MIN_FRACTION = 0.05

BASELINE_DISPLAY_PERCENTILE_MASK = None

REMOVE_SMALL_DISPLAY_PATCHES = True
MIN_DISPLAY_PATCH_PIXELS = 20

SMOOTH_FOR_DISPLAY = True
DISPLAY_SMOOTH_SIGMA = 0.85

PLOT_INTERNAL_PERUSPIIRI = True
PLOT_OUTER_PERUSPIIRI = True

PAD_FRACTION = 0.02

LOSS_VMIN = 0
LOSS_VMAX = 100

# Loss colour scheme keyed to the line figure:
# low loss = pale/blue, middle = near-neutral, high loss = red.
LOSS_CMAP = LinearSegmentedColormap.from_list(
    "loss_cream_amber_burgundy",
    [
        "#fff7bc",
        "#fec44f",
        "#f16913",
        "#bd0026",
        "#67000d",
    ],
    N=256,
)

TARGET_DOMAIN_RGB = (0.90, 0.90, 0.90)
TARGET_DOMAIN_ALPHA = 1.00

LAND_OUT_OF_TARGET_RGB = (0.80, 0.80, 0.80)
LAND_OUT_OF_TARGET_ALPHA = 1.00

WATER_RGB = (0.94, 0.94, 0.94)
WATER_ALPHA = 1.00

BASELINE_UNDERLAY_ALPHA = 0.10
LOSS_ALPHA = 0.96

INTERNAL_BOUNDARY_COLOR = "black"
INTERNAL_BOUNDARY_WIDTH = 0.14
INTERNAL_BOUNDARY_ALPHA = 0.12

OUTER_BOUNDARY_COLOR = "black"
OUTER_BOUNDARY_WIDTH = 0.34
OUTER_BOUNDARY_ALPHA = 0.24

FS_MAP_TITLE = STYLE.fs_title

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class RasterRef:
    profile: dict
    transform: object
    crs: object
    width: int
    height: int
    nodata: float | int | None


# =============================================================================
# COMMON HELPERS
# =============================================================================

def tolerance_label(value: float) -> str:
    return str(value).replace(".", "p")


def baseline_run_name(present: str, tolerance: float) -> str:
    return f"{present}__pm{tolerance_label(tolerance)}deg"


def future_run_name(present: str, future: str, tolerance: float) -> str:
    return (
        f"futurecompare__"
        f"{present}__to__{future}__"
        f"pm{tolerance_label(tolerance)}deg"
    )


def find_run_dir(parent: Path, name: str) -> Path | None:
    direct = parent / name
    if direct.exists():
        return direct

    candidates = [
        p for p in parent.rglob("*")
        if p.is_dir() and name in p.name
    ]

    if not candidates:
        return None

    with_flow = [
        p for p in candidates
        if list(p.rglob(FLOW_RASTER_NAME))
    ]

    if with_flow:
        return sorted(with_flow, key=lambda p: len(str(p)))[0]

    return sorted(candidates, key=lambda p: len(str(p)))[0]


def find_file(root: Path, pattern: str) -> Path | None:
    matches = list(root.rglob(pattern))
    return matches[0] if matches else None


def configure_matplotlib(font_size: float | None = None, dpi: int | None = None) -> None:
    """Apply the shared global Matplotlib style.

    font_size is accepted for backward compatibility, but the figure should
    normally use the typed sizes in STYLE (fs_tick, fs_axis, etc.).
    """
    gps.apply_style(STYLE)
    updates = {
        "font.family": FONT_FAMILY,
        "font.sans-serif": [FONT_FAMILY],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    if font_size is not None:
        updates["font.size"] = font_size
    if dpi is not None:
        updates["savefig.dpi"] = dpi
    plt.rcParams.update(updates)


def export_figure(
    fig: plt.Figure,
    basename: str,
    *,
    export_png: bool,
    export_pdf: bool,
    export_svg: bool,
    dpi: int,
    transparent: bool,
    pad_inches: float = 0.0,
    use_tight_bbox: bool = False,
) -> None:
    """
    Export a fixed-canvas figure.

    For aligned panel exports, keep use_tight_bbox=False and pad_inches=0.
    """
    save_kwargs = {"transparent": transparent, "facecolor": "none", "edgecolor": "none"}

    if use_tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = pad_inches

    width_cm = fig.get_figwidth() * CM_PER_INCH
    height_cm = fig.get_figheight() * CM_PER_INCH

    if export_png:
        out_png = FIG_DIR / f"{basename}.png"
        fig.savefig(out_png, dpi=dpi, **save_kwargs)
        print(f"[OK] wrote {out_png} ({width_cm:.2f} × {height_cm:.2f} cm canvas)")

    if export_pdf:
        out_pdf = FIG_DIR / f"{basename}.pdf"
        fig.savefig(out_pdf, **save_kwargs)
        print(f"[OK] wrote {out_pdf} ({width_cm:.2f} × {height_cm:.2f} cm canvas)")

    if export_svg:
        out_svg = FIG_DIR / f"{basename}.svg"
        fig.savefig(out_svg, **save_kwargs)
        print(f"[OK] wrote {out_svg} ({width_cm:.2f} × {height_cm:.2f} cm canvas)")


def rect_within(panel_rect: list[float], inner_rect: list[float]) -> list[float]:
    """Convert an inner rectangle from panel-relative to figure-relative coordinates."""
    px, py, pw, ph = panel_rect
    ix, iy, iw, ih = inner_rect
    return [
        px + ix * pw,
        py + iy * ph,
        iw * pw,
        ih * ph,
    ]


def bounds_to_match_axes_aspect(
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    target_aspect: float,
    mode: str = "crop",
) -> tuple[float, float, float, float]:
    """
    Return map limits whose aspect matches the physical axes rectangle.

    mode="crop":
        Zooms/crops the map limits so the actual map fills the fixed axes box.
        This preserves geographic aspect and avoids dead space.

    mode="pad":
        Expands the limits so the whole map extent is visible. This also
        preserves geographic aspect, but may leave dead space around the map.
    """
    width = right - left
    height = top - bottom

    if width <= 0 or height <= 0:
        raise ValueError("Invalid map bounds.")

    data_aspect = width / height
    cx = 0.5 * (left + right)
    cy = 0.5 * (bottom + top)

    if mode == "crop":
        if data_aspect > target_aspect:
            # Data are wider than the axes: crop x-limits.
            new_height = height
            new_width = height * target_aspect
        else:
            # Data are taller/narrower than the axes: crop y-limits.
            new_width = width
            new_height = width / target_aspect
    elif mode == "pad":
        if data_aspect > target_aspect:
            # Data are wider than the axes: pad y-limits.
            new_width = width
            new_height = width / target_aspect
        else:
            # Data are taller/narrower than the axes: pad x-limits.
            new_height = height
            new_width = height * target_aspect
    else:
        raise ValueError(f"Unknown MAP_FIT_MODE: {mode!r}")

    new_left = cx - 0.5 * new_width
    new_right = cx + 0.5 * new_width
    new_bottom = cy - 0.5 * new_height
    new_top = cy + 0.5 * new_height

    return new_left, new_right, new_bottom, new_top

def axes_physical_aspect(fig: plt.Figure, ax) -> float:
    """Return physical width / height of an axes rectangle."""
    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()
    return (pos.width * fig_w) / (pos.height * fig_h)


def add_panel_debug_frame(fig: plt.Figure) -> None:
    """Draw a thin figure-border frame for checking external alignment."""
    if not DRAW_PANEL_DEBUG_FRAME:
        return

    ax_frame = fig.add_axes([0, 0, 1, 1], zorder=1000)
    ax_frame.patch.set_alpha(0)
    ax_frame.set_xlim(0, 1)
    ax_frame.set_ylim(0, 1)
    ax_frame.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="black", linewidth=0.5)
    ax_frame.axis("off")


def export_panel_crop(fig: plt.Figure, panel_rect: list[float], basename: str) -> None:
    """
    Export a fixed-size crop from a composite figure.

    panel_rect is [left, bottom, width, height] in figure-relative coordinates.
    bbox_inches must be physical inches, not display/pixel coordinates.
    """
    fig_w, fig_h = fig.get_size_inches()
    left, bottom, width, height = panel_rect

    bbox_inches = Bbox.from_bounds(
        left * fig_w,
        bottom * fig_h,
        width * fig_w,
        height * fig_h,
    )

    common = dict(
        bbox_inches=bbox_inches,
        pad_inches=0,
        transparent=COMPOSITE_TRANSPARENT_BACKGROUND,
        facecolor="none",
        edgecolor="none",
    )

    width_cm = width * fig_w * CM_PER_INCH
    height_cm = height * fig_h * CM_PER_INCH

    if COMPOSITE_EXPORT_PNG:
        out_png = FIG_DIR / f"{basename}.png"
        fig.savefig(out_png, dpi=COMPOSITE_DPI, **common)
        print(f"[OK] wrote {out_png} ({width_cm:.2f} × {height_cm:.2f} cm crop)")

    if COMPOSITE_EXPORT_PDF:
        out_pdf = FIG_DIR / f"{basename}.pdf"
        fig.savefig(out_pdf, **common)
        print(f"[OK] wrote {out_pdf} ({width_cm:.2f} × {height_cm:.2f} cm crop)")

    if COMPOSITE_EXPORT_SVG:
        out_svg = FIG_DIR / f"{basename}.svg"
        fig.savefig(out_svg, **common)
        print(f"[OK] wrote {out_svg} ({width_cm:.2f} × {height_cm:.2f} cm crop)")


# =============================================================================
# RASTER IO
# =============================================================================

def read_ref_raster(path: Path) -> tuple[np.ndarray, RasterRef]:
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


def load_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = src.profile.copy()

    return arr, profile


def read_flow_raster(path: Path) -> np.ndarray:
    arr, _profile = load_raster(path)
    return arr


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


def assert_same_ref(path: Path, ref: RasterRef) -> None:
    with rasterio.open(path) as src:
        checks = [
            ("width", ref.width, src.width),
            ("height", ref.height, src.height),
            ("transform", ref.transform, src.transform),
            ("crs", ref.crs, src.crs),
        ]
    for key, expected, observed in checks:
        if expected != observed:
            raise ValueError(f"Raster {key} mismatch for {path}: {observed} != {expected}")


def read_raster_checked(path: Path, ref: RasterRef) -> np.ndarray:
    assert_same_ref(path, ref)
    return read_flow_raster(path)


def write_geotiff(path: Path, arr: np.ndarray, ref_or_profile: RasterRef | dict) -> None:
    profile = (
        ref_or_profile.profile.copy()
        if isinstance(ref_or_profile, RasterRef)
        else ref_or_profile.copy()
    )
    profile.update(
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
    )

    out = np.where(np.isnan(arr), -9999.0, arr).astype(np.float32)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def raster_extent(profile: dict) -> tuple[float, float, float, float]:
    west, south, east, north = rasterio.transform.array_bounds(
        profile["height"],
        profile["width"],
        profile["transform"],
    )
    return west, east, south, north


# =============================================================================
# DOMAIN / MASK HELPERS
# =============================================================================

def rasterize_polygons_to_mask(gdf: gpd.GeoDataFrame, profile_or_ref: dict | RasterRef) -> np.ndarray:
    if isinstance(profile_or_ref, RasterRef):
        height = profile_or_ref.height
        width = profile_or_ref.width
        transform = profile_or_ref.transform
    else:
        height = profile_or_ref["height"]
        width = profile_or_ref["width"]
        transform = profile_or_ref["transform"]

    geoms = [
        geom for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    ]

    if not geoms:
        raise ValueError("No valid geometries for rasterization.")

    arr = rasterize(
        shapes=[(geom, 1) for geom in geoms],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype=np.uint8,
    )

    return arr.astype(bool)


def rasterize_bareground(path: Path, profile_or_ref: dict | RasterRef) -> np.ndarray:
    gpd.options.io_engine = "pyogrio"
    gdf = gpd.read_file(path)

    crs = profile_or_ref.crs if isinstance(profile_or_ref, RasterRef) else profile_or_ref["crs"]
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    geoms = [
        geom for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    ]

    if not geoms:
        raise ValueError(f"No valid geometries found in {path}")

    if isinstance(profile_or_ref, RasterRef):
        height = profile_or_ref.height
        width = profile_or_ref.width
        transform = profile_or_ref.transform
    else:
        height = profile_or_ref["height"]
        width = profile_or_ref["width"]
        transform = profile_or_ref["transform"]

    bare = rasterize(
        shapes=[(geom, 1.0) for geom in geoms],
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        all_touched=False,
        dtype=np.float32,
    )

    return np.clip(bare, 0.0, 1.0)


def load_predictor_on_grid(path: Path, profile: dict, label_name: str) -> np.ndarray:
    arr, arr_profile = load_raster(path)
    assert_same_grid(profile, arr_profile, label_name)
    return arr


def build_vegetation_domain(profile: dict) -> tuple[np.ndarray, np.ndarray]:
    tree = load_predictor_on_grid(TREE_PATH, profile, "TREE")
    nwn = load_predictor_on_grid(NWN_PATH, profile, "NWN")

    tree = np.clip(np.nan_to_num(tree, nan=0.0), 0.0, 1.0)
    nwn = np.clip(np.nan_to_num(nwn, nan=0.0), 0.0, 1.0)

    bareground = rasterize_bareground(BAREGROUND_VECTOR_PATH, profile)
    nwn_veg = np.where(bareground > 0, 0.0, nwn)
    nwn_veg = np.clip(nwn_veg, 0.0, 1.0)

    vegetation_fraction = np.clip(tree + nwn_veg, 0.0, 1.0)
    vegetation_domain = (
        np.isfinite(vegetation_fraction)
        & (vegetation_fraction >= VEGETATION_MIN_FRACTION)
    )

    return vegetation_domain, vegetation_fraction


def build_water_domain(profile: dict) -> np.ndarray:
    water_any = np.zeros((profile["height"], profile["width"]), dtype=bool)

    for path, label_name in [
        (WATER_PATH, "WATER"),
        (OCEAN_PATH, "OCEAN"),
    ]:
        if not path.exists():
            print(f"[WARN] missing {label_name} raster: {path}")
            continue

        arr = load_predictor_on_grid(path, profile, label_name)
        water_any |= np.nan_to_num(arr, nan=0.0) >= WATER_MIN_FRACTION

    return water_any


# =============================================================================
# NUMERIC HELPERS
# =============================================================================

def mean_rasters(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("No rasters supplied for averaging.")
    return np.nanmean(np.stack(arrays, axis=0), axis=0)


def total_positive_current_from_array(current: np.ndarray) -> float:
    current_clean = np.where(np.isfinite(current) & (current > 0), current, 0.0)
    return float(np.nansum(current_clean))


def total_positive_current(arr: np.ndarray, mask_arr: np.ndarray | None = None) -> float:
    if mask_arr is None:
        vals = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    else:
        vals = np.where(mask_arr & np.isfinite(arr) & (arr > 0), arr, 0.0)
    return float(np.nansum(vals))


def weighted_current_total(current: np.ndarray, weight: np.ndarray) -> float:
    current_clean = np.where(np.isfinite(current) & (current > 0), current, 0.0)
    weight_clean = np.where(np.isfinite(weight), weight, 0.0)
    return float(np.nansum(current_clean * weight_clean))


def log_scale01(arr: np.ndarray) -> np.ndarray:
    out = np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)
    out = np.log1p(out)

    vals = out[np.isfinite(out)]
    if vals.size == 0:
        return out

    p1, p99 = np.nanpercentile(vals, [1, 99])
    out = np.clip(out, p1, p99)

    if np.isclose(p1, p99):
        return np.zeros_like(out)

    return (out - p1) / (p99 - p1)


def smooth_for_display(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
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


def remove_small_patches(mask_arr: np.ndarray, min_pixels: int) -> np.ndarray:
    structure = np.ones((3, 3), dtype=int)
    labeled, n_labels = label(mask_arr, structure=structure)

    if n_labels == 0:
        return mask_arr

    counts = np.bincount(labeled.ravel())
    keep_labels = np.where(counts >= min_pixels)[0]
    keep_labels = keep_labels[keep_labels != 0]

    return np.isin(labeled, keep_labels)


def make_rgba(mask_arr: np.ndarray, rgb: tuple[float, float, float], alpha: float) -> np.ndarray:
    rgba = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=float)
    rgba[mask_arr, 0] = rgb[0]
    rgba[mask_arr, 1] = rgb[1]
    rgba[mask_arr, 2] = rgb[2]
    rgba[mask_arr, 3] = alpha
    return rgba


# =============================================================================
# LINE FIGURE DATA
# =============================================================================

def build_landcover_weights() -> tuple[dict[str, np.ndarray], RasterRef]:
    tree, ref = read_ref_raster(TREE_PATH)
    nwn, nwn_ref = read_ref_raster(NWN_PATH)

    if (
        nwn_ref.width != ref.width
        or nwn_ref.height != ref.height
        or nwn_ref.transform != ref.transform
        or nwn_ref.crs != ref.crs
    ):
        raise ValueError("TREE and NWN rasters are not on the same grid.")

    tree = np.clip(np.nan_to_num(tree, nan=0.0), 0.0, 1.0)
    nwn = np.clip(np.nan_to_num(nwn, nan=0.0), 0.0, 1.0)

    bareground = rasterize_bareground(BAREGROUND_VECTOR_PATH, ref)

    nwn_veg = np.where(bareground > 0, 0.0, nwn)
    nwn_veg = np.clip(nwn_veg, 0.0, 1.0)

    weights = {
        "tree": tree.astype(np.float32),
        "nwn_veg": nwn_veg.astype(np.float32),
        "bareground_binary": bareground.astype(np.float32),
    }

    if WRITE_WEIGHT_RASTERS:
        write_geotiff(WEIGHT_DIR / "tree_weight.tif", weights["tree"], ref)
        write_geotiff(WEIGHT_DIR / "nwn_vegetated_weight.tif", weights["nwn_veg"], ref)
        write_geotiff(WEIGHT_DIR / "bareground_binary.tif", weights["bareground_binary"], ref)

    return weights, ref


def get_current_raster(
    parent: Path,
    run_name: str,
    ref: RasterRef | None = None,
) -> np.ndarray | None:
    run_dir = find_run_dir(parent, run_name)

    if run_dir is None:
        print(f"    [MISSING RUN] {run_name}")
        return None

    flow_path = find_file(run_dir, FLOW_RASTER_NAME)

    if flow_path is None:
        print(f"    [MISSING FLOW] {run_name}")
        return None

    if ref is not None:
        return read_raster_checked(flow_path, ref)

    return read_flow_raster(flow_path)


def compute_overall_loss() -> pd.DataFrame:
    rows = []
    scenarios = [
        {"label": "Average → heatwave", "present": "condition_average"},
        {"label": "P90 → heatwave", "present": "condition_p90"},
    ]

    for scenario in scenarios:
        label_name = scenario["label"]
        present = scenario["present"]

        print(f"\n[OVERALL] {label_name}")

        for tol in TOLERANCES:
            print(f"  ±{tol:g}°C")

            base_name = baseline_run_name(present, tol)
            baseline_current = get_current_raster(BASELINE_RUN_PARENT, base_name)

            if baseline_current is None:
                print("    [SKIP] missing baseline")
                continue

            baseline_total = total_positive_current_from_array(baseline_current)

            if baseline_total <= 0:
                print("    [SKIP] baseline total <= 0")
                continue

            future_totals = []

            for heatwave in HEATWAVES:
                fut_name = future_run_name(present, heatwave, tol)
                future_current = get_current_raster(FUTURE_RUN_PARENT, fut_name)

                if future_current is None:
                    continue

                future_totals.append(total_positive_current_from_array(future_current))

            if not future_totals:
                print("    [SKIP] no future values")
                continue

            future_mean = float(np.mean(future_totals))
            retained_percent = 100.0 * future_mean / baseline_total
            loss_percent = 100.0 - retained_percent

            rows.append(
                {
                    "scenario": label_name,
                    "present": present,
                    "metric": "Overall",
                    "tolerance": tol,
                    "baseline_total": baseline_total,
                    "future_mean_total": future_mean,
                    "retained_percent": retained_percent,
                    "loss_percent": loss_percent,
                    "n_heatwaves": len(future_totals),
                }
            )

            print(
                f"    overall: baseline={baseline_total:.6g}, "
                f"future_mean={future_mean:.6g}, "
                f"loss={loss_percent:.3f}%, "
                f"n={len(future_totals)}"
            )

    return pd.DataFrame(rows)


def compute_p90_landcover_loss() -> pd.DataFrame:
    weights, ref = build_landcover_weights()

    rows = []
    present = "condition_p90"
    scenario_label = "P90 → heatwave"

    landcovers = [
        ("Tree cover", "tree"),
        ("Vegetated NWN", "nwn_veg"),
    ]

    print(f"\n[LAND-COVER] {scenario_label}")

    for tol in TOLERANCES:
        print(f"  ±{tol:g}°C")

        base_name = baseline_run_name(present, tol)
        baseline_current = get_current_raster(BASELINE_RUN_PARENT, base_name, ref=ref)

        if baseline_current is None:
            print("    [SKIP] missing baseline")
            continue

        baseline_totals = {
            label_name: weighted_current_total(baseline_current, weights[key])
            for label_name, key in landcovers
        }

        future_totals = {label_name: [] for label_name, _key in landcovers}

        for heatwave in HEATWAVES:
            fut_name = future_run_name(present, heatwave, tol)
            future_current = get_current_raster(FUTURE_RUN_PARENT, fut_name, ref=ref)

            if future_current is None:
                continue

            for label_name, key in landcovers:
                future_totals[label_name].append(
                    weighted_current_total(future_current, weights[key])
                )

        for label_name, _key in landcovers:
            vals = future_totals[label_name]
            baseline_total = baseline_totals[label_name]

            if not vals or baseline_total <= 0:
                continue

            future_mean = float(np.mean(vals))
            retained_percent = 100.0 * future_mean / baseline_total
            loss_percent = 100.0 - retained_percent

            rows.append(
                {
                    "scenario": scenario_label,
                    "present": present,
                    "metric": label_name,
                    "tolerance": tol,
                    "baseline_total": baseline_total,
                    "future_mean_total": future_mean,
                    "retained_percent": retained_percent,
                    "loss_percent": loss_percent,
                    "n_heatwaves": len(vals),
                }
            )

            print(
                f"    {label_name}: baseline={baseline_total:.6g}, "
                f"future_mean={future_mean:.6g}, "
                f"loss={loss_percent:.3f}%, "
                f"n={len(vals)}"
            )

    return pd.DataFrame(rows)


def make_landcover_band_table(overall_df: pd.DataFrame, lc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p90_overall = overall_df[
        (overall_df["scenario"] == "P90 → heatwave")
        & (overall_df["metric"] == "Overall")
    ].copy()

    for tol in TOLERANCES:
        overall_match = p90_overall[np.isclose(p90_overall["tolerance"], tol)]
        if overall_match.empty:
            continue

        lc_match = lc_df[np.isclose(lc_df["tolerance"], tol)]
        tree = lc_match[lc_match["metric"] == "Tree cover"]
        nwn = lc_match[lc_match["metric"] == "Vegetated NWN"]

        if tree.empty or nwn.empty:
            continue

        tree_loss = float(tree.iloc[0]["loss_percent"])
        nwn_loss = float(nwn.iloc[0]["loss_percent"])

        rows.append(
            {
                "tolerance": tol,
                "tree_loss_percent": tree_loss,
                "nwn_veg_loss_percent": nwn_loss,
                "band_low_percent": min(tree_loss, nwn_loss),
                "band_high_percent": max(tree_loss, nwn_loss),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# LINE PANEL DRAWING
# =============================================================================

def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(STYLE.axes_linewidth)
    ax.spines["bottom"].set_linewidth(STYLE.axes_linewidth)
    ax.grid(True, axis="y", color=COL_GRID, linewidth=STYLE.grid_linewidth, alpha=STYLE.grid_alpha)
    ax.grid(False, axis="x")
    ax.tick_params(
        axis="both",
        labelsize=FS_TICK,
        length=STYLE.tick_length,
        width=STYLE.tick_width,
    )


def format_tolerance_ticks(ax) -> None:
    ax.set_xticks(TOLERANCES)
    ax.set_xticklabels([f"{t:g}" for t in TOLERANCES], fontsize=FS_TICK)


def add_tolerance_zones(ax) -> None:
    zone_ymin = LINE_ZONE_YMIN

    ax.axvspan(0.05, 0.75, ymin=zone_ymin, ymax=1.0, color=COL_ZONE_STRICT, zorder=0, clip_on=False)
    ax.axvspan(0.75, 2.5, ymin=zone_ymin, ymax=1.0, color=COL_ZONE_MODERATE, zorder=0, clip_on=False)
    ax.axvspan(2.5, 5.1, ymin=zone_ymin, ymax=1.0, color=COL_ZONE_RELAXED, zorder=0, clip_on=False)

    zone_label_y = LINE_ZONE_LABEL_Y

    for x, label_name in [(0.40, "strict"), (1.60, "moderate"), (3.80, "relaxed")]:
        ax.text(
            x,
            zone_label_y,
            label_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=FS_ZONE,
            color=COL_BLACK,
            alpha=0.95,
            zorder=20,
            clip_on=False,
        )


def draw_line_panel(ax, overall_df: pd.DataFrame, band_df: pd.DataFrame) -> None:
    """Draw the connectivity-loss line panel onto an existing fixed axis."""
    add_tolerance_zones(ax)

    avg = overall_df[
        (overall_df["scenario"] == "Average → heatwave")
        & (overall_df["metric"] == "Overall")
    ].sort_values("tolerance")

    if not avg.empty:
        ax.plot(
            avg["tolerance"].to_numpy(dtype=float),
            avg["loss_percent"].to_numpy(dtype=float),
            color=COL_BLUE,
            marker=AVG_TOTAL_STYLE["marker"],
            linestyle=AVG_TOTAL_STYLE["linestyle"],
            linewidth=AVG_TOTAL_STYLE["linewidth"],
            markersize=7.8,
            zorder=6,
        )

    p90 = overall_df[
        (overall_df["scenario"] == "P90 → heatwave")
        & (overall_df["metric"] == "Overall")
    ].sort_values("tolerance")

    if not p90.empty:
        ax.plot(
            p90["tolerance"].to_numpy(dtype=float),
            p90["loss_percent"].to_numpy(dtype=float),
            color=COL_RED,
            marker=P90_TOTAL_STYLE["marker"],
            linestyle=P90_TOTAL_STYLE["linestyle"],
            linewidth=P90_TOTAL_STYLE["linewidth"],
            markersize=7.4,
            zorder=7,
        )

    if not band_df.empty:
        band_df = band_df.sort_values("tolerance").copy()

        x_band = band_df["tolerance"].to_numpy(dtype=float)
        y_tree = band_df["tree_loss_percent"].to_numpy(dtype=float)
        y_nwn = band_df["nwn_veg_loss_percent"].to_numpy(dtype=float)
        y_low = band_df["band_low_percent"].to_numpy(dtype=float)
        y_high = band_df["band_high_percent"].to_numpy(dtype=float)

        ax.fill_between(x_band, y_low, y_high, color=COL_RED_FILL, alpha=0.28, zorder=2, linewidth=0)
        ax.plot(x_band, y_tree, color=COL_RED, linewidth=1.4, alpha=0.45, zorder=3)
        ax.plot(x_band, y_nwn, color=COL_RED, linewidth=1.4, alpha=0.45, linestyle="--", zorder=3)

        tree_anchor_idx = int(np.argmin(np.abs(x_band - 1.0)))
        nwn_anchor_idx = int(np.argmin(np.abs(x_band - 0.5)))
        tree_box_xy = (1.33, 25)
        nwn_box_xy = (0.7, 4.3)
        connector_style = dict(arrowstyle="-", color=COL_GREY, alpha=0.22, linewidth=1.0)

        ax.annotate("", xy=(float(x_band[tree_anchor_idx]), float(y_tree[tree_anchor_idx])), xytext=tree_box_xy, arrowprops=connector_style, zorder=4)
        ax.annotate("", xy=(float(x_band[nwn_anchor_idx]), float(y_nwn[nwn_anchor_idx])), xytext=nwn_box_xy, arrowprops=connector_style, zorder=4)

        ax.text(
            tree_box_xy[0],
            tree_box_xy[1],
            "trees",
            color=COL_RED,
            fontsize=FS_SMALL_BOX,
            fontfamily=FONT_FAMILY,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=COL_RED_FILL, linewidth=1.0, alpha=0.88),
            zorder=8,
        )
        ax.text(
            nwn_box_xy[0],
            nwn_box_xy[1],
            "other vegetation",
            color=COL_RED,
            fontsize=FS_SMALL_BOX,
            fontfamily=FONT_FAMILY,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=COL_RED_FILL, linewidth=1.0, alpha=0.88),
            zorder=8,
        )

    ax.text(
        3.8,
        62,
        "During heatwaves,\nno analogous climates to\naverage conditions exist.",
        color=COL_BLUE,
        fontsize=FS_BOX,
        fontfamily=FONT_FAMILY,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=COL_BLUE_FILL, linewidth=1.6, alpha=0.95),
        zorder=9,
    )

    ax.text(
        3.8,
        25,
        "Tree-covered areas\nexperience higher relative\nconnectivity loss than\nother vegetated areas.",
        color=COL_RED,
        fontsize=FS_BOX,
        fontfamily=FONT_FAMILY,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=COL_RED_FILL, linewidth=1.6, alpha=0.95),
        zorder=9,
    )

    ax.text(
        0.12,
        90,
        "from mean baseline",
        color=COL_BLUE,
        fontsize=FS_LINE_LABEL,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.72),
        zorder=10,
    )

    ax.text(
        0.12,
        37,
        "from p90 baseline",
        color=COL_RED,
        fontsize=FS_LINE_LABEL,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.72),
        zorder=10,
    )

    ax.axhline(0.0, color=COL_GREY, linestyle="--", linewidth=1.0, alpha=0.55, zorder=1)
    ax.set_xlabel("Thermal tolerance (±°C)", fontsize=FS_AXIS, fontfamily=FONT_FAMILY, labelpad=LINE_XLABEL_PAD)
    ax.set_ylabel("Connectivity loss (%)", fontsize=FS_AXIS, fontfamily=FONT_FAMILY)
    ax.set_ylim(0, 105)
    ax.set_xlim(0.0, 5.35)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    format_tolerance_ticks(ax)
    clean_axes(ax)


# =============================================================================
# MAP PANEL DATA / DRAWING
# =============================================================================

def make_display_mask(
    baseline: np.ndarray,
    loss_raw: np.ndarray,
    target_domain: np.ndarray,
    percentile: float | None,
) -> np.ndarray:
    valid = (
        target_domain
        & np.isfinite(loss_raw)
        & np.isfinite(baseline)
        & (baseline > 0)
    )

    if percentile is None:
        mask_arr = valid
    else:
        vals = baseline[valid]
        if vals.size == 0:
            mask_arr = valid
        else:
            threshold = np.nanpercentile(vals, percentile)
            mask_arr = valid & (baseline >= threshold)

    if REMOVE_SMALL_DISPLAY_PATCHES:
        mask_arr = remove_small_patches(mask_arr, min_pixels=MIN_DISPLAY_PATCH_PIXELS)

    return mask_arr


def prepare_map_panel_data() -> dict:
    """Prepare all raster/vector layers needed for the map panel."""
    tol_label = tolerance_label(MAP_TOLERANCE)

    baseline_name = baseline_run_name(
        present=MAP_PRESENT_CONDITION,
        tolerance=MAP_TOLERANCE,
    )
    baseline_run_dir = find_run_dir(BASELINE_RUN_PARENT, baseline_name)

    if baseline_run_dir is None:
        raise FileNotFoundError(f"Missing baseline run: {baseline_name}")

    baseline_flow_path = find_file(baseline_run_dir, FLOW_RASTER_NAME)

    if baseline_flow_path is None:
        raise FileNotFoundError(f"Missing {FLOW_RASTER_NAME} in {baseline_run_dir}")

    baseline, profile = load_raster(baseline_flow_path)

    print("[BASELINE]")
    print(f"  run  = {baseline_name}")
    print(f"  file = {baseline_flow_path}")

    peruspiiri = gpd.read_file(PERUSPIIRI_PATH).to_crs(profile["crs"])
    helsinki_outline = peruspiiri.dissolve()
    peruspiiri_mask = rasterize_polygons_to_mask(peruspiiri, profile)

    future_arrays: list[np.ndarray] = []
    used_heatwaves: list[str] = []

    print("\n[FUTURE HEATWAVE RUNS]")

    for heatwave in HEATWAVES:
        name = future_run_name(
            present=MAP_PRESENT_CONDITION,
            future=heatwave,
            tolerance=MAP_TOLERANCE,
        )

        run_dir = find_run_dir(FUTURE_RUN_PARENT, name)

        if run_dir is None:
            print(f"  [MISSING RUN] {name}")
            continue

        flow_path = find_file(run_dir, FLOW_RASTER_NAME)

        if flow_path is None:
            print(f"  [MISSING FLOW] {name}")
            continue

        arr, arr_profile = load_raster(flow_path)
        assert_same_grid(profile, arr_profile, label_name=name)

        future_arrays.append(arr)
        used_heatwaves.append(heatwave)

        print(f"  [OK] {heatwave}: {flow_path}")

    if not future_arrays:
        raise RuntimeError("No future heatwave rasters found.")

    heatwave_mean = mean_rasters(future_arrays)

    vegetation_domain, _vegetation_fraction = build_vegetation_domain(profile)
    water_mask = build_water_domain(profile)

    target_domain = peruspiiri_mask & vegetation_domain & (~water_mask)
    water_domain = peruspiiri_mask & water_mask
    out_of_target_domain = peruspiiri_mask & (~target_domain) & (~water_domain)

    print("\n[DISPLAY DOMAIN]")
    print(f"  peruspiiri cells       = {int(np.sum(peruspiiri_mask))}")
    print(f"  target vegetated cells = {int(np.sum(target_domain))}")
    print(f"  water cells            = {int(np.sum(water_domain))}")
    print(f"  out-of-target cells    = {int(np.sum(out_of_target_domain))}")

    raw_valid = (
        target_domain
        & np.isfinite(baseline)
        & np.isfinite(heatwave_mean)
        & (baseline > BASELINE_EPSILON)
    )

    if PIXEL_LOSS_BASELINE_MIN_PERCENTILE is not None:
        denom_vals = baseline[raw_valid]
        if denom_vals.size > 0:
            denom_threshold = np.nanpercentile(denom_vals, PIXEL_LOSS_BASELINE_MIN_PERCENTILE)
            raw_valid = raw_valid & (baseline >= denom_threshold)
            print(
                f"[PIXEL LOSS FILTER] baseline denominator >= "
                f"p{PIXEL_LOSS_BASELINE_MIN_PERCENTILE:g} "
                f"({denom_threshold:.6g})"
            )

    loss_percent_raw = np.full_like(baseline, np.nan, dtype=np.float64)
    loss_percent_raw[raw_valid] = (
        100.0
        * (baseline[raw_valid] - heatwave_mean[raw_valid])
        / baseline[raw_valid]
    )
    loss_percent_clipped = np.clip(loss_percent_raw, 0.0, 100.0)

    baseline_total = total_positive_current(baseline, target_domain)
    future_total = total_positive_current(heatwave_mean, target_domain)
    global_loss = (
        100.0 * (baseline_total - future_total) / baseline_total
        if baseline_total > 0
        else np.nan
    )

    map_vals = loss_percent_raw[np.isfinite(loss_percent_raw)]

    print("\n[LOSS CHECK]")
    print(f"  global target-domain total loss = {global_loss:.3f}%")
    print(f"  baseline total current          = {baseline_total:.6g}")
    print(f"  future mean total current       = {future_total:.6g}")

    if map_vals.size > 0:
        print(f"  pixel-wise mean loss            = {np.nanmean(map_vals):.3f}%")
        print(f"  pixel-wise median loss          = {np.nanmedian(map_vals):.3f}%")
        print(f"  pixel-wise p90 loss             = {np.nanpercentile(map_vals, 90):.3f}%")
        print(f"  pixel-wise max loss             = {np.nanmax(map_vals):.3f}%")
        print("  note: global total loss and pixel-wise % loss are not expected to match.")

    raw_tif = RASTER_OUT_DIR / f"p90_to_heatwave_mean_connectivity_loss_raw_pm{tol_label}deg.tif"
    clipped_tif = RASTER_OUT_DIR / f"p90_to_heatwave_mean_connectivity_loss_clipped_0_100_pm{tol_label}deg.tif"
    target_domain_tif = RASTER_OUT_DIR / f"p90_loss_target_domain_tree_veg_nwn_pm{tol_label}deg.tif"

    write_geotiff(raw_tif, loss_percent_raw, profile)
    write_geotiff(clipped_tif, loss_percent_clipped, profile)
    write_geotiff(target_domain_tif, target_domain.astype(float), profile)

    print(f"\n[OK] wrote {raw_tif}")
    print(f"[OK] wrote {clipped_tif}")
    print(f"[OK] wrote {target_domain_tif}")

    display_mask = make_display_mask(
        baseline=baseline,
        loss_raw=loss_percent_clipped,
        target_domain=target_domain,
        percentile=BASELINE_DISPLAY_PERCENTILE_MASK,
    )

    loss_display = np.where(display_mask, loss_percent_clipped, np.nan)

    if SMOOTH_FOR_DISPLAY:
        loss_display = smooth_for_display(loss_display, sigma=DISPLAY_SMOOTH_SIGMA)
        loss_display = np.where(target_domain, loss_display, np.nan)

    loss_display = np.ma.masked_invalid(loss_display)

    baseline_underlay = np.where(target_domain, baseline, np.nan)
    baseline_underlay = np.ma.masked_invalid(log_scale01(baseline_underlay))

    target_domain_rgba = make_rgba(target_domain, TARGET_DOMAIN_RGB, TARGET_DOMAIN_ALPHA)
    water_rgba = make_rgba(water_domain, WATER_RGB, WATER_ALPHA)
    out_of_target_rgba = make_rgba(out_of_target_domain, LAND_OUT_OF_TARGET_RGB, LAND_OUT_OF_TARGET_ALPHA)

    loss_cmap = LOSS_CMAP.copy()
    loss_cmap.set_bad((1, 1, 1, 0))

    baseline_cmap = plt.get_cmap("Greys").copy()
    baseline_cmap.set_bad((1, 1, 1, 0))

    extent = raster_extent(profile)

    left, bottom, right, top = helsinki_outline.total_bounds
    raster_left, raster_right, raster_bottom, raster_top = extent

    left = max(left, raster_left)
    right = min(right, raster_right)
    bottom = max(bottom, raster_bottom)
    top = min(top, raster_top)

    raster_box = gpd.GeoDataFrame(
        geometry=[box(left, bottom, right, top)],
        crs=profile["crs"],
    )

    peruspiiri_plot = gpd.clip(peruspiiri, raster_box)
    outline_plot = peruspiiri_plot.dissolve()

    pad_x = (right - left) * PAD_FRACTION
    pad_y = (top - bottom) * PAD_FRACTION

    mask_note = (
        "all vegetated baseline-connected cells"
        if BASELINE_DISPLAY_PERCENTILE_MASK is None
        else f"baseline current ≥ p{BASELINE_DISPLAY_PERCENTILE_MASK:g}"
    )

    return {
        "out_of_target_rgba": out_of_target_rgba,
        "water_rgba": water_rgba,
        "target_domain_rgba": target_domain_rgba,
        "baseline_underlay": baseline_underlay,
        "baseline_cmap": baseline_cmap,
        "loss_display": loss_display,
        "loss_cmap": loss_cmap,
        "extent": extent,
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "peruspiiri_plot": peruspiiri_plot,
        "outline_plot": outline_plot,
        "used_heatwaves": used_heatwaves,
        "mask_note": mask_note,
    }


def draw_map_panel(fig: plt.Figure, ax, cax, map_data: dict) -> None:
    """Draw the map panel onto existing axes and colorbar axes."""
    ax.imshow(
        map_data["out_of_target_rgba"],
        extent=map_data["extent"],
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )
    ax.imshow(
        map_data["water_rgba"],
        extent=map_data["extent"],
        origin="upper",
        interpolation="nearest",
        zorder=2,
    )
    ax.imshow(
        map_data["target_domain_rgba"],
        extent=map_data["extent"],
        origin="upper",
        interpolation="nearest",
        zorder=3,
    )
    ax.imshow(
        map_data["baseline_underlay"],
        cmap=map_data["baseline_cmap"],
        vmin=0,
        vmax=1,
        alpha=BASELINE_UNDERLAY_ALPHA,
        interpolation="bilinear",
        extent=map_data["extent"],
        origin="upper",
        zorder=4,
    )

    im = ax.imshow(
        map_data["loss_display"],
        cmap=map_data["loss_cmap"],
        vmin=LOSS_VMIN,
        vmax=LOSS_VMAX,
        alpha=LOSS_ALPHA,
        interpolation="bilinear",
        extent=map_data["extent"],
        origin="upper",
        zorder=5,
    )

    if PLOT_INTERNAL_PERUSPIIRI:
        map_data["peruspiiri_plot"].boundary.plot(
            ax=ax,
            color=INTERNAL_BOUNDARY_COLOR,
            linewidth=INTERNAL_BOUNDARY_WIDTH,
            alpha=INTERNAL_BOUNDARY_ALPHA,
            zorder=6,
        )

    if PLOT_OUTER_PERUSPIIRI:
        map_data["outline_plot"].boundary.plot(
            ax=ax,
            color=OUTER_BOUNDARY_COLOR,
            linewidth=OUTER_BOUNDARY_WIDTH,
            alpha=OUTER_BOUNDARY_ALPHA,
            zorder=7,
        )

    raw_left = map_data["left"] - map_data["pad_x"]
    raw_right = map_data["right"] + map_data["pad_x"]
    raw_bottom = map_data["bottom"] - map_data["pad_y"]
    raw_top = map_data["top"] + map_data["pad_y"]

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

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_ticks_position("left")
    # Put the label on the right side of the colorbar so it sits between the
    # legend and the map instead of being clipped outside the figure canvas.
    cbar.ax.yaxis.set_label_position("right")
    cbar.set_label("Connectivity loss (%)", fontsize=FS_AXIS, fontfamily=FONT_FAMILY, labelpad=8)
    cbar.ax.tick_params(labelsize=FS_TICK, length=STYLE.tick_length, width=STYLE.tick_width)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))


def add_map_title(fig: plt.Figure) -> None:
    if not DRAW_MAP_TITLE:
        return
    fig.text(
        PANEL_MAP_AXES_RECT[0] + 0.5 * PANEL_MAP_AXES_RECT[2],
        PANEL_TITLE_Y,
        "from p90 baseline, ±1°C tolerance",
        ha="center",
        va="bottom",
        fontsize=FS_MAP_TITLE,
        fontfamily=FONT_FAMILY,
    )


def add_line_title(fig: plt.Figure) -> None:
    if not DRAW_LINE_TITLE:
        return
    fig.text(
        0.50,
        PANEL_TITLE_Y,
        "Connectivity loss across thermal tolerance",
        ha="center",
        va="top",
        fontsize=FS_TITLE,
        fontfamily=FONT_FAMILY,
    )


# =============================================================================
# EXPORT ENTRY POINTS
# =============================================================================

def make_aligned_line_panel_export(
    overall_df: pd.DataFrame,
    band_df: pd.DataFrame,
) -> None:
    """
    Export line panel as a fixed-size standalone canvas.

    This is the preferred output for external compositing.
    """
    configure_matplotlib(dpi=COMPOSITE_DPI)

    fig = plt.figure(figsize=(PANEL_WIDTH_IN, PANEL_HEIGHT_IN))
    fig.patch.set_alpha(0.0)

    ax = fig.add_axes(PANEL_LINE_AXES_RECT)
    draw_line_panel(ax, overall_df, band_df)
    add_line_title(fig)
    add_panel_debug_frame(fig)

    export_figure(
        fig,
        COMPOSITE_LINE_PANEL_BASENAME,
        export_png=COMPOSITE_EXPORT_PNG,
        export_pdf=COMPOSITE_EXPORT_PDF,
        export_svg=COMPOSITE_EXPORT_SVG,
        dpi=COMPOSITE_DPI,
        transparent=COMPOSITE_TRANSPARENT_BACKGROUND,
        pad_inches=0,
        use_tight_bbox=False,
    )

    plt.close(fig)


def make_aligned_map_panel_export(map_data: dict) -> None:
    """
    Export map panel as a fixed-size standalone canvas.

    This is the preferred output for external compositing.
    """
    configure_matplotlib(font_size=FS_AXIS, dpi=COMPOSITE_DPI)
    plt.rcParams.update({"axes.linewidth": 0.4})

    fig = plt.figure(figsize=(PANEL_WIDTH_IN, PANEL_HEIGHT_IN))
    fig.patch.set_alpha(0.0)

    ax = fig.add_axes(PANEL_MAP_AXES_RECT)
    cax = fig.add_axes(PANEL_MAP_COLORBAR_RECT)

    draw_map_panel(fig, ax, cax, map_data)
    add_map_title(fig)
    add_panel_debug_frame(fig)

    export_figure(
        fig,
        COMPOSITE_MAP_PANEL_BASENAME,
        export_png=COMPOSITE_EXPORT_PNG,
        export_pdf=COMPOSITE_EXPORT_PDF,
        export_svg=COMPOSITE_EXPORT_SVG,
        dpi=COMPOSITE_DPI,
        transparent=COMPOSITE_TRANSPARENT_BACKGROUND,
        pad_inches=0,
        use_tight_bbox=False,
    )

    plt.close(fig)


def make_composite_figure() -> None:
    """Create aligned individual panel exports and a two-panel composite."""
    configure_matplotlib(dpi=COMPOSITE_DPI)

    overall_df = compute_overall_loss()
    lc_df = compute_p90_landcover_loss()
    band_df = make_landcover_band_table(overall_df, lc_df)

    if SAVE_CSV:
        overall_csv = TABLE_DIR / "f4_overall_connectivity_loss.csv"
        lc_csv = TABLE_DIR / "f4_p90_landcover_weighted_connectivity_loss.csv"
        band_csv = TABLE_DIR / "f4_p90_landcover_band.csv"

        overall_df.to_csv(overall_csv, index=False)
        lc_df.to_csv(lc_csv, index=False)
        band_df.to_csv(band_csv, index=False)

        print(f"\n[OK] wrote {overall_csv}")
        print(f"[OK] wrote {lc_csv}")
        print(f"[OK] wrote {band_csv}")

    map_data = prepare_map_panel_data()

    # Preferred outputs for external compositing.
    make_aligned_line_panel_export(overall_df, band_df)
    make_aligned_map_panel_export(map_data)

    # Optional full composite for checking the joint layout.
    fig = plt.figure(figsize=(COMPOSITE_WIDTH_IN, COMPOSITE_HEIGHT_IN))
    fig.patch.set_alpha(0.0)

    line_ax = fig.add_axes(rect_within(COMPOSITE_LINE_PANEL_RECT, PANEL_LINE_AXES_RECT))
    map_ax = fig.add_axes(rect_within(COMPOSITE_MAP_PANEL_RECT, PANEL_MAP_AXES_RECT))
    map_cax = fig.add_axes(rect_within(COMPOSITE_MAP_PANEL_RECT, PANEL_MAP_COLORBAR_RECT))

    draw_line_panel(line_ax, overall_df, band_df)
    draw_map_panel(fig, map_ax, map_cax, map_data)

    # Composite titles are placed relative to the composite panel containers.
    if DRAW_LINE_TITLE:
        x = COMPOSITE_LINE_PANEL_RECT[0] + 0.5 * COMPOSITE_LINE_PANEL_RECT[2]
        y = COMPOSITE_LINE_PANEL_RECT[1] + PANEL_TITLE_Y * COMPOSITE_LINE_PANEL_RECT[3]
        fig.text(
            x,
            y,
            "Connectivity loss across thermal tolerance",
            ha="center",
            va="top",
            fontsize=FS_TITLE,
            fontfamily=FONT_FAMILY,
        )

    if DRAW_MAP_TITLE:
        map_title_x = COMPOSITE_MAP_PANEL_RECT[0] + (
            PANEL_MAP_AXES_RECT[0] + 0.5 * PANEL_MAP_AXES_RECT[2]
        ) * COMPOSITE_MAP_PANEL_RECT[2]
        map_title_y = COMPOSITE_MAP_PANEL_RECT[1] + PANEL_TITLE_Y * COMPOSITE_MAP_PANEL_RECT[3]
        fig.text(
            map_title_x,
            map_title_y,
            "p90 baseline, ±1°C tolerance",
            ha="center",
            va="bottom",
            fontsize=FS_MAP_TITLE,
            fontfamily=FONT_FAMILY,
        )

    export_figure(
        fig,
        COMPOSITE_FIG_BASENAME,
        export_png=COMPOSITE_EXPORT_PNG,
        export_pdf=COMPOSITE_EXPORT_PDF,
        export_svg=COMPOSITE_EXPORT_SVG,
        dpi=COMPOSITE_DPI,
        transparent=COMPOSITE_TRANSPARENT_BACKGROUND,
        pad_inches=0,
        use_tight_bbox=False,
    )

    if SHOW_PREVIEW:
        plt.show()
    else:
        plt.close(fig)

    print(f"[OK] heatwaves used: {', '.join(map_data['used_heatwaves'])}")
    print(f"[OK] display mask: {map_data['mask_note']}")


def make_line_figure() -> None:
    """
    Legacy line-only entry point.

    Uses the same fixed panel canvas as the aligned export.
    """
    overall_df = compute_overall_loss()
    lc_df = compute_p90_landcover_loss()
    band_df = make_landcover_band_table(overall_df, lc_df)

    if SAVE_CSV:
        overall_csv = TABLE_DIR / "f4_overall_connectivity_loss.csv"
        lc_csv = TABLE_DIR / "f4_p90_landcover_weighted_connectivity_loss.csv"
        band_csv = TABLE_DIR / "f4_p90_landcover_band.csv"

        overall_df.to_csv(overall_csv, index=False)
        lc_df.to_csv(lc_csv, index=False)
        band_df.to_csv(band_csv, index=False)

        print(f"\n[OK] wrote {overall_csv}")
        print(f"[OK] wrote {lc_csv}")
        print(f"[OK] wrote {band_csv}")

    make_aligned_line_panel_export(overall_df, band_df)


def make_map_figure() -> None:
    """
    Legacy map-only entry point.

    Uses the same fixed panel canvas as the aligned export.
    """
    map_data = prepare_map_panel_data()
    make_aligned_map_panel_export(map_data)

    print(f"[OK] heatwaves used: {', '.join(map_data['used_heatwaves'])}")
    print(f"[OK] display mask: {map_data['mask_note']}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if RUN_COMPOSITE_FIGURE:
        make_composite_figure()

    if RUN_LINE_FIGURE:
        make_line_figure()

    if RUN_MAP_FIGURE:
        make_map_figure()


if __name__ == "__main__":
    main()
