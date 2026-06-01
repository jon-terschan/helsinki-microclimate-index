import pyvista as pv
import rasterio
from rasterio.enums import Resampling
import numpy as np
import imageio
from scipy.ndimage import gaussian_filter
import os
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# USER SETTINGS
# =============================================================================

PRED_DIR = "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictions/testday/"
DTM_PATH = "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/predictorstack/DTM_10m_Helsinki.tif"
DATE_STR = "20240601"
OUT_GIF  = "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/figures/haltiala_temperature_day_python.gif"

UTC_HOURS = list(range(1, 24))

# Square bbox around Haltiala
xmin = 25493000
xmax = 25498000
ymin = 6679500
ymax = 6684500

# Visualization
Z_SCALE = 2
EXTRUDE_DEPTH = 130
FRAME_DURATION = 0.6

BG = "#0d1117"
SIDE_COLOR = "#707070"
BASE_COLOR = "#9a9a9a"

# =============================================================================
# HELPERS
# =============================================================================

def get_window(src, xmin, ymin, xmax, ymax):
    window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
    return window.round_offsets().round_lengths()

def read_as_grid(src, window, out_shape=None):
    if out_shape is None:
        return src.read(1, window=window, boundless=True, fill_value=np.nan)
    return src.read(
        1,
        window=window,
        boundless=True,
        fill_value=np.nan,
        out_shape=out_shape,
        resampling=Resampling.bilinear
    )

def format_time(utc):
    local = (utc + 3) % 24
    return f"{local:02d}:00 (UTC+3)"

# =============================================================================
# 1. LOAD DTM
# =============================================================================

with rasterio.open(DTM_PATH) as src:
    dtm_window = get_window(src, xmin, ymin, xmax, ymax)
    dtm_raw = read_as_grid(src, dtm_window)
    dtm_transform = src.window_transform(dtm_window)

dtm_valid_mask = np.isfinite(dtm_raw)

dtm_valid_fraction = dtm_valid_mask.mean()
print(f"DTM valid fraction: {dtm_valid_fraction:.3f}")
print("DTM raw shape:", dtm_raw.shape)

# Fill nodata only for smoothing/geometry
fill_value = np.nanmean(dtm_raw)
if np.isnan(fill_value):
    fill_value = 0.0

dtm_filled = np.where(dtm_valid_mask, dtm_raw, fill_value)
dtm_filled = gaussian_filter(dtm_filled, sigma=3)

dtm_filled = dtm_filled - np.nanmin(dtm_filled)
dtm_scaled = np.sqrt(np.maximum(dtm_filled, 0)) * Z_SCALE

ny, nx = dtm_scaled.shape

# Coordinate grid from raster transform
x = dtm_transform.c + (np.arange(nx) + 0.5) * dtm_transform.a
y = dtm_transform.f + (np.arange(ny) + 0.5) * dtm_transform.e
x, y = np.meshgrid(x, y)

surface = pv.StructuredGrid(x, y, dtm_scaled)

# =============================================================================
# 2. READ ALL TEMPS ON THE SAME GRID AND BUILD A GLOBAL VALID MASK
# =============================================================================

all_files = [
    os.path.join(PRED_DIR, f"pred_{DATE_STR}_{h:02d}00.tif")
    for h in UTC_HOURS
]

mins, maxs = [], []
combined_valid_mask = dtm_valid_mask.copy()

for f in all_files:
    with rasterio.open(f) as src:
        w = get_window(src, xmin, ymin, xmax, ymax)
        r = read_as_grid(src, w, out_shape=(ny, nx))

    mins.append(np.nanmin(r))
    maxs.append(np.nanmax(r))

    # intersection of valid areas across all frames
    combined_valid_mask &= np.isfinite(r)

TEMP_MIN = int(np.floor(min(mins)))
TEMP_MAX = int(np.ceil(max(maxs)))

print(f"Temperature range: {TEMP_MIN} to {TEMP_MAX}")
print(f"Combined valid fraction: {combined_valid_mask.mean():.3f}")
print("DTM shape:", dtm_scaled.shape)

# =============================================================================
# 3. BUILD CLIPPED FOOTPRINT
# =============================================================================

# Keep only cells whose four corners are valid in the combined mask
corner_count = (
    combined_valid_mask[:-1, :-1].astype(int)
    + combined_valid_mask[1:, :-1].astype(int)
    + combined_valid_mask[:-1, 1:].astype(int)
    + combined_valid_mask[1:, 1:].astype(int)
)
cell_keep = corner_count == 4

valid_cell_ids = np.flatnonzero(cell_keep.ravel(order="F"))
print(f"Valid cells: {len(valid_cell_ids)} / {cell_keep.size}")

if len(valid_cell_ids) == 0:
    raise RuntimeError("No valid cells found in the chosen extent.")

# Build the clipped footprint once
surface_clip = surface.extract_cells(valid_cell_ids)
surface_poly_base = surface_clip.extract_surface().triangulate()

if surface_poly_base.n_cells == 0:
    raise RuntimeError("No surface mesh after clipping. Check bbox / nodata mask.")

volume = surface_poly_base.extrude([0, 0, -EXTRUDE_DEPTH])

base = surface_poly_base.copy()
base.points[:, 2] = base.points[:, 2].min() - EXTRUDE_DEPTH

print("Surface bounds:", surface_poly_base.bounds)
print("Base bounds:", base.bounds)

# =============================================================================
# 4. LEGEND SETTINGS
# =============================================================================

scalar_bar_args = dict(
    title = "",
    title_font_size=18,
    label_font_size=12,
    fmt="%.0f",
    n_labels=5,

    vertical=True,
    position_x=0.08,   # move right
    position_y=0.1,    # move up
    width=0.06,
    height=0.6,

    color="white",
    shadow=True
)

# =============================================================================
# 5. RENDER FRAMES
# =============================================================================

frames = []

for i, h in enumerate(UTC_HOURS):
    print(f"Rendering hour {h:02d}")

    with rasterio.open(all_files[i]) as src:
        temp_window = get_window(src, xmin, ymin, xmax, ymax)
        temp = read_as_grid(src, temp_window, out_shape=(ny, nx))

    # Apply the global combined mask
    temp[~combined_valid_mask] = np.nan

    # Attach temperature to the full grid, then clip to the same valid footprint
    surface["temp"] = temp.ravel(order="F")
    surface_clip = surface.extract_cells(valid_cell_ids)
    surface_poly = surface_clip.extract_surface().triangulate()

    if surface_poly.n_cells == 0:
        print("Warning: no mesh after clipping on this frame.")
        print("Temp stats:", np.nanmin(temp), np.nanmax(temp))
        continue

    if i == 0:
        print("Frame 0 surface bounds:", surface_poly.bounds)
        print("Frame 0 finite temp fraction:", np.isfinite(temp).mean())

    plotter = pv.Plotter(off_screen=True)
    plotter.set_background(BG)

    plotter.add_mesh(
        surface_poly,
        scalars="temp",
        cmap="RdYlBu_r",
        clim=[TEMP_MIN, TEMP_MAX],
        interpolate_before_map=False,
        scalar_bar_args=scalar_bar_args,
        smooth_shading=True,
        specular=0.0,
        diffuse=0.9,
        ambient=0.3
    )

    # Base follows the same clipped footprint
    plotter.add_mesh(base, color=BASE_COLOR)

    # Sides from the clipped footprint
    plotter.add_mesh(volume, color=SIDE_COLOR)

    # Camera
    # Center of your data
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2  
    scale = max(xmax - xmin, ymax - ymin)

# Isometric-style camera, but controlled
    plotter.camera_position = [
        (cx - scale*1.1, cy - scale*1.4, scale*0.9),  # camera position
        (cx, cy, 0),                                  # look-at
        (0, 0, 1)                                     # up vector (IMPORTANT)
    ]

# Now rotate ONLY around vertical axis
    plotter.camera.azimuth += 25   # flip north-south
    plotter.camera.zoom(1.2)

    img = plotter.screenshot(window_size=[900, 500])
    plotter.close()

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)

    draw.text(
        (pil_img.width - 750, 100),
        format_time(h),
        fill=(255, 255, 255),
        font=font
    )

    draw.text((70, 120), "°C", fill=(255,255,255), font=font)

    frames.append(np.array(pil_img))

# =============================================================================
# 6. SAVE GIF
# =============================================================================

if len(frames) == 0:
    raise RuntimeError("No frames rendered")

imageio.mimsave(
    OUT_GIF,
    frames,
    duration=FRAME_DURATION,
    loop=0
)

print(f"Done -> {OUT_GIF}")