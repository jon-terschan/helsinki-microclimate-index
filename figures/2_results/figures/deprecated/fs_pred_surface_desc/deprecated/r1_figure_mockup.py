import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib as mpl
import geopandas as gpd

# -----------------------
# FAKE DATA (CONTROLLED DISTRIBUTION)
# -----------------------
np.random.seed(42)

n = 40

region_counts = {
    "CENTER": 10,
    "WEST": 9,
    "NORTH": 8,
    "EAST": 10,
    "ÖSTERSUNDOM": 3
}

regions_raw = np.concatenate([
    np.repeat(r, c) for r, c in region_counts.items()
])

np.random.shuffle(regions_raw)

cooling = np.random.uniform(0.5, 4.0, len(regions_raw))

names_raw = np.array([f"Park {i+1}" for i in range(len(regions_raw))])

region_order = ["CENTER", "WEST", "NORTH", "EAST", "ÖSTERSUNDOM"]

region_colors = {
    "CENTER": "#d73027",
    "NORTH": "#4575b4",
    "EAST": "#91bfdb",
    "WEST": "#fdae61",
    "ÖSTERSUNDOM": "#66bd63",
}

# -----------------------
# GROUPING
# -----------------------
grouped_idx = []
group_info = []

for r in region_order:
    idx = np.where(regions_raw == r)[0]
    if len(idx) == 0:
        continue

    idx_sorted = idx[np.argsort(cooling[idx])[::-1]]
    start = len(grouped_idx)
    grouped_idx.extend(idx_sorted.tolist())
    group_info.append((r, start, len(idx_sorted)))

cooling = cooling[grouped_idx]
regions = regions_raw[grouped_idx]
names = names_raw[grouped_idx]

# -----------------------
# FIGURE
# -----------------------
fig = plt.figure(figsize=(10, 10))

ax_map = fig.add_axes([0.25, 0.25, 0.5, 0.5])
ax_polar = fig.add_axes([0.05, 0.05, 0.9, 0.9], polar=True)

ax_map.set_zorder(1)
ax_polar.set_zorder(3)
ax_polar.set_facecolor("none")

## -----------------------
# MAP (DISSOLVED + FAKE RASTER)
# -----------------------
import geopandas as gpd
from shapely.affinity import translate, scale
from matplotlib.path import Path
from matplotlib.patches import PathPatch

gdf = gpd.read_file(r"C:\Users\terschan\Downloads\crowns\p4\Helsinki_Boundaries.gpkg")

# project if needed
if gdf.crs is None or gdf.crs.is_geographic:
    gdf = gdf.to_crs(epsg=3067)

# ---- dissolve to single geometry ----
gdf = gdf.dissolve()

# ---- normalize to [-1, 1] ----
minx, miny, maxx, maxy = gdf.total_bounds
cx = (minx + maxx) / 2
cy = (miny + maxy) / 2
scale_factor = max(maxx - minx, maxy - miny) / 2
scale_factor *= 1.15  # padding

gdf["geometry"] = gdf["geometry"].apply(
    lambda geom: scale(
        translate(geom, xoff=-cx, yoff=-cy),
        xfact=1/scale_factor,
        yfact=1/scale_factor,
        origin=(0, 0)
    )
)

geom = gdf.geometry.iloc[0]

# -----------------------
# CREATE CLUSTERED FAKE RASTER
# -----------------------
res = 400
x = np.linspace(-1, 1, res)
y = np.linspace(-1, 1, res)
xx, yy = np.meshgrid(x, y)

# --- multiple gaussian hotspots ---
n_blobs = 6
rng = np.random.default_rng(42)

raster = np.zeros_like(xx)

for _ in range(n_blobs):
    cx = rng.uniform(-0.5, 0.5)
    cy = rng.uniform(-0.5, 0.5)
    sigma = rng.uniform(0.15, 0.35)
    amplitude = rng.uniform(0.6, 1.2)

    blob = amplitude * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    raster += blob

# --- add broad spatial variation (very important) ---
raster += 0.3 * np.sin(2 * xx) * np.cos(2 * yy)

# --- fine noise ---
raster += rng.normal(0, 0.05, size=raster.shape)

# normalize
raster = (raster - raster.min()) / (raster.max() - raster.min())
# -----------------------
# SOFT RADIAL BACKGROUND (CIRCULAR)
# -----------------------
bg_res = 400
bx = np.linspace(-1, 1, bg_res)
by = np.linspace(-1, 1, bg_res)
bxx, byy = np.meshgrid(bx, by)

radial = np.sqrt(bxx**2 + byy**2)

# smooth radial falloff
bg = np.exp(-radial**2 * 2.5)

bg_im = ax_map.imshow(
    bg,
    extent=(-1, 1, -1, 1),
    cmap="coolwarm",
    alpha=0.25,
    zorder=0
)

# clip to circle
circle_clip = Circle((0, 0), 1.0, transform=ax_map.transData)
bg_im.set_clip_path(circle_clip)

# show raster
im = ax_map.imshow(
    raster,
    extent=(-1, 1, -1, 1),
    cmap="coolwarm",
    alpha=0.65,
    zorder=1
)
# -----------------------
# COLORBAR (COMPACT + CENTERED)
# -----------------------

cax = fig.add_axes([0.42, 0.28, 0.16, 0.015])

cbar = fig.colorbar(im, cax=cax, orientation="horizontal")

# move label to top
cbar.ax.xaxis.set_label_position('top')
cbar.set_label("Mean temperature offset (°C)", fontsize=8, labelpad=2)

# ticks also on top (optional but cleaner)
cbar.ax.xaxis.set_ticks_position('top')

cbar.set_ticks([0.0, 0.5, 1.0])
cbar.set_ticklabels(["0°C", "1.5°C", "3°C"])

cbar.ax.tick_params(labelsize=7, pad=1)
# -----------------------
# CLIP RASTER TO HELSINKI SHAPE
# -----------------------
def polygon_to_path(polygon):
    vertices = []
    codes = []

    def add_ring(coords):
        coords = list(coords)
        vertices.extend(coords)
        codes.extend([Path.MOVETO] + [Path.LINETO]*(len(coords)-2) + [Path.CLOSEPOLY])

    if polygon.geom_type == "Polygon":
        add_ring(polygon.exterior.coords)
        for interior in polygon.interiors:
            add_ring(interior.coords)

    elif polygon.geom_type == "MultiPolygon":
        for poly in polygon.geoms:
            add_ring(poly.exterior.coords)
            for interior in poly.interiors:
                add_ring(interior.coords)

    return Path(vertices, codes)

clip_path = PathPatch(polygon_to_path(geom), transform=ax_map.transData)
im.set_clip_path(clip_path)

# -----------------------
# OUTLINE
# -----------------------
gdf.boundary.plot(
    ax=ax_map,
    color="black",
    linewidth=1.0,
    zorder=2
)

ax_map.set_xlim(-1, 1)
ax_map.set_ylim(-1, 1)
ax_map.set_aspect("equal")
ax_map.axis("off")
# -----------------------
# POLAR LAYOUT
# -----------------------
hole_r = 8.6

region_sector = {
    "NORTH": (0, np.deg2rad(60)),
    "ÖSTERSUNDOM": (np.deg2rad(60), np.deg2rad(90)),
    "EAST": (np.deg2rad(90), np.deg2rad(150)),
    "CENTER": (np.deg2rad(180), np.deg2rad(240)),
    "WEST": (np.deg2rad(240), np.deg2rad(300)),
}

angles = np.empty(len(cooling))
region_bounds = []

for region, start, count in group_info:
    a0, a1 = region_sector[region]
    span = a1 - a0
    local_slot = span / count

    angles[start:start+count] = a0 + (np.arange(count) + 0.5) * local_slot
    region_bounds.append((a0, span, region, start, count))

# -----------------------
# POLAR AXIS
# -----------------------
ax_polar.set_rorigin(-hole_r)
ax_polar.set_rlim(0, cooling.max() + 3)

ax_polar.set_xticks([])
ax_polar.set_yticks([])
ax_polar.spines["polar"].set_visible(False)
ax_polar.grid(False)

ax_polar.set_theta_offset(np.pi/2)
ax_polar.set_theta_direction(-1)

# -----------------------
# RADIAL RINGS
# -----------------------
theta_dense = np.linspace(0, 2*np.pi, 500)

for val, label in [(2, "2°C"), (3, "3°C")]:
    ax_polar.plot(theta_dense, np.full_like(theta_dense, val),
                  color="gray", linewidth=0.8, alpha=0.6)

    ax_polar.text(np.pi/2, val, label,
                  ha="center", va="bottom", fontsize=9)

# -----------------------
# BARS (SMOOTH WITHIN-BAR GRADIENT)
# -----------------------
bar_width = 0.9 * (angles[1] - angles[0])
n_layers = 18

for angle, val, region in zip(angles, cooling, regions):
    base = np.array(mpl.colors.to_rgb(region_colors[region]))

    for i in range(n_layers):
        f0 = i / n_layers
        f1 = (i + 1) / n_layers

        overlap = val / n_layers * 0.15
        r0 = max(val * f0 - overlap / 2, 0)
        height = val * (f1 - f0) + overlap

        t = f1**1.4
        color = (1 - t) * np.ones(3) + t * base

        ax_polar.bar(
            angle,
            height,
            width=bar_width,
            bottom=r0,
            color=color,
            edgecolor=None,
            linewidth=0,
            antialiased=False,
            zorder=1
        )

# separators
ax_polar.bar(
    angles,
    cooling,
    width=bar_width,
    bottom=0,
    color="none",
    edgecolor="white",
    linewidth=1,
    zorder=2
)

# -----------------------
# LABELS (unchanged)
# -----------------------
for angle, name in zip(angles, names):
    display_angle = (np.pi/2 - angle)
    rotation = np.degrees(display_angle)

    if np.pi/2 < display_angle % (2*np.pi) < 3*np.pi/2:
        rotation += 180
        ha = "right"
    else:
        ha = "left"

    ax_polar.text(angle, 0.25, name, fontsize=6,
                  ha=ha, va="center",
                  rotation=rotation, rotation_mode='anchor')

outer_radius = cooling.max() + 3

for a0, span, region, *_ in region_bounds:
    ax_polar.plot([a0, a0], [0, outer_radius], color="black", linewidth=1)

    label_angle = a0 + span / 2
    ax_polar.text(label_angle, outer_radius + 0.3,
                  region, ha="center", va="center",
                  fontsize=10, fontweight="bold")

for angle, val in zip(angles, cooling):
    rotation = np.degrees(angle)

    if np.pi/2 < angle < 3*np.pi/2:
        rotation += 180
        ha = "right"
    else:
        ha = "left"

    ax_polar.text(angle, val + 0.15, f"{val:.1f}",
                  fontsize=7, ha=ha, va="center",
                  rotation=rotation, rotation_mode='anchor')

handles = [
    mpl.patches.Patch(color=region_colors[r], label=r)
    for r in region_colors
]

fig.legend(handles=handles, loc="lower left")

plt.show()