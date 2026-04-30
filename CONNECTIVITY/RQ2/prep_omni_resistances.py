# build_resistance_surfaces.py
# Creates three citywide Omniscape resistance rasters from your predictor stack.
# Outputs:
#   resistance_v1_neutral.tif
#   resistance_v2_generalist.tif
#   resistance_v3_barrierheavy.tif
#   qc_resistance_maps.png
#
# Assumes all rasters share same CRS / extent / resolution.

from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
STACK = BASE / "predictorstack"
OUT = BASE / "omniscape" / "resistance"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "bldg": STACK / "BLDG_FRAC_10m.tif",
    "imperv": STACK / "IMPERV_FRAC_10m_Helsinki.tif",
    "tree": STACK / "TREE_FRAC_10m.tif",
    "nwn": STACK / "NWN_FRAC_10m.tif",
    "water": STACK / "WATER_FRAC_10m_Helsinki.tif",
    "ocean": STACK / "OCEAN_FRAC_10m_Helsinki.tif",
    "rock": STACK / "ROCK_FRAC_10m_Helsinki.tif",
}

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    return arr, profile

def write_raster(path, arr, profile):
    p = profile.copy()
    p.update(
        dtype="float32",
        count=1,
        compress="deflate",
        tiled=False,
        nodata=-9999
    )
    out = arr.copy()
    out[np.isnan(out)] = -9999
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out.astype("float32"), 1)

def scale_to_1_100(arr):
    valid = np.isfinite(arr)
    vals = arr[valid]
    mn = np.min(vals)
    mx = np.max(vals)

    out = np.full(arr.shape, np.nan, dtype="float32")

    if np.isclose(mn, mx):
        out[valid] = 1
    else:
        out[valid] = 1 + 99 * ((arr[valid] - mn) / (mx - mn))

    return out

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

layers = {}
profile = None

for k, f in FILES.items():
    arr, prof = read_raster(f)
    layers[k] = arr
    if profile is None:
        profile = prof

# shorthand
B = np.nan_to_num(layers["bldg"], nan=0)
I = np.nan_to_num(layers["imperv"], nan=0)
T = np.nan_to_num(layers["tree"], nan=0)
N = np.nan_to_num(layers["nwn"], nan=0)
W = np.nan_to_num(layers["water"], nan=0)
O = np.nan_to_num(layers["ocean"], nan=0)
R = np.nan_to_num(layers["rock"], nan=0)

# ---------------------------------------------------
# MODEL 1: neutral
# ---------------------------------------------------

raw1 = (
    4*B +
    3*I +
    2*W +
    5*O -
    2*T -
    1.5*N -
    0.5*R
)

res1 = scale_to_1_100(raw1)

# ---------------------------------------------------
# MODEL 2: urban generalist
# lower penalties for built surfaces
# ---------------------------------------------------

raw2 = (
    2.5*B +
    2*I +
    1.5*W +
    4*O -
    1.5*T -
    1.0*N -
    0.5*R
)

res2 = scale_to_1_100(raw2)

# ---------------------------------------------------
# MODEL 3: barrier heavy
# very strong urban hostility
# ---------------------------------------------------

raw3 = (
    6*B +
    5*I +
    3*W +
    7*O -
    2*T -
    1.5*N -
    0.25*R
)

res3 = scale_to_1_100(raw3)

# ---------------------------------------------------
# EXPORT
# ---------------------------------------------------

write_raster(OUT / "resistance_v1_neutral.tif", res1, profile)
write_raster(OUT / "resistance_v2_generalist.tif", res2, profile)
write_raster(OUT / "resistance_v3_barrierheavy.tif", res3, profile)

print("GeoTIFFs written.")

# ---------------------------------------------------
# QC MAPS
# ---------------------------------------------------

fig, ax = plt.subplots(1, 3, figsize=(16, 6))

for a, arr, title in zip(
    ax,
    [res1, res2, res3],
    ["Neutral", "Generalist", "Barrier heavy"]
):
    im = a.imshow(arr, cmap="viridis", vmin=1, vmax=100)
    a.set_title(title)
    a.axis("off")

fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.7, label="Resistance")
plt.tight_layout()
plt.savefig(OUT / "qc_resistance_maps.png", dpi=200)
plt.close()

print("QC map written.")