# ============================================================
# IMPROVED GAP DISTANCE MODEL
# Implements:
# 1. Large true canopy openings only
# 2. Pixels far from outer forest edge only
# 3. Distance bins + patch means
# 4. Patch fixed effects model
# 5. Cleaner figure
# ============================================================

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import plotting_extent
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import (
    label,
    generate_binary_structure,
    binary_closing,
    binary_fill_holes,
    binary_dilation,
    distance_transform_edt,
    uniform_filter,
)

import statsmodels.formula.api as smf
import statsmodels.api as sm

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
PRED_ROOT = BASE / "predictions"
STACK = BASE / "predictorstack"

OUT_DIR = BASE / "figures" / "gap_distance_patchFE"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TREE = STACK / "TREE_FRAC_10m.tif"
CHM = STACK / "CHM_10m_MAX.tif"
P90 = PRED_ROOT / "baseline" / "15cm_July_allday_p90" / "pred_20000715_0900.tif"

EVENTS = {
    "2010": PRED_ROOT / "2010" / "20100728" / "pred_20100728_0900.tif",
    "2018": PRED_ROOT / "2018" / "20180717" / "pred_20180717_0900.tif",
    "2021": PRED_ROOT / "2021" / "20210714" / "pred_20210714_0900.tif",
}

# ============================================================
# SETTINGS
# ============================================================

TREE_THRESHOLD = 0.45
MIN_PATCH_HA = 20

OUTER_EDGE_EXCLUSION_M = 50
MAX_GAP_DIST = 500

# canopy opening definition (true gaps)
GAP_CHM_MAX = 8.0          # low canopy
GAP_MIN_AREA_M2 = 400      # >= 20x20 m
GAP_CLOSE_ITERS = 1

ANOMALY_WINDOW = 3
MAX_PER_PATCH = 4000
SEED = 42

DIST_BINS = np.array([0,25,50,100,150,250,350,500])

sns.set_theme(style="whitegrid", context="talk")

# ============================================================
# HELPERS
# ============================================================

def read_raster(path, band=1):
    with rasterio.open(path) as src:
        arr = src.read(band).astype("float32")
        nodata = src.nodata
        extent = plotting_extent(src)
        transform = src.transform

    if nodata is not None:
        arr[arr == nodata] = np.nan

    return arr, extent, transform


def remove_small(mask, min_pixels):
    lab, num = label(mask, generate_binary_structure(2,2))
    if num == 0:
        return mask

    cnt = np.bincount(lab.ravel())
    keep = cnt >= min_pixels
    keep[0] = False
    return keep[lab]


def local_mean(arr, valid, size=3):
    good = valid & np.isfinite(arr)

    x = np.where(good, arr, 0.0)
    w = good.astype(float)

    sx = uniform_filter(x, size=size, mode="nearest") * size * size
    sw = uniform_filter(w, size=size, mode="nearest") * size * size

    return sx / np.where(sw == 0, np.nan, sw)


# ============================================================
# LOAD
# ============================================================

tree, extent, transform = read_raster(TREE)
chm, _, _ = read_raster(CHM)
p90, _, _ = read_raster(P90)

pix_x = abs(transform.a)
pix_y = abs(transform.e)

PIX = pix_x
pixel_area_m2 = pix_x * pix_y
pixel_area_ha = pixel_area_m2 / 10000.0
gap_min_pixels = int(np.ceil(GAP_MIN_AREA_M2 / pixel_area_m2))

# ============================================================
# RESPONSE VARIABLE
# ============================================================

anom = []

for fp in EVENTS.values():
    x, _, _ = read_raster(fp)
    anom.append(x - p90)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    anomaly_mean = np.nanmean(np.stack(anom), axis=0)

# ============================================================
# FOREST MASK
# ============================================================

forest = np.nan_to_num(tree, nan=0) >= TREE_THRESHOLD
forest = binary_closing(forest, structure=np.ones((3,3)))
forest = binary_fill_holes(forest)
forest = binary_dilation(forest, iterations=2)
forest = remove_small(forest, 25)

patch_id, nlab = label(forest, generate_binary_structure(2,2))

keep = np.zeros_like(forest, dtype=bool)

for pid in range(1, nlab+1):
    m = patch_id == pid
    if m.sum() * pixel_area_ha >= MIN_PATCH_HA:
        keep |= m

forest = keep

# relabel after removing small patches
patch_id, nlab = label(forest, generate_binary_structure(2,2))

response_sm = local_mean(anomaly_mean, forest, ANOMALY_WINDOW)
response_sm[~forest] = np.nan

# ============================================================
# EXTRACT PIXELS
# ============================================================

rows = []

for pid in range(1, nlab+1):

    patch = patch_id == pid
    rr, cc = np.where(patch)

    if len(rr) == 0:
        continue

    r0, r1 = rr.min(), rr.max()+1
    c0, c1 = cc.min(), cc.max()+1

    p  = patch[r0:r1, c0:c1]
    cm = chm[r0:r1, c0:c1]
    rs = response_sm[r0:r1, c0:c1]

    # distance from outer forest boundary
    edge = distance_transform_edt(p, sampling=(pix_y,pix_x))
    interior = p & (edge >= OUTER_EDGE_EXCLUSION_M)

    if interior.sum() < 30:
        continue

    # ---------------------------------
    # TRUE GAPS = LOW CHM ONLY
    # ---------------------------------
    gap = interior & np.isfinite(cm) & (cm <= GAP_CHM_MAX)

    gap = binary_closing(gap, iterations=GAP_CLOSE_ITERS)
    gap = remove_small(gap, gap_min_pixels)

    if gap.sum() == 0:
        continue

    # distance to nearest gap
    src = np.ones_like(gap, dtype=np.uint8)
    src[gap] = 0

    d = distance_transform_edt(src, sampling=(pix_y,pix_x))

    use = (
        interior &
        np.isfinite(rs) &
        (d <= MAX_GAP_DIST)
    )

    if use.sum() == 0:
        continue

    tmp = pd.DataFrame({
        "patch_id": pid,
        "dist_gap_m": d[use].astype("float32"),
        "response_sm": rs[use].astype("float32")
    })

    if len(tmp) > MAX_PER_PATCH:
        tmp = tmp.sample(MAX_PER_PATCH, random_state=SEED)

    rows.append(tmp)

if len(rows) == 0:
    raise ValueError("No usable patches found.")

df = pd.concat(rows, ignore_index=True)

print("Pixels:", len(df))
print("Patches:", df["patch_id"].nunique())

# ============================================================
# DISTANCE BINS
# ============================================================

df["dist_bin"] = pd.cut(
    df["dist_gap_m"],
    bins=DIST_BINS,
    include_lowest=True
)

bin_df = (
    df.groupby(["patch_id","dist_bin"], observed=True)["response_sm"]
      .mean()
      .reset_index()
)

bin_df["bin_mid"] = bin_df["dist_bin"].apply(
    lambda x: (x.left + x.right)/2
)

# ============================================================
# PATCH FIXED EFFECTS MODEL
# ============================================================

model = smf.ols(
    "response_sm ~ dist_gap_m + C(patch_id)",
    data=df
).fit(cov_type="HC3")

slope = model.params["dist_gap_m"]
pval = model.pvalues["dist_gap_m"]
ci_low, ci_high = model.conf_int().loc["dist_gap_m"]

print("\nPatch FE model")
print(f"Slope: {slope*100:.3f} °C / 100 m")
print(f"95% CI: {ci_low*100:.3f} to {ci_high*100:.3f}")
print(f"p = {pval:.4g}")

# ============================================================
# SAVE
# ============================================================

df.to_csv(OUT_DIR / "pixel_table.csv", index=False)
bin_df.to_csv(OUT_DIR / "bin_means.csv", index=False)

# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(11,7))

sns.lineplot(
    data=bin_df,
    x="bin_mid",
    y="response_sm",
    estimator="mean",
    errorbar=("ci",95),
    lw=3,
    marker="o",
    ax=ax
)

ax.set_xlabel("Distance to canopy opening (m)")
ax.set_ylabel("Heatwave anomaly (°C)")
ax.set_title("Patch-normalized response to canopy openings")

ax.text(
    0.02, 0.98,
    f"Slope = {slope*100:.3f} °C / 100 m\n"
    f"p = {pval:.3g}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)

plt.tight_layout()
plt.savefig(OUT_DIR / "gap_distance_patchFE.png", dpi=350)
plt.show()

# ============================================================
# MAP OF GAPS USED IN MODEL
# ============================================================

accepted_gap = np.zeros_like(forest, dtype=bool)

for pid in range(1, nlab + 1):

    patch = patch_id == pid
    rr, cc = np.where(patch)
    if len(rr) == 0:
        continue

    r0, r1 = rr.min(), rr.max() + 1
    c0, c1 = cc.min(), cc.max() + 1

    p  = patch[r0:r1, c0:c1]
    cm = chm[r0:r1, c0:c1]

    edge = distance_transform_edt(p, sampling=(pix_y, pix_x))
    interior = p & (edge >= OUTER_EDGE_EXCLUSION_M)

    if interior.sum() < 30:
        continue

    gap = interior & np.isfinite(cm) & (cm <= GAP_CHM_MAX)
    gap = binary_closing(gap, iterations=GAP_CLOSE_ITERS)
    gap = remove_small(gap, gap_min_pixels)

    if gap.sum() == 0:
        continue

    accepted_gap[r0:r1, c0:c1] |= gap

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12,12))

ax.imshow(
    forest,
    extent=extent,
    cmap="Greens",
    alpha=0.35,
    interpolation="nearest"
)

ax.imshow(
    accepted_gap,
    extent=extent,
    cmap="Reds",
    alpha=0.85,
    interpolation="nearest"
)

ax.set_title("Forest mask (green) and gaps used in model (red)")
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")

plt.tight_layout()
plt.savefig(OUT_DIR / "accepted_gaps_map.png", dpi=350)
plt.show()


# ============================================================
# IMPROVED GAP DISTANCE MODEL
# Implements:
# 1. Large true canopy openings only
# 2. Pixels far from outer forest edge only
# 3. Removes first 10 m from gap edge (security buffer)
# 4. Uses UCC + CHM + ROCK mask for robust gap detection
# 5. Distance bins + patch means
# 6. Patch fixed effects model
# 7. Cleaner figure
# 8. Map of accepted gaps
# ============================================================

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import plotting_extent
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import (
    label,
    generate_binary_structure,
    binary_closing,
    binary_fill_holes,
    binary_dilation,
    distance_transform_edt,
    uniform_filter,
)

import statsmodels.formula.api as smf

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
PRED_ROOT = BASE / "predictions"
STACK = BASE / "predictorstack"

OUT_DIR = BASE / "figures" / "gap_distance_patchFE_robust"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TREE = STACK / "TREE_FRAC_10m.tif"
CHM = STACK / "CHM_10m_MAX.tif"
CM = STACK / "CM_loc_HEL.tif"          # multiband
ROCK = STACK / "ROCK_FRAC_10m_Helsinki.tif"

P90 = PRED_ROOT / "baseline" / "15cm_July_allday_p90" / "pred_20000715_1000.tif"

EVENTS = {
    "2010": PRED_ROOT / "2010" / "20100728" / "pred_20100728_1000.tif",
    "2018": PRED_ROOT / "2018" / "20180717" / "pred_20180717_1000.tif",
    "2021": PRED_ROOT / "2021" / "20210714" / "pred_20210714_1000.tif",
}

# ============================================================
# SETTINGS
# ============================================================

TREE_THRESHOLD = 0.45
MIN_PATCH_HA = 20

OUTER_EDGE_EXCLUSION_M = 50
INNER_GAP_BUFFER_M = 10
MAX_GAP_DIST = 500

# robust canopy opening definition
GAP_CHM_MAX = 6.0
GAP_UCC_MAX = 0.25
GAP_MIN_AREA_M2 = 900      # 30 x 30 m
GAP_CLOSE_ITERS = 1

ANOMALY_WINDOW = 3
MAX_PER_PATCH = 4000
SEED = 42

DIST_BINS = np.array([10,25,50,100,150,250,350,500])

sns.set_theme(style="whitegrid", context="talk")

# ============================================================
# HELPERS
# ============================================================

def read_raster(path, band=1):
    with rasterio.open(path) as src:
        arr = src.read(band).astype("float32")
        nodata = src.nodata
        extent = plotting_extent(src)
        transform = src.transform

    if nodata is not None:
        arr[arr == nodata] = np.nan

    return arr, extent, transform


def remove_small(mask, min_pixels):
    lab, num = label(mask, generate_binary_structure(2,2))
    if num == 0:
        return mask

    cnt = np.bincount(lab.ravel())
    keep = cnt >= min_pixels
    keep[0] = False
    return keep[lab]


def local_mean(arr, valid, size=3):
    good = valid & np.isfinite(arr)

    x = np.where(good, arr, 0.0)
    w = good.astype(float)

    sx = uniform_filter(x, size=size, mode="nearest") * size * size
    sw = uniform_filter(w, size=size, mode="nearest") * size * size

    return sx / np.where(sw == 0, np.nan, sw)

# ============================================================
# LOAD
# ============================================================

tree, extent, transform = read_raster(TREE)
chm, _, _ = read_raster(CHM)
ucc, _, _ = read_raster(CM, band=4)   # UCC layer
rock, _, _ = read_raster(ROCK)

p90, _, _ = read_raster(P90)

pix_x = abs(transform.a)
pix_y = abs(transform.e)

pixel_area_m2 = pix_x * pix_y
pixel_area_ha = pixel_area_m2 / 10000.0

gap_min_pixels = int(np.ceil(GAP_MIN_AREA_M2 / pixel_area_m2))

# ============================================================
# RESPONSE VARIABLE
# ============================================================

anom = []

for fp in EVENTS.values():
    x, _, _ = read_raster(fp)
    anom.append(x - p90)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    anomaly_mean = np.nanmean(np.stack(anom), axis=0)

# ============================================================
# FOREST MASK
# ============================================================

forest = np.nan_to_num(tree, nan=0) >= TREE_THRESHOLD
forest = binary_closing(forest, structure=np.ones((3,3)))
forest = binary_fill_holes(forest)
forest = binary_dilation(forest, iterations=2)
forest = remove_small(forest, 25)

patch_id, nlab = label(forest, generate_binary_structure(2,2))

keep = np.zeros_like(forest, dtype=bool)

for pid in range(1, nlab+1):
    m = patch_id == pid
    if m.sum() * pixel_area_ha >= MIN_PATCH_HA:
        keep |= m

forest = keep
patch_id, nlab = label(forest, generate_binary_structure(2,2))

response_sm = local_mean(anomaly_mean, forest, ANOMALY_WINDOW)
response_sm[~forest] = np.nan

# ============================================================
# EXTRACT PIXELS
# ============================================================

rows = []
accepted_gap = np.zeros_like(forest, dtype=bool)

for pid in range(1, nlab+1):

    patch = patch_id == pid
    rr, cc = np.where(patch)

    if len(rr) == 0:
        continue

    r0, r1 = rr.min(), rr.max()+1
    c0, c1 = cc.min(), cc.max()+1

    p   = patch[r0:r1, c0:c1]
    cm  = chm[r0:r1, c0:c1]
    uc  = ucc[r0:r1, c0:c1]
    rk  = rock[r0:r1, c0:c1]
    rs  = response_sm[r0:r1, c0:c1]

    edge = distance_transform_edt(p, sampling=(pix_y,pix_x))
    interior = p & (edge >= OUTER_EDGE_EXCLUSION_M)

    if interior.sum() < 30:
        continue

    # ------------------------------------------------
    # ROBUST GAPS:
    # low CHM + low canopy cover + no rock
    # ------------------------------------------------

    gap = (
        interior &
        np.isfinite(cm) &
        np.isfinite(uc) &
        np.isfinite(rk) &
        (cm <= GAP_CHM_MAX) &
        (uc <= GAP_UCC_MAX) &
        (rk == 0)
    )

    gap = binary_closing(gap, iterations=GAP_CLOSE_ITERS)
    gap = remove_small(gap, gap_min_pixels)

    if gap.sum() == 0:
        continue

    accepted_gap[r0:r1, c0:c1] |= gap

    # distance to nearest gap
    src = np.ones_like(gap, dtype=np.uint8)
    src[gap] = 0

    d = distance_transform_edt(src, sampling=(pix_y,pix_x))

    use = (
        interior &
        np.isfinite(rs) &
        np.isfinite(rk) &
        (rk == 0) &
        (d >= INNER_GAP_BUFFER_M) &
        (d <= MAX_GAP_DIST)
    )

    if use.sum() == 0:
        continue

    tmp = pd.DataFrame({
        "patch_id": pid,
        "dist_gap_m": d[use].astype("float32"),
        "response_sm": rs[use].astype("float32")
    })

    if len(tmp) > MAX_PER_PATCH:
        tmp = tmp.sample(MAX_PER_PATCH, random_state=SEED)

    rows.append(tmp)

if len(rows) == 0:
    raise ValueError("No usable patches found.")

df = pd.concat(rows, ignore_index=True)

print("Pixels:", len(df))
print("Patches:", df["patch_id"].nunique())

# ============================================================
# DISTANCE BINS
# ============================================================

df["dist_bin"] = pd.cut(
    df["dist_gap_m"],
    bins=DIST_BINS,
    include_lowest=True
)

bin_df = (
    df.groupby(["patch_id","dist_bin"], observed=True)["response_sm"]
      .mean()
      .reset_index()
)

bin_df["bin_mid"] = bin_df["dist_bin"].apply(
    lambda x: (x.left + x.right) / 2
)

# ============================================================
# PATCH FIXED EFFECTS MODEL
# ============================================================

model = smf.ols(
    "response_sm ~ dist_gap_m + C(patch_id)",
    data=df
).fit(cov_type="HC3")

slope = model.params["dist_gap_m"]
pval = model.pvalues["dist_gap_m"]
ci_low, ci_high = model.conf_int().loc["dist_gap_m"]

print("\nPatch FE model")
print(f"Slope: {slope*100:.3f} °C / 100 m")
print(f"95% CI: {ci_low*100:.3f} to {ci_high*100:.3f}")
print(f"p = {pval:.4g}")

# ============================================================
# SAVE
# ============================================================

df.to_csv(OUT_DIR / "pixel_table.csv", index=False)
bin_df.to_csv(OUT_DIR / "bin_means.csv", index=False)

# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(11,7))

sns.lineplot(
    data=bin_df,
    x="bin_mid",
    y="response_sm",
    estimator="mean",
    errorbar=("ci",95),
    lw=3,
    marker="o",
    ax=ax
)

ax.set_xlabel("Distance to canopy opening (m)")
ax.set_ylabel("Heatwave anomaly (°C)")
ax.set_title("Patch-normalized response to robust canopy openings")

ax.text(
    0.02, 0.98,
    f"Slope = {slope*100:.3f} °C / 100 m\n"
    f"p = {pval:.3g}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)

plt.tight_layout()
plt.savefig(OUT_DIR / "gap_distance_patchFE.png", dpi=350)
plt.show()

# ============================================================
# MAP OF ACCEPTED GAPS
# ============================================================

fig, ax = plt.subplots(figsize=(12,12))

ax.imshow(
    forest,
    extent=extent,
    cmap="Greens",
    alpha=0.35,
    interpolation="nearest"
)

ax.imshow(
    accepted_gap,
    extent=extent,
    cmap="Reds",
    alpha=0.90,
    interpolation="nearest"
)

ax.set_title("Forest mask (green) and robust canopy openings used (red)")
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")

plt.tight_layout()
plt.savefig(OUT_DIR / "accepted_gaps_map.png", dpi=350)
plt.show()



# this is bullshit, get red of it