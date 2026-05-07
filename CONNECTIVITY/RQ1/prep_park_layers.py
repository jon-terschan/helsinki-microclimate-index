# ======================================================
# HELSINKI HEATWAVE ANALYSIS PIPELINE
# Cleaned version: same outputs, same figures, clearer structure
# ======================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import plotting_extent

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

import seaborn as sns

from shapely.geometry import mapping, box
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from adjustText import adjust_text

# ======================================================
# PATHS
# ======================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")

DISTRICT_FILE = BASE / "kartta_greenareas" / "kaupunginosat.gpkg"

PRED_ROOT = BASE / "predictions"
BASELINE_AVG = PRED_ROOT / "baseline" / "15cm_July_allday"
BASELINE_P90 = PRED_ROOT / "baseline" / "15cm_July_allday_p90"

TREE_TIF = BASE / "predictorstack" / "TREE_FRAC_10m.tif"
NWN_TIF  = BASE / "predictorstack" / "NWN_FRAC_10m.tif"

OUT_DIR = BASE / "figures" / "district_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# SETTINGS
# ======================================================

TARGET_CRS = 3879
PRIMARY_HOURS = range(9, 13)   # 09–12 UTC
FULL_HOURS = range(7, 16)      # 07–15 UTC
TARGET_HOUR = 9                # 12 local summer time

EVENTS = {
    "2010": "20100728",
    "2018": "20180717",
    "2021": "20210714"
}

N_CLUSTERS = 4
EXPORT = True

# ======================================================
# HELPERS
# ======================================================

def tif_name(day, hour):
    return f"pred_{day}_{hour:02d}00.tif"


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        extent = plotting_extent(src)
        crs = src.crs

    if nodata is not None:
        arr[arr == nodata] = np.nan

    return arr, extent, crs


def zonal_mean(path, geom, usemask):
    with rasterio.open(path) as src:
        arr, _ = mask(src, [mapping(geom)], crop=True, filled=True, nodata=np.nan)

    vals = arr[0]
    keep = usemask & (~np.isnan(vals))

    if keep.sum() == 0:
        return np.nan

    return float(np.nanmean(vals[keep]))


def zonal_mean_full(path, usemask):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)

    keep = usemask & (~np.isnan(arr))

    if keep.sum() == 0:
        return np.nan

    return float(np.nanmean(arr[keep]))


# ======================================================
# DISTRICT DATA
# ======================================================

def load_districts():
    gdf = gpd.read_file(DISTRICT_FILE).to_crs(TARGET_CRS)

    with rasterio.open(TREE_TIF) as src:
        extent_poly = box(*src.bounds)

    extent_gdf = gpd.GeoDataFrame(geometry=[extent_poly], crs=TARGET_CRS)
    gdf = gpd.clip(gdf, extent_gdf)

    if "aluejako" in gdf.columns:
        gdf = gdf[gdf["aluejako"].str.upper() == "KAUPUNGINOSA"].copy()

    gdf["district"] = gdf["nimi_fi"] if "nimi_fi" in gdf.columns else gdf.index.astype(str)

    return gdf[["district", "geometry"]].reset_index(drop=True)


def district_masks(geom):
    with rasterio.open(TREE_TIF) as src:
        tree, _ = mask(src, [mapping(geom)], crop=True, filled=True, nodata=np.nan)

    with rasterio.open(NWN_TIF) as src:
        nwn, _ = mask(src, [mapping(geom)], crop=True, filled=True, nodata=np.nan)

    tree = tree[0]
    nwn = nwn[0]

    valid = (~np.isnan(tree)) | (~np.isnan(nwn))
    if valid.sum() == 0:
        return None, None, np.nan, np.nan, np.nan

    tree_mask = (np.nan_to_num(tree) > 0) & valid
    nwn_mask = (np.nan_to_num(nwn) > 0) & valid
    greenmask = tree_mask | nwn_mask

    tree_share = tree_mask.sum() / valid.sum()
    nwn_share = nwn_mask.sum() / valid.sum()
    green_share = greenmask.sum() / valid.sum()

    if greenmask.sum() == 0:
        return None, valid, tree_share, nwn_share, green_share

    return greenmask, valid, tree_share, nwn_share, green_share


# ======================================================
# TIME SERIES EXTRACTION
# ======================================================

def hourly_metrics(geom, greenmask, peakday, year):
    rows = []

    for h in FULL_HOURS:
        evt = PRED_ROOT / year / peakday / tif_name(peakday, h)
        avg = BASELINE_AVG / tif_name("20000715", h)
        p90 = BASELINE_P90 / tif_name("20000715", h)

        if not (evt.exists() and avg.exists() and p90.exists()):
            continue

        evt_m = zonal_mean(evt, geom, greenmask)
        avg_m = zonal_mean(avg, geom, greenmask)
        p90_m = zonal_mean(p90, geom, greenmask)

        rows.append({
            "hour_utc": h,
            "event_temp": evt_m,
            "avg_temp": avg_m,
            "p90_temp": p90_m,
            "offset_vs_avg": evt_m - avg_m,
            "offset_vs_p90": evt_m - p90_m
        })

    return pd.DataFrame(rows)


# ======================================================
# DISTRICT EVENT TABLE
# ======================================================

districts = load_districts()
rows = []

for _, row in districts.iterrows():

    district = row["district"]
    geom = row.geometry

    print("Processing:", district)

    greenmask, validmask, tree_share, nwn_share, green_share = district_masks(geom)

    if greenmask is None:
        continue

    for year, peakday in EVENTS.items():

        ts = hourly_metrics(geom, greenmask, peakday, year)

        if ts.empty:
            continue

        primary = ts[ts["hour_utc"].isin(PRIMARY_HOURS)]

        rows.append({
            "district": district,
            "event_year": year,
            "tree_share": tree_share,
            "nwn_share": nwn_share,
            "green_share": green_share,

            "mean_09_12_event_temp": primary["event_temp"].mean(),
            "mean_09_12_avg_temp": primary["avg_temp"].mean(),
            "mean_09_12_p90_temp": primary["p90_temp"].mean(),
            "mean_09_12_offset_vs_avg": primary["offset_vs_avg"].mean(),
            "mean_09_12_offset_vs_p90": primary["offset_vs_p90"].mean(),
            "max_09_12_offset_vs_p90": primary["offset_vs_p90"].max(),

            "mean_07_15_event_temp": ts["event_temp"].mean(),
            "mean_07_15_offset_vs_avg": ts["offset_vs_avg"].mean(),
            "mean_07_15_offset_vs_p90": ts["offset_vs_p90"].mean(),
            "max_07_15_offset_vs_p90": ts["offset_vs_p90"].max(),

            "n_hours": len(ts)
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "district_heatwave_offsets_events.csv", index=False)

# ======================================================
# DISTRICT SUMMARY
# ======================================================

summary = (
    df.groupby("district", as_index=False)
      .mean(numeric_only=True)
)

summary["n_events"] = df.groupby("district")["event_year"].nunique().values
summary["mean_anomaly"] = summary["mean_09_12_offset_vs_p90"]

summary.to_csv(OUT_DIR / "district_heatwave_offsets_summary.csv", index=False)

# ======================================================
# CLUSTERING
# ======================================================

X = summary[["tree_share", "nwn_share", "mean_anomaly"]]
Xs = StandardScaler().fit_transform(X)

k = min(N_CLUSTERS, len(summary))
km = KMeans(n_clusters=k, random_state=42, n_init=20)

summary["cluster"] = km.fit_predict(Xs)

# ======================================================
# FIGURE 1
# DISTRICT TREE vs NWN
# ======================================================

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 9))

palette = sns.color_palette("tab10", n_colors=k)
sizes = 200 * (0.15 + summary["green_share"])

for c in sorted(summary["cluster"].unique()):
    d = summary[summary["cluster"] == c]

    ax.scatter(
        d["tree_share"],
        d["nwn_share"],
        s=sizes.loc[d.index],
        color=palette[c],
        edgecolor="black",
        linewidth=0.3,
        alpha=0.88,
        label=f"Cluster {c}"
    )

for _, r in summary.iterrows():
    ax.text(r["tree_share"] + 0.002, r["nwn_share"] + 0.002, r["district"], fontsize=8)

ax.axvline(summary["tree_share"].mean(), ls="--", lw=1, color="gray")
ax.axhline(summary["nwn_share"].mean(), ls="--", lw=1, color="gray")

ax.set_xlabel("Tree share")
ax.set_ylabel("NWN share")
ax.set_title("District composition in modeled green domain")
ax.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "district_tree_vs_nwn_share.png", dpi=300)
plt.show()

# ======================================================
# FIGURE 2
# RELATIONSHIPS
# ======================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

pairs = [
    ("tree_share", "Tree cover vs heatwave anomaly"),
    ("nwn_share", "NWN cover vs heatwave anomaly"),
    ("green_share", "Total green share vs heatwave anomaly")
]

for ax, (xvar, title) in zip(axes.flat[:3], pairs):
    sns.regplot(
        data=summary,
        x=xvar,
        y="mean_09_12_offset_vs_p90",
        scatter_kws={"s": 70, "alpha": 0.8},
        line_kws={"lw": 2},
        ax=ax
    )
    ax.set_title(title)

sc = axes[1, 1].scatter(
    summary["tree_share"],
    summary["nwn_share"],
    c=summary["mean_09_12_offset_vs_p90"],
    s=120
)

axes[1, 1].set_title("Green composition colored by anomaly")
plt.colorbar(sc, ax=axes[1, 1])

plt.tight_layout()
plt.show()

# ======================================================
# FIGURE 3
# DISTRICT HEAT DECOMPOSITION
# ======================================================

plotdf = summary.copy()

plotdf["chronic_warmth"] = (
    plotdf["mean_09_12_p90_temp"] -
    plotdf["mean_09_12_avg_temp"]
)

plotdf["acute_amplification"] = plotdf["mean_09_12_offset_vs_p90"]

fig, ax = plt.subplots(figsize=(14, 10))

xmean = plotdf["chronic_warmth"].mean()
ymean = plotdf["acute_amplification"].mean()

sc = ax.scatter(
    plotdf["chronic_warmth"],
    plotdf["acute_amplification"],
    c=plotdf["tree_share"],
    cmap="YlGn",
    s=160,
    edgecolor="black"
)

ax.axvline(xmean, ls="--", color="gray")
ax.axhline(ymean, ls="--", color="gray")

texts = []
for _, r in plotdf.iterrows():
    texts.append(
        ax.text(
            r["chronic_warmth"],
            r["acute_amplification"],
            r["district"],
            fontsize=8
        )
    )

adjust_text(texts, ax=ax)

ax.set_xlabel("Regular summer heat: p90 − mean July")
ax.set_ylabel("Heatwave amplification: heatwave − p90")
ax.set_title("District green area heat burden decomposition")

plt.colorbar(sc, ax=ax, label="Tree share")

plt.tight_layout()
plt.savefig(OUT_DIR / "district_heat_decomposition_labeled.png", dpi=300)
plt.show()

## ======================================================
# FIGURE 4
# HEATWAVE RASTER MAPS
# WITH COLUMN-WISE PERCENTILE CLIPPING
# ======================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

# ------------------------------------------------------
# SETTINGS
# ------------------------------------------------------
TARGET_HOUR = 9          # 12:00 local summer time
LOW_Q = 2               # lower percentile clip
HIGH_Q = 98             # upper percentile clip

# ------------------------------------------------------
# LOAD GREEN MASK
# ------------------------------------------------------
tree, extent, crs = read_raster(TREE_TIF)
nwn, _, _ = read_raster(NWN_TIF)

valid = (~np.isnan(tree)) | (~np.isnan(nwn))
tree_mask = (np.nan_to_num(tree) > 0) & valid
nwn_mask  = (np.nan_to_num(nwn) > 0) & valid
greenmask = tree_mask | nwn_mask

# transparency weighting
veg = np.clip(np.nan_to_num(tree) + np.nan_to_num(nwn), 0, 1)
alpha = 0.08 + 0.92 * (veg ** 2.2)
alpha[~greenmask] = 0

# ------------------------------------------------------
# LOAD ALL EVENT MAPS
# ------------------------------------------------------
maps = {}

for year, peakday in EVENTS.items():

    evt, _, _ = read_raster(
        PRED_ROOT / year / peakday / tif_name(peakday, TARGET_HOUR)
    )

    avg, _, _ = read_raster(
        BASELINE_AVG / tif_name("20000715", TARGET_HOUR)
    )

    p90, _, _ = read_raster(
        BASELINE_P90 / tif_name("20000715", TARGET_HOUR)
    )

    # apply green mask
    evt[~greenmask] = np.nan
    avg[~greenmask] = np.nan
    p90[~greenmask] = np.nan

    maps[year] = {
        "temp": evt,
        "vs_avg": evt - avg,
        "vs_p90": evt - p90
    }

# ------------------------------------------------------
# BUILD COLUMN-WISE ARRAYS
# ------------------------------------------------------
temp_vals = np.concatenate([
    maps[y]["temp"][~np.isnan(maps[y]["temp"])]
    for y in maps
])

avg_vals = np.concatenate([
    maps[y]["vs_avg"][~np.isnan(maps[y]["vs_avg"])]
    for y in maps
])

p90_vals = np.concatenate([
    maps[y]["vs_p90"][~np.isnan(maps[y]["vs_p90"])]
    for y in maps
])

# ------------------------------------------------------
# PERCENTILE-CLIPPED NORMS
# ------------------------------------------------------
temp_norm = Normalize(
    vmin=np.percentile(temp_vals, LOW_Q),
    vmax=np.percentile(temp_vals, HIGH_Q)
)

avg_norm = Normalize(
    vmin=np.percentile(avg_vals, LOW_Q),
    vmax=np.percentile(avg_vals, HIGH_Q)
)

p90_norm = Normalize(
    vmin=np.percentile(p90_vals, LOW_Q),
    vmax=np.percentile(p90_vals, HIGH_Q)
)

# ------------------------------------------------------
# DRAW FUNCTION
# ------------------------------------------------------
def draw_panel(ax, arr, norm, cmap):

    rgba = cmap(norm(arr))
    rgba[..., -1] = alpha
    rgba[np.isnan(arr), -1] = 0

    ax.imshow(
        rgba,
        extent=extent,
        interpolation="bilinear"
    )

    ax.add_patch(Rectangle(
        (extent[0], extent[2]),
        extent[1] - extent[0],
        extent[3] - extent[2],
        fill=False,
        lw=0.5,
        ec="0.75"
    ))

    ax.axis("off")
    ax.set_aspect("equal")

# ------------------------------------------------------
# FIGURE LAYOUT
# ------------------------------------------------------
fig, axes = plt.subplots(
    3, 3,
    figsize=(18, 10),
    constrained_layout=True
)

years = list(EVENTS.keys())

for i, year in enumerate(years):

    draw_panel(
        axes[i, 0],
        maps[year]["temp"],
        temp_norm,
        plt.cm.inferno
    )

    draw_panel(
        axes[i, 1],
        maps[year]["vs_avg"],
        avg_norm,
        plt.cm.Reds
    )

    draw_panel(
        axes[i, 2],
        maps[year]["vs_p90"],
        p90_norm,
        plt.cm.OrRd
    )

    # row labels
    axes[i, 0].text(
        -0.05, 0.5, year,
        transform=axes[i, 0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=13,
        fontweight="bold"
    )

# ------------------------------------------------------
# COLUMN TITLES
# ------------------------------------------------------
titles = [
    "Absolute modeled temperature",
    "Heatwave − average July day (mean)",
    "Heatwave − hot July day (p90)"
]

for j in range(3):
    axes[0, j].set_title(
        titles[j],
        fontsize=12,
        fontweight="bold"
    )

# ------------------------------------------------------
# COLORBARS
# ------------------------------------------------------
sm1 = plt.cm.ScalarMappable(norm=temp_norm, cmap="inferno")
cb1 = fig.colorbar(
    sm1,
    ax=axes[:, 0],
    orientation="horizontal",
    fraction=0.05,
    pad=0.02
)
cb1.set_label("Temperature (°C)")

sm2 = plt.cm.ScalarMappable(norm=avg_norm, cmap="Reds")
cb2 = fig.colorbar(
    sm2,
    ax=axes[:, 1],
    orientation="horizontal",
    fraction=0.05,
    pad=0.02
)
cb2.set_label("Anomaly")

sm3 = plt.cm.ScalarMappable(norm=p90_norm, cmap="OrRd")
cb3 = fig.colorbar(
    sm3,
    ax=axes[:, 2],
    orientation="horizontal",
    fraction=0.05,
    pad=0.02
)
cb3.set_label("Anomaly")

# ------------------------------------------------------
# TITLE
# ------------------------------------------------------
fig.suptitle(
    f"Spatial heatwave signatures across Helsinki urban green surfaces\n"
    f"12:00 local time",
    fontsize=16,
    fontweight="bold"
)

# ------------------------------------------------------
# SAVE
# ------------------------------------------------------
plt.savefig(
    OUT_DIR / "heatwave_multimap_percentile_clipped.png",
    dpi=400,
    bbox_inches="tight"
)

plt.show()