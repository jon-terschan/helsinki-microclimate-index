# ============================================================
# FOREST TRANSECT SINGLE-PANEL FIGURE
# Local-window sampling version
#
# One figure per transect.
# Panel content:
#   - Heatwave anomaly (mean of 2010/2018/2021 minus p90)
#   - DTM
#   - CHM
#   - UCC
#   - ROCK_FRAC
#
# Sampling uses a local square window around each point
# instead of a single pixel to reduce line-placement sensitivity.
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import plotting_extent
from rasterio.transform import rowcol

import matplotlib.pyplot as plt
import seaborn as sns

from shapely.geometry import box
from shapely.ops import linemerge

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
PRED_ROOT = BASE / "predictions"
STACK = BASE / "predictorstack"

TRANSECTS_GPKG = BASE / "figures" / "forest_transects_length.gpkg"
OUT_DIR = BASE / "figures" / "forest_transects_singlepanel_localmean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DTM = STACK / "DTM_10m_Helsinki.tif"
CHM = STACK / "CHM_10m_MAX.tif"
ROCK = STACK / "ROCK_FRAC_10m_Helsinki.tif"
UCC_RASTER = STACK / "CM_loc_Hel.tif"   # band 4 = UCC
P90 = PRED_ROOT / "baseline" / "15cm_July_allday_p90" / "pred_20000715_0900.tif"

EVENTS = {
    "2010": PRED_ROOT / "2010" / "20100728" / "pred_20100728_0900.tif",
    "2018": PRED_ROOT / "2018" / "20180717" / "pred_20180717_0900.tif",
    "2021": PRED_ROOT / "2021" / "20210714" / "pred_20210714_0900.tif",
}

# ============================================================
# SETTINGS
# ============================================================

UCC_BAND = 4
STEP_M = 10

# local mean radius in pixels:
# 2 => 5x5 window, 1 => 3x3 window
LOCAL_RADIUS_PX = 2

# smoothing only for plotting
SMOOTH_WINDOW = 7

# Set to 0, 1, 2, ... for a single transect.
# Set to None to process all transects.
PLOT_TRANSECT_INDEX = None

sns.set_theme(style="whitegrid", context="talk")

# ============================================================
# HELPERS
# ============================================================

def read_raster(path, band=1):
    with rasterio.open(path) as src:
        arr = src.read(band).astype(np.float32)
        nodata = src.nodata
        extent = plotting_extent(src)
        transform = src.transform
        crs = src.crs

    if nodata is not None:
        arr[arr == nodata] = np.nan

    return arr, extent, transform, crs


def sample_local_mean(arr, transform, coords, radius_px=2):
    """
    Mean over a square local window around each coordinate.
    """
    h, w = arr.shape
    out = np.full(len(coords), np.nan, dtype=float)

    for i, (x, y) in enumerate(coords):
        r, c = rowcol(transform, x, y)

        r0 = max(0, r - radius_px)
        r1 = min(h, r + radius_px + 1)
        c0 = max(0, c - radius_px)
        c1 = min(w, c + radius_px + 1)

        win = arr[r0:r1, c0:c1]
        if np.isfinite(win).any():
            out[i] = np.nanmean(win)

    return out


def sample_points_along_line(line, step_m=10):
    length = float(line.length)
    dists = np.arange(0, length, step_m, dtype=float)
    if len(dists) == 0 or dists[-1] < length:
        dists = np.append(dists, length)

    pts = [line.interpolate(d) for d in dists]
    coords = [(p.x, p.y) for p in pts]
    return dists, coords


def smooth_series(values, window=7):
    s = pd.Series(values, dtype=float)
    s = s.interpolate(limit_direction="both")
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def norm01(values):
    v = np.asarray(values, dtype=float)
    lo = np.nanpercentile(v, 2)
    hi = np.nanpercentile(v, 98)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(v), lo, hi

    out = (v - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return out, lo, hi


def map_to_band(values, y0, y1, lo, hi):
    v = np.asarray(values, dtype=float)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.full_like(v, (y0 + y1) / 2.0)

    z = (v - lo) / (hi - lo)
    z = np.clip(z, 0, 1)
    return y0 + z * (y1 - y0)


def label_from_row(row, idx):
    for col in ["site", "site_name", "name", "transect", "id", "label"]:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return f"Transect {idx + 1}"


def single_linestring(geom):
    """
    Return a single LineString from LineString/MultiLineString/GeometryCollection.
    """
    if geom.is_empty:
        return None

    if geom.geom_type == "LineString":
        return geom

    if geom.geom_type == "MultiLineString":
        return max(list(geom.geoms), key=lambda g: g.length)

    if geom.geom_type == "GeometryCollection":
        lines = [g for g in geom.geoms if g.geom_type == "LineString"]
        if lines:
            return max(lines, key=lambda g: g.length)

    try:
        merged = linemerge(geom)
        if merged.geom_type == "LineString":
            return merged
        if merged.geom_type == "MultiLineString":
            return max(list(merged.geoms), key=lambda g: g.length)
    except Exception:
        pass

    return None


# ============================================================
# LOAD RASTERS
# ============================================================

assert DTM.exists(), f"Missing DTM: {DTM}"
assert CHM.exists(), f"Missing CHM: {CHM}"
assert ROCK.exists(), f"Missing ROCK: {ROCK}"
assert UCC_RASTER.exists(), f"Missing UCC raster: {UCC_RASTER}"
assert P90.exists(), f"Missing P90 raster: {P90}"

dtm, dtm_extent, dtm_transform, dtm_crs = read_raster(DTM)
chm, _, _, _ = read_raster(CHM)
rock, _, _, _ = read_raster(ROCK)
ucc, _, _, _ = read_raster(UCC_RASTER, band=UCC_BAND)
p90, _, _, _ = read_raster(P90)

shape = dtm.shape
for arr, name in [
    (chm, "CHM"),
    (rock, "ROCK"),
    (ucc, "UCC"),
    (p90, "P90"),
]:
    if arr.shape != shape:
        raise ValueError(f"{name} shape mismatch: {arr.shape} vs {shape}")

# ============================================================
# LOAD TRANSECTS
# ============================================================

transects = gpd.read_file(TRANSECTS_GPKG)

if transects.empty:
    raise ValueError("Transect GeoPackage is empty.")

transects = transects[
    transects.geometry.type.isin(["LineString", "MultiLineString"])
].copy()

if transects.crs is None:
    raise ValueError("Transects layer has no CRS.")

transects = transects.to_crs(dtm_crs).reset_index(drop=True)

print("Transects loaded:", len(transects))
print("Transect columns:", list(transects.columns))

if PLOT_TRANSECT_INDEX is None:
    indices = list(range(len(transects)))
else:
    indices = [PLOT_TRANSECT_INDEX]

# ============================================================
# OVERVIEW MAP
# ============================================================

fig, ax = plt.subplots(figsize=(12, 11))
ax.imshow(dtm, extent=dtm_extent, cmap="Greys", alpha=0.85)
ax.set_title("Transects over DTM")
ax.axis("off")

colors = sns.color_palette("tab10", n_colors=len(transects))
bbox = box(dtm_extent[0], dtm_extent[2], dtm_extent[1], dtm_extent[3])

for i, row in transects.iterrows():
    geom = row.geometry
    label = label_from_row(row, i)

    clipped = geom.intersection(bbox)
    if clipped.is_empty:
        continue

    clipped = single_linestring(clipped)
    if clipped is None:
        continue

    xs, ys = clipped.xy
    ax.plot(xs, ys, lw=2.5, color=colors[i], label=label)

    mid = clipped.interpolate(0.5, normalized=True)
    ax.text(
        mid.x, mid.y, str(i + 1),
        color="black",
        fontsize=10,
        weight="bold",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5)
    )

ax.legend(loc="lower left", frameon=True, fontsize=9, title="Transects")
plt.tight_layout()
plt.savefig(OUT_DIR / "transects_overview_map.png", dpi=350, bbox_inches="tight")
plt.show()

# ============================================================
# EXTRACT AND PLOT TRANSECT PROFILES
# ============================================================

all_profiles = []

for idx in indices:
    row = transects.iloc[idx]
    label = label_from_row(row, idx)

    geom = row.geometry.intersection(bbox)
    if geom.is_empty:
        print(f"Skipping transect {idx + 1}: empty after clipping.")
        continue

    geom = single_linestring(geom)
    if geom is None:
        print(f"Skipping transect {idx + 1}: unsupported geometry.")
        continue

    dists, coords = sample_points_along_line(geom, step_m=STEP_M)

    # Local-window sampling for each raster
    dtm_vals = sample_local_mean(dtm, dtm_transform, coords, radius_px=LOCAL_RADIUS_PX)
    chm_vals = sample_local_mean(chm, dtm_transform, coords, radius_px=LOCAL_RADIUS_PX)
    ucc_vals = sample_local_mean(ucc, dtm_transform, coords, radius_px=LOCAL_RADIUS_PX)
    rock_vals = sample_local_mean(rock, dtm_transform, coords, radius_px=LOCAL_RADIUS_PX)
    p90_vals = sample_local_mean(p90, dtm_transform, coords, radius_px=LOCAL_RADIUS_PX)

    temp_year = {}
    anomaly_year = {}

    for year, fp in EVENTS.items():
        evt_vals = sample_local_mean(
            read_raster(fp)[0],
            dtm_transform,
            coords,
            radius_px=LOCAL_RADIUS_PX
        )
        temp_year[year] = evt_vals
        anomaly_year[year] = evt_vals - p90_vals

    anomaly_stack = np.vstack([anomaly_year[y] for y in EVENTS.keys()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        anomaly_mean = np.nanmean(anomaly_stack, axis=0)

    prof = pd.DataFrame({
        "transect_index": idx + 1,
        "transect_label": label,
        "distance_m": dists,
        "distance_km": dists / 1000.0,
        "dtm": dtm_vals,
        "chm": chm_vals,
        "ucc": ucc_vals,
        "rock": rock_vals,
        "p90": p90_vals,
        "anomaly_2010": anomaly_year["2010"],
        "anomaly_2018": anomaly_year["2018"],
        "anomaly_2021": anomaly_year["2021"],
        "anomaly_mean": anomaly_mean,
    })

    for col in ["dtm", "chm", "ucc", "rock", "anomaly_mean"]:
        prof[f"{col}_sm"] = smooth_series(prof[col].to_numpy(), window=SMOOTH_WINDOW)

    all_profiles.append(prof)
    prof.to_csv(OUT_DIR / f"transect_{idx + 1:02d}_profile.csv", index=False)

    # ============================================================
    # SINGLE-PANEL FIGURE
    # ============================================================

    bands = {
        "anomaly": (0.0, 1.0),
        "dtm": (1.2, 2.2),
        "chm": (2.4, 3.4),
        "ucc": (3.6, 4.6),
        "rock": (4.8, 5.8),
    }

    x = prof["distance_m"].to_numpy()

    # anomaly scaling for the mean anomaly line only
    anom_y = map_to_band(
        prof["anomaly_mean_sm"],
        *bands["anomaly"],
        *np.nanpercentile(prof["anomaly_mean_sm"], [2, 98])
    )

    dtm_y = map_to_band(
        prof["dtm_sm"],
        *bands["dtm"],
        *np.nanpercentile(prof["dtm_sm"], [2, 98])
    )

    chm_y = map_to_band(
        prof["chm_sm"],
        *bands["chm"],
        *np.nanpercentile(prof["chm_sm"], [2, 98])
    )

    ucc_y = map_to_band(
        prof["ucc_sm"],
        *bands["ucc"],
        *np.nanpercentile(prof["ucc_sm"], [2, 98])
    )

    rock_y = map_to_band(
        prof["rock_sm"],
        *bands["rock"],
        *np.nanpercentile(prof["rock_sm"], [2, 98])
    )

    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor("white")

    # Background bands
    band_fills = [
        (bands["anomaly"], "#fff5f0"),
        (bands["dtm"], "#f2f2f2"),
        (bands["chm"], "#f1faef"),
        (bands["ucc"], "#ecfbf8"),
        (bands["rock"], "#fff8ef"),
    ]
    for (y0, y1), color in band_fills:
        ax.axhspan(y0, y1, color=color, alpha=1.0, zorder=0)

    # Main anomaly line only
    ax.plot(x, anom_y, color="#C00000", lw=3.2, label="Mean anomaly")

    # Other layers
    ax.plot(x, dtm_y, color="0.35", lw=2.2, label="DTM")
    ax.plot(x, chm_y, color="#1B5E20", lw=2.4, label="CHM")
    ax.plot(x, ucc_y, color="#1F9D8A", lw=2.4, label="UCC")
    ax.plot(x, rock_y, color="#8C4B1F", lw=2.4, label="Rock fraction")

    peak_idx = int(np.nanargmax(prof["anomaly_mean_sm"].to_numpy()))
    peak_dist = float(dists[peak_idx])
    peak_val = float(prof.loc[peak_idx, "anomaly_mean_sm"])

    ax.axvline(peak_dist, color="#C00000", ls="--", lw=1.4, alpha=0.8)

    # Left-side labels
    def band_label(y0, y1, text, color="black"):
        ax.text(
            0.008, (y0 + y1) / 2,
            text,
            transform=ax.get_yaxis_transform(),
            ha="left", va="center",
            fontsize=11,
            fontweight="bold",
            color=color
        )

    band_label(*bands["anomaly"], "Heatwave anomaly", color="#C00000")
    band_label(*bands["dtm"], "DTM", color="0.25")
    band_label(*bands["chm"], "Canopy height (CHM)", color="#1B5E20")
    band_label(*bands["ucc"], "Upper canopy cover (UCC)", color="#1F9D8A")
    band_label(*bands["rock"], "Rock fraction", color="#8C4B1F")

    # Right-side range labels
    def range_label(y0, y1, arr, unit):
        lo, hi = np.nanpercentile(arr, [2, 98])
        unit_text = f" {unit}" if unit else ""
        ax.text(
            0.992, (y0 + y1) / 2,
            f"{lo:.2f}–{hi:.2f}{unit_text}",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center",
            fontsize=9,
            color="0.35"
        )

    range_label(*bands["anomaly"], prof["anomaly_mean_sm"], "°C")
    range_label(*bands["dtm"], prof["dtm_sm"], "m")
    range_label(*bands["chm"], prof["chm_sm"], "m")
    range_label(*bands["ucc"], prof["ucc_sm"], "")
    range_label(*bands["rock"], prof["rock_sm"], "")

    # Title and annotation
    ax.text(
        0.01, 0.985,
        f"{label} | transect {idx + 1} | length {geom.length/1000:.2f} km",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=2)
    )

    ax.text(
        0.99, 0.985,
        f"Peak mean anomaly at {peak_dist:.0f} m: {peak_val:.2f} °C",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=2)
    )

    ax.text(
        0.99, 0.01,
        f"Local mean window: {2 * LOCAL_RADIUS_PX + 1}×{2 * LOCAL_RADIUS_PX + 1} pixels",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        color="0.35"
    )

    # Formatting
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.05, bands["rock"][1] + 0.15)
    ax.set_xlabel("Distance along transect (m)")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.15)

    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)

    ax.legend(
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=9
    )

    plt.tight_layout()

    out_fig = OUT_DIR / f"transect_{idx + 1:02d}_singlepanel.png"
    plt.savefig(out_fig, dpi=350, bbox_inches="tight")
    plt.show()

    print("Saved:", out_fig)

# ============================================================
# SAVE COMBINED TABLES
# ============================================================

if all_profiles:
    all_profiles_df = pd.concat(all_profiles, ignore_index=True)
    all_profiles_df.to_csv(OUT_DIR / "transect_all_profiles.csv", index=False)

    summary_rows = []
    for tid, g in all_profiles_df.groupby("transect_index"):
        summary_rows.append({
            "transect_index": tid,
            "transect_label": g["transect_label"].iloc[0],
            "length_km": float(g["distance_m"].max() / 1000.0),
            "mean_anomaly": float(np.nanmean(g["anomaly_mean"])),
            "peak_anomaly": float(np.nanmax(g["anomaly_mean"])),
            "peak_distance_m": float(g.loc[g["anomaly_mean"].idxmax(), "distance_m"]),
            "mean_dtm": float(np.nanmean(g["dtm"])),
            "mean_chm": float(np.nanmean(g["chm"])),
            "mean_ucc": float(np.nanmean(g["ucc"])),
            "mean_rock": float(np.nanmean(g["rock"])),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "transect_summary.csv", index=False)

    print("Saved:", OUT_DIR / "transect_all_profiles.csv")
    print("Saved:", OUT_DIR / "transect_summary.csv")
    print("Saved:", OUT_DIR / "transects_overview_map.png")