import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# FIGURE 2X1 SHOWING WHAT HAPPENS IN JULY IN OUR TRAINING DATA
# THIS NEEDS TO BE OVERWORKED COLOR AND COMPSOTION WISE 
# AND COULD BE 2x2 WITH VALIDATION FIGURE

# ==================================================
# PATHS
# ==================================================
data_dir = Path(r"//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/01_traindataprep/exports_july")
out_dir  = Path(r"//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/figures/drafts")
out_dir.mkdir(parents=True, exist_ok=True)

hourly_file = data_dir / "july_hourly_profile_local.csv"
raw_file    = data_dir / "train_data_july_utc.csv"
# ======================================================
# LOAD
# ======================================================
hourly = pd.read_csv(hourly_file)
df = pd.read_csv(raw_file)

# ======================================================
# TIME HANDLING
# keep source data in UTC, convert only for summaries
# ======================================================
time_col = "time_utc" if "time_utc" in df.columns else "time"

df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
df["time_local"] = df[time_col].dt.tz_convert("Europe/Helsinki")
df["hour_local"] = df["time_local"].dt.hour

# ======================================================
# CLEAN VARIABLES
# ======================================================
# uCC may be 0-1 or 0-100
ucc = df["uCC"].astype(float).copy()
if ucc.max() <= 1.5:
    ucc = ucc * 100.0

df["uCC_pct"] = ucc

# ======================================================
# DAYTIME FILTER (10-18 local)
# ======================================================
df_day = df.loc[
    (df["hour_local"] >= 10) &
    (df["hour_local"] <= 18) &
    df["temp"].notna() &
    df["uCC_pct"].notna()
].copy()

# ======================================================
# SITE-LEVEL SUMMARIES
# one point per sensor
# ======================================================
site = (
    df_day.groupby("sensor_id", as_index=False)
    .agg(
        mean_temp=("temp", "mean"),
        mean_ucc=("uCC_pct", "mean"),
        n_obs=("temp", "size")
    )
)

# ======================================================
# BIN SITE MEANS FOR SMOOTH PANEL B
# ======================================================
bins = np.arange(0, 105, 10)
centers = (bins[:-1] + bins[1:]) / 2

site["bin"] = pd.cut(site["mean_ucc"], bins=bins, include_lowest=True)

binned = (
    site.groupby("bin", observed=False)
    .agg(
        mean_temp=("mean_temp", "mean"),
        sd=("mean_temp", "std"),
        n_sites=("mean_temp", "size")
    )
    .reset_index(drop=True)
)

binned["x"] = centers
binned["se"] = binned["sd"] / np.sqrt(binned["n_sites"])
binned["lo"] = binned["mean_temp"] - 1.96 * binned["se"]
binned["hi"] = binned["mean_temp"] + 1.96 * binned["se"]

# interpolate gaps for visual continuity
for c in ["mean_temp", "lo", "hi"]:
    binned[c] = binned[c].interpolate(limit_direction="both")

# histogram = site counts (not observations)
site_counts, _ = np.histogram(site["mean_ucc"], bins=bins)

# ======================================================
# STYLE
# ======================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300
})

fig, axes = plt.subplots(
    2, 1,
    figsize=(7.4, 8.0),
    constrained_layout=True
)

# ======================================================
# PANEL A
# ======================================================
ax = axes[0]

x = hourly["hour_local"].values
y = hourly["mean_temp"].values
sd = hourly["sd_temp"].values

ax.plot(x, y, lw=2.5)
ax.fill_between(x, y - sd, y + sd, alpha=0.22)


ax.set_xlim(0, 23)
ax.set_xticks(np.arange(0, 24, 3))
ax.set_xlabel("Hour (local time)")
ax.set_ylabel("15cm air temperature (°C)")
ax.set_title("A. Mean July diurnal temperature cycle", loc="left")

# ======================================================
# PANEL B  (SITE-LEVEL, SOLID LIGHT RIBBON + LINE ON TOP)
# replace only Panel B block
# ======================================================

ax = axes[1]

# ---------------------------------
# make main axis draw above twin axis
# ---------------------------------
ax.set_zorder(3)
ax.patch.set_alpha(0)

# ---------------------------------
# 1. histogram (furthest back)
# ---------------------------------
# ---------------------------------
# 1. histogram (furthest back)
# ---------------------------------
from matplotlib.ticker import MaxNLocator

ax2 = ax.twinx()
ax2.set_zorder(1)

ax2.bar(
    centers,
    site_counts,
    width=8.5,
    color="#1f77b4",
    alpha=0.10,
    edgecolor="none",
    zorder=1
)

ax2.set_ylabel("Sensor count")
ax2.set_ylim(0, max(site_counts) * 1.20)

# force integer ticks only (no decimals like 5.0)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

ax2.grid(False)

# ---------------------------------
# 2. ribbon (behind line/points)
# solid light blue, no transparency needed
# ---------------------------------
ax.fill_between(
    binned["x"],
    binned["lo"],
    binned["hi"],
    color="#c7dcef",   # light hue
    alpha=1.0,
    linewidth=0,
    zorder=2
)

# ---------------------------------
# 3. mean line
# ---------------------------------
ax.plot(
    binned["x"],
    binned["mean_temp"],
    color="#1f77b4",
    lw=2.6,
    zorder=4
)

# ---------------------------------
# 4. mean points / site observations (top layer)
# ---------------------------------
ax.scatter(
    site["mean_ucc"],
    site["mean_temp"],
    s=28,
    color="#ff7f0e",
    edgecolor="white",
    linewidth=0.35,
    alpha=0.95,
    zorder=5
)

# ---------------------------------
# formatting
# ---------------------------------
ax.set_xlim(0, 100)
ax.set_xlabel("Mean upper canopy cover (%)")
ax.set_ylabel("Mean 15cm air temperature (°C)")
ax.set_title(
    "B. Recorded site-level July temperatures along the canopy gradient",
    loc="left"
)

ymin = min(site["mean_temp"].min(), binned["lo"].min()) - 0.15
ymax = max(site["mean_temp"].max(), binned["hi"].max()) + 0.15
ax.set_ylim(ymin, ymax)

ax.grid(axis="y", alpha=0.15)
ax.grid(axis="x", alpha=0)
# ======================================================
# SAVE
# ======================================================
png_file = out_dir / "Figure_trainingdata_sitelevel.png"
pdf_file = out_dir / "Figure_trainingdata_sitelevel.pdf"

fig.savefig(png_file, dpi=450, bbox_inches="tight")
fig.savefig(pdf_file, bbox_inches="tight")

plt.show()

print("Saved:")
print(png_file)
print(pdf_file)