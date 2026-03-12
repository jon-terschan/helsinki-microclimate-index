import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# this code takes the recent tmax data
# and compares the precalculated CTX90 thresholds against them 
# to create a list and figures of heatwave events within the last 15 years.
recent_dir     = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\recent_tmax")
threshold_file = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\thresholds\CTX90_thresholds.nc")
output_dir     = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\era5\heatwaves")
output_dir.mkdir(parents=True, exist_ok=True)

# --- LOAD DATA ---
print("Loading recent Tmax...")
files  = sorted(recent_dir.glob("ERA5l_tmax_*.nc"))
ds     = xr.open_mfdataset(files, combine="by_coords")
tmax   = ds["t2m"]
if tmax.mean() > 200:
    tmax = tmax - 273.15
    print("Converted K -> C")

print("Loading CTX90 thresholds...")
thresholds = xr.open_dataset(threshold_file)["ctx90_threshold"]  # (md, lat, lon)

# --- SPATIALLY AVERAGE TMAX OVER HELSINKI DOMAIN ---
# Single spatial mean -> one Tmax time series for detection
tmax_spatial = tmax.mean(dim=["latitude", "longitude"])

# --- BUILD THRESHOLD TIME SERIES MATCHING THE RECENT DATA ---
# Match each date to its md string, then look up threshold
dates  = pd.DatetimeIndex(tmax_spatial.valid_time.values)
md_str = dates.strftime("%m-%d")

# Spatially average thresholds too (consistent with tmax_spatial)
thresh_spatial = thresholds.mean(dim=["latitude", "longitude"])  # (md,)

# Build a threshold array aligned to the date axis
thresh_values = np.array([
    float(thresh_spatial.sel(md=md).values) if md in thresh_spatial.md.values else np.nan
    for md in md_str
])

tmax_values = tmax_spatial.values

# --- DETECT HEATWAVE EVENTS ---
print("Detecting heatwave events...")

above_threshold = tmax_values > thresh_values

# Find runs of consecutive days above threshold
events = []
i = 0
while i < len(above_threshold):
    if above_threshold[i]:
        # Find end of this run
        j = i
        while j < len(above_threshold) and above_threshold[j]:
            j += 1
        run_length = j - i
        if run_length >= 3:
            event_dates    = dates[i:j]
            event_tmax     = tmax_values[i:j]
            event_thresh   = thresh_values[i:j]
            event_anomaly  = event_tmax - event_thresh

            events.append({
                "start":           event_dates[0].strftime("%Y-%m-%d"),
                "end":             event_dates[-1].strftime("%Y-%m-%d"),
                "duration_days":   run_length,
                "intensity_mean":  round(float(np.mean(event_anomaly)), 2),   # mean excess above threshold
                "intensity_max":   round(float(np.max(event_tmax)), 2),        # peak Tmax
                "peak_date":       event_dates[np.argmax(event_tmax)].strftime("%Y-%m-%d"),
                "cumulative_heat": round(float(np.sum(event_anomaly)), 2),     # sum of daily anomalies
            })
        i = j
    else:
        i += 1

# --- OUTPUT ---
df = pd.DataFrame(events)
df = df.sort_values("cumulative_heat", ascending=False).reset_index(drop=True)

print(f"\nFound {len(df)} heatwave events (2010-2024, JJA, Helsinki)\n")
print(df.to_string(index=True))

# Save to CSV
csv_out = output_dir / "heatwave_events.csv"
df.to_csv(csv_out, index=False)
print(f"\nSaved to {csv_out}")

# --- GENERAL PLOT OVERVOEW
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

fig, axes = plt.subplots(5, 3, figsize=(18, 14), sharey=True)  # sharey=True now makes sense
axes = axes.flatten()

years = range(2010, 2025)

for idx, year in enumerate(years):
    ax = axes[idx]
    
    mask = dates.year == year
    y_dates  = dates[mask]
    y_tmax   = tmax_values[mask]
    y_thresh = thresh_values[mask]
    y_above  = y_tmax > y_thresh

    ax.plot(y_dates, y_tmax,   color="tomato", lw=1.2, label="Tmax")
    ax.plot(y_dates, y_thresh, color="navy",   lw=1.0, linestyle="--", label="CTX90 threshold")
    ax.fill_between(y_dates, y_thresh, y_tmax, where=y_above,
                    alpha=0.35, color="orange", label="Above threshold")

    year_events = df[df["start"].str.startswith(str(year)) | df["end"].str.startswith(str(year))]
    for _, ev in year_events.iterrows():
        ax.axvspan(pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"]),
                   alpha=0.15, color="red", zorder=0)
        ax.set_title(f"{year}  ★ HW: {ev['start'][5:]} – {ev['end'][5:]} "
                     f"({ev['duration_days']}d, cum={ev['cumulative_heat']:.1f}°C)",
                     fontsize=7.5, color="darkred")

    if not year_events.empty:
        ax.set_facecolor("#fff5f0")

    if year_events.empty:
        ax.set_title(str(year), fontsize=9)

    ax.set_xlim(pd.Timestamp(f"{year}-06-01"), pd.Timestamp(f"{year}-08-31"))
    ax.set_ylim(15, 35)  # <-- fixed y axis
    ax.set_ylabel("°C", fontsize=8)
    ax.tick_params(axis='x', labelrotation=30, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, alpha=0.3)

for idx in range(len(years), len(axes)):
    axes[idx].set_visible(False)

handles = [
    mpatches.Patch(color="tomato",  label="Tmax"),
    mpatches.Patch(color="navy",    label="CTX90 threshold"),
    mpatches.Patch(color="orange",  alpha=0.5, label="Above threshold"),
    mpatches.Patch(color="red",     alpha=0.3, label="Heatwave event (≥3d)"),
]
fig.legend(handles=handles, loc="lower right", fontsize=9)
fig.suptitle("CTX90 Heatwave Detection — Helsinki 2010–2024 (JJA)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(output_dir / "heatwave_overview_grid.png", dpi=150)
plt.show()

## SELECTED YEARS PLOT OVERVIEW
selected_years = [2010, 2018, 2021]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for idx, year in enumerate(selected_years):
    ax = axes[idx]
    
    mask = dates.year == year
    y_dates  = dates[mask]
    y_tmax   = tmax_values[mask]
    y_thresh = thresh_values[mask]
    y_above  = y_tmax > y_thresh

    # Find peak day
    peak_idx   = np.argmax(y_tmax)
    peak_date  = y_dates[peak_idx]
    peak_temp  = y_tmax[peak_idx]

    # --- Cumulative stats over Jun 15 - Aug 20 window ---
    window_mask = (y_dates >= pd.Timestamp(f"{year}-06-15")) & (y_dates <= pd.Timestamp(f"{year}-08-20"))
    w_tmax   = y_tmax[window_mask]
    w_thresh = y_thresh[window_mask]
    w_dates  = y_dates[window_mask]
    w_above  = w_tmax > w_thresh

    hw_days     = int(np.sum(w_above))
    cum_heat    = float(np.sum((w_tmax - w_thresh)[w_above]))
    mean_int    = float(np.mean((w_tmax - w_thresh)[w_above])) if hw_days > 0 else 0
    peak_window = float(np.max(w_tmax))

    stats_title = (f"{year}  |  Days: {hw_days}  |  "
                   f"Cml. heat: {cum_heat:.1f}°C  |  "
                   f"Avg. intensity.: {mean_int:.2f}°C above 90p  |  "
                   f"Peak Tmax: {peak_window:.1f}°C")

    ax.plot(y_dates, y_tmax,   color="tomato", lw=1.2)
    ax.plot(y_dates, y_thresh, color="navy",   lw=1.0, linestyle="--")
    ax.fill_between(y_dates, y_thresh, y_tmax, where=y_above,
                    alpha=0.35, color="orange")

    # Shade the analysis window
    ax.axvspan(pd.Timestamp(f"{year}-06-15"), pd.Timestamp(f"{year}-08-20"),
               alpha=0.05, color="blue", zorder=0)

    # Mark peak day
    ax.axvline(peak_date, color="darkred", lw=1.5, linestyle=":", zorder=5)
    ax.scatter([peak_date], [peak_temp], color="darkred", zorder=6, s=40)
    ax.annotate(f"Peak: {peak_date.strftime('%m-%d')}\n{peak_temp:.1f}°C",
                xy=(peak_date, peak_temp),
                xytext=(10, -25), textcoords="offset points",
                fontsize=7, color="darkred",
                arrowprops=dict(arrowstyle="-", color="darkred", lw=0.8))

    year_events = df[df["start"].str.startswith(str(year)) | df["end"].str.startswith(str(year))]
    for _, ev in year_events.iterrows():
        ax.axvspan(pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"]),
                   alpha=0.15, color="red", zorder=0)

    ax.set_title(stats_title, fontsize=7.5, color="darkred")
    ax.set_facecolor("#fff5f0")
    ax.set_xlim(pd.Timestamp(f"{year}-06-01"), pd.Timestamp(f"{year}-08-31"))
    ax.set_ylim(15, 35)
    ax.set_ylabel("°C", fontsize=9)
    ax.tick_params(axis='x', labelrotation=30, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, alpha=0.3)

handles = [
    mpatches.Patch(color="tomato",  label="Tmax"),
    mpatches.Patch(color="navy",    label="90th percentile"),
    mpatches.Patch(color="orange",  alpha=0.5, label="> 90th percentile"),
    mpatches.Patch(color="red",     alpha=0.3, label="Event (>=3d)"),
]
fig.legend(handles=handles, loc="lower right", fontsize=9)
fig.suptitle("Helsinki heatwaves 2010, 2018, 2021", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(output_dir / "heatwave_selected.png", dpi=150)
plt.show()