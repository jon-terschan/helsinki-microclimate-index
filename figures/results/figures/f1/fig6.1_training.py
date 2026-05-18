# ============================================================
# FINAL RESULTS FIGURE PANELS
#
# Exports:
#   Panel A: July diurnal temperature cycle
#   Panel B: Site-level July temperatures along canopy gradient
#   Panel C: Temporal evolution of model RMSE
#   Panel D: Diurnal pattern of model bias
#
# Each panel is exported individually as PNG, PDF, SVG.
# A combined 2x2 preview is shown but not exported.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

# ============================================================
# OPTIONAL PACKAGES
# ============================================================

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    lowess = None

try:
    import pyreadr
except ImportError:
    pyreadr = None


# ============================================================
# PATHS
# ============================================================

out_dir = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures\f1"
)
out_dir.mkdir(parents=True, exist_ok=True)

hourly_file = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\figures\r1_figure\july_hourly_profile_local.csv"
)

raw_file = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\figures\r1_figure\train_data_july_utc.csv"
)

validation_csv = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\validation\dt_validation_post-pre.csv"
)

validation_rds = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\validation\dt_validation_post-pre.rds"
)

required_training_files = {
    "hourly_file": hourly_file,
    "raw_file": raw_file,
}

missing_training = [
    name for name, path in required_training_files.items()
    if not path.exists()
]

if missing_training:
    print("Missing required training-data files:")
    for name in missing_training:
        print(f"  {name}: {required_training_files[name]}")
    raise FileNotFoundError("One or more required training-data files are missing.")

if not validation_csv.exists() and not validation_rds.exists():
    print("Missing validation input file. Expected one of:")
    print(f"  CSV: {validation_csv}")
    print(f"  RDS: {validation_rds}")
    raise FileNotFoundError("No validation CSV or RDS file found.")

if validation_csv.exists():
    validation_source = "csv"
elif validation_rds.exists():
    validation_source = "rds"
else:
    validation_source = None

print("Using input files:")
print(f"  hourly_file:       {hourly_file}")
print(f"  raw_file:          {raw_file}")
print(f"  validation source: {validation_source}")
print(f"  validation_csv:    {validation_csv}")
print(f"  validation_rds:    {validation_rds}")
print(f"  out_dir:           {out_dir}")


# ============================================================
# GLOBAL STYLE / EXPORT LOGIC
# ============================================================
# This script intentionally does not define font sizes, rcParams, canvas size,
# or export formats locally. Those come from:
#   \\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures\global_plotting_settings.py

import sys
import importlib
from dataclasses import replace

GLOBAL_STYLE_DIR = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\figures\results\figures"
)

if str(GLOBAL_STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(GLOBAL_STYLE_DIR))

try:
    import global_plotting_settings as gps  # noqa: E402
    # Critical for interactive consoles: after editing global_plotting_settings.py,
    # reload it so this run does not use stale cached values.
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
# Canvas size, font sizes, line widths, colours, and DPI are otherwise inherited
# from global_plotting_settings.STYLE.
STYLE = replace(
    gps.STYLE,
    export_png=True,
    export_pdf=False,
    export_svg=True,
)

# Figure-specific axes rectangle inside the fixed panel canvas.
# This is layout geometry, not typography/export policy.
PANEL_AXES_RECT = (0.13, 0.16, 0.76, 0.76)
STYLE = replace(STYLE, axes_rect=PANEL_AXES_RECT)

# Apply global rcParams explicitly with this resolved STYLE.
gps.apply_style(STYLE)

CM_PER_INCH = gps.CM_PER_INCH

# Local aliases keep the panel drawing code readable, while values still come
# from global_plotting_settings.STYLE.
FONT_FAMILY = STYLE.font_family
DPI_EXPORT = STYLE.dpi_export

COL_BLUE = STYLE.col_blue
COL_BLUE_FILL = STYLE.col_blue_fill
COL_HIST = STYLE.col_hist
COL_HIST_FILL = STYLE.col_hist_fill
COL_RED = STYLE.col_red
COL_RED_FILL = STYLE.col_red_fill
COL_GREY = STYLE.col_grey
COL_GRID = STYLE.col_grid
COL_BLACK = STYLE.col_black

print("Resolved global plotting style:")
print(f"  global file:  {Path(gps.__file__)}")
print(f"  output dir:   {out_dir}")
print(f"  canvas:      {STYLE.panel_width_cm} × {STYLE.panel_height_cm} cm")
print(f"  dpi_export:  {STYLE.dpi_export}")
print(f"  fs_base:     {STYLE.fs_base}")
print(f"  fs_tick:     {STYLE.fs_tick}")
print(f"  fs_axis:     {STYLE.fs_axis}")
print(f"  fs_legend:   {STYLE.fs_legend}")
print(f"  exports:     PNG={STYLE.export_png}, SVG={STYLE.export_svg}, PDF={STYLE.export_pdf}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def new_panel_figure():
    return gps.new_panel_figure(STYLE)


def add_panel_axes(fig):
    return gps.add_panel_axes(fig, rect=STYLE.axes_rect, style=STYLE)


def style_axis(ax):
    gps.style_axis(ax, STYLE)


def set_axis_labels(ax, xlabel=None, ylabel=None):
    gps.set_axis_labels(ax, xlabel=xlabel, ylabel=ylabel, style=STYLE)


def add_small_legend(ax, handles, loc="upper right"):
    return gps.add_legend(ax, handles=handles, loc=loc, style=STYLE)


def make_figure_transparent(fig):
    gps.make_transparent(fig)


def save_single_panel(fig, filename_stem):
    """Save one fixed-canvas panel as PNG and SVG only."""
    make_figure_transparent(fig)

    save_kwargs = {
        "transparent": STYLE.transparent,
        "facecolor": "none",
        "edgecolor": "none",
    }

    if STYLE.use_tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = STYLE.pad_inches

    width_cm = fig.get_figwidth() * CM_PER_INCH
    height_cm = fig.get_figheight() * CM_PER_INCH

    if STYLE.export_png:
        png = out_dir / f"{filename_stem}.png"
        fig.savefig(png, dpi=STYLE.dpi_export, **save_kwargs)
        print(f"[OK] wrote {png} ({width_cm:.2f} × {height_cm:.2f} cm canvas)")

    if STYLE.export_svg:
        svg = out_dir / f"{filename_stem}.svg"
        fig.savefig(svg, **save_kwargs)
        print(f"[OK] wrote {svg} ({width_cm:.2f} × {height_cm:.2f} cm canvas)")


def smooth_lowess(x, y, frac=0.25):
    x = np.asarray(x)
    y = np.asarray(y)

    ok = np.isfinite(x) & np.isfinite(y)
    x_ok = x[ok]
    y_ok = y[ok]

    if len(x_ok) < 5:
        return x, y

    order = np.argsort(x_ok)
    x_sorted = x_ok[order]
    y_sorted = y_ok[order]

    if lowess is not None:
        sm = lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)
        return sm[:, 0], sm[:, 1]

    s = pd.Series(y_sorted)
    win = max(5, int(len(s) * frac))

    if win % 2 == 0:
        win += 1

    y_smooth = s.rolling(
        win,
        center=True,
        min_periods=1
    ).mean().values

    return x_sorted, y_smooth


def read_validation_table(validation_csv, validation_rds):
    """
    Prefer CSV if it exists.
    If CSV does not exist:
      1. Try pyreadr.
      2. If pyreadr is unavailable, call Rscript to convert RDS -> CSV.
    """
    if validation_csv.exists():
        print(f"Reading validation CSV: {validation_csv}")
        return pd.read_csv(validation_csv)

    if not validation_rds.exists():
        raise FileNotFoundError(
            f"No validation CSV or RDS file found:\n"
            f"  CSV: {validation_csv}\n"
            f"  RDS: {validation_rds}"
        )

    if pyreadr is not None:
        print(f"Reading validation RDS with pyreadr: {validation_rds}")
        rds = pyreadr.read_r(str(validation_rds))

        if len(rds) == 0:
            raise ValueError("The RDS file was read, but no object was returned.")

        dt = list(rds.values())[0].copy()

        try:
            dt.to_csv(validation_csv, index=False)
            print(f"Wrote validation CSV for future runs: {validation_csv}")
        except Exception as e:
            print(f"Could not write validation CSV copy: {e}")

        return dt

    print("pyreadr is not installed. Trying to convert RDS to CSV using Rscript...")

    import subprocess
    import shutil

    rscript = shutil.which("Rscript")

    if rscript is None:
        raise ImportError(
            "Validation CSV was not found, pyreadr is not installed, "
            "and Rscript was not found on PATH.\n\n"
            "Fix one of these:\n"
            "  1. Install pyreadr: python -m pip install pyreadr\n"
            "  2. Run the R export manually\n"
            "  3. Add Rscript to your system PATH"
        )

    rds_str = str(validation_rds).replace("\\", "/")
    csv_str = str(validation_csv).replace("\\", "/")

    r_code = f'''
    library(data.table)
    dt <- readRDS("{rds_str}")
    fwrite(dt, "{csv_str}")
    '''

    result = subprocess.run(
        [rscript, "-e", r_code],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Rscript stdout:")
        print(result.stdout)
        print("Rscript stderr:")
        print(result.stderr)
        raise RuntimeError("Rscript failed while converting validation RDS to CSV.")

    if not validation_csv.exists():
        raise FileNotFoundError(
            "Rscript completed, but the validation CSV was still not created."
        )

    print(f"Created validation CSV: {validation_csv}")
    return pd.read_csv(validation_csv)


# ============================================================
# LOAD AND PREPARE TRAINING DATA
# ============================================================

hourly_source = pd.read_csv(hourly_file)  # kept for input/path check; Panel A is rebuilt from raw UTC data
df = pd.read_csv(raw_file)

required_raw_cols = {"uCC", "temp", "sensor_id"}

missing_raw_cols = required_raw_cols - set(df.columns)

if missing_raw_cols:
    raise ValueError(f"Missing columns in raw file: {missing_raw_cols}")

time_col = "time_utc" if "time_utc" in df.columns else "time"

if time_col not in df.columns:
    raise ValueError("Raw training file must contain either 'time_utc' or 'time'.")

df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
df["time_local"] = df[time_col].dt.tz_convert("Europe/Helsinki")
df["hour_local"] = df["time_local"].dt.hour
df["local_month"] = df["time_local"].dt.month


def build_local_hourly_temperature_profile(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Build Panel A directly from raw UTC timestamps after converting to Helsinki
    local time. This avoids UTC/local-day boundary problems in pre-aggregated
    hourly tables.
    """
    panel_a_df = df_in.loc[
        (df_in["local_month"] == 7)
        & df_in["temp"].notna()
        & df_in["hour_local"].notna()
    ].copy()

    hourly_profile = (
        panel_a_df
        .groupby("hour_local", as_index=True)
        .agg(
            mean_temp=("temp", "mean"),
            sd_temp=("temp", "std"),
            n_obs=("temp", "size"),
        )
        .reindex(np.arange(24))
        .rename_axis("hour_local")
        .reset_index()
    )

    hourly_profile["mean_temp"] = hourly_profile["mean_temp"].interpolate(
        limit_direction="both"
    )
    hourly_profile["sd_temp"] = hourly_profile["sd_temp"].interpolate(
        limit_direction="both"
    )

    print("Panel A temperature local-hour coverage:")
    print(hourly_profile[["hour_local", "n_obs"]].to_string(index=False))

    return hourly_profile


def find_ssrd_column(df_in: pd.DataFrame) -> str | None:
    """
    Find the SSRD / incoming radiation column in the raw training table.

    The printout below makes this easy to fix if the project uses a different
    column name.
    """
    exact_candidates = [
        "SSRD",
        "ssrd",
        "ssrd_mean",
        "SSRD_mean",
        "SSRD_J",
        "ssrd_J",
        "ssrd_wm2",
        "SSRD_Wm2",
        "surface_solar_radiation_downwards",
        "surface_solar_radiation_downward",
        "shortwave",
        "shortwave_radiation",
        "solar_radiation",
        "global_radiation",
        "rad",
        "radiation",
    ]

    for col in exact_candidates:
        if col in df_in.columns:
            return col

    for col in df_in.columns:
        key = str(col).lower()
        if (
            "ssrd" in key
            or "solar" in key
            or "shortwave" in key
            or "radiat" in key
            or key in {"rad", "rsds", "swdown"}
        ):
            return col

    return None


def build_local_hourly_ssrd_profile(df_in: pd.DataFrame) -> pd.DataFrame | None:
    """
    Build a local-hour SSRD/radiation profile from the same raw UTC table used
    for Panel A. If no radiation column is found, return None and draw Panel A
    without the contextual background.
    """
    ssrd_col = find_ssrd_column(df_in)

    if ssrd_col is None:
        print("Panel A SSRD background skipped: no SSRD/radiation column found.")
        print("Available raw_file columns:")
        print("  " + ", ".join(map(str, df_in.columns)))
        return None

    panel_ssrd = df_in.loc[
        (df_in["local_month"] == 7)
        & df_in[ssrd_col].notna()
        & df_in["hour_local"].notna()
    ].copy()

    hourly_ssrd_profile = (
        panel_ssrd
        .groupby("hour_local", as_index=True)
        .agg(
            mean_ssrd=(ssrd_col, "mean"),
            n_obs=(ssrd_col, "size"),
        )
        .reindex(np.arange(24))
        .rename_axis("hour_local")
        .reset_index()
    )

    hourly_ssrd_profile["mean_ssrd"] = (
        hourly_ssrd_profile["mean_ssrd"]
        .astype(float)
        .fillna(0.0)
    )

    print(f"Panel A SSRD background column: {ssrd_col}")
    print("Panel A SSRD local-hour coverage:")
    print(hourly_ssrd_profile[["hour_local", "n_obs"]].to_string(index=False))

    return hourly_ssrd_profile


hourly = build_local_hourly_temperature_profile(df)
hourly_ssrd = build_local_hourly_ssrd_profile(df)

ucc = df["uCC"].astype(float).copy()

if ucc.max() <= 1.5:
    ucc = ucc * 100.0

df["uCC_pct"] = ucc

df_day = df.loc[
    (df["hour_local"] >= 10) &
    (df["hour_local"] <= 18) &
    df["temp"].notna() &
    df["uCC_pct"].notna()
].copy()

site = (
    df_day.groupby("sensor_id", as_index=False)
    .agg(
        mean_temp=("temp", "mean"),
        mean_ucc=("uCC_pct", "mean"),
        n_obs=("temp", "size")
    )
)

bins = np.arange(0, 105, 10)
centers = (bins[:-1] + bins[1:]) / 2

site["bin"] = pd.cut(
    site["mean_ucc"],
    bins=bins,
    include_lowest=True
)

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

for c in ["mean_temp", "lo", "hi"]:
    binned[c] = binned[c].interpolate(limit_direction="both")

site_counts, _ = np.histogram(site["mean_ucc"], bins=bins)


# ============================================================
# LOAD AND PREPARE VALIDATION DATA
# ============================================================

dt_hourly = read_validation_table(validation_csv, validation_rds)

required_validation_cols = {"time", "observed", "pred"}
missing_validation_cols = required_validation_cols - set(dt_hourly.columns)

if missing_validation_cols:
    raise ValueError(
        f"Missing columns in validation data: {missing_validation_cols}. "
        "The validation file must contain 'time', 'observed', and 'pred'."
    )

dt_hourly["time"] = pd.to_datetime(
    dt_hourly["time"],
    utc=True,
    errors="coerce"
)

dt_hourly["observed"] = pd.to_numeric(
    dt_hourly["observed"],
    errors="coerce"
)

dt_hourly["pred"] = pd.to_numeric(
    dt_hourly["pred"],
    errors="coerce"
)

dt_hourly = dt_hourly.dropna(
    subset=["time", "observed", "pred"]
).copy()

dt_time = (
    dt_hourly
    .groupby("time", as_index=False)
    .agg(
        observed=("observed", "mean"),
        pred=("pred", "mean")
    )
)

dt_time["err"] = dt_time["pred"] - dt_time["observed"]
dt_time["time_local"] = dt_time["time"].dt.tz_convert("Europe/Helsinki")
dt_time["date"] = dt_time["time_local"].dt.date
dt_time["hour"] = dt_time["time_local"].dt.hour

date_err = (
    dt_time
    .groupby("date", as_index=False)
    .agg(
        RMSE=("err", lambda x: np.sqrt(np.mean(np.square(x)))),
        sd_err=("err", "std")
    )
)

hour_bias = (
    dt_time
    .groupby("hour", as_index=False)
    .agg(
        bias=("err", "mean"),
        sd_err=("err", "std")
    )
)

date_err["date"] = pd.to_datetime(date_err["date"])

ylim_rmse = (
    0,
    np.nanmax(date_err["RMSE"] + date_err["sd_err"]) * 1.05
)

ylim_bias = (
    np.nanmin(hour_bias["bias"] - hour_bias["sd_err"]) * 1.05,
    np.nanmax(hour_bias["bias"] + hour_bias["sd_err"]) * 1.05
)


# ============================================================
# PANEL DRAWING FUNCTIONS
# ============================================================
def draw_panel_a(ax):
    x = hourly["hour_local"].values
    y = hourly["mean_temp"].values
    sd = hourly["sd_temp"].values

    # Fix the y-axis first so the SSRD background can be anchored exactly
    # to the visible x-axis baseline.
    y_min = np.nanmin(y - sd)
    y_max = np.nanmax(y + sd)
    y_pad = 0.04 * (y_max - y_min)

    y_lower = y_min - y_pad
    y_upper = y_max + y_pad

    ax.set_ylim(y_lower, y_upper)

    # -------------------------------------------------
    # Background SSRD / daylight context
    # -------------------------------------------------
    # Drawn in the same data coordinates as the temperature axis.
    # This anchors the radiation band exactly to the x-axis baseline.
    if hourly_ssrd is not None:
        x_ssrd = hourly_ssrd["hour_local"].values
        y_ssrd = hourly_ssrd["mean_ssrd"].values.astype(float)

        if np.isfinite(y_ssrd).any() and np.nanmax(y_ssrd) > 0:
            y_norm = y_ssrd / np.nanmax(y_ssrd)

            band_height = 0.35 * (y_upper - y_lower)
            y_base = y_lower
            y_band = y_base + band_height * y_norm

            ax.fill_between(
                x_ssrd,
                y_base,
                y_band,
                color=COL_HIST,
                alpha=0.3,
                linewidth=0,
                edgecolor="none",
                zorder=0,
            )

            ax.text(
                14,
                y_base + 0.06 * (y_upper - y_lower),
                "solar radiation",
                ha="center",
                va="center",
                fontsize=STYLE.fs_legend,
                color=COL_BLACK,
                alpha=0.5,
                zorder=1,
                path_effects=[
                    pe.withStroke(linewidth=2, foreground="white", alpha=0.75)
                ],
            )

    # -------------------------------------------------
    # Foreground temperature profile
    # -------------------------------------------------
    ax.fill_between(
        x,
        y - sd,
        y + sd,
        color=COL_BLUE_FILL,
        alpha=0.7,
        linewidth=0,
        zorder=3,
    )

    ax.plot(
        x,
        y,
        color=COL_BLUE,
        linewidth=2.4,
        zorder=4,
    )

    ax.set_xlim(0, 23)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 23])
    ax.set_ylim(y_lower, y_upper)

    set_axis_labels(
        ax,
        xlabel="Hour (local time)",
        ylabel="15 cm air temperature (°C)"
    )

    band_handle = Patch(
        facecolor=COL_BLUE_FILL,
        edgecolor="none",
        alpha=1.0,
        label="±SD"
    )

    add_small_legend(
        ax,
        handles=[band_handle],
        loc="upper right"
    )

    style_axis(ax)

def draw_panel_b(ax):
    ax.set_zorder(3)
    ax.patch.set_alpha(0)

    ax2 = ax.twinx()
    ax2.set_zorder(1)
    ax2.patch.set_alpha(0)

    # ---------------------------------
    # Right-axis histogram: sensor count
    # ---------------------------------
    ax2.bar(
        centers,
        site_counts,
        width=8.5,
        color=COL_HIST,
        alpha=0.3,
        edgecolor="none",
        zorder=1
    )

    ax2.set_ylabel(
        "Sensor count",
        fontsize=STYLE.fs_axis,
        fontweight="normal",
        fontfamily=STYLE.font_family,
        color=STYLE.col_black,
    )

    ax2.set_ylim(0, max(site_counts) * 1.20)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(False)

    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(COL_HIST)

    ax2.tick_params(
        axis="y",
        length=STYLE.tick_length,
        width=STYLE.tick_width,
        colors=COL_HIST
    )

    # ---------------------------------
    # Left-axis temperature data
    # Binned mean and CI drawn as interval-aligned steps.
    # This removes the visual floating start/end caused by bin centers.
    # ---------------------------------
    step_x = bins
    step_mean = np.r_[binned["mean_temp"].values, binned["mean_temp"].values[-1]]
    step_lo = np.r_[binned["lo"].values, binned["lo"].values[-1]]
    step_hi = np.r_[binned["hi"].values, binned["hi"].values[-1]]

    ax.fill_between(
        step_x,
        step_lo,
        step_hi,
        step="post",
        color=COL_BLUE_FILL,
        alpha=0.7,
        linewidth=0,
        zorder=2
    )

    ax.step(
        step_x,
        step_mean,
        where="post",
        color=COL_BLUE,
        linewidth=2.4,
        zorder=4
    )

    ax.scatter(
        site["mean_ucc"],
        site["mean_temp"],
        s=30,
        color=COL_BLUE,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.92,
        zorder=5
    )

    ax.set_xlim(0, 100)

    set_axis_labels(
        ax,
        xlabel="Mean upper canopy cover (%)",
        ylabel="Mean 15 cm air temperature (°C)"
    )

    ymin = min(site["mean_temp"].min(), binned["lo"].min()) - 0.15
    ymax = max(site["mean_temp"].max(), binned["hi"].max()) + 0.15
    ax.set_ylim(ymin, ymax)

    # ---------------------------------
    # Full custom legend
    # ---------------------------------
    point_handle = Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markerfacecolor=COL_BLUE,
        markeredgecolor="white",
        markeredgewidth=0.6,
        markersize=7,
        label="on-site mean t."
    )

    line_handle = Line2D(
        [0], [0],
        color=COL_BLUE,
        linewidth=2.4,
        label="mean t. ±95% CI"
    )

    hist_handle = Patch(
        facecolor=COL_HIST,
        edgecolor="none",
        alpha=0.34,
        label="n sensors"
    )

    add_small_legend(
        ax,
        handles=[point_handle, line_handle, hist_handle],
        loc="upper right"
    )

    style_axis(ax)

    # Re-apply coloured axes after shared styling.
    # Keep labels black; colour only ticks and axis spines.
    ax.yaxis.label.set_color("black")
    ax.tick_params(
        axis="y",
        length=STYLE.tick_length,
        width=STYLE.tick_width,
        colors=COL_BLUE
    )
    ax.spines["left"].set_color(COL_BLUE)

    ax2.yaxis.label.set_color("black")
    ax2.tick_params(
        axis="y",
        length=STYLE.tick_length,
        width=STYLE.tick_width,
        colors=COL_HIST
    )
    ax2.spines["right"].set_color(COL_HIST)


def draw_panel_c(ax):
    x = date_err["date"]
    y = date_err["RMSE"]
    sd = date_err["sd_err"]

    ax.fill_between(
        x,
        y - sd,
        y + sd,
        color=COL_RED_FILL,
        alpha=0.7,
        linewidth=0
    )

    ax.plot(
        x,
        y,
        color=COL_RED,
        linewidth=1.1
    )

    x_num = mdates.date2num(x)
    xs, ys = smooth_lowess(x_num, y, frac=0.30)

    ax.plot(
        mdates.num2date(xs),
        ys,
        color=COL_RED,
        linewidth=2.4
    )

    ax.axhline(
        1,
        color=COL_BLACK,
        linestyle=":",
        linewidth=1.1
    )

    ax.set_ylim(*ylim_rmse)

    set_axis_labels(
        ax,
        xlabel="Date",
        ylabel="Mean RMSE (°C)"
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    band_handle = Patch(
        facecolor=COL_RED_FILL,
        edgecolor="none",
        alpha=1,
        label="±SD"
    )

    add_small_legend(
        ax,
        handles=[band_handle],
        loc="upper right"
    )

    style_axis(ax)


def draw_panel_d(ax):
    x = hour_bias["hour"]
    y = hour_bias["bias"]
    sd = hour_bias["sd_err"]

    ax.fill_between(
        x,
        y - sd,
        y + sd,
        color=COL_RED_FILL,
        alpha=0.7,
        linewidth=0
    )

    ax.plot(
        x,
        y,
        color=COL_RED,
        linewidth=2.4
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.1
    )

    ax.set_xlim(0, 23)
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_ylim(*ylim_bias)

    set_axis_labels(
        ax,
        xlabel="Hour (local time)",
        ylabel="Mean Bias (°C)"
    )

    band_handle = Patch(
        facecolor=COL_RED_FILL,
        edgecolor="none",
        alpha=1,
        label="±SD"
    )

    add_small_legend(
        ax,
        handles=[band_handle],
        loc="upper right"
    )

    style_axis(ax)


# ============================================================
# HARD CHECK BEFORE EXPORT
# ============================================================

required_runtime_objects = [
    "hourly", "site", "binned", "site_counts",
    "date_err", "hour_bias", "ylim_rmse", "ylim_bias"
]

missing_runtime_objects = [
    name for name in required_runtime_objects
    if name not in globals()
]

if missing_runtime_objects:
    raise RuntimeError(
        "Not exporting panels because some required objects are missing: "
        + ", ".join(missing_runtime_objects)
    )


# ============================================================
# EXPORT INDIVIDUAL PANELS
# ============================================================

panel_specs = [
    ("Figure_results_panel_A_diurnal_temperature", draw_panel_a),
    ("Figure_results_panel_B_canopy_gradient", draw_panel_b),
    ("Figure_results_panel_C_validation_rmse_time", draw_panel_c),
    ("Figure_results_panel_D_validation_bias_hour", draw_panel_d),
]

for filename_stem, draw_func in panel_specs:
    fig = new_panel_figure()
    ax = add_panel_axes(fig)
    draw_func(ax)
    save_single_panel(fig, filename_stem)
    plt.close(fig)


# ============================================================
# COMBINED 2x2 PREVIEW ONLY
# Not exported.
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(gps.cm_to_in(STYLE.panel_width_cm) * 2, gps.cm_to_in(STYLE.panel_height_cm) * 2),
    constrained_layout=True
)

draw_panel_a(axes[0, 0])
draw_panel_b(axes[0, 1])
draw_panel_c(axes[1, 0])
draw_panel_d(axes[1, 1])

# Keep preview readable on screen.
fig.patch.set_facecolor("white")
for ax in fig.axes:
    ax.set_facecolor("white")

plt.show()