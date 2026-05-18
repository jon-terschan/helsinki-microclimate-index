### LOSS OF RELATIVE CONNETIVITY X ABSOLUTE COURRENT #

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

# =============================================================================
# SETTINGS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")

FUTURE_RUN_PARENT = BASE / "output" / "future_condition_multiruns"
BASELINE_RUN_PARENT = BASE / "output" / "conditional_multiruns"

OUT_DIR = BASE / "output" / "future_condition_comparison_figures" / "summary_metrics_full_range"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLOW_RASTER_NAME = "cum_currmap.tif"

HEATWAVES = [
    "condition_heatwave_2010",
    "condition_heatwave_2018",
    "condition_heatwave_2021",
]

TOLERANCES = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

SCENARIOS = [
    {
        "label": "Average → heatwave",
        "present": "condition_average",
        "color": "#2c7fb8",
    },
    {
        "label": "P90 → heatwave",
        "present": "condition_p90",
        "color": "#f03b20",
    },
]


# =============================================================================
# HELPERS
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


def load_total_current(path: Path) -> float:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    return float(np.sum(arr))


def get_total_for_run(parent: Path, run_name: str) -> float | None:
    run_dir = find_run_dir(parent, run_name)

    if run_dir is None:
        print(f"[MISSING RUN] {run_name}")
        return None

    flow_path = find_file(run_dir, FLOW_RASTER_NAME)

    if flow_path is None:
        print(f"[MISSING FLOW] {run_name}")
        return None

    return load_total_current(flow_path)


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    rows = []

    for scenario in SCENARIOS:
        label = scenario["label"]
        present = scenario["present"]

        print(f"\n[SCENARIO] {label}")

        for tol in TOLERANCES:
            baseline_name = baseline_run_name(present, tol)
            baseline_total = get_total_for_run(BASELINE_RUN_PARENT, baseline_name)

            if baseline_total is None or baseline_total <= 0:
                print(f"  ±{tol:g}°C: no valid baseline")
                continue

            future_totals = []

            for heatwave in HEATWAVES:
                future_name = future_run_name(present, heatwave, tol)
                future_total = get_total_for_run(FUTURE_RUN_PARENT, future_name)

                if future_total is not None:
                    future_totals.append(future_total)

            if not future_totals:
                print(f"  ±{tol:g}°C: no valid heatwave runs")
                continue

            future_mean = float(np.mean(future_totals))
            relative = future_mean / baseline_total

            rows.append(
                {
                    "scenario": label,
                    "present": present,
                    "tolerance": tol,
                    "baseline_total": baseline_total,
                    "future_mean_total": future_mean,
                    "relative_to_matched_baseline": relative,
                    "n_heatwaves": len(future_totals),
                }
            )

            print(
                f"  ±{tol:g}°C: baseline={baseline_total:.6g}, "
                f"future_mean={future_mean:.6g}, "
                f"relative={relative:.3f}x, "
                f"n={len(future_totals)}"
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No valid rows found.")

    out_csv = OUT_DIR / "relative_connectivity_simple_full_range.csv"
    df.to_csv(out_csv, index=False)

    # -------------------------------------------------------------------------
    # AUC SUMMARY
    # -------------------------------------------------------------------------

    auc_rows = []

    for scenario in SCENARIOS:
        label = scenario["label"]
        sub = df[df["scenario"] == label].sort_values("tolerance")

        if sub.empty:
            continue

        x = sub["tolerance"].to_numpy(dtype=float)
        y = sub["relative_to_matched_baseline"].to_numpy(dtype=float)

        auc_rows.append(
            {
                "scenario": label,
                "auc_relative_connectivity": auc_trapezoid(x, y),
                "mean_relative_connectivity": float(np.nanmean(y)),
                "max_relative_connectivity": float(np.nanmax(y)),
            }
        )

    auc_df = pd.DataFrame(auc_rows)
    out_auc_csv = OUT_DIR / "relative_connectivity_simple_auc_full_range.csv"
    auc_df.to_csv(out_auc_csv, index=False)

    # -------------------------------------------------------------------------
    # FIGURE
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.3, 1.0]},
    )

    ax = axes[0]

    for scenario in SCENARIOS:
        label = scenario["label"]
        color = scenario["color"]

        sub = df[df["scenario"] == label].sort_values("tolerance")

        if sub.empty:
            continue

        x = sub["tolerance"].to_numpy(dtype=float)
        y = sub["relative_to_matched_baseline"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.8,
            markersize=7,
            color=color,
            label=label,
        )

        ax.fill_between(
            x,
            0,
            y,
            color=color,
            alpha=0.16,
        )

        ax.annotate(
            f"{y[-1]:.2f}×",
            xy=(x[-1], y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=color,
        )

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
    )

    ax.text(
        0.01,
        1.02,
        "same-tolerance baseline",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=9,
    )

    ax.set_xlabel("Thermal analog tolerance (±°C)")
    ax.set_ylabel("Heatwave mean cumulative current / same-tolerance baseline")
    ax.set_title("A. Relative heatwave analog connectivity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    ax.set_ylim(bottom=0)

    # -------------------------------------------------------------------------
    # AUC PANEL
    # -------------------------------------------------------------------------

    ax = axes[1]

    auc_df = auc_df.set_index("scenario").reindex([s["label"] for s in SCENARIOS]).reset_index()

    bar_colors = [
        next(s["color"] for s in SCENARIOS if s["label"] == row["scenario"])
        for _, row in auc_df.iterrows()
    ]

    ax.bar(
        auc_df["scenario"],
        auc_df["auc_relative_connectivity"],
        color=bar_colors,
        alpha=0.75,
    )

    for i, row in auc_df.iterrows():
        ax.text(
            i,
            row["auc_relative_connectivity"],
            f"{row['auc_relative_connectivity']:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Area under relative-connectivity curve")
    ax.set_title("B. Integrated relative connectivity")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)

    fig.suptitle(
        "Heatwave analog connectivity relative to tolerance-matched baseline networks",
        fontsize=15,
    )

    out_png = OUT_DIR / "relative_connectivity_simple_full_range.png"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] wrote {out_png}")
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_auc_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import rasterio

# =============================================================================
# SETTINGS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")

FUTURE_RUN_PARENT = BASE / "output" / "future_condition_multiruns"
BASELINE_RUN_PARENT = BASE / "output" / "conditional_multiruns"

OUT_DIR = BASE / "output" / "future_condition_comparison_figures" / "summary_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLOW_RASTER_NAME = "cum_currmap.tif"

HEATWAVES = [
    "condition_heatwave_2010",
    "condition_heatwave_2018",
    "condition_heatwave_2021",
]

TOLERANCES = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

SCENARIOS = [
    {
        "label": "Average → heatwave",
        "present": "condition_average",
    },
    {
        "label": "P90 → heatwave",
        "present": "condition_p90",
    },
]

# -----------------------------------------------------------------------------
# COLORS
# -----------------------------------------------------------------------------

COL_BLUE       = "#004488"
COL_BLUE_FILL  = "#BBCCEE"

COL_RED        = "#BB5566"
COL_RED_FILL   = "#E8BCC4"

COL_GREY       = "#666666"
COL_GRID       = "#D0D0D0"

# -----------------------------------------------------------------------------
# EXPORT / PREVIEW
# -----------------------------------------------------------------------------

SHOW_PREVIEW = True
SAVE_FIGURE = True
SAVE_CSV = True

DPI = 300


# =============================================================================
# HELPERS
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


def load_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def total_positive_current(path: Path) -> float:
    arr = load_raster(path)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    return float(np.sum(arr))


def get_total_for_run(parent: Path, run_name: str) -> float | None:
    run_dir = find_run_dir(parent, run_name)

    if run_dir is None:
        print(f"    [MISSING RUN] {run_name}")
        return None

    flow_path = find_file(run_dir, FLOW_RASTER_NAME)

    if flow_path is None:
        print(f"    [MISSING FLOW] {run_name}")
        return None

    return total_positive_current(flow_path)


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=COL_GRID, linewidth=0.8, alpha=0.8)
    ax.grid(False, axis="x")


def format_tolerance_ticks(ax) -> None:
    ax.set_xticks(TOLERANCES)
    ax.set_xticklabels([f"{t:g}" for t in TOLERANCES])


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


# =============================================================================
# DATA BUILD
# =============================================================================

def build_summary_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows = []
    relative_rows = []

    for scenario in SCENARIOS:
        label = scenario["label"]
        present = scenario["present"]

        print(f"\n[SCENARIO] {label}")

        for tol in TOLERANCES:
            print(f"  ±{tol:g}°C")

            # -----------------------------------------------------------------
            # TOLERANCE-MATCHED BASELINE
            # -----------------------------------------------------------------

            base_name = baseline_run_name(present, tol)
            baseline_total = get_total_for_run(BASELINE_RUN_PARENT, base_name)

            if baseline_total is None or baseline_total <= 0:
                print(f"    [SKIP] no valid baseline for {base_name}")
                continue

            print(f"    baseline total = {baseline_total:.6g}")

            all_rows.append(
                {
                    "scenario": label,
                    "present": present,
                    "future": "baseline",
                    "tolerance": tol,
                    "run_type": "baseline",
                    "run_name": base_name,
                    "total_current": baseline_total,
                }
            )

            # -----------------------------------------------------------------
            # HEATWAVE RUNS
            # -----------------------------------------------------------------

            future_totals = []

            for heatwave in HEATWAVES:
                fut_name = future_run_name(present, heatwave, tol)
                fut_total = get_total_for_run(FUTURE_RUN_PARENT, fut_name)

                if fut_total is None:
                    continue

                future_totals.append(fut_total)

                all_rows.append(
                    {
                        "scenario": label,
                        "present": present,
                        "future": heatwave,
                        "tolerance": tol,
                        "run_type": "future_single",
                        "run_name": fut_name,
                        "total_current": fut_total,
                    }
                )

                print(f"    {heatwave}: {fut_total:.6g}")

            if not future_totals:
                print("    [SKIP] no future totals available")
                continue

            future_mean_total = float(np.mean(future_totals))
            relative = future_mean_total / baseline_total
            relative_percent = 100.0 * relative

            all_rows.append(
                {
                    "scenario": label,
                    "present": present,
                    "future": "heatwave_mean",
                    "tolerance": tol,
                    "run_type": "future_mean",
                    "run_name": f"{present}_to_heatwave_mean_pm{tolerance_label(tol)}deg",
                    "total_current": future_mean_total,
                    "relative_to_matched_baseline": relative,
                    "relative_percent": relative_percent,
                    "n_heatwaves": len(future_totals),
                }
            )

            relative_rows.append(
                {
                    "scenario": label,
                    "present": present,
                    "tolerance": tol,
                    "baseline_total": baseline_total,
                    "future_mean_total": future_mean_total,
                    "relative_to_matched_baseline": relative,
                    "relative_percent": relative_percent,
                    "n_heatwaves": len(future_totals),
                }
            )

            print(
                f"    future mean = {future_mean_total:.6g}; "
                f"relative = {relative:.4g} ({relative_percent:.4g}%)"
            )

    full_df = pd.DataFrame(all_rows)
    rel_df = pd.DataFrame(relative_rows)

    if full_df.empty or rel_df.empty:
        raise RuntimeError("No valid rows found.")

    auc_rows = []

    for scenario, sub in rel_df.groupby("scenario"):
        sub = sub.sort_values("tolerance")

        x = sub["tolerance"].to_numpy(dtype=float)
        y = sub["relative_to_matched_baseline"].to_numpy(dtype=float)

        auc_rows.append(
            {
                "scenario": scenario,
                "auc_relative_to_matched_baseline": auc_trapezoid(x, y),
                "mean_relative_to_matched_baseline": float(np.nanmean(y)),
                "max_relative_to_matched_baseline": float(np.nanmax(y)),
            }
        )

    auc_df = pd.DataFrame(auc_rows)

    return full_df, rel_df, auc_df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_summary_figure(full_df: pd.DataFrame, rel_df: pd.DataFrame) -> None:
    avg_rel = rel_df[rel_df["scenario"] == "Average → heatwave"].sort_values("tolerance")
    p90_rel = rel_df[rel_df["scenario"] == "P90 → heatwave"].sort_values("tolerance")

    baseline_df = full_df[full_df["run_type"] == "baseline"].copy()
    future_df = full_df[full_df["run_type"] == "future_mean"].copy()

    avg_base = baseline_df[baseline_df["scenario"] == "Average → heatwave"].sort_values("tolerance")
    p90_base = baseline_df[baseline_df["scenario"] == "P90 → heatwave"].sort_values("tolerance")

    avg_future = future_df[future_df["scenario"] == "Average → heatwave"].sort_values("tolerance")
    p90_future = future_df[future_df["scenario"] == "P90 → heatwave"].sort_values("tolerance")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    # -------------------------------------------------------------------------
    # PANEL A: RELATIVE CONNECTIVITY (%)
    # -------------------------------------------------------------------------

    ax = axes[0]

    x_avg = avg_rel["tolerance"].to_numpy(dtype=float)
    y_avg = avg_rel["relative_percent"].to_numpy(dtype=float)

    x_p90 = p90_rel["tolerance"].to_numpy(dtype=float)
    y_p90 = p90_rel["relative_percent"].to_numpy(dtype=float)

    if len(x_avg) > 0:
        ax.fill_between(
            x_avg,
            0,
            y_avg,
            color=COL_BLUE_FILL,
            alpha=0.9,
            zorder=1,
        )

        ax.plot(
            x_avg,
            y_avg,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=COL_BLUE,
            label="Average → heatwave",
            zorder=4,
        )

    if len(x_p90) > 0:
        # Only fill red where P90 exceeds the average curve at shared tolerances.
        merged = pd.merge(
            avg_rel[["tolerance", "relative_percent"]].rename(columns={"relative_percent": "avg_y"}),
            p90_rel[["tolerance", "relative_percent"]].rename(columns={"relative_percent": "p90_y"}),
            on="tolerance",
            how="inner",
        ).sort_values("tolerance")

        if not merged.empty:
            x_shared = merged["tolerance"].to_numpy(dtype=float)
            avg_shared = merged["avg_y"].to_numpy(dtype=float)
            p90_shared = merged["p90_y"].to_numpy(dtype=float)

            ax.fill_between(
                x_shared,
                avg_shared,
                p90_shared,
                where=(p90_shared > avg_shared),
                interpolate=True,
                color=COL_RED_FILL,
                alpha=0.85,
                zorder=2,
            )

        ax.plot(
            x_p90,
            y_p90,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=COL_RED,
            label="P90 → heatwave",
            zorder=5,
        )

    ax.axhline(
        100.0,
        color=COL_GREY,
        linestyle="--",
        linewidth=1.2,
        alpha=0.95,
    )

    ax.text(
        0.015,
        100.8,
        "100% = same as tolerance-matched baseline",
        color=COL_GREY,
        fontsize=9,
        ha="left",
        va="bottom",
    )

    ax.set_title("A. Relative heatwave analog connectivity")
    ax.set_xlabel(
        "Thermal analog tolerance (±°C)\n"
        "0.1 = near-exact analogs · 5 = highly relaxed matching"
    )
    ax.set_ylabel("Connectivity relative to matched baseline (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(bottom=0)
    format_tolerance_ticks(ax)
    clean_axes(ax)
    ax.legend(frameon=False, loc="best")

    # -------------------------------------------------------------------------
    # PANEL B: ABSOLUTE TOTAL CUMULATIVE CURRENT
    # -------------------------------------------------------------------------

    ax = axes[1]

    if not avg_base.empty:
        ax.plot(
            avg_base["tolerance"],
            avg_base["total_current"],
            marker="o",
            linewidth=2.0,
            markersize=6,
            linestyle="--",
            color=COL_BLUE,
            label="Average baseline",
        )

    if not avg_future.empty:
        ax.plot(
            avg_future["tolerance"],
            avg_future["total_current"],
            marker="o",
            linewidth=2.6,
            markersize=7,
            linestyle="-",
            color=COL_BLUE,
            label="Average → heatwave mean",
        )

    if not p90_base.empty:
        ax.plot(
            p90_base["tolerance"],
            p90_base["total_current"],
            marker="o",
            linewidth=2.0,
            markersize=6,
            linestyle="--",
            color=COL_RED,
            label="P90 baseline",
        )

    if not p90_future.empty:
        ax.plot(
            p90_future["tolerance"],
            p90_future["total_current"],
            marker="o",
            linewidth=2.6,
            markersize=7,
            linestyle="-",
            color=COL_RED,
            label="P90 → heatwave mean",
        )

    ax.set_title("B. Total cumulative current")
    ax.set_xlabel(
        "Thermal analog tolerance (±°C)\n"
        "0.1 = near-exact analogs · 5 = highly relaxed matching"
    )
    ax.set_ylabel("Total cumulative current")
    format_tolerance_ticks(ax)
    clean_axes(ax)
    ax.legend(frameon=False, loc="best")

    fig.suptitle(
        "Heatwave analog connectivity relative to matched baseline networks",
        fontsize=15,
    )

    out_png = OUT_DIR / "summary_relative_and_total_connectivity_fullrange.png"

    if SAVE_FIGURE:
        plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
        print(f"[OK] wrote {out_png}")

    if SHOW_PREVIEW:
        plt.show()
    else:
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    full_df, rel_df, auc_df = build_summary_tables()

    if SAVE_CSV:
        full_csv = OUT_DIR / "connectivity_all_totals_fullrange.csv"
        rel_csv = OUT_DIR / "relative_connectivity_tolerance_matched_fullrange.csv"
        auc_csv = OUT_DIR / "relative_connectivity_auc_fullrange.csv"

        full_df.to_csv(full_csv, index=False)
        rel_df.to_csv(rel_csv, index=False)
        auc_df.to_csv(auc_csv, index=False)

        print(f"\n[OK] wrote {full_csv}")
        print(f"[OK] wrote {rel_csv}")
        print(f"[OK] wrote {auc_csv}")

    plot_summary_figure(full_df, rel_df)


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import rasterio

# =============================================================================
# SETTINGS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")

FUTURE_RUN_PARENT = BASE / "output" / "future_condition_multiruns"
BASELINE_RUN_PARENT = BASE / "output" / "conditional_multiruns"

OUT_DIR = BASE / "output" / "future_condition_comparison_figures" / "summary_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLOW_RASTER_NAME = "cum_currmap.tif"

HEATWAVES = [
    "condition_heatwave_2010",
    "condition_heatwave_2018",
    "condition_heatwave_2021",
]

# full range
TOLERANCES = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

SCENARIOS = [
    {
        "label": "Average → heatwave",
        "present": "condition_average",
    },
    {
        "label": "P90 → heatwave",
        "present": "condition_p90",
    },
]

# =============================================================================
# COLORS
# =============================================================================

COL_BLUE        = "#004488"
COL_BLUE_FILL   = "#BBCCEE"

COL_HIST        = "#DDAA33"
COL_HIST_FILL   = "#E9D8A6"

COL_RED         = "#BB5566"
COL_RED_FILL    = "#E8BCC4"

COL_GREY        = "#666666"
COL_GRID        = "#D0D0D0"

# tolerance-zone fills
COL_ZONE_STRICT   = "#F5F5F5"
COL_ZONE_MODERATE = "#EFEFEF"
COL_ZONE_RELAXED  = "#E8E8E8"

# =============================================================================
# EXPORT / PREVIEW
# =============================================================================

SHOW_PREVIEW = True
SAVE_FIGURE = True
SAVE_CSV = True
DPI = 300

# =============================================================================
# HELPERS
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


def load_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def total_positive_current(path: Path) -> float:
    arr = load_raster(path)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    return float(np.sum(arr))


def get_total_for_run(parent: Path, run_name: str) -> float | None:
    run_dir = find_run_dir(parent, run_name)

    if run_dir is None:
        print(f"    [MISSING RUN] {run_name}")
        return None

    flow_path = find_file(run_dir, FLOW_RASTER_NAME)

    if flow_path is None:
        print(f"    [MISSING FLOW] {run_name}")
        return None

    return total_positive_current(flow_path)


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=COL_GRID, linewidth=0.8, alpha=0.8)
    ax.grid(False, axis="x")


def format_tolerance_ticks(ax) -> None:
    ax.set_xticks(TOLERANCES)
    ax.set_xticklabels([f"{t:g}" for t in TOLERANCES])


def add_tolerance_zones(ax, ymin: float, ymax: float) -> None:
    """
    Visual interpretation bands:
    strict   ~ near-exact analogs
    moderate ~ intermediate flexibility
    relaxed  ~ permissive matching
    """
    ax.axvspan(0.05, 0.75, color=COL_ZONE_STRICT, zorder=0)
    ax.axvspan(0.75, 2.5, color=COL_ZONE_MODERATE, zorder=0)
    ax.axvspan(2.5, 5.1, color=COL_ZONE_RELAXED, zorder=0)

    ytxt = ymax - 0.03 * (ymax - ymin)

    ax.text(0.4, ytxt, "strict", ha="center", va="top", fontsize=9, color=COL_GREY)
    ax.text(1.6, ytxt, "moderate", ha="center", va="top", fontsize=9, color=COL_GREY)
    ax.text(3.8, ytxt, "relaxed", ha="center", va="top", fontsize=9, color=COL_GREY)


def label_line_right(ax, x: np.ndarray, y: np.ndarray, text: str, color: str, dy: float = 0.0) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return

    xv = x[valid]
    yv = y[valid]

    ax.annotate(
        text,
        xy=(xv[-1], yv[-1]),
        xytext=(8, dy),
        textcoords="offset points",
        color=color,
        fontsize=10,
        fontweight="bold",
        va="center",
        ha="left",
    )


# =============================================================================
# DATA BUILD
# =============================================================================

def build_summary_table() -> pd.DataFrame:
    rows = []

    for scenario in SCENARIOS:
        label = scenario["label"]
        present = scenario["present"]

        print(f"\n[SCENARIO] {label}")

        for tol in TOLERANCES:
            print(f"  ±{tol:g}°C")

            # -----------------------------------------------------------------
            # tolerance-matched baseline
            # -----------------------------------------------------------------

            base_name = baseline_run_name(present, tol)
            baseline_total = get_total_for_run(BASELINE_RUN_PARENT, base_name)

            if baseline_total is None or baseline_total <= 0:
                print(f"    [SKIP] no valid baseline for {base_name}")
                continue

            print(f"    baseline total = {baseline_total:.6g}")

            # -----------------------------------------------------------------
            # heatwave runs
            # -----------------------------------------------------------------

            future_totals = []

            for heatwave in HEATWAVES:
                fut_name = future_run_name(present, heatwave, tol)
                fut_total = get_total_for_run(FUTURE_RUN_PARENT, fut_name)

                if fut_total is None:
                    continue

                future_totals.append(fut_total)
                print(f"    {heatwave}: {fut_total:.6g}")

            if not future_totals:
                print("    [SKIP] no future totals available")
                continue

            future_mean_total = float(np.mean(future_totals))
            retained_fraction = future_mean_total / baseline_total
            retained_percent = 100.0 * retained_fraction
            loss_percent = 100.0 - retained_percent

            rows.append(
                {
                    "scenario": label,
                    "present": present,
                    "tolerance": tol,
                    "baseline_total": baseline_total,
                    "future_mean_total": future_mean_total,
                    "retained_fraction": retained_fraction,
                    "retained_percent": retained_percent,
                    "loss_percent": loss_percent,
                    "n_heatwaves": len(future_totals),
                }
            )

            print(
                f"    future mean = {future_mean_total:.6g}; "
                f"retained = {retained_percent:.4g}% ; "
                f"loss = {loss_percent:.4g}%"
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No valid rows found.")

    return df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_summary_figure(df: pd.DataFrame) -> None:
    avg = df[df["scenario"] == "Average → heatwave"].sort_values("tolerance")
    p90 = df[df["scenario"] == "P90 → heatwave"].sort_values("tolerance")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.2),
        constrained_layout=True,
    )

    # -------------------------------------------------------------------------
    # PANEL A: RETAINED CONNECTIVITY
    # -------------------------------------------------------------------------

    ax = axes[0]

    x_avg = avg["tolerance"].to_numpy(dtype=float)
    y_avg = avg["retained_percent"].to_numpy(dtype=float)

    x_p90 = p90["tolerance"].to_numpy(dtype=float)
    y_p90 = p90["retained_percent"].to_numpy(dtype=float)

    ymax_a = 105
    ymin_a = 0

    add_tolerance_zones(ax, ymin=ymin_a, ymax=ymax_a)

    if len(x_avg) > 0:
        ax.plot(
            x_avg,
            y_avg,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=COL_BLUE,
            label="Average → heatwave",
            zorder=3,
        )

    if len(x_p90) > 0:
        ax.plot(
            x_p90,
            y_p90,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=COL_RED,
            label="P90 → heatwave",
            zorder=4,
        )

    ax.axhline(
        100.0,
        color=COL_GREY,
        linestyle="--",
        linewidth=1.2,
        alpha=0.95,
    )

    ax.text(
        0.03,
        100.8,
        "100% = same as tolerance-matched baseline",
        color=COL_GREY,
        fontsize=9,
        ha="left",
        va="bottom",
    )

    label_line_right(ax, x_avg, y_avg, "Average → heatwave", COL_BLUE, dy=-8)
    label_line_right(ax, x_p90, y_p90, "P90 → heatwave", COL_RED, dy=8)

    ax.set_title("A. Relative retained connectivity")
    ax.set_xlabel(
        "Thermal analog tolerance (±°C)\n"
        "0.1 = near-exact analogs · 5 = highly relaxed matching"
    )
    ax.set_ylabel("Connectivity retained relative to matched baseline (%)")
    ax.set_ylim(ymin_a, ymax_a)
    ax.set_xlim(0.0, 5.25)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    format_tolerance_ticks(ax)
    clean_axes(ax)

    # -------------------------------------------------------------------------
    # PANEL B: CONNECTIVITY LOSS
    # -------------------------------------------------------------------------

    ax = axes[1]

    y_avg_loss = avg["loss_percent"].to_numpy(dtype=float)
    y_p90_loss = p90["loss_percent"].to_numpy(dtype=float)

    ymax_b = 105
    ymin_b = 0

    add_tolerance_zones(ax, ymin=ymin_b, ymax=ymax_b)

    if len(x_avg) > 0:
        ax.plot(
            x_avg,
            y_avg_loss,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=COL_BLUE,
            zorder=3,
        )

    if len(x_p90) > 0:
        ax.plot(
            x_p90,
            y_p90_loss,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=COL_RED,
            zorder=4,
        )

    ax.axhline(
        0.0,
        color=COL_GREY,
        linestyle="--",
        linewidth=1.2,
        alpha=0.95,
    )

    ax.text(
        0.03,
        1.5,
        "0% = no connectivity loss",
        color=COL_GREY,
        fontsize=9,
        ha="left",
        va="bottom",
    )

    label_line_right(ax, x_avg, y_avg_loss, "Average → heatwave", COL_BLUE, dy=8)
    label_line_right(ax, x_p90, y_p90_loss, "P90 → heatwave", COL_RED, dy=-8)

    ax.set_title("B. Connectivity loss")
    ax.set_xlabel(
        "Thermal analog tolerance (±°C)\n"
        "0.1 = near-exact analogs · 5 = highly relaxed matching"
    )
    ax.set_ylabel("Connectivity loss relative to matched baseline (%)")
    ax.set_ylim(ymin_b, ymax_b)
    ax.set_xlim(0.0, 5.25)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    format_tolerance_ticks(ax)
    clean_axes(ax)

    # -------------------------------------------------------------------------
    # OVERALL TITLE
    # -------------------------------------------------------------------------

    fig.suptitle(
        "Heatwave analog connectivity relative to tolerance-matched baseline networks",
        fontsize=15,
    )

    out_png = OUT_DIR / "summary_relative_retained_and_loss_connectivity.png"

    if SAVE_FIGURE:
        plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
        print(f"\n[OK] wrote {out_png}")

    if SHOW_PREVIEW:
        plt.show()
    else:
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    df = build_summary_table()

    if SAVE_CSV:
        out_csv = OUT_DIR / "relative_retained_and_loss_connectivity.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[OK] wrote {out_csv}")

    plot_summary_figure(df)


if __name__ == "__main__":
    main()