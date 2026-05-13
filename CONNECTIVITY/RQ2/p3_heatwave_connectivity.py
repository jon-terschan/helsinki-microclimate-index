#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# =============================================================================
# SETTINGS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")

RUN_PARENT = BASE / "output" / "future_condition_multiruns"

OUT_DIR = BASE / "output" / "future_condition_comparison_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# choose one present condition
PRESENT_CONDITION = "condition_average"

# heatwaves to compare
HEATWAVES = [
    "condition_heatwave_2010",
    "condition_heatwave_2018",
    "condition_heatwave_2021",
]

# tolerances to show in the 1x4 figure
TOLERANCES = [1.0, 2.0, 3.0]

# use this tolerance to define the shared display range
REFERENCE_TOLERANCE = 3.0

# choose "cum_currmap.tif" or "normalized_cum_currmap.tif"
FLOW_RASTER_NAME = "cum_currmap.tif"

# display options
USE_LOG_SCALE = True
LOW_PERCENTILE = 1
HIGH_PERCENTILE = 99


# =============================================================================
# HELPERS
# =============================================================================

def tolerance_label(value: float) -> str:
    return str(value).replace(".", "p")


def run_name(present: str, future: str, tolerance: float) -> str:
    return (
        f"futurecompare__"
        f"{present}__to__{future}__"
        f"pm{tolerance_label(tolerance)}deg"
    )


def find_run_dir(run_parent: Path, name: str) -> Path:
    candidates = [
        p for p in run_parent.rglob("*")
        if p.is_dir() and name in p.name
    ]

    if not candidates:
        raise FileNotFoundError(f"No run folder found for: {name}")

    # prefer folders that already contain the requested flow raster
    with_flow = [
        p for p in candidates
        if list(p.rglob(FLOW_RASTER_NAME))
    ]

    if with_flow:
        return sorted(with_flow, key=lambda p: len(str(p)))[0]

    return sorted(candidates, key=lambda p: len(str(p)))[0]


def find_file(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} in {root}")
    return matches[0]


def load_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

    return arr


def transform_for_display(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)

    if USE_LOG_SCALE:
        arr = np.where(arr <= 0, np.nan, arr)
        arr = np.log1p(arr)

    return arr


def get_shared_range(reference_arr: np.ndarray) -> tuple[float, float]:
    vals = reference_arr[np.isfinite(reference_arr)]

    if vals.size == 0:
        raise ValueError("Reference raster has no finite values.")

    vmin, vmax = np.nanpercentile(
        vals,
        [LOW_PERCENTILE, HIGH_PERCENTILE],
    )

    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

    if np.isclose(vmin, vmax):
        raise ValueError("Reference raster has no useful value range.")

    return float(vmin), float(vmax)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    for heatwave in HEATWAVES:
        print(f"\n[HEATWAVE] {heatwave}")

        rasters = {}

        # ---------------------------------------------------------------------
        # LOAD ALL TOLERANCE RASTERS
        # ---------------------------------------------------------------------

        for tol in TOLERANCES:
            name = run_name(
                present=PRESENT_CONDITION,
                future=heatwave,
                tolerance=tol,
            )

            this_run_dir = find_run_dir(
                RUN_PARENT,
                name,
            )

            flow_path = find_file(
                this_run_dir,
                FLOW_RASTER_NAME,
            )

            arr = load_raster(flow_path)
            arr_display = transform_for_display(arr)

            rasters[tol] = arr_display

            vals = arr[np.isfinite(arr) & (arr > 0)]
            if vals.size:
                print(
                    f"  ±{tol:g}°C: "
                    f"raw max={np.nanmax(vals):.6g}, "
                    f"p99={np.nanpercentile(vals, 99):.6g}, "
                    f"positive_cells={vals.size}"
                )
            else:
                print(f"  ±{tol:g}°C: no positive current")

        # ---------------------------------------------------------------------
        # SHARED RANGE FROM MOST PERMISSIVE TOLERANCE
        # ---------------------------------------------------------------------

        reference = rasters[REFERENCE_TOLERANCE]
        vmin, vmax = get_shared_range(reference)

        print(
            f"  shared display range from ±{REFERENCE_TOLERANCE:g}°C: "
            f"vmin={vmin:.6g}, vmax={vmax:.6g}"
        )

        # ---------------------------------------------------------------------
        # PLOT 1x4
        # ---------------------------------------------------------------------

        fig, axes = plt.subplots(
            1,
            len(TOLERANCES),
            figsize=(20, 6),
            constrained_layout=True,
        )

        if len(TOLERANCES) == 1:
            axes = [axes]

        last_im = None

        for ax, tol in zip(axes, TOLERANCES):
            arr = rasters[tol]

            last_im = ax.imshow(
                arr,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_title(f"±{tol:g}°C")
            ax.axis("off")

        title_scale = "log-scaled" if USE_LOG_SCALE else "raw"
        fig.suptitle(
            f"{heatwave}: cumulative current ({title_scale}); shared range from ±{REFERENCE_TOLERANCE:g}°C",
            fontsize=14,
        )

        cbar = fig.colorbar(
            last_im,
            ax=axes,
            fraction=0.025,
            pad=0.02,
        )
        cbar.set_label("Display value")

        out_png = OUT_DIR / f"compare_cum_current_{PRESENT_CONDITION}_to_{heatwave}.png"

        plt.savefig(
            out_png,
            dpi=250,
            bbox_inches="tight",
        )
        plt.close()

        print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# =============================================================================
# SETTINGS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")

RUN_PARENT = BASE / "output" / "future_condition_multiruns"

OUT_DIR = BASE / "output" / "future_condition_comparison_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_FLOW_FILE = (
    BASE
    / "output"
    / "conditional_multiruns"
    / "condition_average__pm1p0deg"
    / "cum_currmap.tif"
)

PRESENT_CONDITION = "condition_average"

HEATWAVES = [
    "condition_heatwave_2010",
    "condition_heatwave_2018",
    "condition_heatwave_2021",
]

# removed 0.5; baseline is shown first, then future analog runs
TOLERANCES = [1.0, 2.0, 3.0, 4.0]

FLOW_RASTER_NAME = "cum_currmap.tif"

# Display is fixed to baseline current range
USE_LOG_SCALE = True
LOW_PERCENTILE = 1
HIGH_PERCENTILE = 99


# =============================================================================
# HELPERS
# =============================================================================

def tolerance_label(value: float) -> str:
    return str(value).replace(".", "p")


def run_name(present: str, future: str, tolerance: float) -> str:
    return (
        f"futurecompare__"
        f"{present}__to__{future}__"
        f"pm{tolerance_label(tolerance)}deg"
    )


def find_run_dir(run_parent: Path, name: str) -> Path:
    candidates = [
        p for p in run_parent.rglob("*")
        if p.is_dir() and name in p.name
    ]

    if not candidates:
        raise FileNotFoundError(f"No run folder found for: {name}")

    with_flow = [
        p for p in candidates
        if list(p.rglob(FLOW_RASTER_NAME))
    ]

    if with_flow:
        return sorted(with_flow, key=lambda p: len(str(p)))[0]

    return sorted(candidates, key=lambda p: len(str(p)))[0]


def find_file(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} in {root}")
    return matches[0]


def load_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def transform_for_display(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)

    if USE_LOG_SCALE:
        arr = np.where(arr <= 0, np.nan, arr)
        arr = np.log1p(arr)

    return arr


def get_baseline_range(baseline_arr: np.ndarray) -> tuple[float, float]:
    vals = baseline_arr[np.isfinite(baseline_arr)]

    if vals.size == 0:
        raise ValueError("Baseline raster has no finite values.")

    vmin, vmax = np.nanpercentile(vals, [LOW_PERCENTILE, HIGH_PERCENTILE])

    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

    if np.isclose(vmin, vmax):
        raise ValueError("Baseline raster has no useful value range.")

    return float(vmin), float(vmax)


def print_stats(label: str, arr_raw: np.ndarray) -> None:
    vals = arr_raw[np.isfinite(arr_raw) & (arr_raw > 0)]

    if vals.size == 0:
        print(f"  {label}: no positive current")
        return

    print(
        f"  {label}: "
        f"raw max={np.nanmax(vals):.6g}, "
        f"p95={np.nanpercentile(vals, 95):.6g}, "
        f"p99={np.nanpercentile(vals, 99):.6g}, "
        f"positive_cells={vals.size}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not BASELINE_FLOW_FILE.exists():
        raise FileNotFoundError(f"Missing baseline flow file: {BASELINE_FLOW_FILE}")

    baseline_raw = load_raster(BASELINE_FLOW_FILE)
    baseline_display = transform_for_display(baseline_raw)

    vmin, vmax = get_baseline_range(baseline_display)

    print("[BASELINE RANGE]")
    print(f"  file = {BASELINE_FLOW_FILE}")
    print_stats("baseline ±1°C", baseline_raw)
    print(f"  shared display range from baseline: vmin={vmin:.6g}, vmax={vmax:.6g}")

    for heatwave in HEATWAVES:
        print(f"\n[HEATWAVE] {heatwave}")

        rasters_display = {
            "Baseline ±1°C": baseline_display
        }

        rasters_raw = {
            "Baseline ±1°C": baseline_raw
        }

        for tol in TOLERANCES:
            name = run_name(
                present=PRESENT_CONDITION,
                future=heatwave,
                tolerance=tol,
            )

            this_run_dir = find_run_dir(RUN_PARENT, name)
            flow_path = find_file(this_run_dir, FLOW_RASTER_NAME)

            arr_raw = load_raster(flow_path)
            arr_display = transform_for_display(arr_raw)

            label = f"Future ±{tol:g}°C"
            rasters_display[label] = arr_display
            rasters_raw[label] = arr_raw

            print_stats(label, arr_raw)

        # ---------------------------------------------------------------------
        # PLOT 1x4: baseline + ±1, ±2, ±3 future runs
        # ---------------------------------------------------------------------

        labels = list(rasters_display.keys())

        fig, axes = plt.subplots(
            1,
            len(labels),
            figsize=(22, 6),
            constrained_layout=True,
        )

        last_im = None

        for ax, label in zip(axes, labels):
            last_im = ax.imshow(
                rasters_display[label],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(label)
            ax.axis("off")

        title_scale = "log-scaled" if USE_LOG_SCALE else "raw"
        fig.suptitle(
            f"{PRESENT_CONDITION} → {heatwave}: cumulative current ({title_scale}); "
            f"display range fixed to baseline ±1°C",
            fontsize=14,
        )

        cbar = fig.colorbar(
            last_im,
            ax=axes,
            fraction=0.025,
            pad=0.02,
        )
        cbar.set_label("Display value, fixed to baseline range")

        out_png = OUT_DIR / f"compare_baseline_range_{PRESENT_CONDITION}_to_{heatwave}.png"

        plt.savefig(out_png, dpi=250, bbox_inches="tight")
        plt.close()

        print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()


### p90 vs heatwave
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# =============================================================================
# SETTINGS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")

RUN_PARENT = BASE / "output" / "future_condition_multiruns"

OUT_DIR = BASE / "output" / "future_condition_comparison_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_FLOW_FILE = (
    BASE
    / "output"
    / "conditional_multiruns"
    / "condition_p90__pm0p5deg"
    / "cum_currmap.tif"
)

PRESENT_CONDITION = "condition_p90"

HEATWAVES = [
    "condition_heatwave_2010",
    "condition_heatwave_2018",
    "condition_heatwave_2021",
]

TOLERANCES = [0.5, 1.0, 2.0]

FLOW_RASTER_NAME = "cum_currmap.tif"

USE_LOG_SCALE = True
LOW_PERCENTILE = 1
HIGH_PERCENTILE = 99


# =============================================================================
# HELPERS
# =============================================================================

def tolerance_label(value: float) -> str:
    return str(value).replace(".", "p")


def run_name(present: str, future: str, tolerance: float) -> str:
    return (
        f"futurecompare__"
        f"{present}__to__{future}__"
        f"pm{tolerance_label(tolerance)}deg"
    )


def find_run_dir(run_parent: Path, name: str) -> Path | None:
    candidates = [
        p for p in run_parent.rglob("*")
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
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def transform_for_display(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)

    if USE_LOG_SCALE:
        arr = np.where(arr <= 0, np.nan, arr)
        arr = np.log1p(arr)

    return arr


def get_baseline_range(baseline_arr: np.ndarray) -> tuple[float, float]:
    vals = baseline_arr[np.isfinite(baseline_arr)]

    if vals.size == 0:
        raise ValueError("Baseline raster has no finite values.")

    vmin, vmax = np.nanpercentile(vals, [LOW_PERCENTILE, HIGH_PERCENTILE])

    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

    if np.isclose(vmin, vmax):
        raise ValueError("Baseline raster has no useful value range.")

    return float(vmin), float(vmax)


def print_stats(label: str, arr_raw: np.ndarray) -> None:
    vals = arr_raw[np.isfinite(arr_raw) & (arr_raw > 0)]

    if vals.size == 0:
        print(f"  {label}: no positive current")
        return

    print(
        f"  {label}: "
        f"raw max={np.nanmax(vals):.6g}, "
        f"p95={np.nanpercentile(vals, 95):.6g}, "
        f"p99={np.nanpercentile(vals, 99):.6g}, "
        f"positive_cells={vals.size}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not BASELINE_FLOW_FILE.exists():
        raise FileNotFoundError(f"Missing baseline flow file: {BASELINE_FLOW_FILE}")

    baseline_raw = load_raster(BASELINE_FLOW_FILE)
    baseline_display = transform_for_display(baseline_raw)

    vmin, vmax = get_baseline_range(baseline_display)

    print("[BASELINE RANGE]")
    print(f"  file = {BASELINE_FLOW_FILE}")
    print_stats("p90 baseline ±0.5°C", baseline_raw)
    print(f"  shared display range from p90 baseline ±0.5°C: vmin={vmin:.6g}, vmax={vmax:.6g}")

    for heatwave in HEATWAVES:
        print(f"\n[HEATWAVE] {heatwave}")

        rasters_display = {
            "P90 baseline ±0.5°C": baseline_display
        }

        rasters_raw = {
            "P90 baseline ±0.5°C": baseline_raw
        }

        for tol in TOLERANCES:
            name = run_name(
                present=PRESENT_CONDITION,
                future=heatwave,
                tolerance=tol,
            )

            this_run_dir = find_run_dir(RUN_PARENT, name)

            if this_run_dir is None:
                print(f"  Future ±{tol:g}°C: missing run folder")
                continue

            flow_path = find_file(this_run_dir, FLOW_RASTER_NAME)

            if flow_path is None:
                print(f"  Future ±{tol:g}°C: missing {FLOW_RASTER_NAME}")
                continue

            arr_raw = load_raster(flow_path)
            arr_display = transform_for_display(arr_raw)

            label = f"Future ±{tol:g}°C"
            rasters_display[label] = arr_display
            rasters_raw[label] = arr_raw

            print_stats(label, arr_raw)

        labels = list(rasters_display.keys())

        if len(labels) == 1:
            print("  [SKIP FIGURE] no future runs available for this heatwave")
            continue

        fig, axes = plt.subplots(
            1,
            len(labels),
            figsize=(5.5 * len(labels), 6),
            constrained_layout=True,
        )

        if len(labels) == 1:
            axes = [axes]

        last_im = None

        for ax, label in zip(axes, labels):
            last_im = ax.imshow(
                rasters_display[label],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(label)
            ax.axis("off")

        title_scale = "log-scaled" if USE_LOG_SCALE else "raw"
        fig.suptitle(
            f"{PRESENT_CONDITION} → {heatwave}: cumulative current ({title_scale}); "
            f"display range fixed to p90 baseline ±0.5°C",
            fontsize=14,
        )

        cbar = fig.colorbar(
            last_im,
            ax=axes,
            fraction=0.025,
            pad=0.02,
        )
        cbar.set_label("Display value, fixed to p90 baseline ±0.5°C range")

        out_png = OUT_DIR / f"compare_p90_baseline_range_{PRESENT_CONDITION}_to_{heatwave}.png"

        plt.savefig(out_png, dpi=250, bbox_inches="tight")
        plt.close()

        print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()

### baseline average vs averag heatwave
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# =============================================================================
# SETTINGS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")

RUN_PARENT = BASE / "output" / "future_condition_multiruns"

OUT_DIR = BASE / "output" / "future_condition_comparison_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# More restrictive baseline reference
BASELINE_RUN_DIR = (
    BASE
    / "output"
    / "conditional_multiruns"
    / "condition_average__pm0p5deg"
)

BASELINE_FLOW_FILE = BASELINE_RUN_DIR / "cum_currmap.tif"
BASELINE_FLOW_POTENTIAL_FILE = BASELINE_RUN_DIR / "flow_potential.tif"

PRESENT_CONDITION = "condition_average"

HEATWAVES = [
    "condition_heatwave_2010",
    "condition_heatwave_2018",
    "condition_heatwave_2021",
]

TOLERANCES = [2.0, 3.0, 4.0]

FLOW_RASTER_NAME = "cum_currmap.tif"
FLOW_POTENTIAL_NAME = "flow_potential.tif"

USE_LOG_SCALE = True
LOW_PERCENTILE = 1
HIGH_PERCENTILE = 99

# Overlay: show only upper part of current values
CURRENT_OVERLAY_PERCENTILE = 70

# Background transparency
FLOW_POTENTIAL_ALPHA = 0.3

WRITE_AVERAGE_TIFS = False


# =============================================================================
# HELPERS
# =============================================================================

def tolerance_label(value: float) -> str:
    return str(value).replace(".", "p")


def run_name(present: str, future: str, tolerance: float) -> str:
    return (
        f"futurecompare__"
        f"{present}__to__{future}__"
        f"pm{tolerance_label(tolerance)}deg"
    )


def find_run_dir(run_parent: Path, name: str) -> Path:
    candidates = [
        p for p in run_parent.rglob("*")
        if p.is_dir() and name in p.name
    ]

    if not candidates:
        raise FileNotFoundError(f"No run folder found for: {name}")

    with_flow = [
        p for p in candidates
        if list(p.rglob(FLOW_RASTER_NAME))
    ]

    if with_flow:
        return sorted(with_flow, key=lambda p: len(str(p)))[0]

    return sorted(candidates, key=lambda p: len(str(p)))[0]


def find_file(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} in {root}")
    return matches[0]


def load_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = src.profile.copy()
    return arr, profile


def write_geotiff(path: Path, arr: np.ndarray, profile: dict) -> None:
    profile = profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
    )

    out = np.where(np.isnan(arr), -9999.0, arr).astype(np.float32)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def transform_for_display(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)

    if USE_LOG_SCALE:
        arr = np.where(arr <= 0, np.nan, arr)
        arr = np.log1p(arr)

    return arr


def get_range(arr: np.ndarray) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        raise ValueError("Raster has no finite values.")

    vmin, vmax = np.nanpercentile(vals, [LOW_PERCENTILE, HIGH_PERCENTILE])

    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

    if np.isclose(vmin, vmax):
        raise ValueError("Raster has no useful value range.")

    return float(vmin), float(vmax)


def threshold_overlay(arr: np.ndarray, percentile: float) -> np.ndarray:
    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        return arr

    threshold = np.nanpercentile(vals, percentile)

    return np.where(arr >= threshold, arr, np.nan)


def print_stats(label: str, arr_raw: np.ndarray) -> None:
    vals = arr_raw[np.isfinite(arr_raw) & (arr_raw > 0)]

    if vals.size == 0:
        print(f"  {label}: no positive current")
        return

    print(
        f"  {label}: "
        f"raw max={np.nanmax(vals):.6g}, "
        f"p50={np.nanpercentile(vals, 50):.6g}, "
        f"p95={np.nanpercentile(vals, 95):.6g}, "
        f"p99={np.nanpercentile(vals, 99):.6g}, "
        f"positive_cells={vals.size}"
    )


def mean_rasters(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("No rasters supplied for averaging.")

    stack = np.stack(arrays, axis=0)
    return np.nanmean(stack, axis=0).astype(np.float32)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not BASELINE_FLOW_FILE.exists():
        raise FileNotFoundError(f"Missing baseline flow file: {BASELINE_FLOW_FILE}")

    if not BASELINE_FLOW_POTENTIAL_FILE.exists():
        raise FileNotFoundError(f"Missing baseline flow potential file: {BASELINE_FLOW_POTENTIAL_FILE}")

    baseline_raw, baseline_profile = load_raster(BASELINE_FLOW_FILE)
    baseline_flow_potential_raw, _ = load_raster(BASELINE_FLOW_POTENTIAL_FILE)

    baseline_display = transform_for_display(baseline_raw)
    baseline_flow_potential_display = transform_for_display(baseline_flow_potential_raw)

    current_vmin, current_vmax = get_range(baseline_display)
    background_vmin, background_vmax = get_range(baseline_flow_potential_display)

    print("[BASELINE RANGE]")
    print(f"  current file = {BASELINE_FLOW_FILE}")
    print(f"  flow potential file = {BASELINE_FLOW_POTENTIAL_FILE}")
    print_stats("baseline ±0.5°C current", baseline_raw)
    print_stats("baseline ±0.5°C flow potential", baseline_flow_potential_raw)
    print(
        f"  shared current display range from baseline ±0.5°C: "
        f"vmin={current_vmin:.6g}, vmax={current_vmax:.6g}"
    )

    averaged_current_by_tolerance: dict[float, np.ndarray] = {}
    averaged_flow_potential_by_tolerance: dict[float, np.ndarray] = {}

    for tol in TOLERANCES:
        print(f"\n[TOLERANCE] ±{tol:g}°C")

        heatwave_current_arrays = []
        heatwave_flow_potential_arrays = []

        for heatwave in HEATWAVES:
            name = run_name(
                present=PRESENT_CONDITION,
                future=heatwave,
                tolerance=tol,
            )

            this_run_dir = find_run_dir(RUN_PARENT, name)

            current_path = find_file(this_run_dir, FLOW_RASTER_NAME)
            flow_potential_path = find_file(this_run_dir, FLOW_POTENTIAL_NAME)

            current_raw, _ = load_raster(current_path)
            flow_potential_raw, _ = load_raster(flow_potential_path)

            heatwave_current_arrays.append(current_raw)
            heatwave_flow_potential_arrays.append(flow_potential_raw)

            print_stats(f"{heatwave} ±{tol:g}°C current", current_raw)
            print_stats(f"{heatwave} ±{tol:g}°C flow potential", flow_potential_raw)

        avg_current = mean_rasters(heatwave_current_arrays)
        avg_flow_potential = mean_rasters(heatwave_flow_potential_arrays)

        averaged_current_by_tolerance[tol] = avg_current
        averaged_flow_potential_by_tolerance[tol] = avg_flow_potential

        print_stats(f"HEATWAVE MEAN ±{tol:g}°C current", avg_current)
        print_stats(f"HEATWAVE MEAN ±{tol:g}°C flow potential", avg_flow_potential)

        if WRITE_AVERAGE_TIFS:
            current_out_tif = OUT_DIR / (
                f"avg_heatwave_current_{PRESENT_CONDITION}_pm{tolerance_label(tol)}deg.tif"
            )
            flow_potential_out_tif = OUT_DIR / (
                f"avg_heatwave_flow_potential_{PRESENT_CONDITION}_pm{tolerance_label(tol)}deg.tif"
            )

            write_geotiff(current_out_tif, avg_current, baseline_profile)
            write_geotiff(flow_potential_out_tif, avg_flow_potential, baseline_profile)

            print(f"[OK] wrote {current_out_tif}")
            print(f"[OK] wrote {flow_potential_out_tif}")

    # -------------------------------------------------------------------------
    # PLOT: baseline + averaged heatwave tolerance maps
    # -------------------------------------------------------------------------

    panels: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "Baseline ±0.5°C",
            baseline_raw,
            baseline_flow_potential_raw,
        )
    ]

    for tol in TOLERANCES:
        panels.append(
            (
                f"Heatwave mean ±{tol:g}°C",
                averaged_current_by_tolerance[tol],
                averaged_flow_potential_by_tolerance[tol],
            )
        )

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.2 * len(panels), 6),
        constrained_layout=True,
    )

    if len(panels) == 1:
        axes = [axes]

    last_im = None

    for ax, (label, current_raw, flow_potential_raw) in zip(axes, panels):
        current_display = transform_for_display(current_raw)
        flow_potential_display = transform_for_display(flow_potential_raw)

        current_overlay = threshold_overlay(
            current_display,
            CURRENT_OVERLAY_PERCENTILE,
        )

        ax.imshow(
            flow_potential_display,
            cmap="Greys",
            vmin=background_vmin,
            vmax=background_vmax,
            alpha=FLOW_POTENTIAL_ALPHA,
        )

        last_im = ax.imshow(
            current_overlay,
            cmap="viridis",
            vmin=current_vmin,
            vmax=current_vmax,
        )

        ax.set_title(label)
        ax.axis("off")

    title_scale = "log-scaled" if USE_LOG_SCALE else "raw"
    fig.suptitle(
        f"{PRESENT_CONDITION} → mean heatwave connectivity ({title_scale}); "
        f"current display range fixed to baseline ±0.5°C; "
        f"current overlay shows values ≥ p{CURRENT_OVERLAY_PERCENTILE:g}",
        fontsize=14,
    )

    cbar = fig.colorbar(
        last_im,
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Cumulative current display value, fixed to baseline ±0.5°C")

    out_png = OUT_DIR / (
        f"compare_baseline_pm0p5_vs_heatwave_mean_{PRESENT_CONDITION}_overlay.png"
    )

    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] wrote {out_png}")


if __name__ == "__main__":
    main()

# p90 vs heatwave
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# =============================================================================
# SETTINGS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")

RUN_PARENT = BASE / "output" / "future_condition_multiruns"

OUT_DIR = BASE / "output" / "future_condition_comparison_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Restrictive p90 baseline reference
BASELINE_RUN_DIR = (
    BASE
    / "output"
    / "conditional_multiruns"
    / "condition_p90__pm0p5deg"
)

BASELINE_FLOW_FILE = BASELINE_RUN_DIR / "cum_currmap.tif"
BASELINE_FLOW_POTENTIAL_FILE = BASELINE_RUN_DIR / "flow_potential.tif"

PRESENT_CONDITION = "condition_p90"

HEATWAVES = [
    "condition_heatwave_2010",
    "condition_heatwave_2018",
    "condition_heatwave_2021",
]

# Compare p90 → heatwave using the three lowest tolerances
TOLERANCES = [0.5, 1.0, 2.0]

FLOW_RASTER_NAME = "cum_currmap.tif"
FLOW_POTENTIAL_NAME = "flow_potential.tif"

USE_LOG_SCALE = True
LOW_PERCENTILE = 1
HIGH_PERCENTILE = 99

# Overlay: show only upper part of current values
CURRENT_OVERLAY_PERCENTILE = 70

# Background transparency
FLOW_POTENTIAL_ALPHA = 0.3

WRITE_AVERAGE_TIFS = False


# =============================================================================
# HELPERS
# =============================================================================

def tolerance_label(value: float) -> str:
    return str(value).replace(".", "p")


def run_name(present: str, future: str, tolerance: float) -> str:
    return (
        f"futurecompare__"
        f"{present}__to__{future}__"
        f"pm{tolerance_label(tolerance)}deg"
    )


def find_run_dir(run_parent: Path, name: str) -> Path:
    candidates = [
        p for p in run_parent.rglob("*")
        if p.is_dir() and name in p.name
    ]

    if not candidates:
        raise FileNotFoundError(f"No run folder found for: {name}")

    with_flow = [
        p for p in candidates
        if list(p.rglob(FLOW_RASTER_NAME))
    ]

    if with_flow:
        return sorted(with_flow, key=lambda p: len(str(p)))[0]

    return sorted(candidates, key=lambda p: len(str(p)))[0]


def find_file(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} in {root}")
    return matches[0]


def load_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = src.profile.copy()
    return arr, profile


def write_geotiff(path: Path, arr: np.ndarray, profile: dict) -> None:
    profile = profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
    )

    out = np.where(np.isnan(arr), -9999.0, arr).astype(np.float32)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def transform_for_display(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)

    if USE_LOG_SCALE:
        arr = np.where(arr <= 0, np.nan, arr)
        arr = np.log1p(arr)

    return arr


def get_range(arr: np.ndarray) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        raise ValueError("Raster has no finite values.")

    vmin, vmax = np.nanpercentile(vals, [LOW_PERCENTILE, HIGH_PERCENTILE])

    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

    if np.isclose(vmin, vmax):
        raise ValueError("Raster has no useful value range.")

    return float(vmin), float(vmax)


def threshold_overlay(arr: np.ndarray, percentile: float) -> np.ndarray:
    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        return arr

    threshold = np.nanpercentile(vals, percentile)

    return np.where(arr >= threshold, arr, np.nan)


def print_stats(label: str, arr_raw: np.ndarray) -> None:
    vals = arr_raw[np.isfinite(arr_raw) & (arr_raw > 0)]

    if vals.size == 0:
        print(f"  {label}: no positive current")
        return

    print(
        f"  {label}: "
        f"raw max={np.nanmax(vals):.6g}, "
        f"p50={np.nanpercentile(vals, 50):.6g}, "
        f"p95={np.nanpercentile(vals, 95):.6g}, "
        f"p99={np.nanpercentile(vals, 99):.6g}, "
        f"positive_cells={vals.size}"
    )


def mean_rasters(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("No rasters supplied for averaging.")

    stack = np.stack(arrays, axis=0)
    return np.nanmean(stack, axis=0).astype(np.float32)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not BASELINE_FLOW_FILE.exists():
        raise FileNotFoundError(f"Missing baseline flow file: {BASELINE_FLOW_FILE}")

    if not BASELINE_FLOW_POTENTIAL_FILE.exists():
        raise FileNotFoundError(
            f"Missing baseline flow potential file: {BASELINE_FLOW_POTENTIAL_FILE}"
        )

    baseline_raw, baseline_profile = load_raster(BASELINE_FLOW_FILE)
    baseline_flow_potential_raw, _ = load_raster(BASELINE_FLOW_POTENTIAL_FILE)

    baseline_display = transform_for_display(baseline_raw)
    baseline_flow_potential_display = transform_for_display(baseline_flow_potential_raw)

    current_vmin, current_vmax = get_range(baseline_display)
    background_vmin, background_vmax = get_range(baseline_flow_potential_display)

    print("[BASELINE RANGE]")
    print(f"  current file = {BASELINE_FLOW_FILE}")
    print(f"  flow potential file = {BASELINE_FLOW_POTENTIAL_FILE}")
    print_stats("p90 baseline ±0.5°C current", baseline_raw)
    print_stats("p90 baseline ±0.5°C flow potential", baseline_flow_potential_raw)
    print(
        f"  shared current display range from p90 baseline ±0.5°C: "
        f"vmin={current_vmin:.6g}, vmax={current_vmax:.6g}"
    )

    averaged_current_by_tolerance: dict[float, np.ndarray] = {}
    averaged_flow_potential_by_tolerance: dict[float, np.ndarray] = {}

    for tol in TOLERANCES:
        print(f"\n[TOLERANCE] ±{tol:g}°C")

        heatwave_current_arrays = []
        heatwave_flow_potential_arrays = []

        for heatwave in HEATWAVES:
            name = run_name(
                present=PRESENT_CONDITION,
                future=heatwave,
                tolerance=tol,
            )

            try:
                this_run_dir = find_run_dir(RUN_PARENT, name)
                current_path = find_file(this_run_dir, FLOW_RASTER_NAME)
                flow_potential_path = find_file(this_run_dir, FLOW_POTENTIAL_NAME)
            except FileNotFoundError as e:
                print(f"  [MISSING] {heatwave} ±{tol:g}°C: {e}")
                continue

            current_raw, _ = load_raster(current_path)
            flow_potential_raw, _ = load_raster(flow_potential_path)

            heatwave_current_arrays.append(current_raw)
            heatwave_flow_potential_arrays.append(flow_potential_raw)

            print_stats(f"{heatwave} ±{tol:g}°C current", current_raw)
            print_stats(f"{heatwave} ±{tol:g}°C flow potential", flow_potential_raw)

        if not heatwave_current_arrays:
            print(f"  [SKIP] no runs available for ±{tol:g}°C")
            continue

        avg_current = mean_rasters(heatwave_current_arrays)
        avg_flow_potential = mean_rasters(heatwave_flow_potential_arrays)

        averaged_current_by_tolerance[tol] = avg_current
        averaged_flow_potential_by_tolerance[tol] = avg_flow_potential

        print_stats(f"HEATWAVE MEAN ±{tol:g}°C current", avg_current)
        print_stats(f"HEATWAVE MEAN ±{tol:g}°C flow potential", avg_flow_potential)

        if WRITE_AVERAGE_TIFS:
            current_out_tif = OUT_DIR / (
                f"avg_heatwave_current_{PRESENT_CONDITION}_pm{tolerance_label(tol)}deg.tif"
            )
            flow_potential_out_tif = OUT_DIR / (
                f"avg_heatwave_flow_potential_{PRESENT_CONDITION}_pm{tolerance_label(tol)}deg.tif"
            )

            write_geotiff(current_out_tif, avg_current, baseline_profile)
            write_geotiff(flow_potential_out_tif, avg_flow_potential, baseline_profile)

            print(f"[OK] wrote {current_out_tif}")
            print(f"[OK] wrote {flow_potential_out_tif}")

    # -------------------------------------------------------------------------
    # PLOT: p90 baseline + averaged p90→heatwave tolerance maps
    # -------------------------------------------------------------------------

    panels: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "P90 baseline ±0.5°C",
            baseline_raw,
            baseline_flow_potential_raw,
        )
    ]

    for tol in TOLERANCES:
        if tol not in averaged_current_by_tolerance:
            continue

        panels.append(
            (
                f"Heatwave mean ±{tol:g}°C",
                averaged_current_by_tolerance[tol],
                averaged_flow_potential_by_tolerance[tol],
            )
        )

    if len(panels) == 1:
        raise RuntimeError("No heatwave comparison panels available.")

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.2 * len(panels), 6),
        constrained_layout=True,
    )

    if len(panels) == 1:
        axes = [axes]

    last_im = None

    for ax, (label, current_raw, flow_potential_raw) in zip(axes, panels):
        current_display = transform_for_display(current_raw)
        flow_potential_display = transform_for_display(flow_potential_raw)

        current_overlay = threshold_overlay(
            current_display,
            CURRENT_OVERLAY_PERCENTILE,
        )

        ax.imshow(
            flow_potential_display,
            cmap="Greys",
            vmin=background_vmin,
            vmax=background_vmax,
            alpha=FLOW_POTENTIAL_ALPHA,
        )

        last_im = ax.imshow(
            current_overlay,
            cmap="viridis",
            vmin=current_vmin,
            vmax=current_vmax,
        )

        ax.set_title(label)
        ax.axis("off")

    title_scale = "log-scaled" if USE_LOG_SCALE else "raw"
    fig.suptitle(
        f"{PRESENT_CONDITION} → mean heatwave connectivity ({title_scale}); "
        f"current display range fixed to p90 baseline ±0.5°C; "
        f"current overlay shows values ≥ p{CURRENT_OVERLAY_PERCENTILE:g}",
        fontsize=14,
    )

    cbar = fig.colorbar(
        last_im,
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Cumulative current display value, fixed to p90 baseline ±0.5°C")

    out_png = OUT_DIR / (
        f"compare_p90_pm0p5_vs_heatwave_mean_{PRESENT_CONDITION}_overlay.png"
    )

    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] wrote {out_png}")


if __name__ == "__main__":
    main()



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

OUT_DIR = BASE / "output" / "future_condition_comparison_figures" / "summary_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FUTURE_RUN_PARENT = BASE / "output" / "future_condition_multiruns"
BASELINE_RUN_PARENT = BASE / "output" / "conditional_multiruns"

FLOW_RASTER_NAME = "cum_currmap.tif"
FLOW_POTENTIAL_NAME = "flow_potential.tif"

HEATWAVES = [
    "condition_heatwave_2010",
    "condition_heatwave_2018",
    "condition_heatwave_2021",
]

TOLERANCES_AVG = [0.5, 1.0, 2.0, 3.0, 4.0]
TOLERANCES_P90 = [0.5, 1.0, 2.0, 3.0, 4.0]

BASELINE_RUNS = [
    {
        "group": "Average baseline",
        "present": "condition_average",
        "tolerance": 0.5,
        "run_name": "condition_average__pm0p5deg",
    },
    {
        "group": "P90 baseline",
        "present": "condition_p90",
        "tolerance": 0.5,
        "run_name": "condition_p90__pm0p5deg",
    },
]

FUTURE_GROUPS = [
    {
        "group": "Average → heatwave mean",
        "present": "condition_average",
        "tolerances": TOLERANCES_AVG,
    },
    {
        "group": "P90 → heatwave mean",
        "present": "condition_p90",
        "tolerances": TOLERANCES_P90,
    },
]


# =============================================================================
# HELPERS
# =============================================================================

def tolerance_label(value: float) -> str:
    return str(value).replace(".", "p")


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


def raster_metrics(arr: np.ndarray) -> dict:
    finite = np.isfinite(arr)
    positive = finite & (arr > 0)
    vals = arr[positive]

    out = {
        "positive_cells": int(np.sum(positive)),
        "finite_cells": int(np.sum(finite)),
        "total": float(np.nansum(arr)),
        "mean_positive": np.nan,
        "median_positive": np.nan,
        "p95_positive": np.nan,
        "p99_positive": np.nan,
        "max_positive": np.nan,
    }

    if vals.size > 0:
        out.update(
            {
                "mean_positive": float(np.nanmean(vals)),
                "median_positive": float(np.nanmedian(vals)),
                "p95_positive": float(np.nanpercentile(vals, 95)),
                "p99_positive": float(np.nanpercentile(vals, 99)),
                "max_positive": float(np.nanmax(vals)),
            }
        )

    return out


def summarize_run(
    group: str,
    present: str,
    future: str,
    tolerance: float,
    run_type: str,
    run_name: str,
    run_dir: Path,
) -> dict:
    current_path = find_file(run_dir, FLOW_RASTER_NAME)
    flow_potential_path = find_file(run_dir, FLOW_POTENTIAL_NAME)

    if current_path is None:
        raise FileNotFoundError(f"Missing {FLOW_RASTER_NAME} in {run_dir}")

    if flow_potential_path is None:
        raise FileNotFoundError(f"Missing {FLOW_POTENTIAL_NAME} in {run_dir}")

    current = load_raster(current_path)
    flow_potential = load_raster(flow_potential_path)

    current_metrics = raster_metrics(current)
    fp_metrics = raster_metrics(flow_potential)

    row = {
        "group": group,
        "present": present,
        "future": future,
        "tolerance": tolerance,
        "run_type": run_type,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "current_file": str(current_path),
        "flow_potential_file": str(flow_potential_path),
    }

    for k, v in current_metrics.items():
        row[f"current_{k}"] = v

    for k, v in fp_metrics.items():
        row[f"flow_potential_{k}"] = v

    return row


def mean_rows(rows: list[dict], group: str, present: str, tolerance: float) -> dict:
    numeric_keys = [
        k for k, v in rows[0].items()
        if isinstance(v, (int, float, np.integer, np.floating))
    ]

    out = {
        "group": group,
        "present": present,
        "future": "heatwave_mean",
        "tolerance": tolerance,
        "run_type": "future_mean",
        "run_name": f"{present}_to_heatwave_mean_pm{tolerance_label(tolerance)}deg",
        "run_dir": "",
        "current_file": "",
        "flow_potential_file": "",
        "n_heatwaves_available": len(rows),
    }

    for key in numeric_keys:
        vals = np.array([r[key] for r in rows], dtype=float)
        out[key] = float(np.nanmean(vals))

    return out


def plot_metric_lines(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_png: Path,
    title: str,
    include_baselines: bool = True,
) -> None:
    plt.figure(figsize=(9, 6))

    future_df = df[df["run_type"] == "future_mean"].copy()

    for group, sub in future_df.groupby("group"):
        sub = sub.sort_values("tolerance")
        plt.plot(
            sub["tolerance"],
            sub[metric],
            marker="o",
            label=group,
        )

    if include_baselines:
        baseline_df = df[df["run_type"] == "baseline"].copy()

        for _, row in baseline_df.iterrows():
            plt.axhline(
                row[metric],
                linestyle="--",
                linewidth=1,
                alpha=0.8,
                label=f"{row['group']} ±{row['tolerance']:g}°C",
            )

    plt.xlabel("Condition tolerance (±°C)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def plot_active_fraction(
    df: pd.DataFrame,
    out_png: Path,
) -> None:
    df = df.copy()
    df["current_active_fraction"] = (
        df["current_positive_cells"] / df["current_finite_cells"]
    )

    plot_metric_lines(
        df,
        metric="current_active_fraction",
        ylabel="Fraction of finite cells with current > 0",
        out_png=out_png,
        title="Spatial extent of active cumulative current",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    rows: list[dict] = []

    # -------------------------------------------------------------------------
    # BASELINE RUNS
    # -------------------------------------------------------------------------

    print("[BASELINES]")

    for item in BASELINE_RUNS:
        run_dir = find_run_dir(BASELINE_RUN_PARENT, item["run_name"])

        if run_dir is None:
            print(f"  [MISSING] {item['run_name']}")
            continue

        row = summarize_run(
            group=item["group"],
            present=item["present"],
            future="none",
            tolerance=item["tolerance"],
            run_type="baseline",
            run_name=item["run_name"],
            run_dir=run_dir,
        )

        rows.append(row)

        print(
            f"  [OK] {item['run_name']}: "
            f"current_total={row['current_total']:.6g}, "
            f"flow_potential_total={row['flow_potential_total']:.6g}, "
            f"active_cells={row['current_positive_cells']}"
        )

    # -------------------------------------------------------------------------
    # FUTURE RUNS + HEATWAVE MEANS
    # -------------------------------------------------------------------------

    print("\n[FUTURE RUNS]")

    for group in FUTURE_GROUPS:
        group_name = group["group"]
        present = group["present"]

        for tolerance in group["tolerances"]:
            heatwave_rows = []

            for heatwave in HEATWAVES:
                rn = future_run_name(present, heatwave, tolerance)
                run_dir = find_run_dir(FUTURE_RUN_PARENT, rn)

                if run_dir is None:
                    print(f"  [MISSING] {rn}")
                    continue

                try:
                    row = summarize_run(
                        group=group_name,
                        present=present,
                        future=heatwave,
                        tolerance=tolerance,
                        run_type="future_single",
                        run_name=rn,
                        run_dir=run_dir,
                    )
                except FileNotFoundError as e:
                    print(f"  [MISSING OUTPUT] {rn}: {e}")
                    continue

                rows.append(row)
                heatwave_rows.append(row)

                print(
                    f"  [OK] {rn}: "
                    f"current_total={row['current_total']:.6g}, "
                    f"flow_potential_total={row['flow_potential_total']:.6g}, "
                    f"active_cells={row['current_positive_cells']}"
                )

            if heatwave_rows:
                avg_row = mean_rows(
                    heatwave_rows,
                    group=group_name,
                    present=present,
                    tolerance=tolerance,
                )
                rows.append(avg_row)

                print(
                    f"  [MEAN] {avg_row['run_name']}: "
                    f"current_total={avg_row['current_total']:.6g}, "
                    f"flow_potential_total={avg_row['flow_potential_total']:.6g}, "
                    f"active_cells={avg_row['current_positive_cells']:.1f}"
                )

    if not rows:
        raise RuntimeError("No runs found.")

    df = pd.DataFrame(rows)

    out_csv = OUT_DIR / "connectivity_summary_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] wrote {out_csv}")

    # -------------------------------------------------------------------------
    # IMPACTFUL SUMMARY PLOTS
    # -------------------------------------------------------------------------

    plot_metric_lines(
        df,
        metric="current_total",
        ylabel="Total cumulative current",
        out_png=OUT_DIR / "summary_total_cumulative_current.png",
        title="Total cumulative current across condition tolerances",
    )

    plot_metric_lines(
        df,
        metric="flow_potential_total",
        ylabel="Total flow potential",
        out_png=OUT_DIR / "summary_total_flow_potential.png",
        title="Total flow potential across condition tolerances",
    )

    plot_metric_lines(
        df,
        metric="current_positive_cells",
        ylabel="Cells with cumulative current > 0",
        out_png=OUT_DIR / "summary_active_current_cells.png",
        title="Spatial extent of active cumulative current",
    )

    plot_active_fraction(
        df,
        out_png=OUT_DIR / "summary_active_current_fraction.png",
    )

    plot_metric_lines(
        df,
        metric="current_p95_positive",
        ylabel="P95 cumulative current among active cells",
        out_png=OUT_DIR / "summary_p95_cumulative_current.png",
        title="Upper-tail cumulative current among active cells",
    )

    print("[OK] wrote summary plots")


if __name__ == "__main__":
    main()

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

OUT_DIR = BASE / "output" / "future_condition_comparison_figures" / "summary_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLOW_RASTER_NAME = "cum_currmap.tif"

HEATWAVES = [
    "condition_heatwave_2010",
    "condition_heatwave_2018",
    "condition_heatwave_2021",
]

SCENARIOS = [
    {
        "label": "Average → heatwave",
        "present": "condition_average",
        "tolerances": [0.5, 1.0, 2.0, 3.0],
    },
    {
        "label": "P90 → heatwave",
        "present": "condition_p90",
        "tolerances": [0.5, 1.0, 2.0, 3.0],
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


def load_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def total_positive_current(path: Path) -> float:
    arr = load_raster(path)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0)
    return float(np.sum(arr))


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

        for tol in scenario["tolerances"]:
            print(f"  ±{tol:g}°C")

            # -----------------------------------------------------------------
            # TOLERANCE-MATCHED BASELINE
            # -----------------------------------------------------------------

            base_name = baseline_run_name(present, tol)
            base_dir = find_run_dir(BASELINE_RUN_PARENT, base_name)

            if base_dir is None:
                print(f"    [MISSING BASELINE] {base_name}")
                continue

            base_flow_path = find_file(base_dir, FLOW_RASTER_NAME)

            if base_flow_path is None:
                print(f"    [MISSING BASELINE FLOW] {base_name}")
                continue

            baseline_total = total_positive_current(base_flow_path)

            print(f"    baseline total = {baseline_total:.6g}")

            # -----------------------------------------------------------------
            # FUTURE HEATWAVE RUNS
            # -----------------------------------------------------------------

            future_totals = []

            for heatwave in HEATWAVES:
                fut_name = future_run_name(present, heatwave, tol)
                fut_dir = find_run_dir(FUTURE_RUN_PARENT, fut_name)

                if fut_dir is None:
                    print(f"    [MISSING FUTURE] {fut_name}")
                    continue

                fut_flow_path = find_file(fut_dir, FLOW_RASTER_NAME)

                if fut_flow_path is None:
                    print(f"    [MISSING FUTURE FLOW] {fut_name}")
                    continue

                fut_total = total_positive_current(fut_flow_path)
                future_totals.append(fut_total)

                print(f"    {heatwave}: {fut_total:.6g}")

            if not future_totals:
                print("    [SKIP] no future totals available")
                continue

            future_mean_total = float(np.mean(future_totals))

            if baseline_total > 0:
                relative = future_mean_total / baseline_total
            else:
                relative = np.nan

            rows.append(
                {
                    "scenario": label,
                    "present": present,
                    "tolerance": tol,
                    "baseline_total": baseline_total,
                    "future_mean_total": future_mean_total,
                    "relative_to_matched_baseline": relative,
                    "n_heatwaves": len(future_totals),
                }
            )

            print(
                f"    future mean = {future_mean_total:.6g}; "
                f"relative = {relative:.4g}"
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No matched baseline/future rows found.")

    out_csv = OUT_DIR / "relative_connectivity_tolerance_matched_baseline.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] wrote {out_csv}")

    # -------------------------------------------------------------------------
    # AUC SUMMARY
    # -------------------------------------------------------------------------

    auc_rows = []

    for scenario, sub in df.groupby("scenario"):
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
    out_auc_csv = OUT_DIR / "relative_connectivity_tolerance_matched_auc.csv"
    auc_df.to_csv(out_auc_csv, index=False)
    print(f"[OK] wrote {out_auc_csv}")

    # -------------------------------------------------------------------------
    # FIGURE
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )

    colors = {
        "Average → heatwave": "#2c7fb8",
        "P90 → heatwave": "#f03b20",
    }

    ax = axes[0]

    for scenario, sub in df.groupby("scenario"):
        sub = sub.sort_values("tolerance")

        x = sub["tolerance"].to_numpy(dtype=float)
        y = sub["relative_to_matched_baseline"].to_numpy(dtype=float)

        color = colors.get(scenario, "black")

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=color,
            label=scenario,
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
    ax.set_ylabel("Future mean cumulative current / same-tolerance baseline")
    ax.set_title("A. Heatwave analog connectivity relative to matched baseline")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    ax.set_ylim(bottom=0)

    # -------------------------------------------------------------------------
    # AUC BAR PANEL
    # -------------------------------------------------------------------------

    ax = axes[1]

    auc_df = auc_df.sort_values("scenario")
    bar_colors = [colors.get(s, "grey") for s in auc_df["scenario"]]

    ax.bar(
        auc_df["scenario"],
        auc_df["auc_relative_to_matched_baseline"],
        color=bar_colors,
        alpha=0.75,
    )

    for i, row in auc_df.reset_index(drop=True).iterrows():
        ax.text(
            i,
            row["auc_relative_to_matched_baseline"],
            f"{row['auc_relative_to_matched_baseline']:.2f}",
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

    out_png = OUT_DIR / "summary_relative_connectivity_tolerance_matched_baseline.png"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()