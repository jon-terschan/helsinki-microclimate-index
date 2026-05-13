#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# =============================================================================
# SETTINGS
# =============================================================================

JULIA_EXE = r"C:\Program Files\Julia-1.10.11\bin\julia.exe"

BASE = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape"
)

SOURCE_FILE = (
    BASE /
    "sources" / "thermal_refugia_simple_v1" / 
    "source_p90_coolness_stability.tif"
)

RESISTANCE_FILE = (
    BASE /
    "resistance" /
    "resistance_parsimonious_v4.tif"
)

CONDITION_FILE = (
    BASE /
    "sources" / "thermal_refugia_simple_v1" /
    "condition_average_surface.tif"
)

OUT_ROOT = BASE / "output"

RUN_NAME = "conditional_baseline_connectivity"

RADIUS = 100
BLOCK_SIZE = 21

CONDITION_LOWER = -1
CONDITION_UPPER = 1

PARALLELIZE = True

# =============================================================================
# HELPERS
# =============================================================================

def write_ini(path: Path, project_dir: Path):

    lines = [

        # ---------------------------------------------------------------------
        # CORE INPUTS
        # ---------------------------------------------------------------------

        f"resistance_file = {RESISTANCE_FILE}",
        f"source_file = {SOURCE_FILE}",

        # ---------------------------------------------------------------------
        # CONDITIONAL CONNECTIVITY
        # ---------------------------------------------------------------------

        "conditional = true",
        "n_conditions = 1",

        f"condition1_file = {CONDITION_FILE}",

        "comparison1 = within",

        f"condition1_lower = {CONDITION_LOWER}",
        f"condition1_upper = {CONDITION_UPPER}",

        # ---------------------------------------------------------------------
        # MOVING WINDOW
        # ---------------------------------------------------------------------

        f"radius = {RADIUS}",
        f"block_size = {BLOCK_SIZE}",

        # ---------------------------------------------------------------------
        # OUTPUT
        # ---------------------------------------------------------------------

        f"project_name = {project_dir}",

        "calc_normalized_current = true",
        "calc_flow_potential = true",

        f"parallelize = {str(PARALLELIZE).lower()}",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_omniscape(ini_path: Path):

    code = f'using Omniscape; run_omniscape(raw"{ini_path}")'

    subprocess.run(
        [JULIA_EXE, "-e", code],
        check=True,
    )


def find_file(root: Path, pattern: str):

    matches = list(root.rglob(pattern))

    return matches[0] if matches else None


def load_raster(path: Path):

    with rasterio.open(path) as src:

        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

    return arr


def log_scale(arr):

    arr = np.where(arr < 0, np.nan, arr)

    arr = np.log1p(arr)

    p1, p99 = np.nanpercentile(arr, [1, 99])

    arr = np.clip(arr, p1, p99)

    return (arr - p1) / (p99 - p1)


def rescale01(arr):

    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        return arr

    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    if np.isclose(vmin, vmax):
        return np.zeros_like(arr)

    return (arr - vmin) / (vmax - vmin)


def plot(arr, title, outpath):

    plt.figure(figsize=(8, 8))

    plt.imshow(arr, cmap="viridis")

    plt.title(title)

    plt.axis("off")

    plt.colorbar(
        fraction=0.046,
        pad=0.04,
    )

    plt.savefig(
        outpath,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

RUN_PARENT = OUT_ROOT / f"{RUN_NAME}_run"

CONFIG_PATH = OUT_ROOT / f"{RUN_NAME}.ini"

FIG_DIR = OUT_ROOT / f"{RUN_NAME}_figs"


def main():

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Omniscape automatically creates a new output folder internally.
    #
    # Therefore:
    # - do NOT create RUN_PARENT beforehand
    # - search recursively one level deeper afterwards
    # -------------------------------------------------------------------------

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # WRITE CONFIG
    # -------------------------------------------------------------------------

    write_ini(
        CONFIG_PATH,
        RUN_PARENT,
    )

    print("[RUN] conditional omniscape")

    run_omniscape(CONFIG_PATH)

    print("[DONE] omniscape solve")

    # -------------------------------------------------------------------------
    # FIND OUTPUTS
    # -------------------------------------------------------------------------

    norm_current_path = find_file(
        OUT_ROOT,
        "normalized_cum_currmap.tif",
    )

    flow_path = find_file(
        OUT_ROOT,
        "flow_potential.tif",
    )

    if norm_current_path is None:
        raise FileNotFoundError(
            "normalized_cum_currmap.tif not found"
        )

    print(f"[FOUND] normalized current: {norm_current_path}")

    if flow_path is not None:
        print(f"[FOUND] flow potential: {flow_path}")

    # -------------------------------------------------------------------------
    # NORMALIZED CURRENT
    # -------------------------------------------------------------------------

    norm_current = load_raster(norm_current_path)

    norm_current_log = log_scale(norm_current)

    plot(
        norm_current_log,
        "conditional normalized current",
        FIG_DIR / "normalized_current.png",
    )

    # -------------------------------------------------------------------------
    # FLOW POTENTIAL
    # -------------------------------------------------------------------------

    if flow_path is not None:

        flow = load_raster(flow_path)

        flow_log = log_scale(flow)

        plot(
            flow_log,
            "conditional flow potential",
            FIG_DIR / "flow_potential.png",
        )

    # -------------------------------------------------------------------------
    # SOURCE
    # -------------------------------------------------------------------------

    source = load_raster(SOURCE_FILE)

    source = rescale01(source)

    plot(
        source,
        "baseline coolness source",
        FIG_DIR / "source_strength.png",
    )

    # -------------------------------------------------------------------------
    # CONDITION SURFACE
    # -------------------------------------------------------------------------

    condition = load_raster(CONDITION_FILE)

    condition_scaled = rescale01(condition)

    plot(
        condition_scaled,
        "condition surface",
        FIG_DIR / "condition_surface.png",
    )

    print(f"[OK] figures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()


    #!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

# =============================================================================
# SETTINGS
# =============================================================================

RUN_DIR = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape\output\conditional_baseline_connectivity_run"
)

# choose:
# "normalized_cum_currmap.tif"
# "cum_currmap.tif"
FLOW_RASTER_NAME = "cum_currmap.tif"

# pathway extraction
SMOOTH_SIGMA = 1.2
THRESHOLD_PERCENTILE = 95

# =============================================================================
# HELPERS
# =============================================================================

def find_file(root: Path, pattern: str):

    matches = list(root.rglob(pattern))

    if not matches:
        raise FileNotFoundError(f"Could not find: {pattern}")

    return matches[0]


def load_raster(path: Path):

    with rasterio.open(path) as src:

        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        profile = src.profile.copy()

    return arr, profile


def log_scale(arr):

    arr = np.where(arr <= 0, np.nan, arr)

    arr = np.log1p(arr)

    vals = arr[np.isfinite(arr)]

    p1, p99 = np.percentile(vals, [1, 99])

    arr = np.clip(arr, p1, p99)

    return (arr - p1) / (p99 - p1)


def write_geotiff(path: Path, arr: np.ndarray, profile: dict):

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


# =============================================================================
# MAIN
# =============================================================================

def main():

    figs_dir = RUN_DIR / "dominant_pathways"
    figs_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # LOAD FLOW
    # -------------------------------------------------------------------------

    flow_path = find_file(
        RUN_DIR,
        FLOW_RASTER_NAME,
    )

    print(f"[INFO] using: {flow_path}")

    flow, profile = load_raster(flow_path)

    # -------------------------------------------------------------------------
    # SMOOTH
    # -------------------------------------------------------------------------

    smoothed = flow.copy()

    nanmask = ~np.isfinite(smoothed)

    smoothed[nanmask] = 0

    smoothed = gaussian_filter(
        smoothed,
        sigma=SMOOTH_SIGMA,
    )

    smoothed[nanmask] = np.nan

    # -------------------------------------------------------------------------
    # LOG SCALE
    # -------------------------------------------------------------------------

    scaled = log_scale(smoothed)

    # -------------------------------------------------------------------------
    # EXTRACT DOMINANT PATHWAYS
    # -------------------------------------------------------------------------

    vals = scaled[np.isfinite(scaled)]

    threshold = np.percentile(
        vals,
        THRESHOLD_PERCENTILE,
    )

    dominant = np.where(
        scaled >= threshold,
        scaled,
        np.nan,
    )

    # -------------------------------------------------------------------------
    # EXPORT RASTER
    # -------------------------------------------------------------------------

    dominant_out = (
        figs_dir /
        "dominant_pathways.tif"
    )

    write_geotiff(
        dominant_out,
        dominant,
        profile,
    )

    # -------------------------------------------------------------------------
    # FIGURE
    # -------------------------------------------------------------------------

    plt.figure(figsize=(10, 10))

    plt.imshow(
        scaled,
        cmap="Greys",
        alpha=0.35,
    )

    plt.imshow(
        dominant,
        cmap="inferno",
    )

    plt.title(
        f"Dominant pathways (top {100 - THRESHOLD_PERCENTILE:.1f}%)"
    )

    plt.axis("off")

    plt.colorbar(
        fraction=0.046,
        pad=0.04,
    )

    fig_out = (
        figs_dir /
        "dominant_pathways.png"
    )

    plt.savefig(
        fig_out,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"[OK] wrote: {dominant_out}")
    print(f"[OK] wrote: {fig_out}")


if __name__ == "__main__":
    main()