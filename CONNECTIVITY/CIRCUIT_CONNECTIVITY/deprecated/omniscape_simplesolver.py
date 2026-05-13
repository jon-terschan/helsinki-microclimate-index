#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

# =============================================================================
# DEBUG MASTER SETTINGS — CHANGE THESE ONLY
# =============================================================================

SELECTED_SOURCE_SCENARIO = "ref_p95"     # ref_p75 / ref_p90 / ref_p95
CLIMATE_STATE = "baseline_avg"           # baseline_avg / baseline_p90 / h2010 / h2018 / h2021
RESISTANCE_NAME = "parsimonious_bldgNA"           # permissive / conservative / parsimonious

RADIUS = 100                              # 50 = 500 m at 10 m pixels
BLOCK_SIZE = 21                           # odd integer, can be changed for sensitivity tests
SOURCE_THRESHOLD = 0                     # optional; keep 0 unless you want to hard-threshold sources
PARALLELIZE = True
MAX_PARALLEL = None                      # set to an int if you want to limit threads
CALC_FLOW_POTENTIAL = True
CALC_NORMALIZED_CURRENT = True
SOURCE_FROM_RESISTANCE = False
WRITE_RAW_CURRMAP = False                # turn on if you want raw current for diagnostics
SKIP_IF_COMPLETE = False                 # set True to avoid rerunning if outputs already exist

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")
SOURCE_ROOT = BASE / "sources" / SELECTED_SOURCE_SCENARIO
RESISTANCE_ROOT = BASE / "resistance"
OUT_ROOT = BASE / "output" / "debug"

JULIA_EXE = r"C:\Program Files\Julia-1.10.11\bin\julia.exe"

# =============================================================================
# HELPERS
# =============================================================================

def find_source_file(source_root: Path, climate_state: str) -> Path:
    source_dir = source_root / climate_state
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    candidates = sorted(source_dir.glob("*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No source tif found in {source_dir}")

    # Prefer the canonical naming if present
    preferred = [p for p in candidates if p.name.startswith(f"omni_source_{climate_state}_")]
    return preferred[0] if preferred else candidates[0]


def build_run_name() -> str:
    return f"{SELECTED_SOURCE_SCENARIO}__{CLIMATE_STATE}__{RESISTANCE_NAME}__r{RADIUS:03d}_b{BLOCK_SIZE:02d}"


def write_ini(
    out_path: Path,
    resistance: Path,
    source: Path,
    radius: int,
    block_size: int,
    project_name: Path,
) -> None:
    lines = [
        f"resistance_file = {resistance}",
        f"source_file = {source}",
        f"radius = {radius}",
        f"block_size = {block_size}",
        f"project_name = {project_name}",
        f"source_threshold = {SOURCE_THRESHOLD}",
        f"source_from_resistance = {str(SOURCE_FROM_RESISTANCE).lower()}",
        f"calc_flow_potential = {str(CALC_FLOW_POTENTIAL).lower()}",
        f"calc_normalized_current = {str(CALC_NORMALIZED_CURRENT).lower()}",
        f"parallelize = {str(PARALLELIZE).lower()}",
    ]
    if MAX_PARALLEL is not None:
        lines.append(f"max_parallel = {int(MAX_PARALLEL)}")
    if WRITE_RAW_CURRMAP:
        lines.append("write_raw_currmap = true")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_omniscape(ini_path: Path) -> None:
    code = f'using Omniscape; run_omniscape(raw"{ini_path}")'
    subprocess.run([JULIA_EXE, "-e", code], check=True)


def is_complete(run_dir: Path) -> bool:
    return any(run_dir.rglob("normalized_cum_currmap.tif"))


def write_run_log(run_dir: Path, params: dict) -> None:
    (run_dir / "run_log.json").write_text(json.dumps(params, indent=2), encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    resistance_path = RESISTANCE_ROOT / f"resistance_{RESISTANCE_NAME}.tif"
    if not resistance_path.exists():
        raise FileNotFoundError(f"Missing resistance file: {resistance_path}")

    source_file = find_source_file(SOURCE_ROOT, CLIMATE_STATE)

    run_name = build_run_name()
    run_dir = OUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ini_path = run_dir / "config.ini"

    params = {
        "source_scenario": SELECTED_SOURCE_SCENARIO,
        "climate_state": CLIMATE_STATE,
        "source_file": str(source_file),
        "resistance_name": RESISTANCE_NAME,
        "resistance_file": str(resistance_path),
        "radius_pixels": RADIUS,
        "radius_meters": RADIUS * 10,
        "block_size": BLOCK_SIZE,
        "source_threshold": SOURCE_THRESHOLD,
        "project_name": str(run_dir),
        "run_dir": str(run_dir),
        "parallelize": PARALLELIZE,
        "max_parallel": MAX_PARALLEL,
        "calc_flow_potential": CALC_FLOW_POTENTIAL,
        "calc_normalized_current": CALC_NORMALIZED_CURRENT,
        "write_raw_currmap": WRITE_RAW_CURRMAP,
    }

    write_run_log(run_dir, params)

    if SKIP_IF_COMPLETE and is_complete(run_dir):
        print(f"[SKIP] {run_name} (already complete)")
        return

    write_ini(
        out_path=ini_path,
        resistance=resistance_path,
        source=source_file,
        radius=RADIUS,
        block_size=BLOCK_SIZE,
        project_name=run_dir,
    )

    print(f"[RUN] {run_name}")
    print(f"       source = {source_file}")
    print(f"       resistance = {resistance_path}")
    print(f"       radius = {RADIUS} px")
    print(f"       block_size = {BLOCK_SIZE}")
    run_omniscape(ini_path)
    print("[DONE]")


if __name__ == "__main__":
    main()


#

#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# =============================================================================
# CHANGE THIS
# =============================================================================

RUN_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape\output\conditional_baseline_connectivity_run_1")

# =============================================================================
# HELPERS
# =============================================================================

def load_raster(path):
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
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if np.isclose(vmin, vmax):
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def clip_source(arr, low=1, high=99):
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return arr
    p_low, p_high = np.percentile(vals, [low, high])
    return np.clip(arr, p_low, p_high)


def plot(arr, title, outpath):
    plt.figure(figsize=(8, 8))
    plt.imshow(arr, cmap="viridis")
    plt.title(title)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def find_file(run_dir, pattern):
    matches = list(run_dir.rglob(pattern))
    return matches[0] if matches else None


def read_source_from_ini(ini_path):
    if not ini_path.exists():
        return None

    with open(ini_path) as f:
        for line in f:
            if "source_file" in line:
                return Path(line.split("=")[1].strip())

    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    figs_dir = RUN_DIR / "figs"
    figs_dir.mkdir(exist_ok=True)

    # ---- find outputs ----
    norm_current_path = find_file(RUN_DIR, "normalized_cum_currmap.tif")
    cum_current_path = find_file(RUN_DIR, "cum_currmap.tif")
    flow_path = find_file(RUN_DIR, "flow_potential.tif")

    if norm_current_path is None:
        raise FileNotFoundError("normalized_cum_currmap.tif not found")

    # ---- load + plot normalized current ----
    norm_current = load_raster(norm_current_path)
    norm_current_log = log_scale(norm_current)
    plot(norm_current_log, "normalized current (log)", figs_dir / "normalized_current.png")

    # ---- cumulative current ----
    if cum_current_path:
        cum = load_raster(cum_current_path)
        cum_log = log_scale(cum)
        plot(cum_log, "cumulative current (log)", figs_dir / "cum_current.png")

    # ---- flow potential ----
    if flow_path:
        flow = load_raster(flow_path)
        flow_log = log_scale(flow)
        plot(flow_log, "flow potential (log)", figs_dir / "flow.png")

    # ---- source ----
    ini_path = RUN_DIR / "config.ini"
    source_path = read_source_from_ini(ini_path)

    if source_path and source_path.exists():
        source = load_raster(source_path)
        source = rescale01(clip_source(source))
        plot(source, "source (scaled)", figs_dir / "source.png")

    print(f"[OK] Figures written to {figs_dir}")


if __name__ == "__main__":
    main()


#### THRESHOLD FILTER

#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

# =============================================================================
# MASTER SETTINGS
# =============================================================================

SELECTED_SOURCE_SCENARIO = "ref_p95"   # ref_p75 / ref_p90 / ref_p95
CLIMATE_STATE = "baseline_avg"         # baseline_avg / baseline_p90 / h2010 / h2018 / h2021
RESISTANCE_NAME = "parsimonious_bldgNA"       # permissive / conservative / parsimonious

RADIUS = 100
BLOCK_SIZE = 21

# two-run debug comparison:
SOURCE_THRESHOLD_A = 0
SOURCE_THRESHOLD_B = 0.9

CALC_FLOW_POTENTIAL = True
CALC_NORMALIZED_CURRENT = True
SOURCE_FROM_RESISTANCE = False
PARALLELIZE = True
MAX_PARALLEL = None
WRITE_RAW_CURRMAP = False

SKIP_IF_COMPLETE = False

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")
SOURCE_ROOT = BASE / "sources" / SELECTED_SOURCE_SCENARIO
RESISTANCE_ROOT = BASE / "resistance"
OUT_ROOT = BASE / "output" / "debug"

JULIA_EXE = r"C:\Program Files\Julia-1.10.11\bin\julia.exe"

# =============================================================================
# HELPERS
# =============================================================================

def find_source_file(source_root: Path, climate_state: str) -> Path:
    source_dir = source_root / climate_state
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    candidates = sorted(source_dir.glob("*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No source tif found in {source_dir}")

    preferred = [p for p in candidates if p.name.startswith(f"omni_source_{climate_state}_")]
    return preferred[0] if preferred else candidates[0]


def build_run_name(threshold: float) -> str:
    thr = f"{threshold:g}".replace(".", "p")
    return f"{SELECTED_SOURCE_SCENARIO}__{CLIMATE_STATE}__{RESISTANCE_NAME}__r{RADIUS:03d}_b{BLOCK_SIZE:02d}__st{thr}"


def write_ini(
    out_path: Path,
    resistance: Path,
    source: Path,
    radius: int,
    block_size: int,
    project_name: Path,
    source_threshold: float,
) -> None:
    lines = [
        f"resistance_file = {resistance}",
        f"source_file = {source}",
        f"radius = {radius}",
        f"block_size = {block_size}",
        f"project_name = {project_name}",
        f"source_threshold = {source_threshold}",
        f"source_from_resistance = {str(SOURCE_FROM_RESISTANCE).lower()}",
        f"calc_flow_potential = {str(CALC_FLOW_POTENTIAL).lower()}",
        f"calc_normalized_current = {str(CALC_NORMALIZED_CURRENT).lower()}",
        f"parallelize = {str(PARALLELIZE).lower()}",
    ]
    if MAX_PARALLEL is not None:
        lines.append(f"max_parallel = {int(MAX_PARALLEL)}")
    if WRITE_RAW_CURRMAP:
        lines.append("write_raw_currmap = true")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_omniscape(ini_path: Path) -> None:
    code = f'using Omniscape; run_omniscape(raw"{ini_path}")'
    subprocess.run([JULIA_EXE, "-e", code], check=True)


def find_output(run_dir: Path, pattern: str) -> Path | None:
    hits = sorted(run_dir.rglob(pattern))
    return hits[0] if hits else None


def load_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def shared_log_scale(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.where(a < 0, np.nan, a)
    b = np.where(b < 0, np.nan, b)

    a = np.log1p(a)
    b = np.log1p(b)

    vals = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if vals.size == 0:
        return a, b

    p1, p99 = np.nanpercentile(vals, [1, 99])
    if np.isclose(p1, p99):
        return a, b

    a = np.clip(a, p1, p99)
    b = np.clip(b, p1, p99)
    a = (a - p1) / (p99 - p1)
    b = (b - p1) / (p99 - p1)
    return a, b


def plot_three_panel(a: np.ndarray, b: np.ndarray, title: str, outpath: Path) -> None:
    diff = b - a
    vmax = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 99) if np.isfinite(diff).any() else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    im0 = axes[0].imshow(a, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("run A")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(b, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("run B")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title("B - A")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_run_log(run_dir: Path, params: dict) -> None:
    (run_dir / "run_log.json").write_text(json.dumps(params, indent=2), encoding="utf-8")


def run_case(source_threshold: float) -> Path:
    resistance_path = RESISTANCE_ROOT / f"resistance_{RESISTANCE_NAME}.tif"
    if not resistance_path.exists():
        raise FileNotFoundError(f"Missing resistance file: {resistance_path}")

    source_file = find_source_file(SOURCE_ROOT, CLIMATE_STATE)

    run_name = build_run_name(source_threshold)
    run_dir = OUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ini_path = run_dir / "config.ini"

    params = {
        "source_scenario": SELECTED_SOURCE_SCENARIO,
        "climate_state": CLIMATE_STATE,
        "source_file": str(source_file),
        "resistance_name": RESISTANCE_NAME,
        "resistance_file": str(resistance_path),
        "radius_pixels": RADIUS,
        "radius_meters": RADIUS * 10,
        "block_size": BLOCK_SIZE,
        "source_threshold": source_threshold,
        "project_name": str(run_dir),
        "run_dir": str(run_dir),
        "parallelize": PARALLELIZE,
        "max_parallel": MAX_PARALLEL,
        "calc_flow_potential": CALC_FLOW_POTENTIAL,
        "calc_normalized_current": CALC_NORMALIZED_CURRENT,
        "write_raw_currmap": WRITE_RAW_CURRMAP,
    }
    write_run_log(run_dir, params)

    if SKIP_IF_COMPLETE and find_output(run_dir, "normalized_cum_currmap.tif") is not None:
        print(f"[SKIP] {run_name} (already complete)")
        return run_dir

    write_ini(
        out_path=ini_path,
        resistance=resistance_path,
        source=source_file,
        radius=RADIUS,
        block_size=BLOCK_SIZE,
        project_name=run_dir,
        source_threshold=source_threshold,
    )

    print(f"[RUN] {run_name}")
    run_omniscape(ini_path)
    return run_dir


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    run_a = run_case(SOURCE_THRESHOLD_A)
    run_b = run_case(SOURCE_THRESHOLD_B)

    # find outputs
    a_norm = find_output(run_a, "normalized_cum_currmap.tif")
    b_norm = find_output(run_b, "normalized_cum_currmap.tif")
    a_cum = find_output(run_a, "cum_currmap.tif")
    b_cum = find_output(run_b, "cum_currmap.tif")
    a_flow = find_output(run_a, "flow_potential.tif")
    b_flow = find_output(run_b, "flow_potential.tif")

    if not all([a_norm, b_norm, a_cum, b_cum, a_flow, b_flow]):
        raise FileNotFoundError("One or more Omniscape outputs are missing.")

    figs_dir = OUT_ROOT / "debug_compare"
    figs_dir.mkdir(exist_ok=True)

    # normalized current
    a, b = shared_log_scale(load_raster(a_norm), load_raster(b_norm))
    plot_three_panel(a, b, "normalized current (log scaled)", figs_dir / "normalized_current_compare.png")

    # cumulative current
    a, b = shared_log_scale(load_raster(a_cum), load_raster(b_cum))
    plot_three_panel(a, b, "cumulative current (log scaled)", figs_dir / "cum_current_compare.png")

    # flow potential
    a, b = shared_log_scale(load_raster(a_flow), load_raster(b_flow))
    plot_three_panel(a, b, "flow potential (log scaled)", figs_dir / "flow_potential_compare.png")

    print(f"[OK] comparison figures written to {figs_dir}")


if __name__ == "__main__":
    main()


### only one threshold

#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

# =============================================================================
# MASTER SETTINGS
# =============================================================================

SELECTED_SOURCE_SCENARIO = "ref_p95"
CLIMATE_STATE = "baseline_avg"
RESISTANCE_NAME = "parsimonious_v4"

RADIUS = 100
BLOCK_SIZE = 21

SOURCE_THRESHOLD = 0.8   # p90 cutoff: keep strongest ~10% of sources

CALC_FLOW_POTENTIAL = True
CALC_NORMALIZED_CURRENT = True
SOURCE_FROM_RESISTANCE = False
PARALLELIZE = True
MAX_PARALLEL = None
WRITE_RAW_CURRMAP = False

SKIP_IF_COMPLETE = False

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")
SOURCE_ROOT = BASE / "sources" / SELECTED_SOURCE_SCENARIO
RESISTANCE_ROOT = BASE / "resistance"
OUT_ROOT = BASE / "output" / "debug_single"

JULIA_EXE = r"C:\Program Files\Julia-1.10.11\bin\julia.exe"

# =============================================================================
# HELPERS
# =============================================================================

def find_source_file(source_root: Path, climate_state: str) -> Path:
    source_dir = source_root / climate_state
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    candidates = sorted(source_dir.glob("*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No source tif found in {source_dir}")

    preferred = [p for p in candidates if p.name.startswith(f"omni_source_{climate_state}_")]
    return preferred[0] if preferred else candidates[0]


def build_run_name() -> str:
    thr = f"{SOURCE_THRESHOLD:g}".replace(".", "p")
    return f"{SELECTED_SOURCE_SCENARIO}__{CLIMATE_STATE}__{RESISTANCE_NAME}__r{RADIUS:03d}_b{BLOCK_SIZE:02d}__st{thr}"


def write_ini(
    out_path: Path,
    resistance: Path,
    source: Path,
    radius: int,
    block_size: int,
    project_name: Path,
) -> None:
    lines = [
        f"resistance_file = {resistance}",
        f"source_file = {source}",
        f"radius = {radius}",
        f"block_size = {block_size}",
        f"project_name = {project_name}",
        f"source_threshold = {SOURCE_THRESHOLD}",
        f"source_from_resistance = {str(SOURCE_FROM_RESISTANCE).lower()}",
        f"calc_flow_potential = {str(CALC_FLOW_POTENTIAL).lower()}",
        f"calc_normalized_current = {str(CALC_NORMALIZED_CURRENT).lower()}",
        f"parallelize = {str(PARALLELIZE).lower()}",
    ]
    if MAX_PARALLEL is not None:
        lines.append(f"max_parallel = {int(MAX_PARALLEL)}")
    if WRITE_RAW_CURRMAP:
        lines.append("write_raw_currmap = true")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_omniscape(ini_path: Path) -> None:
    code = f'using Omniscape; run_omniscape(raw"{ini_path}")'
    subprocess.run([JULIA_EXE, "-e", code], check=True)


def find_output(run_dir: Path, pattern: str) -> Path | None:
    hits = sorted(run_dir.rglob(pattern))
    return hits[0] if hits else None


def load_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def shared_percentile_scale(a: np.ndarray, b: np.ndarray, p_low=1, p_high=99) -> tuple[np.ndarray, np.ndarray]:
    vals = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if vals.size == 0:
        return a, b

    lo, hi = np.nanpercentile(vals, [p_low, p_high])
    if np.isclose(lo, hi):
        return a, b

    a = np.clip(a, lo, hi)
    b = np.clip(b, lo, hi)
    a = (a - lo) / (hi - lo)
    b = (b - lo) / (hi - lo)
    return a, b


def shared_log_scale(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.where(a < 0, np.nan, a)
    b = np.where(b < 0, np.nan, b)

    a = np.log1p(a)
    b = np.log1p(b)
    return shared_percentile_scale(a, b, 1, 99)


def plot(arr: np.ndarray, title: str, outpath: Path, cmap: str = "viridis", vmin=None, vmax=None) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def write_run_log(run_dir: Path, params: dict) -> None:
    (run_dir / "run_log.json").write_text(json.dumps(params, indent=2), encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    resistance_path = RESISTANCE_ROOT / f"resistance_{RESISTANCE_NAME}.tif"
    if not resistance_path.exists():
        raise FileNotFoundError(f"Missing resistance file: {resistance_path}")

    source_file = find_source_file(SOURCE_ROOT, CLIMATE_STATE)

    run_name = build_run_name()
    state_dir = OUT_ROOT / run_name
    state_dir.mkdir(parents=True, exist_ok=True)

    project_dir = state_dir / "project"
    project_dir.mkdir(parents=True, exist_ok=True)

    ini_path = state_dir / "config.ini"

    params = {
        "source_scenario": SELECTED_SOURCE_SCENARIO,
        "climate_state": CLIMATE_STATE,
        "source_file": str(source_file),
        "resistance_name": RESISTANCE_NAME,
        "resistance_file": str(resistance_path),
        "radius_pixels": RADIUS,
        "radius_meters": RADIUS * 10,
        "block_size": BLOCK_SIZE,
        "source_threshold": SOURCE_THRESHOLD,
        "project_name": str(project_dir),
        "run_dir": str(state_dir),
        "parallelize": PARALLELIZE,
        "max_parallel": MAX_PARALLEL,
        "calc_flow_potential": CALC_FLOW_POTENTIAL,
        "calc_normalized_current": CALC_NORMALIZED_CURRENT,
        "write_raw_currmap": WRITE_RAW_CURRMAP,
    }
    write_run_log(state_dir, params)

    write_ini(
        out_path=ini_path,
        resistance=resistance_path,
        source=source_file,
        radius=RADIUS,
        block_size=BLOCK_SIZE,
        project_name=project_dir,
    )

    print(f"[RUN] {run_name}")
    print(f"      source = {source_file}")
    print(f"      resistance = {resistance_path}")
    print(f"      project = {project_dir}")
    run_omniscape(ini_path)

    norm_current_path = find_output(project_dir, "normalized_cum_currmap.tif")
    cum_current_path = find_output(project_dir, "cum_currmap.tif")
    flow_path = find_output(project_dir, "flow_potential.tif")

    if norm_current_path is None:
        raise FileNotFoundError("normalized_cum_currmap.tif not found")

    figs_dir = state_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    norm = load_raster(norm_current_path)
    plot(norm, "normalized current", figs_dir / "normalized_current.png")

    if cum_current_path is not None:
        cum = load_raster(cum_current_path)
        plot(cum, "cumulative current", figs_dir / "cum_current.png")

    if flow_path is not None:
        flow = load_raster(flow_path)
        plot(flow, "flow potential", figs_dir / "flow_potential.png")

    src = load_raster(source_file)
    src = np.where(src < 0, np.nan, src)
    src = np.clip(src, np.nanpercentile(src[np.isfinite(src)], 1), np.nanpercentile(src[np.isfinite(src)], 99))
    src = (src - np.nanmin(src)) / (np.nanmax(src) - np.nanmin(src))
    plot(src, "source", figs_dir / "source.png")

    print(f"[OK] figures written to {figs_dir}")
    print("[DONE]")


if __name__ == "__main__":
    main()


### comparison figs
    #!/usr/bin/env python3
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# =============================================================================
# CHANGE THESE
# =============================================================================

RUN_A = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape\output\debug_single\ref_p95__baseline_avg__parsimonious_bldgNA__r100_b21__st0p9"
)

RUN_B = Path(
    r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape\output\debug\ref_p95__baseline_avg__parsimonious_bldgNA__r100_b21_1"
)

# output folder
OUTDIR = RUN_B / "comparison_figs"

# =============================================================================
# HELPERS
# =============================================================================

def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

    return arr


def logscale(arr):
    arr = np.where(arr < 0, np.nan, arr)
    arr = np.log1p(arr)

    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        return arr

    p1, p99 = np.percentile(vals, [1, 99])

    arr = np.clip(arr, p1, p99)

    return (arr - p1) / (p99 - p1)


def diffscale(arr):
    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        return arr, 1

    vmax = np.percentile(np.abs(vals), 99)

    if vmax <= 0:
        vmax = 1

    return arr, vmax


def find_output(run_dir, keyword):
    """
    Omniscape writes files inside nested folders.
    We recursively search for filenames containing keyword.
    """

    matches = list(run_dir.rglob(f"*{keyword}*.tif"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find output containing '{keyword}' in:\n{run_dir}"
        )

    return matches[0]


def plot_single(arr, title, outpath, cmap="viridis", vmin=None, vmax=None):
    plt.figure(figsize=(8, 8))

    im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.title(title)
    plt.axis("off")

    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_side_by_side(a, b, diff, title_a, title_b, title_diff, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # shared scaling for A/B
    vals = np.concatenate([
        a[np.isfinite(a)],
        b[np.isfinite(b)]
    ])

    p1, p99 = np.percentile(vals, [1, 99])

    # A
    im0 = axes[0].imshow(a, cmap="viridis", vmin=p1, vmax=p99)
    axes[0].set_title(title_a)
    axes[0].axis("off")

    # B
    im1 = axes[1].imshow(b, cmap="viridis", vmin=p1, vmax=p99)
    axes[1].set_title(title_b)
    axes[1].axis("off")

    # difference
    diff, vmax = diffscale(diff)

    im2 = axes[2].imshow(
        diff,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax
    )

    axes[2].set_title(title_diff)
    axes[2].axis("off")

    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTDIR.mkdir(exist_ok=True)

    products = {
        "normalized": "normalized_cum_currmap",
        "cumulative": "cum_currmap",
        "flow": "flow_potential",
    }

    for label, keyword in products.items():

        print(f"[INFO] processing: {label}")

        path_a = find_output(RUN_A, keyword)
        path_b = find_output(RUN_B, keyword)

        arr_a = logscale(load_raster(path_a))
        arr_b = logscale(load_raster(path_b))

        diff = arr_b - arr_a

        # single outputs
        plot_single(
            arr_a,
            f"{label} — run A",
            OUTDIR / f"{label}_runA.png"
        )

        plot_single(
            arr_b,
            f"{label} — run B",
            OUTDIR / f"{label}_runB.png"
        )

        # comparison
        plot_side_by_side(
            arr_a,
            arr_b,
            diff,
            "run A",
            "run B",
            "B - A",
            OUTDIR / f"{label}_comparison.png"
        )

    print(f"[OK] wrote figures to:\n{OUTDIR}")


if __name__ == "__main__":
    main()