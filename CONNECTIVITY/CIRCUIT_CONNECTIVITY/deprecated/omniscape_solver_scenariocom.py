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
BASELINE_STATE = "baseline_avg"
HEATWAVE_STATE = "h2018"

RESISTANCE_NAME = "parsimonious"

RADIUS = 100
BLOCK_SIZE = 21

SOURCE_THRESHOLD = 0.0  # same for both states

CALC_FLOW_POTENTIAL = True
CALC_NORMALIZED_CURRENT = True
SOURCE_FROM_RESISTANCE = False
PARALLELIZE = True
MAX_PARALLEL = None
WRITE_RAW_CURRMAP = False

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")
SOURCE_ROOT = BASE / "sources" / SELECTED_SOURCE_SCENARIO
RESISTANCE_ROOT = BASE / "resistance"
OUT_ROOT = BASE / "output" / "debug_comparisons"

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


def unique_path(parent: Path, stem: str) -> Path:
    candidate = parent / stem
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate_i = parent / f"{stem}_{i}"
        if not candidate_i.exists():
            return candidate_i
        i += 1


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


def plot_pair_with_diff(
    a: np.ndarray,
    b: np.ndarray,
    title_a: str,
    title_b: str,
    title_diff: str,
    outpath: Path,
    cmap: str = "viridis",
    diff_cmap: str = "RdBu_r",
) -> None:
    diff = b - a
    vals = diff[np.isfinite(diff)]
    vmax = np.nanpercentile(np.abs(vals), 99) if vals.size else 1.0
    if np.isclose(vmax, 0):
        vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    im0 = axes[0].imshow(a, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title(title_a)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(b, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(title_b)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap=diff_cmap, vmin=-vmax, vmax=vmax)
    axes[2].set_title(title_diff)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_run_log(run_dir: Path, params: dict) -> None:
    (run_dir / "run_log.json").write_text(json.dumps(params, indent=2), encoding="utf-8")


def run_case(state_name: str, project_stem: str) -> dict:
    resistance_path = RESISTANCE_ROOT / f"resistance_{RESISTANCE_NAME}.tif"
    if not resistance_path.exists():
        raise FileNotFoundError(f"Missing resistance file: {resistance_path}")

    source_file = find_source_file(SOURCE_ROOT, state_name)

    state_dir = OUT_ROOT / project_stem / state_name
    state_dir.mkdir(parents=True, exist_ok=True)

    project_dir = unique_path(state_dir, "project")
    ini_path = state_dir / "config.ini"

    params = {
        "source_scenario": SELECTED_SOURCE_SCENARIO,
        "state": state_name,
        "source_file": str(source_file),
        "resistance_name": RESISTANCE_NAME,
        "resistance_file": str(resistance_path),
        "radius_pixels": RADIUS,
        "radius_meters": RADIUS * 10,
        "block_size": BLOCK_SIZE,
        "source_threshold": SOURCE_THRESHOLD,
        "project_name": str(project_dir),
        "run_dir": str(state_dir),
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

    print(f"[RUN] {state_name}")
    print(f"      source = {source_file}")
    print(f"      resistance = {resistance_path}")
    print(f"      project = {project_dir}")
    run_omniscape(ini_path)

    return {
        "state": state_name,
        "state_dir": state_dir,
        "project_dir": project_dir,
        "config": ini_path,
        "source_file": source_file,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    project_stem = f"{SELECTED_SOURCE_SCENARIO}__{BASELINE_STATE}_vs_{HEATWAVE_STATE}__{RESISTANCE_NAME}__r{RADIUS:03d}_b{BLOCK_SIZE:02d}"
    comparison_root = OUT_ROOT / project_stem
    comparison_root.mkdir(parents=True, exist_ok=True)

    baseline = run_case(BASELINE_STATE, project_stem)
    heatwave = run_case(HEATWAVE_STATE, project_stem)

    a_norm = find_output(baseline["project_dir"], "normalized_cum_currmap.tif")
    b_norm = find_output(heatwave["project_dir"], "normalized_cum_currmap.tif")
    a_cum = find_output(baseline["project_dir"], "cum_currmap.tif")
    b_cum = find_output(heatwave["project_dir"], "cum_currmap.tif")
    a_flow = find_output(baseline["project_dir"], "flow_potential.tif")
    b_flow = find_output(heatwave["project_dir"], "flow_potential.tif")

    if not all([a_norm, b_norm, a_cum, b_cum, a_flow, b_flow]):
        raise FileNotFoundError("One or more Omniscape outputs are missing.")

    figs_dir = comparison_root / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    src_a = load_raster(baseline["source_file"])
    src_b = load_raster(heatwave["source_file"])
    src_a, src_b = shared_percentile_scale(src_a, src_b, 1, 99)
    plot_pair_with_diff(
        src_a,
        src_b,
        f"{BASELINE_STATE} source",
        f"{HEATWAVE_STATE} source",
        "source B - A",
        figs_dir / "source_comparison.png",
    )

    a, b = shared_log_scale(load_raster(a_norm), load_raster(b_norm))
    plot_pair_with_diff(
        a,
        b,
        f"{BASELINE_STATE} normalized current",
        f"{HEATWAVE_STATE} normalized current",
        "normalized current B - A",
        figs_dir / "normalized_current_comparison.png",
    )

    a, b = shared_log_scale(load_raster(a_cum), load_raster(b_cum))
    plot_pair_with_diff(
        a,
        b,
        f"{BASELINE_STATE} cumulative current",
        f"{HEATWAVE_STATE} cumulative current",
        "cumulative current B - A",
        figs_dir / "cum_current_comparison.png",
    )

    a, b = shared_log_scale(load_raster(a_flow), load_raster(b_flow))
    plot_pair_with_diff(
        a,
        b,
        f"{BASELINE_STATE} flow potential",
        f"{HEATWAVE_STATE} flow potential",
        "flow potential B - A",
        figs_dir / "flow_potential_comparison.png",
    )

    print(f"[OK] outputs in {comparison_root}")
    print(f"[OK] figures in {figs_dir}")


if __name__ == "__main__":
    main()