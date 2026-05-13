#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# =============================================================================
# SETTINGS
# =============================================================================

JULIA_EXE = r"C:\Program Files\Julia-1.10.11\bin\julia.exe"

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")

SOURCE_FILE = BASE / "sources" / "source_p90_coolness_stability.tif"
RESISTANCE_FILE = BASE / "resistance" / "resistance_v5.tif"

OUT_ROOT = BASE / "output"
RUN_PARENT = OUT_ROOT / "conditional_multiruns"
CONFIG_DIR = OUT_ROOT / "conditional_multirun_configs"

RUN_PARENT.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    {
        "name": "condition_average",
        "file": BASE / "conditions" / "condition_average" / "condition_average.tif",
    },
    {
        "name": "condition_p90",
        "file": BASE / "conditions" / "condition_p90" / "condition_p90.tif",
    },
]

TOLERANCES = [4.0, 5.0]

RADIUS = 100
BLOCK_SIZE = 7

PARALLELIZE = True
CALC_FLOW_POTENTIAL = True
CALC_NORMALIZED_CURRENT = True
WRITE_RAW_CURRMAP = True

SKIP_IF_COMPLETE = True


# =============================================================================
# HELPERS
# =============================================================================

def tolerance_label(value: float) -> str:
    return str(value).replace(".", "p")


def build_run_name(condition_name: str, tolerance: float) -> str:
    return f"{condition_name}__pm{tolerance_label(tolerance)}deg"


def write_ini(
    ini_path: Path,
    project_name: Path,
    condition_file: Path,
    tolerance: float,
) -> None:
    lines = [
        f"resistance_file = {RESISTANCE_FILE}",
        f"source_file = {SOURCE_FILE}",
        "conditional = true",
        "n_conditions = 1",
        f"condition1_file = {condition_file}",
        "comparison1 = within",
        f"condition1_lower = {-tolerance}",
        f"condition1_upper = {tolerance}",
        f"radius = {RADIUS}",
        f"block_size = {BLOCK_SIZE}",
        f"project_name = {project_name}",
        f"calc_flow_potential = {str(CALC_FLOW_POTENTIAL).lower()}",
        f"calc_normalized_current = {str(CALC_NORMALIZED_CURRENT).lower()}",
        f"write_raw_currmap = {str(WRITE_RAW_CURRMAP).lower()}",
        f"parallelize = {str(PARALLELIZE).lower()}",
    ]

    ini_path.write_text("\n".join(lines), encoding="utf-8")


def run_omniscape(ini_path: Path) -> None:
    code = f'using Omniscape; run_omniscape(raw"{ini_path}")'
    subprocess.run([JULIA_EXE, "-e", code], check=True)


def find_file(root: Path, pattern: str) -> Path | None:
    matches = list(root.rglob(pattern))
    return matches[0] if matches else None


def find_actual_run_dir(project_name: Path, run_name: str) -> Path:
    if project_name.exists() and find_file(project_name, "normalized_cum_currmap.tif"):
        return project_name

    candidates = [
        p for p in RUN_PARENT.rglob("*")
        if p.is_dir()
        and run_name in p.name
        and find_file(p, "normalized_cum_currmap.tif")
    ]

    if candidates:
        return sorted(candidates, key=lambda p: len(str(p)))[0]

    raise FileNotFoundError(f"Could not locate output folder for {run_name}")


def is_complete(project_name: Path, run_name: str) -> bool:
    try:
        run_dir = find_actual_run_dir(project_name, run_name)
        return find_file(run_dir, "normalized_cum_currmap.tif") is not None
    except FileNotFoundError:
        return False


def load_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

    return arr


def log_scale(arr: np.ndarray) -> np.ndarray:
    arr = np.where(arr <= 0, np.nan, arr)
    arr = np.log1p(arr)

    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return arr

    p1, p99 = np.percentile(vals, [1, 99])
    arr = np.clip(arr, p1, p99)

    if np.isclose(p1, p99):
        return np.zeros_like(arr)

    return (arr - p1) / (p99 - p1)


def rescale01(arr: np.ndarray) -> np.ndarray:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return arr

    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    if np.isclose(vmin, vmax):
        return np.zeros_like(arr)

    return (arr - vmin) / (vmax - vmin)


def plot(arr: np.ndarray, title: str, outpath: Path, cmap: str = "viridis") -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(arr, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def make_figures(run_dir: Path, condition_file: Path) -> None:
    fig_dir = run_dir / "figs"
    fig_dir.mkdir(exist_ok=True)

    norm_current_path = find_file(run_dir, "normalized_cum_currmap.tif")
    cum_current_path = find_file(run_dir, "cum_currmap.tif")
    flow_path = find_file(run_dir, "flow_potential.tif")

    if norm_current_path is None:
        raise FileNotFoundError(f"normalized_cum_currmap.tif not found in {run_dir}")

    norm_current = load_raster(norm_current_path)
    plot(
        log_scale(norm_current),
        "normalized cumulative current (log-scaled)",
        fig_dir / "normalized_current.png",
    )

    if cum_current_path is not None:
        cum_current = load_raster(cum_current_path)
        plot(
            log_scale(cum_current),
            "cumulative current (log-scaled)",
            fig_dir / "cumulative_current.png",
        )

    if flow_path is not None:
        flow = load_raster(flow_path)
        plot(
            log_scale(flow),
            "flow potential (log-scaled)",
            fig_dir / "flow_potential.png",
        )

    source = load_raster(SOURCE_FILE)
    plot(
        rescale01(source),
        "source strength",
        fig_dir / "source_strength.png",
    )

    condition = load_raster(condition_file)
    plot(
        rescale01(condition),
        "condition surface",
        fig_dir / "condition_surface.png",
    )


def write_run_log(run_dir: Path, params: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_log.json").write_text(
        json.dumps(params, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE_FILE}")

    if not RESISTANCE_FILE.exists():
        raise FileNotFoundError(f"Missing resistance file: {RESISTANCE_FILE}")

    for condition in CONDITIONS:
        condition_name = condition["name"]
        condition_file = condition["file"]

        if not condition_file.exists():
            raise FileNotFoundError(f"Missing condition file: {condition_file}")

        print(f"\n=== CONDITION: {condition_name} ===")

        for tolerance in TOLERANCES:
            run_name = build_run_name(condition_name, tolerance)
            project_name = RUN_PARENT / run_name
            ini_path = CONFIG_DIR / f"{run_name}.ini"

            print(f"\n[SETUP] {run_name}")
            print(f"        condition = {condition_file}")
            print(f"        tolerance = ±{tolerance} °C")

            if SKIP_IF_COMPLETE and is_complete(project_name, run_name):
                print(f"[SKIP] {run_name} already complete")
                continue

            write_ini(
                ini_path=ini_path,
                project_name=project_name,
                condition_file=condition_file,
                tolerance=tolerance,
            )

            params = {
                "run_name": run_name,
                "source_file": str(SOURCE_FILE),
                "resistance_file": str(RESISTANCE_FILE),
                "condition_name": condition_name,
                "condition_file": str(condition_file),
                "condition_tolerance_degrees": tolerance,
                "condition_lower": -tolerance,
                "condition_upper": tolerance,
                "radius_pixels": RADIUS,
                "radius_meters": RADIUS * 10,
                "block_size": BLOCK_SIZE,
                "parallelize": PARALLELIZE,
                "calc_flow_potential": CALC_FLOW_POTENTIAL,
                "calc_normalized_current": CALC_NORMALIZED_CURRENT,
                "write_raw_currmap": WRITE_RAW_CURRMAP,
                "project_name": str(project_name),
                "config_file": str(ini_path),
            }

            log_dir = RUN_PARENT / f"{run_name}_metadata"
            write_run_log(log_dir, params)

            print(f"[RUN] {run_name}")
            run_omniscape(ini_path)
            print(f"[DONE] {run_name}")

            run_dir = find_actual_run_dir(project_name, run_name)
            make_figures(run_dir, condition_file)

            print(f"[OK] completed: {run_dir}")


if __name__ == "__main__":
    main()