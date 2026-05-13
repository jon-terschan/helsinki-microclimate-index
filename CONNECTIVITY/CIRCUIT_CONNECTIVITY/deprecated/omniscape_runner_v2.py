#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

# =============================================================================
# MASTER PARAMETERS
# =============================================================================

SELECTED_SOURCE_SCENARIO = "ref_p95"   # change this only: ref_p75 / ref_p90 / ref_p95
BLOCK_SIZE = 7
RADII = [50, 75, 100]                  # 500 m, 750 m, 1000 m at 10 m pixels
SKIP_COMPLETED = True                  # skip runs that already have outputs
OVERWRITE_INI = True                   # rewrite config.ini and run_log.json each time

CLIMATE_STATES = [
    "baseline_avg",
    "baseline_p90",
    "h2010",
    "h2018",
    "h2021",
]

RESISTANCE_LAYERS = [
    {
        "name": "permissive",
        "path": "resistance_permissive.tif",
    },
    {
        "name": "conservative",
        "path": "resistance_conservative.tif",
    },
]

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")
SOURCE_ROOT = BASE / "sources" / SELECTED_SOURCE_SCENARIO
RESISTANCE_ROOT = BASE / "resistance"
OUT_ROOT = BASE / "output" / SELECTED_SOURCE_SCENARIO

JULIA_EXE = r"C:\Program Files\Julia-1.10.11\bin\julia.exe"

# =============================================================================
# HELPERS
# =============================================================================

def find_source_file(climate_state: str) -> Path:
    """
    Finds the one source file inside:
    sources/<ref_pXX>/<climate_state>/
    """
    source_dir = SOURCE_ROOT / climate_state
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    candidates = sorted(source_dir.glob(f"omni_source_{climate_state}_*.tif"))
    if not candidates:
        candidates = sorted(source_dir.glob("*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No source tif found in {source_dir}")

    return candidates[0]


def write_ini(out_path: Path, resistance: Path, source: Path, radius: int, block_size: int, project_name: Path) -> None:
    content = f"""
resistance_file = {str(resistance)}
source_file = {str(source)}
radius = {radius}
block_size = {block_size}
project_name = {str(project_name)}

source_from_resistance = false
calc_flow_potential = true
calc_normalized_current = true
parallelize = true
""".strip()

    out_path.write_text(content, encoding="utf-8")


def run_omniscape(ini_path: Path) -> None:
    # Use a direct argument list to avoid shell quoting problems.
    code = f'using Omniscape; run_omniscape(raw"{ini_path}")'
    subprocess.run([JULIA_EXE, "-e", code], check=True)


def build_run_name(climate_state: str, resistance_name: str, radius: int, block_size: int) -> str:
    return f"{climate_state}__{resistance_name}__r{radius:03d}_b{block_size:02d}"


def is_completed(run_dir: Path) -> bool:
    return any(run_dir.rglob("normalized_cum_currmap.tif"))


def write_run_log(run_dir: Path, params: dict) -> None:
    (run_dir / "run_log.json").write_text(json.dumps(params, indent=2), encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for climate_state in CLIMATE_STATES:
        source_file = find_source_file(climate_state)

        for resistance in RESISTANCE_LAYERS:
            resistance_path = RESISTANCE_ROOT / resistance["path"]
            if not resistance_path.exists():
                raise FileNotFoundError(f"Missing resistance file: {resistance_path}")

            for radius in RADII:
                run_name = build_run_name(
                    climate_state=climate_state,
                    resistance_name=resistance["name"],
                    radius=radius,
                    block_size=BLOCK_SIZE,
                )

                run_dir = OUT_ROOT / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                ini_path = run_dir / "config.ini"

                params = {
                    "source_scenario": SELECTED_SOURCE_SCENARIO,
                    "climate_state": climate_state,
                    "source_file": str(source_file),
                    "resistance_name": resistance["name"],
                    "resistance_file": str(resistance_path),
                    "radius_pixels": radius,
                    "radius_meters": radius * 10,
                    "block_size": BLOCK_SIZE,
                    "project_name": str(run_dir),
                    "run_dir": str(run_dir),
                }

                write_run_log(run_dir, params)

                if SKIP_COMPLETED and is_completed(run_dir):
                    print(f"[SKIP] {run_name} (already has outputs)")
                    continue

                write_ini(
                    out_path=ini_path,
                    resistance=resistance_path,
                    source=source_file,
                    radius=radius,
                    block_size=BLOCK_SIZE,
                    project_name=run_dir,
                )

                print(f"[RUN] {run_name}")
                run_omniscape(ini_path)

    print("All runs complete.")


if __name__ == "__main__":
    main()


#### FIGURE GENERATOR
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

# =============================================================================
# MASTER PARAMETERS
# =============================================================================

SELECTED_SOURCE_SCENARIO = "ref_p75"   # same value as in the runner
SKIP_EXISTING_FIGS = True

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA\omniscape")
OUT_ROOT = BASE / "output" / SELECTED_SOURCE_SCENARIO

# =============================================================================
# HELPERS
# =============================================================================

def load_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def log_scale(arr: np.ndarray) -> np.ndarray:
    arr = np.where(arr < 0, np.nan, arr)
    arr = np.log1p(arr)

    p1, p99 = np.nanpercentile(arr, [1, 99])
    if np.isclose(p1, p99):
        out = np.zeros_like(arr, dtype=np.float32)
        out[np.isfinite(arr)] = 1.0
        return out

    arr = np.clip(arr, p1, p99)
    return (arr - p1) / (p99 - p1)


def rescale01(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=np.float32)

    if not np.any(valid):
        return out

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if np.isclose(vmin, vmax):
        out[valid] = 1.0
        return out

    out[valid] = (arr[valid] - vmin) / (vmax - vmin)
    return out


def clip_source(arr: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return arr
    p_low, p_high = np.percentile(vals, [low, high])
    return np.clip(arr, p_low, p_high)


def plot(arr: np.ndarray, title: str, outpath: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(arr, cmap="viridis")
    plt.title(title)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def find_first(run_dir: Path, pattern: str) -> Path | None:
    hits = sorted(run_dir.rglob(pattern))
    return hits[0] if hits else None


def read_source_path_from_ini(ini_path: Path) -> Path | None:
    if not ini_path.exists():
        return None
    for line in ini_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("source_file"):
            _, value = line.split("=", 1)
            return Path(value.strip())
    return None


def process_run(run_dir: Path) -> None:
    figs_dir = run_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    if SKIP_EXISTING_FIGS and (figs_dir / "refugia_index.png").exists():
        print(f"[SKIP] {run_dir.name} (figs exist)")
        return

    ini_path = run_dir / "config.ini"
    source_path = read_source_path_from_ini(ini_path)

    norm_current = find_first(run_dir, "normalized_cum_currmap.tif")
    cum_current = find_first(run_dir, "cum_currmap.tif")
    flow = find_first(run_dir, "flow_potential.tif")

    if norm_current is None:
        print(f"[SKIP] {run_dir.name} (missing normalized current)")
        return

    current = load_raster(norm_current)
    current_log = log_scale(current)
    plot(current_log, "normalized current (log)", figs_dir / "current_log.png")

    if cum_current is not None:
        arr = log_scale(load_raster(cum_current))
        plot(arr, "cumulative current (log)", figs_dir / "cum_current_log.png")

    if flow is not None:
        arr = log_scale(load_raster(flow))
        plot(arr, "flow potential (log)", figs_dir / "flow_log.png")

    if source_path is not None and source_path.exists():
        source = load_raster(source_path)
        source_scaled = rescale01(clip_source(source))
        refugia = source_scaled * current_log

        plot(source_scaled, "source (scaled)", figs_dir / "source.png")
        plot(refugia, "refugia index", figs_dir / "refugia_index.png")

    print(f"[OK] {run_dir.name}")


def main() -> None:
    if not OUT_ROOT.exists():
        raise FileNotFoundError(f"Missing output root: {OUT_ROOT}")

    for cfg in sorted(OUT_ROOT.rglob("config.ini")):
        process_run(cfg.parent)

    print("All figure generation complete.")


if __name__ == "__main__":
    main()