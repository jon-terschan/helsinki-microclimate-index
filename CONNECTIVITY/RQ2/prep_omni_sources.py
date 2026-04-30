# omniscape source surface prep
# here we create the surface like this
# first average peak solar hours 12-16 local time
# then take max value within target domain (green areas)
# then substract local temperature mean 
# then we invert the difference (low = bad because temp is close to the max, high good because its far away from max
# then scale to 0-1 range for use as source in omniscape

#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
import matplotlib.pyplot as plt

BASE_DIR = Path(r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\scripts\DATA")
PREDICTIONS_DIR = BASE_DIR / "predictions"
PREDICTORSTACK_DIR = BASE_DIR / "predictorstack"
OUT_DIR = BASE_DIR / "omniscape" / "sources"

NWN_PATH = PREDICTORSTACK_DIR / "NWN_FRAC_10m.tif"
TREE_PATH = PREDICTORSTACK_DIR / "TREE_FRAC_10m.tif"

# Keep only the actual heatwave years.
TARGET_YEARS = {"2010", "2018", "2021"}

# UTC hours corresponding to 12:00-16:00 local time in your setup.
UTC_HOURS = {9, 10, 11, 12, 13}

VALID_MASK_THRESHOLD = 0.0
QC_HEATWAVE = "2018"
WRITE_NODATA_OUTSIDE_DOMAIN = True


@dataclass
class RasterRef:
    profile: dict
    transform: object
    crs: object
    width: int
    height: int
    nodata: float | int | None


def parse_pred_file(path: Path) -> Tuple[str, int, int]:
    m = re.match(r"pred_(\d{8})_(\d{2})(\d{2})\.tif$", path.name)
    if not m:
        raise ValueError(f"Unexpected prediction filename format: {path.name}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def find_prediction_files(base_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for tif in base_dir.rglob("pred_*.tif"):
        date_str, hour, minute = parse_pred_file(tif)
        year = date_str[:4]
        if year not in TARGET_YEARS:
            continue
        if hour in UTC_HOURS:
            groups.setdefault(year, []).append(tif)
    for year in groups:
        groups[year] = sorted(groups[year])
    return groups


def read_raster(path: Path) -> Tuple[np.ndarray, RasterRef]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        ref = RasterRef(src.profile.copy(), src.transform, src.crs, src.width, src.height, nodata)
    return arr, ref


def read_mask(path: Path, ref: RasterRef, name: str) -> np.ndarray:
    with rasterio.open(path) as src:
        if (src.width, src.height) != (ref.width, ref.height) or src.transform != ref.transform or src.crs != ref.crs:
            raise ValueError(f"{name} does not match prediction grid. Reproject/resample first or use aligned rasters.")
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
    return arr


def average_rasters(paths: List[Path]) -> Tuple[np.ndarray, RasterRef]:
    if not paths:
        raise ValueError("No rasters to average.")

    arrays = []
    ref = None
    for p in paths:
        arr, this_ref = read_raster(p)
        if ref is None:
            ref = this_ref
        else:
            if (this_ref.width, this_ref.height) != (ref.width, ref.height) or this_ref.transform != ref.transform or this_ref.crs != ref.crs:
                raise ValueError(f"Raster grid mismatch: {p}")
        arrays.append(arr)

    stack = np.stack(arrays, axis=0)
    valid = np.isfinite(stack)
    count = valid.sum(axis=0).astype(np.float32)
    summed = np.nansum(stack, axis=0)
    mean = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    np.divide(summed, count, out=mean, where=count > 0)
    return mean, ref  # type: ignore[arg-type]


def build_valid_domain(nwn: np.ndarray, tree: np.ndarray) -> np.ndarray:
    return (np.nan_to_num(nwn, nan=0.0) > VALID_MASK_THRESHOLD) | (np.nan_to_num(tree, nan=0.0) > VALID_MASK_THRESHOLD)


def scale_0_1(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    vals = arr[mask & np.isfinite(arr)]
    if vals.size == 0:
        raise ValueError("No valid cells available for scaling.")
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if np.isclose(vmin, vmax):
        out[mask] = 1.0
    else:
        out[mask] = (arr[mask] - vmin) / (vmax - vmin)
    return out


def write_geotiff(path: Path, arr: np.ndarray, ref: RasterRef, nodata: float = -9999.0) -> None:
    profile = ref.profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=2,
        tiled=False,
        nodata=nodata if WRITE_NODATA_OUTSIDE_DOMAIN else None,
    )

    to_write = arr.astype(np.float32)
    if WRITE_NODATA_OUTSIDE_DOMAIN:
        to_write = np.where(np.isnan(to_write), nodata, to_write)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(to_write, 1)


def make_qc_plot(path: Path, source: np.ndarray, heatwave_mean: np.ndarray, domain: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    im0 = axes[0].imshow(heatwave_mean, cmap="inferno")
    axes[0].set_title("Heatwave mean temperature")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(source, cmap="viridis", vmin=0, vmax=1)
    axes[1].contour(domain.astype(np.uint8), levels=[0.5], colors="white", linewidths=0.8)
    axes[1].set_title("Scaled source surface")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = find_prediction_files(PREDICTIONS_DIR)
    if not groups:
        raise RuntimeError(
            f"No prediction files found under {PREDICTIONS_DIR} for TARGET_YEARS={sorted(TARGET_YEARS)} "
            f"and UTC_HOURS={sorted(UTC_HOURS)}."
        )

    first_file = next(iter(next(iter(groups.values()))))
    _, ref = read_raster(first_file)

    nwn = read_mask(NWN_PATH, ref, "NWN_FRAC")
    tree = read_mask(TREE_PATH, ref, "TREE_FRAC")
    valid_domain = build_valid_domain(nwn, tree)

    for year, files in sorted(groups.items()):
        heatwave_mean, ref2 = average_rasters(files)
        if (ref2.width, ref2.height) != (ref.width, ref.height) or ref2.transform != ref.transform or ref2.crs != ref.crs:
            raise ValueError(f"Heatwave {year}: averaged raster grid does not match reference grid.")

        masked = np.where(valid_domain, heatwave_mean, np.nan)
        max_temp = float(np.nanmax(masked))
        if not np.isfinite(max_temp):
            raise ValueError(f"Heatwave {year}: no valid cells inside target domain.")

        source_raw = max_temp - heatwave_mean
        source_scaled = scale_0_1(source_raw, valid_domain)

        if WRITE_NODATA_OUTSIDE_DOMAIN:
            source_scaled = np.where(valid_domain, source_scaled, np.nan)
        else:
            source_scaled = np.where(valid_domain, source_scaled, 0.0)

        out_tif = OUT_DIR / f"omni_source_{year}_9-13_max.tif"
        write_geotiff(out_tif, source_scaled, ref)
        print(f"[OK] {year}: averaged {len(files)} rasters, max within domain = {max_temp:.3f}, wrote {out_tif}")

        if year == QC_HEATWAVE:
            qc_png = OUT_DIR / f"qc_{year}_source_surface.png"
            make_qc_plot(qc_png, source_scaled, masked, valid_domain, title=f"Omniscape source surface QC ({year})")
            print(f"[OK] QC plot written to {qc_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
