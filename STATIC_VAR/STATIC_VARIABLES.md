# Static variables

This folder contains the R scripts that build the static predictor raster library used by Helmi. Most scripts read source raster and vector data, rasterize vector and point inputs to the 1 m DTM-based base grid, derive per-pixel predictor layers, and aggregate the final results to the 10 m prediction grid used by HELMI. Sources used in STATIC_VAR include:

* ALS-derived point clouds and canopy height data derived from the 2021 City of Helsinki airborne lidar data.
* the 2021 City of Helsinki digital terrain model
* the 2024 Helsinki region land cover data set

The folder is organized by predictor group, with the main scripts grouped as follows:

* `canopy_metrics/local_canopy_metrics.R` for local canopy metrics.
* `canopy_metrics/focal_canopy_metrics.R` for angle-corrected focal canopy metrics.
* `canopy_metrics/pre_tile_chm.R`, `canopy_metrics/chm_postprocessing.R`, and `canopy_metrics/post_tiling_merge.R` for canopy height model preprocessing, hole filling, and tile merging.
* `topography/topo_metrics.R` for topography, water fraction, and water distance metrics.
* `buildings/building_predictors.R` and `buildings/building_height.R` for building metrics and building height.
* `land cover/lulc_metrics.R` for land cover-derived rock fraction, impervious fraction, and related surface masks.
* `general/grid_template.R`, `general/prediction_mask.R`, `general/raster_merge.R`, and `general/stack_predictors.R` for workflow support, template creation, raster merging, and predictor stacking.

## Vegetation

### Local canopy metrics

Local canopy metrics are computed directly from ALS return density above ground on a per-pixel basis. The local script `canopy_metrics/local_canopy_metrics.R` reads classified ALS returns, rasterizes vertical point density and occlusion indicators to the 1 m DTM grid, and derives canopy closure (CC), uncorrected canopy closure (uCC), and effective plant area index (PAIe). These 1 m layers are then aggregated to the 10 m prediction grid.

These metrics are “local” because they do not use neighborhood context, are computed vertically for each pixel, and do not account for hemispherical or angular effects. The workflow in STATIC_VAR first prepares 1 m canopy rasters, then resamples and aggregates them to the 10 m model grid. The local canopy workflow can be run locally in a few hours; HPC variants exist for tiled, parallel batch reruns across the full study area.

### Focal canopy metrics

Here, you will also find deprecated attempts to angle-correct the same metrics. They work in principle, but the resulting gain was disproportionate to the increased processing time. The focal canopy metrics script `canopy_metrics/focal_canopy_metrics.R` applies angle-weighted corrections to gap fraction, PAI, and canopy closure using ALS mean scan angle information. It reads ALS returns and scan angle metadata, applies neighborhood-based smoothing and weighting, and computes corrected canopy metrics on the 1 m base grid before aggregation to 10 m.

Because the focal calculations require neighborhood information, tiles must be padded to avoid edge degradation. These metrics are closer to hemispherical biophysical estimates than the local metrics, but they remain empirically derived approximations. ALS scan angles in Helsinki are typically limited to about 20°, so the focal metrics should be interpreted as truncated geometry-corrected predictors rather than full radiative-transfer outputs.

### Canopy max height

The canopy height model is built from high-resolution (0.5 m) ALS returns - the pre-processing of which is a whole different step. In STATIC_VAR, the canopy height workflow is implemented by `canopy_metrics/pre_tile_chm.R`, `canopy_metrics/chm_postprocessing.R`, and `canopy_metrics/post_tiling_merge.R`. These scripts tile the CHM, fill missing pixels by propagating nearby maximum values, and merge the results into a final raster.

Water pixels are rasterized from the land cover data and explicitly set to 0 m to avoid triangulation artifacts and NA values affecting subsequent calculations.

The final canopy maximum height predictor is derived by aggregating the post-processed CHM to 10 m resolution.

## Topography

Topography predictors are derived from the 2021 Helsinki DTM by the script `topography/topo_metrics.R`. This script rasterizes the DTM to the 1 m base grid, computes elevation, slope, slope aspect encoded as eastness and southness, ruggedness (standard deviation of slope), and a topographic position index using a 40 m moving window. These intermediate 1 m layers are then aggregated consistently to the 10 m prediction footprint.

## Water

Water predictors are generated from land cover water masks and distance-to-water calculations. In STATIC_VAR, these metrics are produced by `topography/topo_metrics.R`, which creates inland water and ocean fraction rasters and computes Euclidean distance rasters to the nearest water edge. The derived water outputs are then used by `general/prediction_mask.R` to exclude water bodies from the model domain.

## Buildings

Building predictors are generated from building footprint and height data. The STATIC_VAR building scripts `buildings/building_predictors.R` and `buildings/building_height.R` rasterize building presence and height rasters, compute distance-to-building metrics, and derive additional building-height indicators, all aligned to the 10 m prediction grid.

## Other land cover

### Rocky outcrops

Rocky outcrops are extracted from the land cover classification and represented as a bare rock fraction raster. The script `land cover/lulc_metrics.R` computes this rock fraction together with impervious surface fractions and related land-cover masks. Helsinki’s thin soil and exposed bedrock are important local modifiers of microclimate, so STATIC_VAR includes explicit rock outcrop masking and fraction calculation.

### Vegetated surfaces

Vegetation structure is primarily captured by canopy metrics, but STATIC_VAR also produces a factorized vegetation mask from land cover data via `land cover/lulc_metrics.R`. This mask encodes broad surface types such as trees, shrubs, grass, low vegetation, and non-vegetated surfaces, providing a simple categorical surface-type predictor for the model.
