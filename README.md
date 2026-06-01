<img src="https://github.com/jon-terschan/scripts/blob/main/figures/github/helmilogo_github.png" width="50%">

# A predictive model of Summer near-ground temperatures in Helsinki urban green spaces🌲☀️
<img src="https://github.com/jon-terschan/helsinki-microclimate-index/blob/main/figures/github/temperature_cycle_github.gif" width="100%">

Helmi is a random forest model that predicts **hourly near-ground air temperatures in Helsinki parks and urban forests** during the **leaf-on period (Summer)** at a **spatial resolution of 10 meters**. Helmi combines field observations from the [Helsinki Microclimate and Phenology Observatory (HELMO-HELPO)](https://www.helsinki.fi/en/researchgroups/tree-d-lab/research/urban-microclimate-phenology-observatories) with [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview) meteorological data, canopy structure derived from the [City of Helsinki's airborne laser scanning data](https://hri.fi/data/en_GB/dataset/helsingin-laserkeilausaineistot), and [land cover data](https://www.hsy.fi/en/environmental-information/open-data/avoin-data---sivut/helsinki-region-land-cover-dataset/). Helmi was co-released with [PUBLICATION REFERENCE] and most of the associated data is available [ZENODO LINK]. Here is what HELMI predictions look like:

This repository contains the code for the (pre-)processing of predictors, the tuning/training, and the code related to the publication. Note that some scripts in this repo are written for high-performance computing (HPC) systems, in particular the Finnish Scientific Computational Center's (CSC) [Puhti supercomputer](https://docs.csc.fi/computing/systems-puhti/) (decommissioned in Spring 2026), which we used to process airborne laser scanning (ALS) data, to tune the model and to generate predictions.

## Performance
On average, HELMI's temperature predictions differed from observed values by about 0.6 °C (MAE: 0.63 +- 0.14 °C). Occasionally, larger errors occured, and when taken into account, the typical overall prediction error to be expected is about 1 °C (RMSE: ~0.97 +- 0.23 °C). No systematic over- or underestimation was observed. These observations were retrieved from spatiotemporal cross-validation (k = 25) in the training and tuning phase. When tested against independent sensor data from Kumpula Botanical Garden, a small botanical garden with a semi-dense canopy layer, performance for the Summer of 2024 was slightly worse (about 1–1.5 °C RMSE):

<img src="https://github.com/jon-terschan/helsinki-microclimate-index/blob/main/figures/github/validation_summary_github.png" width="100%">

HELMI's prediction tends to be slightly too cold at night and too warm during the day. Temperature bias is highest during the morning-midday, presumably because the model overestimates solar heating, which, in reality is attenuated by sun angle and microclimate effects the model does not capture well.

## Limitations
HELMI combines large-scale meteorological data (ERA5-Land) with sensor-level observations made primarily in forest-dominated environments with moderate terrain variation and limited built infrastructure. Overall, its training dataset is weighted toward closed-canopy forest systems, and highly anthropogenic environments are not well-represented. Thus, Helmi is expected to perform best in:

* Dense forest environments dominated by mature vegetation.
* Larger parks with a substantial amount of canopy cover.
* Gentle terrain and mid-range elevations, i.e., between 0-40 meters above sea level.
* low built fraction.

We expect HELMI's prediction certainty to degrade in:

* open and low-canopy landscapes and mixed systems, such as forest edges or landscaped parks and gardens as they are not extensively represented in the sensor data
* whenever temperature conditions may fluctuate rapidly, such as in the morning or in the beginning of the leaf-on period.

Apart from the reported validation, HELMI's performance outside of the well represented feature combinations has not been extensively validated and should be interpreted with caution. Generally, we do not expect HELMI to be accurate in:

* highly urban settings dominated by non-natural impervious surfaces, i.e., in the concrete jungle.
* extreme topographies.
* leaf-off conditions, i.e. seasons outside of Summer, as it was trained from May to September.
* forecasting settings, due to the lack of training data and lagged predictors.

## Planned changes

* Test a change in model to XGBoost.
* Incorporate detailed cloud information (e.g., hourly METEOSAT cloud masks).
* Add variants: If interpolation is the goal and no historic data is needed, it would be smart to try out MEPS as ambient reference
* Add additional ERA5-Land fields (radiation flow, wind) as dynamic predictors.  
* In-depth feature analysis and feature pruning to reduce the operational complexity of model training and predicting.

## Citation

If you use Helmi predictions or implementations, please cite the associated publication and the model:

**Primary reference**  
Terschanski, J. (Year). *Title of the article*. Journal Name. DOI

**Helmi**  
Terschanski, J. (2026). *HELMI — Helsinki Microclimate Index* (Version 0.0.1).  
GitHub repository: https://github.com/jon-terschan/helsinki-microclimate-index  
DOI: ENTER DOI WHEN READY

## BibTex

## Technical documentation

* [Airborne laser scanning derived metrics](/documentation/ALS_PROCESSING.md)
* [Other static predictors](/documentation/STATIC_VARIABLES.md)
* [Dynamic predictors](/documentation/DYNAMIC_VARIABLES.md)
* [Modeling](/documentation/MODELING.md)

---

## Acknowledgements

This publication builds on many excellent open-source software, packages and libraries. Many thanks to the authors and maintainers who make this work possible. ❤️
We also want to acknowledge the computational resources contributed by CSC here. See [CREDITS.md](CREDITS.md) for details.

For acknowledgements related to the research in which Helmi was published and the connectivity analysis, please check out the corresponding publication.