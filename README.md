![A conceptualized logo of the Helsinki Microclimate Index](https://github.com/jon-terschan/scripts/blob/main/figures/helmilogo_github.png)

# HELMI - Helsinki Microclimate Index: A predictive model of Summer near-ground temperatures in Helsinki urban green spaces🌲☀️

Helmi is a random forest model that predicts **hourly near-ground air temperatures in Helsinki parks and urban forests** during the **leaf-on period (Summer)** at a **spatial resolution of 10 meters**. Helmi combines field observations from the [Helsinki Microclimate and Phenology Observatory (HELMO-HELPO)](https://www.helsinki.fi/en/researchgroups/tree-d-lab/research/urban-microclimate-phenology-observatories) with [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview) meteorological data, canopy structure derived from the[City of Helsinki's airborne laser scanning data](https://hri.fi/data/en_GB/dataset/helsingin-laserkeilausaineistot), and [land cover data](https://www.hsy.fi/en/environmental-information/open-data/avoin-data---sivut/helsinki-region-land-cover-dataset/). Helmi was co-released with [PUBLICATION REFERENCE] and most of the associated data is available [ZENODO LINK].

This repository contains the code for the (pre-)processing of predictors and tuning/training of Helmi is available. Some scripts in this repo are written for high-performance computing (HPC) systems, in particular the Finnish Scientific Computational Center's (CSC) [Puhti supercomputer](https://docs.csc.fi/computing/systems-puhti/) (decommissioned in Spring 2026), which we used to process airborne laser scanning (ALS) data, to tune the model and to generate predictions.

## Performance
On average, HELMI's temperature predictions differ from observed values by about 0.6 °C (MAE: 0.63 +- 0.14 °C). Occasionally, larger errors occur, and when these are taken into account, the typical overall prediction error is about 1 °C (RMSE: ~0.97 +- 0.23 °C). No systematic over- or underestimation was observed.

Performance was assessed by tuning across 25 spatiotemporal cross-validation folds. In addition, the production model was externally validated over independent data from Kumpula Botanical garden.

## Limitations
HELMI's training data is made up of sensor-level observations located primarily in forest-dominated environments with moderate terrain variation and limited built infrastructure. Overall, the dataset is weighted toward closed-canopy forest systems, and highly anthropogenic environments are not represented. Thus, Helmi is expected to perform best in:

* Dense forest environments dominated by mature vegetation.
* Gentle terrain and mid-range elevations, i.e., between 0-40 meters above sea level.
* low built fraction.

We expect HELMI's predictions accuracy to degrade in open and low-canopy systems, as well as mixed systems (forest edges) as they are not extensively represented in the sensor data.

HELMI's performance outside of the represented feature combinations has not been validated and should be interpreted with caution. Generally, we do not expect HELMI to be accurated in:

* highly urban settings dominated by non-natural impervious surfaces.
* extreme topographies
* leaf-off conditions, i.e. seasons outside of Summer.
* forecasting.

## Planned changes

* Test a change in model to XGBoost.
* Incorporate detailed cloud information (e.g., hourly METEOSAT cloud masks).
* Add variants: If interpolation is the goal and no historic data is needed, it would be smart to try out MEPS as ambient reference
* Add additional ERA5-Land fields (radiation flow, wind) as dynamic predictors.  
* In-depth feature analysis and feature pruning to reduce the operational complexity of model training and predicting.

## Citation

If you use HELMI in academic work, please cite the associated publication and the model:

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