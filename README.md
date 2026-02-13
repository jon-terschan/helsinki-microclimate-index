# HELMI - Helsinki Microclimate Index: A predictive model of Summer near-ground temperatures in Helsinki urban green spaces🌲☀️

Helmi is a random forest model that can predict **hourly near-ground air temperatures in Helsinki parks and urban forests** during the **leaf-on period (Summer)** at a **spatial resolution of 10 meters**. Helmi converges in-situ temperature observations from the [Helsinki Microclimate and Phenology Observatory (HELMO-HELPO)](https://www.helsinki.fi/en/researchgroups/tree-d-lab/research/urban-microclimate-phenology-observatories) with open data canopy structure, sky occlusion, meteorological reference data, and land cover (see Predictors). Helmi was co-released with [PUBLICATION REFERENCE] and most of the associated data is available [ZENODO LINK].

The full codebase for the (pre-)processing of predictors and tuning/training of Helmi is available. Some scripts are designed to run on high-performance computing (HPC) systems, in particular the Finnish Scientific Computational Center's (CSC) [Puhti supercomputer](https://docs.csc.fi/computing/systems-puhti/) (decommissioned in Spring 2026), which we used to process airborne laser scanning (ALS) data, to tune the model and to generate predictions.

## Performance

We tuned hyperparameters by minimizing mean errors (RMSE, MSE, and bias) across 25 spatiotemporal cross-validation folds. Helmi's performance with the best performing hyperparameter set:

* Mean RMSE
* Mean MSE
* R2 (treat with care)

Helmi's production version was also externally validated by predicting over an external data set from Kumpula Botanical Garden.

## Limitations

Helmi cannot predict temperatures

* in non-park and forests urban environments, i.e., in and around "concrete jungle", as it was trained on temperature observations in vegetated urban areas.

* in Seasons other than Summer and leaf-off conditions, as it was trained on observations from Helsinki's leaf-on period (May to September).

* in the future. Although it can be used as a baseline to study microclimatic response under hypothetical changes in the ambient climate (for example, in resarch), it is not, strictly speaking, a forecasting model.

We assume Helmi's predictive performance to degrade

* in urban green areas with a high amount of urban-natural mixed matter, such as gardens, sports fields, and so on. Although Helmi was trained with some data from gardens, it was not specifically designed to perform well in these environments.

## Planned improvements

* Test a change in model to XGBoost.
* Incorporate detailed cloud information (e.g., hourly METEOSAT cloud masks).
* Add variants: If interpolation is the goal and no historic data is needed, it would be smart to try out MEPS as ambient reference
* Add additional ERA5-Land fields (radiation flow, wind) as dynamic predictors.  

## Acknowledgements

This publication builds on many excellent open-source software, packages and libraries. Many thanks to the authors and maintainers who make this work possible. ❤️
We also want to acknowledge the computational resources contributed by CSC here. See [CREDITS.md](CREDITS.md) for details.

For acknowledgements related to the research in which Helmi was published and the connectivity analysis, please check out the corresponding publication.

## Citation

If you use HELMI in academic work, please cite the associated publication and the model:

**Primary reference**  
Terschanski, J. (Year). *Title of the article*. Journal Name. DOI

**Helmi**  
Terschanski, J. (2026). *HELMI — Helsinki Microclimate Index* (Version 0.0.1).  
GitHub repository: https://github.com/jon-terschan/helsinki-microclimate-index  
DOI: ENTER DOI WHEN READY

## BibTex

---

## Technical documentation

* [Airborne laser scanning derived metrics](/documentation/ALS_PROCESSING.md)
* [Other static predictors](/documentation/STATIC_VARIABLES.md)
* [Dynamic predictors](/documentation/DYNAMIC_VARIABLES.md)
* [Modeling](/documentation/MODELING.md)
