# HELMI - Helsinki Microclimate Index: A predictive machine-learning model of Summer near-ground temperatures in Helsinki parks and urban forests.

A short summary of the model, performance metrics and limitations. The model's target domain is urban parks and forests. 

## Performance
Performance overview.

## Limitations
Limitations overview.

## Improvements
Proposed changes to improve the accuracy of future versions:
* Change model to XGBoost, probably marginal increases but still.
* Incorporate detailed cloud information (e.g., hourly METEOSAT cloud masks) into the model.
* Add more ERA5-Land fields (radiation flow, wind) as dynamic predictors.  

## Acknowledgements
This publication builds on many excellent open-source software, packages and libraries. Many thanks to the authors and maintainers who make this work possible. ❤️
See [CREDITS.md](CREDITS.md) for details.

For acknowledgements related to the research in which Helmi was published, please check out the corresponding publication.

## A note on HPC
High-performance computing (HPC) was used to (pre-)process airborne laser scanning (ALS) tiles and for model training, testing, and predictions. ALS data sets are often composed of hundreds of tiles, making them suitable for (almost) embarassingly parallel HPC processing. We used CSC's Puhti supercomputer and want to acknowledge the computational resources contributed by CSC here. Unfortunately, Puhti will be decommissioned in Spring 2026. Although this may affect the reproducibility of the provided scripts, it should be (relatively) straightforward to adapt the steps to a different HPC system, as the general SLURM/Lustre-related logic will be similar.

## Technical documentation
More in-depth technical documentation of the repository content can be found in: 