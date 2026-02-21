# Modeling
We are using R's ranger.

## 01_data
In this stage, all earlier work converges into the train data assembly. Here, we convert the sensor data to long format, aggregate to hourly resolution, filter out out-of-soil observations, combine sensor geometries with the raw data, extract the static predictors at the sensor locations and load and join ERA5-Land derived predictors. We also encode day and date into sine/cosine. 

Notably, most of the training data assembly is handled in R, but the ERA5-Land variable extraction is done in Python because the handling of netcdf files in R is (opinion alert) atrocious.

## 02_model

* 02_training.R for tuning the final model. Despite being the first script, this is actually the one that is supposed to run last, after hyperparameter tuning results are retrieved.

* 03_tuning_local.R locally prepares and exports all the deterministic components of the hyperparameter tuning, namely the spatiotemporal cross-validation folds and the hyperparameter tuning grid. These should be set up here and never be changed after the fact.

* 03.2_tuning_local_test.R is a local test for debugging the tuning loop on a very limited sample. 

* 04_tuning_HPC.R is the actual worker script that should be run in parallel on the HPC system. The number of folds parameter is only for debugging and running limited test runs, while tuning the script should run on all available folds.

* 05_tuning_results.R is again a local script, used to aggregate the tuning results into summary metrics and perhaps figures. The resulting table is used to select the best fit for the final model in 02_training.R

* 06_SHAP_values.R calculates SHAPLEY additive values to interpret predictor importance in the final model. This is quite expensive and can be run either locally, or with small adaptions on HPC.

## 03_predictions
As the name implies, this contains script to predict using the final model. There is one script to predict a single time stamp locally, and another one written to run embarassingly parallel on HPC, as well as a batch script for the same purpose.

# predictors
## Notes

* 10 m building fraction is a nice looking raster layer but in terms of modeling its probably noise that can be cut, because there is no variance of it represented in the training data. 50 m building fraction is the one we should keep.

* Lagged ERA5 predictors are nice for forecasting, but if they dont increase the model by a lot, I will cut them. Including lagged predictors provided negligible improvement in predictive performance (<0.03 °C reduction in RMSE), I think its thus best to drop them and retain a more parsimonious model structure for reasons of operational ease.
