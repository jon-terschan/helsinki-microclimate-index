
# Predicting

1. First, we generate deterministic files "needed" for the HPC implementation in 05_deter_filegen.R. We enshrine the feature order in the model in a txt file called feature_order.txt to make sure that the merged predictor stack (static + dynamic vars)
is fully deterministic between runs and is technically a superfluous step. More importantly,
we define on which dates to predict here by exporting a prediction_schedule.rds. Important: The number of rows in this schedule is upper bound of the HPC array job, because we use this file as a lookup and indexing. So, make sure that the batch file (.sh) is based around whatever you define here.

2. Then, we tested the prediction logic locally. Generally, our predictors come from two different sources: static variables come from a predefined raster stack we created in earlier steps. They are trivial to handle, because they do not change over the predicted time period. Dynamic variables like ERA5 are a bit trickier and need much more attention so most of the script logic is concerned with extracting ERA5-Land data from the netcdf and turn into a single point in time raster that can be stacked onto the rest of the predictors. Also, we need to recompute the cosine/sine encoded datetime variables here. Once this is done, we predict and immediately write, hopefully using terras chunking logic and making things a bit more RAM efficient.

3. In the final step, we take the prediction logic from the former file and change a few things to make it work as part of a batch job on Puhti. Most importantly, we have to change the indexing and output logic to make use of the predetermined prediction schedule, as well as all the filepaths.