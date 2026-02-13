# OTHER STATIC VARIABLES
We created many scripts generating additional static predictors for the model related to topography, water presence and built-up matter. Some examples include:

* Elevation, slope, slopeaspect (Eastness/Southness), and ruggedness
* Water presence, distance to oceans/inland water bodies
* Building presence, height, and distance from buildings
* Presence of other impervious surfaces (concrete roads and sealed surfaces)
* Rocky outcrop presence

Data sets used for this were the outcomes of our own ALS processing (LINK), the 2021 [City of Helsinki digital elevation model](https://hri.fi/data/en_GB/dataset/helsingin-korkeusmalli), and the 2024 [Helsinki region land cover data set](https://www.hsy.fi/en/environmental-information/open-data/avoin-data---sivut/helsinki-region-land-cover-dataset/).

These are simple repetitive data preparation and rasterization operations, that do not warrant long documentation. We usually rasterized vector data to the same 1 m grid template based on the DTM and then calculated whichever metric to the same 10 m prediction grid.

# VEGETATION
## CANOPY MAX HEIGHT
During (LINK) the ALS processing, we created a canopy height model of Helsinki with a very high spatial resolution (0.5 m). We post-processed the CHM by filling missing pixels with the maximum nearest neighbor value in two passes, to remove some of the calculation artifacts. The resulting CHM still contains some holes, but not inside the relevant domain (vegetated surface). We also used the rasterized water mask from the land cover data set to set all water pixels to a height of 0 m, as triangulation artifacts and NAs would otherwise mess with the SVF estimation (LINK).

From the post-processed canopy height model, we derived canopy maximum height on a resolution of 10 meters.

# TOPOGRAPHY

We derived elevation, slope, slopeaspect (encoded as Eastness and Southness), as well as ruggedness (standard deviation of slope) and topographic position index with a 40 m window size from the 2021 [City of Helsinki digital elevation model](https://hri.fi/data/en_GB/dataset/helsingin-korkeusmalli). Their DTM is also derived from airborne laser scanning, presumably the same data, but we assumed it to be of a higher quality than the one we processed by ourselves.

# WATER
# BUILDINGS 
# OTHER LAND COVER
## ROCKY OUTCROPS
Helsinki's soil layer is notoriously thin in many places and rocky outcrops are a characteristic sight for the city. Urban parks and forests are no different in this regard. We assumed that bare rock formations have a unique microclimate and a considerable impact on the localized climatic conditions and thus included bare rock fraction in the model.

## VEGETATED SURFACES
Presumably, most information relevant to vegetation is already encoded in the canopy structure, but we created a very simple factorized vegetation mask from land cover data encoding rudimentary vegetation differences (trees, low vegetation).