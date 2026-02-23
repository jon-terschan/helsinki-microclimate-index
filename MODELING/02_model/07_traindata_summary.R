# the purpose of this script is to summarize the feature
# space defined by the train data. from that info, we can already infer 
# limitations and target domains of the final model

library(dplyr)
library(sf)

# load train data
train_joined <- readRDS("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/01_traindataprep/06_train_data.rds")
glimpse(train_joined) # check its the right data

# summary statistic
sensor_static <- train_joined %>%
  st_drop_geometry() %>%
  group_by(sensor_id) %>%
  summarise(
    n_rows = n(),
    n_temp_obs = sum(!is.na(temp)),
    CC = first(CC),
    UCC = first(uCC),
    PAI = first(PAI),
    elev_10 = first(elev_10),
    slope = first(slope),
    bld_fr_10 = first(bldg_fr_10),
    bld_fr_50 = first(bldg_fr_50),
    bld_dist = first(bldg_dis),
    chm_max_10m = first(chm_max_10m),
    nwn_fr_10 = first(nwn_fr_10),
    tree_fr_10 = first(tree_fr_10),
    rock_fr_10 = first(rock_fr_10),
    oce_dis = first(oce_dis),
    ruggedness = first(ruggedness),
    .groups = "drop"
  )

write.csv(sensor_static, "//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/01_traindataprep/train_data_summary.csv", row.names = FALSE)
train_sum <- read.csv("//ad.helsinki.fi/home/t/terschan/Desktop/paper1/scripts/DATA/modeling/01_traindataprep/train_data_summary.csv")

# quantify feature space
train_sum %>%
  select(-sensor_id) %>%   # remove IDs or other non-predictors
  summarise(
    across(
      everything(),
      list(
        min = ~min(.x, na.rm = TRUE),
        q05 = ~quantile(.x, 0.05, na.rm = TRUE),
        median = ~median(.x, na.rm = TRUE),
        mean = ~mean(.x, na.rm = TRUE),
        q95 = ~quantile(.x, 0.95, na.rm = TRUE),
        max = ~max(.x, na.rm = TRUE),
        sd = ~sd(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    )
  )
