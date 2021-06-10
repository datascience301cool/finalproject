
# Boosted tree tuning

# Load packages
library(tidyverse)
library(tidymodels)
library(tictoc)

# Set seed
set.seed(57)

# Load required objects
load('tuning/data/vehicle_setup.rda')

# Define model
btree <- boost_tree(
  mode = 'regression',
  sample_size = tune()
) %>%
  set_engine('xgboost')

# Set-up tuning grid
btree_params <- parameters(btree)

# Define tuning grid
btree_grid <- grid_regular(btree_params, levels = 5)

# Workflow
btree_wflow <- workflow() %>%
  add_model(btree) %>%
  add_recipe(vehicle_recipe)

# Tuning/fitting
tic("Boosted Tree")
btree_tune <- btree_wflow %>%
  tune_grid(
    resamples = model_folds,
    grid = btree_grid
  )
toc(log = T)

# Save runtime info
btree_runtime <- tic.log(format = TRUE)

# Write out results & workflow
save(btree_tune, btree_wflow, btree_runtime, file = "training/data/btree_tune.rda")
