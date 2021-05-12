
# Random forest tuning

# Load packages
library(tidyverse)
library(tidymodels)

# Set seed
set.seed(57)

# Load required objects
load("data/vehicles_setup.rda")

# Define model
rf_model <- rand_forest(
  mode = "regression", 
  mtry = tune(), 
  min_n = tune()
) %>% 
  set_engine("ranger", importance = "impurity")

# # Check tuning parameters
# parameters(rf_model)

# Set-up tuning grid
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(2, 15)))

# Define grid
rf_grid <- grid_regular(rf_params, levels = 5)

# Random forest workflow
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(vehicles_recipe)

# Tuning/fitting
rf_tune <- rf_workflow %>% 
  tune_grid(
    resamples = vehicles_fold, 
    grid = rf_grid
  )

# Write out results & workflow
save(rf_tune, rf_workflow, file = "data/rf_tune.rda")
