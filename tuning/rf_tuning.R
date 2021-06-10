
# Random forest tuning

# Load packages
library(tidyverse)
library(tidymodels)
library(tictoc)

# Set seed
set.seed(57)

# Load required objects
load("tuning/data/vehicle_setup.rda")

# Define model
rf_model <- rand_forest(
  mode = "regression", 
  mtry = tune(), 
  min_n = tune()
) %>% 
  set_engine("ranger")

# Set-up tuning grid
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(2, 17)))

# Define grid
rf_grid <- grid_regular(rf_params, levels = 5)

# Workflow
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(vehicle_recipe)

# Tuning/fitting
tic("Random Forest")
rf_tune <- rf_workflow %>% 
  tune_grid(
    resamples = model_folds, 
    grid = rf_grid
  )
toc(log = TRUE)

# Save runtime info
rf_runtime <- tic.log(format = TRUE)

# Write out results & workflow
save(rf_tune, rf_workflow, rf_runtime, file = "tuning/data/rf_tune.rda")
