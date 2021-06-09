# Single layer neural network tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# load required objects ----
load("data/vehicle_setup.rda")

# Define model ----
slnn_model <- mlp(
  mode = "regression", 
  hidden_units = tune(), 
  penalty = tune()
) %>% 
  set_engine("nnet")

# Check tuning parameters
parameters(slnn_model)

# set-up tuning grid ----
slnn_params <- parameters(slnn_model)

# define tuning grid
slnn_grid <- grid_regular(slnn_params, levels = 5)

# workflow ----
slnn_workflow <- workflow() %>% 
  add_model(slnn_model) %>% 
  add_recipe(vehicle_recipe)

# Tuning/fitting ----
tic("Single Layer Neural Network")
slnn_tune <- slnn_workflow %>% 
  tune_grid(
    resamples = model_folds, 
    grid = slnn_grid
  )
toc(log = TRUE)

# save runtime info
slnn_runtime <- tic.log(format = TRUE)

# Write out results & workflow
save(slnn_tune, slnn_workflow, slnn_runtime, file = "data/slnn_tune.rda")
