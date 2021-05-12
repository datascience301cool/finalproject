
# Knn tuning

# Load packages
library(tidyverse)
library(tidymodels)

# Set seed
set.seed(57)

# Load required objects
load("data/vehicles_setup.rda")

# Define model
knn_model <- nearest_neighbor(
  mode = "regression", 
  neighbors = tune()
) %>% 
  set_engine("kknn")

# # Check tuning parameters
# parameters(knn_model)

# Set-up tuning grid
knn_params <- parameters(knn_model)

# Define grid
knn_grid <- grid_regular(knn_params, levels = 5)

# K-nearest neighbors workflow
knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(vehicles_recipe)

# Tuning/fitting
knn_tune <- knn_workflow %>% 
  tune_grid(
    resamples = vehicles_fold, 
    grid = knn_grid
  )

# Write out results & workflow
save(knn_tune, knn_workflow, file = "data/knn_tune.rda")
