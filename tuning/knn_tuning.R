
# K-nearest neighbors tuning

# Load packages
library(tidyverse)
library(tidymodels)
library(tictoc)

# Set seed
set.seed(57)

# Load required objects
load("tuning/data/vehicle_setup.rda")

# Define model
knn_model <- nearest_neighbor(
  mode = "regression", 
  neighbors = tune()
) %>% 
  set_engine("kknn")

# Set-up tuning grid
knn_params <- parameters(knn_model)

# Define tuning grid
knn_grid <- grid_regular(knn_params, levels = 5)

# Workflow
knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(vehicle_recipe)

# Tuning/fitting
tic("Nearest Neighbors")
knn_tune <- knn_workflow %>% 
  tune_grid(
    resamples = model_folds, 
    grid = knn_grid
  )
toc(log = TRUE)

# Save runtime info
knn_runtime <- tic.log(format = TRUE)

# Write out results & workflow
save(knn_tune, knn_workflow, knn_runtime, file = "tuning/data/knn_tune.rda")
