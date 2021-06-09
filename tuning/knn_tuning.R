# Nearest neighbors tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# load required objects ----
load("data/vehicle_setup.rda")

# Define model ----
knn_model <- nearest_neighbor(
  mode = "regression", 
  neighbors = tune()
) %>% 
  set_engine("kknn")

# Check tuning parameters
parameters(knn_model)

# set-up tuning grid ----
knn_params <- parameters(knn_model)

# define tuning grid
knn_grid <- grid_regular(knn_params, levels = 5)

# workflow ----
knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(vehicle_recipe)

# Tuning/fitting ----
tic("Nearest Neighbors")
knn_tune <- knn_workflow %>% 
  tune_grid(
    resamples = model_folds, 
    grid = knn_grid
  )
toc(log = TRUE)

# save runtime info
knn_runtime <- tic.log(format = TRUE)

# Write out results & workflow
save(knn_tune, knn_workflow, knn_runtime, file = "data/knn_tune.rda")
