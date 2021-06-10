
# Support vector machine with radial basis function tuning

# Load packages
library(tidyverse)
library(tidymodels)
library(tictoc)

# Set seed
set.seed(2021)

# Load required objects
load("tuning/data/vehicle_setup.rda")

# Define Model
svmrb_model <- svm_rbf(
  mode = "regression",
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")

# Set-up tuning Grid
svmrb_param <- parameters(svmrb_model) 

# Define tuning grid
svmrb_grid <- grid_regular(svmrb_param, levels = 5)

# Workflow
svmrb_workflow <- workflow() %>% 
  add_model(svmrb_model) %>% 
  add_recipe(vehicle_recipe)

# Tuning/fitting
tic("SVM Radial Basis")
svmrb_tuned <- svmrb_workflow %>% 
  tune_grid(
    resamples = model_folds, 
    grid = svmrb_grid
  )
toc(log = TRUE)

# Save runtime info
svmrb_runtime <- tic.log(format = TRUE)

# Write out results & workflow
save(svmrb_tuned, svmrb_workflow, svmrb_runtime, file = "tuning/data/svmrb_tuned.rda")
