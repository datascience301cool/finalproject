# Elastic net tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# load required objects ----
load("data/vehicle_setup.rda")

# Define model ----
en_model <- linear_reg(
  mode = "regression", 
  mixture = tune(), 
  penalty = tune()
) %>% 
  set_engine("glmnet")

# Check tuning parameters
parameters(en_model)

# set-up tuning grid ----
en_params <- parameters(en_model)

# define tuning grid
en_grid <- grid_regular(en_params, levels = 5)

# workflow ----
en_workflow <- workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(vehicle_recipe)

# Tuning/fitting ----
tic("Elastic Net")
en_tune <- en_workflow %>% 
  tune_grid(
    resamples = model_folds, 
    grid = en_grid
  )
toc(log = TRUE)

# save runtime info
en_runtime <- tic.log(format = TRUE)

# Write out results & workflow
save(en_tune, en_workflow, en_runtime, file = "data/en_tune.rda")
