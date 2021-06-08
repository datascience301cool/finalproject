# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(splitstackshape)
library(tictoc)
library(earth)
library(kernlab)

# Seed
set.seed(2021)

# load file
load(file = "data/vehicles.rda")

load(file = "data/vehicles_strat.rda")

# split data
vehicles_strat <- vehicles %>% 
  stratified("price", 0.05)

save(vehicles_strat, file = "data/vehicles_strat.rda")

vehicle_split <- vehicles_strat %>%
  initial_split(prop = .8, strata = price)

vehicle_train <- training(vehicle_split)

vehicle_test <- testing(vehicle_split)

# recipe
vehicles_recipe <- recipe(price ~ year + manufacturer + fuel + odometer + title_status + transmission, data = vehicles_strat) %>% 
  step_medianimpute(year, odometer) %>% 
  step_modeimpute(manufacturer, fuel, title_status, transmission) %>% 
  step_other(all_nominal(), -transmission, -fuel) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric(), -all_outcomes())

#tuning
model_folds <- vfold_cv(vehicle_train, v = 5, repeats = 3)


# Define Model
svmrb_model <- svm_rbf(
  mode = "regression",
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")

# Tuning Grid
svmrb_param <- parameters(svmrb_model) 

# Define Tuning Grid
svmrb_grid <- grid_regular(svmrb_param, levels = 5)

# Workflow
svmrb_workflow <- workflow() %>% 
  add_model(svmrb_model) %>% 
  add_recipe(vehicles_recipe)

# Tuning/fitting ----

tic("SVM Radial Basis")
# Pace tuning code in hear
svmrb_tuned <- svmrb_workflow %>% 
  tune_grid(model_folds, grid = svmrb_grid)

toc(log = TRUE)

# save runtime info
svmrb_runtime <- tic.log(format = TRUE)

# Write out results & workflow
write_rds(svmrb_runtime, "svmrbtime.rds")
write_rds(svmrb_tuned, "svmrb_results.rds")

select_best(svmrb_tuned, metric = "accuracy")
show_best(svmrb_tuned, metric = "accuracy")




