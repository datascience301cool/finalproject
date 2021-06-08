# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(splitstackshape)

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

# Preston's recipe
vehicles_recipe <- recipe(price ~ year + manufacturer + condition + cylinders + fuel + odometer + title_status + transmission + drive + type + state_region, data = vehicle_train) %>% 
  step_medianimpute(year, odometer) %>% 
  step_modeimpute(manufacturer, condition, cylinders, fuel, title_status, transmission, drive, type) %>% 
  step_nzv(all_predictors()) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_normalize(all_predictors())

#tuning
model_folds <- vfold_cv(vehicle_train, v = 5, repeats = 3)

### ELASTIC NET

# Define Model
en_model <- logistic_reg(
  mode = "classification",
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

# Tuning Grid
en_params <- parameters(en_model)

# Define Tuning Grid
en_grid <- grid_regular(en_params, levels = 5)

# Workflow
en_workflow <- workflow() %>%
  add_model(en_model) %>%
  add_recipe(vehicles_recipe)

# Tuning/fitting ----

en_tuned <- en_workflow %>% 
  tune_grid(model_folds, grid = en_grid)

# save files
write_rds(en_tuned, "en_result.rds")

save(en_tuned, en_workflow, file = "model_info/en_tuned.rds")

# results
en_workflow_tuned <- en_workflow %>% 
  finalize_workflow(select_best(en_tuned, metrics = class_metrics))

en_results <- fit(en_workflow_tuned, vehicle_train)



