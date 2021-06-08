# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(lubridate)
library(kknn)
library(xgboost)
library(numbers)
library(caret)
library(earth)
library(vip)
library(pdp)
library(nnet)
library(ridge)
library(tictoc)
library(kernlab)

# Seed
set.seed(2021)

load(file = "data/vehicles.rda")

load(file = "data/vehicles_strat.rda")

vehicle_split <- vehicles_strat %>%
  initial_split(prop = .8, strata = price)

vehicle_train <- training(vehicle_split)

vehicle_test <- testing(vehicle_split)

#vehicle_recipe <- recipe(price ~ ., data = vehicles_strat) %>%
#  step_center(all_predictors()) %>%
#  step_scale(all_predictors())

# Preston's recipe
vehicles_recipe <- recipe(price ~ year + manufacturer + condition + cylinders + fuel + odometer + title_status + transmission + drive + type + state_region, data = vehicle_train) %>% 
  step_medianimpute(year, odometer) %>% 
  step_modeimpute(manufacturer, condition, cylinders, fuel, title_status, transmission, drive, type) %>% 
  step_nzv(all_predictors()) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_normalize(all_predictors())

#1: ridge model
mlp_model <- mlp(hidden_units = tune(), penalty = tune()) %>% 
  set_engine("nnet", trace = 0) %>% 
  set_mode("regression")

#tuning
mlp_params <- parameters(mlp_model) #%>%
#  update(prod_degree = prod_degree(c(0, 10)),
#    num_terms = num_terms(c(0, 10)))

mlp_grid <- grid_regular(mlp_params, levels = 1)

mlp_workflow <- workflow() %>%
  add_model(mlp_model) %>%
  add_recipe(vehicles_recipe)

# Tuning/fitting ----
tic("NEURAL NETWORK")

model_folds <- vfold_cv(vehicle_train, v = 5, repeats = 3)

mlp_metric <- metric_set(rmse)

class_metrics <- metric_set(rmse, rsq)

#mlp_tuned <- mlp_workflow %>% 
#  tune_grid(model_folds, grid = mlp_grid)

mlp_tuned <- mlp_workflow %>%
  tune_grid(
    resamples = model_folds,
    grid = mlp_grid,
    metrics = class_metrics
  )

#mlp_tuned_final <- mlp_workflow %>% 
#  finalize_workflow(select_best(mlp_tuned, metric = "rmse"))

#toc(log = TRUE)

#mlp_results <- fit(mlp_tuned_final, vehicle_train)

#mlp_predict <- predict(mlp_results, new_data = vehicle_test) %>% 
#  bind_cols(vehicle_test %>% select(price))

#mlp_accuracy <- accuracy(mlp_predict, truth = price, estimate = .pred)

# save runtime info
#mlp_time <- tic.log(format = TRUE)

#write_rds(mlp_accuracy, "mlp_model.rds")
write_rds(mlp_tuned, "data/mlp_tuned.rds")
write_rds(mlp_workflow, "data/mlp_workflow.rds")

