library(tidyverse)
library(tidymodels)
library(tictoc)
library(janitor)
library(splitstackshape)
library(recipes)


load('data/vehicle_setup.rda')



train_set <- training(vehicle_split)
test_set <- testing(vehicle_split)

vehicle_recipe <- recipe(price ~ year + manufacturer + fuel + odometer + title_status + transmission, data = train_set) %>% 
  step_medianimpute(year, odometer) %>% 
  step_modeimpute(manufacturer, fuel, title_status, transmission) %>% 
  step_other(all_nominal(), -transmission, -fuel) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric(), -all_outcomes())



btree <- boost_tree(
  mode = 'regression',
  sample_size = tune()
) %>%
  set_engine('xgboost')

params <- parameters(btree)
btree_grid <- grid_regular(params, levels = 5)

btree_wflow <- workflow() %>%
  add_model(btree) %>%
  add_recipe(vehicle_recipe)

tic("Boosted Tree")
btree_tune <- btree_wflow %>%
  tune_grid(
    resamples = model_folds,
    grid = btree_grid
  )
toc(log = T)

btree_runtime <- tic.log(format = TRUE)
save(btree_tune, btree_wflow, btree_runtime, file = "data/btree_tune.rda")



btree_workflow_tuned <- btree_wflow %>% 
  finalize_workflow(select_best(btree_tune, metrics = class_metrics))


btree_results <- fit(btree_workflow_tuned, train_set)



btree_predict <- predict(btree_results, new_data = test_set) %>% 
  bind_cols(test_set %>% select(price))


save(btree_predict, file = 'data/btree_predictions.rda')

select_best(btree_workflow_tuned, metric = 'rmse')


