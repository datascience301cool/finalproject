#Notes:

#Stratify the data (because it'll take too long to run if not)
#5%, 20k rows

#NOT RANDOM FOREST OR BOOSTED TREE (takes too long)
#Neural net - Edwin
#Elastic net - Lauren
#Random forest - Preston
#Boosted tree - Josh

#Predicting the price of a car based off of variables

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(splitstackshape)
library(janitor)
# Seed
set.seed(2021)

load(file = "data/vehicles.rda")
# vehicles <- vehicles %>% 
#   clean_names() %>%
#   select(-c(id, url, region_url, vin, image_url, description)) %>%
#   mutate(state_region = 
#            case_when(
#              state %in% c("wv", "va", "ky", "nc", "sc", "tn", "ar", "la", "al", "ms", "ga", "fl") ~ "southeast", 
#              state %in% c("me", "vt", "nh", "ma", "ct", "ri", "ny", "nj", "pa", "de", "md", "dc") ~ "northeast", 
#              state %in% c("nd", "sd", "mn", "wi", "ia", "ne", "ks", "mo", "il", "in", "mi", "oh") ~ "midwest", 
#              state %in% c("ok", "tx", "nm", "az") ~ "southwest", 
#              state %in% c("ak", "hi", "wa", "or", "id", "mt", "wy", "co", "ut", "nv", "ca") ~ "west"
#            )
#   ) %>%
#   mutate(condition = factor(condition)) %>% 
#   mutate(cylinders = factor(cylinders)) %>% 
#   mutate(fuel = factor(fuel)) %>% 
#   mutate(title_status = factor(title_status)) %>% 
#   mutate(transmission = factor(transmission)) %>% 
#   mutate(drive = factor(drive)) %>% 
#   mutate(size = factor(size)) %>% 
#   mutate(type = factor(type)) %>% 
#   mutate(paint_color = factor(paint_color)) %>% 
#   mutate(state_region = factor(state_region)) %>% 
#   mutate(manufacturer = factor(manufacturer)) %>%
#   filter(price > 1000 & price < 100000) %>% 
#   mutate(price = log10(price)) %>%
#   filter(odometer < 300000) %>% 
#   filter(year > 1993)
# 
# save(vehicles, file = "data/vehicles.rda")

#vehicles_strat <- vehicles %>% 
#  stratified("price", 0.05)

load(file = "data/vehicles_strat.rda")

#save(vehicles_strat, file = "data/vehicles_strat.rda")

set.seed(57)

vehicle_split <- vehicles_strat %>%
  initial_split(prop = .8, strata = price)

vehicle_train <- training(vehicle_split)

vehicle_test <- testing(vehicle_split)

vehicle_recipe <- recipe(price ~ ., data = vehicles_strat) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

# Data inspection for recipe
skimr::skim_without_charts(vehicle_train)

vehicle_train %>% 
  select(price, year, odometer, lat, long) %>% 
  cor(use = "complete.obs")

vehicle_train %>% 
  ggplot(aes(manufacturer, price)) + 
  geom_point()

vehicle_train %>% 
  ggplot(aes(fuel, price)) + 
  geom_point()

vehicle_train %>% 
  ggplot(aes(title_status, price)) + 
  geom_point()

vehicle_train %>% 
  ggplot(aes(transmission, price)) + 
  geom_point()

vehicle_train %>% 
  ggplot(aes(type, price)) + 
  geom_point()

vehicle_train %>% 
  ggplot(aes(state_region, price)) + 
  geom_point()

# Preston's recipe
vehicle_recipe <- recipe(price ~ year + manufacturer + fuel + odometer + title_status + transmission, data = vehicle_train) %>% 
  step_impute_median(year, odometer) %>% 
  step_impute_mode(manufacturer, fuel, title_status, transmission) %>% 
  step_other(all_nominal(), -transmission, -fuel) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric(), -all_outcomes())

vehicle_recipe %>% prep(vehicle_train) %>% bake(new_data = NULL) %>% view()

# load required objects ----
#load("model_info/wildfires_setup.rda")

#tuning
model_folds <- vfold_cv(vehicle_train, v = 5, repeats = 3)

# save tuning objects
save(model_folds, vehicle_recipe, vehicle_split, file = "data/vehicle_setup.rda")

# Define model ----
knn_model <- nearest_neighbor(
  mode = "classification",
  neighbors = tune()
) %>%
  set_engine("kknn")

# # check tuning parameters
# parameters(knn_model)

# set-up tuning grid ----
knn_params <- parameters(knn_model) %>%
  update(neighbors = neighbors(range = c(1,40)))

# define grid
knn_grid <- grid_regular(knn_params, levels = 15)

# workflow ----
knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(vehicle_recipe)

class_metrics <- metric_set(precision, recall, sensitivity, specificity, f_meas, accuracy, roc_auc)

# Tuning/fitting ----
knn_tune <- knn_workflow %>%
  tune_grid(
    resamples = model_folds,
    grid = knn_grid,
    metrics = class_metrics
  )
# Write out results & workflow
save(knn_tune, knn_workflow, file = "model_info/knn_tune.rda")

knn_tune








### ELASTIC NET

#New recipe

# Define Model
en_model <- linear_reg(
  mode = "regression",
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
  add_recipe(vehicle_recipe)

# Tuning/fitting ----

en_tuned <- en_workflow %>% 
  tune_grid(
    resamples = model_folds, 
    grid = en_grid
  )

# save files
write_rds(en_tuned, "en_result.rds")

save(en_tuned, en_workflow, file = "model_info/en_tuned.rds")

# results
en_workflow_tuned <- en_workflow %>% 
  finalize_workflow(select_best(en_tuned, metrics = class_metrics))

en_results <- fit(en_workflow_tuned, vehicle_train)


#NEURAL NET

mlp_model <- mlp(hidden_units = tune(), penalty = tune()) %>% 
  set_engine("nnet", trace = 0) %>% 
  set_mode("regression")



