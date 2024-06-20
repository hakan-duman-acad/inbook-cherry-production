# THE SOURCE CODE OF THE ANALYSIS
## Required packages installation if not already installed
packs <- c(
  "tidyverse", "tidymodels", "tsibble", "fable", "feasts", "timetk", "lubridate", "kknn", "ranger", "rpart", "nnet"
)
new_packs <- packs[!(packs %in% installed.packages()[, "Package"])]
if (length(new_packs) > 0) install.packages(new_packs)
library(tidyverse)
library(tidymodels)
library(tsibble)
library(fable)
library(feasts)
library(timetk)
library(lubridate)
tidymodels_prefer()
## The data is loading
cherries <- read_delim(file = "./data/FAOSTAT_data_en_6-5-2024-cherries-production.csv")
## Filtering and Preparing the data
data <- cherries |>
  filter(Area == "Türkiye", Item == "Cherries") |>
  select(Year, Value) |>
  set_names(c("date", "value")) |>
  mutate(date = as_date(paste0(date, "-01-01")))

data1 <- tsibble(data, index = date)
## Ploting time series
data |>
  ggplot(aes(x = date, y = value / 1000)) +
  geom_line() +
  xlab("") +
  ylab("Production Quantity (kt)") +
  ggtitle("")
## KPSS unit-root tests
data1 |>
  features(value, feasts::unitroot_kpss)
data1 |>
  features(log(value), feasts::unitroot_kpss)
data1 |>
  mutate(diff_value = difference(log(value))) |>
  features(diff_value, feasts::unitroot_kpss)
## Added lagged features for 1, 2, and 3 steps back
data <- data |>
  mutate(
    lValue = log(value),
    lagLValue = dplyr::lag(lValue),
    dValue = difference(lValue),
    lag_1_dValue = dplyr::lag(dValue, 1),
    lag_2_dValue = dplyr::lag(dValue, 2),
    lag_3_dValue = dplyr::lag(dValue, 3),
  ) |>
  na.omit()
## Splitting data into training and testing sets
## Performing k-fold cross validation (k = 3, 5 repeats) on training data
set.seed(2024)
splits <- data |>
  time_series_split(assess = 10, cumulative = TRUE)
n_folds <- vfold_cv(training(splits), v = 3, repeats = 5)
## Training DT model
tune_spec <-
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune()
  ) |>
  set_engine("rpart") |>
  set_mode("regression")
grid <- grid_regular(cost_complexity(),
  tree_depth(),
  levels = 5
)
wf <- workflow() |>
  add_model(tune_spec) |>
  add_formula(dValue ~ lag_1_dValue + lag_2_dValue + lag_3_dValue)

res <-
  wf |>
  tune_grid(
    resamples = n_folds,
    grid = grid
  )
best_sub_model <- res |>
  select_best(metric = "rmse")
best_sub_model
final_wf <-
  wf |>
  finalize_workflow(best_sub_model)
final_fit <- final_wf |>
  last_fit(splits)
final_model_wf <- extract_workflow(final_fit)
final_model <- final_model_wf |>
  extract_fit_parsnip()
final_dt <- final_model
## Training RF model
tune_spec <-
  rand_forest(
    mtry = tune(),
    trees = tune(),
    min_n = tune()
  ) |>
  set_engine("ranger") |>
  set_mode("regression")
rf_param <-
  tune_spec |>
  parameters()
rf_param <-
  rf_param |>
  finalize(x = training(splits) |>
    select(lag_1_dValue, lag_2_dValue, lag_3_dValue))
grid <- rf_param |> grid_regular(levels = 5)
wf <- workflow() |>
  add_model(tune_spec) |>
  add_formula(dValue ~ lag_1_dValue + lag_2_dValue + lag_3_dValue)
res <-
  wf |>
  tune_grid(
    resamples = n_folds,
    grid = grid
  )
res <- read_rds("./results_forest.Rds")
wf <- read_rds("./workflow_forest.Rds")
best_sub_model <- res |>
  select_best(metric = "rmse")
best_sub_model
final_wf <-
  wf |>
  finalize_workflow(best_sub_model)
final_fit <- final_wf |>
  last_fit(splits)
final_model_wf <- extract_workflow(final_fit)
final_model <- final_model_wf |>
  extract_fit_parsnip()
final_rf <- final_model
## Training KNN model
tune_spec <-
  nearest_neighbor(
    neighbors = tune(),
    weight_func = tune(),
    dist_power = tune()
  ) |>
  set_engine("kknn") |>
  set_mode("regression")
grid <- grid_regular(neighbors(), weight_func(),
  dist_power(),
  levels = 5
)
wf <- workflow() |>
  add_model(tune_spec) |>
  add_formula(dValue ~ lag_1_dValue + lag_2_dValue + lag_3_dValue)
res <-
  wf |>
  tune_grid(
    resamples = n_folds,
    grid = grid
  )
best_sub_model <- res |>
  select_best(metric = "rmse")
best_sub_model
final_wf <-
  wf |>
  finalize_workflow(best_sub_model)
final_fit <- final_wf |>
  last_fit(splits)
final_model_wf <- extract_workflow(final_fit)
final_model <- final_model_wf |>
  extract_fit_parsnip()
final_knn <- final_model
## Training ANN model
tune_spec <-
  mlp(
    hidden_units = tune(),
    penalty = tune(),
    epochs = tune(),
  ) |>
  set_engine("nnet") |>
  set_mode("regression")
grid <- grid_regular(hidden_units(), penalty(), epochs(),
  levels = 5
)
wf <- workflow() |>
  add_model(tune_spec) |>
  add_formula(dValue ~ lag_1_dValue + lag_2_dValue + lag_3_dValue)
res <-
  wf |>
  tune_grid(
    resamples = n_folds,
    grid = grid
  )
best_sub_model <- res |>
  select_best(metric = "rmse")
best_sub_model
final_wf <-
  wf |>
  finalize_workflow(best_sub_model)
final_fit <- final_wf |>
  last_fit(splits)
final_model_wf <- extract_workflow(final_fit)
final_model <- final_model_wf |>
  extract_fit_parsnip()
final_ann <- final_model
## Training LR model
tune_spec <-
  linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")
wf <- workflow() |>
  add_model(tune_spec) |>
  add_formula(dValue ~ lag_1_dValue + lag_2_dValue + lag_3_dValue)

final_fit <- wf |> last_fit(splits)
final_model_wf <- extract_workflow(final_fit)
final_model <- final_model_wf |>
  extract_fit_parsnip()
final_lr <- final_model
## Evaluating Model performances
new_data <- data |>
  mutate(
    DT = predict(final_dt, data) |> unlist(),
    DT = exp(DT + lagLValue)
  ) |>
  mutate(
    RF = predict(final_rf, data) |> unlist(),
    RF = exp(RF + lagLValue)
  ) |>
  mutate(
    LR = predict(final_lr, data) |> unlist(),
    LR = exp(LR + lagLValue)
  ) |>
  mutate(
    KNN = predict(final_knn, data) |> unlist(),
    KNN = exp(KNN + lagLValue)
  ) |>
  mutate(
    ANN = predict(final_ann, data) |> unlist(),
    ANN = exp(ANN + lagLValue)
  ) |>
  rename(OBS = value) |>
  mutate(type = ifelse(date <= ymd("2012-01-01"), "training", "testing"))
## Ploting model performances
plot_data <- new_data |>
  select(date, type, OBS, DT:ANN) |>
  pivot_longer(OBS:ANN) |>
  na.omit()
plot_data |>
  ggplot(aes(x = date, y = value, color = name, linetype = type)) +
  geom_line(alpha = .7) +
  geom_vline(color = "red", xintercept = as.numeric(ymd("2012-01-01")), linetype = "dashed") +
  scale_linetype_manual(values = c("dashed", "solid")) +
  scale_color_manual(values = c("red", "green", "blue", "orange", "black", "dark green")) +
  geom_line(
    data = plot_data |> filter(name == "OBS"),
    aes(x = date, y = value)
  ) +
  theme(legend.title = element_blank()) +
  xlab("") +
  ylab("")
## A function to collect and summarize all forecasting performance metrics for all models across datasets
metrics <- function() {
  data <- plot_data
  grid <- expand_grid(
    .type = c("training", "testing", "all"),
    .model = c("LR", "DT", "RF", "KNN", "ANN")
  )
  ret_val <- NULL
  for (i in 1:nrow(grid)) {
    n_data <- data
    .type <- grid$.type[i]
    .model <- grid$.model[i]
    if (!.type == "all") {
      n_data <- n_data |>
        filter(grepl(pattern = .type, x = type))
    }
    obs <- n_data |>
      filter(name == "OBS") |>
      select(date, value) |>
      rename(obs = value)
    preds <- n_data |>
      filter(name == .model) |>
      select(date, value) |>
      rename(.pred = value)
    obs_pred_data <- left_join(obs, preds, by = "date") |>
      select(-date)
    .metrics <- obs_pred_data |>
      mutate(
        mn = mean(obs),
        res = (obs - .pred)^2,
        tot = (obs - mn)^2
      ) |>
      summarise(
        RMSE = sqrt(mean((obs - .pred)^2)),
        MAE = mean(abs(obs - .pred)),
        MAPE = 100 * mean(abs(obs - .pred) / abs(obs)),
        R2 = 1 - (sum(res) / sum(tot))
      ) |>
      add_column(model = .model, type = .type)
    ret_val <- rbind(ret_val, .metrics)
  }
  return(ret_val)
}
## Collecting metrics
metrics()
## LR Model:  Checking for Residual Mean Assumption
mean(residuals(final_lr$fit))
## LR Model: Testing for Serial Correlation (Jung-Box test)
Box.test(residuals(final_lr$fit), lag = 10, type = "Ljung-Box")
## LR Model: Normality of Residuals
shapiro.test(residuals(final_lr$fit))
## Function for h-step-ahead forecasting with provided method
foreCast <- function(temp = data, h = 10, model = final_ann) {
  for (i in 1:h) {
    dVal <- predict(model, new_data = tail(temp, 1))$.pred
    lags <- tail(temp, 1)[, 5:7] |> unlist()
    new_row <- c(NA, NA, NA, tail(temp$lValue, 1), dVal, lags)
    new_row <- as_tibble(matrix(new_row, nrow = 1))
    colnames(new_row) <- colnames(temp)
    new_row <- new_row |>
      mutate(
        lValue = lagLValue + dValue,
        value = exp(lValue),
        date = tail(temp$date, 1) + years(1)
      )
    temp <- rbind(temp, new_row)
  }
  return(temp |> select(date, value) |> tail(h))
}
## Combining Forecast Results with Original Observations
obs <- data |>
  select(date, value) |>
  mutate(model = "OBS")
forecast_ann <- foreCast() |> mutate(model = "ANN")
forecast_lr <- foreCast(model = final_lr) |> mutate(model = "LR")
forecast_knn <- foreCast(model = final_knn) |> mutate(model = "KNN")
forecast_rf <- foreCast(model = final_rf) |> mutate(model = "RF")
forecast_dt <- foreCast(model = final_dt) |> mutate(model = "DT")
forecast_data <- rbind(obs, forecast_ann, forecast_lr, forecast_knn, forecast_rf, forecast_dt)
## Ploting time series and the forecasts
forecast_data |>
  ggplot(aes(x = date, y = value, color = model)) +
  geom_line(alpha = .7) +
  scale_color_manual(values = c("red", "green", "blue", "orange", "black", "dark green")) +
  theme(legend.title = element_blank()) +
  xlab("") +
  ylab("")
