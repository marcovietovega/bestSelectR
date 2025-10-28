# Helper function to create test data
create_test_data <- function(n = 50, p = 4, seed = 123) {
  set.seed(seed)
  x_matrix <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(x_matrix) <- paste0("X", 1:p)

  if (p >= 3) {
    linear_combo <- x_matrix[, 1] + 0.5 * x_matrix[, 2] - 0.3 * x_matrix[, 3]
  } else if (p == 2) {
    linear_combo <- x_matrix[, 1] + 0.5 * x_matrix[, 2]
  } else {
    linear_combo <- x_matrix[, 1]
  }

  y <- rbinom(n, 1, plogis(linear_combo))

  list(X = x_matrix, y = y)
}

create_deterministic_data <- function() {
  x_matrix <- matrix(
    c(
      1,
      2,
      5,
      2,
      4,
      3,
      3,
      1,
      2,
      4,
      3,
      12,
      5,
      5,
      15,
      6,
      1,
      20,
      7,
      2,
      1,
      8,
      3,
      25
    ),
    nrow = 8,
    ncol = 3,
    byrow = TRUE
  )

  colnames(x_matrix) <- c("X1", "X2", "X3")
  y <- ifelse(x_matrix[, 3] > 7, 1, 0)

  list(X = x_matrix, y = y)
}

test_that("bestSubset returns correct object structure", {
  data <- create_test_data()

  result <- bestSubset(
    data$X,
    data$y,
    max_variables = 3,
    top_n = 3,
    metric = "auc"
  )

  expect_s3_class(result, "bestSubset")
  expect_equal(result$call_info$n_observations, nrow(data$X))
  expect_equal(result$call_info$n_predictors, ncol(data$X))
})

test_that("bestSubset works with minimal parameters", {
  data <- create_test_data(n = 20, p = 3)

  result <- bestSubset(data$X, data$y)
  expect_lte(nrow(result$models), 5)
})

test_that("max_variables parameter works correctly", {
  data <- create_test_data(p = 5)

  result <- bestSubset(data$X, data$y, max_variables = 2, top_n = 3)

  max_vars_in_models <- max(result$models$n_variables)
  expect_lte(max_vars_in_models, 2)
})

test_that("top_n parameter controls number of returned models", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y, top_n = 3)

  expect_lte(nrow(result$models), 3)
})

test_that("metric parameter affects model ranking", {
  data <- create_test_data(n = 40, p = 4) # Larger dataset for clearer differences

  result_auc <- bestSubset(data$X, data$y, top_n = 5, metric = "auc")
  result_acc <- bestSubset(data$X, data$y, top_n = 5, metric = "accuracy")

  expect_true(all(diff(result_auc$models$auc) <= 0))
  expect_true(all(diff(result_acc$models$accuracy) <= 0))
})

test_that("function accepts data.frame inputs", {
  data <- create_test_data()
  x_df <- as.data.frame(data$X)

  result <- bestSubset(x_df, data$y, top_n = 2)
  # Best model should have valid accuracy/AUC, others may be NA (optimization)
  expect_true(result$best_model$accuracy >= 0 & result$best_model$accuracy <= 1)
  expect_true(result$best_model$auc >= 0 & result$best_model$auc <= 1)
  # Check that non-NA values are valid
  expect_true(all(
    result$models$accuracy[!is.na(result$models$accuracy)] >= 0 &
      result$models$accuracy[!is.na(result$models$accuracy)] <= 1
  ))
  expect_true(all(
    result$models$auc[!is.na(result$models$auc)] >= 0 &
      result$models$auc[!is.na(result$models$auc)] <= 1
  ))
})

test_that("function handles edge case: single variable", {
  data <- create_test_data(p = 1)

  expect_warning(
    result <- bestSubset(data$X, data$y),
    "top_n.*exceeds available models."
  )
  expect_lte(nrow(result$models), 1)
  expect_true(all(result$models$n_variables <= 1))
})

test_that("best_model contains expected information", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y, top_n = 1)

  best_model <- result$best_model
  expect_true(all(is.finite(best_model$coefficients)))
  expect_gte(best_model$n_variables, 1)
  expect_lte(best_model$n_variables, ncol(data$X))
})

test_that("models data frame has correct structure", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y, top_n = 3)

  models_df <- result$models
  # Check non-NA values are valid (optimization may leave some as NA)
  expect_true(all(
    models_df$auc[!is.na(models_df$auc)] >= 0 &
      models_df$auc[!is.na(models_df$auc)] <= 1
  ))
  expect_true(all(
    models_df$accuracy[!is.na(models_df$accuracy)] >= 0 &
      models_df$accuracy[!is.na(models_df$accuracy)] <= 1
  ))
  expect_true(all(models_df$deviance >= 0))
})

test_that("cross-validation functionality works", {
  data <- create_test_data()

  result_cv <- bestSubset(
    data$X,
    data$y,
    cross_validation = TRUE,
    cv_folds = 3,
    top_n = 2
  )

  expect_true(result_cv$call_info$use_cv)
  # Check non-NA values are valid (optimization may leave some as NA)
  expect_true(all(
    result_cv$models$accuracy[!is.na(result_cv$models$accuracy)] >= 0 &
      result_cv$models$accuracy[!is.na(result_cv$models$accuracy)] <= 1
  ))
  expect_gte(nrow(result_cv$models), 1)
})

test_that("cv_folds and cv_repeats reject NaN and Inf", {
  data <- create_test_data()

  # cv_folds with Inf
  expect_error(
    bestSubset(data$X, data$y, cross_validation = TRUE, cv_folds = Inf),
    "cv_folds must be a finite number"
  )

  # cv_folds with NaN
  expect_error(
    bestSubset(data$X, data$y, cross_validation = TRUE, cv_folds = NaN),
    "cv_folds must be a finite number"
  )

  # cv_repeats with Inf
  expect_error(
    bestSubset(
      data$X,
      data$y,
      cross_validation = TRUE,
      cv_folds = 5,
      cv_repeats = Inf
    ),
    "cv_repeats must be a finite number"
  )
})

test_that("cv_seed validates numeric input", {
  data <- create_test_data()

  # Non-numeric string
  expect_error(
    bestSubset(
      data$X,
      data$y,
      cross_validation = TRUE,
      cv_folds = 3,
      cv_seed = "abc"
    ),
    "cv_seed must be a single numeric value or NULL"
  )

  # NaN
  expect_error(
    bestSubset(
      data$X,
      data$y,
      cross_validation = TRUE,
      cv_folds = 3,
      cv_seed = NaN
    ),
    "cv_seed must be a finite number or NULL"
  )
})

test_that("cv_folds > 20 triggers warning", {
  data <- create_test_data()

  expect_warning(
    bestSubset(
      data$X,
      data$y,
      cross_validation = TRUE,
      cv_folds = 25,
      cv_seed = 123
    ),
    "cv_folds = 25 may result in very slow cross-validation"
  )
})

test_that("algorithm identifies strong relationships", {
  data <- create_deterministic_data()

  expect_warning(
    result <- bestSubset(data$X, data$y, top_n = 3),
    "Perfect separation detected for variable\\(s\\): X3"
  )

  expect_gte(max(result$models$auc), 0.9)
  expect_true(3 %in% result$best_model$variables)
})

test_that("invalid X input throws appropriate errors", {
  data <- create_test_data()

  expect_error(bestSubset("not_a_matrix", data$y))
  expect_error(bestSubset(letters[1:10], data$y))

  x_wrong_dim <- matrix(1:10, 5, 2)
  expect_error(bestSubset(x_wrong_dim, data$y))
})

test_that("predictor matrix rejects Inf and NaN values", {
  data <- create_test_data()

  # Inf values
  X_inf <- data$X
  X_inf[1, 2] <- Inf
  expect_error(
    bestSubset(X_inf, data$y),
    "X contains infinite \\(Inf/-Inf\\) or NaN values"
  )

  # -Inf values
  X_neg_inf <- data$X
  X_neg_inf[3, 1] <- -Inf
  expect_error(
    bestSubset(X_neg_inf, data$y),
    "X contains infinite \\(Inf/-Inf\\) or NaN values"
  )

  # NaN values
  X_nan <- data$X
  X_nan[2, 3] <- NaN
  expect_error(
    bestSubset(X_nan, data$y),
    "X contains infinite \\(Inf/-Inf\\) or NaN values"
  )
})

test_that("categorical data throws helpful error message", {
  df <- data.frame(
    x1 = rnorm(20),
    x2 = rnorm(20),
    category1 = factor(c(rep('A', 10), rep('B', 10))),
    category2 = factor(sample(letters[1:3], 20, replace = TRUE))
  )
  y <- rbinom(20, 1, 0.5)

  expect_error(
    bestSubset(df, y),
    "Categorical variables detected: category1, category2"
  )

  expect_error(
    bestSubset(df, y),
    "bestSelectR requires numeric data"
  )

  X_processed <- model.matrix(~ . - 1, data = df)
  expect_no_error(bestSubset(X_processed, y, top_n = 2))
})

test_that("invalid y input throws appropriate errors", {
  data <- create_test_data()

  y_continuous <- rnorm(nrow(data$X))
  expect_error(bestSubset(data$X, y_continuous))

  y_wrong_values <- c(0, 1, 2, 1, 0)
  x_small <- matrix(rnorm(20), 5, 4)
  expect_error(bestSubset(x_small, y_wrong_values))

  y_wrong_length <- c(0, 1)
  expect_error(bestSubset(data$X, y_wrong_length))
})

test_that("invalid parameter values throw errors", {
  data <- create_test_data()

  expect_error(bestSubset(data$X, data$y, max_variables = -1))
  expect_error(bestSubset(data$X, data$y, top_n = -1))
  expect_error(bestSubset(
    data$X,
    data$y,
    cross_validation = TRUE,
    cv_folds = -1
  ))

  expect_error(bestSubset(
    data$X,
    data$y,
    cross_validation = TRUE,
    cv_folds = nrow(data$X) + 1
  ))
  expect_error(bestSubset(data$X, data$y, metric = "invalid_metric"))

  expect_error(bestSubset(data$X, data$y, max_variables = 0))
  expect_error(bestSubset(data$X, data$y, top_n = 0))
})

test_that("insufficient data throws errors", {
  x_tiny <- matrix(rnorm(4), 1, 4)
  y_tiny <- c(1)
  expect_error(bestSubset(x_tiny, y_tiny))

  expect_error(
    bestSubset(matrix(numeric(0), 0, 0), numeric(0)),
    "Need at least 2 observations"
  )
})

test_that("print method works correctly", {
  data <- create_test_data()
  result <- bestSubset(data$X, data$y, top_n = 1)

  expect_no_error(print(result))
})

test_that("summary method works correctly", {
  data <- create_test_data()
  result <- bestSubset(data$X, data$y, top_n = 1)

  expect_no_error(summary(result))
})

test_that("coef method works correctly", {
  data <- create_test_data()
  result <- bestSubset(data$X, data$y, top_n = 1)

  coeffs <- coef(result)
  expect_true(length(coeffs) >= 1)
  expect_true(!is.null(names(coeffs)))
})

test_that("predict method works correctly", {
  data <- create_test_data()
  result <- bestSubset(data$X, data$y, top_n = 1)

  new_data <- matrix(rnorm(20), nrow = 5, ncol = 4)
  colnames(new_data) <- colnames(data$X)

  pred_prob <- predict(result, new_data, type = "prob")
  expect_equal(length(pred_prob), nrow(new_data))
  expect_true(all(pred_prob >= 0 & pred_prob <= 1))

  pred_class <- predict(result, new_data, type = "class")
  expect_equal(length(pred_class), nrow(new_data))
  expect_true(all(pred_class %in% c(0, 1)))

  new_df <- as.data.frame(matrix(rnorm(20), nrow = 5, ncol = 4))
  colnames(new_df) <- colnames(data$X)

  pred_df <- predict(result, new_df, type = "prob")
  expect_equal(length(pred_df), nrow(new_df))
})

test_that("predict method throws errors for invalid inputs", {
  data <- create_test_data()
  result <- bestSubset(data$X, data$y, top_n = 1)

  wrong_data <- matrix(rnorm(10), nrow = 5, ncol = 2)
  expect_error(predict(result, wrong_data))

  new_data <- matrix(rnorm(20), nrow = 5, ncol = 4)
  expect_error(predict(result, new_data, type = "invalid"))
})

test_that("algorithm correctly identifies variables with deterministic relationships", {
  data <- create_deterministic_data()

  expect_warning(
    result <- bestSubset(data$X, data$y, metric = "accuracy", top_n = 1),
    "Perfect separation detected for variable\\(s\\): X3"
  )

  best_vars <- result$best_model$variables
  expect_true(3 %in% best_vars)
  expect_gte(result$best_model$accuracy, 0.8)
  expect_warning(
    result_auc <- bestSubset(data$X, data$y, metric = "auc", top_n = 3),
    "Perfect separation detected for variable\\(s\\): X3"
  )

  expect_true(3 %in% result_auc$best_model$variables)
})

test_that("mathematical correctness and coefficient validation", {
  data <- create_test_data()
  result <- bestSubset(data$X, data$y, top_n = 1)

  best_model <- result$best_model
  expect_true(all(is.finite(best_model$coefficients)))
  expect_true(all(abs(best_model$coefficients) < 1000))

  best_vars <- best_model$variables
  x_subset <- cbind(1, data$X[, best_vars, drop = FALSE])
  coeffs <- best_model$coefficients

  linear_pred <- x_subset %*% coeffs
  predicted_probs <- 1 / (1 + exp(-linear_pred))
  predicted_classes <- as.integer(predicted_probs > 0.5)

  manual_accuracy <- mean(predicted_classes == data$y)
  expect_equal(manual_accuracy, best_model$accuracy, tolerance = 0.001)
})

test_that("cross-validation results are reproducible with cv_seed", {
  data <- create_test_data()

  cv_result1 <- bestSubset(
    data$X,
    data$y,
    cross_validation = TRUE,
    cv_folds = 3,
    cv_seed = 42,
    top_n = 1
  )

  cv_result2 <- bestSubset(
    data$X,
    data$y,
    cross_validation = TRUE,
    cv_folds = 3,
    cv_seed = 42,
    top_n = 1
  )

  expect_equal(cv_result1$models$auc, cv_result2$models$auc)
  expect_equal(
    cv_result1$models$accuracy,
    cv_result2$models$accuracy
  )
})

test_that("na.fail properly rejects missing data", {
  data <- create_test_data()

  X_missing <- data$X
  X_missing[c(1, 5, 10), 2] <- NA
  expect_error(
    bestSubset(X_missing, data$y, na.action = na.fail),
    "Data contains missing values.*Use na.action = na.omit"
  )

  y_missing <- data$y
  y_missing[3] <- NA

  expect_error(
    bestSubset(data$X, y_missing, na.action = na.fail),
    "Data contains missing values"
  )
})

test_that("invalid na.action throws error", {
  data <- create_test_data()

  # String instead of function
  expect_error(
    bestSubset(data$X, data$y, na.action = "ignore"),
    "na.action must be a function"
  )

  # Invalid function
  expect_error(
    bestSubset(data$X, data$y, na.action = mean),
    "na.action must be na.fail, na.omit, or na.exclude"
  )
})

test_that("na.omit removes missing data cases", {
  data <- create_test_data(n = 30)

  X_missing <- data$X
  X_missing[c(1, 5, 10), 2] <- NA
  X_missing[c(2, 8), 3] <- NA

  # Suppress expected missing data warning
  suppressWarnings({
    result <- bestSubset(X_missing, data$y, na.action = na.omit, top_n = 3)
  })
  expect_equal(result$call_info$n_observations, 25)
  expect_equal(result$call_info$n_observations_original, 30)
  expect_equal(result$call_info$n_removed, 5)

  expect_equal(result$na_info$na_action_used, "na.omit")
  expect_equal(result$na_info$original_n, 30)
  expect_equal(result$na_info$effective_n, 25)
  expect_equal(result$na_info$n_removed, 5)
})

test_that("na.exclude removes missing data but preserves structure", {
  data <- create_test_data(n = 20)

  x_missing <- data$X
  missing_rows <- c(2, 7, 15)
  x_missing[missing_rows, 1] <- NA

  suppressWarnings({
    result <- bestSubset(x_missing, data$y, na.action = na.exclude, top_n = 1)
  })

  expect_equal(result$call_info$n_observations, 17)
  expect_equal(result$na_info$na_action_used, "na.exclude")
  expect_equal(result$na_info$n_removed, 3)

  expect_length(result$na_info$na_map, 20)
  expect_equal(which(is.na(result$na_info$na_map)), missing_rows)

  new_data <- matrix(rnorm(20), nrow = 5, ncol = 4)
  colnames(new_data) <- colnames(data$X)
  new_data[c(1, 3), 2] <- NA

  pred <- predict(result, new_data, type = "prob")
  expect_length(pred, 5)
  expect_true(is.na(pred[1]))
  expect_true(is.na(pred[3]))
  expect_true(!is.na(pred[2]))
})

test_that("prediction handles missing data consistently with training", {
  data <- create_test_data(n = 25)

  X_train_missing <- data$X
  X_train_missing[c(1, 5), 2] <- NA

  # Suppress expected missing data warning
  suppressWarnings({
    result_omit <- bestSubset(
      X_train_missing,
      data$y,
      na.action = na.omit,
      top_n = 2
    )
  })

  new_data <- matrix(rnorm(20), nrow = 5, ncol = 4)
  colnames(new_data) <- colnames(data$X)
  new_data[c(1, 3), 2] <- NA
  expect_warning(
    pred <- predict(result_omit, new_data, type = "prob"),
    "newdata contains missing values.*complete cases"
  )

  expect_length(pred, 3)
  expect_true(all(pred >= 0 & pred <= 1))
  # Suppress expected missing data warning
  suppressWarnings({
    result_exclude <- bestSubset(
      X_train_missing,
      data$y,
      na.action = na.exclude,
      top_n = 2
    )
  })

  pred_exclude <- predict(result_exclude, new_data, type = "prob")

  expect_length(pred_exclude, 5)
  expect_true(is.na(pred_exclude[1]))
  expect_true(is.na(pred_exclude[3]))
  expect_true(!is.na(pred_exclude[2]))
})

test_that("na.fail model behavior with prediction missing values", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y, na.action = na.fail, top_n = 2)

  new_data <- matrix(rnorm(20), nrow = 5, ncol = 4)
  colnames(new_data) <- colnames(data$X)
  new_data[2, 3] <- NA

  pred <- predict(result, new_data, type = "prob")

  expect_length(pred, 5)
  expect_true(is.na(pred[2]))
  expect_true(!is.na(pred[1]))
})

test_that("missing data edge cases are handled correctly", {
  data <- create_test_data(n = 15)

  X_all_missing <- data$X
  X_all_missing[] <- NA

  expect_error(
    bestSubset(X_all_missing, data$y, na.action = na.omit),
    "No complete cases available after removing missing values"
  )

  X_mostly_missing <- data$X
  X_mostly_missing[1:13, ] <- NA
  y_mostly_missing <- data$y
  y_mostly_missing[1:13] <- NA
  y_mostly_missing[14:15] <- c(1, 1)

  expect_error(
    suppressWarnings(bestSubset(
      X_mostly_missing,
      y_mostly_missing,
      na.action = na.omit
    )),
    "Response variable must have exactly 2 distinct values"
  )
})

test_that("cross-validation works with missing data", {
  data <- create_test_data(n = 30)

  X_missing <- data$X
  X_missing[c(1, 5, 10), 2] <- NA
  # Suppress expected missing data warning
  suppressWarnings({
    result <- bestSubset(
      X_missing,
      data$y,
      na.action = na.omit,
      cross_validation = TRUE,
      cv_folds = 3,
      top_n = 2
    )
  })

  expect_true(result$call_info$use_cv)
  expect_equal(result$na_info$na_action_used, "na.omit")
  expect_lt(result$call_info$n_observations, 30)
})

test_that("na_info object contains complete information", {
  data <- create_test_data(n = 20)

  x_missing <- data$X
  missing_rows <- c(2, 7, 15)
  x_missing[missing_rows, 1] <- NA

  suppressWarnings({
    result <- bestSubset(x_missing, data$y, na.action = na.omit, top_n = 2)
  })

  na_info <- result$na_info

  expected_fields <- c(
    "na_action_used",
    "original_n",
    "effective_n",
    "n_removed",
    "na_map"
  )
  expect_true(all(expected_fields %in% names(na_info)))

  expect_equal(na_info$na_action_used, "na.omit")
  expect_equal(na_info$original_n, 20)
  expect_equal(na_info$effective_n, 17)
  expect_equal(na_info$n_removed, 3)
})

test_that("handles boundary case: all observations same class", {
  x_boundary <- matrix(rnorm(20), nrow = 5, ncol = 4)
  y_all_ones <- rep(1, 5)

  expect_error(
    {
      result <- bestSubset(x_boundary, y_all_ones, top_n = 2)
    },
    "Response variable must have exactly 2 distinct values"
  )
})

# Deviance metric tests
test_that("deviance metric works correctly without CV", {
  data <- create_test_data(n = 40, p = 3)

  result <- bestSubset(data$X, data$y, metric = "deviance", top_n = 5)

  # Check class and structure
  expect_s3_class(result, "bestSubset")

  # Deviance should be positive
  expect_true(all(result$models$deviance > 0))

  # Models should be sorted by deviance (ascending - lower is better)
  expect_true(all(diff(result$models$deviance) >= 0))

  # Best model should have lowest deviance
  expect_equal(result$best_model$deviance, min(result$models$deviance))

  # Metric in call_info should be "deviance"
  expect_equal(result$call_info$metric, "deviance")
})

test_that("deviance metric sorting is ascending (lower is better)", {
  data <- create_test_data(n = 40, p = 4)

  result_dev <- bestSubset(data$X, data$y, metric = "deviance", top_n = 5)
  result_auc <- bestSubset(data$X, data$y, metric = "auc", top_n = 5)

  # Deviance should be ascending (lower first)
  deviances <- result_dev$models$deviance
  expect_true(all(diff(deviances) >= 0))

  # AUC should be descending (higher first)
  aucs <- result_auc$models$auc
  expect_true(all(diff(aucs) <= 0))
})

test_that("deviance matches glm calculation", {
  data <- create_test_data(n = 30, p = 3)

  # Fit with bestSubset (all variables)
  # With 3 variables, there are 2^3-1 = 7 possible models
  bs_result <- bestSubset(
    data$X,
    data$y,
    max_variables = 3,
    metric = "deviance",
    top_n = 5
  )

  # Find full model (all 3 variables)
  full_model <- bs_result$models[bs_result$models$n_variables == 3, ]

  # Should have at least one model with all 3 variables
  expect_true(nrow(full_model) > 0)

  # Fit with glm
  glm_model <- glm(data$y ~ data$X, family = binomial)

  # Compare deviances (should match within numerical tolerance)
  expect_equal(full_model$deviance[1], glm_model$deviance, tolerance = 1e-5)
})

test_that("deviance with cross-validation works correctly", {
  # Use default n=50 for CV to ensure sufficient data per fold
  data <- create_test_data()

  result_cv <- bestSubset(
    data$X,
    data$y,
    metric = "deviance",
    cross_validation = TRUE,
    cv_folds = 3,
    cv_seed = 123,
    top_n = 2
  )

  # Check CV was used
  expect_true(result_cv$call_info$use_cv)
  expect_equal(result_cv$call_info$cv_folds, 3)

  # Models should be sorted by deviance
  expect_true(all(diff(result_cv$models$deviance) >= 0))

  # Deviance values should be positive
  expect_true(all(result_cv$models$deviance > 0))
})

test_that("deviance CV results are reproducible with seed", {
  # Use default n=50 for CV to ensure sufficient data per fold
  data <- create_test_data()

  result1 <- bestSubset(
    data$X,
    data$y,
    metric = "deviance",
    cross_validation = TRUE,
    cv_folds = 3,
    cv_seed = 42,
    top_n = 1
  )

  result2 <- bestSubset(
    data$X,
    data$y,
    metric = "deviance",
    cross_validation = TRUE,
    cv_folds = 3,
    cv_seed = 42,
    top_n = 1
  )

  # Results should be identical with same seed
  expect_equal(result1$models$deviance, result2$models$deviance)
  expect_equal(result1$best_model$deviance, result2$best_model$deviance)
})

# ========================
# AIC Metric Tests
# ========================

test_that("AIC metric works correctly", {
  data <- create_test_data(n = 40, p = 4)

  result <- bestSubset(
    data$X,
    data$y,
    max_variables = 3,
    top_n = 5,
    metric = "aic"
  )

  # Check result structure
  expect_s3_class(result, "bestSubset")
  expect_true("aic" %in% names(result$models))
  expect_true("aic" %in% names(result$best_model))

  # AIC values should be positive
  expect_true(all(result$models$aic > 0))
  expect_true(result$best_model$aic > 0)

  # Models should be sorted by AIC (ascending - lower is better)
  expect_true(all(diff(result$models$aic) >= 0))

  # First model should have the lowest AIC
  expect_equal(result$models$aic[1], min(result$models$aic))
})

test_that("BIC metric works correctly", {
  data <- create_test_data(n = 40, p = 4)

  result <- bestSubset(
    data$X,
    data$y,
    max_variables = 3,
    top_n = 5,
    metric = "bic"
  )

  # Check result structure
  expect_s3_class(result, "bestSubset")
  expect_true("bic" %in% names(result$models))
  expect_true("bic" %in% names(result$best_model))

  # BIC values should be positive
  expect_true(all(result$models$bic > 0))
  expect_true(result$best_model$bic > 0)

  # Models should be sorted by BIC (ascending - lower is better)
  expect_true(all(diff(result$models$bic) >= 0))

  # First model should have the lowest BIC
  expect_equal(result$models$bic[1], min(result$models$bic))
})

test_that("AIC and BIC relationship holds", {
  data <- create_test_data(n = 50, p = 4)

  result <- bestSubset(data$X, data$y, max_variables = 3, top_n = 5)

  # For n > 7, BIC should penalize more than AIC (BIC > AIC for same model)
  # Since n = 50, log(n) ≈ 3.91 > 2
  # Best model has all metrics calculated
  expect_true(result$best_model$bic > result$best_model$aic)
  # Check relationship for non-NA values (optimization may leave some NA)
  non_na_idx <- !is.na(result$models$bic) & !is.na(result$models$aic)
  if (any(non_na_idx)) {
    expect_true(all(
      result$models$bic[non_na_idx] > result$models$aic[non_na_idx]
    ))
  }
})

test_that("AIC values are calculated correctly", {
  data <- create_test_data(n = 30, p = 3)

  # Use metric="aic" to ensure AIC is calculated
  result <- bestSubset(
    data$X,
    data$y,
    max_variables = 2,
    top_n = 3,
    metric = "aic"
  )

  # AIC = deviance + 2*k, where k counts predictors only (intercept not penalized)
  # For each model, verify the formula
  for (i in 1:nrow(result$models)) {
    k <- result$models$n_variables[i] # predictors only; intercept not penalized
    expected_aic <- result$models$deviance[i] + 2 * k
    expect_equal(result$models$aic[i], expected_aic, tolerance = 1e-10)
  }
})

test_that("BIC values are calculated correctly", {
  data <- create_test_data(n = 30, p = 3)

  # Use metric="bic" to ensure BIC is calculated
  result <- bestSubset(
    data$X,
    data$y,
    max_variables = 2,
    top_n = 3,
    metric = "bic"
  )

  # BIC = deviance + k*log(n), where k counts predictors only (intercept not penalized)
  n_obs <- nrow(data$X)
  for (i in 1:nrow(result$models)) {
    k <- result$models$n_variables[i] # predictors only; intercept not penalized
    expected_bic <- result$models$deviance[i] + k * log(n_obs)
    expect_equal(result$models$bic[i], expected_bic, tolerance = 1e-10)
  }
})

test_that("AIC metric works with cross-validation", {
  data <- create_test_data(n = 50, p = 3)

  result <- bestSubset(
    data$X,
    data$y,
    max_variables = 2,
    top_n = 3,
    metric = "aic",
    cross_validation = TRUE,
    cv_folds = 5,
    cv_seed = 123
  )

  # Check that results are valid
  expect_s3_class(result, "bestSubset")
  expect_true(all(result$models$aic > 0))
  expect_true(all(diff(result$models$aic) >= 0)) # Sorted ascending
})

test_that("BIC metric works with cross-validation", {
  data <- create_test_data(n = 50, p = 3)

  result <- bestSubset(
    data$X,
    data$y,
    max_variables = 2,
    top_n = 3,
    metric = "bic",
    cross_validation = TRUE,
    cv_folds = 5,
    cv_seed = 123
  )

  # Check that results are valid
  expect_s3_class(result, "bestSubset")
  expect_true(all(result$models$bic > 0))
  expect_true(all(diff(result$models$bic) >= 0)) # Sorted ascending
})

test_that("AIC and BIC metrics produce different model rankings", {
  data <- create_test_data(n = 60, p = 4)

  result_aic <- bestSubset(
    data$X,
    data$y,
    max_variables = 3,
    top_n = 5,
    metric = "aic"
  )
  result_bic <- bestSubset(
    data$X,
    data$y,
    max_variables = 3,
    top_n = 5,
    metric = "bic"
  )

  # BIC penalizes complexity more, so may select simpler models
  # The rankings might differ
  expect_true(length(result_aic$models$aic) > 0)
  expect_true(length(result_bic$models$bic) > 0)

  # Best model should have all metrics calculated
  expect_true(is.finite(result_aic$best_model$aic))
  expect_true(is.finite(result_aic$best_model$bic))
  expect_true(is.finite(result_bic$best_model$aic))
  expect_true(is.finite(result_bic$best_model$bic))

  # Other models (rank 2+) should only have selected metric
  if (nrow(result_aic$models) > 1) {
    expect_true(all(is.na(result_aic$models$bic[-1])))
    expect_true(all(is.na(result_bic$models$aic[-1])))
  }
})

test_that("AIC/BIC work with repeated CV", {
  data <- create_test_data(n = 40, p = 3)

  result_aic <- bestSubset(
    data$X,
    data$y,
    max_variables = 2,
    top_n = 3,
    metric = "aic",
    cross_validation = TRUE,
    cv_folds = 3,
    cv_repeats = 2,
    cv_seed = 456
  )

  result_bic <- bestSubset(
    data$X,
    data$y,
    max_variables = 2,
    top_n = 3,
    metric = "bic",
    cross_validation = TRUE,
    cv_folds = 3,
    cv_repeats = 2,
    cv_seed = 456
  )

  expect_s3_class(result_aic, "bestSubset")
  expect_s3_class(result_bic, "bestSubset")
  expect_true(all(result_aic$models$aic > 0))
  expect_true(all(result_bic$models$bic > 0))
})

test_that("Best model has all metrics, other models have only selected metric", {
  data <- create_test_data(n = 30, p = 3)

  # When using AUC metric, only AUC calculated during search (optimization)
  # But best model gets all metrics calculated at the end
  result <- bestSubset(
    data$X,
    data$y,
    max_variables = 2,
    top_n = 3,
    metric = "auc"
  )

  # All columns should exist in the output
  expect_true("aic" %in% names(result$models))
  expect_true("bic" %in% names(result$models))
  expect_true("accuracy" %in% names(result$models))
  expect_true("auc" %in% names(result$models))

  # Best model should have ALL metrics calculated
  expect_true("aic" %in% names(result$best_model))
  expect_true("bic" %in% names(result$best_model))
  expect_true("accuracy" %in% names(result$best_model))
  expect_true("auc" %in% names(result$best_model))
  expect_true(is.finite(result$best_model$aic))
  expect_true(is.finite(result$best_model$bic))
  expect_true(is.finite(result$best_model$accuracy))
  expect_true(is.finite(result$best_model$auc))

  # Other models (rank 2+) should have NA for non-selected metrics
  if (nrow(result$models) > 1) {
    expect_true(all(is.na(result$models$aic[-1])))
    expect_true(all(is.na(result$models$bic[-1])))
    expect_true(all(is.na(result$models$accuracy[-1])))
  }
})

# ============================================================================
# Parallel Processing Tests
# ============================================================================

test_that("Parallel gives same results as serial", {
  set.seed(123)
  data <- create_test_data(n = 50, p = 6)

  # Serial execution
  result_serial <- bestSubset(
    data$X,
    data$y,
    max_variables = 5,
    top_n = 3,
    n_threads = 1
  )

  # Parallel execution (2 threads)
  result_parallel <- bestSubset(
    data$X,
    data$y,
    max_variables = 5,
    top_n = 3,
    n_threads = 2
  )

  # Best model coefficients should be nearly identical
  # (minor differences due to floating-point arithmetic order in parallel)
  expect_equal(
    result_serial$best_model$coefficients,
    result_parallel$best_model$coefficients,
    tolerance = 1e-6
  )

  # Best model metrics should be identical
  expect_equal(
    result_serial$best_model$accuracy,
    result_parallel$best_model$accuracy,
    tolerance = 1e-10
  )

  expect_equal(
    result_serial$best_model$auc,
    result_parallel$best_model$auc,
    tolerance = 1e-10
  )
})

test_that("Parallel CV gives same results as serial CV", {
  set.seed(123)
  data <- create_test_data(n = 60, p = 5)

  # Serial CV
  result_serial <- bestSubset(
    data$X,
    data$y,
    cross_validation = TRUE,
    cv_folds = 3,
    cv_seed = 42,
    max_variables = 4,
    top_n = 2,
    n_threads = 1
  )

  # Parallel CV
  result_parallel <- bestSubset(
    data$X,
    data$y,
    cross_validation = TRUE,
    cv_folds = 3,
    cv_seed = 42,
    max_variables = 4,
    top_n = 2,
    n_threads = 2
  )

  # Results should be identical with same seed
  expect_equal(
    result_serial$best_model$coefficients,
    result_parallel$best_model$coefficients,
    tolerance = 1e-10
  )
})

test_that("Multiple parallel runs give consistent results", {
  # Run same analysis multiple times to check for race conditions
  results <- lapply(1:5, function(i) {
    set.seed(123) # Same seed for all runs
    data <- create_test_data(n = 50, p = 6)
    bestSubset(data$X, data$y, max_variables = 5, top_n = 2, n_threads = 2)
  })

  # All runs should give identical best models
  for (i in 2:5) {
    expect_equal(
      results[[1]]$best_model$coefficients,
      results[[i]]$best_model$coefficients,
      tolerance = 1e-10,
      info = paste("Run", i, "differs from run 1")
    )
  }
})

test_that("n_threads parameter validation works", {
  data <- create_test_data(n = 30, p = 3)

  # Valid: n_threads = 1
  expect_no_error(bestSubset(data$X, data$y, n_threads = 1))

  # Valid: n_threads = 2
  expect_no_error(bestSubset(data$X, data$y, n_threads = 2))

  # Valid: n_threads = NULL (auto-detect)
  expect_no_error(bestSubset(data$X, data$y, n_threads = NULL))

  # Invalid: n_threads = 0
  expect_error(
    bestSubset(data$X, data$y, n_threads = 0),
    "n_threads must be at least 1"
  )

  # Invalid: n_threads = -1
  expect_error(
    bestSubset(data$X, data$y, n_threads = -1),
    "n_threads must be at least 1"
  )

  # Invalid: n_threads = NaN
  expect_error(
    bestSubset(data$X, data$y, n_threads = NaN),
    "n_threads must be a finite number"
  )

  # Invalid: n_threads = Inf
  expect_error(
    bestSubset(data$X, data$y, n_threads = Inf),
    "n_threads must be a finite number"
  )
})

test_that("n_threads warns when exceeding available cores", {
  data <- create_test_data(n = 30, p = 3)

  n_cores <- parallel::detectCores()

  # Skip test if detectCores() returns NA (unlikely but possible)
  skip_if(is.na(n_cores), "Cannot detect number of cores")

  # Using more threads than available should trigger warning
  expect_warning(
    bestSubset(data$X, data$y, n_threads = n_cores + 5),
    "n_threads.*exceeds available CPU cores.*This may reduce performance"
  )

  # Using exactly available cores should not warn
  expect_no_warning(
    bestSubset(data$X, data$y, n_threads = n_cores)
  )

  # Using fewer cores should not warn
  expect_no_warning(
    bestSubset(data$X, data$y, n_threads = max(1, n_cores - 1))
  )
})

test_that("Parallel processing with different metrics", {
  set.seed(456)
  data <- create_test_data(n = 50, p = 5)

  # Test with each metric
  for (metric in c("accuracy", "auc", "deviance")) {
    result_serial <- bestSubset(
      data$X,
      data$y,
      metric = metric,
      max_variables = 4,
      n_threads = 1
    )

    result_parallel <- bestSubset(
      data$X,
      data$y,
      metric = metric,
      max_variables = 4,
      n_threads = 2
    )

    expect_equal(
      result_serial$best_model$coefficients,
      result_parallel$best_model$coefficients,
      tolerance = 1e-6,
      info = paste("Metric:", metric)
    )
  }
})
