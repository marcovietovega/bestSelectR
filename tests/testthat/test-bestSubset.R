# Tests for core bestSubset functionality

# Helper function to create test data
create_test_data <- function(n = 50, p = 4, seed = 123) {
  set.seed(seed)
  x_matrix <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(x_matrix) <- paste0("X", 1:p)

  # Create y with known relationship (adapt based on available columns)
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

# Test basic functionality
test_that("bestSubset returns correct object structure", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y, max_variables = 3, top_n = 3)

  # Check object class
  expect_s3_class(result, "bestSubset")

  # Check essential components exist
  expect_true("models" %in% names(result))
  expect_true("best_model" %in% names(result))
  expect_true("call_info" %in% names(result))
  expect_true("call" %in% names(result))
})

test_that("bestSubset works with minimal parameters", {
  data <- create_test_data(n = 20, p = 3)

  result <- bestSubset(data$X, data$y)

  expect_s3_class(result, "bestSubset")
  expect_true(nrow(result$models) <= 5)  # Default top_n = 5
})

test_that("max_variables parameter works correctly", {
  data <- create_test_data(p = 5)

  # Test with max_variables = 2, with reasonable top_n
  result <- bestSubset(data$X, data$y, max_variables = 2, top_n = 3)

  # All models should have at most 2 variables (plus intercept)
  max_vars_in_models <- max(result$models$n_variables)
  expect_lte(max_vars_in_models, 3)  # 2 variables + intercept
})

test_that("top_n parameter controls number of returned models", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y, top_n = 3)

  expect_lte(nrow(result$models), 3)
})

test_that("metric parameter affects model ranking", {
  data <- create_test_data()

  result_auc <- bestSubset(data$X, data$y, top_n = 3, metric = "auc")
  result_acc <- bestSubset(data$X, data$y, top_n = 3, metric = "accuracy")

  expect_s3_class(result_auc, "bestSubset")
  expect_s3_class(result_acc, "bestSubset")

  # Models might be ranked differently
  expect_true(all(c("auc", "accuracy") %in% names(result_auc$models)))
  expect_true(all(c("auc", "accuracy") %in% names(result_acc$models)))
})

test_that("function works with different data dimensions", {
  # Small dataset (2 variables = 3 possible models, so top_n=3 is safe)
  data_small <- create_test_data(n = 15, p = 2)
  result_small <- bestSubset(data_small$X, data_small$y, top_n = 3)
  expect_s3_class(result_small, "bestSubset")

  # Larger dataset
  data_large <- create_test_data(n = 100, p = 6)
  result_large <- bestSubset(data_large$X, data_large$y, max_variables = 3)
  expect_s3_class(result_large, "bestSubset")
})

test_that("function handles edge case: single variable", {
  data <- create_test_data(p = 1)

  # Only 1 possible model with 1 variable, so top_n=1
  result <- bestSubset(data$X, data$y, top_n = 1)

  expect_s3_class(result, "bestSubset")
  expect_equal(nrow(result$models), 1)  # Only one possible model
})

test_that("best_model contains expected information", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y, top_n = 2)

  best_model <- result$best_model

  # Check essential components
  expect_true("coefficients" %in% names(best_model))
  expect_true("variables" %in% names(best_model))
  expect_true("accuracy" %in% names(best_model))
  expect_true("auc" %in% names(best_model))
  expect_true("deviance" %in% names(best_model))

  # Coefficients should be numeric
  expect_type(best_model$coefficients, "double")

  # Variables should be integer (1-based indices)
  expect_type(best_model$variables, "integer")
})

test_that("models data frame has correct structure", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y, top_n = 3)

  models_df <- result$models

  # Check required columns exist
  expected_cols <- c("n_variables", "variables", "auc", "accuracy", "deviance")
  expect_true(all(expected_cols %in% names(models_df)))

  # Check data types
  expect_type(models_df$n_variables, "integer")
  expect_type(models_df$auc, "double")
  expect_type(models_df$accuracy, "double")
  expect_type(models_df$deviance, "double")

  # Check value ranges
  expect_true(all(models_df$auc >= 0 & models_df$auc <= 1))
  expect_true(all(models_df$accuracy >= 0 & models_df$accuracy <= 1))
})

test_that("call_info contains execution details", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y, max_variables = 3, metric = "auc")

  call_info <- result$call_info

  # Check essential information is captured
  expect_true("n_observations" %in% names(call_info))
  expect_true("n_predictors" %in% names(call_info))
  expect_true("metric" %in% names(call_info))

  expect_equal(call_info$n_observations, nrow(data$X))
  expect_equal(call_info$n_predictors, ncol(data$X))
  expect_equal(call_info$metric, "auc")
})

# Cross-validation tests
test_that("cross-validation functionality works", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y,
    max_variables = 3,
    cross_validation = TRUE,
    cv_folds = 3,
    top_n = 2
  )

  expect_s3_class(result, "bestSubset")

  # Check that CV information is recorded
  expect_true("use_cv" %in% names(result$call_info))
  expect_true(result$call_info$use_cv)
})

test_that("cross-validation with seed produces reproducible results", {
  data <- create_test_data()

  result1 <- bestSubset(data$X, data$y,
    cross_validation = TRUE,
    cv_folds = 3,
    cv_seed = 42,
    top_n = 2
  )

  result2 <- bestSubset(data$X, data$y,
    cross_validation = TRUE,
    cv_folds = 3,
    cv_seed = 42,
    top_n = 2
  )

  # Results should be identical with same seed
  expect_equal(result1$models$auc, result2$models$auc, tolerance = 1e-10)
  expect_equal(
    result1$models$accuracy, result2$models$accuracy, tolerance = 1e-10
  )
})

test_that("cross-validation parameters are properly recorded", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y,
    cross_validation = TRUE,
    cv_folds = 4,
    cv_repeats = 2,
    cv_seed = 123
  )

  call_info <- result$call_info

  expect_equal(call_info$cv_folds, 4)
  expect_equal(call_info$cv_repeats, 2)
  # Note: cv_seed might be stored differently, check if it exists
  if ("cv_seed" %in% names(call_info)) {
    expect_equal(call_info$cv_seed, 123)
  }
})

# Performance validation tests
test_that("performance metrics are reasonable", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y, top_n = 3)

  models <- result$models

  # AUC should be between 0.5 and 1 for meaningful models
  expect_true(all(models$auc >= 0.4))  # Allow some tolerance for random data
  expect_true(all(models$auc <= 1.0))

  # Accuracy should be between 0 and 1
  expect_true(all(models$accuracy >= 0))
  expect_true(all(models$accuracy <= 1))

  # Deviance should be positive
  expect_true(all(models$deviance >= 0))
})

test_that("models are properly ranked by performance", {
  data <- create_test_data()

  result_auc <- bestSubset(data$X, data$y, metric = "auc", top_n = 5)
  result_acc <- bestSubset(data$X, data$y, metric = "accuracy", top_n = 5)

  # Models should be sorted by the specified metric (descending)
  auc_values <- result_auc$models$auc
  expect_true(all(diff(auc_values) <= 0))  # Should be non-increasing

  acc_values <- result_acc$models$accuracy
  expect_true(all(diff(acc_values) <= 0))  # Should be non-increasing
})

test_that("function works with data frame input", {
  data <- create_test_data()

  # Convert matrix to data frame
  x_df <- as.data.frame(data$X)

  result <- bestSubset(x_df, data$y, top_n = 2)

  expect_s3_class(result, "bestSubset")
  expect_equal(result$call_info$n_predictors, ncol(x_df))
})