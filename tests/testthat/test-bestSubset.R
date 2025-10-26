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
  expect_lte(max_vars_in_models, 3)
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
  expect_true(all(result$models$accuracy >= 0 & result$models$accuracy <= 1))
  expect_true(all(result$models$auc >= 0 & result$models$auc <= 1))
})

test_that("function handles edge case: single variable", {
  data <- create_test_data(p = 1)

  expect_warning(
    result <- bestSubset(data$X, data$y),
    "top_n.*exceeds available models."
  )
  expect_lte(nrow(result$models), 2)
  expect_true(all(result$models$n_variables <= 2))
})

test_that("best_model contains expected information", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y, top_n = 1)

  best_model <- result$best_model
  expect_true(all(is.finite(best_model$coefficients)))
  expect_gte(best_model$n_variables, 1)
  expect_lte(best_model$n_variables, ncol(data$X) + 1)
})

test_that("models data frame has correct structure", {
  data <- create_test_data()

  result <- bestSubset(data$X, data$y, top_n = 3)

  models_df <- result$models
  expect_true(all(models_df$auc >= 0 & models_df$auc <= 1))
  expect_true(all(models_df$accuracy >= 0 & models_df$accuracy <= 1))
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
  expect_true(all(
    result_cv$models$accuracy >= 0 & result_cv$models$accuracy <= 1
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
    bestSubset(data$X, data$y, cross_validation = TRUE, cv_folds = 5, cv_repeats = Inf),
    "cv_repeats must be a finite number"
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
