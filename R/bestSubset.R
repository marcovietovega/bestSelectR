#' Best Subset Selection for Logistic Regression
#'
#' Performs best subset selection for logistic regression to find
#' the optimal subset of predictors based on various metrics.
#'
#' @param X A numeric matrix or data frame of predictor variables (n x p).
#'   Categorical variables are not supported - use model.matrix() to create
#'   dummy variables first.
#' @param y A numeric vector of binary outcomes containing exactly 0 and 1 values.
#'   Other binary formats require conversion
#' @param max_variables Maximum number of variables to consider in subsets.
#'   If NULL (default), considers all variables. Values > 20 will trigger
#'   a warning about computational complexity.
#' @param top_n Number of top models to return (default: 5, max: 10)
#' @param metric Selection metric. One of "accuracy", "auc" (default), "deviance", "aic", or "bic".
#'   For "accuracy" and "auc", higher values indicate better models.
#'   For "deviance", "aic", and "bic", lower values indicate better models.
#'   AIC (Akaike Information Criterion) = deviance + 2k, and BIC (Bayesian Information Criterion)
#'   = deviance + k*log(n), where n is the sample size and k is the number of predictors only
#'   (the intercept is not penalized). Both AIC and BIC penalize model complexity, with BIC
#'   applying a stronger penalty for larger datasets.
#' @param cross_validation Logical indicating whether to use cross-validation (default: FALSE)
#' @param cv_folds Number of cross-validation folds (default: 5)
#' @param cv_repeats Number of cross-validation repeats (default: 1)
#' @param cv_seed Random seed for cross-validation reproducibility (default: NULL)
#' @param na.action How to handle missing values. One of na.fail (default), na.omit, or na.exclude
#' @param n_threads Number of threads for parallel processing (default: NULL for auto-detection).
#'   Use 1 for serial execution, or specify a positive integer for a specific thread count.
#'   Set to NULL to automatically use all available cores minus 1.
#' @param omp_schedule Optional OpenMP schedule string to tune parallel work-stealing at runtime
#'   when n_threads > 1. Examples: "guided,32", "dynamic,64". If NULL (default), the
#'   system environment (OMP_SCHEDULE) is used as-is. This is ignored when n_threads = 1.
#'
#' @details
#' Selection is performed by evaluating only the chosen metric (plus deviance) for every
#' candidate subset, then sorting models by that metric and keeping the top_n models. Full
#' metric values for all measures are computed only for the single best model after selection.
#'
#' Cross-validation affects selection differently depending on the metric:
#' - For "accuracy" and "auc", the ranking uses cross-validated averages when
#'   `cross_validation = TRUE`.
#' - For "deviance", "aic", and "bic", the ranking always uses the full-data values even when
#'   `cross_validation = TRUE`. In these cases CV is not used to choose the subset but can still be
#'   used downstream for model assessment if desired.
#'
#' @return A list with class "bestSubset" containing:
#' \describe{
#'   \item{models}{Data frame with results for top models}
#'   \item{best_model}{List with detailed information about the best model}
#'   \item{call_info}{Information about the function call and data}
#'   \item{call}{The original function call}
#' }
#'
#' @examples
#' # Generate sample data (small for fast execution)
#' set.seed(123)
#' X <- matrix(rnorm(20*3), nrow=20, ncol=3)
#' colnames(X) <- paste0("X", 1:3)
#' y <- rbinom(20, 1, plogis(X[,1] + 0.5*X[,2]))
#'
#' # Basic best subset selection
#' result <- bestSubset(X, y, max_variables=2, top_n=3)
#' print(result)
#' summary(result)
#'
#' @export
bestSubset <- function(
  X,
  y,
  max_variables = NULL,
  top_n = 5,
  metric = c("auc", "accuracy", "deviance", "aic", "bic"),
  cross_validation = FALSE,
  cv_folds = 5,
  cv_repeats = 1,
  cv_seed = NULL,
  na.action = na.fail,
  n_threads = NULL,
  omp_schedule = NULL
) {
  call <- match.call()

  include_intercept <- TRUE
  max_iterations <- 100
  tolerance <- 1e-6

  X <- validate_input_matrix(X, "X")
  validate_dimensions(X, y)

  na_result <- handle_missing_values(X, y, na.action)
  X_clean <- na_result$X_clean
  y_clean <- na_result$y_clean

  validate_data_requirements(X_clean, y_clean)

  metric <- match.arg(metric)

  validated_params <- validate_parameters(
    max_variables,
    top_n,
    metric,
    cross_validation,
    cv_folds,
    cv_repeats,
    ncol(X_clean),
    nrow(X_clean)
  )

  detect_perfect_separation(X_clean, y_clean)

  if (!is.null(validated_params$max_variables) && validated_params$max_variables > 60) {
    stop(paste0(
      "max_variables cannot exceed 60.\n",
      "Reason: Best subset selection beyond 60 variables is computationally infeasible.\n",
      "Solution: Set max_variables <= 60, or use regularization methods (LASSO, elastic net)."
    ), call. = FALSE)
  }

  actual_max_vars <- if (is.null(validated_params$max_variables)) {
    ncol(X_clean)
  } else {
    min(validated_params$max_variables, ncol(X_clean))
  }
  warn_computational_complexity(ncol(X_clean), actual_max_vars)

  max_variables <- validated_params$max_variables
  top_n <- validated_params$top_n
  metric <- validated_params$metric

  if (is.null(max_variables)) {
    max_variables <- -1L
  } else {
    max_variables <- as.integer(max_variables)
  }

  if (is.null(cv_seed)) {
    cv_seed <- -1L
  } else {
    if (!is.numeric(cv_seed) || length(cv_seed) != 1) {
      stop("cv_seed must be a single numeric value or NULL")
    }
    if (is.nan(cv_seed) || is.infinite(cv_seed)) {
      stop("cv_seed must be a finite number or NULL")
    }
    cv_seed <- as.integer(cv_seed)
  }

  # Handle n_threads parameter
  if (is.null(n_threads)) {
    # Auto-detect: use all available cores minus 1
    n_threads <- max(1L, parallel::detectCores() - 1L)
  } else {
    if (!is.numeric(n_threads) || length(n_threads) != 1) {
      stop("n_threads must be a single numeric value or NULL")
    }
    if (is.nan(n_threads) || is.infinite(n_threads)) {
      stop("n_threads must be a finite number")
    }
    if (n_threads < 1) {
      stop("n_threads must be at least 1 (use 1 for serial execution)")
    }
    n_threads <- as.integer(n_threads)

    # Warn if n_threads exceeds available cores
    n_cores_available <- parallel::detectCores()
    if (!is.na(n_cores_available) && n_threads > n_cores_available) {
      warning(
        "n_threads (",
        n_threads,
        ") exceeds available CPU cores (",
        n_cores_available,
        "). This may reduce performance.",
        call. = FALSE
      )
    }
  }

  # Apply optional OpenMP schedule override (must be set before entering C++ parallel regions)
  if (!is.null(omp_schedule)) {
    if (!is.character(omp_schedule) || length(omp_schedule) != 1L) {
      stop(
        "omp_schedule must be a single character string like 'guided,32' or NULL"
      )
    }
    Sys.setenv(OMP_SCHEDULE = omp_schedule)
  }

  # Calculate number of subsets to evaluate
  n_predictors <- ncol(X_clean)
  if (max_variables == -1 || max_variables > n_predictors) {
    n_subsets <- 2^n_predictors - 1
  } else {
    n_subsets <- sum(sapply(1:max_variables, function(k) {
      choose(n_predictors, k)
    }))
  }

  # Print computational settings
  cat("Computational Settings:\n")
  cat("  Threads used:", n_threads, "\n")
  # Check if OpenMP is available by testing if _OPENMP is defined in compiled code
  # We approximate this by checking if n_threads > 1 was accepted
  if (n_threads > 1) {
    cat("  OpenMP: Enabled\n")
    # Show schedule if available
    sch <- Sys.getenv("OMP_SCHEDULE", unset = NA_character_)
    if (!is.na(sch) && nzchar(sch)) {
      cat("  OMP_SCHEDULE:", sch, "\n")
    }
  } else {
    cat("  OpenMP: Serial mode (1 thread)\n")
  }
  cat("  Evaluating", format(n_subsets, big.mark = ","), "subset models\n\n")

  result <- best_subset_selection(
    X_clean,
    y_clean,
    max_variables = max_variables,
    top_n = top_n,
    metric = metric,
    use_cv = cross_validation,
    cv_folds = cv_folds,
    cv_repeats = cv_repeats,
    cv_seed = cv_seed,
    include_intercept = include_intercept,
    max_iterations = max_iterations,
    tolerance = tolerance,
    n_threads = n_threads
  )

  result$na_info <- list(
    na_action_used = na_result$na_action_used,
    original_n = na_result$original_n,
    effective_n = na_result$effective_n,
    n_removed = na_result$original_n - na_result$effective_n,
    na_map = na_result$na_map
  )

  result$call_info$n_observations_original <- na_result$original_n
  result$call_info$n_observations <- na_result$effective_n
  result$call_info$n_removed <- na_result$original_n - na_result$effective_n

  result$call <- call
  class(result) <- "bestSubset"

  return(result)
}
