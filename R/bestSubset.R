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
#' @param metric Selection metric. One of "accuracy" or "auc" (default)
#' @param cross_validation Logical indicating whether to use cross-validation (default: FALSE)
#' @param cv_folds Number of cross-validation folds (default: 5)
#' @param cv_repeats Number of cross-validation repeats (default: 1)
#' @param cv_seed Random seed for cross-validation reproducibility (default: NULL)
#' @param na.action How to handle missing values. One of na.fail (default), na.omit, or na.exclude
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
  metric = c("auc", "accuracy"),
  cross_validation = FALSE,
  cv_folds = 5,
  cv_repeats = 1,
  cv_seed = NULL,
  na.action = na.fail
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

  warn_computational_complexity(ncol(X_clean), validated_params$max_variables)

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
    tolerance = tolerance
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
