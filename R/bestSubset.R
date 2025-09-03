#' Best Subset Selection for Logistic Regression
#'
#' Performs best subset selection for logistic regression to find
#' the optimal subset of predictors based on various metrics.
#'
#' @param X A numeric matrix or data frame of predictor variables (n x p)
#' @param y A numeric vector of binary outcomes (0/1)
#' @param max_variables Maximum number of variables to consider in subsets.
#'   If NULL (default), considers all variables.
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
#' \dontrun{
#' # Generate sample data
#' set.seed(123)
#' n <- 100
#' X <- matrix(rnorm(n*4), nrow=n, ncol=4)
#' colnames(X) <- paste0("X", 1:4)
#' y <- rbinom(n, 1, plogis(X[,1] + 0.5*X[,2] - 0.3*X[,3]))
#'
#' # Basic best subset selection
#' result <- bestSubset(X, y, max_variables=3, top_n=5)
#' print(result)
#' summary(result)
#'
#' # With cross-validation
#' result_cv <- bestSubset(X, y, cross_validation=TRUE, cv_folds=5, metric="auc")
#' print(result_cv)
#'
#' # Make predictions
#' predictions <- predict(result, X)
#' probabilities <- predict(result, X, type="prob")
#' }
#'
#' @export
bestSubset <- function(X, y, 
                      max_variables = NULL,
                      top_n = 5,
                      metric = c("auc", "accuracy"),
                      cross_validation = FALSE,
                      cv_folds = 5,
                      cv_repeats = 1,
                      cv_seed = NULL,
                      na.action = na.fail) {
  
  call <- match.call()
  
  # Set fixed internal parameters
  include_intercept <- TRUE
  max_iterations <- 100
  tolerance <- 1e-6
  
  # Input validation
  X <- validate_input_matrix(X, "X")
  validate_dimensions(X, y)
  
  # Handle missing values
  na_result <- handle_missing_values(X, y, na.action)
  X_clean <- na_result$X_clean
  y_clean <- na_result$y_clean
  
  metric <- match.arg(metric)
  
  # Parameter validation
  validated_params <- validate_parameters(max_variables, top_n, metric,
                                        cross_validation, cv_folds, cv_repeats,
                                        ncol(X_clean), nrow(X_clean))
  
  # Essential data validation
  validate_data_requirements(X_clean, y_clean)
  
  # Perfect separation detection
  detect_perfect_separation(X_clean, y_clean)
  
  # Computational complexity warning
  warn_computational_complexity(ncol(X_clean), validated_params$max_variables)
  
  # Use validated parameters
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
    cv_seed <- as.integer(cv_seed)
  }

  result <- best_subset_selection(
    X_clean, y_clean,
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
  
  # Add missing value information to result
  result$na_info <- list(
    na_action_used = na_result$na_action_used,
    original_n = na_result$original_n,
    effective_n = na_result$effective_n,
    n_removed = na_result$original_n - na_result$effective_n,
    na_map = na_result$na_map
  )
  
  # Update call info with missing value information
  result$call_info$n_observations_original <- na_result$original_n
  result$call_info$n_observations <- na_result$effective_n
  result$call_info$n_removed <- na_result$original_n - na_result$effective_n
  
  result$call <- call
  class(result) <- "bestSubset"
  
  return(result)
}
