validate_parameters <- function(max_variables, top_n, metric, 
                               cross_validation, cv_folds, cv_repeats, 
                               n_predictors, n_obs) {
    
    validated <- list()
    
    # Validate max_variables
    if (!is.null(max_variables)) {
        if (!is.numeric(max_variables) || length(max_variables) != 1 || max_variables < 1) {
            stop("max_variables must be a positive integer")
        }

        max_variables <- as.integer(max_variables)

        if (max_variables > n_predictors) {
            warning(paste0("max_variables (", max_variables, ") exceeds predictors (",
                          n_predictors, "). Using all ", n_predictors, " predictors."),
                   call. = FALSE)
            max_variables <- n_predictors
        }

        validated$max_variables <- max_variables
    } else {
        validated$max_variables <- NULL
    }
    
    # Validate top_n
    if (!is.numeric(top_n) || length(top_n) != 1 || top_n < 1) {
        stop("top_n must be at least 1")
    }
    
    top_n <- as.integer(top_n)
    
    # Limit top_n to maximum of 10 for readable output
    if (top_n > 10) {
        warning("top_n limited to maximum of 10 models for readable output", call. = FALSE)
        top_n <- 10
    }
    
    # Calculate maximum possible models
    actual_p <- if (is.null(max_variables)) n_predictors else max_variables
    max_models <- 2^min(actual_p, 15) - 1  # Limit to prevent overflow
    
    if (top_n > max_models) {
        warning(paste0("top_n (", top_n, ") exceeds available models (", 
                      max_models, "). Returning all models."), call. = FALSE)
        top_n <- max_models
    }
    
    validated$top_n <- top_n
    
    # Validate metric
    valid_metrics <- c("auc", "accuracy")
    if (!metric %in% valid_metrics) {
        stop("metric must be 'auc' or 'accuracy'")
    }
    
    validated$metric <- metric
    
    # Validate cross-validation parameters (only when CV is enabled)
    if (cross_validation) {
        if (!is.numeric(cv_folds) || length(cv_folds) != 1) {
            stop("cv_folds must be a numeric value")
        }

        if (is.nan(cv_folds) || is.infinite(cv_folds)) {
            stop("cv_folds must be a finite number")
        }

        if (cv_folds < 2) {
            stop("cv_folds must be at least 2")
        }

        cv_folds <- as.integer(cv_folds)

        if (cv_folds > n_obs) {
            stop(paste0("cv_folds (", cv_folds, ") cannot exceed observations (", n_obs, ")"))
        }

        if (cv_folds > 20) {
            warning(paste0("cv_folds = ", cv_folds, " may result in very slow cross-validation. ",
                          "Consider using cv_folds <= 20 for reasonable performance."),
                   call. = FALSE)
        }

        if (!is.numeric(cv_repeats) || length(cv_repeats) != 1) {
            stop("cv_repeats must be a numeric value")
        }

        if (is.nan(cv_repeats) || is.infinite(cv_repeats)) {
            stop("cv_repeats must be a finite number")
        }

        if (cv_repeats < 1) {
            stop("cv_repeats must be at least 1")
        }

        cv_repeats <- as.integer(cv_repeats)
        
        validated$cv_folds <- cv_folds
        validated$cv_repeats <- cv_repeats
    }
    
    return(validated)
}

validate_data_requirements <- function(X_clean, y_clean) {
    
    # Minimum sample size
    if (nrow(X_clean) < 2) {
        stop("Need at least 2 observations")
    }
    
    # Minimum predictors
    if (ncol(X_clean) < 1) {
        stop("Need at least 1 predictor variable")
    }
    
    # Response variable validation
    if (!is.numeric(y_clean)) {
        stop("Response variable y must be numeric")
    }
    
    unique_values <- unique(y_clean)
    if (length(unique_values) != 2) {
        stop("Response variable must have exactly 2 distinct values")
    }
    
    if (!all(y_clean %in% c(0, 1))) {
        stop("Response variable must contain only 0 and 1 values")
    }
    
    return(TRUE)
}

detect_perfect_separation <- function(X_clean, y_clean) {
    
    n_predictors <- ncol(X_clean)
    separated_vars <- character(0)
    
    for (j in 1:n_predictors) {
        x_j <- X_clean[, j]
        
        # Check if this predictor perfectly separates the classes
        # Simple approach: check if all 0s have x < threshold and all 1s have x >= threshold
        y0_values <- x_j[y_clean == 0]
        y1_values <- x_j[y_clean == 1]
        
        if (length(y0_values) > 0 && length(y1_values) > 0) {
            max_y0 <- max(y0_values)
            min_y1 <- min(y1_values)
            
            # Perfect separation if no overlap
            if (max_y0 < min_y1) {
                separated_vars <- c(separated_vars, paste0("X", j))
            }
            
            # Also check the other direction
            max_y1 <- max(y1_values)
            min_y0 <- min(y0_values)
            
            if (max_y1 < min_y0) {
                separated_vars <- c(separated_vars, paste0("X", j))
            }
        }
    }
    
    # Remove duplicates
    separated_vars <- unique(separated_vars)
    
    if (length(separated_vars) > 0) {
        warning(paste0("Perfect separation detected for variable(s): ", 
                      paste(separated_vars, collapse = ", "), 
                      ". Results may be unreliable."), call. = FALSE)
    }
    
    return(separated_vars)
}

warn_computational_complexity <- function(n_predictors, max_variables_used) {

    actual_p <- if (is.null(max_variables_used)) n_predictors else max_variables_used
    total_models <- 2^actual_p - 1

    # Warn when searching more than 20 variables
    if (actual_p > 20) {
        warning(paste0(
            "Large search space: 2^", actual_p, " - 1 = ",
            format(total_models, big.mark = ",", scientific = TRUE),
            " models.\n",
            "Computation may be very slow or infeasible. ",
            "Consider setting max_variables <= 20 for reasonable performance."
        ), call. = FALSE)
    }

    return(invisible(total_models))
}