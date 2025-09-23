#' @export
print.bestSubset <- function(x, ...) {
    cat("Best Subset Selection for Logistic Regression\n")
    cat("=============================================\n\n")
    
    info <- x$call_info
    cat("Data Information:\n")
    
    if (!is.null(info$n_observations_original) && info$n_observations_original != info$n_observations) {
        cat("  ", format_model_summary_line(info$n_observations_original, info$n_observations), "\n")
    } else {
        cat("  Observations:", info$n_observations, "\n")
    }
    
    cat("  Predictors:", info$n_predictors, "\n")
    cat("  Models evaluated:", info$n_models_evaluated, "\n\n")
    
    cat("Selection Settings:\n")
    cat("  Metric:", info$metric, "\n")
    cat("  Max variables:", if(info$max_variables == info$n_predictors) "all" else info$max_variables, "\n")
    cat("  Cross-validation:", if(info$use_cv) "Yes" else "No")
    if (info$use_cv) {
        cat(" (", info$cv_folds, "-fold", if(info$cv_repeats > 1) paste(",", info$cv_repeats, "repeats"), ")", sep="")
    }
    cat("\n")
    if (!is.null(x$na_info) && x$na_info$na_action_used != "none") {
        cat("  Missing values:", x$na_info$na_action_used, "\n")
    }
    cat("\n")
    
    cat("Best Model:\n")
    best <- x$best_model
    cat("  Variables:", best$n_variables, "\n")
    cat("  Accuracy:", sprintf("%.4f", best$accuracy), "\n")
    cat("  AUC:", sprintf("%.4f", best$auc), "\n")
    cat("  Deviance:", sprintf("%.4f", best$deviance), "\n\n")
    }

#' @export
summary.bestSubset <- function(object, ...) {
    cat("Best Subset Selection Results\n")
    cat("============================\n\n")
    
    print(object)
    
    cat("\nTop Models:\n")
    models_df <- object$models
    
    display_df <- data.frame(
        Rank = models_df$rank,
        Variables = models_df$variables,
        N_Vars = models_df$n_variables,
        Accuracy = sprintf("%.4f", models_df$accuracy),
        AUC = sprintf("%.4f", models_df$auc),
        Deviance = sprintf("%.4f", models_df$deviance),
        stringsAsFactors = FALSE
    )
    
    print(display_df, row.names = FALSE)
    
    cat("\nBest Model Coefficients:\n")
    best <- object$best_model
    coef_names <- create_coefficient_names(best$variables, object$call_info$include_intercept)
    
    coef_df <- data.frame(
        Coefficient = coef_names,
        Estimate = sprintf("%.6f", best$coefficients),
        stringsAsFactors = FALSE
    )
    
    print(coef_df, row.names = FALSE)
    
    if (!is.null(object$na_info) && object$na_info$n_removed > 0) {
        cat(paste("\nNote: Results based on", object$na_info$effective_n, 
                 "complete observations out of", object$na_info$original_n, "total observations.\n"))
        cat(paste(object$na_info$n_removed, "observations removed due to missing values.\n"))
    }
}

#' Predict method for bestSubset objects
#'
#' @param object A bestSubset object from bestSubset()
#' @param newdata A matrix or data frame of new predictor values
#' @param type Type of prediction: "class" for binary predictions or "prob" for probabilities
#' @param ... Additional arguments (currently unused)
#'
#' @return A numeric vector of predictions
#'
#' @export
predict.bestSubset <- function(object, newdata, type = c("class", "prob"), ...) {
    type <- match.arg(type)
    
    if (missing(newdata)) {
        stop("newdata is required for predictions")
    }
    
    newdata <- validate_input_matrix(newdata, "newdata")
    
    if (ncol(newdata) < max(object$best_model$variables)) {
        stop("newdata must have at least ", max(object$best_model$variables), " columns")
    }
    
    # Handle missing values in new data consistently with training
    if (!is.null(object$na_info) && object$na_info$na_action_used %in% c("na.omit", "na.exclude")) {
        # Check for missing values in newdata
        na_check <- check_for_missing(newdata, rep(0, nrow(newdata)))  # dummy y for check
        
        if (na_check$has_missing) {
            if (object$na_info$na_action_used == "na.exclude") {
                # For na.exclude, pad results with NA
                complete_cases <- na_check$complete_cases
                newdata_clean <- newdata[complete_cases, , drop = FALSE]
                
                if (nrow(newdata_clean) == 0) {
                    return(rep(NA, nrow(newdata)))
                }
                
                predictions_clean <- predict_best_subset(object$best_model, newdata_clean, type)
                
                # Pad results
                predictions <- rep(NA, nrow(newdata))
                predictions[complete_cases] <- predictions_clean
                
                return(predictions)
                
            } else {
                # For na.omit, just use complete cases
                warning("newdata contains missing values. Using complete cases only.", call. = FALSE)
                complete_cases <- na_check$complete_cases
                newdata <- newdata[complete_cases, , drop = FALSE]
                
                if (nrow(newdata) == 0) {
                    return(numeric(0))
                }
            }
        }
    } else if (!is.null(object$na_info) && object$na_info$na_action_used == "na.fail") {
        # Check for any missing values and fail if found
        if (any(is.na(newdata))) {
            stop("newdata contains missing values. Model was fitted with na.action = na.fail")
        }
    }
    
    predictions <- predict_best_subset(object$best_model, newdata, type)
    
    return(predictions)
}

#' Extract coefficients from bestSubset objects
#'
#' @param object A bestSubset object from bestSubset()
#' @param model_rank Which ranked model to extract coefficients from (default: 1 = best model)
#' @param ... Additional arguments (currently unused)
#'
#' @return A named numeric vector of coefficients
#'
#' @export
coef.bestSubset <- function(object, model_rank = 1, ...) {
    if (model_rank < 1 || model_rank > nrow(object$models)) {
        stop("model_rank must be between 1 and ", nrow(object$models))
    }
    
    if (model_rank == 1) {
        coeffs <- object$best_model$coefficients
        variables <- object$best_model$variables
    } else {
        stop("Extracting coefficients from non-best models not yet implemented")
    }
    
    coef_names <- create_coefficient_names(variables, object$call_info$include_intercept)
    names(coeffs) <- coef_names
    
    return(coeffs)
}