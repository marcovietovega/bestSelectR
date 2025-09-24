validate_input_matrix <- function(X, var_name = "X") {
    if (!is.matrix(X) && !is.data.frame(X)) {
        stop(paste(var_name, "must be a matrix or data frame"))
    }

    if (is.data.frame(X)) {
        non_numeric <- !sapply(X, is.numeric)
        if (any(non_numeric)) {
            categorical_vars <- names(X)[non_numeric]
            stop(
                "Categorical variables detected: ",
                paste(categorical_vars, collapse = ", "),
                ".\nbestSelectR requires numeric data."
            )
        }
        X <- as.matrix(X)
    }

    return(X)
}

validate_dimensions <- function(X, y) {
    if (length(y) != nrow(X)) {
        stop("X and y must have the same number of observations")
    }
    return(TRUE)
}

format_variables_string <- function(variables) {
    if (length(variables) == 0) {
        return("(none)")
    }
    return(paste0("X", variables, collapse = ","))
}

create_coefficient_names <- function(variables, include_intercept = TRUE) {
    coef_names <- character(0)

    if (include_intercept) {
        coef_names <- c(coef_names, "(Intercept)")
    }

    if (length(variables) > 0) {
        var_names <- paste0("X", variables)
        coef_names <- c(coef_names, var_names)
    }

    return(coef_names)
}

format_model_summary_line <- function(n_original, n_effective) {
    if (n_original == n_effective) {
        return(paste0("Observations: ", n_effective))
    } else {
        return(paste0(
            "Observations: ",
            n_effective,
            " of ",
            n_original,
            " (",
            n_original - n_effective,
            " removed due to missing values)"
        ))
    }
}
