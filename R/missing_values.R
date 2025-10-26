check_for_missing <- function(X, y) {
    X_na_count <- sum(is.na(X))
    y_na_count <- sum(is.na(y))
    
    if (X_na_count == 0 && y_na_count == 0) {
        return(list(has_missing = FALSE, message = NULL))
    }
    
    messages <- character(0)
    
    if (y_na_count > 0) {
        messages <- c(messages, paste("Response variable y contains", y_na_count, "missing values"))
    }
    
    if (X_na_count > 0) {
        na_by_col <- colSums(is.na(X))
        na_cols <- which(na_by_col > 0)
        col_names <- paste0("X", na_cols)
        messages <- c(messages, paste("Predictors contain", X_na_count, "missing values in columns:", paste(col_names, collapse = ", ")))
    }
    
    complete_cases <- complete.cases(cbind(X, y))
    n_complete <- sum(complete_cases)
    n_total <- nrow(X)
    n_removed <- n_total - n_complete
    
    if (n_removed > 0) {
        messages <- c(messages, paste(n_removed, "observations will be removed due to missing values"))
        messages <- c(messages, paste("Effective sample size:", n_complete, "out of", n_total, "observations"))
    }
    
    return(list(
        has_missing = TRUE,
        message = paste(messages, collapse = ". "),
        n_total = n_total,
        n_complete = n_complete,
        n_removed = n_removed,
        complete_cases = complete_cases
    ))
}

create_na_map <- function(original_length, complete_indices) {
    na_map <- rep(NA, original_length)
    na_map[complete_indices] <- seq_along(complete_indices)
    return(na_map)
}

handle_missing_values <- function(X, y, na.action) {
    if (!is.matrix(X)) {
        X <- as.matrix(X)
    }
    
    if (!is.vector(y)) {
        y <- as.vector(y)
    }
    
    if (nrow(X) != length(y)) {
        stop("X and y must have the same number of observations")
    }

    # Validate na.action parameter
    if (!is.function(na.action)) {
        stop("na.action must be a function (na.fail, na.omit, or na.exclude)")
    }

    if (!identical(na.action, na.fail) &&
        !identical(na.action, na.omit) &&
        !identical(na.action, na.exclude)) {
        stop("na.action must be na.fail, na.omit, or na.exclude")
    }

    na_info <- check_for_missing(X, y)
    
    if (!na_info$has_missing) {
        return(list(
            X_clean = X,
            y_clean = y,
            na_action_used = "none",
            original_n = nrow(X),
            effective_n = nrow(X),
            complete_cases = rep(TRUE, nrow(X)),
            na_map = NULL,
            message = "No missing values found"
        ))
    }
    
    if (identical(na.action, na.fail)) {
        stop(paste("Data contains missing values.", na_info$message, 
                  "Use na.action = na.omit to automatically remove incomplete cases, or clean your data first."))
    }
    
    if (identical(na.action, na.omit) || identical(na.action, na.exclude)) {
        complete_cases <- na_info$complete_cases
        
        if (sum(complete_cases) == 0) {
            stop("No complete cases available after removing missing values")
        }
        
        if (sum(complete_cases) < 2) {
            stop("Insufficient complete cases (need at least 2 observations)")
        }
        
        X_clean <- X[complete_cases, , drop = FALSE]
        y_clean <- y[complete_cases]
        
        na_map <- if (identical(na.action, na.exclude)) {
            create_na_map(length(y), which(complete_cases))
        } else {
            NULL
        }
        
        action_name <- if (identical(na.action, na.omit)) "na.omit" else "na.exclude"
        
        warning(paste("Missing values detected.", na_info$message), call. = FALSE)
        
        return(list(
            X_clean = X_clean,
            y_clean = y_clean,
            na_action_used = action_name,
            original_n = na_info$n_total,
            effective_n = na_info$n_complete,
            complete_cases = complete_cases,
            na_map = na_map,
            message = na_info$message
        ))
    }
    
    stop("na.action must be na.fail, na.omit, or na.exclude")
}

pad_results_for_exclude <- function(results_vector, na_map) {
    if (is.null(na_map)) {
        return(results_vector)
    }
    
    padded <- rep(NA, length(na_map))
    complete_positions <- !is.na(na_map)
    padded[complete_positions] <- results_vector
    
    return(padded)
}