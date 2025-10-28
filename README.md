# Best Subset Selection for Logistic Regression

**bestSelectR** helps you find the best subset of predictor variables for logistic regression models, especially when your outcome is binary (e.g., 0 or 1). It works by checking all possible combinations of your variables and ranks them based on how well they predict the outcome.

This is useful when you're not sure which variables are most important. Instead of using stepwise selection or picking variables manually, **bestSelectR** tests every subset and finds the combinations that perform best. It also includes cross-validation to help avoid overfitting and give more reliable performance estimates.

## Table of Contents

- [Installation](#installation)
- [Test Plan for bestSelectR](#test-plan-for-bestselectr)
  - [1. Setup and Installation](#1-setup-and-installation)
  - [2. Main Functionality](#2-main-functionality)
    - [2.1. Variable Counting Verification](#21-variable-counting-verification)
  - [3. Cross-Validation Functionality](#3-cross-validation-functionality)
  - [4. Missing Data Handling](#4-missing-data-handling)
  - [5. Prediction Functionality](#5-prediction-functionality)
  - [6. Error and Edge Cases](#6-error-and-edge-cases)
    - [6.5. NaN/Inf Validation Tests](#65-naninf-validation-tests)
    - [6.6. Column Order Dependency Tests](#66-column-order-dependency-tests)
    - [6.7. High-Dimensional Scalability Tests](#67-high-dimensional-scalability-tests)
    - [6.8. Numerical Stability Tests](#68-numerical-stability-tests)
  - [7. Real Dataset Example (mtcars)](#7-real-dataset-example-mtcars)
- [Requirements](#requirements)
- [Function Reference](#function-reference)
- [Output Summary](#output-summary)
- [More Examples](#more-examples)
- [Performance Metrics](#performance-metrics)
- [Usage Tips](#usage-tips)
- [Documentation](#documentation)

## Installation

Install directly from GitHub:

```r
devtools::install_github("marcovietovega/bestSelectR",
  build_vignettes = TRUE,
  INSTALL_opts = "--install-tests"
)
```

You'll need `devtools` for installation. All other dependencies will be installed automatically.

## Test Plan for bestSelectR

This section provides a test plan for **bestSelectR** that allows reviewers to verify the package's functionality step-by-step.

### 1. Setup and Installation

1. Install `devtools` if not already installed:

```r
install.packages("devtools")
```

2. Install the package from GitHub:

```r
devtools::install_github("marcovietovega/bestSelectR",
  build_vignettes = TRUE,
  INSTALL_opts = "--install-tests"
)
```

3. Load the package:

```r
library(bestSelectR)
```

**Expected outcome**: Package loads without errors. Vignette should be available:

```r
vignette("bestSelectR-tutorial", package = "bestSelectR")
```

### 2. Main Functionality

Test the core best subset selection functionality:

```r
# Load built-in mtcars dataset
data(mtcars)

# Use transmission (am) as binary outcome: 0=automatic, 1=manual
X <- as.matrix(mtcars[, c("mpg", "hp", "wt", "qsec")])  # Select 4 predictors
y <- mtcars$am

# Run best subset selection
result <- bestSubset(X, y, max_variables = 3, top_n = 5)

print(result)
summary(result)
```

**Expected outcome**:

- `print(result)` shows summary information (best model information, selection settings, data information)
- `summary(result)` displays detailed results including coefficients and top models
- AUC values should be between 0.5 and 1.0

#### 2.1. Variable Counting Verification

Verify that variable counts correctly exclude the intercept from predictor count:

```r
# Extract best model information
best_model <- result$best_model

# Get number of variables (should count predictors, not including intercept)
n_vars_reported <- best_model$n_variables

# Count actual predictors selected (intercept is index -1 or separate)
n_predictors <- length(best_model$variables)

# Get coefficients (includes intercept)
coeffs <- coef(result)
n_coeffs <- length(coeffs)

cat("Variable count verification:\n")
cat("  Reported n_variables:", n_vars_reported, "\n")
cat("  Number of predictor indices:", n_predictors, "\n")
cat("  Number of coefficients:", n_coeffs, "\n")

# Verification: n_coeffs should equal n_predictors + 1 (for intercept)
if (n_coeffs == n_predictors + 1) {
  cat("  Status: CORRECT - Intercept properly separated from predictor count\n")
} else {
  cat("  Status: CHECK - Unexpected relationship between counts\n")
}

# Verify coefficient names
cat("  Coefficient names:", paste(names(coeffs), collapse = ", "), "\n")
if (names(coeffs)[1] == "(Intercept)") {
  cat("  Status: CORRECT - Intercept explicitly named\n")
}
```

**Expected outcome**:

- `n_variables` should count predictors only (excluding intercept)
- Number of coefficients should be `n_predictors + 1` (including intercept)
- Coefficient names should clearly show "(Intercept)" as first element
- Verification confirms intercept is handled separately in counts

### 3. Cross-Validation Functionality

Test cross-validation for more reliable performance estimates:

```r
result_cv <- bestSubset(X, y,
                        max_variables = 3,
                        cross_validation = TRUE,
                        cv_folds = 5,
                        metric = "auc")

print(result_cv)
summary(result_cv)
```

**Expected outcome**:

- Results should include CV-based performance metrics
- Performance may differ from non-CV results
- Should show "Cross-validation: Yes (5-fold)" in output

### 4. Missing Data Handling

Test different approaches to handling missing values:

```r
# Create dataset with missing values
X_missing <- X
set.seed(123)
X_missing[sample(length(X_missing), 5)] <- NA

# Test na.omit (removes rows with missing values)
result_omit <- bestSubset(X_missing, y, na.action = na.omit)
print(result_omit)

# Test na.exclude (removes rows but preserves indices)
result_exclude <- bestSubset(X_missing, y, na.action = na.exclude)

# Test na.fail (should produce an error)
try({
  result_fail <- bestSubset(X_missing, y, na.action = na.fail)
}, silent = TRUE)
cat("na.fail correctly produced an error\n")
```

**Expected outcome**:

- `na.omit` and `na.exclude` run successfully with reduced sample size
- `na.fail` produces an error about missing values
- Output should indicate number of observations removed

### 5. Prediction Functionality

Test prediction methods on new data:

```r
# Create new data for predictions (same structure as original)
new_data <- matrix(c(
  20.0, 150, 3.0, 18.0,  # Car 1
  15.0, 300, 4.5, 16.0,  # Car 2
  25.0, 100, 2.5, 20.0   # Car 3
), nrow = 3, ncol = 4, byrow = TRUE)
colnames(new_data) <- colnames(X)

# Get probability predictions
pred_prob <- predict(result, new_data, type = "prob")
cat("Probability predictions:\n")
print(pred_prob)

# Get class predictions
pred_class <- predict(result, new_data, type = "class")
cat("Class predictions:\n")
print(pred_class)

# Extract coefficients
coeffs <- coef(result)
cat("Model coefficients:\n")
print(coeffs)
```

**Expected outcome**:

- `pred_prob` returns probabilities between 0 and 1
- `pred_class` returns binary predictions (0 or 1)
- `coef()` returns named coefficient vector

### 6. Error and Edge Cases

Test error handling for invalid inputs:

```r
# Test non-binary outcome variable
y_invalid <- sample(1:3, nrow(mtcars), replace = TRUE)
try({
  bestSubset(X, y_invalid)
}, silent = TRUE)
cat("Non-binary y correctly produced an error\n")

# Test top_n greater than maximum allowed (10)
try({
  bestSubset(X, y, top_n = 15)
}, silent = TRUE)
cat("top_n > 10 correctly produced an error\n")

# Test categorical data error (create factor data)
df_categorical <- data.frame(
  mpg = mtcars$mpg[1:20],
  transmission = factor(c(rep("auto", 10), rep("manual", 10)))
)
y_small <- mtcars$am[1:20]

try({
  bestSubset(df_categorical, y_small)
}, silent = TRUE)
cat("Categorical data correctly produced error\n")
```

**Expected outcome**:

- Non-binary `y` produces error about binary outcomes required
- `top_n > 10` produces error about maximum limit
- Categorical data produces helpful error message with conversion guidance

### 6.5. NaN/Inf Validation Tests

Test handling of NaN and Inf values in predictors:

```r
# Create dataset with NaN values
X_nan <- X
X_nan[1, 1] <- NaN

try({
  bestSubset(X_nan, y)
}, silent = TRUE)
cat("NaN values correctly detected\n")

# Create dataset with Inf values
X_inf <- X
X_inf[2, 2] <- Inf

try({
  bestSubset(X_inf, y)
}, silent = TRUE)
cat("Inf values correctly detected\n")

# Create dataset with -Inf values
X_neginf <- X
X_neginf[3, 3] <- -Inf

try({
  bestSubset(X_neginf, y)
}, silent = TRUE)
cat("-Inf values correctly detected\n")
```

**Expected outcome**:

- NaN values should be detected and handled (either error or warning)
- Inf and -Inf values should be detected and handled appropriately
- Error messages should be informative about the issue

### 6.6. Column Order Dependency Tests

Test prediction behavior with different column orders:

```r
# Train model with specific column order
X_train <- as.matrix(mtcars[1:25, c("mpg", "hp", "wt", "qsec")])
y_train <- mtcars$am[1:25]

result <- bestSubset(X_train, y_train, max_variables = 2)

# Test 1: Predictions with same column order
X_test_same <- as.matrix(mtcars[26:32, c("mpg", "hp", "wt", "qsec")])
pred_same <- predict(result, X_test_same, type = "prob")
cat("Predictions with same column order:\n")
print(pred_same)

# Test 2: Predictions with different column order (reordered columns)
X_test_reorder <- as.matrix(mtcars[26:32, c("wt", "mpg", "qsec", "hp")])
pred_reorder <- predict(result, X_test_reorder, type = "prob")
cat("Predictions with reordered columns:\n")
print(pred_reorder)

# Test 3: Verify predictions differ when columns are reordered
if (!identical(pred_same, pred_reorder)) {
  cat("WARNING: Predictions depend on column order (positional matching)\n")
  cat("Users should ensure newdata has columns in same order as training data\n")
}
```

**Expected outcome**:

- Predictions with same column order should work correctly
- Predictions with reordered columns will produce different results (current behavior)
- Test documents the positional matching behavior
- Users should be aware that column order matters

### 6.7. High-Dimensional Scalability Tests

Test warnings and limits for high-dimensional models:

```r
# Create high-dimensional dataset
set.seed(123)
X_high <- matrix(rnorm(32 * 12), nrow = 32, ncol = 12)
y_high <- rbinom(32, 1, 0.5)

# Test warning for large search space
result_warn <- bestSubset(X_high, y_high, max_variables = 16)
cat("Test with max_variables = 16 completed (should show warning)\n")

# Test hard cap at max_variables = 60
X_wide <- matrix(rnorm(100 * 65), nrow = 100, ncol = 65)
y_wide <- rbinom(100, 1, 0.5)
tryCatch({
  result_cap <- bestSubset(X_wide, y_wide, max_variables = 65)
}, error = function(e) {
  cat("Test with max_variables = 65 failed as expected (exceeds limit of 60)\n")
})

# Test warning for large search space (>10 variables)
X_medium <- matrix(rnorm(32 * 11), nrow = 32, ncol = 11)
result_medium <- bestSubset(X_medium, y_high, max_variables = 11)
cat("Test with 11 variables completed (should show computational warning)\n")
```

**Expected outcome**:

- max_variables > 15 triggers warning about large search space
- max_variables > 60 causes an error (hard computational limit)
- Models with >10 variables show computational complexity warnings
- All warnings should be informative and guide users

### 6.8. Numerical Stability Tests

Test handling of perfect separation and extreme scenarios:

```r
# Create perfectly separated data
set.seed(123)
X_sep <- matrix(rnorm(30 * 3), nrow = 30, ncol = 3)
X_sep[, 1] <- c(rep(-5, 15), rep(5, 15))  # Perfect separation on X1
y_sep <- c(rep(0, 15), rep(1, 15))

# Test perfect separation warning
result_sep <- bestSubset(X_sep, y_sep, max_variables = 2)
cat("Perfect separation test completed\n")

# Verify model still fits (coefficients may be extreme)
coeffs_sep <- coef(result_sep)
cat("Coefficients with perfect separation:\n")
print(coeffs_sep)

if (any(abs(coeffs_sep) > 10)) {
  cat("NOTE: Large coefficients detected, indicating possible separation\n")
}

# Test near-perfect separation
X_near <- X_sep
X_near[c(15, 16), 1] <- 0  # Add some overlap
result_near <- bestSubset(X_near, y_sep, max_variables = 2)
cat("Near-perfect separation test completed\n")
```

**Expected outcome**:

- Perfect separation should trigger warning message
- Model should still converge (coefficients may be large)
- Near-perfect separation should work without issues
- Warning messages should mention affected variables

### 7. Real Dataset Example (mtcars)

Test with the complete mtcars dataset to demonstrate real-world usage:

```r
# Use all numeric predictors to predict car transmission type
data(mtcars)
X_full <- as.matrix(mtcars[, c("mpg", "cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "gear", "carb")])
y_full <- mtcars$am

# Find best predictors for car transmission type
result_full <- bestSubset(X_full, y_full,
                         max_variables = 4,
                         top_n = 3,
                         metric = "auc")

cat("Results for predicting transmission type (am) from car characteristics:\n")
summary(result_full)

# Show which variables are most predictive
best_vars <- result_full$best_model$variables
var_names <- colnames(X_full)[best_vars]
cat("Most predictive variables:", paste(var_names, collapse = ", "), "\n")
```

**Expected outcome**:

- Results show which car characteristics best predict transmission type
- AUC values indicate how well the models discriminate between automatic/manual
- Coefficient signs make intuitive sense (e.g., lighter cars more likely to be manual)

## Requirements

- R version 3.5 or higher
- C++ compiler (for installation)
- `devtools` package (for GitHub installation)

## Function Reference

```r
bestSubset(X, y, max_variables = NULL, top_n = 5, metric = "auc",
           cross_validation = FALSE, cv_folds = 5, cv_repeats = 1,
           cv_seed = NULL, na.action = na.fail, n_threads = NULL)
```

**Required arguments:**

- `X`: A matrix or data frame of predictor variables (must be numeric)
- `y`: A binary outcome vector (must contain only 0 and 1)

**Optional parameters:**

- `max_variables`: The maximum number of variables to include in a model (default: use all)
- `top_n`: Number of top-performing models to return (default: 5, maximum allowed: 10)
- `metric`: Metric used to rank the models. Options are:
  - `"auc"`: Area under the ROC curve (default, higher is better)
  - `"accuracy"`: Classification accuracy (higher is better)
  - `"deviance"`: Model deviance (lower is better, measures model fit)
  - `"aic"`: Akaike Information Criterion (lower is better)
  - `"bic"`: Bayesian Information Criterion (lower is better)
- `cross_validation`: Set to `TRUE` to enable k-fold cross-validation (default: `FALSE`)
- `cv_folds`: Number of folds to use if cross-validation is enabled (default: 5)
- `cv_repeats`: Number of times to repeat cross-validation (default: 1)
- `cv_seed`: Set a random seed for reproducibility during cross-validation (default: `NULL`)
- `na.action`: How to handle missing values. Options:
  - `na.fail`: stop with an error if any values are missing (default)
  - `na.omit`: drop rows with missing values
  - `na.exclude`: drop rows, but keep row alignment for predictions
- `n_threads`: Number of threads for parallel processing (default: `NULL` for auto-detection)
  - Use `NULL` to automatically use all available cores minus 1 (recommended)
  - Use `1` for serial execution
  - Specify a positive integer for custom thread count
  - Parallel processing provides significant speedup (typically 4-5x with multiple cores)

## Output Summary

The function gives you:

- List of best models ranked by performance
- Coefficients for the best model
- Performance scores (accuracy, AUC)
- Easy-to-read summaries

## More examples

### Example 1: Limit variables and models

```r
# Only use up to 3 variables, show top 3 models
result <- bestSubset(X, y, max_variables = 3, top_n = 3)
```

### Example 2: Get coefficients

```r
result <- bestSubset(X, y)
coefficients <- coef(result)
print(coefficients)
```

### Example 3: Make predictions

```r
# Train model
result <- bestSubset(X, y)

# Predict on new data
new_X <- matrix(rnorm(20), nrow = 5, ncol = 4)
predictions <- predict(result, new_X, type = "prob")
```

### Example 4: Handle missing data

```r
# Data with missing values
X[c(1,5,10), 1] <- NA  # Make some values missing

# Option 1: Remove cases with missing data
result1 <- bestSubset(X, y, na.action = na.omit)

# Option 2: Stop if missing data found (default)
result2 <- bestSubset(X, y, na.action = na.fail)  # Will give error

# Option 3: Remove cases but keep result positions
result3 <- bestSubset(X, y, na.action = na.exclude)
```

### Example 5: Complete example with summary output

```r
# Create sample data
set.seed(123)
X <- matrix(rnorm(100), nrow = 20, ncol = 5)
y <- rbinom(20, 1, 0.5)

# Run best subset selection
result <- bestSubset(X, y, top_n = 3)

# See basic results
print(result)
# Shows: data info, settings, and best model summary

# See detailed results
summary(result)
# Shows:
# - Top 3 models with performance scores
# - Best model coefficients with names
# - Sample size information

# Extract just the coefficients
coef(result)
# (Intercept)          X1          X3
#   0.123456    0.789012   -0.345678

# Make predictions
new_data <- matrix(rnorm(25), nrow = 5, ncol = 5)
predictions <- predict(result, new_data, type = "prob")
classes <- predict(result, new_data, type = "class")
```

### Example 6: Using deviance for model selection

```r
# Create sample data
set.seed(123)
X <- matrix(rnorm(100), nrow = 50, ncol = 3)
colnames(X) <- c("X1", "X2", "X3")
y <- rbinom(50, 1, plogis(X[,1] + 0.5*X[,2]))

# Select models based on deviance (lower is better)
result <- bestSubset(X, y, metric = "deviance", top_n = 5)

# Models are ranked from lowest to highest deviance
print(result)
summary(result)

# Use deviance with cross-validation for more robust selection
result_cv <- bestSubset(X, y,
                        metric = "deviance",
                        cross_validation = TRUE,
                        cv_folds = 5,
                        top_n = 3)

# Compare with glm deviance
glm_model <- glm(y ~ X, family = binomial)
cat("GLM deviance:", glm_model$deviance, "\n")
cat("bestSelectR deviance:", result$best_model$deviance, "\n")
# Should match closely!
```

## Performance Metrics

The package uses standard metrics to evaluate classification models:

- **Accuracy**: Proportion of correctly predicted cases (range: 0–1; higher is better)
- **AUC (Area Under the ROC Curve)**: Measures how well the model distinguishes between classes
  - AUC of 0.5 means no better than random guessing
  - AUC of 1.0 means perfect separation
  - AUC > 0.7 is typically considered good in practice
- **Deviance**: Measures model fit to the data (lower is better)
  - Calculated as -2 × log-likelihood
  - Useful for comparing nested models
  - Directly comparable to R's `glm()` deviance
  - Supports both standard fitting and cross-validation
\
- **AIC (Akaike Information Criterion)**: Trade-off between goodness-of-fit and complexity
  - AIC = deviance + 2k
  - Here, k counts the number of predictors only; the intercept is not penalized
  - Lower is better
- **BIC (Bayesian Information Criterion)**: Stronger penalty for complexity on larger datasets
  - BIC = deviance + k * log(n)
  - n is the number of observations; k counts predictors only (intercept not penalized)
  - Lower is better

Notes on selection and cross-validation:
- During the subset search, only the chosen metric (and deviance) is computed for each subset to rank models; all metrics are computed only for the best model afterward.
- When `cross_validation = TRUE`, CV-averaged scores are used for ranking only for `metric = "accuracy"` or `"auc"`. For `"deviance"`, `"aic"`, and `"bic"`, ranking is based on full-data values even when CV is enabled.

## Usage Tips

1. **Limit model size for speed**:
   If your dataset has many predictors, use `max_variables` to reduce computation time.
2. **Enable cross-validation for reliability**:
   Use `cross_validation = TRUE` to get more stable performance estimates, especially with small datasets.
3. **Response variable requirements**:
   The outcome `y` must contain exactly 0 and 1 values. Other binary formats require conversion:

   ```r
   # Convert factor to 0/1
   y <- as.numeric(factor_var) - 1

   # Convert logical to 0/1
   y <- as.numeric(logical_var)

   # Convert other numeric binary to 0/1
   y <- ifelse(original_y == "success_value", 1, 0)
   ```

4. **Handle categorical predictors**:
   The package requires all predictors to be numeric. If you have categorical variables, transform them to dummy variables first using `model.matrix()`:
   ```r
   # Convert categorical data to dummy variables
   df <- data.frame(x1 = rnorm(20), category = factor(c(rep('A', 10), rep('B', 10))))
   X_processed <- model.matrix(~ . - 1, data = df)
   result <- bestSubset(X_processed, y)
   ```
5. **Handle missing data**:
   If your predictors contain missing values, either preprocess the data or set the `na.action` argument (e.g., `na.omit` or `na.exclude`).

## Documentation

For documentation see the package vignette:

```r
vignette("bestSelectR-tutorial", package = "bestSelectR")
```
