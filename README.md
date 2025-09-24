# Best Subset Selection for Logistic Regression

**bestSelectR** helps you find the best subset of predictor variables for logistic regression models, especially when your outcome is binary (e.g., 0 or 1). It works by checking all possible combinations of your variables and ranks them based on how well they predict the outcome.

This is useful when you're not sure which variables are most important. Instead of using stepwise selection or picking variables manually, **bestSelectR** tests every subset and finds the combinations that perform best. It also includes cross-validation to help avoid overfitting and give more reliable performance estimates.

## Table of Contents

- [Installation](#installation)
- [Test Functions](#test-functions)
  - [Basic Example (Without Cross-Validation)](#basic-example-without-cross-validation)
  - [Example with Cross-Validation](#example-with-cross-validation)
  - [Example with Missing Data](#example-with-missing-data)
- [Test Plan for bestSelectR](#test-plan-for-bestselectr)
  - [1. Setup and Installation](#1-setup-and-installation)
  - [2. Basic Functionality](#2-basic-functionality)
  - [3. Cross-Validation Functionality](#3-cross-validation-functionality)
  - [4. Missing Data Handling](#4-missing-data-handling)
  - [5. Prediction Functionality](#5-prediction-functionality)
  - [6. Error and Edge Cases](#6-error-and-edge-cases)
  - [7. Real Dataset Example (mtcars)](#7-real-dataset-example-mtcars)
  - [8. Report Instructions](#8-report-instructions)
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
devtools::install_github("marcovietovega/bestSelectR", build_vignettes = TRUE)
```

You'll need `devtools` for installation. All other dependencies will be installed automatically.

## Test Functions

### Basic Example (Without Cross-Validation)

```r
library(bestSelectR)

# Create sample data
set.seed(123)
X <- matrix(rnorm(50*4), nrow=50, ncol=4)
colnames(X) <- paste0("X", 1:4)
y <- rbinom(50, 1, plogis(X[,1] + 0.5*X[,2] - 0.3*X[,3]))

# Find best variable combinations
result <- bestSubset(X, y, max_variables=3, top_n=5)

# See the results
print(result)
summary(result)
```

This example creates a dataset with 4 predictors and tests all subsets of up to 3 variables, ranking models by AUC.

### Example with Cross-Validation

```r
# Use cross-validation for more reliable results
result_cv <- bestSubset(X, y,
                        max_variables=3,
                        cross_validation=TRUE,
                        cv_folds=5,
                        metric="auc")

# View results
print(result_cv)
summary(result_cv)
```

Cross-validation splits the data into k folds and evaluates models on held-out data, providing more reliable performance estimates.

### Example with Missing Data

```r
# Create data with some missing values
X_missing <- X
X_missing[sample(length(X_missing), 20)] <- NA  # Add 20 missing values randomly

# Handle missing data with listwise deletion
result_omit <- bestSubset(X_missing, y, na.action=na.omit)

print(result_omit)
```

The package supports multiple missing data strategies: `na.fail` (default), `na.omit`, and `na.exclude`.

## Test Plan for bestSelectR

This section provides a test plan for **bestSelectR** that allows reviewers to verify the package's functionality step-by-step.

### 1. Setup and Installation

1. Install `devtools` if not already installed:

```r
install.packages("devtools")
```

2. Install the package from GitHub:

```r
devtools::install_github("marcovietovega/bestSelectR", build_vignettes = TRUE)
```

3. Load the package:

```r
library(bestSelectR)
```

**Expected outcome**: Package loads without errors. Vignette should be available:

```r
vignette("bestSelectR-tutorial", package = "bestSelectR")
```

### 2. Basic Functionality

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

### 7. Real Dataset Example (mtcars)

Test with the complete mtcars dataset to demonstrate real-world usage:

```r
# Use all numeric predictors to predict transmission type
data(mtcars)
X_full <- as.matrix(mtcars[, c("mpg", "cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "gear", "carb")])
y_full <- mtcars$am

# Find best predictors for transmission type
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
           cv_seed = NULL, na.action = na.fail)
```

**Required arguments:**

- `X`: A matrix or data frame of predictor variables (must be numeric)
- `y`: A binary outcome vector (must contain only 0 and 1)

**Optional parameters:**

- `max_variables`: The maximum number of variables to include in a model (default: use all)
- `top_n`: Number of top-performing models to return (default: 5, maximum allowed: 10)
- `metric`: Metric used to rank the models. Options are:
  - `"auc"`: Area under the ROC curve (default)
  - `"accuracy"`: Classification accuracy
- `cross_validation`: Set to `TRUE` to enable k-fold cross-validation (default: `FALSE`)
- `cv_folds`: Number of folds to use if cross-validation is enabled (default: 5)
- `cv_repeats`: Number of times to repeat cross-validation (default: 1)
- `cv_seed`: Set a random seed for reproducibility during cross-validation (default: `NULL`)
- `na.action`: How to handle missing values. Options:
  - `na.fail`: stop with an error if any values are missing (default)
  - `na.omit`: drop rows with missing values
  - `na.exclude`: drop rows, but keep row alignment for predictions

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

## Performance Metrics

The package uses standard metrics to evaluate classification models:

- **Accuracy**: Proportion of correctly predicted cases (range: 0–1; higher is better)
- **AUC (Area Under the ROC Curve)**: Measures how well the model distinguishes between classes
  - AUC of 0.5 means no better than random guessing
  - AUC of 1.0 means perfect separation
  - AUC > 0.7 is typically considered good in practice

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
