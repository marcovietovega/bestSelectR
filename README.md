# Best Subset Selection for Logistic Regression

This package helps you find the best combination of variables for predicting binary outcomes with logistic regression. It automatically tests every possible combination of predictor variables and ranks them by accuracy or AUC. It also uses cross validation to make sure the results are reliable.

## Installation

Install directly from GitHub:

```r
devtools::install_github("marcovietovega/bestSelectR")
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

### Example with Missing Data

```r
# Create data with some missing values
X_missing <- X
X_missing[sample(length(X_missing), 20)] <- NA  # Add 20 missing values randomly

# Fail if any missing values are found (default)
result_fail <- bestSubset(X_missing, y, na.action=na.fail)

print(result_fail)
```

## Requirements

- R version 3.5 or higher
- C++ compiler (for installation)
- `devtools` package (for GitHub installation)

## Main function: bestSubset()

```r
bestSubset(X, y, max_variables = NULL, top_n = 5, metric = "auc",
           cross_validation = FALSE, cv_folds = 5, cv_repeats = 1,
           cv_seed = NULL, na.action = na.fail)
```

**What you need to provide:**

- `X`: Your variables (matrix or data frame)
- `y`: Your outcome (must be 0 and 1 only)

**What you can change:**

- `max_variables`: Limit how many variables to use (default: use all)
- `top_n`: How many best models to show (default: 5, max: 10)
- `metric`: How to pick best models - "auc" or "accuracy" (default: "auc")
- `cross_validation`: Use cross-validation? TRUE or FALSE (default: FALSE)
- `cv_folds`: How many groups for cross-validation (default: 5)
- `cv_repeats`: How many times to repeat cross-validation (default: 1)
- `cv_seed`: Random number seed (default: none)
- `na.action`: What to do with missing data (default: stop and tell you)

## What you get back

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

### Example 5: Function fixes problems automatically

```r
# Ask for too many variables - function will fix it
result <- bestSubset(X, y, max_variables = 100)  # X only has 4 variables
# Warning: "max_variables (100) exceeds predictors (4). Using all 4 predictors."

# Ask for too many models to show - function will limit it
result <- bestSubset(X, y, top_n = 50)
# Warning: "top_n limited to maximum of 10 models for readable output"
```

### Example 6: Complete example with summary output

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

## Important notes

### Data requirements

- `y` must be 0 and 1 only (not TRUE/FALSE or other values)
- `X` must be numbers only
- Need at least 2 observations

### Speed

- Small datasets (≤ 10 variables): Very fast
- Medium datasets (10-15 variables): Fast
- Large datasets (> 15 variables): Slower, use `max_variables` to limit

### Missing data

- Default: Stop if any missing data found (`na.fail`) - encourages data cleaning
- Alternative: Remove missing cases (`na.omit`) - automatic cleaning
- Advanced: Remove cases but keep positions (`na.exclude`) - for predictions
- See warnings about how many cases removed

### Smart features

- Checks your data automatically and gives helpful errors
- Fixes small problems (like asking for too many variables) with warnings
- Limits output to 10 best models maximum (more is not useful)
- Warns you if computation will be slow with many variables

## What the results mean

### Performance metrics

- **Accuracy**: Percentage of correct predictions
- **AUC**: Area Under Curve (0.5 = random, 1.0 = perfect)
- **Deviance**: Lower is better (technical measure)

### Model selection

- By default, uses AUC (better for unbalanced data)
- Can switch to accuracy if preferred
- Shows top models in order of performance

## Getting help

If something doesn't work:

1. Check your data has only 0 and 1 in `y`
2. Check for missing values
3. Try with smaller `max_variables` if slow
4. Read error messages - they explain the problem
