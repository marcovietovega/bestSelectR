#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <bitset>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <Rcpp.h>
#include "BestSubsetSelector.hpp"

using namespace std;

// Constructor
BestSubsetSelector::BestSubsetSelector(const MatrixXd &X_input, const VectorXd &y_input)
    : X(X_input), y(y_input), include_intercept(true), max_variables(-1),
      top_n_models(10), metric("accuracy"), convergence_tolerance(1e-6),
      max_iterations(100), use_cross_validation(false), cv_folds(5),
      cv_repeats(3), cv_seed(-1), is_fitted(false)
{
    if (X.rows() != y.rows())
    {
        throw std::invalid_argument("X and y must have the same number of rows");
    }

    n_observations = X.rows();
    n_variables = X.cols();

    // Default: consider all variables
    if (max_variables < 0)
    {
        max_variables = n_variables;
    }
}

// Constructor with intercept option
BestSubsetSelector::BestSubsetSelector(const MatrixXd &X_input, const VectorXd &y_input,
                                       bool add_intercept)
    : BestSubsetSelector(X_input, y_input)
{
    include_intercept = add_intercept;
}

// Generate all possible variable subsets using bit manipulation
std::vector<std::vector<int>> BestSubsetSelector::generateVariableSubsets()
{
    return generateVariableSubsets(max_variables);
}

std::vector<std::vector<int>> BestSubsetSelector::generateVariableSubsets(int max_size)
{
    std::vector<std::vector<int>> subsets;
    int total_vars = n_variables;
    int max_subset_size = std::min(max_size, total_vars);

    // Generate all possible combinations
    // Start from 1 (skip empty set) to 2^n_variables - 1
    for (int i = 1; i < (1 << total_vars); ++i)
    {
        std::vector<int> subset;

        for (int j = 0; j < total_vars; ++j)
        {
            if (i & (1 << j))
            {
                subset.push_back(j);
            }
        }

        // Only include subsets within size limit
        if (subset.size() <= static_cast<size_t>(max_subset_size))
        {
            subsets.push_back(subset);
        }
    }

    return subsets;
}

// Extract subset of X matrix for given variable indices
MatrixXd BestSubsetSelector::extractSubsetMatrix(const std::vector<int> &variable_indices)
{
    if (variable_indices.empty())
    {
        throw std::invalid_argument("Variable indices cannot be empty");
    }

    std::vector<int> final_indices = include_intercept ? addInterceptColumn(variable_indices) : variable_indices;

    MatrixXd X_subset(n_observations, final_indices.size());

    for (size_t i = 0; i < final_indices.size(); ++i)
    {
        if (final_indices[i] == -1)
        {
            // Intercept column
            X_subset.col(i) = VectorXd::Ones(n_observations);
        }
        else
        {
            X_subset.col(i) = X.col(final_indices[i]);
        }
    }

    return X_subset;
}

// Add intercept column index (-1) to the beginning
std::vector<int> BestSubsetSelector::addInterceptColumn(const std::vector<int> &variable_indices)
{
    std::vector<int> with_intercept;
    with_intercept.push_back(-1); // -1 represents intercept
    with_intercept.insert(with_intercept.end(), variable_indices.begin(), variable_indices.end());
    return with_intercept;
}

// Main fitting method
void BestSubsetSelector::fit()
{
    fit(max_variables, top_n_models, metric);
}

void BestSubsetSelector::fit(int max_vars)
{
    fit(max_vars, top_n_models, metric);
}

void BestSubsetSelector::fit(int max_vars, int top_n)
{
    fit(max_vars, top_n, metric);
}

void BestSubsetSelector::fit(int max_vars, int top_n, const std::string &selection_metric)
{
    max_variables = max_vars;
    top_n_models = top_n;
    metric = selection_metric;

    // Use cross-validation if enabled
    if (use_cross_validation)
    {
        fitWithCrossValidation(max_variables);
        return;
    }

    // Generate all variable subsets
    std::vector<std::vector<int>> subsets = generateVariableSubsets(max_variables);
    int n_subsets = subsets.size();

    // Pre-allocate results for thread safety
    std::vector<SubsetResult> parallel_results(n_subsets);
    std::vector<bool> success_flags(n_subsets, false);

    // Progress reporting setup
    std::atomic<int> completed(0);
    int report_interval = std::max(1, n_subsets / 100);  // Report every 1%

    // Parallel loop over subsets with OpenMP
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 10)
    #endif
    for (int idx = 0; idx < n_subsets; ++idx)
    {
        try
        {
            const auto &subset = subsets[idx];

            // Extract subset matrix
            MatrixXd X_subset = extractSubsetMatrix(subset);

            // Fit logistic regression
            LogisticRegression lr(X_subset, y, max_iterations, convergence_tolerance);
            lr.fit();

            // Create Model wrapper
            Model model;
            std::vector<int> indices_for_model = include_intercept ? addInterceptColumn(subset) : subset;
            model.fitFromLogisticRegression(lr, indices_for_model);

            // Calculate performance metrics
            VectorXd fitted_probs = lr.get_fitted_values();
            VectorXi predictions = lr.predict(X_subset);

            double accuracy = PerformanceEvaluator::calculateAccuracy(predictions, y);
            double auc = PerformanceEvaluator::calculateAUC(fitted_probs, y);

            // Calculate AIC and BIC
            double deviance = model.getDeviance();
            int n_params = X_subset.cols();
            double aic_value = PerformanceEvaluator::calculateAIC(deviance, n_params);
            double bic_value = PerformanceEvaluator::calculateBIC(deviance, n_params, n_observations);

            // Create SubsetResult and store at unique index (thread-safe)
            parallel_results[idx] = SubsetResult(model, accuracy, auc, aic_value, bic_value);
            success_flags[idx] = true;
        }
        catch (const std::exception &e)
        {
            // Continue with other models if one fails
            // success_flags[idx] remains false
        }

        // Update progress
        int current = ++completed;
        if (current % report_interval == 0 || current == n_subsets)
        {
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            {
                Rcpp::Rcout << "\rEvaluated " << current << " / " << n_subsets
                            << " models (" << (100 * current / n_subsets) << "%)"
                            << std::flush;
            }
        }
    }

    // Print completion message
    Rcpp::Rcout << "\rEvaluated " << n_subsets << " / " << n_subsets
                << " models (100%)    \n" << std::flush;

    // Collect successful results (sequential section)
    all_results.clear();
    for (int idx = 0; idx < n_subsets; ++idx)
    {
        if (success_flags[idx])
        {
            all_results.push_back(parallel_results[idx]);
        }
    }

    // Sort results by selected metric
    sortResultsByMetric();

    // Keep top N results
    best_results.clear();
    int n_to_keep = std::min(top_n_models, static_cast<int>(all_results.size()));

    for (int i = 0; i < n_to_keep; ++i)
    {
        best_results.push_back(all_results[i]);
    }

    is_fitted = true;
}

// Sort results by the selected metric
void BestSubsetSelector::sortResultsByMetric()
{
    if (metric == "accuracy" || metric == "auc")
    {
        // Higher is better - sort in descending order
        std::sort(all_results.begin(), all_results.end(),
                  [this](const SubsetResult &a, const SubsetResult &b)
                  {
                      return a.getScore(metric) > b.getScore(metric);
                  });
    }
    else if (metric == "deviance" || metric == "aic" || metric == "bic")
    {
        // Lower is better - sort in ascending order
        std::sort(all_results.begin(), all_results.end(),
                  [this](const SubsetResult &a, const SubsetResult &b)
                  {
                      return a.getScore(metric) < b.getScore(metric);
                  });
    }
    else
    {
        throw std::invalid_argument("Unknown metric: " + metric);
    }
}

// Configuration methods
void BestSubsetSelector::setMaxVariables(int max_vars)
{
    max_variables = max_vars;
}

void BestSubsetSelector::setTopNModels(int top_n)
{
    top_n_models = top_n;
}

void BestSubsetSelector::setMetric(const std::string &selection_metric)
{
    if (selection_metric != "accuracy" && selection_metric != "auc" &&
        selection_metric != "deviance" && selection_metric != "aic" &&
        selection_metric != "bic")
    {
        throw std::invalid_argument("Metric must be 'accuracy', 'auc', 'deviance', 'aic', or 'bic'");
    }
    metric = selection_metric;
}

void BestSubsetSelector::setConvergenceParameters(double tolerance, int max_iter)
{
    convergence_tolerance = tolerance;
    max_iterations = max_iter;
}

void BestSubsetSelector::setIncludeIntercept(bool include)
{
    include_intercept = include;
}

// Results access
std::vector<SubsetResult> BestSubsetSelector::getBestResults() const
{
    if (!is_fitted)
    {
        throw std::runtime_error("Model has not been fitted yet");
    }
    return best_results;
}

std::vector<SubsetResult> BestSubsetSelector::getAllResults() const
{
    if (!is_fitted)
    {
        throw std::runtime_error("Model has not been fitted yet");
    }
    return all_results;
}

SubsetResult BestSubsetSelector::getBestModel() const
{
    if (!is_fitted)
    {
        throw std::runtime_error("Model has not been fitted yet");
    }
    if (best_results.empty())
    {
        throw std::runtime_error("No valid models found");
    }
    return best_results[0];
}

SubsetResult BestSubsetSelector::getBestModel(const std::string &selection_metric) const
{
    if (!is_fitted)
    {
        throw std::runtime_error("Model has not been fitted yet");
    }

    auto results_copy = all_results;

    if (selection_metric == "accuracy" || selection_metric == "auc")
    {
        std::sort(results_copy.begin(), results_copy.end(),
                  [&selection_metric](const SubsetResult &a, const SubsetResult &b)
                  {
                      return a.getScore(selection_metric) > b.getScore(selection_metric);
                  });
    }
    else if (selection_metric == "deviance" || selection_metric == "aic" ||
             selection_metric == "bic")
    {
        std::sort(results_copy.begin(), results_copy.end(),
                  [&selection_metric](const SubsetResult &a, const SubsetResult &b)
                  {
                      return a.getScore(selection_metric) < b.getScore(selection_metric);
                  });
    }

    return results_copy[0];
}

// Model statistics
int BestSubsetSelector::getNumModelsEvaluated() const
{
    return all_results.size();
}

int BestSubsetSelector::getNumVariables() const
{
    return n_variables;
}

int BestSubsetSelector::getNumObservations() const
{
    return n_observations;
}

// Status
bool BestSubsetSelector::isFitted() const
{
    return is_fitted;
}

// Prediction using best model
VectorXd BestSubsetSelector::predict_proba(const MatrixXd &X_new) const
{
    SubsetResult best = getBestModel();

    // Need to extract the appropriate columns from X_new
    std::vector<int> var_indices = best.getVariableIndices();
    MatrixXd X_subset(X_new.rows(), var_indices.size());

    for (size_t i = 0; i < var_indices.size(); ++i)
    {
        if (var_indices[i] == -1)
        {
            // Intercept column
            X_subset.col(i) = VectorXd::Ones(X_new.rows());
        }
        else
        {
            X_subset.col(i) = X_new.col(var_indices[i]);
        }
    }

    return best.predict_proba(X_subset);
}

VectorXi BestSubsetSelector::predict(const MatrixXd &X_new) const
{
    SubsetResult best = getBestModel();

    // Need to extract the appropriate columns from X_new
    std::vector<int> var_indices = best.getVariableIndices();
    MatrixXd X_subset(X_new.rows(), var_indices.size());

    for (size_t i = 0; i < var_indices.size(); ++i)
    {
        if (var_indices[i] == -1)
        {
            // Intercept column
            X_subset.col(i) = VectorXd::Ones(X_new.rows());
        }
        else
        {
            X_subset.col(i) = X_new.col(var_indices[i]);
        }
    }

    return best.predict(X_subset);
}

// Cross-validation configuration methods
void BestSubsetSelector::setCrossValidation(bool use_cv, int folds, int repeats, int seed)
{
    use_cross_validation = use_cv;
    cv_folds = folds;
    cv_repeats = repeats;
    cv_seed = seed;
}

bool BestSubsetSelector::isUsingCrossValidation() const
{
    return use_cross_validation;
}

int BestSubsetSelector::getCVFolds() const
{
    return cv_folds;
}

int BestSubsetSelector::getCVRepeats() const
{
    return cv_repeats;
}

int BestSubsetSelector::getCVSeed() const
{
    return cv_seed;
}

// Cross-validation enabled fit method
void BestSubsetSelector::fitWithCrossValidation(int max_vars)
{
    std::vector<std::vector<int>> subsets = generateVariableSubsets(max_vars);
    int n_subsets = subsets.size();

    // Pre-allocate results for thread safety
    std::vector<SubsetResult> parallel_results(n_subsets);
    std::vector<bool> success_flags(n_subsets, false);

    // Progress reporting setup
    std::atomic<int> completed(0);
    int report_interval = std::max(1, n_subsets / 100);  // Report every 1%

    // Parallel loop over subsets with OpenMP
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 10)
    #endif
    for (int idx = 0; idx < n_subsets; ++idx)
    {
        const auto &subset = subsets[idx];

        if (subset.empty())
            continue;

        try
        {
            // Calculate CV performance
            double cv_performance;

            if (cv_repeats > 1)
            {
                cv_performance = PerformanceEvaluator::calculateRepeatedKFold(
                    X, y, subset, metric, include_intercept, cv_folds, cv_repeats, cv_seed);
            }
            else
            {
                cv_performance = PerformanceEvaluator::calculateKFold(
                    X, y, subset, metric, include_intercept, cv_folds, cv_seed);
            }

            // Extract subset matrix for final model fitting
            MatrixXd X_subset = extractSubsetMatrix(subset);

            // Fit final model on full data for storage and prediction
            LogisticRegression lr(X_subset, y, max_iterations, convergence_tolerance);
            lr.fit();

            Model model;
            std::vector<int> indices_for_model = include_intercept ? addInterceptColumn(subset) : subset;
            model.fitFromLogisticRegression(lr, indices_for_model);

            // Calculate final model metrics (only computed once per subset)
            VectorXd fitted_probs = lr.get_fitted_values();
            VectorXi predictions = lr.predict(X_subset);

            double final_accuracy = 0.0, final_auc = 0.0; // Initialize to avoid warnings

            if (metric == "accuracy")
            {
                // Use CV accuracy for ranking, calculate AUC only once on final model
                final_accuracy = cv_performance;                                 // CV accuracy for ranking
                final_auc = PerformanceEvaluator::calculateAUC(fitted_probs, y); // Final model AUC
            }
            else if (metric == "auc")
            {
                // Use CV AUC for ranking, calculate accuracy only once on final model
                final_accuracy = PerformanceEvaluator::calculateAccuracy(predictions, y); // Final model accuracy
                final_auc = cv_performance;                                               // CV AUC for ranking
            }
            else if (metric == "deviance")
            {
                // Use CV deviance for ranking, calculate accuracy and AUC on final model
                final_accuracy = PerformanceEvaluator::calculateAccuracy(predictions, y); // Final model accuracy
                final_auc = PerformanceEvaluator::calculateAUC(fitted_probs, y);          // Final model AUC
                // Note: CV deviance is stored in cv_performance but not used here
                // SubsetResult will get deviance from model.getDeviance()
            }
            else if (metric == "aic" || metric == "bic")
            {
                // Use CV AIC/BIC for ranking, calculate accuracy and AUC on final model
                final_accuracy = PerformanceEvaluator::calculateAccuracy(predictions, y);
                final_auc = PerformanceEvaluator::calculateAUC(fitted_probs, y);
                // Note: CV AIC/BIC is stored in cv_performance but not used here
                // SubsetResult will get AIC/BIC from calculations below
            }
            else
            {
                throw std::invalid_argument("Unsupported metric: " + metric);
            }

            // Calculate AIC and BIC
            double deviance = model.getDeviance();
            int n_params = indices_for_model.size();
            double aic_value = PerformanceEvaluator::calculateAIC(deviance, n_params);
            double bic_value = PerformanceEvaluator::calculateBIC(deviance, n_params, n_observations);

            // Create SubsetResult and store at unique index (thread-safe)
            parallel_results[idx] = SubsetResult(model, final_accuracy, final_auc, aic_value, bic_value);
            success_flags[idx] = true;
        }
        catch (const std::exception &e)
        {
            // Skip failed subsets
            // success_flags[idx] remains false
            continue;
        }

        // Update progress
        int current = ++completed;
        if (current % report_interval == 0 || current == n_subsets)
        {
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            {
                Rcpp::Rcout << "\rEvaluated " << current << " / " << n_subsets
                            << " models (" << (100 * current / n_subsets) << "%)"
                            << std::flush;
            }
        }
    }

    // Print completion message
    Rcpp::Rcout << "\rEvaluated " << n_subsets << " / " << n_subsets
                << " models (100%)    \n" << std::flush;

    // Collect successful results (sequential section)
    all_results.clear();
    for (int idx = 0; idx < n_subsets; ++idx)
    {
        if (success_flags[idx])
        {
            all_results.push_back(parallel_results[idx]);
        }
    }

    // Sort results by selected metric
    sortResultsByMetric();

    // Keep top N results
    best_results.clear();
    int n_to_keep = std::min(top_n_models, static_cast<int>(all_results.size()));

    for (int i = 0; i < n_to_keep; ++i)
    {
        best_results.push_back(all_results[i]);
    }

    is_fitted = true;
}
