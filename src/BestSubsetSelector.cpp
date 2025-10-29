#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <bitset>
#include <cstdint>
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

// Helper function to generate the next combination in lexicographic order
// Returns false when no more combinations exist
bool next_combination(std::vector<int>& indices, int n, int k) {
    // Find rightmost element that can be incremented
    for (int i = k - 1; i >= 0; --i) {
        if (indices[i] < n - k + i) {
            indices[i]++;
            // Reset all elements to the right
            for (int j = i + 1; j < k; ++j) {
                indices[j] = indices[j - 1] + 1;
            }
            return true;
        }
    }
    return false;
}

std::vector<std::vector<int>> BestSubsetSelector::generateVariableSubsets(int max_size)
{
    std::vector<std::vector<int>> subsets;
    size_t total_vars = static_cast<size_t>(n_variables);
    size_t max_subset_size = std::min(static_cast<size_t>(max_size), total_vars);

    // Generate combinations of each size from 1 to max_subset_size
    // This is much more efficient than enumerating all 2^n subsets
    for (size_t k = 1; k <= max_subset_size; ++k) {
        // Initialize first combination: [0, 1, 2, ..., k-1]
        std::vector<int> indices(k);
        for (size_t i = 0; i < k; ++i) {
            indices[i] = static_cast<int>(i);
        }

        // Add first combination
        subsets.push_back(indices);

        // Generate remaining combinations of size k
        while (next_combination(indices, static_cast<int>(total_vars), static_cast<int>(k))) {
            subsets.push_back(indices);
        }
    }

    return subsets;
}

// Generate subsets in Gray-code order to improve warm-start locality in serial runs
// Gray code is only beneficial when searching all 2^n subsets (small n, large max_variables)
// For large n with small max_variables, use combinatorial generation instead
std::vector<std::vector<int>> BestSubsetSelector::generateVariableSubsetsGrayCode(int max_size)
{
    std::vector<std::vector<int>> subsets;
    size_t total_vars = static_cast<size_t>(n_variables);
    size_t max_subset_size = std::min(static_cast<size_t>(max_size), total_vars);

    if (total_vars == 0)
        return subsets;

    // Calculate how many subsets we'll actually generate
    size_t num_subsets_to_generate = 0;
    for (size_t k = 1; k <= max_subset_size; ++k) {
        // Use binomial coefficient formula carefully to avoid overflow
        double binom = 1.0;
        for (size_t i = 0; i < k; ++i) {
            binom *= static_cast<double>(total_vars - i) / static_cast<double>(i + 1);
        }
        num_subsets_to_generate += static_cast<size_t>(binom);
    }

    // If we're generating a large fraction of all subsets (>50%) and total_vars <= 30,
    // use Gray code for better cache locality
    // Otherwise, use combinatorial generation for efficiency
    const size_t total_possible_subsets = (total_vars <= 62) ? ((1ULL << total_vars) - 1) : SIZE_MAX;

    if (total_vars <= 30 && num_subsets_to_generate > (total_possible_subsets / 2)) {
        // Use Gray code enumeration for small p when we're searching most subsets
        const std::uint64_t total_codes = (1ULL << total_vars);
        subsets.reserve(num_subsets_to_generate);

        for (std::uint64_t i = 1; i < total_codes; ++i)
        {
            std::uint64_t gray = i ^ (i >> 1);
            std::vector<int> subset;
            subset.reserve(max_subset_size);

            for (size_t j = 0; j < total_vars; ++j)
            {
                if (gray & (1ULL << j))
                {
                    subset.push_back(static_cast<int>(j));
                    if (subset.size() > max_subset_size)
                    {
                        break;
                    }
                }
            }

            if (!subset.empty() && subset.size() <= max_subset_size)
            {
                subsets.push_back(std::move(subset));
            }
        }
    } else {
        // Use combinatorial generation for large p or small max_variables
        return generateVariableSubsets(max_size);
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

    // Branch-and-Bound disabled: weak lower bounds make it slower than parallel enumeration.
    // Exhaustive parallel search is faster and simpler for the typical use case.
    // Future work: implement better bounds (e.g., null deviance, partial fits).
    // if ((metric == "aic" || metric == "bic") && n_variables >= 25)
    // {
    //     fitBranchAndBound();
    //     return;
    // }

    // Generate all variable subsets. Use Gray-code order in serial to improve warm-start reuse.
    std::vector<std::vector<int>> subsets = (n_threads_configured == 1)
                                                ? generateVariableSubsetsGrayCode(max_variables)
                                                : generateVariableSubsets(max_variables);
    size_t n_subsets = subsets.size();

    if (n_threads_configured == 1)
    {
        // Serial path with warm starts
        size_t completed = 0;
        // Report progress less often: every 2000 models (or 2% for very large jobs)
        size_t report_interval = std::max(static_cast<size_t>(2000), n_subsets / 50);

        all_results.clear();
        all_results.reserve(n_subsets);

        // Previous fit info for warm start
        std::vector<int> last_var_indices;
        VectorXd last_coeffs;
        bool has_last = false;

        // Preallocate subset buffer to reduce allocations and copies
        int max_cols = std::min(max_variables, n_variables) + (include_intercept ? 1 : 0);
        MatrixXd X_buf(n_observations, max_cols);

        for (size_t idx = 0; idx < n_subsets; ++idx)
        {
            try
            {
                const auto &subset = subsets[idx];
                // Fill X_buf with intercept (optional) and columns for this subset
                int m = 0;
                if (include_intercept)
                {
                    X_buf.col(0).setOnes();
                    m = 1;
                }
                for (int j : subset)
                {
                    X_buf.col(m++) = X.col(j);
                }
                Eigen::Ref<const MatrixXd> X_view = X_buf.leftCols(m);

                LogisticRegression lr(X_view, y, max_iterations, convergence_tolerance);

                // Build indices including intercept for mapping
                std::vector<int> indices_for_model = include_intercept ? addInterceptColumn(subset) : subset;

                if (has_last)
                {
                    int m = static_cast<int>(indices_for_model.size());
                    VectorXd beta0 = VectorXd::Zero(m);

                    // Map previous coefficients by matching variable indices (including intercept marker -1)
                    for (int new_pos = 0; new_pos < m; ++new_pos)
                    {
                        int var_id = indices_for_model[new_pos];
                        for (int old_pos = 0; old_pos < static_cast<int>(last_var_indices.size()); ++old_pos)
                        {
                            if (last_var_indices[old_pos] == var_id)
                            {
                                beta0(new_pos) = last_coeffs(old_pos);
                                break;
                            }
                        }
                    }

                    VectorXd eta0 = X_view * beta0;
                    lr.setInitialGuess(beta0, eta0);
                }

                lr.fit();

                // Create Model wrapper
                Model model;
                model.fitFromLogisticRegression(lr, indices_for_model);

                // Cache for next warm start
                last_var_indices = indices_for_model;
                last_coeffs = lr.getCoefficients();
                has_last = true;

                // Calculate only the metrics needed for selection
                double accuracy = NA_REAL;
                double auc = NA_REAL;
                double aic_value = NA_REAL;
                double bic_value = NA_REAL;

                double deviance = model.getDeviance();
                int n_params = X_view.cols();

                if (metric == "accuracy")
                {
                    VectorXi predictions = lr.predict(X_view);
                    accuracy = PerformanceEvaluator::calculateAccuracy(predictions, y);
                }
                else if (metric == "auc")
                {
                    VectorXd fitted_probs = lr.get_fitted_values();
                    auc = PerformanceEvaluator::calculateAUC(fitted_probs, y);
                }
                else if (metric == "aic")
                {
                    aic_value = PerformanceEvaluator::calculateAIC(deviance, n_params);
                }
                else if (metric == "bic")
                {
                    bic_value = PerformanceEvaluator::calculateBIC(deviance, n_params, n_observations);
                }

                all_results.emplace_back(model, accuracy, auc, aic_value, bic_value);
            }
            catch (const std::exception &e)
            {
                // skip
            }

            completed++;
            if (completed % report_interval == 0 || completed == n_subsets)
            {
                int pct = static_cast<int>((100.0 * completed) / n_subsets);
                Rcpp::Rcout << "\rEvaluated " << completed << " / " << n_subsets
                            << " models (" << pct << "%)" << std::flush;
            }
        }

        Rcpp::Rcout << "\rEvaluated " << n_subsets << " / " << n_subsets
                    << " models (100%)    \n"
                    << std::flush;
    }
    else
    {
        // Parallel path (no warm start) with per-thread subset buffers
        std::vector<SubsetResult> parallel_results(n_subsets);
        std::vector<bool> success_flags(n_subsets, false);

        std::atomic<size_t> completed(0);
        // Report progress less often: every 2000 models (or 2% for very large jobs)
        size_t report_interval = std::max(static_cast<size_t>(2000), n_subsets / 50);

        int max_cols = std::min(max_variables, n_variables) + (include_intercept ? 1 : 0);

#ifdef _OPENMP
#pragma omp parallel
        {
            MatrixXd X_buf(n_observations, max_cols);

#pragma omp for schedule(runtime)
            for (size_t idx = 0; idx < n_subsets; ++idx)
            {
                try
                {
                    const auto &subset = subsets[idx];

                    // Build X_view in thread-local buffer
                    int m = 0;
                    if (include_intercept)
                    {
                        X_buf.col(0).setOnes();
                        m = 1;
                    }
                    for (int j : subset)
                    {
                        X_buf.col(m++) = X.col(j);
                    }
                    Eigen::Ref<const MatrixXd> X_view = X_buf.leftCols(m);

                    // Fit logistic regression
                    LogisticRegression lr(X_view, y, max_iterations, convergence_tolerance);
                    lr.fit();

                    // Create Model wrapper
                    Model model;
                    std::vector<int> indices_for_model = include_intercept ? addInterceptColumn(subset) : subset;
                    model.fitFromLogisticRegression(lr, indices_for_model);

                    // Calculate only the metrics needed for selection (optimize performance)
                    double accuracy = NA_REAL;
                    double auc = NA_REAL;
                    double aic_value = NA_REAL;
                    double bic_value = NA_REAL;

                    // Always calculate deviance (needed for AIC/BIC and is cheap)
                    double deviance = model.getDeviance();
                    int n_params = X_view.cols();

                    // Calculate based on selected metric
                    if (metric == "accuracy")
                    {
                        VectorXi predictions = lr.predict(X_view);
                        accuracy = PerformanceEvaluator::calculateAccuracy(predictions, y);
                    }
                    else if (metric == "auc")
                    {
                        VectorXd fitted_probs = lr.get_fitted_values();
                        auc = PerformanceEvaluator::calculateAUC(fitted_probs, y);
                    }
                    else if (metric == "aic")
                    {
                        aic_value = PerformanceEvaluator::calculateAIC(deviance, n_params);
                    }
                    else if (metric == "bic")
                    {
                        bic_value = PerformanceEvaluator::calculateBIC(deviance, n_params, n_observations);
                    }
                    // For "deviance" metric, deviance is already calculated

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
                size_t current = ++completed;
                if (current % report_interval == 0 || current == n_subsets)
                {
#pragma omp critical
                    {
                        int pct = static_cast<int>((100.0 * current) / n_subsets);
                        Rcpp::Rcout << "\rEvaluated " << current << " / " << n_subsets
                                    << " models (" << pct << "%)"
                                    << std::flush;
                    }
                }
            }
        }
#else
        // Fallback without OpenMP: keep prior behavior
        for (size_t idx = 0; idx < n_subsets; ++idx)
        {
            try
            {
                const auto &subset = subsets[idx];
                MatrixXd X_subset = extractSubsetMatrix(subset);
                LogisticRegression lr(X_subset, y, max_iterations, convergence_tolerance);
                lr.fit();
                Model model;
                std::vector<int> indices_for_model = include_intercept ? addInterceptColumn(subset) : subset;
                model.fitFromLogisticRegression(lr, indices_for_model);
                double accuracy = NA_REAL, auc = NA_REAL, aic_value = NA_REAL, bic_value = NA_REAL;
                double deviance = model.getDeviance();
                int n_params = X_subset.cols();
                if (metric == "accuracy")
                {
                    VectorXi predictions = lr.predict(X_subset);
                    accuracy = PerformanceEvaluator::calculateAccuracy(predictions, y);
                }
                else if (metric == "auc")
                {
                    VectorXd fitted_probs = lr.get_fitted_values();
                    auc = PerformanceEvaluator::calculateAUC(fitted_probs, y);
                }
                else if (metric == "aic")
                {
                    aic_value = PerformanceEvaluator::calculateAIC(deviance, n_params);
                }
                else if (metric == "bic")
                {
                    bic_value = PerformanceEvaluator::calculateBIC(deviance, n_params, n_observations);
                }
                parallel_results[idx] = SubsetResult(model, accuracy, auc, aic_value, bic_value);
                success_flags[idx] = true;
            }
            catch (...)
            {
            }
            size_t current = ++completed;
            if (current % report_interval == 0 || current == n_subsets)
            {
                int pct = static_cast<int>((100.0 * current) / n_subsets);
                Rcpp::Rcout << "\rEvaluated " << current << " / " << n_subsets
                            << " models (" << pct << "%)"
                            << std::flush;
            }
        }
#endif

        // Print completion message
        Rcpp::Rcout << "\rEvaluated " << n_subsets << " / " << n_subsets
                    << " models (100%)    \n"
                    << std::flush;

        // Collect successful results (sequential section)
        all_results.clear();
        for (size_t idx = 0; idx < n_subsets; ++idx)
        {
            if (success_flags[idx])
            {
                all_results.push_back(parallel_results[idx]);
            }
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

    // Calculate all metrics for all top_n models (for complete reporting)
    // This ensures all top models have complete info without refitting
    for (int i = 0; i < static_cast<int>(best_results.size()); ++i)
    {
        SubsetResult &result = best_results[i];
        Model model = result.getModel();
        std::vector<int> var_indices = model.getVariableIndices();

        // Remove intercept marker (-1) from indices to get just predictor indices
        std::vector<int> predictor_indices;
        for (int idx : var_indices)
        {
            if (idx != -1)
            {
                predictor_indices.push_back(idx);
            }
        }

        // Extract subset matrix (this will add intercept if needed)
        MatrixXd X_subset = extractSubsetMatrix(predictor_indices);

        // Use existing model to calculate all metrics (don't refit!)
        VectorXi predictions = model.predict(X_subset);
        double accuracy = PerformanceEvaluator::calculateAccuracy(predictions, y);

        VectorXd fitted_probs = model.predict_proba(X_subset);
        double auc = PerformanceEvaluator::calculateAUC(fitted_probs, y);

        // Get existing deviance and calculate AIC/BIC
        double deviance = model.getDeviance();
        int n_params = X_subset.cols();
        double aic_value = PerformanceEvaluator::calculateAIC(deviance, n_params);
        double bic_value = PerformanceEvaluator::calculateBIC(deviance, n_params, n_observations);

        // Update result with all metrics (preserving the same model)
        best_results[i] = SubsetResult(model, accuracy, auc, aic_value, bic_value);
    }

    is_fitted = true;
}

// Branch-and-Bound fit for AIC/BIC
void BestSubsetSelector::fitBranchAndBound()
{
    // Reset containers
    all_results.clear();
    best_results.clear();
    is_fitted = false;

    // Initialize best score to +inf (we minimize AIC/BIC)
    best_score_so_far = std::numeric_limits<double>::infinity();

    // Start DFS from empty subset
    std::vector<int> current_subset;
    dfsBranchAndBound(0, current_subset);

    // Sort and keep top N
    sortResultsByMetric();

    best_results.clear();
    int n_to_keep = std::min(top_n_models, static_cast<int>(all_results.size()));
    for (int i = 0; i < n_to_keep; ++i)
    {
        best_results.push_back(all_results[i]);
    }

    // Calculate all metrics for all top_n models (for complete reporting)
    // This ensures all top models have complete info without refitting
    for (int i = 0; i < static_cast<int>(best_results.size()); ++i)
    {
        SubsetResult &result = best_results[i];
        Model model = result.getModel();
        std::vector<int> var_indices = model.getVariableIndices();

        // Remove intercept marker (-1) from indices to get just predictor indices
        std::vector<int> predictor_indices;
        for (int idx : var_indices)
        {
            if (idx != -1)
            {
                predictor_indices.push_back(idx);
            }
        }

        MatrixXd X_subset = extractSubsetMatrix(predictor_indices);

        VectorXi predictions = model.predict(X_subset);
        double accuracy = PerformanceEvaluator::calculateAccuracy(predictions, y);

        VectorXd fitted_probs = model.predict_proba(X_subset);
        double auc = PerformanceEvaluator::calculateAUC(fitted_probs, y);

        double deviance = model.getDeviance();
        int n_params = X_subset.cols();
        double aic_value = PerformanceEvaluator::calculateAIC(deviance, n_params);
        double bic_value = PerformanceEvaluator::calculateBIC(deviance, n_params, n_observations);

        best_results[i] = SubsetResult(model, accuracy, auc, aic_value, bic_value);
    }

    is_fitted = true;
}

// Depth-first Branch-and-Bound recursion starting from [start_idx]
void BestSubsetSelector::dfsBranchAndBound(int start_idx, std::vector<int> &current_subset)
{
    int k = static_cast<int>(current_subset.size());

    // Evaluate current subset if non-empty (consistent with previous enumeration)
    if (k > 0)
    {
        try
        {
            MatrixXd X_subset = extractSubsetMatrix(current_subset);

            LogisticRegression lr(X_subset, y, max_iterations, convergence_tolerance);
            lr.fit();

            Model model;
            std::vector<int> indices_for_model = include_intercept ? addInterceptColumn(current_subset) : current_subset;
            model.fitFromLogisticRegression(lr, indices_for_model);

            // Always calculate deviance; compute only AIC/BIC for ranking
            double deviance = model.getDeviance();
            int n_params = X_subset.cols();
            double aic_value = PerformanceEvaluator::calculateAIC(deviance, n_params);
            double bic_value = PerformanceEvaluator::calculateBIC(deviance, n_params, n_observations);

            double score = (metric == "aic") ? aic_value : bic_value;

            // Store full SubsetResult (other metrics NA)
            SubsetResult res(model, NA_REAL, NA_REAL, aic_value, bic_value);
            all_results.push_back(res);

            // Update best score
            if (score < best_score_so_far)
            {
                best_score_so_far = score;
            }
        }
        catch (const std::exception &e)
        {
            // Skip invalid/failed fits and continue
        }
    }

    // If we reached max_variables or no more variables to add, stop
    if (k >= max_variables || start_idx >= n_variables)
    {
        return;
    }

    // Lower bound for any descendant (must add at least one variable):
    // Use deviance lower bound = 0; penalty with k_min_desc = k+1
    int k_min_desc = k + 1;
    if (k_min_desc <= max_variables)
    {
        int n_params_lb = k_min_desc + (include_intercept ? 1 : 0);
        double lower_bound = (metric == "aic")
                                 ? PerformanceEvaluator::calculateAIC(0.0, n_params_lb)
                                 : PerformanceEvaluator::calculateBIC(0.0, n_params_lb, n_observations);

        if (!(lower_bound < best_score_so_far))
        {
            // Prune this subtree
            return;
        }
    }
    else
    {
        return; // cannot add more variables
    }

    // Continue branching by adding next variables
    for (int i = start_idx; i < n_variables; ++i)
    {
        current_subset.push_back(i);
        dfsBranchAndBound(i + 1, current_subset);
        current_subset.pop_back();
    }
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

void BestSubsetSelector::setNumThreads(int n_threads)
{
    if (n_threads <= 0)
        n_threads_configured = 1;
    else
        n_threads_configured = n_threads;
}

// Cross-validation configuration methods (restored)
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

// Cross-validation enabled fit method (restored)
void BestSubsetSelector::fitWithCrossValidation(int max_vars)
{
    std::vector<std::vector<int>> subsets = generateVariableSubsets(max_vars);
    size_t n_subsets = subsets.size();

    // Pre-allocate results for thread safety
    std::vector<SubsetResult> parallel_results(n_subsets);
    std::vector<bool> success_flags(n_subsets, false);

    // Progress reporting setup
    std::atomic<size_t> completed(0);
    // Report progress less often: every 2000 models (or 2% for very large jobs)
    size_t report_interval = std::max(static_cast<size_t>(2000), n_subsets / 50);

// Parallel loop over subsets with OpenMP
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime)
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

            double final_accuracy = NA_REAL;
            double final_auc = NA_REAL;
            double aic_value = NA_REAL;
            double bic_value = NA_REAL;
            double deviance = NA_REAL;

            // Use CV performance for ranking
            if (metric == "accuracy")
            {
                final_accuracy = cv_performance;
            }
            else if (metric == "auc")
            {
                final_auc = cv_performance;
            }
            else if (metric == "deviance")
            {
                deviance = cv_performance;
            }
            else if (metric == "aic")
            {
                aic_value = cv_performance;
            }
            else if (metric == "bic")
            {
                bic_value = cv_performance;
            }
            else
            {
                throw std::invalid_argument("Unsupported metric: " + metric);
            }

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
        size_t current = ++completed;
        if (current % report_interval == 0 || current == n_subsets)
        {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                int pct = static_cast<int>((100.0 * current) / n_subsets);
                Rcpp::Rcout << "\rEvaluated " << current << " / " << n_subsets
                            << " models (" << pct << "%)"
                            << std::flush;
            }
        }
    }

    // Print completion message
    Rcpp::Rcout << "\rEvaluated " << n_subsets << " / " << n_subsets
                << " models (100%)    \n"
                << std::flush;

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

    // Calculate all metrics for top_n models
    for (int i = 0; i < static_cast<int>(best_results.size()); ++i)
    {
        SubsetResult &result = best_results[i];
        Model model = result.getModel();
        std::vector<int> var_indices = model.getVariableIndices();

        double current_accuracy = result.getAccuracy();
        double current_auc = result.getAUC();
        double current_aic = result.getAIC();
        double current_bic = result.getBIC();

        std::vector<int> predictor_indices;
        for (int idx : var_indices)
        {
            if (idx != -1)
            {
                predictor_indices.push_back(idx);
            }
        }

        MatrixXd X_subset = extractSubsetMatrix(predictor_indices);

        VectorXi predictions = model.predict(X_subset);
        double accuracy = (metric == "accuracy") ? current_accuracy : PerformanceEvaluator::calculateAccuracy(predictions, y);

        VectorXd fitted_probs = model.predict_proba(X_subset);
        double auc = (metric == "auc") ? current_auc : PerformanceEvaluator::calculateAUC(fitted_probs, y);

        double deviance = model.getDeviance();
        int n_params = X_subset.cols();
        double aic_value = (metric == "aic") ? current_aic : PerformanceEvaluator::calculateAIC(deviance, n_params);
        double bic_value = (metric == "bic") ? current_bic : PerformanceEvaluator::calculateBIC(deviance, n_params, n_observations);

        best_results[i] = SubsetResult(model, accuracy, auc, aic_value, bic_value);
    }

    is_fitted = true;
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
