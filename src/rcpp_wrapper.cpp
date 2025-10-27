#include <Rcpp.h>
#include <RcppEigen.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "BestSubsetSelector.hpp"
#include "PerformanceEvaluator.hpp"
#include "LogisticRegression.hpp"
#include "Model.hpp"
#include "SubsetResult.hpp"

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace Eigen;

// C++ function for best subset selection
// [[Rcpp::export]]
List best_subset_selection(
    NumericMatrix X_r,
    NumericVector y_r,
    int max_variables = -1,
    int top_n = 10,
    std::string metric = "accuracy",
    bool use_cv = false,
    int cv_folds = 5,
    int cv_repeats = 1,
    int cv_seed = -1,
    bool include_intercept = true,
    int max_iterations = 100,
    double tolerance = 1e-6,
    int n_threads = -1)
{
    try
    {
        // Convert R objects to Eigen
        Map<MatrixXd> X_eigen(as<Map<MatrixXd>>(X_r));
        Map<VectorXd> y_eigen(as<Map<VectorXd>>(y_r));

        // Validate inputs
        if (X_eigen.rows() != y_eigen.rows())
        {
            stop("X and y must have the same number of rows");
        }

        if (X_eigen.rows() < 2)
        {
            stop("Need at least 2 observations");
        }

        // Check y values are binary
        for (int i = 0; i < y_eigen.size(); ++i)
        {
            if (y_eigen(i) != 0.0 && y_eigen(i) != 1.0)
            {
                stop("y must contain only 0 and 1 values");
            }
        }

        // Set default max_variables if not specified
        if (max_variables < 0)
        {
            max_variables = X_eigen.cols();
        }

// Configure OpenMP threads
#ifdef _OPENMP
        int original_threads = omp_get_max_threads();
        if (n_threads > 0)
        {
            omp_set_num_threads(n_threads);
        }
        else if (n_threads == 0)
        {
            // n_threads = 0 means serial execution
            omp_set_num_threads(1);
        }
// n_threads = -1 (default) uses OpenMP default (all available threads)
#endif

        // Create BestSubsetSelector
        BestSubsetSelector selector(X_eigen, y_eigen, include_intercept);

        // Configure selector
        selector.setMaxVariables(max_variables);
        selector.setTopNModels(top_n);
        selector.setMetric(metric);
        selector.setConvergenceParameters(tolerance, max_iterations);

        if (use_cv)
        {
            selector.setCrossValidation(true, cv_folds, cv_repeats, cv_seed);
        }

        // Inform selector of configured threads to enable serial warm starts when n_threads == 1
        int selector_threads = 1;
#ifdef _OPENMP
        if (n_threads > 0)
            selector_threads = n_threads;
        else if (n_threads == -1)
            selector_threads = omp_get_max_threads();
        else
            selector_threads = 1;
#endif
        selector.setNumThreads(selector_threads);

        // Fit the model
        selector.fit();

// Restore original thread count
#ifdef _OPENMP
        omp_set_num_threads(original_threads);
#endif

        // Get results
        std::vector<SubsetResult> best_results = selector.getBestResults();
        SubsetResult best_model = selector.getBestModel();

        // Convert results to R format
        int n_models = best_results.size();
        NumericVector model_accuracy(n_models);
        NumericVector model_auc(n_models);
        NumericVector model_deviance(n_models);
        NumericVector model_aic(n_models);
        NumericVector model_bic(n_models);
        IntegerVector n_variables(n_models);
        CharacterVector variable_names(n_models);

        for (int i = 0; i < n_models; ++i)
        {
            model_accuracy[i] = best_results[i].getAccuracy();
            model_auc[i] = best_results[i].getAUC();
            model_deviance[i] = best_results[i].getDeviance();
            model_aic[i] = best_results[i].getAIC();
            model_bic[i] = best_results[i].getBIC();

            std::vector<int> var_indices = best_results[i].getVariableIndices();

            // Count non-intercept variables
            int n_vars = 0;
            std::string var_str = "";
            for (size_t j = 0; j < var_indices.size(); ++j)
            {
                if (var_indices[j] != -1)
                { // Not intercept
                    if (n_vars > 0)
                        var_str += ",";
                    var_str += "X" + std::to_string(var_indices[j] + 1); // 1-based indexing for R
                    n_vars++;
                }
            }
            if (include_intercept)
            {
                if (n_vars > 0)
                    var_str = "Intercept," + var_str;
                else
                    var_str = "Intercept";
            }

            n_variables[i] = n_vars; // Count predictors only, not intercept
            variable_names[i] = var_str;
        }

        IntegerVector ranks = Range(1, n_models);

        // Create models data frame
        DataFrame models = DataFrame::create(
            Named("rank") = ranks,
            Named("variables") = variable_names,
            Named("n_variables") = n_variables,
            Named("accuracy") = model_accuracy,
            Named("auc") = model_auc,
            Named("deviance") = model_deviance,
            Named("aic") = model_aic,
            Named("bic") = model_bic);

        // Best model coefficients
        VectorXd coeffs = best_model.getModel().getCoefficients();
        NumericVector coefficients(coeffs.data(), coeffs.data() + coeffs.size());

        // Best model variable indices
        std::vector<int> best_var_indices = best_model.getVariableIndices();
        IntegerVector best_variables;
        for (size_t i = 0; i < best_var_indices.size(); ++i)
        {
            if (best_var_indices[i] != -1)
            {                                                      // Not intercept
                best_variables.push_back(best_var_indices[i] + 1); // 1-based for R
            }
        }

        // Best model info
        List best_info = List::create(
            Named("variables") = best_variables,
            Named("coefficients") = coefficients,
            Named("accuracy") = best_model.getAccuracy(),
            Named("auc") = best_model.getAUC(),
            Named("deviance") = best_model.getDeviance(),
            Named("aic") = best_model.getAIC(),
            Named("bic") = best_model.getBIC(),
            Named("n_variables") = best_variables.size()); // Count predictors only, not intercept

        // Call information
        List call_info = List::create(
            Named("n_observations") = X_eigen.rows(),
            Named("n_predictors") = X_eigen.cols(),
            Named("max_variables") = max_variables,
            Named("top_n") = top_n,
            Named("metric") = metric,
            Named("use_cv") = use_cv,
            Named("cv_folds") = cv_folds,
            Named("cv_repeats") = cv_repeats,
            Named("include_intercept") = include_intercept,
            Named("n_models_evaluated") = selector.getNumModelsEvaluated());

        // Return comprehensive results
        return List::create(
            Named("models") = models,
            Named("best_model") = best_info,
            Named("call_info") = call_info);
    }
    catch (const std::exception &e)
    {
        stop("C++ error: " + std::string(e.what()));
    }
}

// C++ function for predictions
// [[Rcpp::export]]
NumericVector predict_best_subset(
    List model_info,
    NumericMatrix X_new,
    std::string type = "class")
{
    try
    {
        // Extract model information
        IntegerVector variables = model_info["variables"];
        NumericVector coefficients = model_info["coefficients"];

        // Convert to Eigen
        Map<MatrixXd> X_eigen(as<Map<MatrixXd>>(X_new));

        // Validate inputs
        if (variables.size() > X_eigen.cols())
        {
            stop("Model requires more variables than provided in X_new");
        }

        // Create subset matrix
        int n_obs = X_eigen.rows();
        int n_vars = variables.size();
        bool has_intercept = (coefficients.size() == n_vars + 1);

        MatrixXd X_subset(n_obs, coefficients.size());

        int col_idx = 0;
        if (has_intercept)
        {
            X_subset.col(0) = VectorXd::Ones(n_obs); // Intercept
            col_idx = 1;
        }

        // Add selected variables (convert from 1-based to 0-based indexing)
        for (int i = 0; i < n_vars; ++i)
        {
            int var_idx = variables[i] - 1; // Convert to 0-based
            if (var_idx >= X_eigen.cols())
            {
                stop("Variable index out of bounds");
            }
            X_subset.col(col_idx + i) = X_eigen.col(var_idx);
        }

        // Calculate linear combination
        VectorXd coeffs_eigen(coefficients.size());
        for (int i = 0; i < coefficients.size(); ++i)
        {
            coeffs_eigen(i) = coefficients[i];
        }

        VectorXd linear_comb = X_subset * coeffs_eigen;

        // Calculate predictions based on type
        NumericVector predictions(n_obs);

        if (type == "prob")
        {
            // Return probabilities
            for (int i = 0; i < n_obs; ++i)
            {
                predictions[i] = 1.0 / (1.0 + exp(-linear_comb(i)));
            }
        }
        else if (type == "class")
        {
            // Return binary predictions
            for (int i = 0; i < n_obs; ++i)
            {
                double prob = 1.0 / (1.0 + exp(-linear_comb(i)));
                predictions[i] = (prob >= 0.5) ? 1.0 : 0.0;
            }
        }
        else
        {
            stop("type must be 'class' or 'prob'");
        }

        return predictions;
    }
    catch (const std::exception &e)
    {
        stop("C++ error in prediction: " + std::string(e.what()));
    }
}
