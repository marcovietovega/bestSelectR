#ifndef BEST_SUBSET_SELECTOR_HPP
#define BEST_SUBSET_SELECTOR_HPP

#include <Eigen/Dense>
#include <atomic>
#include <vector>
#include <string>
#include "SubsetResult.hpp"
#include "LogisticRegression.hpp"
#include "PerformanceEvaluator.hpp"
#include "Model.hpp"

using namespace Eigen;

class BestSubsetSelector
{
private:
    MatrixXd X;
    VectorXd y;
    int n_observations;
    int n_variables;
    bool include_intercept;

    int max_variables;
    int top_n_models;
    std::string metric;
    double convergence_tolerance;
    int max_iterations;

    bool use_cross_validation;
    int cv_folds;
    int cv_repeats;
    int cv_seed;

    std::vector<SubsetResult> all_results;
    std::vector<SubsetResult> best_results;
    bool is_fitted;
    std::vector<std::vector<int>> generateVariableSubsets();
    std::vector<std::vector<int>> generateVariableSubsets(int max_size);
    // Generate subsets in Gray-code order (consecutive subsets differ by 1 bit on the full space).
    // When filtering by max_size, some consecutive accepted subsets may differ by >1 bit, but
    // overall improves warm-start locality substantially in serial runs.
    std::vector<std::vector<int>> generateVariableSubsetsGrayCode(int max_size);
    MatrixXd extractSubsetMatrix(const std::vector<int> &variable_indices);
    std::vector<int> addInterceptColumn(const std::vector<int> &variable_indices);
    void sortResultsByMetric();

    // Branch-and-Bound for AIC/BIC
    void fitBranchAndBound();
    void dfsBranchAndBound(int start_idx, std::vector<int> &current_subset);
    std::atomic<double> best_score_so_far; // best AIC/BIC (lower is better)

    // Thread configuration (for enabling serial warm starts)
    int n_threads_configured = 1;

public:
    BestSubsetSelector(const MatrixXd &X_input, const VectorXd &y_input);
    BestSubsetSelector(const MatrixXd &X_input, const VectorXd &y_input,
                       bool add_intercept);

    void fit();
    void fit(int max_vars);
    void fit(int max_vars, int top_n);
    void fit(int max_vars, int top_n, const std::string &selection_metric);
    void fit(int max_vars, int top_n, const std::string &selection_metric,
             bool use_cv, int k_folds = 5, int repeats = 1, int seed = -1);

    void setMaxVariables(int max_vars);
    void setTopNModels(int top_n);
    void setMetric(const std::string &selection_metric);
    void setConvergenceParameters(double tolerance, int max_iter);
    void setIncludeIntercept(bool include);
    void setNumThreads(int n_threads);

    void setCrossValidation(bool use_cv, int k_folds = 5, int repeats = 1, int seed = -1);
    bool isUsingCrossValidation() const;
    int getCVFolds() const;
    int getCVRepeats() const;
    int getCVSeed() const;
    void fitWithCrossValidation(int max_vars);

    std::vector<SubsetResult> getBestResults() const;
    std::vector<SubsetResult> getAllResults() const;
    SubsetResult getBestModel() const;
    SubsetResult getBestModel(const std::string &selection_metric) const;

    int getNumModelsEvaluated() const;
    int getNumVariables() const;
    int getNumObservations() const;

    VectorXd predict_proba(const MatrixXd &X_new) const;
    VectorXi predict(const MatrixXd &X_new) const;

    bool isFitted() const;
};

#endif // BEST_SUBSET_SELECTOR_HPP
