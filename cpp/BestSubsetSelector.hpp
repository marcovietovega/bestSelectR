#ifndef BEST_SUBSET_SELECTOR_HPP
#define BEST_SUBSET_SELECTOR_HPP

#include <Eigen/Dense>
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

    // Algorithm parameters
    int max_variables;
    int top_n_models;
    std::string metric;
    double convergence_tolerance;
    int max_iterations;

    // Cross-validation parameters
    bool use_cross_validation;
    int cv_folds;
    int cv_repeats;
    int cv_seed;

    // Results storage
    std::vector<SubsetResult> all_results;
    std::vector<SubsetResult> best_results;
    bool is_fitted;

    // Private helper methods
    std::vector<std::vector<int>> generateVariableSubsets();
    std::vector<std::vector<int>> generateVariableSubsets(int max_size);
    MatrixXd extractSubsetMatrix(const std::vector<int> &variable_indices);
    std::vector<int> addInterceptColumn(const std::vector<int> &variable_indices);
    void sortResultsByMetric();

public:
    // Constructors
    BestSubsetSelector(const MatrixXd &X_input, const VectorXd &y_input);
    BestSubsetSelector(const MatrixXd &X_input, const VectorXd &y_input,
                       bool add_intercept);

    // Main fitting method
    void fit();
    void fit(int max_vars);
    void fit(int max_vars, int top_n);
    void fit(int max_vars, int top_n, const std::string &selection_metric);
    void fit(int max_vars, int top_n, const std::string &selection_metric,
             bool use_cv, int k_folds = 5, int repeats = 1, int seed = -1);

    // Configuration methods
    void setMaxVariables(int max_vars);
    void setTopNModels(int top_n);
    void setMetric(const std::string &selection_metric);
    void setConvergenceParameters(double tolerance, int max_iter);
    void setIncludeIntercept(bool include);

    // Cross-validation configuration
    void setCrossValidation(bool use_cv, int k_folds = 5, int repeats = 1, int seed = -1);
    bool isUsingCrossValidation() const;
    int getCVFolds() const;
    int getCVRepeats() const;
    int getCVSeed() const;
    void fitWithCrossValidation(int max_vars);

    // Results access
    std::vector<SubsetResult> getBestResults() const;
    std::vector<SubsetResult> getAllResults() const;
    SubsetResult getBestModel() const;
    SubsetResult getBestModel(const std::string &selection_metric) const;

    // Model statistics
    int getNumModelsEvaluated() const;
    int getNumVariables() const;
    int getNumObservations() const;

    // Prediction using best model
    VectorXd predict_proba(const MatrixXd &X_new) const;
    VectorXi predict(const MatrixXd &X_new) const;

    // Status
    bool isFitted() const;
};

#endif // BEST_SUBSET_SELECTOR_HPP
