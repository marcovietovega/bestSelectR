#ifndef PERFORMANCE_EVALUATOR_HPP
#define PERFORMANCE_EVALUATOR_HPP

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

class PerformanceEvaluator
{
public:
    // Core metric calculations
    static double calculateAccuracy(const VectorXi &predictions, const VectorXd &true_labels);
    static double calculateAUC(const VectorXd &probabilities, const VectorXd &true_labels);
    static double calculateDeviance(const VectorXd &probabilities, const VectorXd &true_labels);

    // K-Fold Cross-Validation methods
    static double calculateKFold(const MatrixXd &X, const VectorXd &y,
                                 const std::vector<int> &variable_indices,
                                 const std::string &metric,
                                 bool include_intercept = true,
                                 int k_folds = 5, int seed = -1);

    static double calculateRepeatedKFold(const MatrixXd &X, const VectorXd &y,
                                         const std::vector<int> &variable_indices,
                                         const std::string &metric,
                                         bool include_intercept = true,
                                         int k_folds = 5, int n_repeats = 3, int seed = -1);

private:
    // Helper methods for CV
    static std::vector<VectorXi> createKFoldIndices(int n_samples, int k_folds, int seed = -1);
    static MatrixXd extractSubsetMatrix(const MatrixXd &X, const std::vector<int> &variable_indices,
                                        bool include_intercept);
    static std::vector<int> addInterceptColumn(const std::vector<int> &variable_indices);
};

#endif // PERFORMANCE_EVALUATOR_HPP
