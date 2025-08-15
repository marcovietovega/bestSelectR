#ifndef PERFORMANCE_EVALUATOR_HPP
#define PERFORMANCE_EVALUATOR_HPP

#include <Eigen/Dense>

using namespace Eigen;

class PerformanceEvaluator
{
public:
    // Core metric calculations
    static double calculateAccuracy(const VectorXi &predictions, const VectorXd &true_labels);
    static double calculateAUC(const VectorXd &probabilities, const VectorXd &true_labels);
};

#endif // PERFORMANCE_EVALUATOR_HPP#endif // PERFORMANCE_EVALUATOR_HPP
