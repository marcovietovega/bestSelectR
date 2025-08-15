#ifndef SUBSET_RESULT_HPP
#define SUBSET_RESULT_HPP

#include <Eigen/Dense>
#include <vector>
#include "Model.hpp"

using namespace Eigen;

class SubsetResult
{
private:
    Model model;
    double accuracy;
    double auc;
    std::vector<int> variable_indices;
    int n_variables;
    bool is_valid;

public:
    // Constructors
    SubsetResult();
    SubsetResult(const Model &fitted_model, double acc, double auc_score);

    // Create from fitted model + performance metrics
    void setResult(const Model &fitted_model, double acc, double auc_score);

    // Getters
    Model getModel() const;
    double getAccuracy() const;
    double getAUC() const;
    double getDeviance() const;
    std::vector<int> getVariableIndices() const;
    int getNumVariables() const;
    bool isValid() const;

    // Performance score (can be accuracy, AUC, or custom combination)
    double getScore() const;                          // Returns accuracy by default
    double getScore(const std::string &metric) const; // "accuracy", "auc", "deviance"

    // Comparison operators for ranking
    bool operator<(const SubsetResult &other) const; // Less than (by accuracy)
    bool operator>(const SubsetResult &other) const; // Greater than (by accuracy)

    // Comparison by specific metric
    bool isBetterThan(const SubsetResult &other, const std::string &metric) const;

    // Prediction methods (delegates to Model)
    VectorXd predict_proba(const MatrixXd &X) const;
    VectorXi predict(const MatrixXd &X) const;
};

#endif // SUBSET_RESULT_HPP
