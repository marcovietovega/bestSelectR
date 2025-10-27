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
    double aic;
    double bic;
    std::vector<int> variable_indices;
    int n_variables;
    bool is_valid;

public:
    SubsetResult();
    SubsetResult(const Model &fitted_model, double acc, double auc_score,
                 double aic_value, double bic_value);

    void setResult(const Model &fitted_model, double acc, double auc_score,
                   double aic_value, double bic_value);

    Model getModel() const;
    double getAccuracy() const;
    double getAUC() const;
    double getDeviance() const;
    double getAIC() const;
    double getBIC() const;
    std::vector<int> getVariableIndices() const;
    int getNumVariables() const;
    bool isValid() const;

    double getScore() const;
    double getScore(const std::string &metric) const;

    bool operator<(const SubsetResult &other) const;
    bool operator>(const SubsetResult &other) const;

    bool isBetterThan(const SubsetResult &other, const std::string &metric) const;

    VectorXd predict_proba(const MatrixXd &X) const;
    VectorXi predict(const MatrixXd &X) const;
};

#endif // SUBSET_RESULT_HPP
