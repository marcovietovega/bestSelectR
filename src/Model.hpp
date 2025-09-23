#ifndef MODEL_HPP
#define MODEL_HPP

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

class Model
{
private:
    VectorXd coefficients;
    double deviance;
    std::vector<int> variable_indices; // Which variables were used (for subset selection)
    int n_variables;
    bool is_fitted;

public:
    // Constructors
    Model();
    Model(const VectorXd &coeffs, double dev, const std::vector<int> &var_indices);

    // Fitting from LogisticRegression
    void fitFromLogisticRegression(const class LogisticRegression &lr_model,
                                   const std::vector<int> &var_indices);

    // Prediction methods
    VectorXd predict_proba(const MatrixXd &X) const;
    VectorXi predict(const MatrixXd &X) const;

    // Getters
    VectorXd getCoefficients() const;
    double getDeviance() const;
    std::vector<int> getVariableIndices() const;
    int getNumVariables() const;
    bool isFitted() const;

    // Model comparison
    bool operator<(const Model &other) const; // For sorting by deviance
};

#endif // MODEL_HPP
