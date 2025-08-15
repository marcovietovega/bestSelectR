#include <iostream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include "Model.hpp"
#include "LogisticRegression.hpp"

using namespace std;

// Default constructor
Model::Model() : deviance(0.0), n_variables(0), is_fitted(false)
{
}

// Constructor with fitted parameters
Model::Model(const VectorXd &coeffs, double dev, const std::vector<int> &var_indices)
    : coefficients(coeffs), deviance(dev), variable_indices(var_indices),
      n_variables(var_indices.size()), is_fitted(true)
{
}

// Fit from a LogisticRegression object
void Model::fitFromLogisticRegression(const LogisticRegression &lr_model,
                                      const std::vector<int> &var_indices)
{
    coefficients = lr_model.getCoefficients();
    deviance = lr_model.get_deviance();
    variable_indices = var_indices;
    n_variables = var_indices.size();
    is_fitted = true;
}

// Predict probabilities
VectorXd Model::predict_proba(const MatrixXd &X) const
{
    if (!is_fitted)
    {
        throw std::runtime_error("Model has not been fitted yet");
    }

    if (X.cols() != coefficients.size())
    {
        throw std::invalid_argument("Input matrix columns must match number of coefficients");
    }

    VectorXd eta = X * coefficients;
    VectorXd probs(eta.size());

    for (int i = 0; i < eta.size(); ++i)
    {
        // Use the same logistic function as LogisticRegression
        if (eta(i) >= 0)
        {
            double e = std::exp(-eta(i));
            double p = 1.0 / (1.0 + e);
            probs(i) = std::max(std::min(p, 1.0 - DBL_EPSILON), DBL_EPSILON);
        }
        else
        {
            double e = std::exp(eta(i));
            double p = e / (1.0 + e);
            probs(i) = std::max(std::min(p, 1.0 - DBL_EPSILON), DBL_EPSILON);
        }
    }

    return probs;
}

// Predict binary outcomes
VectorXi Model::predict(const MatrixXd &X) const
{
    VectorXd probs = predict_proba(X);
    VectorXi predictions(probs.size());

    for (int i = 0; i < probs.size(); ++i)
    {
        predictions(i) = (probs(i) >= 0.5) ? 1 : 0;
    }

    return predictions;
}

// Getters
VectorXd Model::getCoefficients() const
{
    return coefficients;
}

double Model::getDeviance() const
{
    return deviance;
}

std::vector<int> Model::getVariableIndices() const
{
    return variable_indices;
}

int Model::getNumVariables() const
{
    return n_variables;
}

bool Model::isFitted() const
{
    return is_fitted;
}

// Comparison operator for sorting (lower deviance = better model)
bool Model::operator<(const Model &other) const
{
    return this->deviance < other.deviance;
}
