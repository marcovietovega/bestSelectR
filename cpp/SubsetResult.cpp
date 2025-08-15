#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "SubsetResult.hpp"

using namespace std;

// Default constructor
SubsetResult::SubsetResult()
    : accuracy(0.0), auc(0.0), n_variables(0), is_valid(false)
{
}

// Constructor with fitted model and performance metrics
SubsetResult::SubsetResult(const Model &fitted_model, double acc, double auc_score)
    : model(fitted_model), accuracy(acc), auc(auc_score),
      variable_indices(fitted_model.getVariableIndices()),
      n_variables(fitted_model.getNumVariables()), is_valid(true)
{
    // Validate inputs
    if (acc < 0.0 || acc > 1.0)
    {
        throw std::invalid_argument("Accuracy must be between 0 and 1");
    }
    if (auc_score < 0.0 || auc_score > 1.0)
    {
        throw std::invalid_argument("AUC must be between 0 and 1");
    }
    if (!fitted_model.isFitted())
    {
        throw std::invalid_argument("Model must be fitted");
    }
}

// Set result from fitted model and metrics
void SubsetResult::setResult(const Model &fitted_model, double acc, double auc_score)
{
    if (!fitted_model.isFitted())
    {
        throw std::invalid_argument("Model must be fitted");
    }
    if (acc < 0.0 || acc > 1.0)
    {
        throw std::invalid_argument("Accuracy must be between 0 and 1");
    }
    if (auc_score < 0.0 || auc_score > 1.0)
    {
        throw std::invalid_argument("AUC must be between 0 and 1");
    }

    model = fitted_model;
    accuracy = acc;
    auc = auc_score;
    variable_indices = fitted_model.getVariableIndices();
    n_variables = fitted_model.getNumVariables();
    is_valid = true;
}

// Getters
Model SubsetResult::getModel() const
{
    return model;
}

double SubsetResult::getAccuracy() const
{
    return accuracy;
}

double SubsetResult::getAUC() const
{
    return auc;
}

double SubsetResult::getDeviance() const
{
    return model.getDeviance();
}

std::vector<int> SubsetResult::getVariableIndices() const
{
    return variable_indices;
}

int SubsetResult::getNumVariables() const
{
    return n_variables;
}

bool SubsetResult::isValid() const
{
    return is_valid;
}

// Get performance score (Accuracy by default)
double SubsetResult::getScore() const
{
    return accuracy;
}

// Get performance score by specific metric
double SubsetResult::getScore(const std::string &metric) const
{
    if (metric == "accuracy")
    {
        return accuracy;
    }
    else if (metric == "auc")
    {
        return auc;
    }
    else if (metric == "deviance")
    {
        return model.getDeviance();
    }
    else
    {
        throw std::invalid_argument("Unknown metric: " + metric + ". Use 'accuracy', 'auc', or 'deviance'");
    }
}

// Comparison operators (higher accuracy = better)
bool SubsetResult::operator<(const SubsetResult &other) const
{
    return this->accuracy < other.accuracy;
}

bool SubsetResult::operator>(const SubsetResult &other) const
{
    return this->accuracy > other.accuracy;
}

// Compare by specific metric
bool SubsetResult::isBetterThan(const SubsetResult &other, const std::string &metric) const
{
    if (metric == "accuracy" || metric == "auc")
    {
        return this->getScore(metric) > other.getScore(metric); // Higher is better
    }
    else if (metric == "deviance")
    {
        return this->getScore(metric) < other.getScore(metric); // Lower is better
    }
    else
    {
        throw std::invalid_argument("Unknown metric: " + metric);
    }
}

// Prediction methods (delegate to Model)
VectorXd SubsetResult::predict_proba(const MatrixXd &X) const
{
    if (!is_valid)
    {
        throw std::runtime_error("SubsetResult is not valid");
    }
    return model.predict_proba(X);
}

VectorXi SubsetResult::predict(const MatrixXd &X) const
{
    if (!is_valid)
    {
        throw std::runtime_error("SubsetResult is not valid");
    }
    return model.predict(X);
}
