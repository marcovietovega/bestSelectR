#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <Eigen/Dense>

using namespace Eigen;

class LogisticRegression
{
private:
    MatrixXd X;
    VectorXd y;
    VectorXd beta; // coefficients
    VectorXd eta;  // current linear predictor
    int n, p;
    int max_iterations;
    double rel_tolerance; // epsilon

    // Helper methods
    static inline double plogis_clip(double t);
    double deviance_from_eta(const VectorXd &eta_in) const;
    VectorXd irls_proposal_beta_from_eta(const VectorXd &eta_curr) const;
    void initialize();

public:
    // Constructor
    LogisticRegression(const MatrixXd &X_in, const VectorXd &y_in,
                       int max_iter = 50, double tol = 1e-8);

    // Main fitting method
    void fit();

    // Getters
    VectorXd getCoefficients() const;

    // Prediction methods
    VectorXd predict_proba(const MatrixXd &X_new) const;
    VectorXi predict(const MatrixXd &X_new) const;
    VectorXd get_fitted_values() const;

    // Model diagnostics
    double get_deviance() const;
};

#endif // LOGISTIC_REGRESSION_HPP
