#include <iostream>
#include <iomanip>
#include <cfloat>    // DBL_EPSILON
#include <algorithm> // std::min
#include "LogisticRegression.hpp"

using namespace std;

double LogisticRegression::plogis_clip(double t)
{
    if (t >= 0)
    {
        double e = std::exp(-t);
        double p = 1.0 / (1.0 + e);
        if (p < DBL_EPSILON)
            return DBL_EPSILON;
        if (p > 1.0 - DBL_EPSILON)
            return 1.0 - DBL_EPSILON;
        return p;
    }
    else
    {
        double e = std::exp(t);
        double p = e / (1.0 + e);
        if (p < DBL_EPSILON)
            return DBL_EPSILON;
        if (p > 1.0 - DBL_EPSILON)
            return 1.0 - DBL_EPSILON;
        return p;
    }
}

double LogisticRegression::deviance_from_eta(const VectorXd &eta_in) const
{
    double dev = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double mu = plogis_clip(eta_in(i));

        if (y(i) == 1.0)
        {
            if (mu > DBL_EPSILON)
            {
                dev += -2.0 * std::log(mu);
            }
            else
            {
                dev += 1e10;
            }
        }
        else
        {
            if ((1.0 - mu) > DBL_EPSILON)
            {
                dev += -2.0 * std::log(1.0 - mu);
            }
            else
            {
                dev += 1e10;
            }
        }
    }
    return dev;
}

VectorXd LogisticRegression::irls_proposal_beta_from_eta(const VectorXd &eta_curr) const
{
    VectorXd sw(n), z(n);
    for (int i = 0; i < n; ++i)
    {
        double mu = plogis_clip(eta_curr(i));
        double dmu = mu * (1.0 - mu);
        sw(i) = std::sqrt(dmu);
        z(i) = eta_curr(i) + (y(i) - mu) / dmu;
    }

    MatrixXd Xw = X;
    VectorXd zw = z;
    for (int i = 0; i < n; ++i)
    {
        Xw.row(i) *= sw(i);
        zw(i) *= sw(i);
    }

    ColPivHouseholderQR<MatrixXd> qr(Xw);
    double qr_tol = std::min(1e-7, rel_tolerance / 1000.0); // min(1e-7, eps/1000)
    qr.setThreshold(qr_tol);

    return qr.solve(zw); // proposed beta
}

// Initialization
void LogisticRegression::initialize()
{
    beta = VectorXd::Zero(p);
    eta = VectorXd(n);
    for (int i = 0; i < n; ++i)
    {
        double mu0 = (y(i) + 0.5) / 2.0; // mustart for binomial with weights=1
        if (mu0 <= DBL_EPSILON)
            mu0 = DBL_EPSILON;
        if (mu0 >= 1.0 - DBL_EPSILON)
            mu0 = 1.0 - DBL_EPSILON;
        eta(i) = std::log(mu0 / (1.0 - mu0)); // etastart
    }
}

// Constructor
LogisticRegression::LogisticRegression(const MatrixXd &X_in, const VectorXd &y_in,
                                       int max_iter, double tol)
    : X(X_in), y(y_in), max_iterations(max_iter), rel_tolerance(tol)
{
    n = X.rows();
    p = X.cols();
    beta = VectorXd::Zero(p);
    eta = VectorXd::Zero(n);
}

// Main fitting method
void LogisticRegression::fit()
{
    initialize();

    // Use the CURRENT MODEL deviance (beta=0 ⇒ mu=0.5) as the baseline
    double dev_old = deviance_from_eta(X * beta);

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        VectorXd beta_before = beta;
        VectorXd eta_before = eta;

        // Proposed coefficients and eta from current eta
        VectorXd beta_prop = irls_proposal_beta_from_eta(eta_before);
        VectorXd eta_prop = X * beta_prop;
        double dev_new = deviance_from_eta(eta_prop);

        // Step-halving if deviance didn't drop
        if (!(dev_new < dev_old))
        {
            for (int k = 0; k < 30 && !(dev_new < dev_old); ++k)
            {
                beta_prop = 0.5 * (beta_before + beta_prop);
                eta_prop = X * beta_prop;
                dev_new = deviance_from_eta(eta_prop);
            }
        }

        beta = beta_prop;
        eta = eta_prop;

        double crit = std::fabs(dev_new - dev_old) / (std::fabs(dev_new) + 0.1);

        if (crit <= rel_tolerance)
        {
            return; // converged
        }

        dev_old = dev_new;
    }
}

// Getter methods
VectorXd LogisticRegression::getCoefficients() const
{
    return beta;
}

// Prediction methods
VectorXd LogisticRegression::predict_proba(const MatrixXd &X_new) const
{
    VectorXd eta_new = X_new * beta;
    VectorXd probs(eta_new.size());
    for (int i = 0; i < eta_new.size(); ++i)
    {
        probs(i) = plogis_clip(eta_new(i));
    }
    return probs;
}

VectorXi LogisticRegression::predict(const MatrixXd &X_new) const
{
    VectorXd probs = predict_proba(X_new);
    VectorXi predictions(probs.size());
    for (int i = 0; i < probs.size(); ++i)
    {
        predictions(i) = (probs(i) >= 0.5) ? 1 : 0;
    }
    return predictions;
}

VectorXd LogisticRegression::get_fitted_values() const
{
    return predict_proba(X);
}

double LogisticRegression::get_deviance() const
{
    return deviance_from_eta(eta);
}
