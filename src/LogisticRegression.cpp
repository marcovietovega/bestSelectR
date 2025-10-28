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
    // Vectorized deviance computation with clipping
    ArrayXd eta_arr = eta_in.array();
    ArrayXd mu = 1.0 / (1.0 + (-eta_arr).exp());
    // Clip to avoid log(0)
    const double eps = DBL_EPSILON;
    mu = mu.max(eps).min(1.0 - eps);

    ArrayXd y_arr = y.array();
    ArrayXd dev_terms = (-2.0) * (y_arr * mu.log() + (1.0 - y_arr) * (1.0 - mu).log());
    return dev_terms.sum();
}

VectorXd LogisticRegression::irls_proposal_beta_from_eta(const VectorXd &eta_curr) const
{
    // Ensure buffers are allocated
    if (sw.size() != n)
    {
        sw.resize(n);
        z.resize(n);
        zw.resize(n);
    }
    if (Xw.rows() != n || Xw.cols() != p)
    {
        Xw.resize(n, p);
    }

    // Compute working response and sqrt(weights)
    for (int i = 0; i < n; ++i)
    {
        double mu = plogis_clip(eta_curr(i));
        double dmu = mu * (1.0 - mu);
        sw(i) = std::sqrt(dmu);
        z(i) = eta_curr(i) + (y(i) - mu) / dmu;
    }

    // Xw = X .* sw (row-wise scaling via broadcasting)
    Xw = X; // copy once
    Xw = Xw.array().colwise() * sw.array();

    // zw = z .* sw
    zw = z.array() * sw.array();

    // Solve weighted least squares via normal equations with Cholesky
    // XtWX = (Xw^T Xw), XtWz = (Xw^T zw)
    MatrixXd XtWX = Xw.transpose() * Xw;
    VectorXd XtWz = Xw.transpose() * zw;

    // Try LLT (Cholesky) first
    Eigen::LLT<MatrixXd> llt;
    llt.compute(XtWX);

    bool llt_ok = (llt.info() == Eigen::Success);
    if (!llt_ok)
    {
        // Add a tiny ridge and retry
        double max_diag = XtWX.diagonal().cwiseAbs().maxCoeff();
        double ridge = std::max(1e-12, 1e-8 * max_diag);
        XtWX.diagonal().array() += ridge;
        llt.compute(XtWX);
        llt_ok = (llt.info() == Eigen::Success);
    }

    if (llt_ok)
    {
        return llt.solve(XtWz);
    }
    else
    {
        // Fallback: use QR on Xw
        ColPivHouseholderQR<MatrixXd> qr(Xw);
        double qr_tol = std::min(1e-7, rel_tolerance / 1000.0); // min(1e-7, eps/1000)
        qr.setThreshold(qr_tol);
        return qr.solve(zw);
    }
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

// Overloaded constructor to accept block expressions without materializing temporaries
LogisticRegression::LogisticRegression(const Ref<const MatrixXd> &X_in, const VectorXd &y_in,
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
    if (has_initial_guess && beta_initial.size() == p && eta_initial.size() == n)
    {
        beta = beta_initial;
        eta = eta_initial;
    }
    else
    {
        initialize();
    }

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
                // Halve step in parameter and update eta via linearity: X * (avg beta) = avg(etas)
                beta_prop = 0.5 * (beta_before + beta_prop);
                eta_prop = 0.5 * (eta_before + eta_prop);
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
    ArrayXd mu = 1.0 / (1.0 + (-eta_new.array()).exp());
    const double eps = DBL_EPSILON;
    mu = mu.max(eps).min(1.0 - eps);
    return mu.matrix();
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

void LogisticRegression::setInitialGuess(const VectorXd &beta0, const VectorXd &eta0)
{
    // Basic validation (sizes checked in fit)
    beta_initial = beta0;
    eta_initial = eta0;
    has_initial_guess = true;
}
