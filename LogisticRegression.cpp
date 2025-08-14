#include <iostream>
#include <iomanip>
#include <cfloat>    // DBL_EPSILON
#include <algorithm> // std::min
#include <Eigen/Dense>

using namespace std;
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
    double rel_tolerance; // glm.control(epsilon)

    // Stable logistic with clipping
    static inline double plogis_clip(double t)
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

    // Deviance from a given eta (binomial 0/1)
    double deviance_from_eta(const VectorXd &eta_in) const
    {
        double dev = 0.0;
        for (int i = 0; i < n; ++i)
        {
            double mu = plogis_clip(eta_in(i));
            dev += (y(i) == 1.0) ? -2.0 * std::log(mu)
                                 : -2.0 * std::log(1.0 - mu);
        }
        return dev;
    }

    // One IRLS proposal using the CURRENT eta
    VectorXd irls_proposal_beta_from_eta(const VectorXd &eta_curr) const
    {
        VectorXd sw(n), z(n);
        for (int i = 0; i < n; ++i)
        {
            double mu = plogis_clip(eta_curr(i));
            double dmu = mu * (1.0 - mu);           // d_mu/d_eta for logit
            sw(i) = std::sqrt(dmu);                 // sqrt(W) with W = mu(1-mu)
            z(i) = eta_curr(i) + (y(i) - mu) / dmu; // working response
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

    // eta <- logit(mustart), keep beta = 0
    void initialize()
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

public:
    LogisticRegression(const MatrixXd &X_in, const VectorXd &y_in,
                       int max_iter = 50, double tol = 1e-8)
        : X(X_in), y(y_in), max_iterations(max_iter), rel_tolerance(tol)
    {
        n = X.rows();
        p = X.cols();
        beta = VectorXd::Zero(p);
        eta = VectorXd::Zero(n);
    }

    void fit()
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

    VectorXd getCoefficients() const { return beta; }
};

int main()
{
    MatrixXd X(5, 2);
    X << 1, 1,
        1, 2,
        1, 3,
        1, 4,
        1, 5;

    VectorXd y(5);
    y << 0, 0, 1, 1, 1;

    LogisticRegression model(X, y);
    model.fit();

    VectorXd coeffs = model.getCoefficients();
    cout << "Coefficients:\n";
    cout << fixed << setprecision(6) << coeffs.transpose() << "\n";
    return 0;
}
