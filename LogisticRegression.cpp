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
    double rel_tolerance; // epsilon

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

    // Deviance from a given eta (binomial 0/1) - matches R's calculation
    double deviance_from_eta(const VectorXd &eta_in) const
    {
        double dev = 0.0;
        for (int i = 0; i < n; ++i)
        {
            double mu = plogis_clip(eta_in(i));

            // R's binomial deviance calculation
            if (y(i) == 1.0)
            {
                if (mu > DBL_EPSILON)
                {
                    dev += -2.0 * std::log(mu);
                }
                else
                {
                    dev += 1e10; // Large penalty for mu near 0 when y=1
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
                    dev += 1e10; // Large penalty for mu near 1 when y=0
                }
            }
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

    // Predict probabilities for new data
    VectorXd predict_proba(const MatrixXd &X_new) const
    {
        VectorXd eta_new = X_new * beta;
        VectorXd probs(eta_new.size());
        for (int i = 0; i < eta_new.size(); ++i)
        {
            probs(i) = plogis_clip(eta_new(i));
        }
        return probs;
    }

    // Predict binary classifications (0/1) using 0.5 threshold
    VectorXi predict(const MatrixXd &X_new) const
    {
        VectorXd probs = predict_proba(X_new);
        VectorXi predictions(probs.size());
        for (int i = 0; i < probs.size(); ++i)
        {
            predictions(i) = (probs(i) >= 0.5) ? 1 : 0;
        }
        return predictions;
    }

    // Get fitted probabilities for training data
    VectorXd get_fitted_values() const
    {
        return predict_proba(X);
    }

    // Get current deviance
    double get_deviance() const
    {
        return deviance_from_eta(eta);
    }
};

int main()
{
    MatrixXd X(8, 3); // intercept + 2 predictors
    X << 1, 180, 80,  // intercept, chol, bp
        1, 200, 90,
        1, 220, 85,
        1, 240, 95,
        1, 250, 100,
        1, 180, 75,
        1, 300, 120,
        1, 320, 130;

    VectorXd y(8);
    y << 0, 0, 0, 1, 1, 0, 1, 1; // binary outcomes

    LogisticRegression model(X, y);
    model.fit();

    cout << "=== C++ LogisticRegression Results ===\n";
    cout << fixed << setprecision(6);

    // Test 1: Coefficients
    VectorXd coeffs = model.getCoefficients();
    cout << "Coefficients: " << coeffs.transpose() << "\n";

    // Test 2: Fitted probabilities
    VectorXd fitted_probs = model.get_fitted_values();
    cout << "Fitted probabilities: " << fitted_probs.transpose() << "\n";

    // Test 3: Binary predictions
    VectorXi predictions = model.predict(X);
    cout << "Binary predictions: " << predictions.transpose() << "\n";

    // Test 4: Deviance
    double deviance = model.get_deviance();
    cout << "Deviance: " << scientific << setprecision(6) << deviance << "\n";

    // Test 5: Predictions on new data
    MatrixXd X_new(3, 3);
    X_new << 1, 190, 85, 1, 280, 110, 1, 160, 70;

    VectorXd new_probs = model.predict_proba(X_new);
    VectorXi new_preds = model.predict(X_new);
    cout << fixed << setprecision(6);
    cout << "New data probabilities: " << new_probs.transpose() << "\n";
    cout << "New data predictions: " << new_preds.transpose() << "\n";
    return 0;
}
