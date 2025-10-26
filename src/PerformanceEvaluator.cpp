#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <cfloat>
#include "PerformanceEvaluator.hpp"
#include "LogisticRegression.hpp"

using namespace std;

// Calculate classification accuracy
double PerformanceEvaluator::calculateAccuracy(const VectorXi &predictions, const VectorXd &true_labels)
{
    if (predictions.size() != true_labels.size())
    {
        throw std::invalid_argument("Predictions and true labels must have the same size");
    }

    int correct = 0;
    for (int i = 0; i < predictions.size(); ++i)
    {
        if (predictions(i) == (int)true_labels(i))
        {
            correct++;
        }
    }

    return (double)correct / predictions.size();
}

// Calculate Area Under the ROC Curve (AUC)
double PerformanceEvaluator::calculateAUC(const VectorXd &probabilities, const VectorXd &true_labels)
{
    if (probabilities.size() != true_labels.size())
    {
        throw std::invalid_argument("Probabilities and true labels must have the same size");
    }

    int n = probabilities.size();

    // Create pairs of (probability, true_label) for sorting
    vector<pair<double, int>> prob_label_pairs;
    for (int i = 0; i < n; ++i)
    {
        prob_label_pairs.push_back({probabilities(i), (int)true_labels(i)});
    }

    // Sort by probability in descending order
    sort(prob_label_pairs.begin(), prob_label_pairs.end(),
         [](const pair<double, int> &a, const pair<double, int> &b)
         {
             return a.first > b.first; // Higher probability first
         });

    // Count positive and negative cases
    int n_pos = 0, n_neg = 0;
    for (int i = 0; i < n; ++i)
    {
        if ((int)true_labels(i) == 1)
            n_pos++;
        else
            n_neg++;
    }

    // Handle edge cases
    if (n_pos == 0 || n_neg == 0)
    {
        return 0.5; // Random classifier when all labels are the same
    }

    // Calculate AUC
    double auc = 0.0;
    int tp = 0; // true positives

    for (int i = 0; i < n; ++i)
    {
        if (prob_label_pairs[i].second == 1)
        {
            tp++; // Found a positive case
        }
        else
        {
            // Found a negative case
            auc += (double)tp / n_pos / n_neg;
        }
    }

    return auc;
}

// Calculate Deviance
double PerformanceEvaluator::calculateDeviance(const VectorXd &probabilities, const VectorXd &true_labels)
{
    if (probabilities.size() != true_labels.size())
    {
        throw std::invalid_argument("Probabilities and true labels must have the same size");
    }

    double dev = 0.0;
    for (int i = 0; i < true_labels.size(); ++i)
    {
        double p = probabilities(i);

        // Clip probabilities to avoid log(0)
        if (p < DBL_EPSILON)
            p = DBL_EPSILON;
        if (p > 1.0 - DBL_EPSILON)
            p = 1.0 - DBL_EPSILON;

        if (true_labels(i) == 1.0)
        {
            dev += -2.0 * std::log(p);
        }
        else
        {
            dev += -2.0 * std::log(1.0 - p);
        }
    }

    return dev;
}

// K-Fold Cross-Validation method
double PerformanceEvaluator::calculateKFold(const MatrixXd &X, const VectorXd &y,
                                            const std::vector<int> &variable_indices,
                                            const std::string &metric,
                                            bool include_intercept,
                                            int k_folds, int seed)
{
    if (k_folds < 2)
    {
        throw std::invalid_argument("k_folds must be at least 2");
    }

    int n_samples = X.rows();
    if (k_folds > n_samples)
    {
        throw std::invalid_argument("k_folds cannot be larger than number of samples");
    }

    // Validate metric
    if (metric != "accuracy" && metric != "auc" && metric != "deviance")
    {
        throw std::invalid_argument("Unsupported metric: " + metric + ". Supported: 'accuracy', 'auc', 'deviance'");
    }

    // Create fold indices
    std::vector<VectorXi> fold_indices = createKFoldIndices(n_samples, k_folds, seed);

    double total_score = 0.0;
    int successful_folds = 0;

    for (int fold = 0; fold < k_folds; ++fold)
    {
        try
        {
            // Create train/test split
            VectorXi test_mask = VectorXi::Zero(n_samples);
            for (int i = 0; i < fold_indices[fold].size(); ++i)
            {
                test_mask(fold_indices[fold](i)) = 1;
            }

            // Extract train data
            std::vector<int> train_idx;
            std::vector<int> test_idx;
            for (int i = 0; i < n_samples; ++i)
            {
                if (test_mask(i) == 0)
                {
                    train_idx.push_back(i);
                }
                else
                {
                    test_idx.push_back(i);
                }
            }

            // Create train/test matrices
            MatrixXd X_train(train_idx.size(), X.cols());
            VectorXd y_train(train_idx.size());
            MatrixXd X_test(test_idx.size(), X.cols());
            VectorXd y_test(test_idx.size());

            for (size_t i = 0; i < train_idx.size(); ++i)
            {
                X_train.row(i) = X.row(train_idx[i]);
                y_train(i) = y(train_idx[i]);
            }

            for (size_t i = 0; i < test_idx.size(); ++i)
            {
                X_test.row(i) = X.row(test_idx[i]);
                y_test(i) = y(test_idx[i]);
            }

            // Extract subset matrices
            MatrixXd X_train_subset = extractSubsetMatrix(X_train, variable_indices, include_intercept);
            MatrixXd X_test_subset = extractSubsetMatrix(X_test, variable_indices, include_intercept);

            // Fit model on training data
            LogisticRegression lr(X_train_subset, y_train);
            lr.fit();

            // Calculate metric for this fold
            double fold_score = 0.0;
            if (metric == "accuracy")
            {
                VectorXi predictions = lr.predict(X_test_subset);
                fold_score = calculateAccuracy(predictions, y_test);
            }
            else if (metric == "auc")
            {
                VectorXd probabilities = lr.predict_proba(X_test_subset);
                fold_score = calculateAUC(probabilities, y_test);
            }
            else if (metric == "deviance")
            {
                VectorXd probabilities = lr.predict_proba(X_test_subset);
                fold_score = calculateDeviance(probabilities, y_test);
            }
            else
            {
                // This should never happen due to validation at function start
                throw std::invalid_argument("Invalid metric: " + metric);
            }

            total_score += fold_score;
            successful_folds++;
        }
        catch (const std::exception &e)
        {
            // Skip this fold if model fitting fails
            continue;
        }
    }

    if (successful_folds == 0)
    {
        throw std::runtime_error("All CV folds failed to fit");
    }

    return total_score / successful_folds;
}

// Repeated K-Fold Cross-Validation method
double PerformanceEvaluator::calculateRepeatedKFold(const MatrixXd &X, const VectorXd &y,
                                                    const std::vector<int> &variable_indices,
                                                    const std::string &metric,
                                                    bool include_intercept,
                                                    int k_folds, int n_repeats, int seed)
{
    if (n_repeats < 1)
    {
        throw std::invalid_argument("n_repeats must be at least 1");
    }

    double total_score = 0.0;
    int successful_repeats = 0;

    for (int repeat = 0; repeat < n_repeats; ++repeat)
    {
        try
        {
            // Use different seed for each repeat (if seed provided)
            int repeat_seed = (seed == -1) ? -1 : seed + repeat;

            double repeat_score = calculateKFold(X, y, variable_indices, metric,
                                                 include_intercept, k_folds, repeat_seed);
            total_score += repeat_score;
            successful_repeats++;
        }
        catch (const std::exception &e)
        {
            // Skip this repeat if it fails
            continue;
        }
    }

    if (successful_repeats == 0)
    {
        throw std::runtime_error("All repeated CV runs failed");
    }

    return total_score / successful_repeats;
}

// Helper method to create k-fold indices
std::vector<VectorXi> PerformanceEvaluator::createKFoldIndices(int n_samples, int k_folds, int seed)
{
    // Create indices vector
    std::vector<int> indices(n_samples);
    for (int i = 0; i < n_samples; ++i)
    {
        indices[i] = i;
    }

    // Shuffle indices if seed is provided
    if (seed != -1)
    {
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    else
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    // Create folds
    std::vector<VectorXi> folds(k_folds);
    int fold_size = n_samples / k_folds;
    int remainder = n_samples % k_folds;

    int start_idx = 0;
    for (int fold = 0; fold < k_folds; ++fold)
    {
        int current_fold_size = fold_size + (fold < remainder ? 1 : 0);
        folds[fold] = VectorXi(current_fold_size);

        for (int i = 0; i < current_fold_size; ++i)
        {
            folds[fold](i) = indices[start_idx + i];
        }

        start_idx += current_fold_size;
    }

    return folds;
}

// Helper method to extract subset matrix
MatrixXd PerformanceEvaluator::extractSubsetMatrix(const MatrixXd &X,
                                                   const std::vector<int> &variable_indices,
                                                   bool include_intercept)
{
    if (variable_indices.empty())
    {
        throw std::invalid_argument("Variable indices cannot be empty");
    }

    std::vector<int> final_indices = include_intercept ? addInterceptColumn(variable_indices) : variable_indices;

    MatrixXd X_subset(X.rows(), final_indices.size());

    for (size_t i = 0; i < final_indices.size(); ++i)
    {
        if (final_indices[i] == -1)
        {
            // Intercept column
            X_subset.col(i) = VectorXd::Ones(X.rows());
        }
        else
        {
            X_subset.col(i) = X.col(final_indices[i]);
        }
    }

    return X_subset;
}

// Helper method to add intercept column
std::vector<int> PerformanceEvaluator::addInterceptColumn(const std::vector<int> &variable_indices)
{
    std::vector<int> with_intercept;
    with_intercept.push_back(-1); // -1 represents intercept
    with_intercept.insert(with_intercept.end(), variable_indices.begin(), variable_indices.end());
    return with_intercept;
}
