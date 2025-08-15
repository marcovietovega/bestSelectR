#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include "PerformanceEvaluator.hpp"

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

    // Calculate AUC using the trapezoidal rule
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
            // Found a negative case - add area: height = tp/n_pos, width = 1/n_neg
            auc += (double)tp / n_pos / n_neg;
        }
    }

    return auc;
}
