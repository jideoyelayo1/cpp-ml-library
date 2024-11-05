#ifndef KNN_REGRESSOR_HPP
#define KNN_REGRESSOR_HPP

#include <vector>
#include <cmath>
#include <algorithm>

/**
 * @file KNNRegressor.hpp
 * @brief Implementation of the K-Nearest Neighbors Regressor.
 */

/**
 * @class KNNRegressor
 * @brief K-Nearest Neighbors Regressor for regression tasks.
 */
class KNNRegressor {
public:
    /**
     * @brief Constructs a KNNRegressor.
     * @param k The number of neighbors to consider.
     */
    explicit KNNRegressor(int k = 3);

    /**
     * @brief Destructor for KNNRegressor.
     */
    ~KNNRegressor();

    /**
     * @brief Fits the regressor to the training data.
     * @param X A vector of feature vectors (training data).
     * @param y A vector of target values (training labels).
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

    /**
     * @brief Predicts target values for the given input data.
     * @param X A vector of feature vectors (test data).
     * @return A vector of predicted target values.
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

private:
    int k;  ///< Number of neighbors to consider.
    std::vector<std::vector<double>> X_train;  ///< Training data features.
    std::vector<double> y_train;  ///< Training data target values.

    /**
     * @brief Computes the Euclidean distance between two feature vectors.
     * @param a The first feature vector.
     * @param b The second feature vector.
     * @return The Euclidean distance.
     */
    double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) const;

    /**
     * @brief Predicts the target value for a single sample.
     * @param x The feature vector of the sample.
     * @return The predicted target value.
     */
    double predict_sample(const std::vector<double>& x) const;
};

KNNRegressor::KNNRegressor(int k) : k(k) {}

KNNRegressor::~KNNRegressor() {}

void KNNRegressor::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    X_train = X;
    y_train = y;
}

std::vector<double> KNNRegressor::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> predictions;
    predictions.reserve(X.size());
    for (const auto& x : X) {
        predictions.push_back(predict_sample(x));
    }
    return predictions;
}

double KNNRegressor::euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) const {
    double distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

double KNNRegressor::predict_sample(const std::vector<double>& x) const {
    // Vector to store distances and corresponding target values
    std::vector<std::pair<double, double>> distances;
    distances.reserve(X_train.size());

    // Compute distances to all training samples
    for (size_t i = 0; i < X_train.size(); ++i) {
        double dist = euclidean_distance(x, X_train[i]);
        distances.emplace_back(dist, y_train[i]);
    }

    // Find the k nearest neighbors
    std::nth_element(distances.begin(), distances.begin() + k, distances.end(),
                     [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                         return a.first < b.first;
                     });

    // Compute the average of the target values of the k nearest neighbors
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        sum += distances[i].second;
    }
    return sum / k;
}

#endif // KNN_REGRESSOR_HPP
