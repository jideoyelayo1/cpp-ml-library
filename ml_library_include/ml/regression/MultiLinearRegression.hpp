#ifndef MULTILINEAR_REGRESSION_HPP
#define MULTILINEAR_REGRESSION_HPP

#include <vector>
#include <stdexcept>
#include <numeric>
#include <cmath>

/**
 * @file MultilinearRegression.hpp
 * @brief A simple implementation of Multi Linear Regression.
 */

/**
 * @class Multilinear Regression
 * @brief A class that implements Multilinear Regression for predicting values
 * based on multiple features.
 */
class MultilinearRegression {
public:
    /**
     * @brief Constructs the MultilinearRegression model with the given learning rate and number of iterations.
     *
     * @param learningRate The rate at which the model learns (default 0.01).
     * @param iterations The number of iterations for the gradient descent (default 1000).
     */
    MultilinearRegression(double learningRate = 0.01, int iterations = 1000)
        : learningRate(learningRate), iterations(iterations) {}

    /**
     * @brief Trains the Multilinear Regression model on the provided data.
     *
     * @param features A vector of vectors, where each sub-vector represents the features for one data point.
     * @param target A vector containing the target values corresponding to each data point.
     * @throw std::invalid_argument If the number of features does not match the target size.
     */
    void train(const std::vector<std::vector<double>>& features, const std::vector<double>& target) {
        if (features.empty() || features.size() != target.size()) {
            throw std::invalid_argument("Features and target data sizes do not match.");
        }

        int numFeatures = features[0].size();
        weights.resize(numFeatures, 0.0);  // Initialize weights

        for (int i = 0; i < iterations; ++i) {
            gradientDescentStep(features, target);
        }
    }

    /**
     * @brief Predicts the output for a given set of features.
     *
     * @param features A vector containing feature values for a single data point.
     * @return The predicted value.
     */
    double predict(const std::vector<double>& features) const {
        return std::inner_product(weights.begin(), weights.end(), features.begin(), 0.0);
    }

private:
    double learningRate;            ///< The learning rate for gradient descent.
    int iterations;                 ///< The number of iterations for training.
    std::vector<double> weights;    ///< The weights for the model.

    /**
     * @brief Performs a single iteration of gradient descent to update the model weights.
     *
     * @param features A vector of vectors containing the feature data.
     * @param target A vector containing the target values.
     */
    void gradientDescentStep(const std::vector<std::vector<double>>& features, const std::vector<double>& target) {
        std::vector<double> gradients(weights.size(), 0.0);

        for (size_t i = 0; i < features.size(); ++i) {
            double prediction = predict(features[i]);
            double error = prediction - target[i];

            for (size_t j = 0; j < weights.size(); ++j) {
                gradients[j] += error * features[i][j];
            }
        }

        for (size_t j = 0; j < weights.size(); ++j) {
            weights[j] -= (learningRate / features.size()) * gradients[j];
        }
    }
};

#endif // MULTILINEAR_REGRESSION_HPP
