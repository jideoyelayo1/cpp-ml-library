#ifndef MULTILINEAR_REGRESSION_HPP
#define MULTILINEAR_REGRESSION_HPP

#include <vector>
#include <stdexcept>
#include <numeric>
#include <cmath>

/**
 * @file MultilinearRegression.hpp
 * @brief A simple implementation of Multilinear Regression with improvements.
 */

/**
 * @class MultilinearRegression
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
     * @param regularizationParameter The regularization parameter lambda (default 0.0, no regularization).
     */
    MultilinearRegression(double learningRate = 0.01, int iterations = 1000, double regularizationParameter = 0.0)
        : learningRate_(learningRate), iterations_(iterations), lambda_(regularizationParameter) {}

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

        size_t numSamples = features.size();
        size_t numFeatures = features[0].size();

        // Validate that all feature vectors have the same size
        for (const auto& feature : features) {
            if (feature.size() != numFeatures) {
                throw std::invalid_argument("All feature vectors must have the same number of elements.");
            }
        }

        // Initialize weights and bias if they haven't been initialized yet
        if (weights_.empty()) {
            weights_.resize(numFeatures, 0.0);
            bias_ = 0.0;
        }

        for (int iter = 0; iter < iterations_; ++iter) {
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
        if (features.size() != weights_.size()) {
            throw std::invalid_argument("Feature vector size does not match the number of weights.");
        }
        double result = std::inner_product(weights_.begin(), weights_.end(), features.begin(), 0.0);
        result += bias_;
        return result;
    }

    /**
     * @brief Gets the current weights of the model.
     *
     * @return A vector containing the weights.
     */
    std::vector<double> getWeights() const {
        return weights_;
    }

    /**
     * @brief Gets the current bias of the model.
     *
     * @return The bias term.
     */
    double getBias() const {
        return bias_;
    }

private:
    double learningRate_;            ///< The learning rate for gradient descent.
    int iterations_;                 ///< The number of iterations for training.
    double lambda_;                  ///< Regularization parameter (lambda).
    std::vector<double> weights_;    ///< The weights for the model.
    double bias_ = 0.0;              ///< Bias term.

    /**
     * @brief Performs a single iteration of gradient descent to update the model weights.
     *
     * @param features A vector of vectors containing the feature data.
     * @param target A vector containing the target values.
     */
    void gradientDescentStep(const std::vector<std::vector<double>>& features, const std::vector<double>& target) {
        size_t numSamples = features.size();
        size_t numFeatures = weights_.size();

        std::vector<double> gradients(numFeatures, 0.0);
        double biasGradient = 0.0;

        for (size_t i = 0; i < numSamples; ++i) {
            double prediction = predict(features[i]);
            double error = prediction - target[i];

            for (size_t j = 0; j < numFeatures; ++j) {
                gradients[j] += (error * features[i][j]) + (lambda_ * weights_[j]);
            }

            biasGradient += error;
        }

        // Update weights and bias
        for (size_t j = 0; j < numFeatures; ++j) {
            weights_[j] -= (learningRate_ / numSamples) * gradients[j];
        }

        bias_ -= (learningRate_ / numSamples) * biasGradient;
    }
};

#endif // MULTILINEAR_REGRESSION_HPP
