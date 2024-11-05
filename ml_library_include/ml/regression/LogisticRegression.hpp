#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>

/**
 * @file LogisticRegression.hpp
 * @brief A simple implementation of Logistic Regression with improvements.
 */
/**
 * @class LogisticRegression
 * @brief Logistic Regression model for binary classification tasks.
 */
class LogisticRegression {
public:
    /**
     * @brief Constructor initializing the learning rate and iteration count.
     * @param learningRate The rate at which the model learns.
     * @param iterations Number of training iterations.
     * @param useBias Whether to include a bias term.
     */
    LogisticRegression(double learningRate = 0.01, int iterations = 1000, bool useBias = true)
        : learningRate_(learningRate), iterations_(iterations), useBias_(useBias) {}

    /**
     * @brief Train the model using features and labels.
     * @param features Input feature matrix.
     * @param labels Binary labels (0 or 1).
     */
    void train(const std::vector<std::vector<double>>& features, const std::vector<int>& labels) {
        if (features.empty() || labels.empty()) {
            throw std::invalid_argument("Features and labels must not be empty.");
        }
        if (features.size() != labels.size()) {
            throw std::invalid_argument("The number of feature vectors must match the number of labels.");
        }

        size_t numSamples = features.size();
        size_t numFeatures = features[0].size();

        // Validate that all feature vectors have the same size
        for (const auto& feature : features) {
            if (feature.size() != numFeatures) {
                throw std::invalid_argument("All feature vectors must have the same number of elements.");
            }
        }

        // Initialize weights if they haven't been initialized yet
        if (weights_.empty()) {
            weights_ = std::vector<double>(numFeatures, 0.0);
            if (useBias_) {
                bias_ = 0.0;
            }
        }

        for (int iter = 0; iter < iterations_; ++iter) {
            std::vector<double> gradients(numFeatures, 0.0);
            double biasGradient = 0.0;

            for (size_t i = 0; i < numSamples; ++i) {
                double prediction = predictProbability(features[i]);
                double error = prediction - labels[i];

                for (size_t j = 0; j < numFeatures; ++j) {
                    gradients[j] += error * features[i][j];
                }

                if (useBias_) {
                    biasGradient += error;
                }
            }

            // Update weights and bias
            for (size_t j = 0; j < numFeatures; ++j) {
                weights_[j] -= learningRate_ * (gradients[j] / numSamples);
            }

            if (useBias_) {
                bias_ -= learningRate_ * (biasGradient / numSamples);
            }
        }
    }

    /**
     * @brief Predicts the class label for a given input.
     * @param features Input feature vector.
     * @return Predicted class label (0 or 1).
     */
    int predict(const std::vector<double>& features) const {
        return predictProbability(features) >= 0.5 ? 1 : 0;
    }

    /**
     * @brief Predicts the probability of class 1 for a given input.
     * @param features Input feature vector.
     * @return Probability of the input belonging to class 1.
     */
    double predictProbability(const std::vector<double>& features) const {
        if (features.size() != weights_.size()) {
            throw std::invalid_argument("Feature vector size does not match the number of weights.");
        }

        double z = std::inner_product(features.begin(), features.end(), weights_.begin(), 0.0);
        if (useBias_) {
            z += bias_;
        }
        return sigmoid(z);
    }

private:
    double learningRate_;               ///< Learning rate for gradient descent
    int iterations_;                    ///< Number of training iterations
    std::vector<double> weights_;       ///< Model weights
    double bias_ = 0.0;                 ///< Bias term
    bool useBias_;                      ///< Whether to use a bias term

    /**
     * @brief Sigmoid activation function (numerically stable version).
     * @param z Linear combination of inputs and weights.
     * @return Result of applying the sigmoid function to z.
     */
    double sigmoid(double z) const {
        if (z >= 0) {
            double exp_neg_z = std::exp(-z);
            return 1.0 / (1.0 + exp_neg_z);
        } else {
            double exp_z = std::exp(z);
            return exp_z / (1.0 + exp_z);
        }
    }
};

#endif // LOGISTIC_REGRESSION_HPP
