#include "MultiLinearRegression.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>

MultilinearRegression::MultilinearRegression(double learningRate, int iterations)
    : learningRate(learningRate), iterations(iterations) {}

void MultilinearRegression::train(const std::vector<std::vector<double>>& features, const std::vector<double>& target) {
    if (features.empty() || features.size() != target.size()) {
        throw std::invalid_argument("Features and target data sizes do not match.");
    }

    int numFeatures = features[0].size();
    weights.resize(numFeatures, 0.0);  // Initialize weights

    for (int i = 0; i < iterations; ++i) {
        gradientDescentStep(features, target);
    }
}

void MultilinearRegression::gradientDescentStep(const std::vector<std::vector<double>>& features, const std::vector<double>& target) {
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

double MultilinearRegression::predict(const std::vector<double>& features) const {
    return std::inner_product(weights.begin(), weights.end(), features.begin(), 0.0);
}
