#include "../../ml_library_include/ml/regression/LogisticRegression.hpp"
#include <cmath>
#include <vector>

LogisticRegression::LogisticRegression(double learningRate, int iterations)
    : learningRate_(learningRate), iterations_(iterations) {}

void LogisticRegression::train(const std::vector<std::vector<double>>& features, const std::vector<int>& labels) {
    int numFeatures = features[0].size();
    weights_ = std::vector<double>(numFeatures, 0.0);

    for (int iter = 0; iter < iterations_; ++iter) {
        for (size_t i = 0; i < features.size(); ++i) {
            double prediction = predictProbability(features[i]);
            for (int j = 0; j < numFeatures; ++j) {
                weights_[j] += learningRate_ * (labels[i] - prediction) * features[i][j];
            }
        }
    }
}

int LogisticRegression::predict(const std::vector<double>& features) const {
    return predictProbability(features) >= 0.5 ? 1 : 0;
}

double LogisticRegression::sigmoid(double z) const {
    return 1.0 / (1.0 + std::exp(-z));
}

double LogisticRegression::predictProbability(const std::vector<double>& features) const {
    double z = 0.0;
    for (size_t i = 0; i < features.size(); ++i) {
        z += weights_[i] * features[i];
    }
    return sigmoid(z);
}
