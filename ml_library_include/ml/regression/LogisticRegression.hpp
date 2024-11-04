#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <vector>
#include <cmath>

/**
 * @brief Logistic Regression model for binary classification tasks.
 */
class LogisticRegression {
public:
    /**
     * @brief Constructor initializing the learning rate and iteration count.
     * @param learningRate The rate at which the model learns.
     * @param iterations Number of training iterations.
     */
    LogisticRegression(double learningRate, int iterations)
        : learningRate_(learningRate), iterations_(iterations) {}

    /**
     * @brief Train the model using features and labels.
     * @param features Input feature matrix.
     * @param labels Binary labels (0 or 1).
     */
    void train(const std::vector<std::vector<double>>& features, const std::vector<int>& labels) {
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

    /**
     * @brief Predicts the class label for a given input.
     * @param features Input feature vector.
     * @return Predicted class label (0 or 1).
     */
    int predict(const std::vector<double>& features) const {
        return predictProbability(features) >= 0.5 ? 1 : 0;
    }

private:
    double learningRate_;               ///< Learning rate for gradient descent
    int iterations_;                    ///< Number of training iterations
    std::vector<double> weights_;       ///< Model weights

    /**
     * @brief Sigmoid activation function.
     * @param z Linear combination of inputs and weights.
     * @return Result of applying the sigmoid function to z.
     */
    double sigmoid(double z) const {
        return 1.0 / (1.0 + std::exp(-z));
    }

    /**
     * @brief Predicts the probability of class 1 for a given input.
     * @param features Input feature vector.
     * @return Probability of the input belonging to class 1.
     */
    double predictProbability(const std::vector<double>& features) const {
        double z = 0.0;
        for (size_t i = 0; i < features.size(); ++i) {
            z += weights_[i] * features[i];
        }
        return sigmoid(z);
    }
};

#endif // LOGISTIC_REGRESSION_HPP
