#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <vector>

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
    LogisticRegression(double learningRate, int iterations);

    /**
     * @brief Train the model using features and labels.
     * @param features Input feature matrix.
     * @param labels Binary labels (0 or 1).
     */
    void train(const std::vector<std::vector<double>>& features, const std::vector<int>& labels);

    /**
     * @brief Predicts the class label for a given input.
     * @param features Input feature vector.
     * @return Predicted class label (0 or 1).
     */
    int predict(const std::vector<double>& features) const;

private:
    double learningRate_;        ///< Learning rate for gradient descent
    int iterations_;             ///< Number of training iterations
    std::vector<double> weights_;///< Model weights

    double sigmoid(double z) const;
    double predictProbability(const std::vector<double>& features) const;
};

#endif // LOGISTIC_REGRESSION_HPP
