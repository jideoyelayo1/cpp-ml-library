#ifndef KNN_CLASSIFIER_HPP
#define KNN_CLASSIFIER_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>

/**
 * @file KNNClassifier.hpp
 * @brief Implementation of the K-Nearest Neighbors Classifier.
 */

/**
 * @class KNNClassifier
 * @brief K-Nearest Neighbors Classifier for classification tasks.
 */
class KNNClassifier {
public:
    /**
     * @brief Constructs a KNNClassifier.
     * @param k The number of neighbors to consider.
     */
    explicit KNNClassifier(int k = 3);

    /**
     * @brief Destructor for KNNClassifier.
     */
    ~KNNClassifier();

    /**
     * @brief Fits the classifier to the training data.
     * @param X A vector of feature vectors (training data).
     * @param y A vector of target class labels (training labels).
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

    /**
     * @brief Predicts class labels for the given input data.
     * @param X A vector of feature vectors (test data).
     * @return A vector of predicted class labels.
     */
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

private:
    int k;  ///< Number of neighbors to consider.
    std::vector<std::vector<double>> X_train;  ///< Training data features.
    std::vector<int> y_train;  ///< Training data labels.

    /**
     * @brief Computes the Euclidean distance between two feature vectors.
     * @param a The first feature vector.
     * @param b The second feature vector.
     * @return The Euclidean distance.
     */
    double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) const;

    /**
     * @brief Predicts the class label for a single sample.
     * @param x The feature vector of the sample.
     * @return The predicted class label.
     */
    int predict_sample(const std::vector<double>& x) const;
};

KNNClassifier::KNNClassifier(int k) : k(k) {}

KNNClassifier::~KNNClassifier() {}

void KNNClassifier::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    X_train = X;
    y_train = y;
}

std::vector<int> KNNClassifier::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> predictions;
    predictions.reserve(X.size());
    for (const auto& x : X) {
        predictions.push_back(predict_sample(x));
    }
    return predictions;
}

double KNNClassifier::euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) const {
    double distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

int KNNClassifier::predict_sample(const std::vector<double>& x) const {
    // Vector to store distances and corresponding labels
    std::vector<std::pair<double, int>> distances;
    distances.reserve(X_train.size());

    // Compute distances to all training samples
    for (size_t i = 0; i < X_train.size(); ++i) {
        double dist = euclidean_distance(x, X_train[i]);
        distances.emplace_back(dist, y_train[i]);
    }

    // Sort distances
    std::nth_element(distances.begin(), distances.begin() + k, distances.end(),
                     [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                         return a.first < b.first;
                     });

    // Get the labels of the k nearest neighbors
    std::unordered_map<int, int> class_counts;
    for (int i = 0; i < k; ++i) {
        int label = distances[i].second;
        class_counts[label]++;
    }

    // Determine the majority class
    int max_count = 0;
    int majority_class = -1;
    for (const auto& [label, count] : class_counts) {
        if (count > max_count) {
            max_count = count;
            majority_class = label;
        }
    }

    return majority_class;
}

#endif // KNN_CLASSIFIER_HPP
