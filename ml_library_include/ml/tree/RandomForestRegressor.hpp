#ifndef RANDOM_FOREST_REGRESSOR_HPP
#define RANDOM_FOREST_REGRESSOR_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <random>
#include <memory>

/**
 * @file RandomForestRegressor.hpp
 * @brief A simple implementation of Random Forest Regression.
 */

/**
 * @class RandomForestRegressor
 * @brief Implements a Random Forest Regressor.
 */
class RandomForestRegressor {
public:
    /**
     * @brief Constructs a RandomForestRegressor.
     * @param n_estimators The number of trees in the forest.
     * @param max_depth The maximum depth of the tree.
     * @param min_samples_split The minimum number of samples required to split an internal node.
     * @param max_features The number of features to consider when looking for the best split. Defaults to sqrt(num_features).
     */
    RandomForestRegressor(int n_estimators = 10, int max_depth = 5, int min_samples_split = 2, int max_features = -1);

    /**
     * @brief Destructor for RandomForestRegressor.
     */
    ~RandomForestRegressor() = default;

    /**
     * @brief Fits the model to the training data.
     * @param X A vector of feature vectors.
     * @param y A vector of target values.
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

    /**
     * @brief Predicts target values for given input data.
     * @param X A vector of feature vectors.
     * @return A vector of predicted target values.
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

private:
    struct Node {
        bool is_leaf;
        double value;
        int feature_index;
        double threshold;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        Node()
            : is_leaf(false), value(0.0), feature_index(-1), threshold(0.0), left(nullptr), right(nullptr) {}
    };

    struct DecisionTree {
        std::unique_ptr<Node> root;
        int max_depth;
        int min_samples_split;
        int max_features;
        std::mt19937 random_engine;

        DecisionTree(int max_depth, int min_samples_split, int max_features);
        ~DecisionTree() = default;
        void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
        double predict_sample(const std::vector<double>& x) const;

    private:
        std::unique_ptr<Node> build_tree(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int depth);
        double calculate_mse(const std::vector<double>& y) const;
        void split_dataset(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int feature_index, double threshold,
                           std::vector<std::vector<double>>& X_left, std::vector<double>& y_left,
                           std::vector<std::vector<double>>& X_right, std::vector<double>& y_right) const;
    };

    int n_estimators;
    int max_depth;
    int min_samples_split;
    int max_features;
    std::vector<std::unique_ptr<DecisionTree>> trees;
    std::mt19937 random_engine;

    void bootstrap_sample(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                          std::vector<std::vector<double>>& X_sample, std::vector<double>& y_sample);
};

RandomForestRegressor::RandomForestRegressor(int n_estimators, int max_depth, int min_samples_split, int max_features)
    : n_estimators(n_estimators), max_depth(max_depth), min_samples_split(min_samples_split), max_features(max_features) {
    std::random_device rd;
    random_engine.seed(rd());
}

void RandomForestRegressor::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    // Set max_features if not set
    int actual_max_features = max_features;
    if (actual_max_features == -1) {
        actual_max_features = static_cast<int>(std::sqrt(X[0].size()));
    }

    for (int i = 0; i < n_estimators; ++i) {
        std::vector<std::vector<double>> X_sample;
        std::vector<double> y_sample;
        bootstrap_sample(X, y, X_sample, y_sample);

        auto tree = std::make_unique<DecisionTree>(max_depth, min_samples_split, actual_max_features);
        tree->fit(X_sample, y_sample);
        trees.push_back(std::move(tree));
    }
}

std::vector<double> RandomForestRegressor::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> predictions(X.size(), 0.0);
    for (const auto& tree : trees) {
        for (size_t i = 0; i < X.size(); ++i) {
            predictions[i] += tree->predict_sample(X[i]);
        }
    }
    for (auto& pred : predictions) {
        pred /= n_estimators;
    }
    return predictions;
}

void RandomForestRegressor::bootstrap_sample(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                                             std::vector<std::vector<double>>& X_sample, std::vector<double>& y_sample) {
    size_t n_samples = X.size();
    std::uniform_int_distribution<size_t> dist(0, n_samples - 1);

    for (size_t i = 0; i < n_samples; ++i) {
        size_t index = dist(random_engine);
        X_sample.push_back(X[index]);
        y_sample.push_back(y[index]);
    }
}

RandomForestRegressor::DecisionTree::DecisionTree(int max_depth, int min_samples_split, int max_features)
    : root(nullptr), max_depth(max_depth), min_samples_split(min_samples_split), max_features(max_features) {
    std::random_device rd;
    random_engine.seed(rd());
}

void RandomForestRegressor::DecisionTree::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    root = build_tree(X, y, 0);
}

double RandomForestRegressor::DecisionTree::predict_sample(const std::vector<double>& x) const {
    const Node* node = root.get();
    while (!node->is_leaf) {
        if (x[node->feature_index] <= node->threshold) {
            node = node->left.get();
        } else {
            node = node->right.get();
        }
    }
    return node->value;
}

std::unique_ptr<RandomForestRegressor::Node> RandomForestRegressor::DecisionTree::build_tree(
    const std::vector<std::vector<double>>& X, const std::vector<double>& y, int depth) {
    auto node = std::make_unique<Node>();

    // Check stopping criteria
    if (depth >= max_depth || y.size() < static_cast<size_t>(min_samples_split)) {
        node->is_leaf = true;
        node->value = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
        return node;
    }

    double best_mse = std::numeric_limits<double>::max();
    int best_feature_index = -1;
    double best_threshold = 0.0;
    std::vector<std::vector<double>> best_X_left, best_X_right;
    std::vector<double> best_y_left, best_y_right;

    int num_features = X[0].size();
    std::vector<int> features_indices(num_features);
    std::iota(features_indices.begin(), features_indices.end(), 0);

    // Randomly select features without replacement
    std::shuffle(features_indices.begin(), features_indices.end(), random_engine);
    if (max_features < num_features) {
        features_indices.resize(max_features);
    }

    for (int feature_index : features_indices) {
        // Get all possible thresholds
        std::vector<double> feature_values;
        feature_values.reserve(X.size());
        for (const auto& x : X) {
            feature_values.push_back(x[feature_index]);
        }
        std::sort(feature_values.begin(), feature_values.end());
        feature_values.erase(std::unique(feature_values.begin(), feature_values.end()), feature_values.end());

        std::vector<double> thresholds;
        thresholds.reserve(feature_values.size() - 1);
        for (size_t i = 1; i < feature_values.size(); ++i) {
            thresholds.push_back((feature_values[i - 1] + feature_values[i]) / 2.0);
        }

        // Evaluate each threshold
        for (double threshold : thresholds) {
            std::vector<std::vector<double>> X_left, X_right;
            std::vector<double> y_left, y_right;
            split_dataset(X, y, feature_index, threshold, X_left, y_left, X_right, y_right);

            if (y_left.empty() || y_right.empty())
                continue;

            double mse_left = calculate_mse(y_left);
            double mse_right = calculate_mse(y_right);
            double mse = (mse_left * y_left.size() + mse_right * y_right.size()) / y.size();

            if (mse < best_mse) {
                best_mse = mse;
                best_feature_index = feature_index;
                best_threshold = threshold;
                best_X_left = std::move(X_left);
                best_X_right = std::move(X_right);
                best_y_left = std::move(y_left);
                best_y_right = std::move(y_right);
            }
        }
    }

    // If no split improves the mse, make this a leaf node
    if (best_feature_index == -1) {
        node->is_leaf = true;
        node->value = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
        return node;
    }

    // Recursively build the left and right subtrees
    node->feature_index = best_feature_index;
    node->threshold = best_threshold;
    node->left = build_tree(best_X_left, best_y_left, depth + 1);
    node->right = build_tree(best_X_right, best_y_right, depth + 1);
    return node;
}

double RandomForestRegressor::DecisionTree::calculate_mse(const std::vector<double>& y) const {
    double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double mse = std::transform_reduce(y.begin(), y.end(), 0.0, std::plus<>(), [mean](double val) {
        double diff = val - mean;
        return diff * diff;
    });
    return mse / y.size();
}

void RandomForestRegressor::DecisionTree::split_dataset(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                                                        int feature_index, double threshold,
                                                        std::vector<std::vector<double>>& X_left, std::vector<double>& y_left,
                                                        std::vector<std::vector<double>>& X_right, std::vector<double>& y_right) const {
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][feature_index] <= threshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }
}

#endif // RANDOM_FOREST_REGRESSOR_HPP
