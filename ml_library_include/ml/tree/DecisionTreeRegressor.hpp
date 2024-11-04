#ifndef DECISION_TREE_REGRESSOR_HPP
#define DECISION_TREE_REGRESSOR_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

/**
 * @file DecisionTreeRegressor.hpp
 * @brief A simple implementation of Decision Tree Regression.
 */

/**
 * @class DecisionTreeRegressor
 * @brief Implements a Decision Tree Regressor.
 */
class DecisionTreeRegressor {
public:
    /**
     * @brief Constructs a DecisionTreeRegressor.
     * @param max_depth The maximum depth of the tree.
     * @param min_samples_split The minimum number of samples required to split an internal node.
     */
    DecisionTreeRegressor(int max_depth = 5, int min_samples_split = 2);

    /**
     * @brief Destructor for DecisionTreeRegressor.
     */
    ~DecisionTreeRegressor();

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
        Node* left;
        Node* right;

        Node() : is_leaf(false), value(0.0), feature_index(-1), threshold(0.0), left(nullptr), right(nullptr) {}
    };

    Node* root;
    int max_depth;
    int min_samples_split;

    Node* build_tree(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int depth);
    double calculate_mse(const std::vector<double>& y) const;
    void split_dataset(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int feature_index, double threshold,
                       std::vector<std::vector<double>>& X_left, std::vector<double>& y_left,
                       std::vector<std::vector<double>>& X_right, std::vector<double>& y_right) const;
    double predict_sample(const std::vector<double>& x, Node* node) const;
    void delete_tree(Node* node);
};

DecisionTreeRegressor::DecisionTreeRegressor(int max_depth, int min_samples_split)
    : root(nullptr), max_depth(max_depth), min_samples_split(min_samples_split) {}

DecisionTreeRegressor::~DecisionTreeRegressor() {
    delete_tree(root);
}

void DecisionTreeRegressor::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    root = build_tree(X, y, 0);
}

std::vector<double> DecisionTreeRegressor::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> predictions;
    for (const auto& x : X) {
        predictions.push_back(predict_sample(x, root));
    }
    return predictions;
}

DecisionTreeRegressor::Node* DecisionTreeRegressor::build_tree(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int depth) {
    Node* node = new Node();

    // Check stopping criteria
    if (depth >= max_depth || y.size() < min_samples_split) {
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
    for (int feature_index = 0; feature_index < num_features; ++feature_index) {
        // Get all possible thresholds
        std::vector<double> feature_values;
        for (const auto& x : X) {
            feature_values.push_back(x[feature_index]);
        }
        std::sort(feature_values.begin(), feature_values.end());
        std::vector<double> thresholds;
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
                best_X_left = X_left;
                best_X_right = X_right;
                best_y_left = y_left;
                best_y_right = y_right;
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

double DecisionTreeRegressor::calculate_mse(const std::vector<double>& y) const {
    double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double mse = 0.0;
    for (double val : y) {
        mse += (val - mean) * (val - mean);
    }
    return mse / y.size();
}

void DecisionTreeRegressor::split_dataset(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
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

double DecisionTreeRegressor::predict_sample(const std::vector<double>& x, Node* node) const {
    if (node->is_leaf) {
        return node->value;
    }
    if (x[node->feature_index] <= node->threshold) {
        return predict_sample(x, node->left);
    } else {
        return predict_sample(x, node->right);
    }
}

void DecisionTreeRegressor::delete_tree(Node* node) {
    if (node != nullptr) {
        delete_tree(node->left);
        delete_tree(node->right);
        delete node;
    }
}

#endif // DECISION_TREE_REGRESSOR_HPP
