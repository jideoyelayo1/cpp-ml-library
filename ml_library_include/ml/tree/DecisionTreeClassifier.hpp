#ifndef DECISION_TREE_CLASSIFIER_HPP
#define DECISION_TREE_CLASSIFIER_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <map>
#include <cmath>

/**
 * @file DecisionTreeClassifier.hpp
 * @brief A simple implementation of Decision Tree Classification.
 */

/**
 * @class DecisionTreeClassifier
 * @brief Implements a Decision Tree Classifier.
 */
class DecisionTreeClassifier {
public:
    /**
     * @brief Constructs a DecisionTreeClassifier.
     * @param max_depth The maximum depth of the tree.
     * @param min_samples_split The minimum number of samples required to split an internal node.
     */
    DecisionTreeClassifier(int max_depth = 5, int min_samples_split = 2);

    /**
     * @brief Destructor for DecisionTreeClassifier.
     */
    ~DecisionTreeClassifier();

    /**
     * @brief Fits the model to the training data.
     * @param X A vector of feature vectors.
     * @param y A vector of target class labels.
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

    /**
     * @brief Predicts class labels for given input data.
     * @param X A vector of feature vectors.
     * @return A vector of predicted class labels.
     */
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

private:
    struct Node {
        bool is_leaf;
        int value; // Class label for leaf nodes
        int feature_index;
        double threshold;
        Node* left;
        Node* right;

        Node() : is_leaf(false), value(0), feature_index(-1), threshold(0.0), left(nullptr), right(nullptr) {}
    };

    Node* root;
    int max_depth;
    int min_samples_split;

    Node* build_tree(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int depth);
    double calculate_gini(const std::vector<int>& y) const;
    void split_dataset(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int feature_index, double threshold,
                       std::vector<std::vector<double>>& X_left, std::vector<int>& y_left,
                       std::vector<std::vector<double>>& X_right, std::vector<int>& y_right) const;
    int predict_sample(const std::vector<double>& x, Node* node) const;
    void delete_tree(Node* node);
};

DecisionTreeClassifier::DecisionTreeClassifier(int max_depth, int min_samples_split)
    : root(nullptr), max_depth(max_depth), min_samples_split(min_samples_split) {}

DecisionTreeClassifier::~DecisionTreeClassifier() {
    delete_tree(root);
}

void DecisionTreeClassifier::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    root = build_tree(X, y, 0);
}

std::vector<int> DecisionTreeClassifier::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> predictions;
    for (const auto& x : X) {
        predictions.push_back(predict_sample(x, root));
    }
    return predictions;
}

DecisionTreeClassifier::Node* DecisionTreeClassifier::build_tree(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int depth) {
    Node* node = new Node();

    // Check stopping criteria
    if (depth >= max_depth || y.size() < static_cast<size_t>(min_samples_split) || calculate_gini(y) == 0.0) {
        node->is_leaf = true;
        // Majority class label
        std::map<int, int> class_counts;
        for (int label : y) {
            class_counts[label]++;
        }
        node->value = std::max_element(class_counts.begin(), class_counts.end(),
                                       [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                                           return a.second < b.second;
                                       })->first;
        return node;
    }

    double best_gini = std::numeric_limits<double>::max();
    int best_feature_index = -1;
    double best_threshold = 0.0;
    std::vector<std::vector<double>> best_X_left, best_X_right;
    std::vector<int> best_y_left, best_y_right;

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
            std::vector<int> y_left, y_right;
            split_dataset(X, y, feature_index, threshold, X_left, y_left, X_right, y_right);

            if (y_left.empty() || y_right.empty())
                continue;

            double gini_left = calculate_gini(y_left);
            double gini_right = calculate_gini(y_right);
            double gini = (gini_left * y_left.size() + gini_right * y_right.size()) / y.size();

            if (gini < best_gini) {
                best_gini = gini;
                best_feature_index = feature_index;
                best_threshold = threshold;
                best_X_left = X_left;
                best_X_right = X_right;
                best_y_left = y_left;
                best_y_right = y_right;
            }
        }
    }

    // If no split improves the Gini impurity, make this a leaf node
    if (best_feature_index == -1) {
        node->is_leaf = true;
        // Majority class label
        std::map<int, int> class_counts;
        for (int label : y) {
            class_counts[label]++;
        }
        node->value = std::max_element(class_counts.begin(), class_counts.end(),
                                       [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                                           return a.second < b.second;
                                       })->first;
        return node;
    }

    // Recursively build the left and right subtrees
    node->feature_index = best_feature_index;
    node->threshold = best_threshold;
    node->left = build_tree(best_X_left, best_y_left, depth + 1);
    node->right = build_tree(best_X_right, best_y_right, depth + 1);
    return node;
}

double DecisionTreeClassifier::calculate_gini(const std::vector<int>& y) const {
    std::map<int, int> class_counts;
    for (int label : y) {
        class_counts[label]++;
    }
    double impurity = 1.0;
    size_t total = y.size();
    for (const auto& class_count : class_counts) {
        double prob = static_cast<double>(class_count.second) / total;
        impurity -= prob * prob;
    }
    return impurity;
}

void DecisionTreeClassifier::split_dataset(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
                                           int feature_index, double threshold,
                                           std::vector<std::vector<double>>& X_left, std::vector<int>& y_left,
                                           std::vector<std::vector<double>>& X_right, std::vector<int>& y_right) const {
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

int DecisionTreeClassifier::predict_sample(const std::vector<double>& x, Node* node) const {
    if (node->is_leaf) {
        return node->value;
    }
    if (x[node->feature_index] <= node->threshold) {
        return predict_sample(x, node->left);
    } else {
        return predict_sample(x, node->right);
    }
}

void DecisionTreeClassifier::delete_tree(Node* node) {
    if (node != nullptr) {
        delete_tree(node->left);
        delete_tree(node->right);
        delete node;
    }
}

#endif // DECISION_TREE_CLASSIFIER_HPP
