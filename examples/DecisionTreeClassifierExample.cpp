#include "../ml_library_include/ml/tree/DecisionTreeClassifier.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// Helper function to check if predictions match actual values
inline bool approxEqual(int a, int b) {
    return a == b;
}

// Test function for Decision Tree Classifier
void testDecisionTreeClassifier() {
    // Sample dataset
    std::vector<std::vector<double>> X = {
        {2.771244718, 1.784783929},
        {1.728571309, 1.169761413},
        {3.678319846, 2.81281357},
        {3.961043357, 2.61995032},
        {2.999208922, 2.209014212},
        {7.497545867, 3.162953546},
        {9.00220326,  3.339047188},
        {7.444542326, 0.476683375},
        {10.12493903, 3.234550982},
        {6.642287351, 3.319983761}
    };
    std::vector<int> y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    // Create and train the DecisionTreeClassifier model
    DecisionTreeClassifier model(3, 2); // Parameters: max depth = 3, min samples = 2
    model.fit(X, y);

    // Make predictions
    std::vector<int> predictions = model.predict(X);

    // Output and check predictions
    std::cout << "Decision Tree Classification Predictions:" << std::endl;
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Predicted: " << predictions[i] << ", Actual: " << y[i] << std::endl;
        if (approxEqual(predictions[i], y[i])) {
            std::cout << "Test passed: Prediction matches actual value." << std::endl;
        } else {
            std::cerr << "Test failed: Prediction is " << predictions[i] << ", expected " << y[i] << "." << std::endl;
        }
    }
}

// Only include main if TEST_DECISION_TREE_CLASSIFICATION is defined
#ifdef TEST_DECISION_TREE_CLASSIFICATION
int main() {
    testDecisionTreeClassifier();
    return 0;
}
#endif
