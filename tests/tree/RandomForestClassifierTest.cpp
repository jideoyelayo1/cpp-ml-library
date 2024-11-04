#include "../../ml_library_include/ml/tree/RandomForestClassifier.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../TestUtils.hpp"

int main() {
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

    // Create and train the RandomForestClassifier model
    RandomForestClassifier model(25, 5, 2); // Parameters: 25 trees, max depth = 5, min samples = 2
    model.fit(X, y);

    // Make predictions
    std::vector<int> predictions = model.predict(X);

    // Verify predictions by comparing them with expected values
    size_t correctCount = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Predicted: " << predictions[i] << ", Actual: " << y[i] << std::endl;
        if (predictions[i] == y[i]) {
            ++correctCount;
        }
    }

    // Calculate accuracy
    double accuracy = static_cast<double>(correctCount) / y.size();
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    // Assert that accuracy is within acceptable range
    assert(accuracy >= 0.9 && "Accuracy is below acceptable threshold.");

    std::cout << "Random Forest Classification Basic Test passed." << std::endl;
    return 0;
}
