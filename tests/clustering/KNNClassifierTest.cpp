#include "../ml_library_include/ml/clustering/KNNClassifier.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../TestUtils.hpp"

int main() {
    // Training data
    std::vector<std::vector<double>> X_train = {
        {1.0, 2.0},
        {1.5, 1.8},
        {5.0, 8.0},
        {8.0, 8.0},
        {1.0, 0.6},
        {9.0, 11.0}
    };
    std::vector<int> y_train = {0, 0, 1, 1, 0, 1};

    // Test data
    std::vector<std::vector<double>> X_test = {
        {1.0, 1.0}, // Expected class: 0
        {8.0, 9.0}, // Expected class: 1
        {0.0, 0.0}  // Expected class: 0
    };

    // Expected classes for test data
    std::vector<int> expected_classes = {0, 1, 0};

    // Create and train the KNN classifier with k = 3
    KNNClassifier knn(3);
    knn.fit(X_train, y_train);

    // Make predictions
    std::vector<int> predictions = knn.predict(X_test);

    // Verify predictions by comparing them with expected values
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Sample " << i << " predicted class: " << predictions[i] 
                  << ", Expected class: " << expected_classes[i] << std::endl;
        assert(predictions[i] == expected_classes[i] && "KNN prediction does not match expected class.");
    }

    // Inform user of successful test
    std::cout << "KNN Classifier Basic Test passed." << std::endl;

    return 0;
}
