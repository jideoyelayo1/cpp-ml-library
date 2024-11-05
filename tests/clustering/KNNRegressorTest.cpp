#include "../ml_library_include/ml/clustering/KNNRegressor.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../TestUtils.hpp"


int main() {
    // Training data
    std::vector<std::vector<double>> X_train = {
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0}
    };
    std::vector<double> y_train = {2.0, 3.0, 4.0, 5.0, 6.0};

    // Test data
    std::vector<std::vector<double>> X_test = {
        {1.5}, // Expected output ~2.5
        {2.5}, // Expected output ~3.5
        {3.5}  // Expected output ~4.5
    };
    std::vector<double> expected_values = {2.5, 3.5, 4.5};

    // Create and train the KNN regressor with k = 2
    KNNRegressor knn(2);
    knn.fit(X_train, y_train);

    // Make predictions
    std::vector<double> predictions = knn.predict(X_test);

    // Verify predictions by comparing them with expected values
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Sample " << i << " predicted value: " << predictions[i]
                  << ", Expected value: " << expected_values[i] << std::endl;
        assert(approxEqual(predictions[i], expected_values[i], 0.1) && 
               "KNN regression prediction does not match expected value.");
    }

    // Inform user of successful test
    std::cout << "KNN Regressor Basic Test passed." << std::endl;

    return 0;
}
