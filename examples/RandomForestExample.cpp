#include "../ml_library_include/ml/tree/RandomForestRegressor.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// Helper function for approximate equality check
inline bool approxEqual(double a, double b, double tolerance = 0.1) {
    return std::fabs(a - b) < tolerance;
}

// Test function for Random Forest Regression
void testRandomForestRegression() {
    // Sample dataset
    std::vector<std::vector<double>> X = {
        {5.1, 3.5, 1.4},
        {4.9, 3.0, 1.4},
        {6.2, 3.4, 5.4},
        {5.9, 3.0, 5.1}
    };
    std::vector<double> y = {0.2, 0.2, 2.3, 1.8};

    // Create and train the RandomForestRegressor model
    RandomForestRegressor model(10, 5, 2); // Parameters: 10 trees, max depth = 5, min samples = 2
    model.fit(X, y);

    // Make predictions
    std::vector<double> predictions = model.predict(X);

    // Output and check predictions
    std::cout << "Random Forest Regression Predictions:" << std::endl;
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Predicted: " << predictions[i] << ", Actual: " << y[i] << std::endl;
        if (approxEqual(predictions[i], y[i])) {
            std::cout << "Test passed: Prediction is within tolerance." << std::endl;
        } else {
            std::cerr << "Test failed: Prediction is " << predictions[i] << ", expected ~" << y[i] << "." << std::endl;
        }
    }
}

// Only include main if TEST_RANDOM_FOREST_REGRESSION is defined
#ifdef TEST_RANDOM_FOREST_REGRESSION
int main() {
    testRandomForestRegression();
    return 0;
}
#endif
