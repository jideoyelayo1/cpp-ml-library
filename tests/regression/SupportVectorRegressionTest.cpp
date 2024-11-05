#include "../ml_library_include/ml/regression/SupportVectorRegression.hpp"
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath> // For std::abs

int main() {
    // Create and train the model
    SupportVectorRegression svr(1.0, 0.1, SupportVectorRegression::KernelType::RBF, 3, 0.1);

    // Training data
    std::vector<std::vector<double>> X_train = {
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0}
    };
    std::vector<double> y_train = {1.5, 2.0, 2.5, 3.0, 3.5};

    // Ensure that training runs without errors
    svr.fit(X_train, y_train);

    // Test data
    std::vector<std::vector<double>> X_test = {
        {1.5},
        {2.5},
        {3.5}
    };

    // Expected predictions (approximate values)
    std::vector<double> expected_predictions = {1.75, 2.25, 2.75};

    // Make predictions
    std::vector<double> predictions = svr.predict(X_test);

    // Check that predictions are close to expected values
    for (size_t i = 0; i < predictions.size(); ++i) {
        // Allow a small tolerance due to potential numerical differences
        double tolerance = 0.1;
        assert(std::abs(predictions[i] - expected_predictions[i]) < tolerance);
    }

    // Inform user of successful test
    std::cout << "Support Vector Regression Basic Test passed." << std::endl;

    return 0;
}
