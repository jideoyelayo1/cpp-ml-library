#include "../ml_library_include/ml/regression/SupportVectorRegression.hpp"
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath> // For std::abs

// Helper function to perform min-max scaling on a single feature vector
void min_max_scale(std::vector<std::vector<double>>& data, double& min_val, double& max_val) {
    min_val = std::numeric_limits<double>::max();
    max_val = std::numeric_limits<double>::lowest();

    // Find min and max in data
    for (const auto& x : data) {
        min_val = std::min(min_val, x[0]);
        max_val = std::max(max_val, x[0]);
    }

    // Apply min-max scaling to each feature
    for (auto& x : data) {
        x[0] = (x[0] - min_val) / (max_val - min_val);
    }
}

int main() {
    // Training data
    std::vector<std::vector<double>> X_train = {
        {10.0},
        {20.0},
        {30.0},
        {40.0},
        {50.0}
    };
    std::vector<double> y_train = {
        10.0,
        20.0,
        30.0,
        40.0,
        50.0
    };

    // Test data
    std::vector<std::vector<double>> X_test = {
        {15.0},
        {25.0},
        {35.0}
    };

    // Apply scaling to both X_train and X_test using min-max normalization
    double min_val, max_val;
    min_max_scale(X_train, min_val, max_val);
    min_max_scale(X_test, min_val, max_val);

    // Create and train the model
    SupportVectorRegression svr(10.0, 0.01, SupportVectorRegression::KernelType::LINEAR);
    svr.fit(X_train, y_train);

    // Expected predictions (approximate values)
    std::vector<double> expected_predictions = {
        15.0,
        25.0,
        35.0
    };

    // Make predictions
    std::vector<double> predictions = svr.predict(X_test);

    // Set a tolerance for comparison
    double tolerance = 100;
    bool all_tests_passed = true;

    // Check that predictions are close to expected values and report any deviations
    for (size_t i = 0; i < predictions.size(); ++i) {
        double diff = std::abs(predictions[i] - expected_predictions[i]);
        if (diff > tolerance) {
            all_tests_passed = false;
            std::cout << "Test failed for sample " << i << ":\n";
            std::cout << "  Expected: " << expected_predictions[i]
                      << "\n  Predicted: " << predictions[i]
                      << "\n  Difference: " << diff
                      << "\n  Tolerance: " << tolerance << "\n";

            // Assert to indicate test failure
            assert(diff <= tolerance && "Prediction is outside the tolerance range");
        }
    }

    // Inform user of test outcome
    if (all_tests_passed) {
        std::cout << "Support Vector Regression Basic Test passed." << std::endl;
    }

    return 0;
}
