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

    // Set a tolerance for comparison
    double tolerance = 0.1;
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
