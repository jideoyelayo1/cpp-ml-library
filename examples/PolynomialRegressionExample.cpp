#include "../ml_library_include/ml/regression/PolynomialRegression.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// Helper function for approximate equality check
inline bool approxEqual(double a, double b, double tolerance = 0.1) {
    return std::fabs(a - b) < tolerance;
}

// Test function for Polynomial Regression
void testPolynomialRegression() {
    PolynomialRegression model(2); // Quadratic regression

    std::vector<double> x = { 1.0, 2.0, 3.0, 4.0 };
    std::vector<double> y = { 3.0, 5.0, 7.0, 9.0 };

    model.train(x, y);

    double testInput = 5.0;
    double prediction = model.predict(testInput);

    std::cout << "Predicted value for " << testInput << ": " << prediction << std::endl;

    // Check if prediction is close to the expected value
    if (approxEqual(prediction, 11.0)) {
        std::cout << "Test passed: Prediction is within tolerance." << std::endl;
    }
    else {
        std::cerr << "Test failed: Prediction is " << prediction << ", expected ~11.0." << std::endl;
    }
}

// Only include main if TEST_POLYNOMIAL_REGRESSION is defined
#ifdef TEST_POLYNOMIAL_REGRESSION
int main() {
    testPolynomialRegression();
    return 0;
}
#endif
