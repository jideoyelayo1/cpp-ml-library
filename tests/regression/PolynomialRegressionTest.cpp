#include "../../ml_library_include/ml/regression/PolynomialRegression.hpp"
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>

// Custom approximation function for comparing doubles
inline bool approxEqual(double a, double b, double epsilon = 0.1) {
    return std::fabs(a - b) <= epsilon;
}

inline void testPolynomialRegression() {
    PolynomialRegression model(2); // Quadratic regression

    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0};

    // Ensure that training runs without errors
    model.train(x, y);

    // Test the prediction
    double prediction = model.predict(5.0);

    // Check that the prediction is approximately as expected
    assert(approxEqual(prediction, 11.0, 0.1));

    // Inform user of successful test
    std::cout << "Polynomial Regression Basic Test passed." << std::endl;
}
