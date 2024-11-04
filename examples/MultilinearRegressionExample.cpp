#include "../ml_library_include/ml/regression/MultiLinearRegression.hpp"
#include <vector>
#include <iostream>
#include <cmath>

// Helper function for approximate equality check
inline bool approxEqual(double a, double b, double tolerance = 0.1) {
    return std::fabs(a - b) < tolerance;
}

void testMultilinearRegression() {
    MultilinearRegression model(0.01, 1000);

    std::vector<std::vector<double>> features = {
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0}
    };

    std::vector<double> target = { 3.0, 5.0, 7.0, 9.0 };

    try {
        model.train(features, target);
        std::cout << "Training passed." << std::endl;
    }
    catch (...) {
        std::cerr << "Training failed with an exception!" << std::endl;
        return;
    }

    std::vector<double> testFeatures = { 5.0, 6.0 };
    double prediction = model.predict(testFeatures);

    if (approxEqual(prediction, 11.0)) {
        std::cout << "Test passed: Prediction is within tolerance." << std::endl;
    }
    else {
        std::cerr << "Test failed: Prediction is " << prediction << ", expected ~11.0." << std::endl;
    }
}

// Only include main if TEST_MULTILINEAR_REGRESSION is defined
#ifdef TEST_MULTILINEAR_REGRESSION
int main() {
    testMultilinearRegression();
    return 0;
}
#endif
