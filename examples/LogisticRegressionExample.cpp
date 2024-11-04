#include "../ml_library_include/ml/regression/LogisticRegression.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// Test function for Logistic Regression
void testLogisticRegression() {
    LogisticRegression model(0.1, 1000);

    std::vector<std::vector<double>> features = {
        {0.0, 0.0},
        {1.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };

    std::vector<int> labels = { 0, 1, 1, 0 };

    model.train(features, labels);

    std::vector<double> testInput = { 1.0, 1.0 };
    int predictedClass = model.predict(testInput);

    std::cout << "Predicted class for {1.0, 1.0}: " << predictedClass << std::endl;

    // Optionally add a check to verify the predicted class
    if (predictedClass == 1) {
        std::cout << "Test passed: Correct prediction." << std::endl;
    }
    else {
        std::cerr << "Test failed: Prediction was " << predictedClass << ", expected 1." << std::endl;
    }
}

// Only include main if TEST_LOGISTIC_REGRESSION is defined
#ifdef TEST_LOGISTIC_REGRESSION
int main() {
    testLogisticRegression();
    return 0;
}
#endif
