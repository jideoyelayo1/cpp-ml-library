#include "ml/regression/MultiLinearRegression.h"
#include <vector>
#include <iostream>
#include <cassert>

bool approxEqual(double a, double b, double epsilon = 0.1) {
    return std::abs(a - b) < epsilon;
}

void testMultilinearRegression() {
    MultilinearRegression model(0.01, 1000);

    std::vector<std::vector<double>> features = {
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0}
    };

    std::vector<double> target = {3.0, 5.0, 7.0, 9.0};

    // Ensure that training runs without errors
    model.train(features, target);

    // Test the prediction
    std::vector<double> testFeatures = {5.0, 6.0};
    double prediction = model.predict(testFeatures);

    // Assert with approximate equality
    assert(approxEqual(prediction, 11.0));

    // Inform user of successful test
    std::cout << "MultilinearRegression Basic Test passed." << std::endl;
}
