#include "LogisticRegression.hpp"
#include <vector>
#include <iostream>
#include <cassert>

void testLogisticRegression() {
    LogisticRegression model(0.1, 1000);

    std::vector<std::vector<double>> features = {
        {0.0, 0.0},
        {1.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };

    std::vector<int> labels = {0, 1, 1, 0};

    // Ensure that training runs without errors
    model.train(features, labels);

    // Test the prediction
    std::vector<double> testFeatures = {1.0, 1.0};
    int prediction = model.predict(testFeatures);

    // Check that prediction is as expected
    assert(prediction == 1);

    // Inform user of successful test
    std::cout << "Logistic Regression Basic Test passed." << std::endl;
}
