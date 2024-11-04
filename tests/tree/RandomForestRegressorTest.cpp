#include "../../ml_library_include/ml/tree/RandomForestRegressor.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../TestUtils.hpp"

int main() {
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

    // Verify predictions by comparing them with expected values
    for (size_t i = 0; i < predictions.size(); ++i) {
        assert(approxEqual(predictions[i], y[i], 0.1) && "Prediction does not match expected value.");
    }

    // Inform user of successful test
    std::cout << "Random Forest Regression Basic Test passed." << std::endl;

    return 0;
}
