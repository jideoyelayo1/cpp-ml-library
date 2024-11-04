#include "../../ml_library_include/ml/tree/DecisionTreeRegressor.hpp"
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

    // Create and train the model
    DecisionTreeRegressor model(5, 2); // Parameters: max depth = 5, min samples = 2
    model.fit(X, y);

    // Make predictions
    std::vector<double> predictions = model.predict(X);

    // Check predictions by comparing them with expected values
    for (size_t i = 0; i < predictions.size(); ++i) {
        assert(approxEqual(predictions[i], y[i], 0.1) && "Prediction does not match expected value.");
    }

    // Inform user of successful test
    std::cout << "Decision Tree Regression Basic Test passed." << std::endl;

    return 0;
}
